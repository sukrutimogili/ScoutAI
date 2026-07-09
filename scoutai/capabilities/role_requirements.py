"""
extract_role_requirements and generate_rubric capabilities — fixed-node (§4.1).
Caching layer for both, keyed on content hash + version strings (§9).

Spec references: §4.1, §8, §9.
ADR references: ADR-5 (model roles, not hardcoded models).

Caching rules (§9):
- RoleProfile cached by jd_hash
- Rubric cached by (jd_hash, rubric_config_version)
- Cache key also includes prompt_version and schema_version so stale
  cached objects are not served after a prompt or schema change.
- TTL controlled by config.cache.ttl_seconds.
- Deliberate non-caching: candidate_agent internals (per §9).

Failure modes:
- LLM failure → fail closed: raises ProviderError, not silently returns empty object.
- Schema-invalid output → one repair retry (§7.3), then fail closed.
- Cache miss → normal LLM call path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Optional

from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.capabilities.prompts import (
    EXTRACT_ROLE_REQUIREMENTS_SYSTEM,
    EXTRACT_ROLE_REQUIREMENTS_USER,
    GENERATE_RUBRIC_SYSTEM,
    GENERATE_RUBRIC_USER,
)
from scoutai.config import ScoutAIConfig
from scoutai.schemas import RequiredSkill, RoleProfile, Rubric, RubricCriterion

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _sha256(text: str) -> str:
    """Return a hex sha256 hash of a string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_json_response(raw: str, context: str) -> dict[str, Any]:
    """
    Extract and parse a JSON object from an LLM response.

    Strips markdown code fences if present. Raises ValueError on failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"[{context}] No JSON object in response: {raw[:300]!r}")

    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"[{context}] Invalid JSON: {e}. Raw: {raw[:300]!r}") from e


def _call_llm_with_retry(
    model: Any,
    messages: list[Any],
    parse_fn: Any,
    capability: str,
    config: ScoutAIConfig,
    model_id: str,
    router: ModelRouter,
) -> dict[str, Any]:
    """
    Call an LLM model with retry policy and schema repair-retry (§6, §7.3).

    Returns parsed dict on success. Raises ProviderError on exhaustion.
    """
    last_error: Optional[Exception] = None
    for attempt in range(config.retry.max_attempts):
        try:
            response = model.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            result = parse_fn(raw)
            router.record_call_result(model_id, success=True)
            return result
        except (ValueError, json.JSONDecodeError) as parse_err:
            # §7.3: one repair-prompt retry on schema validation failure
            if attempt == 0:
                logger.warning(
                    "LLM output failed schema validation, attempting repair retry",
                    extra={"capability": capability, "error": str(parse_err)},
                )
                last_error = parse_err
                continue
            router.record_call_result(model_id, success=False)
            last_error = parse_err
            break
        except Exception as call_err:
            wait_ms = config.retry.backoff_ms[min(attempt, len(config.retry.backoff_ms) - 1)]
            logger.warning(
                "LLM call failed",
                extra={
                    "capability": capability,
                    "attempt": attempt + 1,
                    "error": str(call_err),
                },
            )
            if attempt < config.retry.max_attempts - 1:
                time.sleep(wait_ms / 1000.0)
            else:
                router.record_call_result(model_id, success=False)
                last_error = call_err

    raise ProviderError(
        provider="llm",
        capability=capability,
        reason=f"Failed after {config.retry.max_attempts} attempts. Last: {last_error}",
    )


# ── In-memory cache ───────────────────────────────────────────────────────────


class _CacheEntry:
    __slots__ = ("value", "created_at")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.created_at = time.monotonic()

    def is_expired(self, ttl_seconds: int) -> bool:
        return (time.monotonic() - self.created_at) >= ttl_seconds


class SessionCache:
    """
    Thread-safe in-memory cache for RoleProfile and Rubric objects.

    Cache keys include content hash + prompt_version + schema_version to
    ensure prompt or schema changes invalidate stale entries (§9).

    This is a session-scoped cache — it lives for the duration of one pipeline run.
    For cross-session caching with TTL, a Redis backend can replace this without
    changing the API (config.cache.backend = "redis").
    """

    def __init__(self, ttl_seconds: int = 86400, enabled: bool = True) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._lock = Lock()
        self._ttl = ttl_seconds
        self._enabled = enabled

    def get(self, key: str) -> Optional[Any]:
        if not self._enabled:
            return None
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if entry.is_expired(self._ttl):
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: Any) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._store[key] = _CacheEntry(value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


def _role_profile_cache_key(
    jd_hash: str, prompt_version: str, schema_version: str
) -> str:
    """Cache key for RoleProfile — tied to JD content + prompt + schema versions (§9)."""
    return f"role_profile:{jd_hash}:pv={prompt_version}:sv={schema_version}"


def _rubric_cache_key(
    jd_hash: str, rubric_config_version: str, prompt_version: str, schema_version: str
) -> str:
    """Cache key for Rubric — tied to JD content + rubric config + versions (§9)."""
    return f"rubric:{jd_hash}:rcv={rubric_config_version}:pv={prompt_version}:sv={schema_version}"


# ── extract_role_requirements ─────────────────────────────────────────────────


def _parse_role_profile_response(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM response for extract_role_requirements."""
    data = _parse_json_response(raw, "extract_role_requirements")
    required = {"title"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"extract_role_requirements response missing fields: {missing}")
    return data


def _build_role_profile(data: dict[str, Any], jd_hash: str) -> RoleProfile:
    """Construct a validated RoleProfile from parsed LLM output."""
    raw_skills = data.get("required_skills", [])
    skills = []
    for s in raw_skills:
        if isinstance(s, dict):
            try:
                skills.append(
                    RequiredSkill(
                        name=s.get("name", ""),
                        description=s.get("description", ""),
                        required=bool(s.get("required", True)),
                        weight=float(s.get("weight", 1.0)),
                    )
                )
            except Exception as e:
                logger.warning(
                    "Skipping malformed skill entry",
                    extra={"entry": s, "error": str(e)},
                )

    return RoleProfile(
        title=str(data.get("title", "Unknown Role")),
        summary=str(data.get("summary", "")),
        required_skills=skills,
        experience_requirements=[
            str(r) for r in data.get("experience_requirements", [])
        ],
        education_requirements=[
            str(r) for r in data.get("education_requirements", [])
        ],
        soft_skills=[str(s) for s in data.get("soft_skills", [])],
        jd_hash=jd_hash,
        extracted_at=_now_iso(),
    )


def extract_role_requirements(
    jd_text: str,
    config: ScoutAIConfig,
    router: ModelRouter,
    cache: Optional[SessionCache] = None,
) -> RoleProfile:
    """
    Extract structured role requirements from a job description.

    Fixed-node capability (§4.1). Uses the high_context model role.
    Results are cached by jd_hash + prompt_version + schema_version (§9).

    Args:
        jd_text: Raw job description text.
        config: ScoutAI configuration.
        router: Model router.
        cache: Optional session cache. Pass None to disable caching.

    Returns:
        Validated RoleProfile.

    Raises:
        ProviderError: If LLM call fails after all retries.
        ValueError: If jd_text is empty.
    """
    if not jd_text or not jd_text.strip():
        raise ValueError("jd_text must not be empty")

    # Truncate if needed (§7.2)
    if len(jd_text) > config.security.max_jd_chars:
        logger.warning(
            "JD text truncated",
            extra={
                "original_length": len(jd_text),
                "truncated_to": config.security.max_jd_chars,
            },
        )
        jd_text = jd_text[: config.security.max_jd_chars]

    jd_hash = _sha256(jd_text)
    prompt_version = config.prompt_versions.get("extract_role_requirements", "1.0.0")
    schema_version = config.schema_version

    # ── Cache lookup (§9) ─────────────────────────────────────────────────────
    if cache is not None:
        cache_key = _role_profile_cache_key(jd_hash, prompt_version, schema_version)
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(
                "Cache hit for extract_role_requirements",
                extra={"jd_hash": jd_hash[:8]},
            )
            return cached

    # ── LLM call ──────────────────────────────────────────────────────────────
    model, model_id = router.get_model("extract_role_requirements")

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=EXTRACT_ROLE_REQUIREMENTS_SYSTEM),
        HumanMessage(content=EXTRACT_ROLE_REQUIREMENTS_USER.format(jd_text=jd_text)),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_role_profile_response,
        capability="extract_role_requirements",
        config=config,
        model_id=model_id,
        router=router,
    )

    role_profile = _build_role_profile(data, jd_hash)

    # ── Cache store ───────────────────────────────────────────────────────────
    if cache is not None:
        cache.set(cache_key, role_profile)
        logger.info(
            "Cached role_profile",
            extra={"jd_hash": jd_hash[:8], "title": role_profile.title},
        )

    logger.info(
        "extract_role_requirements completed",
        extra={"title": role_profile.title, "skill_count": len(role_profile.required_skills)},
    )
    return role_profile


# ── generate_rubric ───────────────────────────────────────────────────────────


def _parse_rubric_response(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM response for generate_rubric."""
    data = _parse_json_response(raw, "generate_rubric")
    if "criteria" not in data:
        raise ValueError("generate_rubric response missing 'criteria' field")
    if not isinstance(data["criteria"], list) or len(data["criteria"]) == 0:
        raise ValueError("generate_rubric 'criteria' must be a non-empty list")
    return data


def _build_rubric(
    data: dict[str, Any], jd_hash: str, rubric_config_version: str
) -> Rubric:
    """Construct a validated Rubric from parsed LLM output."""
    criteria = []
    seen_names: set[str] = set()

    for c in data.get("criteria", []):
        if not isinstance(c, dict):
            continue
        name = str(c.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        try:
            criteria.append(
                RubricCriterion(
                    name=name,
                    description=str(c.get("description", "")),
                    weight=float(c.get("weight", 1.0)),
                    category=str(c.get("category", "required_skills")),
                    examples_of_sufficient_evidence=[
                        str(e) for e in c.get("examples_of_sufficient_evidence", [])
                    ],
                )
            )
        except Exception as e:
            logger.warning(
                "Skipping malformed criterion",
                extra={"name": name, "error": str(e)},
            )

    if not criteria:
        raise ValueError("No valid criteria could be built from LLM response")

    return Rubric(
        criteria=criteria,
        jd_hash=jd_hash,
        rubric_config_version=rubric_config_version,
        generated_at=_now_iso(),
    )


def generate_rubric(
    role_profile: RoleProfile,
    config: ScoutAIConfig,
    router: ModelRouter,
    cache: Optional[SessionCache] = None,
) -> Rubric:
    """
    Generate a scoring rubric from a role profile.

    Fixed-node capability (§4.1). Uses the high_context model role.
    Cached by (jd_hash, rubric_config_version, prompt_version, schema_version) (§9).

    Args:
        role_profile: The structured role profile (output of extract_role_requirements).
        config: ScoutAI configuration.
        router: Model router.
        cache: Optional session cache.

    Returns:
        Validated Rubric.

    Raises:
        ProviderError: If LLM call fails after all retries.
    """
    jd_hash = role_profile.jd_hash
    rubric_config_version = config.rubric.config_version
    prompt_version = config.prompt_versions.get("generate_rubric", "1.0.0")
    schema_version = config.schema_version

    # ── Cache lookup (§9) ─────────────────────────────────────────────────────
    if cache is not None:
        cache_key = _rubric_cache_key(
            jd_hash, rubric_config_version, prompt_version, schema_version
        )
        cached = cache.get(cache_key)
        if cached is not None:
            logger.info(
                "Cache hit for generate_rubric",
                extra={"jd_hash": jd_hash[:8]},
            )
            return cached

    # ── LLM call ──────────────────────────────────────────────────────────────
    model, model_id = router.get_model("generate_rubric")

    from langchain_core.messages import HumanMessage, SystemMessage

    role_profile_json = role_profile.model_dump_json(indent=2)
    messages = [
        SystemMessage(content=GENERATE_RUBRIC_SYSTEM),
        HumanMessage(
            content=GENERATE_RUBRIC_USER.format(role_profile_json=role_profile_json)
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_rubric_response,
        capability="generate_rubric",
        config=config,
        model_id=model_id,
        router=router,
    )

    rubric = _build_rubric(data, jd_hash, rubric_config_version)

    # ── Cache store ───────────────────────────────────────────────────────────
    if cache is not None:
        cache.set(cache_key, rubric)
        logger.info(
            "Cached rubric",
            extra={"jd_hash": jd_hash[:8], "criteria_count": len(rubric.criteria)},
        )

    logger.info(
        "generate_rubric completed",
        extra={"criteria_count": len(rubric.criteria)},
    )
    return rubric
