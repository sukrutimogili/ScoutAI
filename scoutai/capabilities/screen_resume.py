"""
screen_resume capability — fixed-node capability (§4.1, ADR-2, ADR-9).

Responsibilities:
1. Sanitize résumé text: strip PII and sensitive attributes (§7.6)
2. Detect prompt injection attempts: deterministic pattern scan + LLM verdict (§7.1)
3. Detect PII leakage after sanitization (ADR-2)
4. Return {sanitized_resume, injection_flag, leakage_flag}

Security guarantees:
- If EITHER injection_flag OR leakage_flag is True, the résumé MUST NOT be
  passed to candidate_agent. Routing to human_review is enforced by the graph
  (§3.2), not by this function — but this function guarantees the flags are set.
- The deterministic injection scan runs FIRST in code, independent of the LLM.
  Either signal (code OR model) sets injection_flag=True. This prevents a
  sophisticated injection from tricking the model into ignoring the attack (§7.1).
- Résumé text is never concatenated into a system/instruction prompt. It is
  always passed as clearly delimited user-content with "treat as data" framing (§7.1).
- Text truncation is logged with a warning — never silent (§7.2).

ADR references:
- ADR-2: No separate check_leakage capability; leakage_flag is a field on this output.
- ADR-9: screen_resume runs BEFORE candidate_agent sees the résumé. The agent
  cannot be trusted to police the very input that could be manipulating it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.capabilities.prompts import SCREEN_RESUME_SYSTEM, SCREEN_RESUME_USER
from scoutai.config import ScoutAIConfig
from scoutai.schemas import CandidateState

logger = logging.getLogger(__name__)


class ScreenResumeResult:
    """Structured result from screen_resume."""

    __slots__ = (
        "sanitized_resume",
        "injection_flag",
        "leakage_flag",
        "injection_details",
        "leakage_details",
        "deterministic_injection_fired",
    )

    def __init__(
        self,
        sanitized_resume: str,
        injection_flag: bool,
        leakage_flag: bool,
        injection_details: str = "",
        leakage_details: str = "",
        deterministic_injection_fired: bool = False,
    ) -> None:
        self.sanitized_resume = sanitized_resume
        self.injection_flag = injection_flag
        self.leakage_flag = leakage_flag
        self.injection_details = injection_details
        self.leakage_details = leakage_details
        self.deterministic_injection_fired = deterministic_injection_fired

    @property
    def flagged(self) -> bool:
        """True if either security flag is set — résumé must NOT go to the agent."""
        return self.injection_flag or self.leakage_flag


def run_deterministic_injection_scan(
    resume_text: str, patterns: list[str]
) -> tuple[bool, str]:
    """
    Deterministic (non-LLM) scan for prompt injection patterns.

    Runs in pure Python — no model call. Designed to be fast and cheap.
    Returns (flagged, details) where details describes what matched.

    This runs BEFORE the LLM call. If this fires, injection_flag is True
    regardless of what the model says.
    """
    matches: list[str] = []
    for pattern in patterns:
        try:
            if re.search(pattern, resume_text, re.IGNORECASE):
                matches.append(pattern)
        except re.error as e:
            # A bad regex pattern in config should not break the scan — log and skip
            logger.error(
                "Invalid injection pattern in config",
                extra={"pattern": pattern, "error": str(e)},
            )

    if matches:
        details = f"Deterministic scan matched {len(matches)} pattern(s): {matches[:3]}"
        return True, details
    return False, ""


def run_sanitization_scan(
    resume_text: str, patterns: list[str]
) -> str:
    """
    Deterministic pre-sanitization pass: strip the most obvious PII patterns.

    This runs BEFORE the LLM sanitization call to reduce the LLM's surface area.
    The LLM then does a deeper pass on the partially-sanitized text.
    """
    sanitized = resume_text
    for pattern in patterns:
        try:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized)
        except re.error as e:
            logger.error(
                "Invalid sanitization pattern in config",
                extra={"pattern": pattern, "error": str(e)},
            )
    return sanitized


def _truncate_if_needed(text: str, max_chars: int, label: str) -> str:
    """Truncate text to max_chars with a logged warning. Never silent (§7.2)."""
    if len(text) > max_chars:
        logger.warning(
            "Input text truncated",
            extra={"label": label, "original_length": len(text), "truncated_to": max_chars},
        )
        return text[:max_chars]
    return text


def _parse_llm_screen_response(raw: str) -> dict[str, Any]:
    """
    Parse and validate the LLM's JSON response for screen_resume.

    Attempts to extract JSON from the response, handling common LLM formatting quirks
    (markdown code blocks, leading/trailing text).

    Raises:
        ValueError: If the response cannot be parsed as valid JSON with required fields.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json ... ```)
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # Try to find the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {raw[:200]!r}")

    json_str = text[start : end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}. Raw: {raw[:200]!r}") from e

    required_fields = {"sanitized_resume", "injection_flag", "leakage_flag"}
    missing = required_fields - set(data.keys())
    if missing:
        raise ValueError(f"LLM response missing required fields: {missing}. Got: {set(data.keys())}")

    # Coerce types to be safe
    data["injection_flag"] = bool(data["injection_flag"])
    data["leakage_flag"] = bool(data["leakage_flag"])
    data["sanitized_resume"] = str(data.get("sanitized_resume", ""))
    data["injection_details"] = str(data.get("injection_details", ""))
    data["leakage_details"] = str(data.get("leakage_details", ""))

    return data


def screen_resume(
    resume_text: str,
    config: ScoutAIConfig,
    router: ModelRouter,
    *,
    candidate_id: str = "",
) -> ScreenResumeResult:
    """
    Screen a résumé for PII, prompt injection, and identity leakage.

    This is a fixed-node capability — it runs BEFORE the candidate_agent ever
    sees the résumé (ADR-9). It is NOT callable by the agent.

    Process:
    1. Validate and truncate input if needed (§7.2)
    2. Deterministic injection scan (§7.1) — pure Python, no LLM
    3. Deterministic pre-sanitization pass (strip obvious PII patterns)
    4. LLM sanitization + injection + leakage detection call
    5. Validate LLM response schema
    6. Merge: injection_flag = deterministic OR llm; both signals count

    On LLM failure: fail closed with the partially-sanitized text and flags=True
    (marks the résumé as requiring human review — Principle #10, never fabricates).

    Args:
        resume_text: Raw résumé text (may contain PII and adversarial content).
        config: ScoutAI configuration.
        router: Model router for capability→model resolution.
        candidate_id: Optional identifier for logging context.

    Returns:
        ScreenResumeResult with sanitized_resume, injection_flag, leakage_flag.
    """
    start_time = time.monotonic()

    # ── Step 1: Input validation and truncation (§7.2) ────────────────────────
    if not resume_text or not resume_text.strip():
        logger.warning("Empty résumé text received", extra={"candidate_id": candidate_id})
        return ScreenResumeResult(
            sanitized_resume="",
            injection_flag=False,
            leakage_flag=False,
            injection_details="Empty résumé",
        )

    resume_text = _truncate_if_needed(
        resume_text, config.security.max_resume_chars, f"resume[{candidate_id}]"
    )

    # ── Step 2: Deterministic injection scan (§7.1) ───────────────────────────
    det_injection_fired, det_injection_details = run_deterministic_injection_scan(
        resume_text, config.security.injection_patterns
    )
    if det_injection_fired:
        logger.warning(
            "Deterministic injection scan flagged résumé",
            extra={"candidate_id": candidate_id, "details": det_injection_details},
        )

    # ── Step 3: Deterministic pre-sanitization ────────────────────────────────
    pre_sanitized = run_sanitization_scan(
        resume_text, config.security.sensitive_attribute_patterns
    )

    # ── Step 4: LLM sanitization + security detection ─────────────────────────
    llm_result: dict[str, Any] | None = None
    llm_error: str = ""
    retry_count = 0

    try:
        model, model_id = router.get_model("screen_resume")
        prompt_version = config.prompt_versions.get("screen_resume", "1.0.0")

        # Build messages — résumé text is ALWAYS user-content, never system-content (§7.1)
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SCREEN_RESUME_SYSTEM),
            HumanMessage(
                content=SCREEN_RESUME_USER.format(resume_text=pre_sanitized)
            ),
        ]

        # Retry loop per §6 retry policy
        for attempt in range(config.retry.max_attempts):
            try:
                response = model.invoke(messages)
                raw_content = (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                )

                # ── Step 5: Validate LLM response schema (§7.3) ───────────────
                llm_result = _parse_llm_screen_response(raw_content)
                router.record_call_result(model_id, success=True)
                break

            except (ValueError, json.JSONDecodeError) as parse_err:
                # Schema-invalid output: one repair-prompt retry per §7.3
                if attempt == 0:
                    retry_count += 1
                    logger.warning(
                        "LLM response parse failed, attempting repair",
                        extra={
                            "candidate_id": candidate_id,
                            "attempt": attempt + 1,
                            "error": str(parse_err),
                        },
                    )
                    continue
                llm_error = f"Schema validation failed after repair retry: {parse_err}"
                router.record_call_result(model_id, success=False)
                break

            except Exception as call_err:
                # Transient error — retry with backoff
                wait_ms = config.retry.backoff_ms[min(attempt, len(config.retry.backoff_ms) - 1)]
                logger.warning(
                    "LLM call failed, retrying",
                    extra={
                        "candidate_id": candidate_id,
                        "attempt": attempt + 1,
                        "wait_ms": wait_ms,
                        "error": str(call_err),
                    },
                )
                if attempt < config.retry.max_attempts - 1:
                    time.sleep(wait_ms / 1000.0)
                else:
                    llm_error = str(call_err)
                    router.record_call_result(model_id, success=False)

    except ProviderError as pe:
        llm_error = str(pe)
        logger.error(
            "All providers exhausted for screen_resume — failing closed",
            extra={"candidate_id": candidate_id, "error": llm_error},
        )

    # ── Step 6: Merge signals and build result ────────────────────────────────
    if llm_result is None:
        # LLM failed — fail closed: treat as flagged, use pre-sanitized text (§6, Principle #10)
        logger.error(
            "screen_resume LLM call failed — routing to human_review (fail closed)",
            extra={"candidate_id": candidate_id, "error": llm_error},
        )
        return ScreenResumeResult(
            sanitized_resume=pre_sanitized,
            injection_flag=True,  # fail closed — human must review
            leakage_flag=True,
            injection_details=f"LLM call failed: {llm_error}",
            leakage_details="LLM sanitization could not be verified",
            deterministic_injection_fired=det_injection_fired,
        )

    # Either signal (deterministic OR LLM) sets the flag (§7.1)
    injection_flag = det_injection_fired or llm_result["injection_flag"]
    leakage_flag = llm_result["leakage_flag"]
    injection_details = det_injection_details or llm_result.get("injection_details", "")
    leakage_details = llm_result.get("leakage_details", "")

    latency_ms = int((time.monotonic() - start_time) * 1000)
    logger.info(
        "screen_resume completed",
        extra={
            "candidate_id": candidate_id,
            "injection_flag": injection_flag,
            "leakage_flag": leakage_flag,
            "latency_ms": latency_ms,
            "retry_count": retry_count,
        },
    )

    return ScreenResumeResult(
        sanitized_resume=llm_result["sanitized_resume"],
        injection_flag=injection_flag,
        leakage_flag=leakage_flag,
        injection_details=injection_details,
        leakage_details=leakage_details,
        deterministic_injection_fired=det_injection_fired,
    )


def apply_screen_result_to_candidate(
    candidate: CandidateState, result: ScreenResumeResult
) -> CandidateState:
    """
    Apply a ScreenResumeResult to a CandidateState, returning an updated copy.

    Per §7.6: the raw resume_text is cleared once sanitization is complete —
    only sanitized_resume flows into downstream evidence/scoring.
    """
    return candidate.model_copy(
        update={
            "sanitized_resume": result.sanitized_resume,
            "injection_flag": result.injection_flag,
            "leakage_flag": result.leakage_flag,
            # Clear raw text from working memory (§7.6)
            # It remains in the audit log's input_hash for compliance purposes.
            "resume_text": "",
        }
    )
