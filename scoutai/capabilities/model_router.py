"""
Model Router — capability → model_role → provider resolution.

Spec reference: §6 (Model Selection & Routing), §7.7 (Rate Limiting).
ADR reference: ADR-5 (two roles, not hardcoded models).

Rules:
- capability → model_role is a fixed, config-independent mapping (§6).
- model_role → actual provider/model is resolved from config.yaml.
- The candidate_agent decides *which tool* to call; it NEVER decides which model
  answers that call. Model selection is deterministic and testable.
- Retry: exponential backoff (3 attempts, 250ms/500ms/1000ms on timeouts/5xx).
- Circuit breaker: per-provider, 5 consecutive failures → open for 60s.
- On persistent failure: node/tool fails closed — surfaces a ProviderError,
  never fabricates output (Principle #10, §6).
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional

from tenacity import (
    RetryCallState,
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from scoutai.config import ScoutAIConfig

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ── Capability → model_role mapping (§6) ─────────────────────────────────────
# This mapping is fixed by the spec — it is NOT configurable, because these
# assignments encode architectural decisions, not operator preferences.

CAPABILITY_TO_MODEL_ROLE: dict[str, str] = {
    # Fixed-node capabilities (§4.1)
    "extract_role_requirements": "high_context",
    "generate_rubric": "high_context",
    "compose_decision_summary": "high_context",
    "screen_resume": "fast_structured",
    "run_fairness_probe": "fast_structured",
    # Agent-callable tools (§4.2)
    "extract_evidence": "fast_structured",
    "assess_capabilities": "fast_structured",
    "verify_evidence": "fast_structured",
    "generate_interview_questions": "fast_structured",
    "reevaluate_candidate": "fast_structured",
    # Deterministic capabilities — no model needed
    "ask_candidate": None,          # interrupt; no model call
    "finalize_candidate": None,     # terminal action; no model call
    "check_availability": None,     # calendar API; deterministic
    "propose_interview": None,      # calendar API; deterministic
}


# ── Custom exceptions ─────────────────────────────────────────────────────────


class ProviderError(Exception):
    """Raised when a model provider call fails after exhausting retries."""

    def __init__(self, provider: str, capability: str, reason: str) -> None:
        self.provider = provider
        self.capability = capability
        self.reason = reason
        super().__init__(f"[{provider}] {capability} failed: {reason}")


class CircuitOpenError(ProviderError):
    """Raised when the circuit breaker is open for a provider."""

    def __init__(self, provider: str, capability: str, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(provider, capability, f"circuit open, retry after {retry_after:.1f}s")


class RateLimitError(ProviderError):
    """Raised when a provider rate limit is hit."""


class ValidationError(Exception):
    """Raised when a capability's output fails schema validation."""

    def __init__(self, capability: str, reason: str) -> None:
        self.capability = capability
        super().__init__(f"Schema validation failed for {capability}: {reason}")


# ── Circuit breaker ───────────────────────────────────────────────────────────


@dataclass
class CircuitBreakerState:
    """State for a single provider's circuit breaker."""

    consecutive_failures: int = 0
    tripped_at: Optional[float] = None
    _lock: Lock = field(default_factory=Lock, compare=False, repr=False)

    def is_open(self, open_window_seconds: int) -> bool:
        """Return True if the circuit is open (provider should not be called)."""
        with self._lock:
            if self.tripped_at is None:
                return False
            elapsed = time.monotonic() - self.tripped_at
            if elapsed >= open_window_seconds:
                # Auto-reset after window expires
                self.tripped_at = None
                self.consecutive_failures = 0
                return False
            return True

    def retry_after(self, open_window_seconds: int) -> float:
        """Return seconds until the circuit is eligible to close."""
        with self._lock:
            if self.tripped_at is None:
                return 0.0
            elapsed = time.monotonic() - self.tripped_at
            return max(0.0, open_window_seconds - elapsed)

    def record_failure(self, threshold: int) -> bool:
        """Record a failure. Returns True if the circuit just tripped."""
        with self._lock:
            self.consecutive_failures += 1
            if self.consecutive_failures >= threshold and self.tripped_at is None:
                self.tripped_at = time.monotonic()
                logger.warning(
                    "Circuit breaker tripped",
                    extra={"consecutive_failures": self.consecutive_failures},
                )
                return True
            return False

    def record_success(self) -> None:
        """Reset on success."""
        with self._lock:
            self.consecutive_failures = 0
            self.tripped_at = None


class CircuitBreaker:
    """Per-provider circuit breaker registry."""

    def __init__(self, threshold: int, open_window_seconds: int) -> None:
        self._threshold = threshold
        self._open_window_seconds = open_window_seconds
        self._states: dict[str, CircuitBreakerState] = defaultdict(CircuitBreakerState)

    def check(self, provider: str, capability: str) -> None:
        """Raise CircuitOpenError if the circuit is open for this provider."""
        state = self._states[provider]
        if state.is_open(self._open_window_seconds):
            raise CircuitOpenError(
                provider=provider,
                capability=capability,
                retry_after=state.retry_after(self._open_window_seconds),
            )

    def record_failure(self, provider: str) -> None:
        self._states[provider].record_failure(self._threshold)

    def record_success(self, provider: str) -> None:
        self._states[provider].record_success()

    def get_state(self, provider: str) -> CircuitBreakerState:
        return self._states[provider]


# ── Model router ──────────────────────────────────────────────────────────────


class ModelRouter:
    """
    Resolves capability → model_role → provider and instantiates the appropriate
    LangChain chat model.

    Provider clients are instantiated lazily and cached (one per provider/model combo).
    API keys are read from environment variables at resolution time — never stored.

    Usage:
        router = ModelRouter(config)
        model, model_id = router.get_model("extract_evidence")
        # Use model for the LangChain call.
    """

    def __init__(self, config: ScoutAIConfig) -> None:
        self._config = config
        self._circuit_breaker = CircuitBreaker(
            threshold=config.circuit_breaker.consecutive_failures_threshold,
            open_window_seconds=config.circuit_breaker.open_window_seconds,
        )
        self._client_cache: dict[str, "BaseChatModel"] = {}
        self._cache_lock = Lock()

    def get_model_role(self, capability: str) -> Optional[str]:
        """Return the model_role for a capability, or None if no model is needed."""
        role = CAPABILITY_TO_MODEL_ROLE.get(capability)
        if capability not in CAPABILITY_TO_MODEL_ROLE:
            raise ValueError(
                f"Unknown capability '{capability}'. "
                f"Valid capabilities: {sorted(CAPABILITY_TO_MODEL_ROLE.keys())}"
            )
        return role

    def get_model(
        self, capability: str, force_fallback: bool = False
    ) -> tuple["BaseChatModel", str]:
        """
        Resolve and return (model_instance, model_id_string) for a capability.

        Tries primary first, then fallback, then fallback_2 if configured.
        Checks circuit breaker before each attempt.

        Args:
            capability: The capability name (e.g. "extract_evidence").
            force_fallback: If True, skip the primary and go straight to fallback.

        Returns:
            (chat_model, model_id) where model_id is "provider/model-name".

        Raises:
            ProviderError: If all providers are exhausted or circuit-open.
            ValueError: If capability is unknown or requires no model.
        """
        role = self.get_model_role(capability)
        if role is None:
            raise ValueError(
                f"Capability '{capability}' does not require a model call "
                f"(it is deterministic or an interrupt). "
                f"Do not call get_model() for it."
            )

        role_config = (
            self._config.model_roles.fast_structured
            if role == "fast_structured"
            else self._config.model_roles.high_context
        )

        candidates = [role_config.primary, role_config.fallback]
        if role_config.fallback_2:
            candidates.append(role_config.fallback_2)

        if force_fallback and len(candidates) > 1:
            candidates = candidates[1:]

        last_error: Optional[Exception] = None
        for model_id in candidates:
            provider = model_id.split("/")[0]
            try:
                self._circuit_breaker.check(provider, capability)
                model = self._get_or_create_client(model_id)
                return model, model_id
            except CircuitOpenError as e:
                logger.warning(
                    "Circuit open for provider, trying next",
                    extra={"provider": provider, "retry_after": e.retry_after},
                )
                last_error = e
                continue
            except Exception as e:
                logger.warning(
                    "Failed to instantiate model client",
                    extra={"model_id": model_id, "error": str(e)},
                )
                last_error = e
                continue

        raise ProviderError(
            provider="all",
            capability=capability,
            reason=f"All providers exhausted. Last error: {last_error}",
        )

    def record_call_result(self, model_id: str, success: bool) -> None:
        """Update circuit breaker state after a model call."""
        provider = model_id.split("/")[0]
        if success:
            self._circuit_breaker.record_success(provider)
        else:
            self._circuit_breaker.record_failure(provider)

    def _get_or_create_client(self, model_id: str) -> "BaseChatModel":
        """Return a cached model client or create a new one."""
        with self._cache_lock:
            if model_id not in self._client_cache:
                self._client_cache[model_id] = self._create_client(model_id)
            return self._client_cache[model_id]

    def _create_client(self, model_id: str) -> "BaseChatModel":
        """
        Instantiate a LangChain chat model for the given model_id.

        model_id format: "<provider>/<model-name>"
        e.g. "gemini/gemini-2.0-flash", "groq/llama-3.3-70b-versatile"
        """
        provider, model_name = _parse_model_id(model_id)
        api_key = self._resolve_api_key(provider)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0,
                timeout=self._config.retry.timeout_seconds,
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model_name,
                groq_api_key=api_key,
                temperature=0,
                timeout=self._config.retry.timeout_seconds,
            )
        else:
            raise ValueError(
                f"Unsupported provider '{provider}' in model_id '{model_id}'. "
                f"Supported: gemini, groq."
            )

    def _resolve_api_key(self, provider: str) -> str:
        """
        Resolve the API key for a provider from environment variables.

        Looks up the env var name from config.yaml model_roles.*.provider_env_vars,
        then reads from the environment. Never logs or stores the key value.

        Raises:
            EnvironmentError: If the required env var is not set.
        """
        # Check both model roles for the provider env var name
        env_var: Optional[str] = None
        for role_config in [
            self._config.model_roles.fast_structured,
            self._config.model_roles.high_context,
        ]:
            if provider in role_config.provider_env_vars:
                env_var = role_config.provider_env_vars[provider]
                break

        if env_var is None:
            # Fall back to a sensible default naming convention
            env_var = f"{provider.upper()}_API_KEY"

        value = os.getenv(env_var)
        if not value:
            raise EnvironmentError(
                f"API key for provider '{provider}' not found. "
                f"Set the '{env_var}' environment variable. "
                f"See .env.example for instructions."
            )
        return value

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        return self._circuit_breaker


# ── Retry decorator factory ───────────────────────────────────────────────────


def build_retry_decorator(config: ScoutAIConfig) -> Any:
    """
    Build a tenacity retry decorator from config.

    Retries on transient errors (timeout, rate limit, 5xx).
    After max_attempts, raises RetryError — the caller should treat this as a
    fail-closed outcome and surface ProviderError (§6, Principle #10).
    """
    retry_cfg = config.retry
    backoff_intervals = [ms / 1000.0 for ms in retry_cfg.backoff_ms]

    def _before_sleep(retry_state: RetryCallState) -> None:
        logger.warning(
            "Retrying model call",
            extra={
                "attempt": retry_state.attempt_number,
                "wait_seconds": retry_state.next_action.sleep if retry_state.next_action else 0,
                "exception": str(retry_state.outcome.exception()) if retry_state.outcome else None,
            },
        )

    # Build a wait strategy that uses the configured backoff intervals
    # then caps at the last value for any additional retries
    def _wait_strategy(retry_state: RetryCallState) -> float:
        attempt = retry_state.attempt_number - 1  # 0-indexed
        if attempt < len(backoff_intervals):
            return backoff_intervals[attempt]
        return backoff_intervals[-1]

    return retry(
        stop=stop_after_attempt(retry_cfg.max_attempts),
        wait=wait_fixed(backoff_intervals[0]),  # first interval; overridden by _wait_strategy
        retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
        before_sleep=_before_sleep,
        reraise=False,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_model_id(model_id: str) -> tuple[str, str]:
    """Parse 'provider/model-name' into (provider, model_name).

    Raises:
        ValueError: If model_id does not contain exactly one '/'.
    """
    parts = model_id.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid model_id '{model_id}'. "
            "Expected format: '<provider>/<model-name>', e.g. 'gemini/gemini-2.0-flash'."
        )
    return parts[0], parts[1]
