"""
S2 unit tests: Model router — capability→role mapping, circuit breaker, fallback logic.

All tests use mocked clients — no live API keys are required.
Tests cover:
- capability → model_role resolution
- invalid capability raises ValueError
- deterministic capabilities return role=None
- circuit breaker trips and opens on consecutive failures
- circuit breaker auto-resets after window expires
- fallback provider is used when primary circuit is open
- _parse_model_id validation
- retry decorator is constructable from config
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from scoutai.capabilities.model_router import (
    CAPABILITY_TO_MODEL_ROLE,
    CircuitBreaker,
    CircuitBreakerState,
    CircuitOpenError,
    ModelRouter,
    ProviderError,
    _parse_model_id,
    build_retry_decorator,
)
from scoutai.config import load_config

CONFIG_PATH = "config.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def router(config):
    return ModelRouter(config)


# ── Capability → model_role mapping ──────────────────────────────────────────


class TestCapabilityRoleMapping:
    def test_high_context_capabilities(self, router):
        for cap in ["extract_role_requirements", "generate_rubric", "compose_decision_summary"]:
            assert router.get_model_role(cap) == "high_context", f"Expected high_context for {cap}"

    def test_fast_structured_capabilities(self, router):
        fast_caps = [
            "screen_resume",
            "extract_evidence",
            "assess_capabilities",
            "verify_evidence",
            "generate_interview_questions",
            "reevaluate_candidate",
            "run_fairness_probe",
        ]
        for cap in fast_caps:
            assert router.get_model_role(cap) == "fast_structured", f"Expected fast_structured for {cap}"

    def test_no_model_capabilities_return_none(self, router):
        """Deterministic/interrupt capabilities don't need a model."""
        for cap in ["ask_candidate", "finalize_candidate", "check_availability", "propose_interview"]:
            assert router.get_model_role(cap) is None, f"Expected None for {cap}"

    def test_unknown_capability_raises(self, router):
        with pytest.raises(ValueError, match="Unknown capability"):
            router.get_model_role("nonexistent_capability")

    def test_all_capabilities_in_mapping(self):
        """Every key in the mapping must be in one of the three buckets."""
        high_ctx = {"extract_role_requirements", "generate_rubric", "compose_decision_summary"}
        fast = {
            "screen_resume", "extract_evidence", "assess_capabilities", "verify_evidence",
            "generate_interview_questions", "reevaluate_candidate", "run_fairness_probe",
        }
        no_model = {"ask_candidate", "finalize_candidate", "check_availability", "propose_interview"}
        all_defined = high_ctx | fast | no_model
        assert set(CAPABILITY_TO_MODEL_ROLE.keys()) == all_defined


# ── get_model raises for no-model capabilities ───────────────────────────────


class TestGetModelNoModel:
    def test_ask_candidate_raises(self, router):
        with pytest.raises(ValueError, match="does not require a model call"):
            router.get_model("ask_candidate")

    def test_finalize_candidate_raises(self, router):
        with pytest.raises(ValueError, match="does not require a model call"):
            router.get_model("finalize_candidate")


# ── Circuit breaker ───────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker(threshold=5, open_window_seconds=60)
        cb.check("gemini", "test_cap")  # should not raise

    def test_trips_after_threshold_failures(self):
        cb = CircuitBreaker(threshold=3, open_window_seconds=60)
        for _ in range(3):
            cb.record_failure("gemini")
        with pytest.raises(CircuitOpenError):
            cb.check("gemini", "extract_evidence")

    def test_does_not_trip_before_threshold(self):
        cb = CircuitBreaker(threshold=5, open_window_seconds=60)
        for _ in range(4):
            cb.record_failure("gemini")
        cb.check("gemini", "extract_evidence")  # should not raise — still 1 below threshold

    def test_success_resets_state(self):
        cb = CircuitBreaker(threshold=3, open_window_seconds=60)
        for _ in range(2):
            cb.record_failure("gemini")
        cb.record_success("gemini")
        state = cb.get_state("gemini")
        assert state.consecutive_failures == 0

    def test_auto_resets_after_window(self):
        cb = CircuitBreaker(threshold=1, open_window_seconds=1)
        cb.record_failure("gemini")
        # Circuit should be open immediately
        with pytest.raises(CircuitOpenError):
            cb.check("gemini", "test")
        # Wait for window to expire
        time.sleep(1.1)
        cb.check("gemini", "test")  # should not raise now

    def test_independent_per_provider(self):
        cb = CircuitBreaker(threshold=3, open_window_seconds=60)
        for _ in range(3):
            cb.record_failure("gemini")
        # groq should still be closed
        cb.check("groq", "extract_evidence")  # should not raise

    def test_retry_after_is_positive_when_open(self):
        cb = CircuitBreaker(threshold=1, open_window_seconds=60)
        cb.record_failure("gemini")
        state = cb.get_state("gemini")
        assert state.retry_after(60) > 0.0


# ── ModelRouter fallback logic ────────────────────────────────────────────────


class TestModelRouterFallback:
    def test_uses_fallback_when_primary_circuit_open(self, config):
        """When primary provider's circuit is open, router should use fallback."""
        router = ModelRouter(config)

        # Trip the primary provider's circuit
        for _ in range(config.circuit_breaker.consecutive_failures_threshold):
            router.circuit_breaker.record_failure("gemini")

        # Mock the _create_client method to avoid real HTTP calls
        mock_model = MagicMock()
        with patch.object(router, "_create_client", return_value=mock_model) as mock_create:
            model, model_id = router.get_model("extract_evidence")
            # Should have called _create_client with the fallback, not primary
            # Primary is "gemini/..." — fallback is "groq/..."
            called_id = mock_create.call_args[0][0]
            assert called_id.startswith("groq/"), f"Expected groq fallback, got {called_id}"

    def test_uses_primary_when_circuit_closed(self, config):
        """When no circuit is open, router uses the primary model."""
        router = ModelRouter(config)
        mock_model = MagicMock()
        with patch.object(router, "_create_client", return_value=mock_model) as mock_create:
            model, model_id = router.get_model("extract_evidence")
            called_id = mock_create.call_args[0][0]
            assert called_id == config.model_roles.fast_structured.primary

    def test_raises_when_all_providers_exhausted(self, config):
        """If all providers are circuit-open, ProviderError should be raised."""
        router = ModelRouter(config)
        threshold = config.circuit_breaker.consecutive_failures_threshold

        # Trip all possible providers
        for provider in ["gemini", "groq"]:
            for _ in range(threshold):
                router.circuit_breaker.record_failure(provider)

        with pytest.raises(ProviderError):
            router.get_model("extract_evidence")

    def test_force_fallback_skips_primary(self, config):
        """force_fallback=True should skip the primary provider."""
        router = ModelRouter(config)
        mock_model = MagicMock()
        with patch.object(router, "_create_client", return_value=mock_model) as mock_create:
            model, model_id = router.get_model("extract_evidence", force_fallback=True)
            called_id = mock_create.call_args[0][0]
            assert called_id != config.model_roles.fast_structured.primary


# ── record_call_result ────────────────────────────────────────────────────────


class TestRecordCallResult:
    def test_record_success_resets_failures(self, router):
        router.circuit_breaker.record_failure("gemini")
        router.record_call_result("gemini/gemini-2.0-flash", success=True)
        state = router.circuit_breaker.get_state("gemini")
        assert state.consecutive_failures == 0

    def test_record_failure_increments_count(self, router):
        router.record_call_result("gemini/gemini-2.0-flash", success=False)
        state = router.circuit_breaker.get_state("gemini")
        assert state.consecutive_failures == 1


# ── _parse_model_id ───────────────────────────────────────────────────────────


class TestParseModelId:
    def test_valid_gemini(self):
        provider, model = _parse_model_id("gemini/gemini-2.0-flash")
        assert provider == "gemini"
        assert model == "gemini-2.0-flash"

    def test_valid_groq(self):
        provider, model = _parse_model_id("groq/llama-3.3-70b-versatile")
        assert provider == "groq"
        assert model == "llama-3.3-70b-versatile"

    def test_no_slash_raises(self):
        with pytest.raises(ValueError, match="Invalid model_id"):
            _parse_model_id("gemini-flash")

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError, match="Invalid model_id"):
            _parse_model_id("/model-name")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError, match="Invalid model_id"):
            _parse_model_id("gemini/")

    def test_model_with_slashes_in_name(self):
        """Model names may not have additional slashes — first split only."""
        provider, model = _parse_model_id("groq/llama-3/70b")
        assert provider == "groq"
        assert model == "llama-3/70b"


# ── Retry decorator ───────────────────────────────────────────────────────────


class TestBuildRetryDecorator:
    def test_builds_successfully(self, config):
        decorator = build_retry_decorator(config)
        assert decorator is not None

    def test_decorated_function_retries(self, config):
        """A decorated function that raises TransientError should be retried."""
        decorator = build_retry_decorator(config)
        call_count = 0

        @decorator
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < config.retry.max_attempts:
                raise TimeoutError("transient")
            return "ok"

        result = flaky_function()
        assert result == "ok"
        assert call_count == config.retry.max_attempts
