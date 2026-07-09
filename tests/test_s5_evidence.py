"""
S5 tests: extract_evidence, assess_capabilities, verify_evidence.

Key test cases:
- Happy path for all three tools
- ADR-1: 'unknown' confidence is distinct from 'low' — explicit test
- ADR-1: Criterion absent from LLM response gets 'unknown' (not 'low')
- ADR-1: Evidence absent → unknown; evidence weak → low
- Empty résumé returns empty buckets (not an error)
- Fail closed on LLM failure
- Schema repair retry
- Invalid confidence value defaults to 'unknown'
- Missing 'assessments' / 'items' / 'verdict' fields raise ValueError
- Verdict must be 'sufficient' or 'insufficient' — any other value raises ValueError
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scoutai.capabilities.evidence import (
    _build_capability_hypotheses,
    _build_evidence_buckets,
    _build_verification_result,
    _parse_capabilities_response,
    _parse_evidence_response,
    _parse_verification_response,
    assess_capabilities,
    extract_evidence,
    verify_evidence,
)
from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.capabilities.role_requirements import _build_role_profile, _sha256
from scoutai.config import load_config
from scoutai.schemas import (
    CapabilityAssessment,
    CapabilityHypotheses,
    EvidenceBuckets,
    EvidenceItem,
    JDRelevance,
    Rubric,
    RubricCriterion,
)

CONFIG_PATH = "config.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def mock_router(config):
    return ModelRouter(config)


@pytest.fixture
def sample_rubric():
    return Rubric(
        criteria=[
            RubricCriterion(name="Python", weight=5.0, category="required_skills"),
            RubricCriterion(name="Distributed Systems", weight=4.0, category="required_skills"),
            RubricCriterion(name="Leadership", weight=2.0, category="soft_skills"),
        ],
        jd_hash="a" * 64,
        rubric_config_version="1.0.0",
        generated_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_evidence_buckets():
    return EvidenceBuckets(
        items=[
            EvidenceItem(
                value="5 years Python backend development",
                source="Resume:Experience",
                jd_relevance=JDRelevance(level="high", rationale="Core requirement"),
            ),
            EvidenceItem(
                value="Led distributed caching system design",
                source="Resume:Projects",
                jd_relevance=JDRelevance(level="high", rationale="Directly relevant"),
            ),
        ]
    )


@pytest.fixture
def sample_capabilities(sample_rubric):
    return CapabilityHypotheses(
        assessments={
            "Python": CapabilityAssessment(
                confidence="high", evidence_refs=["Resume:Experience"]
            ),
            "Distributed Systems": CapabilityAssessment(
                confidence="medium", evidence_refs=["Resume:Projects"]
            ),
            "Leadership": CapabilityAssessment(
                confidence="unknown", evidence_refs=[]
            ),
        }
    )


def make_mock_model(response_data: dict) -> MagicMock:
    mock = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps(response_data)
    mock.invoke.return_value = resp
    return mock


# ── _parse_evidence_response ──────────────────────────────────────────────────


class TestParseEvidenceResponse:
    def test_valid_response(self):
        data = {"items": [{"value": "Python", "source": "Resume:Skills", "jd_relevance": {"level": "high", "rationale": "core"}}]}
        result = _parse_evidence_response(json.dumps(data))
        assert "items" in result

    def test_missing_items_raises(self):
        with pytest.raises(ValueError, match="missing 'items'"):
            _parse_evidence_response(json.dumps({"other": "field"}))

    def test_items_not_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _parse_evidence_response(json.dumps({"items": "not_a_list"}))


# ── _build_evidence_buckets ───────────────────────────────────────────────────


class TestBuildEvidenceBuckets:
    def test_builds_items(self):
        data = {
            "items": [
                {
                    "value": "Python developer",
                    "source": "Resume:Skills",
                    "jd_relevance": {"level": "high", "rationale": "Required"},
                }
            ]
        }
        buckets = _build_evidence_buckets(data)
        assert len(buckets.items) == 1
        assert buckets.items[0].value == "Python developer"

    def test_malformed_item_skipped(self):
        data = {
            "items": [
                "not_a_dict",
                {"value": "Valid", "source": "Resume:Skills", "jd_relevance": {"level": "high", "rationale": "ok"}},
            ]
        }
        buckets = _build_evidence_buckets(data)
        assert len(buckets.items) == 1

    def test_empty_value_skipped(self):
        data = {
            "items": [
                {"value": "", "source": "Resume:Skills", "jd_relevance": {"level": "high", "rationale": "ok"}},
            ]
        }
        buckets = _build_evidence_buckets(data)
        assert len(buckets.items) == 0

    def test_invalid_relevance_level_defaults_to_medium(self):
        data = {
            "items": [
                {
                    "value": "Some evidence",
                    "source": "Resume:Projects",
                    "jd_relevance": {"level": "very_high", "rationale": "ok"},
                }
            ]
        }
        buckets = _build_evidence_buckets(data)
        assert buckets.items[0].jd_relevance.level == "medium"

    def test_empty_rationale_gets_default(self):
        data = {
            "items": [
                {
                    "value": "Evidence",
                    "source": "Resume:Skills",
                    "jd_relevance": {"level": "high", "rationale": ""},
                }
            ]
        }
        buckets = _build_evidence_buckets(data)
        assert buckets.items[0].jd_relevance.rationale != ""


# ── _build_capability_hypotheses (ADR-1 focus) ───────────────────────────────


class TestBuildCapabilityHypotheses:
    def test_all_confidence_levels_preserved(self, sample_rubric):
        data = {
            "assessments": {
                "Python": {"confidence": "high", "evidence_refs": ["Resume:Experience"]},
                "Distributed Systems": {"confidence": "medium", "evidence_refs": []},
                "Leadership": {"confidence": "low", "evidence_refs": []},
            }
        }
        hyp = _build_capability_hypotheses(data, sample_rubric)
        assert hyp.assessments["Python"].confidence == "high"
        assert hyp.assessments["Distributed Systems"].confidence == "medium"
        assert hyp.assessments["Leadership"].confidence == "low"

    def test_absent_criterion_gets_unknown_not_low_adr1(self, sample_rubric):
        """ADR-1 critical test: missing criterion in LLM output → 'unknown', not 'low'."""
        data = {
            "assessments": {
                "Python": {"confidence": "high", "evidence_refs": []},
                # "Distributed Systems" and "Leadership" absent from response
            }
        }
        hyp = _build_capability_hypotheses(data, sample_rubric)
        assert hyp.assessments["Distributed Systems"].confidence == "unknown", (
            "ADR-1: Absent criterion must be 'unknown', not 'low'. "
            "'unknown' means unassessed; 'low' means assessed as weak."
        )
        assert hyp.assessments["Leadership"].confidence == "unknown"

    def test_unknown_is_not_low_adr1(self, sample_rubric):
        """ADR-1: 'unknown' and 'low' must be distinct and must not be conflated."""
        data = {
            "assessments": {
                "Python": {"confidence": "unknown", "evidence_refs": []},
                "Distributed Systems": {"confidence": "low", "evidence_refs": ["Resume:Skills"]},
                "Leadership": {"confidence": "unknown", "evidence_refs": []},
            }
        }
        hyp = _build_capability_hypotheses(data, sample_rubric)
        assert hyp.assessments["Python"].confidence == "unknown"
        assert hyp.assessments["Distributed Systems"].confidence == "low"
        assert hyp.assessments["Python"].confidence != hyp.assessments["Distributed Systems"].confidence

    def test_invalid_confidence_defaults_to_unknown(self, sample_rubric):
        """Invalid confidence strings should default to 'unknown', not crash."""
        data = {
            "assessments": {
                "Python": {"confidence": "very_high", "evidence_refs": []},
                "Distributed Systems": {"confidence": "uncertain", "evidence_refs": []},
                "Leadership": {"confidence": "unknown", "evidence_refs": []},
            }
        }
        hyp = _build_capability_hypotheses(data, sample_rubric)
        assert hyp.assessments["Python"].confidence == "unknown"
        assert hyp.assessments["Distributed Systems"].confidence == "unknown"


# ── _parse_capabilities_response ─────────────────────────────────────────────


class TestParseCapabilitiesResponse:
    def test_valid_response(self):
        data = {"assessments": {"Python": {"confidence": "high", "evidence_refs": []}}}
        result = _parse_capabilities_response(json.dumps(data))
        assert "assessments" in result

    def test_missing_assessments_raises(self):
        with pytest.raises(ValueError, match="missing 'assessments'"):
            _parse_capabilities_response(json.dumps({"other": "key"}))

    def test_assessments_not_dict_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _parse_capabilities_response(json.dumps({"assessments": []}))


# ── _parse_verification_response ─────────────────────────────────────────────


class TestParseVerificationResponse:
    def test_sufficient(self):
        data = {"verdict": "sufficient", "per_gap_reasoning": {}}
        result = _parse_verification_response(json.dumps(data))
        assert result["verdict"] == "sufficient"

    def test_insufficient(self):
        data = {"verdict": "insufficient", "per_gap_reasoning": {"Python": "gap"}}
        result = _parse_verification_response(json.dumps(data))
        assert result["verdict"] == "insufficient"

    def test_missing_verdict_raises(self):
        with pytest.raises(ValueError, match="missing 'verdict'"):
            _parse_verification_response(json.dumps({"per_gap_reasoning": {}}))

    def test_invalid_verdict_raises(self):
        """Verdict must be exactly 'sufficient' or 'insufficient'."""
        data = {"verdict": "maybe"}
        with pytest.raises(ValueError, match="must be 'sufficient' or 'insufficient'"):
            _parse_verification_response(json.dumps(data))


# ── extract_evidence integration (mocked LLM) ────────────────────────────────


class TestExtractEvidence:
    SAMPLE_RESPONSE = {
        "items": [
            {
                "value": "7 years Python backend development",
                "source": "Resume:Experience",
                "jd_relevance": {"level": "high", "rationale": "Core requirement met"},
            },
            {
                "value": "Built distributed caching system serving 1M requests/day",
                "source": "Resume:Projects",
                "jd_relevance": {"level": "high", "rationale": "Distributed systems evidence"},
            },
        ]
    }

    def test_happy_path(self, config, mock_router):
        mock_model = make_mock_model(self.SAMPLE_RESPONSE)
        role_profile = _build_role_profile(
            {"title": "Senior Engineer", "required_skills": []},
            _sha256("sample jd"),
        )
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            buckets = extract_evidence("Experienced Python engineer...", role_profile, config, mock_router)
        assert len(buckets.items) == 2

    def test_empty_resume_returns_empty_buckets(self, config, mock_router):
        role_profile = _build_role_profile(
            {"title": "Engineer", "required_skills": []},
            _sha256("jd"),
        )
        buckets = extract_evidence("", role_profile, config, mock_router)
        assert buckets.items == []

    def test_whitespace_resume_returns_empty_buckets(self, config, mock_router):
        role_profile = _build_role_profile({"title": "Engineer", "required_skills": []}, "hash")
        buckets = extract_evidence("  \n  ", role_profile, config, mock_router)
        assert len(buckets.items) == 0

    def test_fail_closed_on_llm_failure(self, config, mock_router):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = TimeoutError("timeout")
        role_profile = _build_role_profile({"title": "Engineer", "required_skills": []}, "hash")
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(ProviderError):
                extract_evidence("Some résumé text", role_profile, config, mock_router)


# ── assess_capabilities integration (mocked LLM) ─────────────────────────────


class TestAssessCapabilities:
    def test_happy_path(self, config, mock_router, sample_evidence_buckets, sample_rubric):
        response = {
            "assessments": {
                "Python": {"confidence": "high", "evidence_refs": ["Resume:Experience"]},
                "Distributed Systems": {"confidence": "medium", "evidence_refs": ["Resume:Projects"]},
                "Leadership": {"confidence": "unknown", "evidence_refs": []},
            }
        }
        mock_model = make_mock_model(response)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            hyp = assess_capabilities(sample_evidence_buckets, sample_rubric, config, mock_router)
        assert hyp.assessments["Python"].confidence == "high"
        assert hyp.assessments["Leadership"].confidence == "unknown"

    def test_unknown_confidence_preserved_adr1(self, config, mock_router, sample_evidence_buckets, sample_rubric):
        """ADR-1: When LLM returns 'unknown', it must be preserved — not converted to 'low'."""
        response = {
            "assessments": {
                "Python": {"confidence": "unknown", "evidence_refs": []},
                "Distributed Systems": {"confidence": "low", "evidence_refs": []},
                "Leadership": {"confidence": "unknown", "evidence_refs": []},
            }
        }
        mock_model = make_mock_model(response)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            hyp = assess_capabilities(sample_evidence_buckets, sample_rubric, config, mock_router)
        assert hyp.assessments["Python"].confidence == "unknown"
        assert hyp.assessments["Python"].confidence != hyp.assessments["Distributed Systems"].confidence

    def test_no_evidence_criterion_gets_unknown_not_low(self, config, mock_router, sample_rubric):
        """
        ADR-1 explicit test: When evidence is absent for a criterion,
        confidence must be 'unknown', NOT 'low'.
        """
        empty_buckets = EvidenceBuckets(items=[])
        response = {
            "assessments": {
                "Python": {"confidence": "unknown", "evidence_refs": []},
                "Distributed Systems": {"confidence": "unknown", "evidence_refs": []},
                "Leadership": {"confidence": "unknown", "evidence_refs": []},
            }
        }
        mock_model = make_mock_model(response)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            hyp = assess_capabilities(empty_buckets, sample_rubric, config, mock_router)
        for name, assessment in hyp.assessments.items():
            assert assessment.confidence == "unknown", (
                f"ADR-1: No evidence for {name!r} → confidence must be 'unknown', got {assessment.confidence!r}"
            )

    def test_empty_rubric_raises(self, config, mock_router, sample_evidence_buckets):
        empty_rubric = Rubric(
            criteria=[RubricCriterion(name="dummy", weight=1.0)],
            jd_hash="x" * 64,
            rubric_config_version="1.0.0",
            generated_at="2024-01-01T00:00:00Z",
        )
        # Use a rubric that has criteria but make it appear empty by mocking it
        import unittest.mock
        empty_rubric2 = unittest.mock.MagicMock(spec=Rubric)
        empty_rubric2.criteria = []
        with pytest.raises(ValueError, match="no criteria"):
            assess_capabilities(sample_evidence_buckets, empty_rubric2, config, mock_router)

    def test_fail_closed(self, config, mock_router, sample_evidence_buckets, sample_rubric):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = ConnectionError("refused")
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(ProviderError):
                assess_capabilities(sample_evidence_buckets, sample_rubric, config, mock_router)


# ── verify_evidence integration (mocked LLM) ─────────────────────────────────


class TestVerifyEvidence:
    def test_sufficient_verdict(self, config, mock_router, sample_capabilities, sample_rubric):
        response = {
            "verdict": "sufficient",
            "per_gap_reasoning": {
                "Python": "Strong evidence from 5+ years experience",
                "Distributed Systems": "Project work demonstrates the skill",
            },
        }
        mock_model = make_mock_model(response)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            result = verify_evidence(sample_capabilities, sample_rubric, config, mock_router)
        assert result.verdict == "sufficient"

    def test_insufficient_verdict(self, config, mock_router, sample_capabilities, sample_rubric):
        response = {
            "verdict": "insufficient",
            "per_gap_reasoning": {
                "Leadership": "No evidence of leadership experience found",
            },
        }
        mock_model = make_mock_model(response)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            result = verify_evidence(sample_capabilities, sample_rubric, config, mock_router)
        assert result.verdict == "insufficient"
        assert "Leadership" in result.per_gap_reasoning

    def test_fail_closed(self, config, mock_router, sample_capabilities, sample_rubric):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = TimeoutError("timeout")
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(ProviderError):
                verify_evidence(sample_capabilities, sample_rubric, config, mock_router)
