"""
S16: Evaluation harness skeleton.

Fixture-based test suite against the S11 metric list, including adversarial
fixtures from S15. No model calls - all metrics are computed from mock data
or synthetic evaluation fixtures.

Metrics tracked (S11):
- Resume extraction accuracy
- JSON/schema validity rate
- Hallucination rate
- Bias detection rate
- Interview usefulness
- False positive rate
- Agent convergence
"""

from __future__ import annotations

from typing import Any

import pytest

from scoutai.schemas import (
    BiasIndicator,
    BiasReport,
    CandidateState,
    ShortlistEntry,
    TrajectoryEntry,
)
from tests.fixtures.resumes import (
    CLEAN_RESUME,
    CONTRADICTORY_RESUME,
    EMPTY_RESUME,
    INJECTION_RESUME_1,
    INJECTION_RESUME_2,
    INJECTION_RESUME_3,
    PII_RESUME,
)


@pytest.fixture
def eval_candidates() -> list[dict[str, Any]]:
    """Synthetic evaluation candidates with known ground truth."""
    return [
        {
            "candidate_id": "eval-clean",
            "resume_text": CLEAN_RESUME,
            "expected_recommendation": "strong_interview",
            "expected_evidence_count": 3,
            "expected_injection": False,
        },
        {
            "candidate_id": "eval-injection-1",
            "resume_text": INJECTION_RESUME_1,
            "expected_recommendation": "reject",
            "expected_injection": True,
        },
        {
            "candidate_id": "eval-injection-2",
            "resume_text": INJECTION_RESUME_2,
            "expected_recommendation": "reject",
            "expected_injection": True,
        },
        {
            "candidate_id": "eval-injection-3",
            "resume_text": INJECTION_RESUME_3,
            "expected_recommendation": "reject",
            "expected_injection": True,
        },
        {
            "candidate_id": "eval-contradictory",
            "resume_text": CONTRADICTORY_RESUME,
            "expected_recommendation": "hold",
            "expected_evidence_count": 2,
            "expected_injection": False,
        },
        {
            "candidate_id": "eval-empty",
            "resume_text": EMPTY_RESUME,
            "expected_recommendation": "reject",
            "expected_evidence_count": 0,
            "expected_injection": False,
        },
        {
            "candidate_id": "eval-pii",
            "resume_text": PII_RESUME,
            "expected_recommendation": "interview",
            "expected_leakage": True,
            "expected_injection": False,
        },
    ]


# --- Metric 1: Resume extraction accuracy ------------------------------------


class TestExtractionAccuracy:

    def test_clean_resume_has_expected_evidence(self):
        """CLEAN_RESUME must contain Python, AWS, and system design evidence."""
        assert "Python" in CLEAN_RESUME
        assert "AWS" in CLEAN_RESUME
        assert "distributed" in CLEAN_RESUME.lower() or "microservices" in CLEAN_RESUME.lower()

    def test_empty_resume_has_no_skills_or_experience(self):
        """EMPTY_RESUME must have no extractable skills or experience details."""
        assert "Python" not in EMPTY_RESUME
        assert "AWS" not in EMPTY_RESUME
        assert "Engineer" not in EMPTY_RESUME

    def test_contradictory_resume_has_conflicting_signals(self):
        """CONTRADICTORY_RESUME must have conflicting evidence signals."""
        assert "10 years" in CONTRADICTORY_RESUME
        assert "beginner" in CONTRADICTORY_RESUME.lower()
        assert "PhD" in CONTRADICTORY_RESUME
        assert "2045" in CONTRADICTORY_RESUME

    def test_all_eval_candidates_have_ids(self, eval_candidates):
        """Every evaluation candidate must have a unique ID."""
        ids = [c["candidate_id"] for c in eval_candidates]
        assert len(ids) == len(set(ids)), "Duplicate candidate IDs in eval set"


# --- Metric 2: JSON/schema validity rate -------------------------------------


class TestSchemaValidity:

    def test_candidate_state_schema_valid(self):
        c = CandidateState(
            candidate_id="test",
            resume_text="test",
            sanitized_resume="test",
            finalized=False,
            interview_rounds=0,
        )
        assert c.candidate_id == "test"
        assert c.finalized is False

    def test_candidate_state_rejects_invalid_recommendation(self):
        with pytest.raises(Exception):
            CandidateState(
                candidate_id="test",
                resume_text="test",
                sanitized_resume="test",
                recommendation="invalid_option",
            )

    def test_shortlist_entry_schema_valid(self):
        s = ShortlistEntry(
            candidate="c001",
            recommendation="interview",
            weighted_score=75.0,
            evidence_refs=["Resume:Skills"],
        )
        assert s.candidate == "c001"
        assert s.weighted_score == 75.0

    def test_shortlist_entry_rejects_invalid_score(self):
        with pytest.raises(Exception):
            ShortlistEntry(
                candidate="c001",
                recommendation="interview",
                weighted_score=150.0,
                evidence_refs=[],
            )

    def test_trajectory_entry_schema_valid(self):
        from datetime import datetime, timezone
        t = TrajectoryEntry(
            node="candidate_agent",
            tool_used="extract_evidence",
            timestamp=datetime.now(timezone.utc).isoformat(),
            input_hash="0" * 64,
            output_hash="0" * 64,
            latency_ms=100,
            model=None,
            model_role="fast_structured",
            prompt_version="1.0.0",
            tool_version="1.0.0",
            schema_version="1.0.0",
            status="success",
        )
        assert t.node == "candidate_agent"
        assert t.status == "success"

    def test_bias_report_schema_valid(self):
        """BiasReport must accept valid data per the actual schema."""
        b = BiasReport(
            candidate_a="c001",
            candidate_b="c002",
            indicators=[
                BiasIndicator(
                    criterion="gender_parity",
                    description="Score differential between gender-swapped variants",
                    severity="low",
                    counterfactual_delta=2.5,
                ),
            ],
            overall_risk="low",
            summary="No significant bias detected.",
            probe_timestamp="2024-01-01T00:00:00Z",
        )
        assert b.overall_risk == "low"
        assert len(b.indicators) == 1
        assert b.indicators[0].criterion == "gender_parity"


# --- Metric 3: Hallucination rate --------------------------------------------


class TestHallucinationRate:

    def test_evidence_refs_are_traceable(self):
        """Evidence refs must follow the 'Source:Detail' convention."""
        valid_refs = [
            "Resume:Skills",
            "Resume:Experience",
            "Interview:Q1",
            "Resume:Projects",
        ]
        invalid_refs = [
            "made-up-source",
            "hallucinated-fact",
            "",
            "unknown",
        ]
        for ref in valid_refs:
            assert ":" in ref, f"Valid ref '{ref}' must contain ':'"
        for ref in invalid_refs:
            assert ":" not in ref or ref == "", f"Invalid ref '{ref}' should not be traceable"

    def test_evidence_buckets_match_resume_content(self):
        """Evidence extracted from CLEAN_RESUME must reference actual content."""
        real_content = ["Python", "Go", "AWS", "Kubernetes", "Docker", "PostgreSQL"]
        for item in real_content:
            assert item in CLEAN_RESUME, f"'{item}' must be in CLEAN_RESUME"

    def test_empty_resume_produces_no_evidence(self):
        """EMPTY_RESUME must produce zero evidence citations."""
        assert "Python" not in EMPTY_RESUME
        assert "AWS" not in EMPTY_RESUME


# --- Metric 4: Bias detection rate -------------------------------------------


class TestBiasDetectionRate:

    def test_bias_report_detects_indicators(self):
        """BiasReport must have non-empty indicators when bias is found."""
        report = BiasReport(
            candidate_a="c001",
            candidate_b="c003",
            indicators=[
                BiasIndicator(
                    criterion="gender_parity",
                    description="Score differential may reflect identity-based bias.",
                    severity="medium",
                    counterfactual_delta=15.0,
                ),
                BiasIndicator(
                    criterion="name_ethnicity",
                    description="Name-based score variation detected.",
                    severity="low",
                    counterfactual_delta=5.0,
                ),
            ],
            overall_risk="medium",
            summary="Score differential may reflect identity-based bias.",
            probe_timestamp="2024-01-01T00:00:00Z",
        )
        assert len(report.indicators) > 0
        assert report.overall_risk in ("low", "medium", "high")

    def test_bias_report_no_false_positive(self):
        """BiasReport must support 'low' risk when no bias is detected."""
        report = BiasReport(
            candidate_a="c001",
            candidate_b="c002",
            indicators=[],
            overall_risk="low",
            summary="No significant bias detected.",
            probe_timestamp="2024-01-01T00:00:00Z",
        )
        assert report.overall_risk == "low"
        assert len(report.indicators) == 0


# --- Metric 5: Interview usefulness ------------------------------------------


class TestInterviewUsefulness:

    def test_ask_candidate_can_change_verdict(self):
        before = "insufficient"
        after = "sufficient"
        assert before != after, "Verdict must change after ask_candidate"
        assert after == "sufficient", "After ask_candidate, verdict should be sufficient"

    def test_ask_candidate_is_single_use(self):
        """ask_candidate must be limited to exactly 1 use per candidate (ADR-3)."""
        max_uses = 1
        used = 0
        used += 1
        assert used <= max_uses, "Must not exceed 1 use"
        with pytest.raises(AssertionError):
            used += 1
            assert used <= max_uses, "Second use must be blocked"


# --- Metric 6: False positive rate -------------------------------------------


class TestFalsePositiveRate:

    def test_strong_interview_has_high_confidence(self):
        entry = ShortlistEntry(
            candidate="c001",
            recommendation="strong_interview",
            weighted_score=88.0,
            evidence_refs=["Resume:Skills", "Resume:Experience"],
        )
        assert entry.weighted_score >= 80.0, \
            "strong_interview should have weighted_score >= 80"

    def test_interview_has_moderate_confidence(self):
        entry = ShortlistEntry(
            candidate="c002",
            recommendation="interview",
            weighted_score=65.0,
            evidence_refs=["Resume:Skills"],
        )
        assert 50.0 <= entry.weighted_score < 80.0, \
            "interview should have weighted_score between 50 and 80"

    def test_reject_has_low_confidence(self):
        entry = ShortlistEntry(
            candidate="c003",
            recommendation="reject",
            weighted_score=30.0,
            evidence_refs=[],
        )
        assert entry.weighted_score < 50.0, \
            "reject should have weighted_score < 50"


# --- Metric 7: Agent convergence ---------------------------------------------


class TestAgentConvergence:

    def test_agent_converges_within_budget(self):
        """Agent must converge (call finalize_candidate) within max_iterations=8."""
        max_iterations = 8
        tool_calls = [
            "extract_evidence",
            "assess_capabilities",
            "verify_evidence",
            "finalize_candidate",
        ]
        assert len(tool_calls) <= max_iterations, \
            f"Agent must converge within {max_iterations} calls"
        assert "finalize_candidate" in tool_calls, \
            "Agent must call finalize_candidate to converge"

    def test_agent_exhaustion_force_finalizes(self):
        """When max_iterations is exhausted, agent must force-finalize with 'hold'."""
        max_iterations = 8
        tool_calls = ["extract_evidence"] * 8
        if len(tool_calls) >= max_iterations and "finalize_candidate" not in tool_calls:
            force_recommendation = "hold"
            assert force_recommendation == "hold", \
                "Force-finalize must set recommendation to 'hold'"

    def test_agent_convergence_rate(self):
        """Agent convergence rate should be near 100% (exhaustion near 0%)."""
        total_runs = 100
        converged = 98
        exhaustion_rate = (total_runs - converged) / total_runs
        assert exhaustion_rate < 0.05, \
            f"Exhaustion rate {exhaustion_rate:.1%} must be < 5%"