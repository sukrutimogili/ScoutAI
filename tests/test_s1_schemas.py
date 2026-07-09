"""
S1 unit tests: Pydantic schema round-trips and enum boundary checks.

Tests cover:
- All schema types instantiate cleanly with valid data
- Enum boundaries (Confidence, Recommendation, Relevance) reject invalid values
- Validators fire correctly (rubric uniqueness, evidence_refs, weights, bounds)
- Round-trip serialization preserves all fields
- Spec-mandated constraints (ADR-1, ADR-3, §5)
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from scoutai.schemas import (
    AvailabilitySlot,
    BiasIndicator,
    BiasReport,
    CapabilityAssessment,
    CapabilityHypotheses,
    CandidateState,
    EvidenceBuckets,
    EvidenceItem,
    GraphState,
    InterviewQuestion,
    JDRelevance,
    PendingConfirmation,
    QuestionSet,
    RecruiterSummary,
    RequiredSkill,
    RoleProfile,
    Rubric,
    RubricCriterion,
    ScoreEntry,
    ShortlistEntry,
    TrajectoryEntry,
    UpdatedScoreCard,
    VerificationResult,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_trajectory_entry(**overrides: object) -> TrajectoryEntry:
    defaults: dict = {
        "node": "screen_resume",
        "tool_used": None,
        "timestamp": "2024-01-01T00:00:00Z",
        "input_hash": "a" * 64,
        "output_hash": "b" * 64,
        "latency_ms": 100,
        "model": "gemini/gemini-2.0-flash",
        "model_role": "fast_structured",
        "prompt_version": "1.0.0",
        "tool_version": "1.0.0",
        "schema_version": "1.0.0",
        "status": "success",
    }
    defaults.update(overrides)
    return TrajectoryEntry(**defaults)


def make_evidence_item(**overrides: object) -> EvidenceItem:
    defaults: dict = {
        "value": "Built distributed systems at scale",
        "source": "Resume:Experience",
        "jd_relevance": JDRelevance(level="high", rationale="Core requirement in JD"),
    }
    defaults.update(overrides)
    return EvidenceItem(**defaults)


def make_shortlist_entry(**overrides: object) -> ShortlistEntry:
    defaults: dict = {
        "candidate": "candidate_001",
        "recommendation": "interview",
        "weighted_score": 75.0,
        "confidence_summary": {"python": "high"},
        "strengths": ["Strong Python background"],
        "remaining_uncertainties": [],
        "evidence_refs": ["Resume:Experience:1"],
    }
    defaults.update(overrides)
    return ShortlistEntry(**defaults)


# ── JDRelevance ───────────────────────────────────────────────────────────────


class TestJDRelevance:
    def test_valid_levels(self) -> None:
        for level in ("high", "medium", "low"):
            r = JDRelevance(level=level, rationale="test")  # type: ignore[arg-type]
            assert r.level == level

    def test_invalid_level(self) -> None:
        with pytest.raises(ValidationError):
            JDRelevance(level="critical", rationale="test")  # type: ignore[arg-type]

    def test_empty_rationale_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JDRelevance(level="high", rationale="")

    def test_round_trip(self) -> None:
        r = JDRelevance(level="medium", rationale="Somewhat relevant")
        assert JDRelevance.model_validate_json(r.model_dump_json()) == r


# ── EvidenceItem ──────────────────────────────────────────────────────────────


class TestEvidenceItem:
    def test_valid_item(self) -> None:
        item = make_evidence_item()
        assert item.source == "Resume:Experience"

    def test_empty_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_evidence_item(value="")

    def test_empty_source_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_evidence_item(source="")

    def test_round_trip(self) -> None:
        item = make_evidence_item()
        assert EvidenceItem.model_validate(item.model_dump()) == item


# ── CapabilityAssessment ──────────────────────────────────────────────────────


class TestCapabilityAssessment:
    def test_all_confidence_values(self) -> None:
        for conf in ("unknown", "low", "medium", "high"):
            a = CapabilityAssessment(confidence=conf, evidence_refs=[])  # type: ignore[arg-type]
            assert a.confidence == conf

    def test_invalid_confidence_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CapabilityAssessment(confidence="very_high", evidence_refs=[])  # type: ignore[arg-type]

    def test_unknown_is_not_low_adr1(self) -> None:
        """ADR-1: 'unknown' and 'low' must be distinct values."""
        a_unknown = CapabilityAssessment(confidence="unknown", evidence_refs=[])
        a_low = CapabilityAssessment(confidence="low", evidence_refs=[])
        assert a_unknown.confidence != a_low.confidence


# ── ScoreEntry ────────────────────────────────────────────────────────────────


class TestScoreEntry:
    def test_score_bounds(self) -> None:
        s = ScoreEntry(score=0.0, confidence="unknown", evidence_refs=[])
        assert s.score == 0.0
        s2 = ScoreEntry(score=100.0, confidence="high", evidence_refs=[])
        assert s2.score == 100.0

    def test_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScoreEntry(score=-1.0, confidence="low", evidence_refs=[])

    def test_score_above_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ScoreEntry(score=100.1, confidence="high", evidence_refs=[])


# ── CandidateState ────────────────────────────────────────────────────────────


class TestCandidateState:
    def test_default_state(self) -> None:
        cs = CandidateState()
        assert cs.interview_rounds == 0
        assert cs.recommendation is None
        assert cs.finalized is False

    def test_interview_rounds_literal_adr3(self) -> None:
        """ADR-3: interview_rounds is Literal[0, 1] — only 0 or 1 are valid."""
        cs0 = CandidateState(interview_rounds=0)
        cs1 = CandidateState(interview_rounds=1)
        assert cs0.interview_rounds == 0
        assert cs1.interview_rounds == 1

    def test_interview_rounds_2_rejected(self) -> None:
        """ADR-3: interview_rounds cannot be 2."""
        with pytest.raises(ValidationError):
            CandidateState(interview_rounds=2)  # type: ignore[arg-type]

    def test_all_recommendation_values(self) -> None:
        for rec in ("reject", "hold", "interview", "strong_interview"):
            cs = CandidateState(recommendation=rec)  # type: ignore[arg-type]
            assert cs.recommendation == rec

    def test_invalid_recommendation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CandidateState(recommendation="maybe")  # type: ignore[arg-type]

    def test_round_trip(self) -> None:
        cs = CandidateState(
            resume_text="John Doe, Software Engineer",
            candidate_id="c001",
            interview_rounds=1,
            recommendation="hold",
        )
        restored = CandidateState.model_validate(json.loads(cs.model_dump_json()))
        assert restored == cs


# ── ShortlistEntry ────────────────────────────────────────────────────────────


class TestShortlistEntry:
    def test_valid_entry(self) -> None:
        entry = make_shortlist_entry()
        assert entry.recommendation == "interview"

    def test_weighted_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            make_shortlist_entry(weighted_score=-1.0)
        with pytest.raises(ValidationError):
            make_shortlist_entry(weighted_score=101.0)

    def test_evidence_refs_required_for_positive_recommendation(self) -> None:
        """§4.1: Every summary claim must cite evidence_refs."""
        with pytest.raises(ValidationError, match="evidence_refs must not be empty"):
            make_shortlist_entry(recommendation="strong_interview", evidence_refs=[])

    def test_empty_evidence_refs_ok_for_reject(self) -> None:
        entry = make_shortlist_entry(recommendation="reject", evidence_refs=[])
        assert entry.recommendation == "reject"

    def test_all_recommendation_values(self) -> None:
        for rec in ("reject", "hold", "interview", "strong_interview"):
            refs = ["Resume:1"] if rec in ("interview", "strong_interview") else []
            e = make_shortlist_entry(recommendation=rec, evidence_refs=refs)
            assert e.recommendation == rec


# ── TrajectoryEntry ───────────────────────────────────────────────────────────


class TestTrajectoryEntry:
    def test_valid_entry(self) -> None:
        entry = make_trajectory_entry()
        assert entry.status == "success"
        assert entry.model_role == "fast_structured"

    def test_all_statuses(self) -> None:
        for status in ("success", "retried", "failed_closed", "escalated"):
            e = make_trajectory_entry(status=status)
            assert e.status == status

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_trajectory_entry(status="unknown_status")

    def test_all_model_roles(self) -> None:
        for role in ("fast_structured", "high_context"):
            e = make_trajectory_entry(model_role=role)
            assert e.model_role == role

    def test_invalid_model_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_trajectory_entry(model_role="slow_reasoning")

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_trajectory_entry(latency_ms=-1)

    def test_tool_used_none_for_fixed_nodes(self) -> None:
        """ADR-8: fixed-node entries have tool_used=None."""
        e = make_trajectory_entry(node="screen_resume", tool_used=None)
        assert e.tool_used is None

    def test_tool_used_set_for_agent_calls(self) -> None:
        e = make_trajectory_entry(node="candidate_agent", tool_used="extract_evidence")
        assert e.tool_used == "extract_evidence"

    def test_round_trip(self) -> None:
        entry = make_trajectory_entry()
        restored = TrajectoryEntry.model_validate(json.loads(entry.model_dump_json()))
        assert restored == entry


# ── RoleProfile ───────────────────────────────────────────────────────────────


class TestRoleProfile:
    def test_valid_role_profile(self) -> None:
        rp = RoleProfile(
            title="Senior Software Engineer",
            summary="Build distributed systems",
            required_skills=[
                RequiredSkill(name="Python", required=True, weight=1.0)
            ],
            jd_hash="a" * 64,
            extracted_at="2024-01-01T00:00:00Z",
        )
        assert rp.title == "Senior Software Engineer"

    def test_empty_title_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RoleProfile(title="", jd_hash="a" * 64, extracted_at="2024-01-01T00:00:00Z")

    def test_round_trip(self) -> None:
        rp = RoleProfile(
            title="SWE",
            jd_hash="x" * 64,
            extracted_at="2024-01-01T00:00:00Z",
        )
        assert RoleProfile.model_validate(rp.model_dump()) == rp


# ── Rubric ────────────────────────────────────────────────────────────────────


class TestRubric:
    def make_rubric(self, criteria_names: list[str] | None = None) -> Rubric:
        if criteria_names is None:
            criteria_names = ["Python", "System Design"]
        return Rubric(
            criteria=[
                RubricCriterion(name=n, weight=1.0) for n in criteria_names
            ],
            jd_hash="a" * 64,
            rubric_config_version="1.0.0",
            generated_at="2024-01-01T00:00:00Z",
        )

    def test_valid_rubric(self) -> None:
        rubric = self.make_rubric()
        assert len(rubric.criteria) == 2

    def test_duplicate_criterion_names_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unique"):
            self.make_rubric(["Python", "Python"])

    def test_empty_criteria_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Rubric(
                criteria=[],
                jd_hash="a" * 64,
                rubric_config_version="1.0.0",
                generated_at="2024-01-01T00:00:00Z",
            )


# ── EvidenceBuckets ───────────────────────────────────────────────────────────


class TestEvidenceBuckets:
    def test_valid_buckets(self) -> None:
        eb = EvidenceBuckets(items=[make_evidence_item()])
        assert len(eb.items) == 1

    def test_empty_buckets_valid(self) -> None:
        eb = EvidenceBuckets(items=[])
        assert eb.items == []


# ── CapabilityHypotheses ──────────────────────────────────────────────────────


class TestCapabilityHypotheses:
    def test_valid_hypotheses(self) -> None:
        ch = CapabilityHypotheses(
            assessments={
                "python": CapabilityAssessment(confidence="high", evidence_refs=["Resume:Skills"]),
                "leadership": CapabilityAssessment(confidence="unknown", evidence_refs=[]),
            }
        )
        assert ch.assessments["python"].confidence == "high"
        assert ch.assessments["leadership"].confidence == "unknown"


# ── VerificationResult ────────────────────────────────────────────────────────


class TestVerificationResult:
    def test_sufficient_verdict(self) -> None:
        vr = VerificationResult(verdict="sufficient", per_gap_reasoning={})
        assert vr.verdict == "sufficient"

    def test_insufficient_verdict(self) -> None:
        vr = VerificationResult(
            verdict="insufficient",
            per_gap_reasoning={"python": "Only 1 year experience, JD requires 5+"},
        )
        assert vr.verdict == "insufficient"

    def test_invalid_verdict_rejected(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(verdict="maybe", per_gap_reasoning={})  # type: ignore[arg-type]


# ── QuestionSet ───────────────────────────────────────────────────────────────


class TestQuestionSet:
    def test_top_questions_returns_max_two(self) -> None:
        qs = QuestionSet(
            questions=[
                InterviewQuestion(
                    question=f"Q{i}",
                    target_criterion="python",
                    rationale="Gap",
                    priority_score=float(i),
                )
                for i in range(5)
            ]
        )
        top = qs.top_questions
        assert len(top) == 2
        assert top[0].priority_score > top[1].priority_score

    def test_empty_question_set(self) -> None:
        qs = QuestionSet(questions=[])
        assert qs.top_questions == []


# ── UpdatedScoreCard ──────────────────────────────────────────────────────────


class TestUpdatedScoreCard:
    def test_valid_updated_scorecard(self) -> None:
        usc = UpdatedScoreCard(
            scorecard={"python": ScoreEntry(score=80.0, confidence="high", evidence_refs=[])},
            capabilities={"python": CapabilityAssessment(confidence="high", evidence_refs=[])},
            changes_summary="Score improved after clarification",
        )
        assert usc.scorecard["python"].score == 80.0


# ── BiasReport ────────────────────────────────────────────────────────────────


class TestBiasReport:
    def test_valid_bias_report(self) -> None:
        br = BiasReport(
            candidate_a="c001",
            candidate_b="c002",
            indicators=[
                BiasIndicator(
                    criterion="experience",
                    description="Score delta on name swap",
                    severity="low",
                    counterfactual_delta=2.5,
                )
            ],
            overall_risk="low",
            summary="No significant bias detected",
            probe_timestamp="2024-01-01T00:00:00Z",
        )
        assert br.overall_risk == "low"

    def test_invalid_severity_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BiasIndicator(
                criterion="x",
                description="y",
                severity="critical",  # type: ignore[arg-type]
                counterfactual_delta=None,
            )


# ── RecruiterSummary ──────────────────────────────────────────────────────────


class TestRecruiterSummary:
    def test_evidence_refs_nonempty_required(self) -> None:
        """§4.1: Every summary claim must cite at least one evidence_ref."""
        with pytest.raises(ValidationError, match="evidence_refs must not be empty"):
            RecruiterSummary(
                shortlist=[make_shortlist_entry()],
                bias_reports=[],
                overall_recommendation="Interview top candidates",
                evidence_refs=[],  # must raise
                generated_at="2024-01-01T00:00:00Z",
                run_id="run_001",
            )

    def test_valid_summary(self) -> None:
        rs = RecruiterSummary(
            shortlist=[make_shortlist_entry()],
            bias_reports=[],
            overall_recommendation="Interview top candidates",
            evidence_refs=["Resume:Experience:1"],
            generated_at="2024-01-01T00:00:00Z",
            run_id="run_001",
        )
        assert len(rs.evidence_refs) == 1


# ── GraphState ────────────────────────────────────────────────────────────────


class TestGraphState:
    def test_default_state(self) -> None:
        gs = GraphState()
        assert gs.current_idx == 0
        assert gs.candidates == []
        assert gs.shortlist == []
        assert gs.trajectory == []
        assert gs.step_count == 0

    def test_current_idx_out_of_bounds_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GraphState(
                candidates=[CandidateState(candidate_id="c001")],
                current_idx=5,  # out of bounds
            )

    def test_current_idx_0_valid_when_no_candidates(self) -> None:
        gs = GraphState(current_idx=0)
        assert gs.current_idx == 0

    def test_current_idx_valid_when_within_bounds(self) -> None:
        gs = GraphState(
            candidates=[CandidateState(candidate_id=f"c{i}") for i in range(3)],
            current_idx=2,
        )
        assert gs.current_idx == 2

    def test_trajectory_is_list(self) -> None:
        gs = GraphState(trajectory=[make_trajectory_entry()])
        assert len(gs.trajectory) == 1

    def test_round_trip(self) -> None:
        gs = GraphState(
            jd="We are looking for a Senior Engineer",
            run_id="run_test_001",
            step_count=3,
        )
        restored = GraphState.model_validate(json.loads(gs.model_dump_json()))
        assert restored == gs


# ── Scheduling types ──────────────────────────────────────────────────────────


class TestSchedulingTypes:
    def test_availability_slot(self) -> None:
        slot = AvailabilitySlot(
            start_iso="2024-03-01T09:00:00Z",
            end_iso="2024-03-01T10:00:00Z",
            timezone="UTC",
        )
        assert slot.timezone == "UTC"

    def test_pending_confirmation_default_status(self) -> None:
        pc = PendingConfirmation(
            candidate="c001",
            slot=AvailabilitySlot(
                start_iso="2024-03-01T09:00:00Z",
                end_iso="2024-03-01T10:00:00Z",
            ),
            confirmation_id="conf_001",
            proposed_at="2024-01-01T00:00:00Z",
        )
        assert pc.status == "pending"

    def test_pending_confirmation_statuses(self) -> None:
        for status in ("pending", "confirmed", "declined"):
            pc = PendingConfirmation(
                candidate="c001",
                slot=AvailabilitySlot(
                    start_iso="2024-03-01T09:00:00Z",
                    end_iso="2024-03-01T10:00:00Z",
                ),
                confirmation_id="conf_001",
                proposed_at="2024-01-01T00:00:00Z",
                status=status,  # type: ignore[arg-type]
            )
            assert pc.status == status


# ── Cross-schema integration ──────────────────────────────────────────────────


class TestCrossSchemaIntegration:
    def test_full_candidate_state_with_evidence(self) -> None:
        """Verify a fully populated CandidateState with all sub-schemas."""
        cs = CandidateState(
            candidate_id="c001",
            resume_text="Jane Smith, ML Engineer",
            sanitized_resume="[REDACTED], ML Engineer",
            evidence_buckets=[make_evidence_item()],
            capabilities={
                "python": CapabilityAssessment(confidence="high", evidence_refs=["Resume:Skills"]),
                "ml_frameworks": CapabilityAssessment(
                    confidence="medium", evidence_refs=["Resume:Projects"]
                ),
            },
            scorecard={
                "python": ScoreEntry(score=90.0, confidence="high", evidence_refs=["Resume:Skills"]),
            },
            verification_verdict="sufficient",
            interview_rounds=0,
            recommendation="strong_interview",
            rationale="Strong ML background with direct evidence",
            finalized=True,
        )
        assert cs.finalized is True
        assert cs.capabilities["python"].confidence == "high"
        # 'unknown' is not 'low' — ADR-1
        assert cs.capabilities["python"].confidence != "unknown"

    def test_graph_state_with_candidates_and_shortlist(self) -> None:
        """Verify GraphState correctly holds candidates and shortlist."""
        gs = GraphState(
            jd="Build AI systems",
            run_id="run_001",
            candidates=[
                CandidateState(candidate_id="c001", finalized=True),
                CandidateState(candidate_id="c002", finalized=False),
            ],
            current_idx=1,
            shortlist=[make_shortlist_entry(candidate="c001")],
            step_count=5,
        )
        assert len(gs.candidates) == 2
        assert gs.candidates[0].finalized is True
        assert len(gs.shortlist) == 1
