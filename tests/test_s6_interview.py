"""
S6 tests: generate_interview_questions, ask_candidate (interrupt scaffold),
          reevaluate_candidate, finalize_candidate.

Key test cases:
- generate_interview_questions ranks by weight × (1 - confidence_weight)
- ask_candidate result is_valid() for non-empty answers
- ask_candidate interrupt payload has required fields
- reevaluate_candidate merges new scores without lowering untouched criteria
- finalize_candidate computes weighted_score correctly
- finalize_candidate caps score when any criterion is 'unknown'
- finalize_candidate raises on empty rationale
- finalize_candidate positive recommendations require evidence_refs
- compute_weighted_score: all unknown → capped score
- compute_weighted_score: empty rubric → 0.0
- ADR-3: ask_candidate single-use enforcement is tested in S7 (harness)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from scoutai.capabilities.interview import (
    AskCandidateResult,
    _build_question_set,
    _build_updated_scorecard,
    _parse_questions_response,
    _parse_reevaluation_response,
    compute_weighted_score,
    create_ask_candidate_interrupt,
    finalize_candidate,
    generate_interview_questions,
    reevaluate_candidate,
)
from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.config import load_config
from scoutai.schemas import (
    CapabilityAssessment,
    CapabilityHypotheses,
    CandidateState,
    InterviewQuestion,
    Rubric,
    RubricCriterion,
    ScoreEntry,
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
def sample_capabilities():
    return CapabilityHypotheses(
        assessments={
            "Python": CapabilityAssessment(confidence="high", evidence_refs=["Resume:Skills"]),
            "Distributed Systems": CapabilityAssessment(confidence="low", evidence_refs=["Resume:Projects"]),
            "Leadership": CapabilityAssessment(confidence="unknown", evidence_refs=[]),
        }
    )


@pytest.fixture
def sample_candidate(sample_rubric):
    return CandidateState(
        candidate_id="c001",
        sanitized_resume="Experienced Python engineer",
        scorecard={
            "Python": ScoreEntry(score=85.0, confidence="high", evidence_refs=["Resume:Skills"]),
            "Distributed Systems": ScoreEntry(score=40.0, confidence="low", evidence_refs=["Resume:Projects"]),
            "Leadership": ScoreEntry(score=0.0, confidence="unknown", evidence_refs=[]),
        },
        capabilities={
            "Python": CapabilityAssessment(confidence="high", evidence_refs=["Resume:Skills"]),
            "Distributed Systems": CapabilityAssessment(confidence="low", evidence_refs=["Resume:Projects"]),
            "Leadership": CapabilityAssessment(confidence="unknown", evidence_refs=[]),
        },
        finalized=False,
    )


def make_mock_model(response_data: dict) -> MagicMock:
    mock = MagicMock()
    resp = MagicMock()
    resp.content = json.dumps(response_data)
    mock.invoke.return_value = resp
    return mock


# ── _parse_questions_response ─────────────────────────────────────────────────


class TestParseQuestionsResponse:
    def test_valid(self):
        data = {"questions": [{"question": "Tell me about X", "target_criterion": "Python", "rationale": "Gap", "priority_score": 4.0}]}
        result = _parse_questions_response(json.dumps(data))
        assert "questions" in result

    def test_missing_questions_raises(self):
        with pytest.raises(ValueError, match="missing 'questions'"):
            _parse_questions_response(json.dumps({"other": "field"}))

    def test_questions_not_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            _parse_questions_response(json.dumps({"questions": "not_a_list"}))


# ── _build_question_set ───────────────────────────────────────────────────────


class TestBuildQuestionSet:
    def test_sorts_by_priority_score(self, sample_capabilities, sample_rubric):
        data = {
            "questions": [
                {"question": "Q_low_priority", "target_criterion": "Python", "rationale": "r", "priority_score": 1.0},
                {"question": "Q_high_priority", "target_criterion": "Leadership", "rationale": "r", "priority_score": 9.0},
                {"question": "Q_mid_priority", "target_criterion": "Distributed Systems", "rationale": "r", "priority_score": 3.0},
            ]
        }
        qs = _build_question_set(data, sample_capabilities, sample_rubric)
        # top_questions returns sorted by priority descending
        top = qs.top_questions
        assert len(top) == 2
        assert top[0].question == "Q_high_priority"
        assert top[1].question == "Q_mid_priority"

    def test_auto_computes_priority_when_absent(self, sample_capabilities, sample_rubric):
        """If priority_score not in LLM response, compute from weight × (1 - conf_weight)."""
        data = {
            "questions": [
                # Leadership: weight=2.0, confidence=unknown (0.0) → priority = 2.0 × (1 - 0) = 2.0
                {"question": "Leadership Q", "target_criterion": "Leadership", "rationale": "gap"},
                # Python: weight=5.0, confidence=high (1.0) → priority = 5.0 × (1 - 1.0) = 0.0
                {"question": "Python Q", "target_criterion": "Python", "rationale": "low gap"},
            ]
        }
        qs = _build_question_set(data, sample_capabilities, sample_rubric)
        leadership_q = next(q for q in qs.questions if q.target_criterion == "Leadership")
        python_q = next(q for q in qs.questions if q.target_criterion == "Python")
        assert leadership_q.priority_score > python_q.priority_score

    def test_malformed_question_skipped(self, sample_capabilities, sample_rubric):
        data = {
            "questions": [
                "not_a_dict",
                {"question": "Good Q", "target_criterion": "Python", "rationale": "r", "priority_score": 1.0},
            ]
        }
        qs = _build_question_set(data, sample_capabilities, sample_rubric)
        assert len(qs.questions) == 1

    def test_empty_question_text_skipped(self, sample_capabilities, sample_rubric):
        data = {
            "questions": [
                {"question": "", "target_criterion": "Python", "rationale": "r", "priority_score": 1.0},
            ]
        }
        qs = _build_question_set(data, sample_capabilities, sample_rubric)
        assert len(qs.questions) == 0


# ── AskCandidateResult ────────────────────────────────────────────────────────


class TestAskCandidateResult:
    def test_valid_answer(self):
        result = AskCandidateResult(question="Tell me about Python", answer="I have 5 years exp")
        assert result.is_valid()

    def test_empty_answer_invalid(self):
        result = AskCandidateResult(question="Q", answer="")
        assert not result.is_valid()

    def test_whitespace_answer_invalid(self):
        result = AskCandidateResult(question="Q", answer="   \n  ")
        assert not result.is_valid()

    def test_to_qa_log_entry(self):
        result = AskCandidateResult(question="Q?", answer="A.", question_index=0)
        entry = result.to_qa_log_entry()
        assert entry["question"] == "Q?"
        assert entry["answer"] == "A."
        assert entry["question_index"] == 0


# ── create_ask_candidate_interrupt ────────────────────────────────────────────


class TestCreateAskCandidateInterrupt:
    def test_interrupt_payload_has_required_fields(self):
        question = InterviewQuestion(
            question="Describe your distributed systems experience",
            target_criterion="Distributed Systems",
            rationale="Only low confidence evidence found",
            priority_score=3.0,
        )
        payload = create_ask_candidate_interrupt(question, candidate_id="c001")
        assert payload["type"] == "ask_candidate"
        assert payload["candidate_id"] == "c001"
        assert payload["question"] == "Describe your distributed systems experience"
        assert payload["target_criterion"] == "Distributed Systems"
        assert "rationale" in payload
        assert "priority_score" in payload


# ── _parse_reevaluation_response ──────────────────────────────────────────────


class TestParseReevaluationResponse:
    def test_valid(self):
        data = {
            "scorecard": {"Python": {"score": 90.0, "confidence": "high", "evidence_refs": []}},
            "capabilities": {"Python": {"confidence": "high", "evidence_refs": []}},
            "changes_summary": "Python score improved",
        }
        result = _parse_reevaluation_response(json.dumps(data))
        assert "scorecard" in result

    def test_missing_scorecard_raises(self):
        data = {"capabilities": {}, "changes_summary": ""}
        with pytest.raises(ValueError, match="missing 'scorecard'"):
            _parse_reevaluation_response(json.dumps(data))

    def test_missing_capabilities_raises(self):
        data = {"scorecard": {}, "changes_summary": ""}
        with pytest.raises(ValueError, match="missing 'capabilities'"):
            _parse_reevaluation_response(json.dumps(data))


# ── _build_updated_scorecard ──────────────────────────────────────────────────


class TestBuildUpdatedScorecard:
    def test_merges_updates_with_existing(self, sample_candidate):
        data = {
            "scorecard": {
                "Leadership": {"score": 70.0, "confidence": "medium", "evidence_refs": ["Interview:Q1"]},
            },
            "capabilities": {
                "Leadership": {"confidence": "medium", "evidence_refs": ["Interview:Q1"]},
            },
            "changes_summary": "Leadership score improved after interview",
        }
        updated = _build_updated_scorecard(
            data,
            sample_candidate.scorecard,
            sample_candidate.capabilities,
        )
        # Leadership was updated
        assert updated.scorecard["Leadership"].score == 70.0
        assert updated.scorecard["Leadership"].confidence == "medium"
        # Python was NOT touched — should retain original score
        assert updated.scorecard["Python"].score == 85.0
        assert updated.scorecard["Python"].confidence == "high"

    def test_does_not_lower_untouched_criteria(self, sample_candidate):
        """Interview answer must not lower scores for criteria it doesn't address."""
        data = {
            "scorecard": {
                "Leadership": {"score": 60.0, "confidence": "medium", "evidence_refs": []},
                # Python NOT in response — should stay at 85.0
            },
            "capabilities": {
                "Leadership": {"confidence": "medium", "evidence_refs": []},
            },
            "changes_summary": "Only leadership updated",
        }
        updated = _build_updated_scorecard(
            data,
            sample_candidate.scorecard,
            sample_candidate.capabilities,
        )
        assert updated.scorecard["Python"].score == 85.0, "Untouched criteria must not be lowered"

    def test_score_clamped_to_0_100(self, sample_candidate):
        data = {
            "scorecard": {
                "Python": {"score": 150.0, "confidence": "high", "evidence_refs": []},  # over 100
            },
            "capabilities": {},
            "changes_summary": "test",
        }
        updated = _build_updated_scorecard(data, sample_candidate.scorecard, sample_candidate.capabilities)
        assert updated.scorecard["Python"].score <= 100.0


# ── compute_weighted_score ────────────────────────────────────────────────────


class TestComputeWeightedScore:
    def test_all_high_confidence_no_cap(self, config, sample_rubric):
        scorecard = {
            "Python": ScoreEntry(score=100.0, confidence="high", evidence_refs=[]),
            "Distributed Systems": ScoreEntry(score=100.0, confidence="high", evidence_refs=[]),
            "Leadership": ScoreEntry(score=100.0, confidence="high", evidence_refs=[]),
        }
        score = compute_weighted_score(scorecard, sample_rubric, config)
        assert score == 100.0

    def test_unknown_confidence_caps_score(self, config, sample_rubric):
        """§12: any criterion at 'unknown' caps weighted_score at unknown_confidence_score_cap."""
        scorecard = {
            "Python": ScoreEntry(score=100.0, confidence="high", evidence_refs=[]),
            "Distributed Systems": ScoreEntry(score=100.0, confidence="high", evidence_refs=[]),
            "Leadership": ScoreEntry(score=0.0, confidence="unknown", evidence_refs=[]),  # unknown!
        }
        score = compute_weighted_score(scorecard, sample_rubric, config)
        cap = config.rubric.unknown_confidence_score_cap
        assert score <= cap, f"Score {score} should be capped at {cap} due to 'unknown' criterion"

    def test_empty_rubric_returns_zero(self, config):
        empty_rubric = Rubric(
            criteria=[RubricCriterion(name="x", weight=1.0)],
            jd_hash="a" * 64,
            rubric_config_version="1.0.0",
            generated_at="2024-01-01T00:00:00Z",
        )
        # Simulate empty by using zero-weight rubric
        import unittest.mock
        zero_rubric = unittest.mock.MagicMock(spec=Rubric)
        zero_rubric.criteria = []
        score = compute_weighted_score({}, zero_rubric, config)
        assert score == 0.0

    def test_missing_criterion_in_scorecard_treated_as_unknown(self, config, sample_rubric):
        """If a rubric criterion has no scorecard entry, it counts as unknown → caps score."""
        scorecard = {
            "Python": ScoreEntry(score=100.0, confidence="high", evidence_refs=[]),
            # Distributed Systems and Leadership missing → unknown
        }
        score = compute_weighted_score(scorecard, sample_rubric, config)
        cap = config.rubric.unknown_confidence_score_cap
        assert score <= cap

    def test_weighted_correctly(self, config, sample_rubric):
        """Score is a weighted average, not a simple average."""
        # Python=100 (weight=5), DS=0 (weight=4), Leadership=0 (weight=2)
        # Total weight = 11
        # Weighted sum = (5/11)*100 + (4/11)*0 + (2/11)*0 = 45.45...
        scorecard = {
            "Python": ScoreEntry(score=100.0, confidence="high", evidence_refs=[]),
            "Distributed Systems": ScoreEntry(score=0.0, confidence="high", evidence_refs=[]),
            "Leadership": ScoreEntry(score=0.0, confidence="high", evidence_refs=[]),
        }
        score = compute_weighted_score(scorecard, sample_rubric, config)
        expected = round((5.0 / 11.0) * 100.0, 2)
        assert abs(score - expected) < 0.1


# ── finalize_candidate ────────────────────────────────────────────────────────


class TestFinalizeCandidate:
    def test_happy_path_interview(self, config, sample_candidate, sample_rubric):
        entry = finalize_candidate(
            sample_candidate,
            recommendation="interview",
            rationale="Strong Python, some distributed systems gaps",
            rubric=sample_rubric,
            config=config,
        )
        assert entry.candidate == "c001"
        assert entry.recommendation == "interview"
        assert 0.0 <= entry.weighted_score <= 100.0
        assert len(entry.evidence_refs) > 0  # interview recommendations require refs

    def test_happy_path_reject(self, config, sample_rubric):
        candidate = CandidateState(
            candidate_id="c002",
            scorecard={
                "Python": ScoreEntry(score=10.0, confidence="low", evidence_refs=[]),
                "Distributed Systems": ScoreEntry(score=0.0, confidence="unknown", evidence_refs=[]),
                "Leadership": ScoreEntry(score=0.0, confidence="unknown", evidence_refs=[]),
            },
            capabilities={
                "Python": CapabilityAssessment(confidence="low", evidence_refs=[]),
                "Distributed Systems": CapabilityAssessment(confidence="unknown", evidence_refs=[]),
                "Leadership": CapabilityAssessment(confidence="unknown", evidence_refs=[]),
            },
        )
        entry = finalize_candidate(
            candidate,
            recommendation="reject",
            rationale="Insufficient evidence across all criteria",
            rubric=sample_rubric,
            config=config,
        )
        assert entry.recommendation == "reject"

    def test_empty_rationale_raises(self, config, sample_candidate, sample_rubric):
        with pytest.raises(ValueError, match="rationale must not be empty"):
            finalize_candidate(
                sample_candidate,
                recommendation="hold",
                rationale="",
                rubric=sample_rubric,
                config=config,
            )

    def test_whitespace_rationale_raises(self, config, sample_candidate, sample_rubric):
        with pytest.raises(ValueError, match="rationale must not be empty"):
            finalize_candidate(
                sample_candidate,
                recommendation="hold",
                rationale="   ",
                rubric=sample_rubric,
                config=config,
            )

    def test_strengths_populated_from_high_confidence(self, config, sample_candidate, sample_rubric):
        entry = finalize_candidate(
            sample_candidate,
            recommendation="interview",
            rationale="Good candidate",
            rubric=sample_rubric,
            config=config,
        )
        assert "Python" in entry.strengths  # Python is the only 'high' confidence criterion

    def test_remaining_uncertainties_populated(self, config, sample_candidate, sample_rubric):
        entry = finalize_candidate(
            sample_candidate,
            recommendation="interview",
            rationale="Interview to clarify gaps",
            rubric=sample_rubric,
            config=config,
        )
        # Leadership is 'unknown', Distributed Systems is 'low' → should be in uncertainties
        assert "Leadership" in entry.remaining_uncertainties

    def test_custom_remaining_uncertainties(self, config, sample_candidate, sample_rubric):
        entry = finalize_candidate(
            sample_candidate,
            recommendation="hold",
            rationale="Hold pending more info",
            rubric=sample_rubric,
            config=config,
            remaining_uncertainties=["Custom gap A", "Custom gap B"],
        )
        assert entry.remaining_uncertainties == ["Custom gap A", "Custom gap B"]

    def test_unknown_confidence_caps_weighted_score(self, config, sample_rubric):
        """§12: unknown confidence on any criterion caps the final weighted_score."""
        candidate = CandidateState(
            candidate_id="c003",
            scorecard={
                "Python": ScoreEntry(score=100.0, confidence="high", evidence_refs=["Resume:Skills"]),
                "Distributed Systems": ScoreEntry(score=100.0, confidence="high", evidence_refs=["Resume:Projects"]),
                "Leadership": ScoreEntry(score=0.0, confidence="unknown", evidence_refs=[]),
            },
            capabilities={
                "Python": CapabilityAssessment(confidence="high", evidence_refs=["Resume:Skills"]),
                "Distributed Systems": CapabilityAssessment(confidence="high", evidence_refs=["Resume:Projects"]),
                "Leadership": CapabilityAssessment(confidence="unknown", evidence_refs=[]),
            },
        )
        entry = finalize_candidate(
            candidate,
            recommendation="interview",
            rationale="Strong candidate but leadership unknown",
            rubric=sample_rubric,
            config=config,
        )
        cap = config.rubric.unknown_confidence_score_cap
        assert entry.weighted_score <= cap, (
            f"weighted_score={entry.weighted_score} should be capped at {cap} due to 'unknown' criterion"
        )

    def test_evidence_refs_deduplicated(self, config, sample_candidate, sample_rubric):
        """Evidence refs collected from scorecard + capabilities should not have duplicates."""
        # Add same ref to both scorecard and capabilities
        from copy import deepcopy
        c = deepcopy(sample_candidate)
        c.scorecard["Python"] = ScoreEntry(score=85.0, confidence="high", evidence_refs=["Resume:Skills", "Resume:Skills"])
        entry = finalize_candidate(c, "interview", "Good", sample_rubric, config)
        assert len(entry.evidence_refs) == len(set(entry.evidence_refs))


# ── generate_interview_questions integration (mocked LLM) ────────────────────


class TestGenerateInterviewQuestions:
    SAMPLE_RESPONSE = {
        "questions": [
            {
                "question": "Describe a distributed system you designed end-to-end",
                "target_criterion": "Distributed Systems",
                "rationale": "Only low-confidence evidence found",
                "priority_score": 3.0,
            },
            {
                "question": "Have you led engineering teams?",
                "target_criterion": "Leadership",
                "rationale": "No leadership evidence found",
                "priority_score": 2.0,
            },
        ]
    }

    def test_happy_path(self, config, mock_router, sample_capabilities, sample_rubric):
        mock_model = make_mock_model(self.SAMPLE_RESPONSE)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            qs = generate_interview_questions(sample_capabilities, sample_rubric, config, mock_router)
        assert len(qs.questions) == 2
        top = qs.top_questions
        assert top[0].priority_score >= top[1].priority_score

    def test_fail_closed(self, config, mock_router, sample_capabilities, sample_rubric):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = TimeoutError("timeout")
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(ProviderError):
                generate_interview_questions(sample_capabilities, sample_rubric, config, mock_router)


# ── reevaluate_candidate integration (mocked LLM) ────────────────────────────


class TestReevaluateCandidate:
    def test_updates_leadership_score(self, config, mock_router, sample_candidate, sample_rubric):
        response = {
            "scorecard": {
                "Leadership": {"score": 75.0, "confidence": "medium", "evidence_refs": ["Interview:Q1"]},
            },
            "capabilities": {
                "Leadership": {"confidence": "medium", "evidence_refs": ["Interview:Q1"]},
            },
            "changes_summary": "Leadership demonstrated via interview answer",
        }
        mock_model = make_mock_model(response)
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            updated = reevaluate_candidate(
                sample_candidate,
                question="Have you led teams?",
                answer="Yes, I led a team of 5 at my last job",
                rubric=sample_rubric,
                config=config,
                router=mock_router,
            )
        assert updated.scorecard["Leadership"].score == 75.0
        assert updated.scorecard["Python"].score == 85.0  # unchanged

    def test_fail_closed(self, config, mock_router, sample_candidate, sample_rubric):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = TimeoutError("timeout")
        with patch.object(mock_router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(ProviderError):
                reevaluate_candidate(
                    sample_candidate,
                    question="Q?",
                    answer="A.",
                    rubric=sample_rubric,
                    config=config,
                    router=mock_router,
                )
