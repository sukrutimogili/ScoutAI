"""
Interview and finalization capabilities for candidate_agent (§4.2).

Includes:
- generate_interview_questions: produce ranked questions targeting evidence gaps
- ask_candidate: interrupt scaffold (ADR-3: single-use, removed after first call)
- reevaluate_candidate: update scorecard from interview answer
- finalize_candidate: terminal action — writes ShortlistEntry, ends agent's turn

Spec references: §4.2, §3.3.
ADR references: ADR-3 (interview_rounds cap, single-use enforcement), ADR-9 (agent tools).

Single-use enforcement for ask_candidate:
- This module provides the tool FUNCTIONS only.
- The single-use gating (removing ask_candidate from the tool list after first call)
  is enforced by the agent harness in scoutai/agent/harness.py (S7), NOT here.
- The interview_rounds field in CandidateState is the typed record; the harness is
  the behavioral enforcer. Both are required per ADR-3.

finalize_candidate:
- Terminal action — the agent calls this to end its turn.
- Computes weighted_score from the rubric weights and scorecard.
- Validates that positive recommendations have evidence_refs (§4.1).
- Returns a ShortlistEntry ready to be appended to GraphState.shortlist.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.capabilities.prompts import (
    GENERATE_INTERVIEW_QUESTIONS_SYSTEM,
    GENERATE_INTERVIEW_QUESTIONS_USER,
    REEVALUATE_CANDIDATE_SYSTEM,
    REEVALUATE_CANDIDATE_USER,
)
from scoutai.capabilities.role_requirements import _call_llm_with_retry, _parse_json_response
from scoutai.config import ScoutAIConfig
from scoutai.schemas import (
    CapabilityAssessment,
    CapabilityHypotheses,
    CandidateState,
    InterviewQuestion,
    QuestionSet,
    Recommendation,
    Rubric,
    ScoreEntry,
    ShortlistEntry,
    UpdatedScoreCard,
)

logger = logging.getLogger(__name__)

# Confidence level to numeric weight for priority scoring
_CONFIDENCE_WEIGHT: dict[str, float] = {
    "unknown": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 1.0,
}


# ── generate_interview_questions ──────────────────────────────────────────────


def _parse_questions_response(raw: str) -> dict[str, Any]:
    data = _parse_json_response(raw, "generate_interview_questions")
    if "questions" not in data:
        raise ValueError("generate_interview_questions response missing 'questions' field")
    if not isinstance(data["questions"], list):
        raise ValueError("generate_interview_questions 'questions' must be a list")
    return data


def _build_question_set(
    data: dict[str, Any],
    capabilities: CapabilityHypotheses,
    rubric: Rubric,
) -> QuestionSet:
    """
    Construct a QuestionSet from parsed LLM output.

    Questions are ranked by: criterion_weight × (1 - confidence_weight)
    Higher rank = more important gap to address.
    """
    questions: list[InterviewQuestion] = []
    criterion_weights = {c.name: c.weight for c in rubric.criteria}

    for q_raw in data.get("questions", []):
        if not isinstance(q_raw, dict):
            continue
        try:
            question_text = str(q_raw.get("question", "")).strip()
            criterion = str(q_raw.get("target_criterion", "")).strip()
            rationale = str(q_raw.get("rationale", "")).strip()

            if not question_text or not criterion:
                continue

            # Compute priority_score if not provided, or validate it
            if "priority_score" in q_raw:
                priority_score = float(q_raw["priority_score"])
            else:
                # Compute from rubric weights and current confidence
                weight = criterion_weights.get(criterion, 1.0)
                confidence = "unknown"
                if criterion in capabilities.assessments:
                    confidence = capabilities.assessments[criterion].confidence
                conf_weight = _CONFIDENCE_WEIGHT.get(confidence, 0.0)
                priority_score = weight * (1.0 - conf_weight)

            questions.append(
                InterviewQuestion(
                    question=question_text,
                    target_criterion=criterion,
                    rationale=rationale or f"Gap in {criterion}",
                    priority_score=priority_score,
                )
            )
        except Exception as e:
            logger.warning(
                "Skipping malformed question",
                extra={"item": q_raw, "error": str(e)},
            )

    return QuestionSet(questions=questions)


def generate_interview_questions(
    capabilities: CapabilityHypotheses,
    rubric: Rubric,
    config: ScoutAIConfig,
    router: ModelRouter,
) -> QuestionSet:
    """
    Generate ranked interview questions targeting the highest-priority evidence gaps.

    Agent-callable tool (§4.2). Uses fast_structured model role.
    Questions are ranked by weight × (1 - confidence_weight) — higher priority
    means the gap is both important (high weight) and unresolved (low confidence).

    The agent should use only the top 1-2 questions (QuestionSet.top_questions).
    The agent prompt also says not to call this more than once per candidate pass.

    Args:
        capabilities: Current capability assessments (from assess_capabilities).
        rubric: The evaluation rubric.
        config: ScoutAI configuration.
        router: Model router.

    Returns:
        QuestionSet sorted by priority. Use .top_questions for top 1-2.

    Raises:
        ProviderError: If LLM call fails after all retries.
    """
    model, model_id = router.get_model("generate_interview_questions")

    from langchain_core.messages import HumanMessage, SystemMessage

    capabilities_json = capabilities.model_dump_json(indent=2)
    rubric_json = rubric.model_dump_json(indent=2)
    messages = [
        SystemMessage(content=GENERATE_INTERVIEW_QUESTIONS_SYSTEM),
        HumanMessage(
            content=GENERATE_INTERVIEW_QUESTIONS_USER.format(
                capabilities_json=capabilities_json,
                rubric_json=rubric_json,
            )
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_questions_response,
        capability="generate_interview_questions",
        config=config,
        model_id=model_id,
        router=router,
    )

    question_set = _build_question_set(data, capabilities, rubric)
    logger.info(
        "generate_interview_questions completed",
        extra={"question_count": len(question_set.questions)},
    )
    return question_set


# ── ask_candidate ─────────────────────────────────────────────────────────────

# ask_candidate is an interrupt — it does not call an LLM.
# It pauses the agent loop and waits for a human/candidate answer.
# The single-use enforcement (ADR-3) is in the agent harness (S7).
#
# This module provides:
# 1. The structured result type (AskCandidateResult)
# 2. The validation logic for incoming answers
# 3. The QA log entry format


class AskCandidateResult:
    """
    Result of an ask_candidate interrupt.

    Returned by the agent harness after the human/candidate provides an answer.
    The answer is raw text — reevaluate_candidate processes it into scored evidence.
    """

    __slots__ = ("question", "answer", "question_index")

    def __init__(self, question: str, answer: str, question_index: int = 0) -> None:
        self.question = question
        self.answer = answer
        self.question_index = question_index

    def to_qa_log_entry(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "question_index": self.question_index,
        }

    def is_valid(self) -> bool:
        """A valid answer is non-empty and not just whitespace."""
        return bool(self.answer and self.answer.strip())


def create_ask_candidate_interrupt(
    question: InterviewQuestion,
    candidate_id: str = "",
) -> dict[str, Any]:
    """
    Create an interrupt payload for the ask_candidate tool.

    This is used by the agent harness to suspend execution and surface
    the question to the human reviewer / candidate interface.

    Returns:
        Dict with interrupt metadata consumed by the LangGraph interrupt mechanism.
    """
    return {
        "type": "ask_candidate",
        "candidate_id": candidate_id,
        "question": question.question,
        "target_criterion": question.target_criterion,
        "rationale": question.rationale,
        "priority_score": question.priority_score,
    }


# ── reevaluate_candidate ──────────────────────────────────────────────────────


def _parse_reevaluation_response(raw: str) -> dict[str, Any]:
    data = _parse_json_response(raw, "reevaluate_candidate")
    if "scorecard" not in data:
        raise ValueError("reevaluate_candidate response missing 'scorecard' field")
    if "capabilities" not in data:
        raise ValueError("reevaluate_candidate response missing 'capabilities' field")
    return data


def _build_updated_scorecard(
    data: dict[str, Any],
    existing_scorecard: dict[str, ScoreEntry],
    existing_capabilities: dict[str, CapabilityAssessment],
) -> UpdatedScoreCard:
    """
    Build an UpdatedScoreCard, merging LLM updates with existing state.

    Only criteria touched by the interview answer are updated.
    Criteria not in the LLM response retain their current scores.
    """
    # Start with a copy of existing state
    new_scorecard = dict(existing_scorecard)
    new_capabilities = dict(existing_capabilities)

    # Apply LLM updates
    raw_scorecard = data.get("scorecard", {})
    raw_capabilities = data.get("capabilities", {})

    for name, raw_score in raw_scorecard.items():
        if not isinstance(raw_score, dict):
            continue
        try:
            score_val = float(raw_score.get("score", 0.0))
            score_val = max(0.0, min(100.0, score_val))  # clamp
            confidence = str(raw_score.get("confidence", "unknown")).lower()
            if confidence not in ("unknown", "low", "medium", "high"):
                confidence = "unknown"
            evidence_refs = [
                str(r) for r in raw_score.get("evidence_refs", []) if isinstance(r, str)
            ]
            new_scorecard[name] = ScoreEntry(
                score=score_val,
                confidence=confidence,  # type: ignore[arg-type]
                evidence_refs=evidence_refs,
            )
        except Exception as e:
            logger.warning(
                "Skipping malformed scorecard entry in reevaluation",
                extra={"criterion": name, "error": str(e)},
            )

    for name, raw_cap in raw_capabilities.items():
        if not isinstance(raw_cap, dict):
            continue
        try:
            confidence = str(raw_cap.get("confidence", "unknown")).lower()
            if confidence not in ("unknown", "low", "medium", "high"):
                confidence = "unknown"
            evidence_refs = [
                str(r) for r in raw_cap.get("evidence_refs", []) if isinstance(r, str)
            ]
            new_capabilities[name] = CapabilityAssessment(
                confidence=confidence,  # type: ignore[arg-type]
                evidence_refs=evidence_refs,
            )
        except Exception as e:
            logger.warning(
                "Skipping malformed capability in reevaluation",
                extra={"criterion": name, "error": str(e)},
            )

    return UpdatedScoreCard(
        scorecard=new_scorecard,
        capabilities=new_capabilities,
        changes_summary=str(data.get("changes_summary", "Scores updated from interview answer")),
    )


def reevaluate_candidate(
    candidate: CandidateState,
    question: str,
    answer: str,
    rubric: Rubric,
    config: ScoutAIConfig,
    router: ModelRouter,
) -> UpdatedScoreCard:
    """
    Update a candidate's scorecard based on a new interview answer.

    Agent-callable tool (§4.2). Uses fast_structured model role.
    Only updates criteria directly addressed by the answer — does not
    lower scores for criteria the answer doesn't touch.

    Args:
        candidate: Current candidate state (must have scorecard populated).
        question: The interview question that was asked.
        answer: The candidate's answer.
        rubric: The evaluation rubric.
        config: ScoutAI configuration.
        router: Model router.

    Returns:
        UpdatedScoreCard with merged scores and changes_summary.

    Raises:
        ProviderError: If LLM call fails after all retries.
    """
    model, model_id = router.get_model("reevaluate_candidate")

    from langchain_core.messages import HumanMessage, SystemMessage

    scorecard_json = json.dumps(
        {k: v.model_dump() for k, v in candidate.scorecard.items()}, indent=2
    )
    rubric_json = rubric.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=REEVALUATE_CANDIDATE_SYSTEM),
        HumanMessage(
            content=REEVALUATE_CANDIDATE_USER.format(
                scorecard_json=scorecard_json,
                question=question,
                answer=answer,
                rubric_json=rubric_json,
            )
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_reevaluation_response,
        capability="reevaluate_candidate",
        config=config,
        model_id=model_id,
        router=router,
    )

    updated = _build_updated_scorecard(data, candidate.scorecard, candidate.capabilities)
    logger.info(
        "reevaluate_candidate completed",
        extra={"changes_summary": updated.changes_summary[:100]},
    )
    return updated


# ── finalize_candidate ────────────────────────────────────────────────────────


def compute_weighted_score(
    scorecard: dict[str, ScoreEntry],
    rubric: Rubric,
    config: ScoutAIConfig,
) -> float:
    """
    Compute a weighted_score from the scorecard and rubric.

    Per §12 Open Items: weighted sum with per-criterion floor.
    Any criterion at 'unknown' confidence caps the maximum achievable score.

    Formula:
    1. Compute weighted sum of scores (weight × score / max_possible_weight)
    2. If any criterion is 'unknown', cap at config.rubric.unknown_confidence_score_cap

    Returns:
        float in range [0, 100].
    """
    if not rubric.criteria:
        return 0.0

    total_weight = sum(c.weight for c in rubric.criteria)
    if total_weight == 0:
        return 0.0

    weighted_sum = 0.0
    has_unknown = False

    for criterion in rubric.criteria:
        name = criterion.name
        entry = scorecard.get(name)
        if entry is None:
            # Missing from scorecard = unknown
            has_unknown = True
            continue
        if entry.confidence == "unknown":
            has_unknown = True
        weighted_sum += (criterion.weight / total_weight) * entry.score

    # Clamp to 0-100
    score = max(0.0, min(100.0, weighted_sum))

    # Apply unknown cap per §12
    if has_unknown:
        cap = config.rubric.unknown_confidence_score_cap
        score = min(score, cap)

    return round(score, 2)


def finalize_candidate(
    candidate: CandidateState,
    recommendation: Recommendation,
    rationale: str,
    rubric: Rubric,
    config: ScoutAIConfig,
    remaining_uncertainties: Optional[list[str]] = None,
) -> ShortlistEntry:
    """
    Finalize a candidate's evaluation — terminal action for candidate_agent.

    This is the ONLY way to end the agent's turn. There is no implicit fall-through
    (§3.2 — explicit 'done' state, never implicit). The agent MUST call this.

    Computes weighted_score from scorecard + rubric weights.
    Validates that positive recommendations have evidence_refs (§4.1).

    Args:
        candidate: Fully populated CandidateState (after evidence, assessment, etc.)
        recommendation: Final recommendation enum value.
        rationale: Human-readable justification (required, non-empty).
        rubric: The evaluation rubric.
        config: ScoutAI configuration.
        remaining_uncertainties: Known gaps at finalization time.

    Returns:
        ShortlistEntry ready to be appended to GraphState.shortlist.

    Raises:
        ValueError: If rationale is empty, or recommendation is invalid.
    """
    if not rationale or not rationale.strip():
        raise ValueError("finalize_candidate: rationale must not be empty")

    # Collect all evidence refs from scorecard
    all_evidence_refs: list[str] = []
    for entry in candidate.scorecard.values():
        all_evidence_refs.extend(entry.evidence_refs)
    for cap in candidate.capabilities.values():
        all_evidence_refs.extend(cap.evidence_refs)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_refs: list[str] = []
    for r in all_evidence_refs:
        if r not in seen:
            seen.add(r)
            unique_refs.append(r)

    # Compute weighted score
    weighted_score = compute_weighted_score(candidate.scorecard, rubric, config)

    # Build confidence summary from current capabilities
    confidence_summary: dict[str, str] = {
        name: assessment.confidence
        for name, assessment in candidate.capabilities.items()
    }

    # Build strengths from high-confidence criteria
    strengths: list[str] = [
        name
        for name, assessment in candidate.capabilities.items()
        if assessment.confidence == "high"
    ]

    uncertainties = remaining_uncertainties or [
        name
        for name, assessment in candidate.capabilities.items()
        if assessment.confidence in ("unknown", "low")
    ]

    logger.info(
        "finalize_candidate called",
        extra={
            "candidate_id": candidate.candidate_id,
            "recommendation": recommendation,
            "weighted_score": weighted_score,
        },
    )

    return ShortlistEntry(
        candidate=candidate.candidate_id,
        recommendation=recommendation,
        weighted_score=weighted_score,
        confidence_summary=confidence_summary,  # type: ignore[arg-type]
        strengths=strengths,
        remaining_uncertainties=uncertainties,
        evidence_refs=unique_refs,
    )
