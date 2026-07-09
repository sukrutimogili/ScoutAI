"""
S11 tests: human_review interrupt node and routing.

Key test cases:
1.  human_review_node: approve action writes correct decision to state
2.  human_review_node: reject action writes correct decision to state
3.  human_review_node: edit_recommendation updates shortlist recommendation
4.  human_review_node: request-more-evidence resets candidate finalized flag
5.  human_review_node: invalid action defaults to reject
6.  human_review_node: builds review payload with shortlist summary
7.  route_after_human_review: approve with interview → schedule
8.  route_after_human_review: approve with reject → END
9.  route_after_human_review: reject → END
10. route_after_human_review: edit_recommendation → END
11. route_after_human_review: request-more-evidence → select_candidate
12. route_after_human_review: no decision → END (safe fallback)
13. integration: human_review_node + route_after_human_review complete flow

No live API keys required — all tests use mocked LLM calls.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scoutai.config import load_config
from scoutai.graph.nodes import (
    human_review_node,
    route_after_human_review,
)
from scoutai.schemas import (
    BiasReport,
    CandidateState,
    RecruiterSummary,
    ShortlistEntry,
)

CONFIG_PATH = "config.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


def _make_shortlist_entry(
    candidate_id: str,
    recommendation: str = "interview",
    score: float = 72.0,
) -> ShortlistEntry:
    return ShortlistEntry(
        candidate=candidate_id,
        recommendation=recommendation,  # type: ignore[arg-type]
        weighted_score=score,
        evidence_refs=[f"{candidate_id}:Resume:Skills"],
        strengths=["Strong Python skills"],
        remaining_uncertainties=["Leadership experience unclear"],
        confidence_summary={"Python": "high", "Leadership": "medium"},
    )


def _make_candidate(candidate_id: str, *, finalized: bool = True) -> CandidateState:
    return CandidateState(
        candidate_id=candidate_id,
        resume_text=f"Resume for {candidate_id}",
        sanitized_resume=f"Sanitized for {candidate_id}",
        finalized=finalized,
        interview_rounds=1 if finalized else 0,
        recommendation="interview" if finalized else None,
    )


def _make_recruiter_summary(text: str = "Recommend c001 for interview.") -> RecruiterSummary:
    return RecruiterSummary(
        shortlist=[_make_shortlist_entry("c001")],
        bias_reports=[],
        overall_recommendation=text,
        evidence_refs=["c001:Resume:Skills"],
        generated_at=datetime.now(timezone.utc).isoformat(),
        run_id="test-run",
    )


# ── human_review_node ────────────────────────────────────────────────────────


class TestHumanReviewNode:
    def _state(
        self,
        shortlist: list[ShortlistEntry] | None = None,
        candidates: list[CandidateState] | None = None,
        recruiter_summary: RecruiterSummary | None = None,
        bias_reports: list[BiasReport] | None = None,
    ) -> dict[str, Any]:
        return {
            "shortlist": [e.model_dump() for e in (shortlist or [_make_shortlist_entry("c001")])],
            "candidates": [c.model_dump() for c in (candidates or [_make_candidate("c001")])],
            "recruiter_summary": recruiter_summary or _make_recruiter_summary(),
            "bias_reports": bias_reports or [],
            "step_count": 0,
        }

    def test_approve_writes_correct_decision(self):
        """Approve action must write action='approve' to human_review_decision."""
        state = self._state()

        with patch("langgraph.types.interrupt", return_value={"action": "approve"}):
            result = human_review_node(state)

        decision = result.get("human_review_decision", {})
        assert decision["action"] == "approve"
        assert "overridden_at" in decision

    def test_reject_writes_correct_decision(self):
        """Reject action must write action='reject' to human_review_decision."""
        state = self._state()

        with patch("langgraph.types.interrupt", return_value={"action": "reject"}):
            result = human_review_node(state)

        decision = result.get("human_review_decision", {})
        assert decision["action"] == "reject"

    def test_edit_recommendation_updates_shortlist(self):
        """edit_recommendation must update the shortlist entry's recommendation."""
        entry = _make_shortlist_entry("c001", recommendation="interview")
        state = self._state(shortlist=[entry])

        with patch("langgraph.types.interrupt", return_value={
            "action": "edit_recommendation",
            "candidate_id": "c001",
            "new_recommendation": "hold",
        }):
            result = human_review_node(state)

        # Check the shortlist was updated
        updated_shortlist = result.get("shortlist", [])
        assert len(updated_shortlist) == 1
        rec = updated_shortlist[0].recommendation if hasattr(updated_shortlist[0], "recommendation") else updated_shortlist[0].get("recommendation")
        assert rec == "hold"

    def test_edit_recommendation_updates_last_entry_when_no_candidate_id(self):
        """Without candidate_id, edit_recommendation must update the last entry."""
        entries = [
            _make_shortlist_entry("c001", recommendation="interview"),
            _make_shortlist_entry("c002", recommendation="strong_interview"),
        ]
        state = self._state(shortlist=entries)

        with patch("langgraph.types.interrupt", return_value={
            "action": "edit_recommendation",
            "new_recommendation": "hold",
        }):
            result = human_review_node(state)

        updated_shortlist = result.get("shortlist", [])
        assert len(updated_shortlist) == 2
        # Last entry should be updated
        rec = updated_shortlist[1].recommendation if hasattr(updated_shortlist[1], "recommendation") else updated_shortlist[1].get("recommendation")
        assert rec == "hold"
        # First entry should remain unchanged
        rec0 = updated_shortlist[0].recommendation if hasattr(updated_shortlist[0], "recommendation") else updated_shortlist[0].get("recommendation")
        assert rec0 == "interview"

    def test_request_more_evidence_resets_candidate(self):
        """request-more-evidence must reset finalized flag and interview_rounds."""
        candidate = _make_candidate("c001", finalized=True)
        state = self._state(candidates=[candidate])

        with patch("langgraph.types.interrupt", return_value={
            "action": "request-more-evidence",
            "candidate_id": "c001",
        }):
            result = human_review_node(state)

        updated_candidates = result.get("candidates", [])
        assert len(updated_candidates) == 1
        finalized = updated_candidates[0].get("finalized") if isinstance(updated_candidates[0], dict) else getattr(updated_candidates[0], "finalized")
        assert finalized is False, "Candidate must be unfinalized for re-evaluation"

        interview_rounds = updated_candidates[0].get("interview_rounds") if isinstance(updated_candidates[0], dict) else getattr(updated_candidates[0], "interview_rounds")
        assert interview_rounds == 0, "interview_rounds must be reset to 0 (ADR-3)"

    def test_invalid_action_defaults_to_reject(self):
        """An invalid action must default to 'reject'."""
        state = self._state()

        with patch("langgraph.types.interrupt", return_value={"action": "invalid_action"}):
            result = human_review_node(state)

        decision = result.get("human_review_decision", {})
        assert decision["action"] == "reject"

    def test_builds_review_payload_with_shortlist_summary(self):
        """The interrupt payload must contain shortlist summary with evidence refs count."""
        entry = _make_shortlist_entry("c001", recommendation="interview")
        state = self._state(shortlist=[entry])

        with patch("langgraph.types.interrupt", return_value={"action": "approve"}) as mock_interrupt:
            result = human_review_node(state)

        # Verify the interrupt was called with a payload
        mock_interrupt.assert_called_once()
        payload = mock_interrupt.call_args[0][0]
        assert "shortlist" in payload
        assert len(payload["shortlist"]) == 1
        assert payload["shortlist"][0]["candidate"] == "c001"
        assert payload["shortlist"][0]["evidence_refs_count"] == 1

    def test_increments_step_count(self):
        """human_review_node must increment step_count."""
        state = self._state()
        state["step_count"] = 5

        with patch("langgraph.types.interrupt", return_value={"action": "approve"}):
            result = human_review_node(state)

        assert result["step_count"] == 6

    def test_edit_recommendation_updates_correct_candidate(self):
        """edit_recommendation must update only the specified candidate."""
        entries = [
            _make_shortlist_entry("c001", recommendation="interview"),
            _make_shortlist_entry("c002", recommendation="strong_interview"),
        ]
        state = self._state(shortlist=entries)

        with patch("langgraph.types.interrupt", return_value={
            "action": "edit_recommendation",
            "candidate_id": "c001",
            "new_recommendation": "reject",
        }):
            result = human_review_node(state)

        updated_shortlist = result.get("shortlist", [])
        assert len(updated_shortlist) == 2

        rec0 = updated_shortlist[0].recommendation if hasattr(updated_shortlist[0], "recommendation") else updated_shortlist[0].get("recommendation")
        rec1 = updated_shortlist[1].recommendation if hasattr(updated_shortlist[1], "recommendation") else updated_shortlist[1].get("recommendation")
        assert rec0 == "reject", "c001 must be updated to reject"
        assert rec1 == "strong_interview", "c002 must remain unchanged"


# ── route_after_human_review ─────────────────────────────────────────────────


class TestRouteAfterHumanReview:
    def test_approve_with_interview_returns_schedule(self):
        """Approve with an interview recommendation must route to schedule."""
        state = {
            "human_review_decision": {"action": "approve"},
            "shortlist": [_make_shortlist_entry("c001", recommendation="interview")],
        }
        assert route_after_human_review(state) == "schedule"

    def test_approve_with_strong_interview_returns_schedule(self):
        """Approve with strong_interview must route to schedule."""
        state = {
            "human_review_decision": {"action": "approve"},
            "shortlist": [_make_shortlist_entry("c001", recommendation="strong_interview")],
        }
        assert route_after_human_review(state) == "schedule"

    def test_approve_with_reject_returns_end(self):
        """Approve with a reject recommendation must route to END."""
        state = {
            "human_review_decision": {"action": "approve"},
            "shortlist": [_make_shortlist_entry("c001", recommendation="reject")],
        }
        assert route_after_human_review(state) == "END"

    def test_approve_with_hold_returns_end(self):
        """Approve with a hold recommendation must route to END."""
        state = {
            "human_review_decision": {"action": "approve"},
            "shortlist": [_make_shortlist_entry("c001", recommendation="hold")],
        }
        assert route_after_human_review(state) == "END"

    def test_approve_empty_shortlist_returns_end(self):
        """Approve with empty shortlist must route to END."""
        state = {
            "human_review_decision": {"action": "approve"},
            "shortlist": [],
        }
        assert route_after_human_review(state) == "END"

    def test_reject_returns_end(self):
        """Reject action must route to END."""
        state = {"human_review_decision": {"action": "reject"}}
        assert route_after_human_review(state) == "END"

    def test_edit_recommendation_returns_end(self):
        """edit_recommendation must route to END (changes already applied)."""
        state = {"human_review_decision": {"action": "edit_recommendation"}}
        assert route_after_human_review(state) == "END"

    def test_request_more_evidence_returns_select_candidate(self):
        """request-more-evidence must route to select_candidate."""
        state = {"human_review_decision": {"action": "request-more-evidence", "candidate_id": "c001"}}
        assert route_after_human_review(state) == "select_candidate"

    def test_no_decision_returns_end(self):
        """Missing decision must default to END (safe fallback)."""
        assert route_after_human_review({}) == "END"

    def test_no_action_field_returns_end(self):
        """Decision without action field must default to reject → END."""
        state = {"human_review_decision": {"candidate_id": "c001"}}
        assert route_after_human_review(state) == "END"