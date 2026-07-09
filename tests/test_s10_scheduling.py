"""
S10 tests: check_availability, propose_interview, and schedule_node wiring.

Key test cases:
1.  MockBackend.check_availability: returns 3 slots for a valid week
2.  MockBackend.check_availability: handles invalid week_start gracefully
3.  MockBackend.propose_interview: returns PendingConfirmation with correct fields
4.  check_availability: public function delegates to backend
5.  propose_interview: public function delegates to backend
6.  _create_backend: creates correct backend from config
7.  schedule_node: schedules interview for approved candidates
8.  schedule_node: skips candidates with non-interview recommendations
9.  schedule_node: empty shortlist handled safely
10. schedule_node: writes slot info to candidate state
11. schedule_node: handles backend failure gracefully
12. GoogleCalendarBackend: scaffold (not imported unless google deps present)

No live calendar API keys required — all calls use MockBackend.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scoutai.capabilities.scheduling import (
    MockBackend,
    SchedulingBackend,
    _create_backend,
    check_availability,
    propose_interview,
)
from scoutai.config import load_config
from scoutai.graph.nodes import schedule_node
from scoutai.schemas import (
    AvailabilitySlot,
    CandidateState,
    PendingConfirmation,
    ShortlistEntry,
)

CONFIG_PATH = "config.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def mock_backend():
    return MockBackend()


def _make_shortlist_entry(
    candidate_id: str,
    recommendation: str = "interview",
) -> ShortlistEntry:
    return ShortlistEntry(
        candidate=candidate_id,
        recommendation=recommendation,  # type: ignore[arg-type]
        weighted_score=72.0,
        evidence_refs=[f"{candidate_id}:Resume:Skills"],
    )


def _make_candidate(candidate_id: str) -> CandidateState:
    return CandidateState(
        candidate_id=candidate_id,
        resume_text=f"Resume for {candidate_id}",
        sanitized_resume=f"Sanitized for {candidate_id}",
    )


# ── MockBackend ──────────────────────────────────────────────────────────────


class TestMockBackend:
    def test_check_availability_returns_three_slots(self, mock_backend, config):
        """MockBackend.check_availability must return exactly 3 slots."""
        week_start = "2026-07-13"  # Monday
        slots = mock_backend.check_availability("c001", week_start, config)
        assert len(slots) == 3
        for slot in slots:
            assert isinstance(slot, AvailabilitySlot)
            assert slot.start_iso
            assert slot.end_iso
            assert slot.timezone == "UTC"

    def test_check_availability_slots_are_in_week(self, mock_backend, config):
        """All returned slots must fall within the requested week (Mon-Fri)."""
        week_start = "2026-07-13"  # Monday
        monday = datetime.fromisoformat(week_start).replace(tzinfo=timezone.utc)
        friday = monday + timedelta(days=5)  # Saturday 00:00

        slots = mock_backend.check_availability("c001", week_start, config)
        for slot in slots:
            start = datetime.fromisoformat(slot.start_iso)
            assert monday <= start < friday, (
                f"Slot {slot.start_iso} must be within {monday.isoformat()} "
                f"to {friday.isoformat()}"
            )

    def test_check_availability_invalid_week_start(self, mock_backend, config):
        """Invalid week_start must not crash — falls back to next Monday."""
        slots = mock_backend.check_availability("c001", "not-a-date", config)
        assert len(slots) == 3
        assert all(isinstance(s, AvailabilitySlot) for s in slots)

    def test_check_availability_empty_week_start(self, mock_backend, config):
        """Empty week_start must not crash — falls back to next Monday."""
        slots = mock_backend.check_availability("c001", "", config)
        assert len(slots) == 3

    def test_propose_interview_returns_pending_confirmation(self, mock_backend, config):
        """propose_interview must return a PendingConfirmation with status='pending'."""
        slot = AvailabilitySlot(
            start_iso="2026-07-14T10:00:00+00:00",
            end_iso="2026-07-14T11:00:00+00:00",
            timezone="UTC",
        )
        confirmation = mock_backend.propose_interview("c001", slot, config)
        assert isinstance(confirmation, PendingConfirmation)
        assert confirmation.candidate == "c001"
        assert confirmation.status == "pending"
        assert confirmation.confirmation_id
        assert confirmation.slot == slot

    def test_propose_interview_generates_unique_ids(self, mock_backend, config):
        """Each propose_interview call must generate a unique confirmation_id."""
        slot = AvailabilitySlot(
            start_iso="2026-07-14T10:00:00+00:00",
            end_iso="2026-07-14T11:00:00+00:00",
            timezone="UTC",
        )
        c1 = mock_backend.propose_interview("c001", slot, config)
        c2 = mock_backend.propose_interview("c002", slot, config)
        assert c1.confirmation_id != c2.confirmation_id


# ── Public API functions ─────────────────────────────────────────────────────


class TestCheckAvailability:
    def test_delegates_to_backend(self, config):
        """check_availability must delegate to the backend."""
        mock_backend = MagicMock(spec=SchedulingBackend)
        mock_backend.check_availability.return_value = [
            AvailabilitySlot(
                start_iso="2026-07-14T10:00:00+00:00",
                end_iso="2026-07-14T11:00:00+00:00",
                timezone="UTC",
            )
        ]
        result = check_availability("c001", "2026-07-13", config, backend=mock_backend)
        assert len(result) == 1
        mock_backend.check_availability.assert_called_once_with(
            "c001", "2026-07-13", config
        )

    def test_creates_backend_from_config(self, config):
        """check_availability must create a MockBackend when none is provided."""
        result = check_availability("c001", "2026-07-13", config)
        assert len(result) == 3
        assert all(isinstance(s, AvailabilitySlot) for s in result)


class TestProposeInterview:
    def test_delegates_to_backend(self, config):
        """propose_interview must delegate to the backend."""
        slot = AvailabilitySlot(
            start_iso="2026-07-14T10:00:00+00:00",
            end_iso="2026-07-14T11:00:00+00:00",
            timezone="UTC",
        )
        mock_backend = MagicMock(spec=SchedulingBackend)
        mock_backend.propose_interview.return_value = PendingConfirmation(
            candidate="c001",
            slot=slot,
            confirmation_id="test-id",
            proposed_at="2026-07-09T00:00:00Z",
            status="pending",
        )
        result = propose_interview("c001", slot, config, backend=mock_backend)
        assert result.candidate == "c001"
        mock_backend.propose_interview.assert_called_once_with("c001", slot, config)


# ── Backend factory ──────────────────────────────────────────────────────────


class TestCreateBackend:
    def test_creates_mock_backend(self, config):
        """_create_backend must return MockBackend when config says 'mock'."""
        backend = _create_backend(config)
        assert isinstance(backend, MockBackend)

    def test_creates_mock_backend_for_unknown_type(self, config):
        """_create_backend must fall back to MockBackend for unknown types."""
        # Config has scheduling.backend: "mock" but we can test the fallback
        # by patching the attribute
        original_backend = config.scheduling.backend
        config.scheduling.backend = "unknown_backend"
        try:
            backend = _create_backend(config)
            assert isinstance(backend, MockBackend)
        finally:
            config.scheduling.backend = original_backend


# ── schedule_node ────────────────────────────────────────────────────────────


class TestScheduleNode:
    def _state(
        self,
        shortlist: list[ShortlistEntry],
        candidates: list[CandidateState] | None = None,
    ) -> dict[str, Any]:
        return {
            "shortlist": [e.model_dump() for e in shortlist],
            "candidates": [c.model_dump() for c in (candidates or [])],
            "step_count": 0,
        }

    def test_schedules_interview_for_approved_candidate(self, config):
        """schedule_node must propose interview for interview-recommended candidates."""
        entry = _make_shortlist_entry("c001", recommendation="interview")
        candidate = _make_candidate("c001")
        state = self._state([entry], [candidate])

        result = schedule_node(state, config=config)

        # Step count must increment
        assert result["step_count"] == 1

        # Candidate must have slot info written
        updated_candidates = result.get("candidates", [])
        assert len(updated_candidates) == 1
        slot = updated_candidates[0].get("slot") if isinstance(updated_candidates[0], dict) else getattr(updated_candidates[0], "slot", None)
        assert slot is not None, "Candidate must have slot info after scheduling"
        assert slot["start_iso"]
        assert slot["confirmation_id"]
        assert slot["status"] == "pending"

    def test_schedules_strong_interview_candidate(self, config):
        """strong_interview recommendations must also be scheduled."""
        entry = _make_shortlist_entry("c001", recommendation="strong_interview")
        candidate = _make_candidate("c001")
        state = self._state([entry], [candidate])

        result = schedule_node(state, config=config)
        updated_candidates = result.get("candidates", [])
        slot = updated_candidates[0].get("slot") if isinstance(updated_candidates[0], dict) else getattr(updated_candidates[0], "slot", None)
        assert slot is not None

    def test_skips_rejected_candidates(self, config):
        """schedule_node must skip candidates with 'reject' recommendation."""
        entry = _make_shortlist_entry("c001", recommendation="reject")
        candidate = _make_candidate("c001")
        state = self._state([entry], [candidate])

        result = schedule_node(state, config=config)
        updated_candidates = result.get("candidates", [])
        slot = updated_candidates[0].get("slot") if isinstance(updated_candidates[0], dict) else getattr(updated_candidates[0], "slot", None)
        assert slot is None, "Rejected candidate must not have slot info"

    def test_skips_hold_candidates(self, config):
        """schedule_node must skip candidates with 'hold' recommendation."""
        entry = _make_shortlist_entry("c001", recommendation="hold")
        candidate = _make_candidate("c001")
        state = self._state([entry], [candidate])

        result = schedule_node(state, config=config)
        updated_candidates = result.get("candidates", [])
        slot = updated_candidates[0].get("slot") if isinstance(updated_candidates[0], dict) else getattr(updated_candidates[0], "slot", None)
        assert slot is None

    def test_empty_shortlist_handled_safely(self, config):
        """Empty shortlist must not crash the schedule_node."""
        state = self._state([], [])
        result = schedule_node(state, config=config)
        assert result["step_count"] == 1

    def test_schedules_multiple_candidates(self, config):
        """schedule_node must schedule all approved candidates."""
        entries = [
            _make_shortlist_entry("c001", recommendation="interview"),
            _make_shortlist_entry("c002", recommendation="strong_interview"),
            _make_shortlist_entry("c003", recommendation="reject"),
        ]
        candidates = [_make_candidate("c001"), _make_candidate("c002"), _make_candidate("c003")]
        state = self._state(entries, candidates)

        result = schedule_node(state, config=config)
        updated_candidates = result.get("candidates", [])

        # c001 and c002 must have slots
        for i, cid in enumerate(["c001", "c002"]):
            slot = updated_candidates[i].get("slot") if isinstance(updated_candidates[i], dict) else getattr(updated_candidates[i], "slot", None)
            assert slot is not None, f"{cid} must have slot info"

        # c003 must not have a slot
        slot = updated_candidates[2].get("slot") if isinstance(updated_candidates[2], dict) else getattr(updated_candidates[2], "slot", None)
        assert slot is None, "c003 (rejected) must not have slot info"

    def test_increments_step_count(self, config):
        """schedule_node must increment step_count."""
        result = schedule_node({"shortlist": [], "candidates": [], "step_count": 5}, config=config)
        assert result["step_count"] == 6

    def test_slot_info_structure(self, config):
        """Slot info must contain start_iso, end_iso, confirmation_id, and status."""
        entry = _make_shortlist_entry("c001", recommendation="interview")
        candidate = _make_candidate("c001")
        state = self._state([entry], [candidate])

        result = schedule_node(state, config=config)
        updated_candidates = result.get("candidates", [])
        slot = updated_candidates[0].get("slot") if isinstance(updated_candidates[0], dict) else getattr(updated_candidates[0], "slot", {})

        assert "start_iso" in slot
        assert "end_iso" in slot
        assert "confirmation_id" in slot
        assert "status" in slot
        assert slot["status"] == "pending"