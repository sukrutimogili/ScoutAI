"""
check_availability and propose_interview — deterministic scheduling capabilities (§4.1, S10).

These are fixed-node capabilities — NOT callable by the agent (§7.4, ADR-9).
They are deterministic (calendar API calls), not LLM-driven.

Architecture:
- A SchedulingBackend abstract interface allows swapping between mock and real
  calendar providers without changing business logic.
- MockBackend is the default for demo/testing (config.scheduling.backend == "mock").
- GoogleCalendarBackend is the documented upgrade path when credentials are available.

Spec references: §4.1, §7.4, §10, §12 Open Items.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from scoutai.config import ScoutAIConfig
from scoutai.schemas import AvailabilitySlot, PendingConfirmation

logger = logging.getLogger(__name__)


# ── Scheduling backend interface ──────────────────────────────────────────────


class SchedulingBackend(ABC):
    """Abstract interface for calendar/scheduling backends.

    Implementations:
        MockBackend:      Returns deterministic mock slots (default for demo).
        GoogleCalendarBackend: Real Google Calendar integration (upgrade path).
    """

    @abstractmethod
    def check_availability(
        self, candidate_id: str, week_start: str, config: ScoutAIConfig
    ) -> list[AvailabilitySlot]:
        """Return available time slots for a candidate in the given week.

        Args:
            candidate_id: Unique identifier for the candidate.
            week_start:   ISO 8601 date string for the Monday of the target week.
            config:       ScoutAI configuration.

        Returns:
            List of available AvailabilitySlot objects. May be empty.
        """
        ...

    @abstractmethod
    def propose_interview(
        self, candidate_id: str, slot: AvailabilitySlot, config: ScoutAIConfig
    ) -> PendingConfirmation:
        """Propose an interview at a specific time slot.

        Args:
            candidate_id: Unique identifier for the candidate.
            slot:         The proposed time slot.
            config:       ScoutAI configuration.

        Returns:
            A PendingConfirmation awaiting candidate/interviewer confirmation.
        """
        ...


# ── Mock backend (default for demo/testing) ──────────────────────────────────


class MockBackend(SchedulingBackend):
    """Deterministic mock scheduling backend for demo and testing.

    Returns a fixed set of slots for any candidate/week combination.
    The slots are always in the future relative to the current time, making
    them suitable for both demo runs and tests.
    """

    def check_availability(
        self, candidate_id: str, week_start: str, config: ScoutAIConfig
    ) -> list[AvailabilitySlot]:
        """Return 3 mock slots for the given week."""
        try:
            monday = datetime.fromisoformat(week_start)
            if monday.tzinfo is None:
                monday = monday.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            # Fall back to next Monday if parsing fails
            today = datetime.now(timezone.utc)
            monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
            monday = monday.replace(hour=9, minute=0, second=0, microsecond=0)

        slots = [
            AvailabilitySlot(
                start_iso=(monday + timedelta(days=1, hours=10)).isoformat(),  # Tuesday 10:00
                end_iso=(monday + timedelta(days=1, hours=11)).isoformat(),    # Tuesday 11:00
                timezone="UTC",
            ),
            AvailabilitySlot(
                start_iso=(monday + timedelta(days=2, hours=14)).isoformat(),  # Wednesday 14:00
                end_iso=(monday + timedelta(days=2, hours=15)).isoformat(),    # Wednesday 15:00
                timezone="UTC",
            ),
            AvailabilitySlot(
                start_iso=(monday + timedelta(days=4, hours=9)).isoformat(),   # Friday 09:00
                end_iso=(monday + timedelta(days=4, hours=10)).isoformat(),    # Friday 10:00
                timezone="UTC",
            ),
        ]

        logger.info(
            "MockBackend.check_availability",
            extra={
                "candidate_id": candidate_id,
                "week_start": week_start,
                "slots_count": len(slots),
            },
        )
        return slots

    def propose_interview(
        self, candidate_id: str, slot: AvailabilitySlot, config: ScoutAIConfig
    ) -> PendingConfirmation:
        """Create a PendingConfirmation for the proposed slot."""
        confirmation = PendingConfirmation(
            candidate=candidate_id,
            slot=slot,
            confirmation_id=str(uuid.uuid4()),
            proposed_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
        )

        logger.info(
            "MockBackend.propose_interview",
            extra={
                "candidate_id": candidate_id,
                "slot_start": slot.start_iso,
                "confirmation_id": confirmation.confirmation_id,
            },
        )
        return confirmation


# ── Google Calendar backend (upgrade path) ───────────────────────────────────


class GoogleCalendarBackend(SchedulingBackend):
    """Google Calendar integration for real scheduling.

    This is the documented upgrade path from the mock backend (§12 Open Items).
    Requires valid Google Calendar API credentials configured via the
    GOOGLE_CALENDAR_CREDENTIALS_JSON environment variable.

    NOTE: This is a scaffold for the upgrade path. Full implementation requires
    the google-api-python-client and google-auth-httplib2 packages, which are
    optional dependencies not included in the base pyproject.toml.
    """

    def __init__(self, config: ScoutAIConfig) -> None:
        self._calendar_id: Optional[str] = None
        self._service: Any = None
        self._initialized = False

    def _ensure_initialized(self, config: ScoutAIConfig) -> None:
        """Lazy-initialize the Google Calendar service."""
        if self._initialized:
            return

        import json
        import os

        creds_json = os.getenv(config.scheduling.calendar_env_var)
        if not creds_json:
            raise EnvironmentError(
                f"Google Calendar credentials not found. "
                f"Set the '{config.scheduling.calendar_env_var}' environment variable "
                f"to the path of your service account JSON file."
            )

        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            creds_path = os.path.expanduser(creds_json)
            with open(creds_path, "r") as f:
                creds_info = json.load(f)

            credentials = service_account.Credentials.from_service_account_info(
                creds_info,
                scopes=["https://www.googleapis.com/auth/calendar.readonly"],
            )
            self._service = build("calendar", "v3", credentials=credentials)
            self._calendar_id = creds_info.get("calendar_id", "primary")
            self._initialized = True

            logger.info("GoogleCalendarBackend initialized successfully")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize Google Calendar backend: {e}"
            ) from e

    def check_availability(
        self, candidate_id: str, week_start: str, config: ScoutAIConfig
    ) -> list[AvailabilitySlot]:
        """Query Google Calendar for free slots in the given week."""
        self._ensure_initialized(config)

        try:
            monday = datetime.fromisoformat(week_start)
            if monday.tzinfo is None:
                monday = monday.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            today = datetime.now(timezone.utc)
            monday = today + timedelta(days=(7 - today.weekday()) % 7 or 7)
            monday = monday.replace(hour=0, minute=0, second=0, microsecond=0)

        friday = monday + timedelta(days=5)  # Mon-Fri work week

        try:
            events_result = (
                self._service.events()
                .list(
                    calendarId=self._calendar_id,
                    timeMin=monday.isoformat(),
                    timeMax=friday.isoformat(),
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            events = events_result.get("items", [])

            # Build busy periods from events
            busy_periods: list[tuple[datetime, datetime]] = []
            for event in events:
                start_str = event["start"].get("dateTime", event["start"].get("date"))
                end_str = event["end"].get("dateTime", event["end"].get("date"))
                try:
                    start_dt = datetime.fromisoformat(start_str)
                    end_dt = datetime.fromisoformat(end_str)
                    busy_periods.append((start_dt, end_dt))
                except (ValueError, TypeError):
                    continue

            # Generate free slots (9am-5pm, 1-hour blocks, excluding busy periods)
            slots: list[AvailabilitySlot] = []
            for day_offset in range(5):
                day = monday + timedelta(days=day_offset)
                for hour in range(9, 17):  # 9 AM to 5 PM
                    slot_start = day.replace(hour=hour, minute=0, second=0, microsecond=0)
                    slot_end = slot_start + timedelta(hours=1)

                    # Check if this slot overlaps any busy period
                    is_busy = any(
                        slot_start < busy_end and slot_end > busy_start
                        for busy_start, busy_end in busy_periods
                    )
                    if not is_busy:
                        slots.append(
                            AvailabilitySlot(
                                start_iso=slot_start.isoformat(),
                                end_iso=slot_end.isoformat(),
                                timezone="UTC",
                            )
                        )

            logger.info(
                "GoogleCalendarBackend.check_availability",
                extra={
                    "candidate_id": candidate_id,
                    "week_start": week_start,
                    "free_slots": len(slots),
                    "busy_events": len(events),
                },
            )
            return slots

        except Exception as e:
            logger.error(
                "Google Calendar API call failed",
                extra={"error": str(e)},
            )
            raise

    def propose_interview(
        self, candidate_id: str, slot: AvailabilitySlot, config: ScoutAIConfig
    ) -> PendingConfirmation:
        """Create a calendar event for the proposed interview slot."""
        self._ensure_initialized(config)

        try:
            event = {
                "summary": f"Interview: {candidate_id}",
                "description": f"Scheduled interview with {candidate_id}",
                "start": {
                    "dateTime": slot.start_iso,
                    "timeZone": slot.timezone,
                },
                "end": {
                    "dateTime": slot.end_iso,
                    "timeZone": slot.timezone,
                },
            }

            created_event = (
                self._service.events()
                .insert(calendarId=self._calendar_id, body=event)
                .execute()
            )

            confirmation = PendingConfirmation(
                candidate=candidate_id,
                slot=slot,
                confirmation_id=created_event.get("id", str(uuid.uuid4())),
                proposed_at=datetime.now(timezone.utc).isoformat(),
                status="pending",
            )

            logger.info(
                "GoogleCalendarBackend.propose_interview",
                extra={
                    "candidate_id": candidate_id,
                    "event_id": confirmation.confirmation_id,
                    "slot_start": slot.start_iso,
                },
            )
            return confirmation

        except Exception as e:
            logger.error(
                "Google Calendar event creation failed",
                extra={"error": str(e)},
            )
            raise


# ── Backend factory ──────────────────────────────────────────────────────────


def _create_backend(config: ScoutAIConfig) -> SchedulingBackend:
    """Factory: create the appropriate SchedulingBackend from config."""
    backend_type = config.scheduling.backend.lower()
    if backend_type == "mock":
        return MockBackend()
    elif backend_type == "google_calendar":
        return GoogleCalendarBackend(config)
    else:
        logger.warning(
            "Unknown scheduling backend '%s', falling back to mock",
            backend_type,
        )
        return MockBackend()


# ── Public API ───────────────────────────────────────────────────────────────


def check_availability(
    candidate_id: str,
    week_start: str,
    config: ScoutAIConfig,
    backend: Optional[SchedulingBackend] = None,
) -> list[AvailabilitySlot]:
    """Check available interview time slots for a candidate.

    Fixed-node capability — NOT callable by the agent (§7.4, ADR-9).
    Deterministic (calendar API), not LLM-driven.

    Args:
        candidate_id: Unique identifier for the candidate.
        week_start:   ISO 8601 date string for the Monday of the target week.
        config:       ScoutAI configuration.
        backend:      Optional pre-configured backend. Created from config if None.

    Returns:
        List of available AvailabilitySlot objects. May be empty if no slots
        are available in the requested week.

    Raises:
        RuntimeError: If the calendar backend fails (e.g. API error).
    """
    if backend is None:
        backend = _create_backend(config)

    return backend.check_availability(candidate_id, week_start, config)


def propose_interview(
    candidate_id: str,
    slot: AvailabilitySlot,
    config: ScoutAIConfig,
    backend: Optional[SchedulingBackend] = None,
) -> PendingConfirmation:
    """Propose an interview at a specific time slot.

    Fixed-node capability — NOT callable by the agent (§7.4, ADR-9).
    Deterministic (calendar API), not LLM-driven.

    Args:
        candidate_id: Unique identifier for the candidate.
        slot:         The proposed time slot (from check_availability).
        config:       ScoutAI configuration.
        backend:      Optional pre-configured backend. Created from config if None.

    Returns:
        A PendingConfirmation awaiting candidate/interviewer confirmation.

    Raises:
        RuntimeError: If the calendar backend fails (e.g. API error).
    """
    if backend is None:
        backend = _create_backend(config)

    return backend.propose_interview(candidate_id, slot, config)