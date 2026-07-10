"""
Tests for runtime session management (S13.6).

Exercises the session module API through a scripted multi-turn interrupt/resume
sequence. No CLI, no UI — pure session-level interaction.

Key test cases:
1. start_run: returns thread_id with correct format
2. start_run: parses JD and resumes, creates initial state
3. get_state: returns initial state for a started run
4. get_state: raises ValueError for unknown thread_id
5. list_active_runs: returns active runs
6. reset: clears active threads
7. resume_with_decision: resumes a completed run (no-interrupt path)
8. Concurrent runs: multiple independent runs

No live API keys required — tests use the graph with mock data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

from scoutai.runtime.session import (
    get_state,
    list_active_runs,
    reset,
    resume_with_decision,
    start_run,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def cleanup():
    """Ensure session state is cleaned up before and after each test."""
    reset()
    yield
    reset()


@pytest.fixture
def jd_file() -> Generator[Path, None, None]:
    """Create a temporary JD file."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "jd.txt"
        path.write_text(
            "Software Engineer. Requirements: Python, AWS, 5+ years experience.",
            encoding="utf-8",
        )
        yield path


@pytest.fixture
def resume_files() -> Generator[list[Path], None, None]:
    """Create temporary resume files."""
    with tempfile.TemporaryDirectory() as d:
        paths = [
            Path(d) / "alice.txt",
            Path(d) / "bob.txt",
            Path(d) / "charlie.txt",
        ]
        paths[0].write_text("Alice: 5 years Python, 3 years AWS", encoding="utf-8")
        paths[1].write_text("Bob: 3 years Python, 1 year AWS", encoding="utf-8")
        paths[2].write_text("Charlie: 1 year Python", encoding="utf-8")
        yield paths


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestStartRun:
    def test_returns_thread_id(self, jd_file, resume_files):
        """start_run must return a non-empty string thread_id."""
        tid = start_run(jd_file, resume_files)
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_thread_id_includes_run_prefix(self, jd_file, resume_files):
        """start_run must return a thread_id starting with 'run_'."""
        tid = start_run(jd_file, resume_files)
        assert tid.startswith("run_")

    def test_accepts_custom_run_id(self, jd_file, resume_files):
        """start_run must accept a custom run_id."""
        tid = start_run(jd_file, resume_files, run_id="my-custom-run")
        assert tid == "my-custom-run"

    def test_creates_initial_state(self, jd_file, resume_files):
        """start_run must create initial state with candidates."""
        tid = start_run(jd_file, resume_files)
        state = get_state(tid)
        assert "run_id" in state
        assert "jd" in state
        assert "candidates" in state
        assert len(state["candidates"]) == 3

    def test_parses_resume_names(self, jd_file, resume_files):
        """start_run must use filenames as candidate IDs."""
        tid = start_run(jd_file, resume_files)
        state = get_state(tid)
        candidate_ids = [c["candidate_id"] for c in state["candidates"]]
        assert "alice" in candidate_ids
        assert "bob" in candidate_ids
        assert "charlie" in candidate_ids

    def test_raises_on_missing_resume(self, jd_file):
        """start_run must raise FileNotFoundError for missing resume."""
        with pytest.raises(FileNotFoundError):
            start_run(jd_file, [Path("nonexistent.txt")])


class TestGetState:
    def test_returns_initial_state(self, jd_file, resume_files):
        """get_state must return the initial state for a started run."""
        tid = start_run(jd_file, resume_files)
        state = get_state(tid)
        assert state["step_count"] == 0
        assert len(state["candidates"]) == 3

    def test_raises_for_unknown_thread(self):
        """get_state must raise ValueError for unknown thread_id."""
        with pytest.raises(ValueError, match="No active run found"):
            get_state("unknown-thread")

    def test_raises_after_reset(self, jd_file, resume_files):
        """get_state must raise ValueError after reset."""
        tid = start_run(jd_file, resume_files)
        reset()
        with pytest.raises(ValueError, match="No active run found"):
            get_state(tid)


class TestListActiveRuns:
    def test_returns_empty_initially(self):
        """list_active_runs must return empty list when no runs started."""
        runs = list_active_runs()
        assert runs == []

    def test_returns_active_run(self, jd_file, resume_files):
        """list_active_runs must include a started run."""
        start_run(jd_file, resume_files)
        runs = list_active_runs()
        assert len(runs) == 1
        assert runs[0]["candidate_count"] == 3
        assert runs[0]["status"] == "in_progress"

    def test_multiple_runs(self, jd_file, resume_files):
        """list_active_runs must list multiple independent runs."""
        tid1 = start_run(jd_file, resume_files, run_id="run-one")
        tid2 = start_run(jd_file, resume_files, run_id="run-two")
        runs = list_active_runs()
        assert len(runs) == 2
        thread_ids = {r["thread_id"] for r in runs}
        assert tid1 in thread_ids
        assert tid2 in thread_ids


class TestReset:
    def test_clears_active_threads(self, jd_file, resume_files):
        """reset must clear all active threads."""
        start_run(jd_file, resume_files)
        reset()
        assert list_active_runs() == []

    def test_idempotent(self):
        """reset must be idempotent."""
        reset()
        reset()  # Should not raise


class TestResumeWithDecision:
    def test_resume_completes_run(self, jd_file, resume_files):
        """resume_with_decision must complete a run and return final state."""
        tid = start_run(jd_file, resume_files, run_id="resume-test")

        # Mock the graph invoke to avoid API key requirement
        from scoutai.runtime.session import _graph, _config
        from unittest.mock import MagicMock, patch

        mock_result = {"run_id": "resume-test", "step_count": 42, "shortlist": []}
        _graph.invoke = MagicMock(return_value=mock_result)

        result = resume_with_decision(tid, {"action": "approve"})
        assert isinstance(result, dict)
        assert "run_id" in result
        assert result["run_id"] == "resume-test"

    def test_raises_for_unknown_thread(self):
        """resume_with_decision must raise ValueError for unknown thread_id."""
        with pytest.raises(ValueError, match="No active run found"):
            resume_with_decision("unknown-thread", {"action": "approve"})

    def test_raises_after_reset(self, jd_file, resume_files):
        """resume_with_decision must raise ValueError after reset."""
        tid = start_run(jd_file, resume_files)
        reset()
        with pytest.raises(ValueError, match="No active run found"):
            resume_with_decision(tid, {"action": "approve"})
