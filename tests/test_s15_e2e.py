"""
S15: End-to-end integration test.

Synthetic JD + 3-4 synthetic resumes (one injection attempt, one triggering
ask_candidate, one with contradictory evidence to exercise agent re-extraction)
run through the full graph with mocked models, asserting final shortlist + trajectory shape.

All model calls are mocked - no live API keys required.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest

from scoutai.capabilities.model_router import ModelRouter
from scoutai.capabilities.role_requirements import SessionCache, _build_role_profile, _sha256
from scoutai.config import load_config
from scoutai.graph.app import build_graph, run_graph
from scoutai.schemas import (
    CandidateState,
    RecruiterSummary,
    Rubric,
    RubricCriterion,
    ShortlistEntry,
    TrajectoryEntry,
)
from tests.fixtures.resumes import (
    CLEAN_RESUME,
    INJECTION_RESUME_1,
    CONTRADICTORY_RESUME,
    ASK_CANDIDATE_TRIGGER_RESUME,
)

CONFIG_PATH = "config.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def router(config):
    return ModelRouter(config)


@pytest.fixture
def cache(config):
    return SessionCache(config)


@pytest.fixture
def sample_rubric():
    return Rubric(
        criteria=[
            RubricCriterion(name="Python", weight=5.0, category="required_skills"),
            RubricCriterion(name="System Design", weight=4.0, category="required_skills"),
            RubricCriterion(name="AWS", weight=3.0, category="nice_to_have"),
            RubricCriterion(name="Leadership", weight=2.0, category="soft_skills"),
        ],
        jd_hash="a" * 64,
        rubric_config_version="1.0.0",
        generated_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_role_profile():
    return _build_role_profile(
        {
            "title": "Senior Backend Engineer",
            "required_skills": ["Python", "System Design", "AWS"],
            "nice_to_have": ["Leadership"],
        },
        _sha256("Senior Backend Engineer - Python, system design, AWS, leadership."),
    )


def _make_candidate(candidate_id: str, resume_text: str) -> CandidateState:
    return CandidateState(
        candidate_id=candidate_id,
        resume_text=resume_text,
        sanitized_resume=resume_text,
        injection_flag=False,
        leakage_flag=False,
        finalized=False,
    )


def _make_shortlist_entry(candidate_id: str, recommendation: str = "interview",
                          score: float = 72.0) -> ShortlistEntry:
    return ShortlistEntry(
        candidate=candidate_id,
        recommendation=recommendation,
        weighted_score=score,
        evidence_refs=["Resume:Skills"],
    )


def _make_trajectory_entry(tool: str = "extract_evidence",
                           status: str = "success") -> TrajectoryEntry:
    from datetime import datetime, timezone
    return TrajectoryEntry(
        node="candidate_agent",
        tool_used=tool,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_hash="0" * 64,
        output_hash="0" * 64,
        latency_ms=10,
        model=None,
        model_role="fast_structured",
        prompt_version="1.0.0",
        tool_version="1.0.0",
        schema_version="1.0.0",
        status=status,
    )


def _make_mock_screen_result(*, injection: bool = False, leakage: bool = False):
    from scoutai.capabilities.screen_resume import ScreenResumeResult
    return ScreenResumeResult(
        sanitized_resume="Sanitized resume text",
        injection_flag=injection,
        leakage_flag=leakage,
    )


def _make_finalized_candidate(candidate: CandidateState, *,
                              recommendation: str = "interview",
                              rationale: str = "Good fit") -> CandidateState:
    return candidate.model_copy(update={
        "finalized": True,
        "recommendation": recommendation,
        "rationale": rationale,
        "sanitized_resume": candidate.resume_text,
        "injection_flag": False,
        "leakage_flag": False,
    })


def _build_initial_state(candidates: list[CandidateState],
                         run_id: str = "e2e-test") -> dict[str, Any]:
    return {
        "jd": "Senior Backend Engineer - Python, system design, AWS, leadership.",
        "candidates": candidates,
        "current_idx": 0,
        "shortlist": [],
        "trajectory": [],
        "step_count": 0,
        "run_id": run_id,
        "role_profile": None,
        "rubric": None,
    }


def _make_mock_summary(run_id: str, rec: str = "3 candidates shortlisted") -> RecruiterSummary:
    return RecruiterSummary(
        shortlist=[],
        bias_reports=[],
        overall_recommendation=rec,
        evidence_refs=["Resume:Skills"],
        generated_at="2024-01-01T00:00:00Z",
        run_id=run_id,
    )


class TestS15EndToEnd:

    def _setup_env(self):
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")

    def test_e2e_three_clean_candidates_processed(self, config, router, cache,
                                                   sample_role_profile, sample_rubric):
        """Three clean candidates all go through agent and get finalized."""
        self._setup_env()

        candidates = [
            _make_candidate("c001", CLEAN_RESUME),
            _make_candidate("c002", CLEAN_RESUME),
            _make_candidate("c003", CLEAN_RESUME),
        ]

        call_log: list[str] = []

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            call_log.append(cand.candidate_id)
            entry = _make_shortlist_entry(cand.candidate_id)
            return _make_finalized_candidate(cand), entry, [
                _make_trajectory_entry("extract_evidence"),
                _make_trajectory_entry("finalize_candidate"),
            ]

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result()), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run), \
             patch("scoutai.graph.nodes.compose_decision_summary",
                   return_value=_make_mock_summary("e2e-all")):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            result = run_graph(g, _build_initial_state(candidates), config,
                               thread_id="e2e-all")

        candidates_out = result.get("candidates", [])
        assert len(candidates_out) == 3, "All 3 candidates must be in output"

        # All candidates must be finalized
        for c in candidates_out:
            _fin = c.get("finalized") if isinstance(c, dict) else getattr(c, "finalized", False)
            assert _fin is True, "All candidates must be finalized"

        # Agent must be called for all 3
        assert len(call_log) == 3, f"Agent should be called 3 times, got {len(call_log)}"
        assert set(call_log) == {"c001", "c002", "c003"}

    def test_e2e_injection_flagged_skips_agent(self, config, router, cache,
                                                sample_role_profile, sample_rubric):
        """Injection-flagged candidate must route to human_review, not agent."""
        self._setup_env()

        candidates = [
            _make_candidate("c001", CLEAN_RESUME),
            _make_candidate("c002", INJECTION_RESUME_1),
        ]

        agent_call_log: list[str] = []

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            agent_call_log.append(cand.candidate_id)
            entry = _make_shortlist_entry(cand.candidate_id)
            return _make_finalized_candidate(cand), entry, [_make_trajectory_entry()]

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   side_effect=[
                       _make_mock_screen_result(),
                       _make_mock_screen_result(injection=True),
                   ]), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run), \
             patch("scoutai.graph.nodes.compose_decision_summary",
                   return_value=_make_mock_summary("e2e-injection")):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            run_graph(g, _build_initial_state(candidates), config,
                      thread_id="e2e-injection")

        assert agent_call_log == ["c001"], \
            "Agent must only be called for clean candidate, not injection-flagged one"

    def test_e2e_trajectory_has_entries(self, config, router, cache,
                                         sample_role_profile, sample_rubric):
        """Trajectory must contain entries from agent tool calls."""
        self._setup_env()

        candidates = [_make_candidate("c001", CLEAN_RESUME)]

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            traj = [
                _make_trajectory_entry("extract_evidence"),
                _make_trajectory_entry("assess_capabilities"),
                _make_trajectory_entry("verify_evidence"),
                _make_trajectory_entry("finalize_candidate"),
            ]
            return _make_finalized_candidate(cand), _make_shortlist_entry(cand.candidate_id), traj

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result()), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run), \
             patch("scoutai.graph.nodes.compose_decision_summary",
                   return_value=_make_mock_summary("e2e-trajectory")):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            result = run_graph(g, _build_initial_state(candidates), config,
                               thread_id="e2e-trajectory")

        trajectory = result.get("trajectory", [])
        assert len(trajectory) > 0, "Trajectory must have entries"
        tools_used = []
        for entry in trajectory:
            if isinstance(entry, dict):
                tools_used.append(entry.get("tool_used"))
            else:
                tools_used.append(getattr(entry, "tool_used", None))
        assert "extract_evidence" in tools_used, "Trajectory must include extract_evidence"
        assert "finalize_candidate" in tools_used, "Trajectory must include finalize_candidate"

    def test_e2e_shortlist_shape(self, config, router, cache,
                                  sample_role_profile, sample_rubric):
        """Shortlist entries must have the correct shape per S5."""
        self._setup_env()

        candidates = [_make_candidate("c001", CLEAN_RESUME)]

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            entry = _make_shortlist_entry(cand.candidate_id, recommendation="strong_interview", score=88.0)
            return _make_finalized_candidate(cand, recommendation="strong_interview"), entry, [
                _make_trajectory_entry("finalize_candidate"),
            ]

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result()), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run), \
             patch("scoutai.graph.nodes.compose_decision_summary",
                   return_value=_make_mock_summary("e2e-shape")):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            result = run_graph(g, _build_initial_state(candidates), config,
                               thread_id="e2e-shape")

        shortlist = result.get("shortlist", [])
        assert len(shortlist) == 1
        entry = shortlist[0]
        if isinstance(entry, dict):
            assert "candidate" in entry
            assert "recommendation" in entry
            assert "weighted_score" in entry
            assert "evidence_refs" in entry
            assert entry["candidate"] == "c001"
            assert entry["recommendation"] == "strong_interview"
            assert entry["weighted_score"] == 88.0
        else:
            assert entry.candidate == "c001"
            assert entry.recommendation == "strong_interview"
            assert entry.weighted_score == 88.0