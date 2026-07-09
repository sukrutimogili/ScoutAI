"""
S8 tests: outer graph wiring + per-candidate loop.

Covers:
1. select_candidate routing: picks first unfinalized, signals done (current_idx=-1)
2. route_after_select: returns "screen_resume" or "fairness_probe"
3. route_after_screen: returns "human_review" (flagged) or "candidate_agent" (clean)
4. route_after_human_review: returns "END" (stub default)
5. End-to-end per-candidate loop with 2 clean candidates (both loop through agent)
6. Injection-flagged candidate routes to human_review, not candidate_agent
7. All candidates finalized → loop exits to fairness_probe path
8. Graph has exactly 14 edges per §3.2
9. build_graph / run_graph exports work correctly

All model calls are mocked — no live API keys required.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scoutai.capabilities.role_requirements import SessionCache, _build_role_profile, _sha256
from scoutai.config import load_config
from scoutai.capabilities.model_router import ModelRouter
from scoutai.graph.app import build_graph, run_graph
from scoutai.graph.nodes import (
    route_after_human_review,
    route_after_screen,
    route_after_select,
    select_candidate_node,
)
from scoutai.schemas import (
    CandidateState,
    CapabilityAssessment,
    Rubric,
    RubricCriterion,
    ShortlistEntry,
    TrajectoryEntry,
)

CONFIG_PATH = "config.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────────────


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
            RubricCriterion(name="Leadership", weight=2.0, category="soft_skills"),
        ],
        jd_hash="a" * 64,
        rubric_config_version="1.0.0",
        generated_at="2024-01-01T00:00:00Z",
    )


@pytest.fixture
def sample_role_profile():
    return _build_role_profile(
        {"title": "Senior Python Engineer", "required_skills": []},
        _sha256("sample jd"),
    )


def _make_candidate(candidate_id: str, *, finalized: bool = False,
                    injection_flag: bool = False) -> CandidateState:
    """Build a minimal CandidateState for test use."""
    return CandidateState(
        candidate_id=candidate_id,
        resume_text=f"Resume text for {candidate_id}",
        sanitized_resume=f"Sanitized resume for {candidate_id}",
        injection_flag=injection_flag,
        leakage_flag=False,
        finalized=finalized,
    )


def _make_shortlist_entry(candidate_id: str) -> ShortlistEntry:
    return ShortlistEntry(
        candidate=candidate_id,
        recommendation="interview",
        weighted_score=72.0,
        evidence_refs=["Resume:Skills"],
    )


def _make_trajectory_entry(tool: str = "extract_evidence") -> TrajectoryEntry:
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
        status="success",
    )



# ── select_candidate_node ─────────────────────────────────────────────────────


class TestSelectCandidateNode:
    def test_picks_first_unfinalized(self):
        """select_candidate sets current_idx to first candidate where finalized=False."""
        state = {
            "candidates": [
                _make_candidate("c001", finalized=True),
                _make_candidate("c002", finalized=False),
                _make_candidate("c003", finalized=False),
            ],
            "step_count": 0,
        }
        result = select_candidate_node(state)
        assert result["current_idx"] == 1

    def test_picks_idx_zero_when_none_finalized(self):
        state = {
            "candidates": [
                _make_candidate("c001", finalized=False),
                _make_candidate("c002", finalized=False),
            ],
            "step_count": 0,
        }
        result = select_candidate_node(state)
        assert result["current_idx"] == 0

    def test_signals_all_done_with_sentinel(self):
        """When all candidates are finalized, current_idx must be -1 (§3.3)."""
        state = {
            "candidates": [
                _make_candidate("c001", finalized=True),
                _make_candidate("c002", finalized=True),
            ],
            "step_count": 0,
        }
        result = select_candidate_node(state)
        assert result["current_idx"] == -1

    def test_empty_candidates_signals_done(self):
        """Empty candidate list → current_idx=-1 (nothing to process)."""
        state = {"candidates": [], "step_count": 0}
        result = select_candidate_node(state)
        assert result["current_idx"] == -1

    def test_increments_step_count(self):
        state = {"candidates": [_make_candidate("c001")], "step_count": 5}
        result = select_candidate_node(state)
        assert result["step_count"] == 6

    def test_works_with_dict_candidates(self):
        """select_candidate must work when candidates are plain dicts (LangGraph state)."""
        state = {
            "candidates": [
                {"candidate_id": "c001", "finalized": True},
                {"candidate_id": "c002", "finalized": False},
            ],
            "step_count": 0,
        }
        result = select_candidate_node(state)
        assert result["current_idx"] == 1


# ── route_after_select ────────────────────────────────────────────────────────


class TestRouteAfterSelect:
    def test_routes_to_screen_resume_when_candidate_pending(self):
        assert route_after_select({"current_idx": 0}) == "screen_resume"

    def test_routes_to_screen_resume_for_any_positive_idx(self):
        assert route_after_select({"current_idx": 3}) == "screen_resume"

    def test_routes_to_fairness_probe_when_sentinel(self):
        """current_idx == -1 means all done → fairness_probe."""
        assert route_after_select({"current_idx": -1}) == "fairness_probe"


# ── route_after_screen ────────────────────────────────────────────────────────


class TestRouteAfterScreen:
    def _state(self, injection: bool = False, leakage: bool = False, idx: int = 0):
        return {
            "candidates": [{"injection_flag": injection, "leakage_flag": leakage}],
            "current_idx": idx,
        }

    def test_clean_resume_routes_to_agent(self):
        assert route_after_screen(self._state()) == "candidate_agent"

    def test_injection_flag_routes_to_human_review(self):
        assert route_after_screen(self._state(injection=True)) == "human_review"

    def test_leakage_flag_routes_to_human_review(self):
        assert route_after_screen(self._state(leakage=True)) == "human_review"

    def test_both_flags_routes_to_human_review(self):
        assert route_after_screen(self._state(injection=True, leakage=True)) == "human_review"

    def test_invalid_idx_routes_to_human_review(self):
        """Out-of-bounds index → human_review (safe fallback)."""
        state = {"candidates": [], "current_idx": 5}
        assert route_after_screen(state) == "human_review"


# ── route_after_human_review ──────────────────────────────────────────────────


class TestRouteAfterHumanReview:
    def test_stub_returns_end(self):
        """S11 stub always returns END — the test guards this contract."""
        assert route_after_human_review({}) == "END"



# ── Graph structure ───────────────────────────────────────────────────────────


class TestGraphStructure:
    def test_graph_has_14_edges(self, config, router):
        """§3.2: The outer graph must have exactly 14 edges."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")
        g = build_graph(config, router, interrupt_before_human_review=False)
        edges = list(g.get_graph().edges)
        assert len(edges) == 14, f"Expected 14 edges, got {len(edges)}: {edges}"

    def test_graph_has_all_required_nodes(self, config, router):
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")
        g = build_graph(config, router, interrupt_before_human_review=False)
        node_ids = set(g.get_graph().nodes.keys())
        required = {
            "setup_role", "build_rubric", "select_candidate",
            "screen_resume", "candidate_agent", "fairness_probe",
            "compose_summary", "human_review", "schedule",
        }
        assert required.issubset(node_ids), f"Missing nodes: {required - node_ids}"

    def test_build_graph_exports_accessible(self, config, router):
        """build_graph and run_graph are importable from scoutai.graph."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")
        from scoutai.graph import build_graph as bg, run_graph as rg
        assert callable(bg)
        assert callable(rg)



# ── End-to-end per-candidate loop (mocked) ────────────────────────────────────


def _make_mock_screen_result(*, injection: bool = False, leakage: bool = False):
    """Build a ScreenResumeResult-like mock."""
    from scoutai.capabilities.screen_resume import ScreenResumeResult
    return ScreenResumeResult(
        sanitized_resume="Sanitized resume text",
        injection_flag=injection,
        leakage_flag=leakage,
    )


def _make_finalized_candidate(candidate: CandidateState) -> CandidateState:
    """Return a copy of the candidate marked as finalized."""
    return candidate.model_copy(update={
        "finalized": True,
        "recommendation": "interview",
        "rationale": "Good fit",
        "sanitized_resume": "Sanitized resume text",
        "injection_flag": False,
        "leakage_flag": False,
    })


class TestEndToEndLoop:
    """
    End-to-end tests for the per-candidate loop.

    Strategy: patch at the capability level so the graph wiring is exercised
    but no real LLM calls are made.

    Patched:
      - scoutai.graph.nodes.extract_role_requirements → returns a RoleProfile
      - scoutai.graph.nodes.generate_rubric           → returns a Rubric
      - scoutai.graph.nodes.screen_resume             → returns ScreenResumeResult
      - scoutai.agent.harness.AgentHarness.run        → returns (finalized_candidate,
                                                          shortlist_entry, trajectory)
    """

    def _build_initial_state(
        self, candidates: list[CandidateState], run_id: str = "test-run"
    ) -> dict[str, Any]:
        return {
            "jd": "Senior Python Engineer — 5+ years, leadership experience.",
            "candidates": candidates,
            "current_idx": 0,
            "shortlist": [],
            "trajectory": [],
            "step_count": 0,
            "run_id": run_id,
            "role_profile": None,
            "rubric": None,
        }

    def test_single_clean_candidate_completes_loop(
        self, config, router, cache, sample_role_profile, sample_rubric
    ):
        """A single clean candidate should be finalized and appear on the shortlist."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")

        candidate = _make_candidate("c001")
        entry = _make_shortlist_entry("c001")
        traj = [_make_trajectory_entry()]

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            return _make_finalized_candidate(cand), entry, traj

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result()), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            result = run_graph(g, self._build_initial_state([candidate]), config,
                               thread_id="test-single")

        candidates_out = result.get("candidates", [])
        assert len(candidates_out) == 1
        finalized = candidates_out[0]
        fin_flag = finalized.get("finalized") if isinstance(finalized, dict) \
            else getattr(finalized, "finalized", False)
        assert fin_flag is True, "Candidate must be finalized after the loop"

        shortlist = result.get("shortlist", [])
        assert len(shortlist) == 1
        rec = shortlist[0].get("recommendation") if isinstance(shortlist[0], dict) \
            else shortlist[0].recommendation
        assert rec == "interview"

    def test_two_clean_candidates_both_finalized(
        self, config, router, cache, sample_role_profile, sample_rubric
    ):
        """Both candidates must pass through candidate_agent and be finalized."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")

        candidates = [_make_candidate("c001"), _make_candidate("c002")]

        call_log: list[str] = []

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            call_log.append(cand.candidate_id)
            entry = _make_shortlist_entry(cand.candidate_id)
            return _make_finalized_candidate(cand), entry, [_make_trajectory_entry()]

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result()), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            result = run_graph(g, self._build_initial_state(candidates), config,
                               thread_id="test-two")

        assert set(call_log) == {"c001", "c002"}, \
            f"Both candidates must pass through agent. Got: {call_log}"

        shortlist = result.get("shortlist", [])
        assert len(shortlist) == 2

    def test_injection_flagged_candidate_skips_agent(
        self, config, router, cache, sample_role_profile, sample_rubric
    ):
        """
        A candidate with injection_flag=True must route to human_review,
        NOT to candidate_agent. The agent harness must not be called.
        """
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")

        candidate = _make_candidate("c_bad")
        agent_call_log: list[str] = []

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            agent_call_log.append(cand.candidate_id)
            return _make_finalized_candidate(cand), _make_shortlist_entry(cand.candidate_id), []

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result(injection=True)), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            run_graph(g, self._build_initial_state([candidate]), config,
                      thread_id="test-flagged")

        assert agent_call_log == [], \
            "AgentHarness.run must NOT be called for injection-flagged résumés"

    def test_all_candidates_finalized_reaches_fairness_probe(
        self, config, router, cache, sample_role_profile, sample_rubric
    ):
        """After all candidates are finalized, the loop must reach fairness_probe."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")

        candidates = [_make_candidate("c001"), _make_candidate("c002")]
        fairness_call_log: list[int] = []

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            return _make_finalized_candidate(cand), _make_shortlist_entry(cand.candidate_id), []

        def mock_fairness_probe(state, config=None, router=None):
            # Patched via scoutai.graph.app.fairness_probe_node so the _fairness_probe
            # closure in build_graph picks it up. Must accept config/router kwargs
            # because the closure calls: fairness_probe_node(state, config=..., router=...)
            fairness_call_log.append(1)
            return {"step_count": state.get("step_count", 0) + 1}

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result()), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run), \
             patch("scoutai.graph.app.fairness_probe_node",
                   mock_fairness_probe):

            # Build graph inside the patch so the _fairness_probe closure captures
            # the patched fairness_probe_node from scoutai.graph.nodes.
            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            run_graph(g, self._build_initial_state(candidates), config,
                      thread_id="test-fairness")

        assert fairness_call_log, \
            "fairness_probe_node must be called after all candidates are finalized"

        assert fairness_call_log, \
            "fairness_probe_node must be called after all candidates are finalized"

    def test_step_count_increments_per_node(
        self, config, router, cache, sample_role_profile, sample_rubric
    ):
        """step_count must increase as nodes execute."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")

        candidate = _make_candidate("c001")

        def mock_harness_run(self_harness, cand, role_profile, rubric):
            return _make_finalized_candidate(cand), _make_shortlist_entry(cand.candidate_id), []

        with patch("scoutai.graph.nodes.extract_role_requirements",
                   return_value=sample_role_profile), \
             patch("scoutai.graph.nodes.generate_rubric",
                   return_value=sample_rubric), \
             patch("scoutai.graph.nodes.screen_resume",
                   return_value=_make_mock_screen_result()), \
             patch("scoutai.agent.harness.AgentHarness.run",
                   mock_harness_run):

            g = build_graph(config, router, cache=cache,
                            interrupt_before_human_review=False)
            result = run_graph(g, self._build_initial_state([candidate]), config,
                               thread_id="test-step-count")

        assert result.get("step_count", 0) > 0, \
            "step_count must be > 0 after a complete run"

    def test_run_graph_applies_recursion_limit(self, config, router, cache):
        """run_graph must pass recursion_limit=40 from config to the graph."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")

        invoke_configs: list[dict] = []

        g = build_graph(config, router, cache=cache,
                        interrupt_before_human_review=False)
        original_invoke = g.invoke

        def tracking_invoke(state, config=None, **kwargs):
            if config:
                invoke_configs.append(config)
            # Return a minimal result so the test doesn't hang
            return {"step_count": 0, "candidates": [], "shortlist": []}

        g.invoke = tracking_invoke
        run_graph(g, {"jd": "", "candidates": []}, config, thread_id="test-rl")

        assert invoke_configs, "invoke must be called"
        assert invoke_configs[0].get("recursion_limit") == config.graph.recursion_limit



# ── Regression: StateGraph(dict) silently drops untouched keys ────────────────


class TestStateGraphTypedDictRegression:
    """
    Regression guard for the LangGraph 0.2 state-merging behaviour.

    With StateGraph(dict), keys that no node explicitly writes are silently
    dropped after the first node execution. This caused `candidates` to
    disappear after `setup_role` ran (only returning `role_profile` and
    `step_count`), making `select_candidate` see an empty list and immediately
    route to `fairness_probe`.

    The fix: declare the state schema as a TypedDict (GraphStateDict) so
    LangGraph tracks all declared fields from the start, regardless of whether
    every node touches them.

    If someone "simplifies" GraphStateDict back to a plain dict later, this
    test will fail and explain why.
    """

    def test_typeddict_state_preserves_untouched_keys(self):
        """GraphStateDict schema must preserve keys no node writes."""
        from typing import TypedDict as TD
        from langgraph.graph import StateGraph as SG, START, END
        from langgraph.checkpoint.memory import MemorySaver

        class Schema(TD, total=False):
            x: int
            untouched: list

        # node_a only writes 'x'; 'untouched' is never written by any node
        sg = SG(Schema)
        sg.add_node("node_a", lambda s: {"x": 99})
        sg.add_edge(START, "node_a")
        sg.add_edge("node_a", END)
        compiled = sg.compile(checkpointer=MemorySaver())

        result = compiled.invoke(
            {"x": 0, "untouched": [1, 2, 3]},
            config={"configurable": {"thread_id": "reg-typeddict"}},
        )
        assert result.get("untouched") == [1, 2, 3], (
            "TypedDict schema must preserve keys that no node writes. "
            "If this fails, GraphStateDict was changed back to plain dict."
        )

    def test_plain_dict_state_drops_untouched_keys(self):
        """
        Documents the plain-dict behaviour we deliberately avoid.

        StateGraph(dict) drops keys no node touches. This test pins that
        behaviour so it's explicit — not a bug we discovered by accident.
        """
        from langgraph.graph import StateGraph as SG, START, END
        from langgraph.checkpoint.memory import MemorySaver

        sg = SG(dict)
        sg.add_node("node_a", lambda s: {"x": 99})
        sg.add_edge(START, "node_a")
        sg.add_edge("node_a", END)
        compiled = sg.compile(checkpointer=MemorySaver())

        result = compiled.invoke(
            {"x": 0, "untouched": [1, 2, 3]},
            config={"configurable": {"thread_id": "reg-dict"}},
        )
        # This is the WRONG behaviour — documented here, not used in production.
        assert result.get("untouched") is None, (
            "StateGraph(dict) is expected to drop untouched keys. "
            "If LangGraph changed this, the regression guard in "
            "GraphStateDict may no longer be necessary (but verify first)."
        )

    def test_build_graph_uses_typed_state_schema(self, config, router):
        """build_graph must use GraphStateDict (TypedDict), not plain dict."""
        import os
        os.environ.setdefault("GOOGLE_API_KEY", "test")
        os.environ.setdefault("GROQ_API_KEY", "test")
        from scoutai.graph.app import GraphStateDict
        from typing import get_type_hints
        # GraphStateDict must declare 'candidates' so LangGraph tracks it
        assert "candidates" in GraphStateDict.__annotations__, \
            "GraphStateDict must declare 'candidates' field"
        assert "jd" in GraphStateDict.__annotations__, \
            "GraphStateDict must declare 'jd' field"
        # Verify the graph is compiled from a TypedDict schema, not plain dict
        g = build_graph(config, router, interrupt_before_human_review=False)
        # The graph's input schema should contain our declared keys
        schema = g.get_input_schema()
        assert schema is not None
