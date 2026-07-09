"""
Outer LangGraph graph assembly for ScoutAI (S8).

Compiles the deterministic outer graph per §3.2. The agent has NO influence
over this graph shape — it only operates inside the `candidate_agent` node.

Graph topology (§3.2):

    START
      → setup_role
      → build_rubric
      → select_candidate
          →? screen_resume         (candidates remain)
          →? fairness_probe        (all candidates finalized)
      screen_resume
          →? human_review          (injection_flag or leakage_flag)
          →? candidate_agent       (clean)
      candidate_agent → select_candidate   (per-candidate loop)
      fairness_probe → compose_summary → human_review
      human_review
          →? schedule              (approved + interview recommendation)
          →? select_candidate      (recruiter requests more evidence)
          →? END                   (rejected / hold / edit)
      schedule → END

Node functions live in scoutai/graph/nodes.py (implemented in S7).
This module assembles and compiles the graph only.

Public API:
    build_graph(config, router, cache) → CompiledStateGraph

    run_graph(graph, initial_state, config) — convenience wrapper that sets
    the recursion_limit from ScoutAIConfig (§3.3) so callers don't have to
    remember to pass it manually.

Spec reference: §3.2, §3.3.
ADR references: ADR-4 (deterministic outer graph), ADR-9 (agent in one node).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from scoutai.capabilities.model_router import ModelRouter
from scoutai.capabilities.role_requirements import SessionCache
from scoutai.config import ScoutAIConfig
from scoutai.graph.nodes import (
    build_rubric_node,
    candidate_agent_node,
    compose_summary_node,
    fairness_probe_node,
    human_review_node,
    route_after_human_review,
    route_after_screen,
    route_after_select,
    schedule_node,
    screen_resume_node,
    select_candidate_node,
    setup_role_node,
)

logger = logging.getLogger(__name__)


def build_graph(
    config: ScoutAIConfig,
    router: ModelRouter,
    cache: Optional[SessionCache] = None,
    *,
    checkpointer: Any = None,
    interrupt_before_human_review: bool = True,
) -> Any:
    """
    Build and compile the ScoutAI outer LangGraph graph.

    The compiled graph is the entry point for all pipeline runs. It accepts
    a plain dict (matching the GraphState shape from §5) as initial state and
    runs the full hiring pipeline from JD ingestion through to shortlist output.

    Dependency injection:
        All nodes that need config/router/cache receive them via functools.partial
        at graph-build time. LangGraph passes only `state: dict` to nodes —
        injecting dependencies this way keeps nodes testable and avoids globals.

    recursion_limit:
        Per §3.3, the outer graph is capped at 40 recursive steps. This is passed
        at invoke time via `run_graph()` or directly in the LangGraph config dict:
            graph.invoke(state, config={"recursion_limit": 40})
        It cannot be baked into the compiled graph object in LangGraph 0.2.x —
        use `run_graph()` to get this automatically.

    Args:
        config:
            Fully validated ScoutAIConfig.
        router:
            ModelRouter with provider clients ready.
        cache:
            Optional SessionCache for rubric/role-profile caching (§9).
            A fresh in-memory cache is created when None.
        checkpointer:
            LangGraph checkpointer for mid-run resume (§7.8). Defaults to an
            in-memory MemorySaver — sufficient for single-process runs and tests.
            Pass a persistent checkpointer (e.g. SqliteSaver) for production.
        interrupt_before_human_review:
            When True the graph pauses before `human_review` so the caller can
            inject a recruiter decision. Set False in tests to run straight through
            the S11 stub without pausing.

    Returns:
        A compiled CompiledStateGraph ready to invoke.
    """
    if cache is None:
        cache = SessionCache(config)

    # ── Bind config/router/cache into nodes via functools.partial ─────────────
    # LangGraph node callables receive only (state: dict) → dict.
    # We close over config/router/cache here so each node is a clean function.

    setup_role = functools.partial(
        setup_role_node, config=config, router=router, cache=cache
    )
    build_rubric = functools.partial(
        build_rubric_node, config=config, router=router, cache=cache
    )
    screen_resume = functools.partial(
        screen_resume_node, config=config, router=router
    )
    candidate_agent = functools.partial(
        candidate_agent_node, config=config, router=router
    )
    fairness_probe = functools.partial(
        fairness_probe_node, config=config, router=router
    )
    compose_summary = functools.partial(
        compose_summary_node, config=config, router=router
    )
    # select_candidate, human_review, schedule take only state — no extra deps.

    # ── StateGraph ─────────────────────────────────────────────────────────────
    # Plain dict as the state schema. LangGraph merges node return dicts into
    # state automatically; validated Pydantic models are used inside nodes.
    graph = StateGraph(dict)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("setup_role", setup_role)
    graph.add_node("build_rubric", build_rubric)
    graph.add_node("select_candidate", select_candidate_node)
    graph.add_node("screen_resume", screen_resume)
    graph.add_node("candidate_agent", candidate_agent)
    graph.add_node("fairness_probe", fairness_probe)
    graph.add_node("compose_summary", compose_summary)
    graph.add_node("human_review", human_review_node)
    graph.add_node("schedule", schedule_node)

    # ── Fixed edges ────────────────────────────────────────────────────────────
    # START → setup_role → build_rubric → select_candidate
    graph.add_edge(START, "setup_role")
    graph.add_edge("setup_role", "build_rubric")
    graph.add_edge("build_rubric", "select_candidate")

    # candidate_agent → select_candidate  (per-candidate loop, §3.2)
    graph.add_edge("candidate_agent", "select_candidate")

    # fairness_probe → compose_summary → human_review  (post-loop path, §3.2)
    graph.add_edge("fairness_probe", "compose_summary")
    graph.add_edge("compose_summary", "human_review")

    # schedule → END
    graph.add_edge("schedule", END)

    # ── Conditional edges ──────────────────────────────────────────────────────

    # select_candidate: route to screen_resume (candidates remain) or
    # fairness_probe (all candidates finalized — §3.3 stopping condition).
    graph.add_conditional_edges(
        "select_candidate",
        route_after_select,
        {
            "screen_resume": "screen_resume",
            "fairness_probe": "fairness_probe",
        },
    )

    # screen_resume: route to human_review (injection/leakage flag — §7.1)
    # or candidate_agent (clean résumé — normal path).
    graph.add_conditional_edges(
        "screen_resume",
        route_after_screen,
        {
            "human_review": "human_review",
            "candidate_agent": "candidate_agent",
        },
    )

    # human_review: route to schedule (approved interview candidates), back to
    # select_candidate (recruiter requests more evidence), or END (rejected/hold).
    graph.add_conditional_edges(
        "human_review",
        route_after_human_review,
        {
            "schedule": "schedule",
            "select_candidate": "select_candidate",
            "END": END,
        },
    )

    # ── Compile ────────────────────────────────────────────────────────────────
    if checkpointer is None:
        checkpointer = MemorySaver()

    interrupt_nodes = ["human_review"] if interrupt_before_human_review else []

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_nodes if interrupt_nodes else None,
    )

    logger.info(
        "ScoutAI outer graph compiled",
        extra={
            "nodes": list(graph.nodes.keys()),
            "interrupt_before": interrupt_nodes,
            "recursion_limit": config.graph.recursion_limit,
        },
    )

    return compiled


def run_graph(
    compiled_graph: Any,
    initial_state: dict[str, Any],
    config: ScoutAIConfig,
    *,
    thread_id: str = "default",
    extra_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Invoke the compiled graph with the correct recursion_limit from config.

    This is the recommended way to run the pipeline — it automatically applies
    the recursion_limit (§3.3) and thread_id (required by MemorySaver checkpointer).

    Args:
        compiled_graph: The compiled graph returned by build_graph().
        initial_state:  Initial GraphState dict (jd, candidates list, etc.).
        config:         ScoutAIConfig — provides recursion_limit.
        thread_id:      Checkpointer thread identifier (unique per run).
        extra_config:   Additional LangGraph config overrides.

    Returns:
        Final graph state dict after the run completes or hits an interrupt.
    """
    run_config: dict[str, Any] = {
        "recursion_limit": config.graph.recursion_limit,
        "configurable": {"thread_id": thread_id},
    }
    if extra_config:
        run_config.update(extra_config)

    return compiled_graph.invoke(initial_state, config=run_config)
