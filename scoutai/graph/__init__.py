"""LangGraph outer graph assembly and node definitions."""

from scoutai.graph.app import build_graph, run_graph
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

__all__ = [
    # Graph factory
    "build_graph",
    "run_graph",
    # Nodes
    "setup_role_node",
    "build_rubric_node",
    "select_candidate_node",
    "screen_resume_node",
    "candidate_agent_node",
    "fairness_probe_node",
    "compose_summary_node",
    "human_review_node",
    "schedule_node",
    # Routing functions
    "route_after_select",
    "route_after_screen",
    "route_after_human_review",
]
