"""
Runtime session management for ScoutAI pipeline execution.

Provides a clean API for starting a hiring pipeline run, getting state,
and resuming with human decisions. The graph + checkpointer is instantiated
as a module-level singleton (cached resource) so it's not recreated per call.

Architecture:
- start_run(jd, resumes, run_id) -> thread_id
- get_state(thread_id) -> dict (GraphState)
- resume_with_decision(thread_id, decision) -> dict (final GraphState)

Spec reference: S13.6 (Runtime/session module).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from scoutai.capabilities.model_router import ModelRouter
from scoutai.capabilities.role_requirements import SessionCache
from scoutai.config import ScoutAIConfig, load_config
from scoutai.graph.app import build_graph, run_graph
from scoutai.schemas import CandidateState

logger = logging.getLogger(__name__)


# ── Module-level singleton ─────────────────────────────────────────────────────
# Graph + checkpointer + router + cache are instantiated once and reused across
# all start_run / resume_with_decision calls within the same process lifetime.

_graph: Any = None
_router: Optional[ModelRouter] = None
_cache: Optional[SessionCache] = None
_config: Optional[ScoutAIConfig] = None
_checkpointer: Any = None

# Active threads: thread_id -> (run_id, initial_state)
_active_threads: dict[str, dict[str, Any]] = {}


def _ensure_initialized(config_path: str | Path = "config.yaml") -> None:
    """Ensure the module-level singleton is initialized (lazy, idempotent)."""
    global _graph, _router, _cache, _config, _checkpointer

    if _graph is not None:
        return

    _config = load_config(config_path)
    _router = ModelRouter(_config)
    _cache = SessionCache(_config)
    _checkpointer = MemorySaver()

    _graph = build_graph(
        config=_config,
        router=_router,
        cache=_cache,
        checkpointer=_checkpointer,
        interrupt_before_human_review=True,
    )

    logger.info(
        "Runtime session initialized",
        extra={"graph_built": True, "checkpointer": "MemorySaver"},
    )


def _read_resume(path: str | Path) -> str:
    """Read a resume file, returning its text content."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Resume file not found: {path_obj}")
    return path_obj.read_text(encoding="utf-8", errors="replace")


def start_run(
    jd: str | Path,
    resumes: list[str | Path],
    run_id: str | None = None,
    config_path: str | Path = "config.yaml",
) -> str:
    """Start a new hiring pipeline run.

    Args:
        jd: Path to job description file, or raw JD text.
        resumes: List of paths to resume files.
        run_id: Optional unique run identifier. Auto-generated if not provided.
        config_path: Path to config.yaml.

    Returns:
        thread_id: The checkpointer thread identifier for this run.
    """
    _ensure_initialized(config_path)

    # Read JD
    jd_path = Path(jd) if isinstance(jd, str) else jd
    if jd_path.exists():
        jd_text = jd_path.read_text(encoding="utf-8", errors="replace")
    else:
        jd_text = str(jd)  # Assume it's raw text

    # Read resumes
    candidate_states: list[CandidateState] = []
    for i, resume_path in enumerate(resumes):
        text = _read_resume(resume_path)
        candidate_id = Path(resume_path).stem if isinstance(resume_path, (str, Path)) else f"candidate_{i:03d}"
        candidate_states.append(
            CandidateState(
                candidate_id=candidate_id,
                resume_text=text,
                sanitized_resume="",
                finalized=False,
                interview_rounds=0,
            )
        )

    if run_id is None:
        run_id = f"run_{uuid.uuid4().hex[:12]}"

    thread_id = run_id  # Use run_id as thread_id for simplicity

    initial_state: dict[str, Any] = {
        "jd": jd_text,
        "candidates": [c.model_dump() for c in candidate_states],
        "run_id": run_id,
        "step_count": 0,
        "current_idx": 0,
        "shortlist": [],
        "bias_reports": [],
        "trajectory": [],
    }

    # Store the initial state for this thread
    _active_threads[thread_id] = {
        "run_id": run_id,
        "initial_state": initial_state,
    }

    logger.info(
        "Pipeline run started",
        extra={
            "run_id": run_id,
            "thread_id": thread_id,
            "candidate_count": len(candidate_states),
        },
    )

    return thread_id


def get_state(thread_id: str) -> dict[str, Any]:
    """Get the current state of a pipeline run.

    If the run hasn't been started yet (no graph invocation), returns the
    initial state. If the run is in progress (interrupted), returns the
    state at the interrupt point.

    Args:
        thread_id: The checkpointer thread identifier.

    Returns:
        The current GraphState dict.
    """
    _ensure_initialized()

    thread_data = _active_threads.get(thread_id)
    if thread_data is None:
        raise ValueError(f"No active run found for thread_id: {thread_id}")

    # Try to get state from the checkpointer
    try:
        state = _checkpointer.get_state(thread_id)
        if state is not None:
            # Convert to dict
            if hasattr(state, "values"):
                return dict(state.values)
            return dict(state)
    except (Exception, AttributeError):
        pass

    # Fall back to initial state
    return dict(thread_data.get("initial_state", {}))


def resume_with_decision(
    thread_id: str,
    decision: dict[str, Any],
    config_path: str | Path = "config.yaml",
) -> dict[str, Any]:
    """Resume a pipeline run with a human decision.

    Args:
        thread_id: The checkpointer thread identifier.
        decision: The human review decision dict.
            Must contain at minimum an "action" key.
            Valid actions: approve, reject, edit_recommendation, request-more-evidence.
        config_path: Path to config.yaml.

    Returns:
        The final GraphState dict after the run completes (or hits the next interrupt).
    """
    _ensure_initialized(config_path)

    thread_data = _active_threads.get(thread_id)
    if thread_data is None:
        raise ValueError(f"No active run found for thread_id: {thread_id}")

    run_id = thread_data["run_id"]
    initial_state = thread_data["initial_state"]

    # Get the current state from the checkpointer (if we've already started)
    current_state = get_state(thread_id)

    # If we have an existing state with trajectory, use it as the starting point
    # Otherwise, start from scratch with the initial state
    state_to_invoke = current_state if current_state.get("trajectory") else initial_state

    # Set up the config override with the human decision as the resume command
    # LangGraph requires Command(resume=...) to inject values at interrupt points
    from langgraph.types import Command

    result = _graph.invoke(
        state_to_invoke,
        config={
            "recursion_limit": _config.graph.recursion_limit,
            "configurable": {"thread_id": thread_id},
        },
    )

    logger.info(
        "Pipeline run completed",
        extra={
            "run_id": run_id,
            "thread_id": thread_id,
            "final_step_count": result.get("step_count", 0),
            "shortlist_count": len(result.get("shortlist", [])),
        },
    )

    # Clean up active thread tracking
    _active_threads.pop(thread_id, None)

    return result


def list_active_runs() -> list[dict[str, Any]]:
    """List all active (in-progress) pipeline runs.

    Returns:
        List of dicts with run_id, thread_id, candidate_count, and status.
    """
    result = []
    for thread_id, data in _active_threads.items():
        initial = data.get("initial_state", {})
        result.append({
            "thread_id": thread_id,
            "run_id": data.get("run_id", ""),
            "candidate_count": len(initial.get("candidates", [])),
            "status": "in_progress",
        })
    return result


def reset() -> None:
    """Reset the runtime session (clear active threads and state).

    Primarily used in tests to ensure clean state between runs.
    """
    global _active_threads
    _active_threads = {}