"""
Metrics counters and operational logging for ScoutAI (§10, S13).

Architecture:
- MetricsCollector: thread-safe counters for node/tool executions, latencies,
  model usage, and agent-convergence metrics (§11).
- timed_node: decorator that wraps a graph node function with timing and
  metrics recording.
- OperationalLogger: structured JSON logging wrapper that emits node-level
  timing and model-usage events when config.logging.include_node_timings
  or config.logging.include_model_usage is enabled.

Spec references: §10 (operational logging), §11 (evaluation metrics).
"""

from __future__ import annotations

import functools
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Optional

from scoutai.config import ScoutAIConfig

logger = logging.getLogger(__name__)


# ── Metrics data structures ───────────────────────────────────────────────────


@dataclass
class NodeMetrics:
    """Accumulated metrics for a single graph node or tool.

    Spec reference: §10 (operational logging), §11 (evaluation).
    """

    node_name: str
    call_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    error_count: int = 0
    last_called_at: Optional[str] = None

    @property
    def avg_latency_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return round(self.total_latency_ms / self.call_count, 2)

    def record(self, latency_ms: float, error: bool = False) -> None:
        self.call_count += 1
        self.total_latency_ms += latency_ms
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        if error:
            self.error_count += 1
        self.last_called_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_name": self.node_name,
            "call_count": self.call_count,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": self.avg_latency_ms,
            "min_latency_ms": round(self.min_latency_ms, 2) if self.call_count > 0 else 0.0,
            "max_latency_ms": round(self.max_latency_ms, 2) if self.call_count > 0 else 0.0,
            "error_count": self.error_count,
            "last_called_at": self.last_called_at,
        }


@dataclass
class ModelUsageMetrics:
    """Accumulated model usage metrics per model role.

    Spec reference: §10 (include_model_usage).
    """

    model_role: str
    call_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0

    def record(self, latency_ms: float, error: bool = False) -> None:
        self.call_count += 1
        self.total_latency_ms += latency_ms
        if error:
            self.error_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_role": self.model_role,
            "call_count": self.call_count,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "avg_latency_ms": round(self.total_latency_ms / self.call_count, 2) if self.call_count > 0 else 0.0,
            "error_count": self.error_count,
        }


@dataclass
class AgentConvergenceMetrics:
    """Agent convergence metrics per candidate (§11).

    Tracks how many tool calls the agent needed, whether it converged
    (called finalize_candidate) or was force-finalized, and how many
    times it used each tool.

    Spec reference: §11 (evaluation).
    """

    candidate_id: str
    total_tool_calls: int = 0
    converged: bool = False
    force_finalized: bool = False
    tool_usage: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    ask_candidate_used: bool = False
    final_recommendation: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "total_tool_calls": self.total_tool_calls,
            "converged": self.converged,
            "force_finalized": self.force_finalized,
            "tool_usage": dict(self.tool_usage),
            "ask_candidate_used": self.ask_candidate_used,
            "final_recommendation": self.final_recommendation,
        }


# ── MetricsCollector ──────────────────────────────────────────────────────────


class MetricsCollector:
    """Thread-safe metrics collector for the entire pipeline run.

    Accumulates per-node, per-model-role, and per-candidate-agent metrics
    across a single graph invocation. Designed to be created once per run
    and passed through the graph via the config or a closure.

    Spec references: §10, §11.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._node_metrics: dict[str, NodeMetrics] = {}
        self._model_usage: dict[str, ModelUsageMetrics] = {}
        self._agent_convergence: dict[str, AgentConvergenceMetrics] = {}
        self._run_started_at: Optional[str] = None
        self._run_ended_at: Optional[str] = None

    def start_run(self) -> None:
        """Mark the start of a pipeline run."""
        with self._lock:
            self._run_started_at = datetime.now(timezone.utc).isoformat()

    def end_run(self) -> None:
        """Mark the end of a pipeline run."""
        with self._lock:
            self._run_ended_at = datetime.now(timezone.utc).isoformat()

    def record_node(
        self,
        node_name: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record a single node execution.

        Args:
            node_name: The graph node name (e.g. "screen_resume", "candidate_agent").
            latency_ms: Execution latency in milliseconds.
            error: Whether the node execution resulted in an error.
        """
        with self._lock:
            if node_name not in self._node_metrics:
                self._node_metrics[node_name] = NodeMetrics(node_name=node_name)
            self._node_metrics[node_name].record(latency_ms, error=error)

    def record_model_usage(
        self,
        model_role: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Record a model invocation.

        Args:
            model_role: The model role used ("fast_structured" or "high_context").
            latency_ms: Model call latency in milliseconds.
            error: Whether the model call resulted in an error.
        """
        with self._lock:
            if model_role not in self._model_usage:
                self._model_usage[model_role] = ModelUsageMetrics(model_role=model_role)
            self._model_usage[model_role].record(latency_ms, error=error)

    def record_agent_convergence(
        self,
        candidate_id: str,
        total_tool_calls: int,
        converged: bool,
        force_finalized: bool,
        tool_usage: dict[str, int],
        ask_candidate_used: bool,
        final_recommendation: Optional[str] = None,
    ) -> None:
        """Record agent convergence metrics for a candidate.

        Args:
            candidate_id: The candidate identifier.
            total_tool_calls: Total number of tool calls made by the agent.
            converged: Whether the agent called finalize_candidate.
            force_finalized: Whether the agent was force-finalized (budget exhausted).
            tool_usage: Dict mapping tool name to call count.
            ask_candidate_used: Whether ask_candidate was invoked.
            final_recommendation: The final recommendation, if any.
        """
        with self._lock:
            metrics = AgentConvergenceMetrics(
                candidate_id=candidate_id,
                total_tool_calls=total_tool_calls,
                converged=converged,
                force_finalized=force_finalized,
                tool_usage=tool_usage,
                ask_candidate_used=ask_candidate_used,
                final_recommendation=final_recommendation,
            )
            self._agent_convergence[candidate_id] = metrics

    def get_node_metrics(self, node_name: str) -> Optional[NodeMetrics]:
        with self._lock:
            return self._node_metrics.get(node_name)

    def get_all_node_metrics(self) -> dict[str, NodeMetrics]:
        with self._lock:
            return dict(self._node_metrics)

    def get_all_model_usage(self) -> dict[str, ModelUsageMetrics]:
        with self._lock:
            return dict(self._model_usage)

    def get_all_agent_convergence(self) -> dict[str, AgentConvergenceMetrics]:
        with self._lock:
            return dict(self._agent_convergence)

    def summary(self) -> dict[str, Any]:
        """Produce a complete metrics summary dict.

        Returns:
            A dict with node_metrics, model_usage, agent_convergence,
            and run timing information.
        """
        with self._lock:
            return {
                "run_started_at": self._run_started_at,
                "run_ended_at": self._run_ended_at,
                "node_metrics": {
                    name: m.to_dict() for name, m in self._node_metrics.items()
                },
                "model_usage": {
                    role: m.to_dict() for role, m in self._model_usage.items()
                },
                "agent_convergence": {
                    cid: m.to_dict() for cid, m in self._agent_convergence.items()
                },
            }

    def log_summary(self) -> None:
        """Emit the full metrics summary as a structured log line."""
        s = self.summary()
        logger.info(
            "Pipeline run metrics summary",
            extra={"metrics": s},
        )


# ── Timed node wrapper ────────────────────────────────────────────────────────


def timed_node(node_name: str, metrics_collector: MetricsCollector) -> Callable:
    """Decorator that wraps a graph node function with timing and metrics.

    Usage:
        @timed_node("screen_resume", metrics)
        def screen_resume_node(state, config, router):
            ...

    The decorator records:
    - Execution latency (ms)
    - Error count (if the node raises an exception)
    - Last-called timestamp

    Args:
        node_name: The logical name for this node (used in metrics output).
        metrics_collector: The MetricsCollector instance for this run.

    Returns:
        A decorator that wraps the node function.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            error = False
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                error = True
                raise
            finally:
                latency_ms = (time.monotonic() - start) * 1000
                metrics_collector.record_node(node_name, latency_ms, error=error)
        return wrapper
    return decorator


# ── Operational logger ────────────────────────────────────────────────────────


class OperationalLogger:
    """Structured operational logging wrapper.

    Emits JSON-structured log lines for node executions and model calls
    when the corresponding config flags are enabled.

    Spec reference: §10.
    """

    def __init__(self, config: ScoutAIConfig) -> None:
        self._include_timings = config.logging.include_node_timings
        self._include_model_usage = config.logging.include_model_usage
        self._log_format = config.logging.format

    def log_node_execution(
        self,
        node_name: str,
        latency_ms: float,
        state_keys: Optional[list[str]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a node execution event.

        Args:
            node_name: The graph node name.
            latency_ms: Execution latency in milliseconds.
            state_keys: Keys in the state that were modified (optional).
            error: Error message if the node failed (optional).
        """
        if not self._include_timings:
            return

        event = {
            "event": "node_execution",
            "node": node_name,
            "latency_ms": round(latency_ms, 2),
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if state_keys:
            event["state_keys_modified"] = state_keys

        if self._log_format == "json":
            logger.info(json.dumps(event))
        else:
            logger.info(
                f"Node [{node_name}] completed in {latency_ms:.0f}ms"
                + (f" — ERROR: {error}" if error else "")
            )

    def log_model_call(
        self,
        model_role: str,
        model_id: str,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Log a model invocation event.

        Args:
            model_role: The model role ("fast_structured" or "high_context").
            model_id: The resolved model identifier (e.g. "gemini/gemini-2.0-flash").
            latency_ms: Model call latency in milliseconds.
            error: Error message if the call failed (optional).
        """
        if not self._include_model_usage:
            return

        event = {
            "event": "model_call",
            "model_role": model_role,
            "model_id": model_id,
            "latency_ms": round(latency_ms, 2),
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._log_format == "json":
            logger.info(json.dumps(event))
        else:
            logger.info(
                f"Model [{model_role}] {model_id} completed in {latency_ms:.0f}ms"
                + (f" — ERROR: {error}" if error else "")
            )

    def log_agent_iteration(
        self,
        candidate_id: str,
        iteration: int,
        tool_name: str,
        latency_ms: float,
        error: Optional[str] = None,
    ) -> None:
        """Log a single agent tool-call iteration.

        Args:
            candidate_id: The candidate being evaluated.
            iteration: The iteration number (1-based).
            tool_name: The tool that was called.
            latency_ms: Tool execution latency in milliseconds.
            error: Error message if the tool call failed (optional).
        """
        if not self._include_timings:
            return

        event = {
            "event": "agent_iteration",
            "candidate_id": candidate_id,
            "iteration": iteration,
            "tool": tool_name,
            "latency_ms": round(latency_ms, 2),
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if self._log_format == "json":
            logger.info(json.dumps(event))
        else:
            logger.info(
                f"Agent [{candidate_id}] iter {iteration}: {tool_name} "
                f"({latency_ms:.0f}ms)"
                + (f" — ERROR: {error}" if error else "")
            )