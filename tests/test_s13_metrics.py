"""
S13 tests: operational logging and metrics counters (§10, §11).

Key test cases:
1.  NodeMetrics: records call count, latency, min/max/avg
2.  NodeMetrics: tracks error count separately
3.  ModelUsageMetrics: records call count and latency per model role
4.  AgentConvergenceMetrics: tracks tool usage and convergence
5.  MetricsCollector: records and retrieves node metrics
6.  MetricsCollector: records and retrieves model usage
7.  MetricsCollector: records agent convergence metrics
8.  MetricsCollector: summary produces complete dict
9.  MetricsCollector: start_run/end_run timestamps
10. MetricsCollector: thread-safe concurrent recording
11. timed_node: wraps function with timing
12. timed_node: records error on exception
13. OperationalLogger: logs node execution when enabled
14. OperationalLogger: logs model call when enabled
15. OperationalLogger: respects include_node_timings flag
16. OperationalLogger: respects include_model_usage flag
17. OperationalLogger: agent iteration logging

No live API keys required.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from scoutai.audit.metrics import (
    AgentConvergenceMetrics,
    MetricsCollector,
    ModelUsageMetrics,
    NodeMetrics,
    OperationalLogger,
    timed_node,
)
from scoutai.config import load_config

CONFIG_PATH = "config.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def metrics() -> MetricsCollector:
    return MetricsCollector()


# ── NodeMetrics ───────────────────────────────────────────────────────────────


class TestNodeMetrics:
    def test_records_call_count(self):
        """NodeMetrics must increment call_count on each record."""
        m = NodeMetrics(node_name="test")
        assert m.call_count == 0
        m.record(100.0)
        assert m.call_count == 1
        m.record(50.0)
        assert m.call_count == 2

    def test_tracks_latency_bounds(self):
        """NodeMetrics must track min, max, and total latency."""
        m = NodeMetrics(node_name="test")
        m.record(100.0)
        m.record(50.0)
        m.record(200.0)
        assert m.min_latency_ms == 50.0
        assert m.max_latency_ms == 200.0
        assert m.total_latency_ms == 350.0

    def test_avg_latency(self):
        """NodeMetrics must compute average latency."""
        m = NodeMetrics(node_name="test")
        m.record(100.0)
        m.record(200.0)
        assert m.avg_latency_ms == 150.0

    def test_avg_latency_zero_calls(self):
        """NodeMetrics avg_latency must be 0 when no calls recorded."""
        m = NodeMetrics(node_name="test")
        assert m.avg_latency_ms == 0.0

    def test_tracks_error_count(self):
        """NodeMetrics must increment error_count when error=True."""
        m = NodeMetrics(node_name="test")
        m.record(100.0, error=True)
        m.record(50.0, error=False)
        m.record(200.0, error=True)
        assert m.error_count == 2

    def test_to_dict(self):
        """NodeMetrics.to_dict must return a serializable dict."""
        m = NodeMetrics(node_name="test")
        m.record(100.0)
        d = m.to_dict()
        assert d["node_name"] == "test"
        assert d["call_count"] == 1
        assert d["avg_latency_ms"] == 100.0
        assert "last_called_at" in d


# ── ModelUsageMetrics ─────────────────────────────────────────────────────────


class TestModelUsageMetrics:
    def test_records_call_count(self):
        """ModelUsageMetrics must increment call_count."""
        m = ModelUsageMetrics(model_role="fast_structured")
        m.record(100.0)
        assert m.call_count == 1

    def test_tracks_latency(self):
        """ModelUsageMetrics must track total latency."""
        m = ModelUsageMetrics(model_role="fast_structured")
        m.record(100.0)
        m.record(200.0)
        assert m.total_latency_ms == 300.0

    def test_tracks_errors(self):
        """ModelUsageMetrics must track error count."""
        m = ModelUsageMetrics(model_role="high_context")
        m.record(100.0, error=True)
        m.record(50.0, error=False)
        assert m.error_count == 1

    def test_to_dict(self):
        """ModelUsageMetrics.to_dict must return a serializable dict."""
        m = ModelUsageMetrics(model_role="fast_structured")
        m.record(100.0)
        d = m.to_dict()
        assert d["model_role"] == "fast_structured"
        assert d["call_count"] == 1
        assert d["avg_latency_ms"] == 100.0


# ── AgentConvergenceMetrics ───────────────────────────────────────────────────


class TestAgentConvergenceMetrics:
    def test_tracks_tool_usage(self):
        """AgentConvergenceMetrics must track tool usage counts."""
        m = AgentConvergenceMetrics(candidate_id="c001")
        m.tool_usage["extract_evidence"] += 1
        m.tool_usage["assess_capabilities"] += 2
        assert m.tool_usage["extract_evidence"] == 1
        assert m.tool_usage["assess_capabilities"] == 2

    def test_to_dict(self):
        """AgentConvergenceMetrics.to_dict must return a serializable dict."""
        m = AgentConvergenceMetrics(
            candidate_id="c001",
            total_tool_calls=5,
            converged=True,
            force_finalized=False,
            tool_usage={"extract_evidence": 1, "assess_capabilities": 2},
            ask_candidate_used=True,
            final_recommendation="interview",
        )
        d = m.to_dict()
        assert d["candidate_id"] == "c001"
        assert d["total_tool_calls"] == 5
        assert d["converged"] is True
        assert d["force_finalized"] is False
        assert d["ask_candidate_used"] is True
        assert d["final_recommendation"] == "interview"
        assert d["tool_usage"]["extract_evidence"] == 1


# ── MetricsCollector ──────────────────────────────────────────────────────────


class TestMetricsCollector:
    def test_records_node_metrics(self, metrics):
        """MetricsCollector must record and retrieve node metrics."""
        metrics.record_node("screen_resume", 100.0)
        metrics.record_node("screen_resume", 200.0)
        node_m = metrics.get_node_metrics("screen_resume")
        assert node_m is not None
        assert node_m.call_count == 2
        assert node_m.total_latency_ms == 300.0

    def test_records_model_usage(self, metrics):
        """MetricsCollector must record and retrieve model usage."""
        metrics.record_model_usage("fast_structured", 500.0)
        metrics.record_model_usage("fast_structured", 300.0)
        usage = metrics.get_all_model_usage()
        assert "fast_structured" in usage
        assert usage["fast_structured"].call_count == 2
        assert usage["fast_structured"].total_latency_ms == 800.0

    def test_records_agent_convergence(self, metrics):
        """MetricsCollector must record agent convergence metrics."""
        metrics.record_agent_convergence(
            candidate_id="c001",
            total_tool_calls=5,
            converged=True,
            force_finalized=False,
            tool_usage={"extract_evidence": 2, "finalize_candidate": 1},
            ask_candidate_used=True,
            final_recommendation="interview",
        )
        convergence = metrics.get_all_agent_convergence()
        assert "c001" in convergence
        assert convergence["c001"].converged is True
        assert convergence["c001"].total_tool_calls == 5

    def test_summary_contains_all_sections(self, metrics):
        """MetricsCollector.summary must return a complete dict."""
        metrics.start_run()
        metrics.record_node("setup_role", 50.0)
        metrics.record_model_usage("fast_structured", 100.0)
        metrics.end_run()

        s = metrics.summary()
        assert "run_started_at" in s
        assert "run_ended_at" in s
        assert "node_metrics" in s
        assert "model_usage" in s
        assert "agent_convergence" in s
        assert "setup_role" in s["node_metrics"]
        assert "fast_structured" in s["model_usage"]

    def test_start_end_run_timestamps(self, metrics):
        """MetricsCollector must set run timestamps."""
        assert metrics.summary()["run_started_at"] is None
        assert metrics.summary()["run_ended_at"] is None
        metrics.start_run()
        assert metrics.summary()["run_started_at"] is not None
        metrics.end_run()
        assert metrics.summary()["run_ended_at"] is not None

    def test_multiple_nodes_independent(self, metrics):
        """MetricsCollector must keep node metrics independent."""
        metrics.record_node("setup_role", 50.0)
        metrics.record_node("screen_resume", 100.0)
        all_nodes = metrics.get_all_node_metrics()
        assert len(all_nodes) == 2
        assert all_nodes["setup_role"].call_count == 1
        assert all_nodes["screen_resume"].call_count == 1

    def test_node_error_recording(self, metrics):
        """MetricsCollector must record errors per node."""
        metrics.record_node("candidate_agent", 1000.0, error=True)
        metrics.record_node("candidate_agent", 500.0, error=False)
        node_m = metrics.get_node_metrics("candidate_agent")
        assert node_m is not None
        assert node_m.error_count == 1
        assert node_m.call_count == 2


# ── timed_node ────────────────────────────────────────────────────────────────


class TestTimedNode:
    def test_wraps_function_with_timing(self, metrics):
        """timed_node must wrap a function and record timing."""
        @timed_node("test_node", metrics)
        def my_func(state: dict) -> dict:
            time.sleep(0.01)  # 10ms
            return {"result": "done"}

        result = my_func({"input": "data"})
        assert result["result"] == "done"

        node_m = metrics.get_node_metrics("test_node")
        assert node_m is not None
        assert node_m.call_count == 1
        assert node_m.total_latency_ms >= 5.0  # at least 5ms

    def test_records_error_on_exception(self, metrics):
        """timed_node must record an error when the wrapped function raises."""
        @timed_node("failing_node", metrics)
        def failing_func(state: dict) -> dict:
            raise ValueError("Something went wrong")

        with pytest.raises(ValueError):
            failing_func({"input": "data"})

        node_m = metrics.get_node_metrics("failing_node")
        assert node_m is not None
        assert node_m.call_count == 1
        assert node_m.error_count == 1

    def test_preserves_function_name(self, metrics):
        """timed_node must preserve the wrapped function's __name__."""
        @timed_node("test", metrics)
        def my_special_func(state: dict) -> dict:
            return {"ok": True}

        assert my_special_func.__name__ == "my_special_func"


# ── OperationalLogger ─────────────────────────────────────────────────────────


class TestOperationalLogger:
    def test_logs_node_execution_when_enabled(self, config):
        """OperationalLogger must log node execution when include_node_timings is True."""
        config.logging.include_node_timings = True
        oplog = OperationalLogger(config)

        with patch("scoutai.audit.metrics.logger") as mock_logger:
            oplog.log_node_execution("screen_resume", 100.0, state_keys=["candidates"])

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        # Should contain the node name
        assert "screen_resume" in call_args

    def test_skips_node_logging_when_disabled(self, config):
        """OperationalLogger must skip node logging when include_node_timings is False."""
        config.logging.include_node_timings = False
        oplog = OperationalLogger(config)

        with patch("scoutai.audit.metrics.logger") as mock_logger:
            oplog.log_node_execution("screen_resume", 100.0)

        mock_logger.info.assert_not_called()

    def test_logs_model_call_when_enabled(self, config):
        """OperationalLogger must log model calls when include_model_usage is True."""
        config.logging.include_model_usage = True
        oplog = OperationalLogger(config)

        with patch("scoutai.audit.metrics.logger") as mock_logger:
            oplog.log_model_call("fast_structured", "gemini/gemini-2.0-flash", 500.0)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "fast_structured" in call_args

    def test_skips_model_logging_when_disabled(self, config):
        """OperationalLogger must skip model logging when include_model_usage is False."""
        config.logging.include_model_usage = False
        oplog = OperationalLogger(config)

        with patch("scoutai.audit.metrics.logger") as mock_logger:
            oplog.log_model_call("fast_structured", "gemini/gemini-2.0-flash", 500.0)

        mock_logger.info.assert_not_called()

    def test_logs_agent_iteration(self, config):
        """OperationalLogger must log agent iterations."""
        config.logging.include_node_timings = True
        oplog = OperationalLogger(config)

        with patch("scoutai.audit.metrics.logger") as mock_logger:
            oplog.log_agent_iteration("c001", 1, "extract_evidence", 200.0)

        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "c001" in call_args
        assert "extract_evidence" in call_args

    def test_json_format_emits_valid_json(self, config):
        """OperationalLogger must emit valid JSON when format is 'json'."""
        config.logging.include_node_timings = True
        config.logging.format = "json"
        oplog = OperationalLogger(config)

        with patch("scoutai.audit.metrics.logger") as mock_logger:
            oplog.log_node_execution("test_node", 100.0)

        call_args = mock_logger.info.call_args[0][0]
        parsed = json.loads(call_args)
        assert parsed["event"] == "node_execution"
        assert parsed["node"] == "test_node"
        assert parsed["latency_ms"] == 100.0

    def test_logs_error_in_node_execution(self, config):
        """OperationalLogger must include error info when provided."""
        config.logging.include_node_timings = True
        oplog = OperationalLogger(config)

        with patch("scoutai.audit.metrics.logger") as mock_logger:
            oplog.log_node_execution("test_node", 100.0, error="Something failed")

        call_args = mock_logger.info.call_args[0][0]
        assert "Something failed" in call_args
