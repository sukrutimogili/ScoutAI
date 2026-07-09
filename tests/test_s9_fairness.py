"""
S9 tests: run_fairness_probe, compose_decision_summary, and node wiring.

Key test cases:
1.  run_fairness_probe: happy path returns a BiasReport with correct candidate ids
2.  run_fairness_probe: LLM response parsed — indicators built correctly
3.  run_fairness_probe: malformed indicator skipped, valid ones kept
4.  run_fairness_probe: invalid overall_risk raises on parse
5.  run_fairness_probe: fail closed on LLM error (raises ProviderError)
6.  run_fairness_probe_on_shortlist: < 2 candidates returns empty list
7.  run_fairness_probe_on_shortlist: 3 candidates → 2 pairwise probes
8.  run_fairness_probe_on_shortlist: individual probe failure is logged, not raised
9.  compose_decision_summary: happy path returns RecruiterSummary with evidence_refs
10. compose_decision_summary: cited refs not in shortlist fall back to all known refs
11. compose_decision_summary: empty shortlist returns safe minimal summary
12. compose_decision_summary: evidence_refs must be non-empty (schema validation)
13. compose_decision_summary: every cited ref resolves to a real shortlist ref
14. fairness_probe_node: calls run_fairness_probe_on_shortlist, writes bias_reports
15. compose_summary_node: calls compose_decision_summary, writes recruiter_summary
16. compose_summary_node: recruiter_summary.evidence_refs is non-empty

No live API keys required — all LLM calls mocked.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scoutai.capabilities.fairness import (
    _build_bias_report,
    _parse_bias_response,
    _parse_summary_response,
    compose_decision_summary,
    run_fairness_probe,
    run_fairness_probe_on_shortlist,
)
from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.config import load_config
from scoutai.graph.nodes import compose_summary_node, fairness_probe_node
from scoutai.schemas import (
    BiasIndicator,
    BiasReport,
    CapabilityAssessment,
    RecruiterSummary,
    ShortlistEntry,
)

CONFIG_PATH = "config.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def router(config):
    return ModelRouter(config)


def _make_entry(candidate_id: str, refs: list[str] | None = None) -> ShortlistEntry:
    return ShortlistEntry(
        candidate=candidate_id,
        recommendation="interview",
        weighted_score=72.0,
        evidence_refs=refs or [f"{candidate_id}:Resume:Skills"],
    )


def _mock_llm_response(content: str):
    """Build a mock LLM response object."""
    msg = MagicMock()
    msg.content = content
    return msg


# ── _parse_bias_response ──────────────────────────────────────────────────────


class TestParseBiasResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "indicators": [{"criterion": "Python", "description": "gap", "severity": "low"}],
            "overall_risk": "low",
            "summary": "No significant bias detected.",
        })
        data = _parse_bias_response(raw)
        assert data["overall_risk"] == "low"
        assert data["summary"] == "No significant bias detected."

    def test_missing_overall_risk_raises(self):
        raw = json.dumps({"summary": "ok"})
        with pytest.raises(ValueError, match="overall_risk"):
            _parse_bias_response(raw)

    def test_invalid_overall_risk_raises(self):
        raw = json.dumps({"overall_risk": "critical", "summary": "x"})
        with pytest.raises(ValueError, match="overall_risk"):
            _parse_bias_response(raw)

    def test_missing_summary_raises(self):
        raw = json.dumps({"overall_risk": "low"})
        with pytest.raises(ValueError, match="summary"):
            _parse_bias_response(raw)


# ── _build_bias_report ────────────────────────────────────────────────────────


class TestBuildBiasReport:
    def test_builds_correctly(self):
        data = {
            "indicators": [
                {"criterion": "Leadership", "description": "Score gap", "severity": "medium",
                 "counterfactual_delta": 15.0},
            ],
            "overall_risk": "medium",
            "summary": "Moderate risk detected.",
        }
        report = _build_bias_report(data, "c001", "c002")
        assert report.candidate_a == "c001"
        assert report.candidate_b == "c002"
        assert report.overall_risk == "medium"
        assert len(report.indicators) == 1
        assert report.indicators[0].counterfactual_delta == 15.0

    def test_malformed_indicator_skipped(self):
        data = {
            "indicators": ["not-a-dict", {"criterion": "X", "description": "ok", "severity": "low"}],
            "overall_risk": "low",
            "summary": "ok",
        }
        report = _build_bias_report(data, "a", "b")
        assert len(report.indicators) == 1

    def test_invalid_severity_defaults_to_low(self):
        data = {
            "indicators": [{"criterion": "X", "description": "ok", "severity": "EXTREME"}],
            "overall_risk": "low",
            "summary": "ok",
        }
        report = _build_bias_report(data, "a", "b")
        assert report.indicators[0].severity == "low"

    def test_null_counterfactual_delta_allowed(self):
        data = {
            "indicators": [{"criterion": "X", "description": "ok", "severity": "low",
                            "counterfactual_delta": None}],
            "overall_risk": "low",
            "summary": "ok",
        }
        report = _build_bias_report(data, "a", "b")
        assert report.indicators[0].counterfactual_delta is None


# ── run_fairness_probe ────────────────────────────────────────────────────────


class TestRunFairnessProbe:
    def _llm_response(self) -> str:
        return json.dumps({
            "indicators": [
                {"criterion": "Leadership", "description": "Score difference",
                 "severity": "medium", "counterfactual_delta": 10.0}
            ],
            "overall_risk": "medium",
            "summary": "Moderate risk.",
        })

    def test_happy_path_returns_bias_report(self, config, router):
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(self._llm_response())

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            report = run_fairness_probe(
                _make_entry("c001"), _make_entry("c002"), config, router
            )

        assert isinstance(report, BiasReport)
        assert report.candidate_a == "c001"
        assert report.candidate_b == "c002"
        assert report.overall_risk == "medium"

    def test_probe_timestamp_is_set(self, config, router):
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(self._llm_response())

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            report = run_fairness_probe(
                _make_entry("c001"), _make_entry("c002"), config, router
            )

        assert report.probe_timestamp  # non-empty
        # Should be parseable as ISO 8601
        datetime.fromisoformat(report.probe_timestamp)

    def test_fail_closed_on_llm_error(self, config, router):
        mock_model = MagicMock()
        mock_model.invoke.side_effect = Exception("provider down")

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            with pytest.raises(Exception):
                run_fairness_probe(
                    _make_entry("c001"), _make_entry("c002"), config, router
                )


# ── run_fairness_probe_on_shortlist ───────────────────────────────────────────


class TestRunFairnessProbeOnShortlist:
    def test_fewer_than_two_returns_empty(self, config, router):
        result = run_fairness_probe_on_shortlist([_make_entry("c001")], config, router)
        assert result == []

    def test_empty_shortlist_returns_empty(self, config, router):
        result = run_fairness_probe_on_shortlist([], config, router)
        assert result == []

    def test_three_candidates_produces_two_reports(self, config, router):
        shortlist = [_make_entry("c001"), _make_entry("c002"), _make_entry("c003")]
        probe_response = json.dumps({
            "indicators": [],
            "overall_risk": "low",
            "summary": "No bias detected.",
        })
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(probe_response)

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            reports = run_fairness_probe_on_shortlist(shortlist, config, router)

        assert len(reports) == 2
        assert reports[0].candidate_a == "c001"
        assert reports[0].candidate_b == "c002"
        assert reports[1].candidate_a == "c002"
        assert reports[1].candidate_b == "c003"

    def test_individual_probe_failure_is_skipped_not_raised(self, config, router):
        """A single probe failure must not abort the entire shortlist run.

        The side_effect must fail ALL retry attempts for the first pair
        (config.retry.max_attempts times) so _call_llm_with_retry actually
        exhausts its retries and raises ProviderError.
        """
        shortlist = [_make_entry("c001"), _make_entry("c002"), _make_entry("c003")]
        call_count = [0]
        max_attempts = config.retry.max_attempts  # 3

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= max_attempts:
                raise Exception("transient error on first pair")
            return _mock_llm_response(json.dumps({
                "indicators": [], "overall_risk": "low", "summary": "ok",
            }))

        mock_model = MagicMock()
        mock_model.invoke.side_effect = side_effect

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            # Should not raise — first pair failure is swallowed
            reports = run_fairness_probe_on_shortlist(shortlist, config, router)

        # Second pair should succeed
        assert len(reports) == 1
        assert reports[0].candidate_a == "c002"


# ── _parse_summary_response ───────────────────────────────────────────────────


class TestParseSummaryResponse:
    def test_valid_response(self):
        raw = json.dumps({
            "overall_recommendation": "Recommend c001 for interview.",
            "evidence_refs": ["c001:Resume:Skills"],
        })
        data = _parse_summary_response(raw)
        assert data["overall_recommendation"]
        assert data["evidence_refs"] == ["c001:Resume:Skills"]

    def test_missing_overall_recommendation_raises(self):
        raw = json.dumps({"evidence_refs": ["x"]})
        with pytest.raises(ValueError, match="overall_recommendation"):
            _parse_summary_response(raw)

    def test_missing_evidence_refs_raises(self):
        raw = json.dumps({"overall_recommendation": "ok"})
        with pytest.raises(ValueError, match="evidence_refs"):
            _parse_summary_response(raw)

    def test_empty_evidence_refs_raises(self):
        """Empty refs must be rejected at parse time — §4.1 citation requirement."""
        raw = json.dumps({"overall_recommendation": "ok", "evidence_refs": []})
        with pytest.raises(ValueError, match="evidence_refs"):
            _parse_summary_response(raw)


# ── compose_decision_summary ──────────────────────────────────────────────────


class TestComposeDecisionSummary:
    def _summary_response(self, refs: list[str]) -> str:
        return json.dumps({
            "overall_recommendation": "Recommend c001 for interview.",
            "evidence_refs": refs,
        })

    def test_happy_path(self, config, router):
        shortlist = [_make_entry("c001", ["c001:Resume:Skills"])]
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(
            self._summary_response(["c001:Resume:Skills"])
        )

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            summary = compose_decision_summary(shortlist, [], config, router, run_id="r1")

        assert isinstance(summary, RecruiterSummary)
        assert summary.run_id == "r1"
        assert "c001:Resume:Skills" in summary.evidence_refs
        assert len(summary.shortlist) == 1

    def test_every_cited_ref_resolves_to_shortlist(self, config, router):
        """Every evidence_ref in the summary must exist in the shortlist (§4.1)."""
        shortlist = [_make_entry("c001", ["c001:Resume:Projects"])]
        mock_model = MagicMock()
        # LLM cites a real ref and a hallucinated one
        mock_model.invoke.return_value = _mock_llm_response(
            self._summary_response(["c001:Resume:Projects", "hallucinated:ref"])
        )

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            summary = compose_decision_summary(shortlist, [], config, router)

        # Hallucinated ref must be filtered out
        assert "hallucinated:ref" not in summary.evidence_refs
        assert "c001:Resume:Projects" in summary.evidence_refs

    def test_all_cited_refs_hallucinated_falls_back_to_shortlist_refs(self, config, router):
        """If all LLM-cited refs are hallucinated, fall back to all shortlist refs."""
        shortlist = [_make_entry("c001", ["c001:Resume:Skills"])]
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(
            self._summary_response(["completely:hallucinated"])
        )

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            summary = compose_decision_summary(shortlist, [], config, router)

        # Falls back to real shortlist refs — schema constraint satisfied
        assert "c001:Resume:Skills" in summary.evidence_refs

    def test_empty_shortlist_returns_safe_summary(self, config, router):
        """Empty shortlist must not call LLM — returns a safe minimal summary."""
        call_log = []
        mock_model = MagicMock()
        mock_model.invoke.side_effect = lambda *a, **kw: call_log.append(1)

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            summary = compose_decision_summary([], [], config, router)

        assert call_log == [], "LLM must not be called for empty shortlist"
        assert summary.evidence_refs  # non-empty (safe fallback value)
        assert summary.shortlist == []

    def test_evidence_refs_non_empty_schema_enforced(self):
        """RecruiterSummary.evidence_refs=[] must be rejected by schema."""
        with pytest.raises(Exception):
            RecruiterSummary(
                shortlist=[],
                bias_reports=[],
                overall_recommendation="ok",
                evidence_refs=[],   # empty — must fail validation
                generated_at=datetime.now(timezone.utc).isoformat(),
                run_id="test",
            )

    def test_bias_reports_included_in_summary(self, config, router):
        shortlist = [_make_entry("c001", ["c001:Resume:Skills"])]
        bias_report = BiasReport(
            candidate_a="c001", candidate_b="c002",
            overall_risk="low", summary="ok",
            probe_timestamp=datetime.now(timezone.utc).isoformat(),
        )
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(
            self._summary_response(["c001:Resume:Skills"])
        )

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            summary = compose_decision_summary(shortlist, [bias_report], config, router)

        assert len(summary.bias_reports) == 1
        assert summary.bias_reports[0].overall_risk == "low"


# ── Node wiring ───────────────────────────────────────────────────────────────


class TestFairnessProbeNode:
    def _state(self, shortlist: list) -> dict[str, Any]:
        return {"shortlist": shortlist, "step_count": 0}

    def test_writes_bias_reports_to_state(self, config, router):
        shortlist = [
            _make_entry("c001").model_dump(),
            _make_entry("c002").model_dump(),
        ]
        probe_response = json.dumps({
            "indicators": [], "overall_risk": "low", "summary": "no bias",
        })
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(probe_response)

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            result = fairness_probe_node(self._state(shortlist), config=config, router=router)

        assert "bias_reports" in result
        assert len(result["bias_reports"]) == 1

    def test_empty_shortlist_writes_empty_bias_reports(self, config, router):
        result = fairness_probe_node(self._state([]), config=config, router=router)
        assert result["bias_reports"] == []

    def test_increments_step_count(self, config, router):
        result = fairness_probe_node({"shortlist": [], "step_count": 3}, config=config, router=router)
        assert result["step_count"] == 4

    def test_single_candidate_produces_no_probes(self, config, router):
        shortlist = [_make_entry("c001").model_dump()]
        result = fairness_probe_node(self._state(shortlist), config=config, router=router)
        assert result["bias_reports"] == []


class TestComposeSummaryNode:
    def _state(self, shortlist: list, bias_reports: list | None = None) -> dict[str, Any]:
        return {
            "shortlist": shortlist,
            "bias_reports": bias_reports or [],
            "run_id": "run-test",
            "step_count": 0,
        }

    def test_writes_recruiter_summary_to_state(self, config, router):
        shortlist = [_make_entry("c001", ["c001:Resume:Skills"]).model_dump()]
        summary_response = json.dumps({
            "overall_recommendation": "Recommend c001.",
            "evidence_refs": ["c001:Resume:Skills"],
        })
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(summary_response)

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            result = compose_summary_node(self._state(shortlist), config=config, router=router)

        assert "recruiter_summary" in result
        assert isinstance(result["recruiter_summary"], RecruiterSummary)

    def test_evidence_refs_non_empty_in_output(self, config, router):
        """Summary written to state must always have non-empty evidence_refs (§4.1)."""
        shortlist = [_make_entry("c001", ["c001:Resume:Skills"]).model_dump()]
        summary_response = json.dumps({
            "overall_recommendation": "Recommend c001.",
            "evidence_refs": ["c001:Resume:Skills"],
        })
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(summary_response)

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            result = compose_summary_node(self._state(shortlist), config=config, router=router)

        summary = result["recruiter_summary"]
        assert summary.evidence_refs, "recruiter_summary.evidence_refs must not be empty"

    def test_increments_step_count(self, config, router):
        shortlist = [_make_entry("c001", ["c001:Resume:Skills"]).model_dump()]
        summary_response = json.dumps({
            "overall_recommendation": "ok",
            "evidence_refs": ["c001:Resume:Skills"],
        })
        mock_model = MagicMock()
        mock_model.invoke.return_value = _mock_llm_response(summary_response)

        with patch.object(router, "_get_or_create_client", return_value=mock_model):
            result = compose_summary_node(
                self._state(shortlist) | {"step_count": 5},
                config=config, router=router
            )

        assert result["step_count"] == 6

    def test_empty_shortlist_handled_safely(self, config, router):
        """Empty shortlist must not crash — returns minimal safe summary."""
        result = compose_summary_node(self._state([]), config=config, router=router)
        summary = result["recruiter_summary"]
        assert isinstance(summary, RecruiterSummary)
        assert summary.evidence_refs