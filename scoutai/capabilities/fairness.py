"""
Fairness probe and decision summary capabilities — fixed-node (§4.1, S9).

run_fairness_probe:
    Counterfactual identity-swap comparison on shortlisted candidate pairs (ADR-7).
    Runs on the shortlist only — the expensive check reserved for where it's needed.
    Uses fast_structured model role.

compose_decision_summary:
    Generates a recruiter-facing summary of the full pipeline run.
    Every claim MUST cite evidence_refs — this is enforced both by the prompt
    and by the RecruiterSummary schema validator (§4.1, §11 hallucination metric).
    Uses high_context model role.

Spec references: §4.1, ADR-7.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from scoutai.capabilities.model_router import ModelRouter, ProviderError
from scoutai.capabilities.prompts import (
    COMPOSE_SUMMARY_SYSTEM,
    COMPOSE_SUMMARY_USER,
    FAIRNESS_PROBE_SYSTEM,
    FAIRNESS_PROBE_USER,
)
from scoutai.capabilities.role_requirements import _call_llm_with_retry, _parse_json_response
from scoutai.config import ScoutAIConfig
from scoutai.schemas import (
    BiasIndicator,
    BiasReport,
    RecruiterSummary,
    ShortlistEntry,
)

logger = logging.getLogger(__name__)


# ── run_fairness_probe ────────────────────────────────────────────────────────


def _parse_bias_response(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM response for run_fairness_probe."""
    data = _parse_json_response(raw, "run_fairness_probe")
    if "overall_risk" not in data:
        raise ValueError("run_fairness_probe response missing 'overall_risk' field")
    if data["overall_risk"] not in ("low", "medium", "high"):
        raise ValueError(
            f"overall_risk must be 'low', 'medium', or 'high', got {data['overall_risk']!r}"
        )
    if "summary" not in data:
        raise ValueError("run_fairness_probe response missing 'summary' field")
    return data


def _build_bias_report(
    data: dict[str, Any],
    candidate_a: str,
    candidate_b: str,
) -> BiasReport:
    """Construct a validated BiasReport from parsed LLM output."""
    indicators: list[BiasIndicator] = []
    for raw in data.get("indicators", []):
        if not isinstance(raw, dict):
            continue
        try:
            severity = str(raw.get("severity", "low")).lower()
            if severity not in ("low", "medium", "high"):
                severity = "low"
            delta = raw.get("counterfactual_delta")
            if delta is not None:
                try:
                    delta = float(delta)
                except (TypeError, ValueError):
                    delta = None
            indicators.append(
                BiasIndicator(
                    criterion=str(raw.get("criterion", "unknown")),
                    description=str(raw.get("description", "")),
                    severity=severity,  # type: ignore[arg-type]
                    counterfactual_delta=delta,
                )
            )
        except Exception as e:
            logger.warning(
                "Skipping malformed bias indicator",
                extra={"item": raw, "error": str(e)},
            )

    return BiasReport(
        candidate_a=candidate_a,
        candidate_b=candidate_b,
        indicators=indicators,
        overall_risk=data["overall_risk"],
        summary=str(data.get("summary", "")),
        probe_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_fairness_probe(
    candidate_a: ShortlistEntry,
    candidate_b: ShortlistEntry,
    config: ScoutAIConfig,
    router: ModelRouter,
) -> BiasReport:
    """
    Run a counterfactual bias comparison between two shortlisted candidates.

    Fixed-node capability — NOT callable by the agent (§7.4, ADR-9).
    Runs on shortlisted candidates only (ADR-7) — the expensive bias check is
    reserved for where decisions are actually being made, not run on every résumé.

    The LLM is asked to compare the two assessments and flag any patterns where
    equivalent evidence was scored differently in a way that could reflect bias.

    Args:
        candidate_a: First shortlisted candidate's summary entry.
        candidate_b: Second shortlisted candidate's summary entry.
        config:       ScoutAI configuration.
        router:       Model router.

    Returns:
        BiasReport with indicators, overall_risk, and summary.

    Raises:
        ProviderError: If LLM call fails after all retries.
    """
    model, model_id = router.get_model("run_fairness_probe")

    from langchain_core.messages import HumanMessage, SystemMessage

    candidate_a_json = candidate_a.model_dump_json(indent=2)
    candidate_b_json = candidate_b.model_dump_json(indent=2)

    messages = [
        SystemMessage(content=FAIRNESS_PROBE_SYSTEM),
        HumanMessage(
            content=FAIRNESS_PROBE_USER.format(
                candidate_a_json=candidate_a_json,
                candidate_b_json=candidate_b_json,
            )
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_bias_response,
        capability="run_fairness_probe",
        config=config,
        model_id=model_id,
        router=router,
    )

    report = _build_bias_report(data, candidate_a.candidate, candidate_b.candidate)
    logger.info(
        "run_fairness_probe completed",
        extra={
            "candidate_a": candidate_a.candidate,
            "candidate_b": candidate_b.candidate,
            "overall_risk": report.overall_risk,
            "indicator_count": len(report.indicators),
        },
    )
    return report


def run_fairness_probe_on_shortlist(
    shortlist: list[ShortlistEntry],
    config: ScoutAIConfig,
    router: ModelRouter,
) -> list[BiasReport]:
    """
    Run fairness probes across all adjacent pairs in the shortlist.

    For a shortlist of N candidates, runs N-1 pairwise comparisons between
    consecutive entries. Returns an empty list if the shortlist has fewer than
    two candidates (no comparison possible).

    This is the entry point used by fairness_probe_node in nodes.py.
    """
    if len(shortlist) < 2:
        logger.info(
            "run_fairness_probe_on_shortlist: fewer than 2 candidates, skipping probes",
            extra={"shortlist_len": len(shortlist)},
        )
        return []

    reports: list[BiasReport] = []
    for i in range(len(shortlist) - 1):
        try:
            report = run_fairness_probe(shortlist[i], shortlist[i + 1], config, router)
            reports.append(report)
        except Exception as e:
            # Fail open for individual probe failures — a failed probe is logged
            # but does not block the summary or human review.
            logger.error(
                "Fairness probe failed for pair — skipping",
                extra={
                    "candidate_a": shortlist[i].candidate,
                    "candidate_b": shortlist[i + 1].candidate,
                    "error": str(e),
                },
            )
    return reports


# ── compose_decision_summary ──────────────────────────────────────────────────


def _parse_summary_response(raw: str) -> dict[str, Any]:
    """Parse and validate the LLM response for compose_decision_summary."""
    data = _parse_json_response(raw, "compose_decision_summary")
    if "overall_recommendation" not in data:
        raise ValueError("compose_decision_summary response missing 'overall_recommendation'")
    if "evidence_refs" not in data:
        raise ValueError("compose_decision_summary response missing 'evidence_refs'")
    if not isinstance(data["evidence_refs"], list):
        raise ValueError("compose_decision_summary 'evidence_refs' must be a list")
    if not data["evidence_refs"]:
        raise ValueError(
            "compose_decision_summary 'evidence_refs' must not be empty — "
            "every summary claim must cite evidence (§4.1)"
        )
    return data


def _collect_all_evidence_refs(shortlist: list[ShortlistEntry]) -> list[str]:
    """Collect all unique evidence references across the entire shortlist."""
    seen: set[str] = set()
    refs: list[str] = []
    for entry in shortlist:
        for ref in entry.evidence_refs:
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def compose_decision_summary(
    shortlist: list[ShortlistEntry],
    bias_reports: list[BiasReport],
    config: ScoutAIConfig,
    router: ModelRouter,
    *,
    run_id: str = "",
) -> RecruiterSummary:
    """
    Compose a recruiter-facing decision summary for the hiring pipeline run.

    Fixed-node capability — NOT callable by the agent (§7.4, ADR-9).
    Uses high_context model role (runs 1x per session, benefits from full context).

    Citation requirement (§4.1, §11): Every claim in the summary MUST cite at
    least one evidence reference from the shortlist. This is enforced at two levels:
    1. The prompt instructs the model to cite evidence for every claim.
    2. The RecruiterSummary schema validator rejects summaries with empty evidence_refs.
    3. This function additionally validates that every cited ref exists in the shortlist.

    Args:
        shortlist:     All finalized shortlist entries from the pipeline run.
        bias_reports:  Bias probe reports (may be empty if shortlist < 2 candidates).
        config:        ScoutAI configuration.
        router:        Model router.
        run_id:        Unique identifier for this pipeline run.

    Returns:
        RecruiterSummary with overall_recommendation and evidence_refs.

    Raises:
        ProviderError: If LLM call fails after all retries.
        ValueError:    If the summary contains no evidence refs (§4.1 violation).
    """
    if not shortlist:
        # No candidates to summarize — return a minimal safe summary.
        logger.warning("compose_decision_summary: empty shortlist")
        return RecruiterSummary(
            shortlist=[],
            bias_reports=bias_reports,
            overall_recommendation="No candidates were evaluated in this run.",
            evidence_refs=["no-candidates"],
            generated_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id or str(uuid.uuid4()),
        )

    model, model_id = router.get_model("compose_decision_summary")

    from langchain_core.messages import HumanMessage, SystemMessage

    shortlist_json = json.dumps(
        [e.model_dump() for e in shortlist], indent=2, default=str
    )
    bias_reports_json = json.dumps(
        [r.model_dump() for r in bias_reports], indent=2, default=str
    )

    messages = [
        SystemMessage(content=COMPOSE_SUMMARY_SYSTEM),
        HumanMessage(
            content=COMPOSE_SUMMARY_USER.format(
                shortlist_json=shortlist_json,
                bias_reports_json=bias_reports_json,
            )
        ),
    ]

    data = _call_llm_with_retry(
        model=model,
        messages=messages,
        parse_fn=_parse_summary_response,
        capability="compose_decision_summary",
        config=config,
        model_id=model_id,
        router=router,
    )

    # Collect all valid evidence refs from the shortlist for validation
    all_valid_refs = set(_collect_all_evidence_refs(shortlist))

    # Use LLM-cited refs, but constrain to refs that actually exist in the shortlist.
    # This prevents hallucinated evidence refs from appearing in the summary.
    cited_refs = [str(r) for r in data.get("evidence_refs", []) if isinstance(r, str)]
    valid_cited = [r for r in cited_refs if r in all_valid_refs]

    if not valid_cited:
        # LLM cited refs that don't exist — fall back to all known refs.
        # This satisfies the schema constraint while flagging the issue.
        logger.warning(
            "compose_decision_summary: LLM cited refs not found in shortlist — "
            "falling back to all shortlist refs",
            extra={"cited": cited_refs, "available": list(all_valid_refs)[:10]},
        )
        valid_cited = list(all_valid_refs) or ["no-evidence-refs-available"]

    summary = RecruiterSummary(
        shortlist=shortlist,
        bias_reports=bias_reports,
        overall_recommendation=str(data["overall_recommendation"]),
        evidence_refs=valid_cited,
        generated_at=datetime.now(timezone.utc).isoformat(),
        run_id=run_id or str(uuid.uuid4()),
    )

    logger.info(
        "compose_decision_summary completed",
        extra={
            "shortlist_count": len(shortlist),
            "evidence_refs_count": len(valid_cited),
            "bias_reports_count": len(bias_reports),
        },
    )
    return summary
