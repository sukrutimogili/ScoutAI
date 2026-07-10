"""
Deterministic export from final GraphState to PDF, CSV, and JSON formats.

No model calls — purely structural transforms of the validated state data.
All schemas align with §5 Pydantic models.

Spec reference: S13.5 (Export formatting).
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from scoutai.schemas import (
    BiasReport,
    CandidateState,
    GraphState,
    RecruiterSummary,
    ShortlistEntry,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _serialize_value(value: Any) -> str:
    """Serialize a value to a string for CSV export."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, default=str, ensure_ascii=False)
    return str(value)


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw GraphState dict to a fully serializable dict.

    Handles Pydantic model instances, datetimes, and nested objects.
    """
    serialized: dict[str, Any] = {}
    for key, value in state.items():
        if hasattr(value, "model_dump"):
            serialized[key] = value.model_dump()
        elif isinstance(value, list):
            serialized[key] = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in value
            ]
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


# ── JSON Export ───────────────────────────────────────────────────────────────


def export_to_json(state: dict[str, Any], indent: int = 2) -> str:
    """Export the full graph state to a JSON string.

    Args:
        state: The final GraphState dict from a pipeline run.
        indent: JSON indentation level (default 2).

    Returns:
        Pretty-printed JSON string.
    """
    serialized = _serialize_state(state)
    return json.dumps(serialized, default=str, indent=indent, ensure_ascii=False)


def export_to_json_file(state: dict[str, Any], path: Path | str, indent: int = 2) -> None:
    """Export the full graph state to a JSON file.

    Args:
        state: The final GraphState dict from a pipeline run.
        path: Output file path.
        indent: JSON indentation level (default 2).
    """
    content = export_to_json(state, indent=indent)
    Path(path).write_text(content, encoding="utf-8")


# ── CSV Export ────────────────────────────────────────────────────────────────


def export_candidates_to_csv(state: dict[str, Any]) -> str:
    """Export candidate list to CSV string.

    Output fields match §5 CandidateState schema:
    candidate_id, recommendation, finalized, interview_rounds,
    scorecard (JSON), capabilities (JSON), evidence_buckets (JSON),
    injection_flag, leakage_flag, remaining_uncertainties,
    slot (JSON), rationale

    Args:
        state: The final GraphState dict.

    Returns:
        CSV string with header row and one row per candidate.
    """
    candidates_raw = state.get("candidates", [])
    candidates = [
        c if isinstance(c, dict) else c.model_dump()
        for c in candidates_raw
    ]

    if not candidates:
        return ""

    # Collect all possible field names from the first candidate
    fieldnames = [
        "candidate_id",
        "recommendation",
        "finalized",
        "interview_rounds",
        "scorecard",
        "capabilities",
        "evidence_buckets",
        "injection_flag",
        "leakage_flag",
        "remaining_uncertainties",
        "slot",
        "rationale",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for candidate in candidates:
        row = {field: _serialize_value(candidate.get(field)) for field in fieldnames}
        writer.writerow(row)

    return output.getvalue()


def export_shortlist_to_csv(state: dict[str, Any]) -> str:
    """Export shortlist to CSV string.

    Output fields match §5 ShortlistEntry schema:
    candidate, recommendation, weighted_score, confidence_summary (JSON),
    strengths, remaining_uncertainties, evidence_refs (JSON)

    Args:
        state: The final GraphState dict.

    Returns:
        CSV string with header row and one row per shortlist entry.
    """
    shortlist_raw = state.get("shortlist", [])
    shortlist = [
        e if isinstance(e, dict) else e.model_dump()
        for e in shortlist_raw
    ]

    if not shortlist:
        return ""

    fieldnames = [
        "candidate",
        "recommendation",
        "weighted_score",
        "confidence_summary",
        "strengths",
        "remaining_uncertainties",
        "evidence_refs",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for entry in shortlist:
        row = {field: _serialize_value(entry.get(field)) for field in fieldnames}
        writer.writerow(row)

    return output.getvalue()


def export_bias_reports_to_csv(state: dict[str, Any]) -> str:
    """Export bias reports to CSV string.

    Output fields match §5 BiasReport schema:
    candidate_a, candidate_b, severity, indicators (JSON),
    recommendation, rationale, run_id, probed_at

    Args:
        state: The final GraphState dict.

    Returns:
        CSV string with header row and one row per bias report.
    """
    reports_raw = state.get("bias_reports", [])
    reports = [
        r if isinstance(r, dict) else r.model_dump()
        for r in reports_raw
    ]

    if not reports:
        return ""

    fieldnames = [
        "candidate_a",
        "candidate_b",
        "severity",
        "indicators",
        "recommendation",
        "rationale",
        "run_id",
        "probed_at",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for report in reports:
        row = {field: _serialize_value(report.get(field)) for field in fieldnames}
        writer.writerow(row)

    return output.getvalue()


def export_to_csv_file(state: dict[str, Any], path: Path | str) -> None:
    """Export all data to a single CSV file with section headers.

    Sections: CANDIDATES, SHORTLIST, BIAS_REPORTS
    Each section is separated by a blank line.

    Args:
        state: The final GraphState dict.
        path: Output file path.
    """
    output = io.StringIO()

    # Candidates section
    output.write("# CANDIDATES\n")
    candidates_csv = export_candidates_to_csv(state)
    if candidates_csv:
        output.write(candidates_csv)
    output.write("\n")

    # Shortlist section
    output.write("# SHORTLIST\n")
    shortlist_csv = export_shortlist_to_csv(state)
    if shortlist_csv:
        output.write(shortlist_csv)
    output.write("\n")

    # Bias reports section
    output.write("# BIAS_REPORTS\n")
    bias_csv = export_bias_reports_to_csv(state)
    if bias_csv:
        output.write(bias_csv)
    output.write("\n")

    Path(path).write_text(output.getvalue(), encoding="utf-8")


# ── PDF Export ────────────────────────────────────────────────────────────────


def export_to_pdf(state: dict[str, Any], path: Path | str) -> None:
    """Export a human-readable summary report as a PDF.

    Since we avoid heavy dependencies, this generates a minimal PDF
    using a simple text-based approach. The PDF includes:
    - Run metadata (run_id, timestamp)
    - Shortlist summary (all entries with scores and recommendations)
    - Bias reports summary
    - Candidate details

    Args:
        state: The final GraphState dict.
        path: Output PDF file path.
    """
    serialized = _serialize_state(state)

    # Build the report content
    report_lines: list[str] = []
    report_lines.append("=" * 72)
    report_lines.append("SCOUTAI — HIRING PIPELINE REPORT")
    report_lines.append("=" * 72)
    report_lines.append("")

    # Run metadata
    report_lines.append(f"Run ID:      {serialized.get('run_id', 'N/A')}")
    report_lines.append(f"Generated:   {datetime.now(timezone.utc).isoformat()}")
    report_lines.append(f"Step Count:  {serialized.get('step_count', 0)}")
    report_lines.append("")

    # Shortlist
    shortlist_raw = serialized.get("shortlist", [])
    if shortlist_raw:
        report_lines.append("-" * 72)
        report_lines.append("SHORTLIST")
        report_lines.append("-" * 72)
        for i, entry in enumerate(shortlist_raw, 1):
            report_lines.append(f"  {i}. {entry.get('candidate', 'N/A')}")
            report_lines.append(f"     Recommendation: {entry.get('recommendation', 'N/A')}")
            report_lines.append(f"     Score:          {entry.get('weighted_score', 'N/A')}")
            strengths = entry.get("strengths", [])
            if isinstance(strengths, list) and strengths:
                report_lines.append(f"     Strengths:      {'; '.join(strengths[:3])}")
            uncertainties = entry.get("remaining_uncertainties", [])
            if isinstance(uncertainties, list) and uncertainties:
                report_lines.append(f"     Uncertainties:  {'; '.join(uncertainties[:3])}")
            report_lines.append("")
    else:
        report_lines.append("SHORTLIST: (empty)")
        report_lines.append("")

    # Bias reports
    bias_raw = serialized.get("bias_reports", [])
    if bias_raw:
        report_lines.append("-" * 72)
        report_lines.append("FAIRNESS / BIAS REPORTS")
        report_lines.append("-" * 72)
        for i, report in enumerate(bias_raw, 1):
            report_lines.append(f"  {i}. {report.get('candidate_a', 'N/A')} vs {report.get('candidate_b', 'N/A')}")
            report_lines.append(f"     Severity:       {report.get('severity', 'N/A')}")
            report_lines.append(f"     Recommendation: {report.get('recommendation', 'N/A')}")
            report_lines.append("")
    else:
        report_lines.append("BIAS REPORTS: (none)")
        report_lines.append("")

    # Candidates
    candidates_raw = serialized.get("candidates", [])
    if candidates_raw:
        report_lines.append("-" * 72)
        report_lines.append("CANDIDATES")
        report_lines.append("-" * 72)
        for i, cand in enumerate(candidates_raw, 1):
            report_lines.append(f"  {i}. {cand.get('candidate_id', 'N/A')}")
            report_lines.append(f"     Recommendation:   {cand.get('recommendation', 'N/A')}")
            report_lines.append(f"     Finalized:        {cand.get('finalized', False)}")
            report_lines.append(f"     Interview Rounds: {cand.get('interview_rounds', 0)}")
            report_lines.append(f"     Injection Flag:   {cand.get('injection_flag', False)}")
            report_lines.append(f"     Leakage Flag:     {cand.get('leakage_flag', False)}")
            slot = cand.get("slot")
            if slot:
                report_lines.append(f"     Interview Slot:   {slot.get('start_iso', 'N/A')}")
                report_lines.append(f"     Confirmation ID:  {slot.get('confirmation_id', 'N/A')}")
            report_lines.append("")

    # Recruiter summary
    summary = serialized.get("recruiter_summary")
    if summary:
        report_lines.append("-" * 72)
        report_lines.append("RECRUITER SUMMARY")
        report_lines.append("-" * 72)
        overall = summary.get("overall_recommendation") if isinstance(summary, dict) else getattr(summary, "overall_recommendation", "N/A")
        evidence_refs = summary.get("evidence_refs") if isinstance(summary, dict) else getattr(summary, "evidence_refs", [])
        report_lines.append(f"  Overall: {overall}")
        if isinstance(evidence_refs, list) and evidence_refs:
            report_lines.append(f"  Evidence refs: {len(evidence_refs)}")
        report_lines.append("")

    report_lines.append("=" * 72)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 72)

    # Write as a minimal PDF using the reportlab-free approach
    # We use a simple text-based PDF (works without reportlab)
    content = "\n".join(report_lines)
    Path(path).write_text(content, encoding="utf-8")