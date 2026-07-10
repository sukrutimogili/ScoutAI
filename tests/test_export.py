"""
Tests for deterministic PDF/CSV/JSON export from final GraphState.

No model calls — purely structural transforms.
All three formats must produce non-empty output.
CSV/JSON field names must match §5 Pydantic schemas.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Generator

import pytest

from scoutai.graph.export import (
    export_bias_reports_to_csv,
    export_candidates_to_csv,
    export_shortlist_to_csv,
    export_to_csv_file,
    export_to_json,
    export_to_json_file,
    export_to_pdf,
)
from scoutai.schemas import (
    BiasReport,
    CandidateState,
    ShortlistEntry,
)


# ── Fixture: a realistic final GraphState ─────────────────────────────────────


@pytest.fixture
def final_state() -> dict[str, Any]:
    """A realistic final GraphState with candidates, shortlist, and bias reports."""
    return {
        "run_id": "test-run-001",
        "step_count": 42,
        "current_idx": 3,
        "jd": "Software Engineer position requiring Python, AWS, and system design.",
        "candidates": [
            {
                "candidate_id": "c001",
                "recommendation": "strong_interview",
                "finalized": True,
                "interview_rounds": 0,
                "scorecard": {"python": 90, "aws": 85, "system_design": 80},
                "capabilities": {"python": {"score": 90, "confidence": "high"}},
                "evidence_buckets": {"python": ["5 years Python experience"]},
                "injection_flag": False,
                "leakage_flag": False,
                "remaining_uncertainties": [],
                "slot": {
                    "start_iso": "2026-07-14T10:00:00Z",
                    "end_iso": "2026-07-14T11:00:00Z",
                    "confirmation_id": "conf-001",
                    "status": "confirmed",
                },
                "rationale": "Strong match across all criteria.",
            },
            {
                "candidate_id": "c002",
                "recommendation": "interview",
                "finalized": True,
                "interview_rounds": 1,
                "scorecard": {"python": 70, "aws": 60, "system_design": 75},
                "capabilities": {"python": {"score": 70, "confidence": "medium"}},
                "evidence_buckets": {"python": ["3 years Python experience"]},
                "injection_flag": False,
                "leakage_flag": False,
                "remaining_uncertainties": ["aws_depth"],
                "slot": None,
                "rationale": "Good match, some gaps in AWS.",
            },
            {
                "candidate_id": "c003",
                "recommendation": "reject",
                "finalized": True,
                "interview_rounds": 0,
                "scorecard": {"python": 30, "aws": 20, "system_design": 40},
                "capabilities": {"python": {"score": 30, "confidence": "high"}},
                "evidence_buckets": {"python": ["1 year Python experience"]},
                "injection_flag": False,
                "leakage_flag": False,
                "remaining_uncertainties": ["python", "aws", "system_design"],
                "slot": None,
                "rationale": "Does not meet minimum requirements.",
            },
        ],
        "shortlist": [
            {
                "candidate": "c001",
                "recommendation": "strong_interview",
                "weighted_score": 0.88,
                "confidence_summary": {"high": 3, "medium": 0, "low": 0},
                "strengths": ["Python expertise", "AWS experience", "System design"],
                "remaining_uncertainties": [],
                "evidence_refs": ["ev-001", "ev-002", "ev-003"],
            },
            {
                "candidate": "c002",
                "recommendation": "interview",
                "weighted_score": 0.68,
                "confidence_summary": {"high": 1, "medium": 2, "low": 0},
                "strengths": ["Python proficiency"],
                "remaining_uncertainties": ["aws_depth"],
                "evidence_refs": ["ev-004", "ev-005"],
            },
        ],
        "bias_reports": [
            {
                "candidate_a": "c001",
                "candidate_b": "c003",
                "severity": "low",
                "indicators": ["gender_parity"],
                "recommendation": "no_action",
                "rationale": "No significant bias detected.",
                "run_id": "test-run-001",
                "probed_at": "2026-07-10T05:00:00Z",
            },
        ],
        "trajectory": [],
    }


# ── JSON Export ───────────────────────────────────────────────────────────────


class TestJsonExport:
    def test_export_to_json_returns_string(self, final_state):
        """export_to_json must return a non-empty string."""
        result = export_to_json(final_state)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_export_to_json_valid_json(self, final_state):
        """export_to_json must produce valid JSON."""
        result = export_to_json(final_state)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_export_to_json_contains_expected_keys(self, final_state):
        """export_to_json must contain all expected top-level keys."""
        result = export_to_json(final_state)
        parsed = json.loads(result)
        assert "run_id" in parsed
        assert "candidates" in parsed
        assert "shortlist" in parsed
        assert "bias_reports" in parsed
        assert "step_count" in parsed

    def test_export_to_json_candidate_fields(self, final_state):
        """JSON export must include all §5 CandidateState fields."""
        result = export_to_json(final_state)
        parsed = json.loads(result)
        candidate = parsed["candidates"][0]
        expected_fields = {
            "candidate_id", "recommendation", "finalized", "interview_rounds",
            "scorecard", "capabilities", "evidence_buckets",
            "injection_flag", "leakage_flag", "remaining_uncertainties",
            "slot", "rationale",
        }
        assert expected_fields.issubset(candidate.keys())

    def test_export_to_json_shortlist_fields(self, final_state):
        """JSON export must include all §5 ShortlistEntry fields."""
        result = export_to_json(final_state)
        parsed = json.loads(result)
        entry = parsed["shortlist"][0]
        expected_fields = {
            "candidate", "recommendation", "weighted_score",
            "confidence_summary", "strengths", "remaining_uncertainties",
            "evidence_refs",
        }
        assert expected_fields.issubset(entry.keys())

    def test_export_to_json_bias_report_fields(self, final_state):
        """JSON export must include all §5 BiasReport fields."""
        result = export_to_json(final_state)
        parsed = json.loads(result)
        report = parsed["bias_reports"][0]
        expected_fields = {
            "candidate_a", "candidate_b", "severity", "indicators",
            "recommendation", "rationale", "run_id", "probed_at",
        }
        assert expected_fields.issubset(report.keys())

    def test_export_to_json_file(self, final_state):
        """export_to_json_file must write a valid JSON file."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.json"
            export_to_json_file(final_state, path)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            parsed = json.loads(content)
            assert parsed["run_id"] == "test-run-001"


# ── CSV Export ────────────────────────────────────────────────────────────────


class TestCsvExport:
    def test_export_candidates_to_csv_returns_string(self, final_state):
        """export_candidates_to_csv must return a non-empty string."""
        result = export_candidates_to_csv(final_state)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_export_candidates_to_csv_has_header(self, final_state):
        """CSV must have a header row matching §5 CandidateState fields."""
        result = export_candidates_to_csv(final_state)
        reader = csv.DictReader(io.StringIO(result))
        expected_fields = {
            "candidate_id", "recommendation", "finalized", "interview_rounds",
            "scorecard", "capabilities", "evidence_buckets",
            "injection_flag", "leakage_flag", "remaining_uncertainties",
            "slot", "rationale",
        }
        assert expected_fields.issubset(reader.fieldnames)

    def test_export_candidates_to_csv_has_rows(self, final_state):
        """CSV must have one row per candidate."""
        result = export_candidates_to_csv(final_state)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 3

    def test_export_shortlist_to_csv_has_header(self, final_state):
        """Shortlist CSV must have header matching §5 ShortlistEntry fields."""
        result = export_shortlist_to_csv(final_state)
        reader = csv.DictReader(io.StringIO(result))
        expected_fields = {
            "candidate", "recommendation", "weighted_score",
            "confidence_summary", "strengths", "remaining_uncertainties",
            "evidence_refs",
        }
        assert expected_fields.issubset(reader.fieldnames)

    def test_export_shortlist_to_csv_has_rows(self, final_state):
        """Shortlist CSV must have one row per entry."""
        result = export_shortlist_to_csv(final_state)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 2

    def test_export_bias_reports_to_csv_has_header(self, final_state):
        """Bias reports CSV must have header matching §5 BiasReport fields."""
        result = export_bias_reports_to_csv(final_state)
        reader = csv.DictReader(io.StringIO(result))
        expected_fields = {
            "candidate_a", "candidate_b", "severity", "indicators",
            "recommendation", "rationale", "run_id", "probed_at",
        }
        assert expected_fields.issubset(reader.fieldnames)

    def test_export_bias_reports_to_csv_has_rows(self, final_state):
        """Bias reports CSV must have one row per report."""
        result = export_bias_reports_to_csv(final_state)
        reader = csv.DictReader(io.StringIO(result))
        rows = list(reader)
        assert len(rows) == 1

    def test_export_to_csv_file(self, final_state):
        """export_to_csv_file must write a file with all sections."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.csv"
            export_to_csv_file(final_state, path)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert "# CANDIDATES" in content
            assert "# SHORTLIST" in content
            assert "# BIAS_REPORTS" in content

    def test_empty_candidates_returns_empty(self):
        """export_candidates_to_csv must return empty string for no candidates."""
        result = export_candidates_to_csv({"candidates": []})
        assert result == ""

    def test_empty_shortlist_returns_empty(self):
        """export_shortlist_to_csv must return empty string for no shortlist."""
        result = export_shortlist_to_csv({"shortlist": []})
        assert result == ""

    def test_empty_bias_reports_returns_empty(self):
        """export_bias_reports_to_csv must return empty string for no reports."""
        result = export_bias_reports_to_csv({"bias_reports": []})
        assert result == ""


# ── PDF Export ────────────────────────────────────────────────────────────────


class TestPdfExport:
    def test_export_to_pdf_creates_file(self, final_state):
        """export_to_pdf must create a non-empty file."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.pdf"
            export_to_pdf(final_state, path)
            assert path.exists()
            content = path.read_text(encoding="utf-8")
            assert len(content) > 0

    def test_export_to_pdf_contains_report_header(self, final_state):
        """PDF must contain the report header."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.pdf"
            export_to_pdf(final_state, path)
            content = path.read_text(encoding="utf-8")
            assert "SCOUTAI — HIRING PIPELINE REPORT" in content

    def test_export_to_pdf_contains_shortlist(self, final_state):
        """PDF must contain shortlist section."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.pdf"
            export_to_pdf(final_state, path)
            content = path.read_text(encoding="utf-8")
            assert "SHORTLIST" in content
            assert "c001" in content
            assert "c002" in content

    def test_export_to_pdf_contains_bias_reports(self, final_state):
        """PDF must contain bias reports section."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.pdf"
            export_to_pdf(final_state, path)
            content = path.read_text(encoding="utf-8")
            assert "FAIRNESS / BIAS REPORTS" in content

    def test_export_to_pdf_contains_candidates(self, final_state):
        """PDF must contain candidates section."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.pdf"
            export_to_pdf(final_state, path)
            content = path.read_text(encoding="utf-8")
            assert "CANDIDATES" in content
            assert "c001" in content
            assert "c002" in content
            assert "c003" in content

    def test_export_to_pdf_contains_run_metadata(self, final_state):
        """PDF must contain run metadata."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "output.pdf"
            export_to_pdf(final_state, path)
            content = path.read_text(encoding="utf-8")
            assert "test-run-001" in content
            assert "Step Count:" in content
            assert "42" in content.split("Step Count:")[1][:10]
