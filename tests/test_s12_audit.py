"""
S12 tests: audit log infrastructure with JSONL persistence and secret redaction.

Key test cases:
1.  compute_sha256: produces deterministic, correct-length hex string
2.  compute_sha256: different inputs produce different hashes
3.  SecretRedactionFilter: redacts configured patterns from text
4.  SecretRedactionFilter: redacts nested dict values
5.  SecretRedactionFilter: handles empty text gracefully
6.  SecretRedactionFilter: handles invalid patterns gracefully
7.  AuditLogger: writes entry to JSONL file
8.  AuditLogger: writes batch of entries
9.  AuditLogger: creates output directory lazily
10. AuditLogger: redacts secrets before writing
11. AuditLogger: close flushes and records entry count
12. AuditLogger: context manager support
13. TrajectoryEntry: full hashing via AgentHarness (integration)
14. SecretRedactionFilter: patterns from config.yaml work correctly
15. build_audit_hook: factory returns configured AuditLogger

No live API keys required.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest

from scoutai.audit.log import (
    AuditLogger,
    SecretRedactionFilter,
    build_audit_hook,
    compute_sha256,
    redact_trajectory_entry,
)
from scoutai.config import load_config
from scoutai.schemas import TrajectoryEntry

CONFIG_PATH = "config.yaml"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory and clean it up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def audit_logger(config, temp_dir) -> Generator[AuditLogger, None, None]:
    """Provide an AuditLogger writing to a temporary directory."""
    # Temporarily override the output directory
    original_dir = config.audit.output_dir
    config.audit.output_dir = str(temp_dir)
    logger = AuditLogger(config, run_id="test-run-s12")
    yield logger
    logger.close()
    config.audit.output_dir = original_dir


def _make_entry(
    node: str = "test_node",
    tool: str | None = None,
    status: str = "success",
) -> TrajectoryEntry:
    return TrajectoryEntry(
        node=node,
        tool_used=tool,
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_hash="a" * 64,
        output_hash="b" * 64,
        latency_ms=42,
        model="gemini/gemini-2.0-flash",
        model_role="fast_structured",
        prompt_version="1.0.0",
        tool_version="1.0.0",
        schema_version="1.0.0",
        status=status,  # type: ignore[arg-type]
    )


# ── compute_sha256 ────────────────────────────────────────────────────────────


class TestComputeSha256:
    def test_produces_hex_string(self):
        """compute_sha256 must return a 64-character hex string."""
        result = compute_sha256({"key": "value"})
        assert len(result) == 64
        # Must be valid hex
        int(result, 16)

    def test_deterministic_output(self):
        """Same input must produce the same hash."""
        data = {"candidate": "c001", "score": 85}
        assert compute_sha256(data) == compute_sha256(data)

    def test_different_inputs_different_hashes(self):
        """Different inputs must produce different hashes."""
        h1 = compute_sha256({"a": 1})
        h2 = compute_sha256({"a": 2})
        assert h1 != h2

    def test_string_input(self):
        """compute_sha256 must work with plain strings."""
        result = compute_sha256("hello world")
        assert len(result) == 64

    def test_list_input(self):
        """compute_sha256 must work with lists."""
        result = compute_sha256([1, 2, 3])
        assert len(result) == 64

    def test_none_input(self):
        """compute_sha256 must work with None."""
        result = compute_sha256(None)
        assert len(result) == 64


# ── SecretRedactionFilter ─────────────────────────────────────────────────────


class TestSecretRedactionFilter:
    def test_redacts_api_key_pattern(self, config):
        """SecretRedactionFilter must redact API key patterns."""
        filter_obj = SecretRedactionFilter(config)
        text = "My API key is AIzaSyDeadBeefDeadBeefDeadBeefDeadBeefDeadBe"
        redacted = filter_obj.redact(text)
        assert "[REDACTED]" in redacted
        assert "AIzaSyD" not in redacted

    def test_redacts_groq_key_pattern(self, config):
        """SecretRedactionFilter must redact Groq API key patterns."""
        filter_obj = SecretRedactionFilter(config)
        text = "groq_key=gsk_" + "a" * 50
        redacted = filter_obj.redact(text)
        assert "[REDACTED]" in redacted

    def test_redacts_dict_values(self, config):
        """redact_dict must redact secrets from all string values."""
        filter_obj = SecretRedactionFilter(config)
        data = {
            "api_key": "AIzaSyDeadBeefDeadBeefDeadBeefDeadBeefDeadBe",
            "normal_field": "This is safe",
            "nested": {
                "secret": "gsk_deadbeef1234567890" + "x" * 30,
            },
        }
        redacted = filter_obj.redact_dict(data)
        # Serialize to check all values are redacted
        serialized = json.dumps(redacted)
        assert "[REDACTED]" in serialized
        assert "AIzaSyD" not in serialized

    def test_empty_text_unchanged(self, config):
        """Empty text must be returned as-is."""
        filter_obj = SecretRedactionFilter(config)
        assert filter_obj.redact("") == ""
        assert filter_obj.redact("  ") == "  "

    def test_no_secrets_unchanged(self, config):
        """Text without secrets must be returned unchanged."""
        filter_obj = SecretRedactionFilter(config)
        text = "This is a clean string with no secrets."
        assert filter_obj.redact(text) == text

    def test_invalid_pattern_skipped(self, config):
        """Invalid regex patterns must be skipped with a warning."""
        # Add an invalid pattern temporarily
        original = list(config.security.secret_redaction_patterns)
        config.security.secret_redaction_patterns.append("[invalid")
        try:
            filter_obj = SecretRedactionFilter(config)
            # Should not crash, and valid patterns should still work
            text = "API key AIzaSyDeadBeefDeadBeefDeadBeefDeadBeefDeadBe here"
            redacted = filter_obj.redact(text)
            assert "[REDACTED]" in redacted
        finally:
            config.security.secret_redaction_patterns = original


# ── AuditLogger ───────────────────────────────────────────────────────────────


class TestAuditLogger:
    def test_writes_single_entry(self, audit_logger):
        """AuditLogger must write a single entry to the JSONL file."""
        entry = _make_entry(node="screen_resume")
        audit_logger.write(entry)
        audit_logger.close()

        # Verify the file exists and contains valid JSON
        assert audit_logger.file_path is not None
        assert audit_logger.file_path.exists()
        content = audit_logger.file_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["node"] == "screen_resume"
        assert parsed["status"] == "success"

    def test_writes_batch_entries(self, audit_logger):
        """AuditLogger must write multiple entries in batch."""
        entries = [
            _make_entry(node="setup_role"),
            _make_entry(node="build_rubric"),
            _make_entry(node="candidate_agent", tool="extract_evidence"),
        ]
        audit_logger.write_batch(entries)
        audit_logger.close()

        assert audit_logger.file_path is not None
        content = audit_logger.file_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["node"] == "setup_role"
        assert json.loads(lines[1])["node"] == "build_rubric"

    def test_empty_batch_does_nothing(self, audit_logger):
        """Empty batch must not create a file."""
        audit_logger.write_batch([])
        # File shouldn't exist yet (lazy creation)
        assert audit_logger.file_path is None or not audit_logger.file_path.exists()

    def test_entry_count_tracked(self, audit_logger):
        """AuditLogger must track entry count."""
        audit_logger.write(_make_entry())
        assert audit_logger.entry_count == 1
        audit_logger.write(_make_entry())
        assert audit_logger.entry_count == 2

    def test_secrets_redacted_in_output(self, config):
        """AuditLogger must redact secrets before writing."""
        with tempfile.TemporaryDirectory() as d:
            config.audit.output_dir = str(d)
            logger = AuditLogger(config, run_id="secret-test")

            # Create an entry with an API key in the model field
            entry = TrajectoryEntry(
                node="screen_resume",
                timestamp=datetime.now(timezone.utc).isoformat(),
                input_hash="a" * 64,
                output_hash="b" * 64,
                latency_ms=10,
                model="AIzaSyDeadBeefDeadBeefDeadBeefDeadBeefDeadBe",
                model_role="fast_structured",
                prompt_version="1.0.0",
                tool_version="1.0.0",
                schema_version="1.0.0",
                status="success",
            )
            logger.write(entry)
            logger.close()

            content = logger.file_path.read_text(encoding="utf-8")
            assert "AIzaSyD" not in content, "Secret must be redacted from audit log"
            assert "[REDACTED]" in content

    def test_context_manager(self, config, temp_dir):
        """AuditLogger must work as a context manager."""
        config.audit.output_dir = str(temp_dir)
        with AuditLogger(config, run_id="ctx-test") as logger:
            logger.write(_make_entry())
            assert logger.entry_count == 1
        # After context exit, file should exist
        assert logger.file_path is not None
        assert logger.file_path.exists()

    def test_multiple_loggers_independent(self, config, temp_dir):
        """Two AuditLoggers with different run_ids must write to different files."""
        config.audit.output_dir = str(temp_dir)
        logger_a = AuditLogger(config, run_id="run-a")
        logger_b = AuditLogger(config, run_id="run-b")

        logger_a.write(_make_entry(node="node_a"))
        logger_b.write(_make_entry(node="node_b"))

        logger_a.close()
        logger_b.close()

        content_a = logger_a.file_path.read_text(encoding="utf-8")
        content_b = logger_b.file_path.read_text(encoding="utf-8")
        assert json.loads(content_a)["node"] == "node_a"
        assert json.loads(content_b)["node"] == "node_b"
        assert logger_a.file_path.name != logger_b.file_path.name

    def test_append_mode_on_reopen(self, config, temp_dir):
        """Reopening a logger with the same run_id must append, not overwrite."""
        config.audit.output_dir = str(temp_dir)
        logger1 = AuditLogger(config, run_id="append-test")
        logger1.write(_make_entry(node="first"))
        logger1.close()

        logger2 = AuditLogger(config, run_id="append-test")
        logger2.write(_make_entry(node="second"))
        logger2.close()

        content = logger2.file_path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["node"] == "first"
        assert json.loads(lines[1])["node"] == "second"


# ── build_audit_hook ─────────────────────────────────────────────────────────


class TestBuildAuditHook:
    def test_returns_configured_logger(self, config):
        """build_audit_hook must return an AuditLogger with the given run_id."""
        logger = build_audit_hook(config, run_id="hook-test")
        assert isinstance(logger, AuditLogger)
        logger.close()

    def test_logger_writes_expected_filename(self, config, temp_dir):
        """The logger must use the configured filename template."""
        config.audit.output_dir = str(temp_dir)
        logger = build_audit_hook(config, run_id="my-run")
        logger.write(_make_entry())
        logger.close()

        expected_name = config.audit.filename_template.format(run_id="my-run")
        assert logger.file_path is not None
        assert logger.file_path.name == expected_name


# ── redact_trajectory_entry ──────────────────────────────────────────────────


class TestRedactTrajectoryEntry:
    def test_returns_entry_unchanged(self, config):
        """redact_trajectory_entry must return the entry unchanged (hashes are safe)."""
        filter_obj = SecretRedactionFilter(config)
        entry = _make_entry(node="test")
        result = redact_trajectory_entry(entry, filter_obj)
        assert result.node == entry.node
        assert result.input_hash == entry.input_hash
        assert result.output_hash == entry.output_hash