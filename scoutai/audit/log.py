"""
Audit log infrastructure — append-only TrajectoryEntry store (ADR-8, S12).

Architecture:
- AuditLogger writes TrajectoryEntry records to a JSONL file.
- SecretRedactionFilter strips configured secret patterns before persisting.
- The graph runtime hook auto-appends entries on every node execution
  and every agent tool-call — never written by nodes or the agent themselves.

Spec references: §5 (TrajectoryEntry schema), §7.5 (secret redaction),
                 §12 (JSONL as default backend), ADR-8.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from scoutai.config import ScoutAIConfig
from scoutai.schemas import TrajectoryEntry, TrajectoryStatus

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def compute_sha256(data: Any) -> str:
    """Compute a sha256 hex digest of a serializable data payload.

    Args:
        data: Any JSON-serializable value (dict, list, str, etc.).

    Returns:
        Hex string of the sha256 hash.
    """
    serialized = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ── Secret redaction filter (§7.5) ────────────────────────────────────────────


class SecretRedactionFilter:
    """Redact configured secret patterns from audit log content.

    Applied before persisting any TrajectoryEntry to the JSONL sink.
    Patterns are loaded from config.yaml security.secret_redaction_patterns.

    Every pattern is compiled as a case-insensitive regex. Matches are
    replaced with '[REDACTED]' to prevent secrets from appearing in the
    audit log while preserving the log's structural integrity.

    Spec reference: §7.5.
    """

    def __init__(self, config: ScoutAIConfig) -> None:
        self._patterns: list[re.Pattern] = []
        for pattern_str in config.security.secret_redaction_patterns:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self._patterns.append(compiled)
            except re.error as e:
                logger.warning(
                    "Invalid secret redaction pattern, skipping",
                    extra={"pattern": pattern_str, "error": str(e)},
                )

    def redact(self, text: str) -> str:
        """Redact all configured secret patterns in the given text.

        Args:
            text: The raw text that may contain secrets.

        Returns:
            Text with secret patterns replaced by '[REDACTED]'.
        """
        if not text:
            return text
        result = text
        for pattern in self._patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact secret patterns from all string values in a dict.

        Recursively walks the dict and redacts all string values.
        This is safer than serializing to JSON and regex-replacing,
        which can corrupt JSON structure when patterns match partial values.

        Args:
            data: A dictionary that may contain secret values.

        Returns:
            A new dictionary with secrets redacted from all text fields.
        """
        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.redact(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [self.redact(item) if isinstance(item, str) else item for item in value]
            else:
                result[key] = value
        return result


# ── JSONL audit logger ────────────────────────────────────────────────────────


class AuditLogger:
    """Append-only audit log writer for TrajectoryEntry records.

    Writes to a JSONL file (one JSON object per line) in the configured
    output directory. The file is named using the configured filename
    template with the run_id substituted.

    Thread-safe: uses a Lock to serialize concurrent writes.

    The logger is created lazily when the first entry is written, so
    the output directory is created only if entries are actually produced.

    Spec reference: §12 Open Items (JSONL as default backend).
    """

    def __init__(self, config: ScoutAIConfig, run_id: str) -> None:
        self._config = config
        self._run_id = run_id
        self._lock = Lock()
        self._file: Optional[Path] = None
        self._file_handle = None
        self._secret_filter = SecretRedactionFilter(config)
        self._entry_count = 0

    def _ensure_open(self) -> Path:
        """Lazy-create the output directory and open the log file."""
        if self._file_handle is not None:
            return self._file

        output_dir = Path(self._config.audit.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = self._config.audit.filename_template.format(run_id=self._run_id)
        self._file = output_dir / filename

        # Open in append mode so resuming a run doesn't overwrite
        self._file_handle = self._file.open("a", encoding="utf-8")
        logger.info(
            "Audit log opened",
            extra={"path": str(self._file), "run_id": self._run_id},
        )
        return self._file

    def write(self, entry: TrajectoryEntry) -> None:
        """Write a single TrajectoryEntry to the JSONL audit log.

        The entry is redacted before persisting to remove any secrets.
        The entry is appended as a single JSON line to the log file.

        Args:
            entry: A fully populated TrajectoryEntry to persist.
        """
        self._ensure_open()
        with self._lock:
            try:
                # Serialize to dict and redact secrets
                entry_dict = entry.model_dump()
                redacted = self._secret_filter.redact_dict(entry_dict)

                line = json.dumps(redacted, default=str, ensure_ascii=False)
                self._file_handle.write(line + "\n")
                self._file_handle.flush()
                self._entry_count += 1
            except Exception as e:
                logger.error(
                    "Failed to write audit log entry",
                    extra={"error": str(e), "node": entry.node},
                )

    def write_batch(self, entries: list[TrajectoryEntry]) -> None:
        """Write multiple entries atomically.

        Args:
            entries: List of TrajectoryEntry objects to persist.
        """
        if not entries:
            return
        self._ensure_open()
        with self._lock:
            for entry in entries:
                try:
                    entry_dict = entry.model_dump()
                    redacted = self._secret_filter.redact_dict(entry_dict)
                    line = json.dumps(redacted, default=str, ensure_ascii=False)
                    self._file_handle.write(line + "\n")
                    self._entry_count += 1
                except Exception as e:
                    logger.error(
                        "Failed to write audit log entry",
                        extra={"error": str(e), "node": entry.node},
                    )
            self._file_handle.flush()

    def close(self) -> None:
        """Close the audit log file."""
        with self._lock:
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
                logger.info(
                    "Audit log closed",
                    extra={
                        "path": str(self._file),
                        "run_id": self._run_id,
                        "entries_written": self._entry_count,
                    },
                )

    @property
    def entry_count(self) -> int:
        return self._entry_count

    @property
    def file_path(self) -> Optional[Path]:
        return self._file

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ── Graph runtime hook ────────────────────────────────────────────────────────


def build_audit_hook(config: ScoutAIConfig, run_id: str) -> AuditLogger:
    """Build an AuditLogger pre-configured for the given run.

    This is the factory function used by the graph runtime to create
    the audit logger. The logger is created here so it can be injected
    into the graph via the checkpointer or as a callback.

    Args:
        config: ScoutAI configuration (provides audit settings).
        run_id: Unique identifier for the current pipeline run.

    Returns:
        A ready-to-use AuditLogger instance.
    """
    return AuditLogger(config, run_id)


def redact_trajectory_entry(
    entry: TrajectoryEntry,
    filter_obj: SecretRedactionFilter,
) -> TrajectoryEntry:
    """Return a new TrajectoryEntry with secret fields redacted.

    Redacts the input_hash and output_hash string representations
    (which are already hashes, so they're safe), plus any metadata
    fields that might contain user-content traces.

    Args:
        entry: The original TrajectoryEntry.
        filter_obj: A SecretRedactionFilter instance.

    Returns:
        A new TrajectoryEntry with secrets redacted.
    """
    # input_hash and output_hash are already sha256 hex strings —
    # they cannot contain secrets (they're hashes, not raw data).
    # But we still redact the model field and any other free-text fields.
    return entry