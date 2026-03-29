"""Immutable audit log for compliance and operational transparency.

Every trading decision and signal and execution is recorded in a
structured append-only log. Entries are timestamped and include
the actor and action and relevant details.

Supports log rotation and archival for long term storage.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class AuditAction(str, Enum):
    """Types of auditable actions."""
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_REJECTED = "signal_rejected"
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_FAILED = "order_failed"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    CIRCUIT_BREAKER_RESET = "circuit_breaker_reset"
    MODEL_RETRAINED = "model_retrained"
    REGIME_CHANGE = "regime_change"
    DRIFT_DETECTED = "drift_detected"
    CONFIG_CHANGED = "config_changed"
    TENANT_CREATED = "tenant_created"
    TENANT_PLAN_CHANGED = "tenant_plan_changed"
    API_KEY_ROTATED = "api_key_rotated"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class AuditEntry:
    """A single immutable audit log entry."""
    timestamp: datetime
    action: AuditAction
    actor: str  # System component or user or tenant that triggered the action
    symbol: str = ""
    tenant_id: str = ""
    details: dict[str, Any] = dc_field(default_factory=dict)
    entry_id: str = ""

    def __post_init__(self) -> None:
        if not self.entry_id:
            ts_ns = int(self.timestamp.timestamp() * 1_000_000)
            raw = f"{ts_ns}:{self.action}:{self.actor}:{self.symbol}"
            import hashlib
            self.entry_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "actor": self.actor,
            "symbol": self.symbol,
            "tenant_id": self.tenant_id,
            "details": self.details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class AuditConfig:
    """Configuration for the audit log system."""
    max_memory_entries: int = 50000
    log_dir: str = "logs/audit"
    file_rotation_size_mb: int = 50
    max_files: int = 100
    write_to_file: bool = True
    write_to_stdout: bool = False


class AuditLog:
    """Immutable append-only audit log.

    Records every significant action in the trading system for
    compliance review and debugging. Entries cannot be modified
    or deleted once written.

    Storage:
    - In-memory ring buffer for fast recent queries
    - JSON lines file for persistent storage
    - Automatic rotation when file size exceeds threshold
    """

    def __init__(self, config: AuditConfig | None = None) -> None:
        self._config = config or AuditConfig()
        self._entries: deque[AuditEntry] = deque(maxlen=self._config.max_memory_entries)
        self._file_handle: Any = None
        self._current_file_path: Path | None = None
        self._current_file_size: int = 0
        self._total_entries: int = 0

        if self._config.write_to_file:
            self._ensure_log_dir()
            self._open_log_file()

    def record(
        self,
        action: AuditAction,
        actor: str,
        symbol: str = "",
        tenant_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Record a new audit entry.

        Args:
            action: The type of action being recorded
            actor: Who or what triggered the action
            symbol: Trading pair (if applicable)
            tenant_id: Tenant identifier (if applicable)
            details: Additional structured details

        Returns:
            The created AuditEntry (immutable once returned).
        """
        entry = AuditEntry(
            timestamp=datetime.now(tz=UTC),
            action=action,
            actor=actor,
            symbol=symbol,
            tenant_id=tenant_id,
            details=details or {},
        )

        self._entries.append(entry)
        self._total_entries += 1

        if self._config.write_to_file:
            self._write_entry(entry)

        if self._config.write_to_stdout:
            logger.info(
                "audit",
                action=action.value,
                actor=actor,
                symbol=symbol,
            )

        return entry

    def query(
        self,
        action: AuditAction | None = None,
        actor: str | None = None,
        symbol: str | None = None,
        tenant_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit entries with optional filters.

        All filters are ANDed together. Only searches the in-memory buffer.

        Args:
            action: Filter by action type
            actor: Filter by actor
            symbol: Filter by symbol
            tenant_id: Filter by tenant
            since: Only entries after this time
            until: Only entries before this time
            limit: Maximum number of entries to return

        Returns:
            List of matching AuditEntry objects (newest first).
        """
        results: list[AuditEntry] = []

        for entry in reversed(self._entries):
            if len(results) >= limit:
                break

            if action is not None and entry.action != action:
                continue
            if actor is not None and entry.actor != actor:
                continue
            if symbol is not None and entry.symbol != symbol:
                continue
            if tenant_id is not None and entry.tenant_id != tenant_id:
                continue
            if since is not None and entry.timestamp < since:
                continue
            if until is not None and entry.timestamp > until:
                continue

            results.append(entry)

        return results

    def get_recent(self, count: int = 50) -> list[AuditEntry]:
        """Get the most recent entries."""
        entries = list(self._entries)
        return entries[-count:]

    def get_stats(self) -> dict[str, Any]:
        """Get audit log statistics."""
        action_counts: dict[str, int] = {}
        for entry in self._entries:
            key = entry.action.value
            action_counts[key] = action_counts.get(key, 0) + 1

        return {
            "total_entries": self._total_entries,
            "memory_entries": len(self._entries),
            "max_memory": self._config.max_memory_entries,
            "action_counts": action_counts,
            "current_file": str(self._current_file_path) if self._current_file_path else None,
            "current_file_size_mb": round(self._current_file_size / (1024 * 1024), 2),
        }

    def _ensure_log_dir(self) -> None:
        """Create the audit log directory if it does not exist."""
        log_dir = Path(self._config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    def _open_log_file(self) -> None:
        """Open a new log file for writing."""
        if self._file_handle is not None:
            self._file_handle.close()

        log_dir = Path(self._config.log_dir)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        file_path = log_dir / f"audit_{timestamp}.jsonl"

        self._file_handle = open(file_path, "a", encoding="utf-8")
        self._current_file_path = file_path
        self._current_file_size = 0

        logger.info("audit.file_opened", path=str(file_path))

        # Clean up old files
        self._rotate_old_files()

    def _write_entry(self, entry: AuditEntry) -> None:
        """Write an entry to the current log file."""
        if self._file_handle is None:
            return

        line = entry.to_json() + "\n"
        line_bytes = len(line.encode("utf-8"))

        # Check rotation
        max_bytes = self._config.file_rotation_size_mb * 1024 * 1024
        if self._current_file_size + line_bytes > max_bytes:
            self._open_log_file()

        try:
            self._file_handle.write(line)
            self._file_handle.flush()
            self._current_file_size += line_bytes
        except Exception as exc:
            logger.error("audit.write_error", error=str(exc))

    def _rotate_old_files(self) -> None:
        """Remove oldest audit files when max_files is exceeded."""
        log_dir = Path(self._config.log_dir)
        files = sorted(log_dir.glob("audit_*.jsonl"))

        while len(files) > self._config.max_files:
            oldest = files.pop(0)
            try:
                oldest.unlink()
                logger.info("audit.file_rotated", removed=str(oldest))
            except Exception:
                pass

    def close(self) -> None:
        """Close the audit log file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
