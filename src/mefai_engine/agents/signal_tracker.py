"""Signal evolution tracker - monitors how trading signals change over time.

Inspired by MEFAI's existing signal processing pipeline and the concept of
tracking investment thesis evolution with new information.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum

import structlog

from mefai_engine.constants import Direction
from mefai_engine.types import Signal

logger = structlog.get_logger()


class SignalEvolution(StrEnum):
    """How a signal has evolved since creation."""
    STRENGTHENED = "strengthened"
    WEAKENED = "weakened"
    FALSIFIED = "falsified"
    UNCHANGED = "unchanged"
    REVERSED = "reversed"


@dataclass(slots=True)
class TrackedSignal:
    """A signal being tracked over time."""
    original: Signal
    current_confidence: float
    evolution: SignalEvolution
    updates: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    last_update: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    confidence_history: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class SignalTracker:
    """Tracks the evolution of trading signals over time.

    When a new signal comes in for the same symbol, it evaluates whether
    the thesis is strengthening, weakening, or has been falsified.

    This helps avoid:
    - Holding positions where the original thesis no longer holds
    - Flip-flopping between long/short too quickly
    - Missing opportunities when signals strengthen
    """

    def __init__(
        self,
        falsification_threshold: float = 0.3,
        strengthening_delta: float = 0.1,
        max_tracked: int = 50,
    ) -> None:
        self._tracked: dict[str, TrackedSignal] = {}
        self._falsification_threshold = falsification_threshold
        self._strengthening_delta = strengthening_delta
        self._max_tracked = max_tracked

    def track(self, signal: Signal) -> TrackedSignal:
        """Start tracking a new signal or update existing."""
        key = f"{signal.symbol}:{signal.strategy_id}"
        existing = self._tracked.get(key)

        if existing is None:
            tracked = TrackedSignal(
                original=signal,
                current_confidence=signal.confidence,
                evolution=SignalEvolution.UNCHANGED,
                confidence_history=[signal.confidence],
            )
            self._tracked[key] = tracked
            logger.info(
                "signal_tracker.new",
                symbol=signal.symbol,
                direction=signal.direction,
                confidence=signal.confidence,
            )
            return tracked

        return self._update(existing, signal)

    def _update(self, tracked: TrackedSignal, new_signal: Signal) -> TrackedSignal:
        """Update a tracked signal with new information."""
        tracked.updates += 1
        tracked.last_update = datetime.now(tz=timezone.utc)
        tracked.confidence_history.append(new_signal.confidence)
        old_conf = tracked.current_confidence
        tracked.current_confidence = new_signal.confidence

        # Direction reversal = falsified
        if new_signal.direction != tracked.original.direction:
            if new_signal.direction != Direction.FLAT:
                tracked.evolution = SignalEvolution.REVERSED
                tracked.notes.append(
                    f"Direction reversed: {tracked.original.direction} -> {new_signal.direction}"
                )
                logger.warning(
                    "signal_tracker.reversed",
                    symbol=new_signal.symbol,
                    from_dir=tracked.original.direction,
                    to_dir=new_signal.direction,
                )
            else:
                tracked.evolution = SignalEvolution.FALSIFIED
                tracked.notes.append("Signal went FLAT - thesis falsified")
            return tracked

        # Confidence dropped below threshold
        if new_signal.confidence < self._falsification_threshold:
            tracked.evolution = SignalEvolution.FALSIFIED
            tracked.notes.append(
                f"Confidence dropped to {new_signal.confidence:.3f} (threshold: {self._falsification_threshold})"
            )
            return tracked

        # Confidence change
        delta = new_signal.confidence - old_conf
        if delta > self._strengthening_delta:
            tracked.evolution = SignalEvolution.STRENGTHENED
            tracked.notes.append(f"Confidence +{delta:.3f}")
        elif delta < -self._strengthening_delta:
            tracked.evolution = SignalEvolution.WEAKENED
            tracked.notes.append(f"Confidence {delta:.3f}")
        else:
            tracked.evolution = SignalEvolution.UNCHANGED

        return tracked

    def get(self, symbol: str, strategy_id: str) -> TrackedSignal | None:
        """Get tracked signal for a symbol."""
        return self._tracked.get(f"{symbol}:{strategy_id}")

    def get_all(self) -> dict[str, TrackedSignal]:
        """Get all tracked signals."""
        return dict(self._tracked)

    def remove(self, symbol: str, strategy_id: str) -> None:
        """Stop tracking a signal (e.g., after position closed)."""
        key = f"{symbol}:{strategy_id}"
        self._tracked.pop(key, None)

    def get_active_count(self) -> int:
        return len(self._tracked)

    def get_falsified(self) -> list[TrackedSignal]:
        """Get all falsified signals (should be closed)."""
        return [
            t for t in self._tracked.values()
            if t.evolution in (SignalEvolution.FALSIFIED, SignalEvolution.REVERSED)
        ]
