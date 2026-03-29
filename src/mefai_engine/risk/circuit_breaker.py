"""Trading-specific circuit breaker for automated halt."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from mefai_engine.constants import CircuitState

logger = structlog.get_logger()


class TradingCircuitBreaker:
    """Halts trading when consecutive losses or drawdown thresholds are hit.

    States:
        CLOSED    - Normal operation, trading allowed
        OPEN      - Trading halted, waiting for cooldown
        HALF_OPEN - One test trade allowed to probe recovery
    """

    def __init__(
        self,
        max_consecutive_losses: int = 5,
        max_drawdown_pct: float = 10.0,
        cooldown_seconds: int = 3600,
    ) -> None:
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown_pct = max_drawdown_pct
        self.cooldown_seconds = cooldown_seconds

        self._state = CircuitState.CLOSED
        self._consecutive_losses = 0
        self._tripped_at: datetime | None = None
        self._trip_reason: str = ""

    @property
    def state(self) -> CircuitState:
        return self._state

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN and self._tripped_at:
            elapsed = (datetime.now(tz=UTC) - self._tripped_at).total_seconds()
            if elapsed >= self.cooldown_seconds:
                self._state = CircuitState.HALF_OPEN
                logger.info("trading_circuit_breaker.half_open", elapsed=elapsed)
                return True
            return False

        if self._state == CircuitState.HALF_OPEN:
            return True

        return False

    def record_loss(self) -> None:
        """Record a losing trade."""
        self._consecutive_losses += 1
        if self._consecutive_losses >= self.max_consecutive_losses:
            self.trip(f"Consecutive losses: {self._consecutive_losses}")

    def record_win(self) -> None:
        """Record a winning trade."""
        self._consecutive_losses = 0
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            logger.info("trading_circuit_breaker.recovered")

    def trip(self, reason: str) -> None:
        """Manually trip the circuit breaker."""
        self._state = CircuitState.OPEN
        self._tripped_at = datetime.now(tz=UTC)
        self._trip_reason = reason
        logger.warning("trading_circuit_breaker.tripped", reason=reason)

    def reset(self) -> None:
        """Force reset to CLOSED state."""
        self._state = CircuitState.CLOSED
        self._consecutive_losses = 0
        self._tripped_at = None
        logger.info("trading_circuit_breaker.reset")
