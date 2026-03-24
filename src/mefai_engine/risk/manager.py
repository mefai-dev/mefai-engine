"""Central risk manager - every order must pass through here."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from mefai_engine.config import RiskConfig
from mefai_engine.constants import Direction, RiskDecisionType
from mefai_engine.risk.circuit_breaker import TradingCircuitBreaker
from mefai_engine.risk.limits import (
    check_daily_loss,
    check_drawdown,
    check_max_exposure,
    check_position_size,
)
from mefai_engine.risk.pnl_tracker import PnLTracker
from mefai_engine.types import Balance, Position, RiskDecision, Signal

logger = structlog.get_logger()


class RiskManager:
    """Central risk authority.

    Every signal must be evaluated here before execution.
    The risk manager can APPROVE, REDUCE, or REJECT any trade.
    """

    def __init__(self, config: RiskConfig) -> None:
        self._config = config
        self._circuit_breaker = TradingCircuitBreaker(
            max_consecutive_losses=config.max_consecutive_losses,
            max_drawdown_pct=config.max_drawdown_pct,
            cooldown_seconds=config.circuit_breaker_cooldown_seconds,
        )
        self._pnl_tracker = PnLTracker()
        self._daily_loss = 0.0
        self._daily_reset_date: str = ""

    async def evaluate(
        self,
        signal: Signal,
        balance: Balance,
        positions: list[Position],
    ) -> RiskDecision:
        """Evaluate a trading signal against all risk rules.

        Returns:
            RiskDecision with APPROVED, REDUCED, or REJECTED status.
        """
        passed: list[str] = []
        failed: list[str] = []

        # Reset daily loss tracking
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_loss = 0.0
            self._daily_reset_date = today

        # Check circuit breaker first
        if not self._circuit_breaker.can_trade():
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                approved_size_pct=0.0,
                reason="Circuit breaker is OPEN - trading halted",
                checks_failed=["circuit_breaker"],
            )
        passed.append("circuit_breaker")

        # Skip checks for FLAT signals (close positions)
        if signal.direction == Direction.FLAT:
            return RiskDecision(
                decision=RiskDecisionType.APPROVED,
                approved_size_pct=signal.suggested_size_pct,
                reason="Close signal approved",
                checks_passed=["flat_signal"],
            )

        # 1. Position size check
        size_ok, max_size = check_position_size(
            signal.suggested_size_pct, self._config.max_position_pct
        )
        if size_ok:
            passed.append("position_size")
        else:
            failed.append("position_size")

        # 2. Total exposure check
        current_exposure = sum(
            (p.size * p.entry_price) / balance.total * 100
            for p in positions
            if balance.total > 0
        )
        exposure_ok, remaining = check_max_exposure(
            current_exposure,
            signal.suggested_size_pct,
            self._config.max_total_exposure_pct,
        )
        if exposure_ok:
            passed.append("total_exposure")
        else:
            failed.append("total_exposure")

        # 3. Daily loss check
        daily_ok = check_daily_loss(
            self._daily_loss, balance.total, self._config.max_daily_loss_pct
        )
        if daily_ok:
            passed.append("daily_loss")
        else:
            failed.append("daily_loss")
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                approved_size_pct=0.0,
                reason=f"Daily loss limit reached: {self._daily_loss:.2f} USDT",
                checks_passed=passed,
                checks_failed=failed,
            )

        # 4. Drawdown check
        dd_ok = check_drawdown(
            self._pnl_tracker.current_drawdown_pct,
            self._config.max_drawdown_pct,
        )
        if dd_ok:
            passed.append("drawdown")
        else:
            failed.append("drawdown")
            self._circuit_breaker.trip("Max drawdown exceeded")
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                approved_size_pct=0.0,
                reason=f"Drawdown limit exceeded: {self._pnl_tracker.current_drawdown_pct:.1f}%",
                checks_passed=passed,
                checks_failed=failed,
            )

        # Determine final size
        approved_size = min(signal.suggested_size_pct, max_size, remaining)

        if not size_ok or not exposure_ok:
            if approved_size > 0:
                return RiskDecision(
                    decision=RiskDecisionType.REDUCED,
                    approved_size_pct=approved_size,
                    reason=f"Size reduced from {signal.suggested_size_pct:.1f}% to {approved_size:.1f}%",
                    checks_passed=passed,
                    checks_failed=failed,
                )
            return RiskDecision(
                decision=RiskDecisionType.REJECTED,
                approved_size_pct=0.0,
                reason="No room for new position within risk limits",
                checks_passed=passed,
                checks_failed=failed,
            )

        return RiskDecision(
            decision=RiskDecisionType.APPROVED,
            approved_size_pct=approved_size,
            reason="All risk checks passed",
            checks_passed=passed,
        )

    def record_trade_result(self, pnl: float) -> None:
        """Record a closed trade's P&L for tracking."""
        self._pnl_tracker.record(pnl)
        if pnl < 0:
            self._daily_loss += abs(pnl)
            self._circuit_breaker.record_loss()
        else:
            self._circuit_breaker.record_win()

    @property
    def pnl_tracker(self) -> PnLTracker:
        return self._pnl_tracker

    @property
    def circuit_breaker(self) -> TradingCircuitBreaker:
        return self._circuit_breaker
