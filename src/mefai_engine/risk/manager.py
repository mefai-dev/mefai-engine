"""Central risk manager - every order must pass through here."""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from mefai_engine.config import RiskConfig
from mefai_engine.constants import Direction, RiskDecisionType
from mefai_engine.risk.circuit_breaker import TradingCircuitBreaker
from mefai_engine.risk.correlation import CorrelationConfig, CorrelationRiskManager
from mefai_engine.risk.kelly import KellyConfig, KellyCriterion
from mefai_engine.risk.limits import (
    check_daily_loss,
    check_drawdown,
    check_max_exposure,
    check_position_size,
)
from mefai_engine.risk.liquidity import LiquidityConfig, LiquidityFilter
from mefai_engine.risk.pnl_tracker import PnLTracker
from mefai_engine.risk.var import VaRCalculator, VaRConfig
from mefai_engine.types import Balance, Position, RiskDecision, Signal

logger = structlog.get_logger()


class RiskManager:
    """Central risk authority.

    Every signal must be evaluated here before execution.
    The risk manager can APPROVE, REDUCE, or REJECT any trade.

    Integrates:
    - Position size and exposure limits
    - Daily loss and drawdown limits
    - Circuit breaker
    - Kelly criterion position sizing
    - Correlation risk management
    - Value at Risk (VaR / CVaR) limits
    - Liquidity analysis and filtering
    """

    def __init__(
        self,
        config: RiskConfig,
        kelly_config: KellyConfig | None = None,
        correlation_config: CorrelationConfig | None = None,
        var_config: VaRConfig | None = None,
        liquidity_config: LiquidityConfig | None = None,
    ) -> None:
        self._config = config
        self._circuit_breaker = TradingCircuitBreaker(
            max_consecutive_losses=config.max_consecutive_losses,
            max_drawdown_pct=config.max_drawdown_pct,
            cooldown_seconds=config.circuit_breaker_cooldown_seconds,
        )
        self._pnl_tracker = PnLTracker()
        self._daily_loss = 0.0
        self._daily_reset_date: str = ""

        # New risk subsystems
        self._kelly = KellyCriterion(kelly_config or KellyConfig())
        self._correlation = CorrelationRiskManager(correlation_config or CorrelationConfig())
        self._var = VaRCalculator(var_config or VaRConfig())
        self._liquidity = LiquidityFilter(liquidity_config or LiquidityConfig())

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
        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
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

        # 5. Kelly criterion check
        kelly_size = signal.suggested_size_pct
        kelly_result = self._kelly.calculate_from_trades(
            self._pnl_tracker.trade_history,
            confidence=signal.confidence,
        )
        if kelly_result.is_valid:
            kelly_size = min(signal.suggested_size_pct, kelly_result.capped_size_pct)
            passed.append("kelly")
        else:
            # Kelly not valid (not enough trades or negative edge)
            # Do not reject; just note it
            passed.append("kelly_insufficient_data")

        # 6. Correlation check
        current_pos_map: dict[str, float] = {}
        for p in positions:
            pos_pct = (p.size * p.entry_price) / balance.total * 100 if balance.total > 0 else 0
            current_pos_map[p.symbol] = pos_pct

        corr_result = self._correlation.check_correlation_risk(
            symbol=signal.symbol,
            proposed_size_pct=signal.suggested_size_pct,
            current_positions=current_pos_map,
        )
        corr_multiplier = corr_result.suggested_size_multiplier
        if corr_result.is_acceptable:
            passed.append("correlation")
        else:
            failed.append("correlation")

        # 7. VaR check
        var_result = self._var.calculate(balance.total)
        if var_result.is_within_limit:
            passed.append("var")
        else:
            failed.append("var")
            if var_result.breach_type in ("both", "cvar"):
                return RiskDecision(
                    decision=RiskDecisionType.REJECTED,
                    approved_size_pct=0.0,
                    reason=f"VaR limit breach: VaR={var_result.var_pct:.2f}% CVaR={var_result.cvar_pct:.2f}%",
                    checks_passed=passed,
                    checks_failed=failed,
                )

        # 8. Liquidity check
        liq_result = self._liquidity.check(
            symbol=signal.symbol,
            proposed_size_pct=signal.suggested_size_pct,
            portfolio_value=balance.total,
        )
        liq_multiplier = 1.0
        if liq_result.is_acceptable:
            passed.append("liquidity")
        else:
            failed.append("liquidity")
            liq_multiplier = liq_result.adjusted_size_pct / signal.suggested_size_pct if signal.suggested_size_pct > 0 else 0.0

        # Determine final size incorporating all checks
        approved_size = min(
            signal.suggested_size_pct,
            max_size,
            remaining,
            kelly_size,
        )
        approved_size *= corr_multiplier
        approved_size *= liq_multiplier
        approved_size = max(approved_size, 0.0)

        if not size_ok or not exposure_ok or not corr_result.is_acceptable or not liq_result.is_acceptable:
            if approved_size > 0:
                return RiskDecision(
                    decision=RiskDecisionType.REDUCED,
                    approved_size_pct=round(approved_size, 4),
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
            approved_size_pct=round(approved_size, 4),
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

    @property
    def kelly(self) -> KellyCriterion:
        return self._kelly

    @property
    def correlation_manager(self) -> CorrelationRiskManager:
        return self._correlation

    @property
    def var_calculator(self) -> VaRCalculator:
        return self._var

    @property
    def liquidity_filter(self) -> LiquidityFilter:
        return self._liquidity
