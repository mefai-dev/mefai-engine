"""Unit tests for the risk management subsystem."""

from __future__ import annotations

import pytest

from mefai_engine.constants import CircuitState
from mefai_engine.risk.circuit_breaker import TradingCircuitBreaker
from mefai_engine.risk.kelly import KellyConfig, KellyCriterion
from mefai_engine.risk.limits import (
    check_daily_loss,
    check_drawdown,
    check_position_size,
)
from mefai_engine.risk.pnl_tracker import PnLTracker


class TestPnLTracker:
    """Tests for PnL tracking and equity metrics."""

    def test_record_wins_and_losses(self) -> None:
        """Win rate should reflect recorded wins and losses."""
        tracker = PnLTracker(initial_equity=10000.0)
        tracker.record(100.0)
        tracker.record(200.0)
        tracker.record(-50.0)
        tracker.record(150.0)
        tracker.record(-80.0)

        assert tracker.total_trades == 5
        assert tracker.win_rate == pytest.approx(3 / 5)

    def test_profit_factor(self) -> None:
        """Profit factor should be gross profit divided by gross loss."""
        tracker = PnLTracker()
        tracker.record(300.0)
        tracker.record(-100.0)
        tracker.record(200.0)
        tracker.record(-100.0)

        # gross_profit = 500 / gross_loss = 200
        assert tracker.profit_factor == pytest.approx(2.5)

    def test_total_pnl(self) -> None:
        """Total PnL should be the sum of all recorded trades."""
        tracker = PnLTracker()
        tracker.record(100.0)
        tracker.record(-30.0)
        tracker.record(50.0)
        assert tracker.total_pnl == pytest.approx(120.0)

    def test_no_trades_defaults(self) -> None:
        """Win rate and profit factor should handle zero trades gracefully."""
        tracker = PnLTracker()
        assert tracker.win_rate == 0.0
        assert tracker.profit_factor == float("inf")
        assert tracker.total_trades == 0


class TestCircuitBreaker:
    """Tests for the trading circuit breaker."""

    def test_trip_after_consecutive_losses(self) -> None:
        """Circuit breaker should open after N consecutive losses."""
        cb = TradingCircuitBreaker(max_consecutive_losses=3, cooldown_seconds=10)

        assert cb.state == CircuitState.CLOSED
        cb.record_loss()
        cb.record_loss()
        assert cb.state == CircuitState.CLOSED
        cb.record_loss()
        assert cb.state == CircuitState.OPEN

    def test_win_resets_consecutive_count(self) -> None:
        """A win should reset the consecutive loss counter."""
        cb = TradingCircuitBreaker(max_consecutive_losses=3, cooldown_seconds=10)

        cb.record_loss()
        cb.record_loss()
        cb.record_win()
        cb.record_loss()
        cb.record_loss()
        # Should still be closed because win reset the counter
        assert cb.state == CircuitState.CLOSED

    def test_cannot_trade_when_open(self) -> None:
        """Trading should be blocked while circuit breaker is OPEN."""
        cb = TradingCircuitBreaker(max_consecutive_losses=2, cooldown_seconds=99999)
        cb.record_loss()
        cb.record_loss()
        assert cb.state == CircuitState.OPEN
        assert cb.can_trade() is False

    def test_cooldown_recovery(self) -> None:
        """After cooldown the breaker should transition to HALF_OPEN."""
        cb = TradingCircuitBreaker(max_consecutive_losses=2, cooldown_seconds=0)
        cb.record_loss()
        cb.record_loss()
        assert cb.state == CircuitState.OPEN

        # Cooldown is 0 seconds so next check should transition
        assert cb.can_trade() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_recovery_on_win(self) -> None:
        """A win during HALF_OPEN should close the breaker."""
        cb = TradingCircuitBreaker(max_consecutive_losses=2, cooldown_seconds=0)
        cb.record_loss()
        cb.record_loss()
        cb.can_trade()  # transitions to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_win()
        assert cb.state == CircuitState.CLOSED


class TestRiskLimits:
    """Tests for pure risk limit check functions."""

    def test_position_size_within_limit(self) -> None:
        """Should approve position size within the max."""
        ok, size = check_position_size(5.0, 10.0)
        assert ok is True
        assert size == 5.0

    def test_position_size_exceeds_limit(self) -> None:
        """Should reject and cap position size above the max."""
        ok, size = check_position_size(15.0, 10.0)
        assert ok is False
        assert size == 10.0

    def test_daily_loss_within_limit(self) -> None:
        """Daily loss check should pass when below threshold."""
        result = check_daily_loss(200.0, 10000.0, 3.0)
        assert result is True

    def test_daily_loss_exceeded(self) -> None:
        """Daily loss check should fail when above threshold."""
        result = check_daily_loss(400.0, 10000.0, 3.0)
        assert result is False

    def test_drawdown_within_limit(self) -> None:
        """Drawdown check should pass when below max."""
        result = check_drawdown(5.0, 10.0)
        assert result is True

    def test_drawdown_exceeded(self) -> None:
        """Drawdown check should fail when at or above max."""
        result = check_drawdown(12.0, 10.0)
        assert result is False


class TestKellyCriterion:
    """Tests for Kelly criterion position sizing."""

    def test_kelly_known_values(self) -> None:
        """Kelly with known win rate and payoff should return expected size."""
        kelly = KellyCriterion(KellyConfig(
            fraction=1.0,
            max_position_pct=100.0,
            min_win_rate=0.0,
            min_trades=0,
            confidence_scaling=False,
        ))

        # win_rate=0.6 avg_winner=100 avg_loser=100 => b=1
        # full_kelly = (0.6*1 - 0.4)/1 = 0.2 => 20%
        result = kelly.calculate(
            win_rate=0.6,
            avg_winner=100.0,
            avg_loser=100.0,
            n_trades=100,
        )
        assert result.is_valid is True
        assert result.full_kelly_pct == pytest.approx(20.0, abs=0.01)

    def test_kelly_negative_edge(self) -> None:
        """Kelly should return zero size when edge is negative."""
        kelly = KellyCriterion(KellyConfig(
            fraction=0.25,
            min_win_rate=0.0,
            min_trades=0,
        ))
        result = kelly.calculate(
            win_rate=0.3,
            avg_winner=100.0,
            avg_loser=100.0,
            n_trades=100,
        )
        assert result.is_valid is False
        assert result.capped_size_pct == 0.0

    def test_kelly_insufficient_trades(self) -> None:
        """Kelly should reject when trade count is below minimum."""
        kelly = KellyCriterion(KellyConfig(min_trades=30))
        result = kelly.calculate(
            win_rate=0.6,
            avg_winner=150.0,
            avg_loser=100.0,
            n_trades=10,
        )
        assert result.is_valid is False
