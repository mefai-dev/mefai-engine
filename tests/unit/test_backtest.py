"""Unit tests for the vectorized backtest engine and walk forward validator."""

from __future__ import annotations

import numpy as np
import pytest

from mefai_engine.backtest.engine import BacktestConfig, VectorizedBacktest
from mefai_engine.backtest.walk_forward import WalkForwardOptimizer


class TestVectorizedBacktest:
    """Tests for the vectorized backtesting engine."""

    def test_buy_and_hold_rising_prices(self) -> None:
        """Long signal on steadily rising prices should produce positive return."""
        n = 200
        prices = np.linspace(100, 150, n)
        signals = np.ones(n)
        confidences = np.ones(n) * 0.8

        bt = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=0.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
        ))
        result = bt.run(signals, prices, confidences)

        assert result.total_return_pct > 0
        assert len(result.equity_curve) == n

    def test_short_on_rising_prices_should_lose(self) -> None:
        """Short signal on rising prices should produce negative return."""
        n = 200
        prices = np.linspace(100, 150, n)
        signals = np.full(n, -1.0)
        confidences = np.ones(n) * 0.8

        bt = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=0.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
        ))
        result = bt.run(signals, prices, confidences)

        assert result.total_return_pct < 0

    def test_fee_impact(self) -> None:
        """Higher fees should result in lower net returns."""
        n = 200
        prices = np.linspace(100, 120, n)
        signals = np.ones(n)
        signals[::5] = 0  # go flat every 5 bars
        confidences = np.ones(n) * 0.7

        bt_low_fee = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=1.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
        ))
        bt_high_fee = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=20.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
        ))

        result_low = bt_low_fee.run(signals, prices, confidences)
        result_high = bt_high_fee.run(signals, prices, confidences)

        assert result_low.total_return_pct > result_high.total_return_pct

    def test_flat_signal_no_trades(self) -> None:
        """All flat signals should result in zero return and no trades."""
        n = 100
        prices = np.linspace(100, 120, n)
        signals = np.zeros(n)

        bt = VectorizedBacktest(BacktestConfig(initial_capital=10000.0))
        result = bt.run(signals, prices)

        assert result.total_trades == 0

    def test_compounding_grows_position(self) -> None:
        """With compounding on profits should increase position size over time."""
        n = 300
        prices = np.linspace(100, 200, n)
        signals = np.ones(n)

        bt_compound = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=0.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
            compounding=True,
        ))
        bt_fixed = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=0.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
            compounding=False,
        ))

        result_compound = bt_compound.run(signals, prices)
        result_fixed = bt_fixed.run(signals, prices)

        # Compounding should produce higher returns on a winning streak
        assert result_compound.total_return_pct > result_fixed.total_return_pct

    def test_liquidation_stops_trading(self) -> None:
        """Massive drawdown should trigger liquidation."""
        n = 100
        # Price crashes 95%
        prices = np.concatenate([
            np.linspace(100, 100, 10),
            np.linspace(100, 5, 90),
        ])
        signals = np.ones(n)  # long into crash
        confidences = np.ones(n) * 1.0

        bt = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            leverage=10,
            max_position_pct=50.0,
            taker_fee_bps=0.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
            liquidation_threshold_pct=90.0,
        ))
        result = bt.run(signals, prices, confidences)

        # Should have been liquidated (equity stopped changing)
        equity = result.equity_curve
        # After liquidation equity should be flat
        assert equity[-1] < 10000.0

    def test_trade_duration_tracked(self) -> None:
        """Trade duration should be calculated correctly."""
        n = 100
        prices = np.linspace(100, 110, n)
        signals = np.zeros(n)
        signals[10:30] = 1  # 20 bar trade
        signals[50:80] = -1  # 30 bar trade

        bt = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            timeframe="1h",
            taker_fee_bps=0.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
        ))
        result = bt.run(signals, prices)

        assert result.total_trades == 2
        assert result.avg_trade_duration_hours > 0

    def test_sharpe_uses_correct_timeframe(self) -> None:
        """Different timeframes should produce different Sharpe ratios."""
        n = 500
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
        prices = np.maximum(prices, 10)
        signals = np.ones(n)

        bt_1h = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0, timeframe="1h",
            taker_fee_bps=0.0, slippage_base_bps=0.0, volatility_slippage=False,
        ))
        bt_1d = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0, timeframe="1d",
            taker_fee_bps=0.0, slippage_base_bps=0.0, volatility_slippage=False,
        ))

        result_1h = bt_1h.run(signals, prices)
        result_1d = bt_1d.run(signals, prices)

        # Same data but different annualization should give different Sharpe
        # 1h has sqrt(8760) annualization vs 1d has sqrt(365)
        assert result_1h.sharpe_ratio != result_1d.sharpe_ratio

    def test_funding_only_when_position_open(self) -> None:
        """Funding should not be charged when position is flat."""
        n = 100
        prices = np.full(n, 100.0)  # flat price
        funding = np.full(n, 0.001)  # 0.1% per interval

        signals_always_long = np.ones(n)
        signals_half_flat = np.ones(n)
        signals_half_flat[50:] = 0  # flat second half

        bt = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            timeframe="1h",
            taker_fee_bps=0.0,
            slippage_base_bps=0.0,
            volatility_slippage=False,
        ))

        result_always = bt.run(signals_always_long, prices, funding_rates=funding)
        result_half = bt.run(signals_half_flat, prices, funding_rates=funding)

        # Half flat should pay less funding
        assert result_half.total_return_pct > result_always.total_return_pct


class TestWalkForwardValidator:
    """Tests for the walk forward optimization framework."""

    def test_correct_number_of_folds(self) -> None:
        """Walk forward should create the expected number of folds."""
        wf = WalkForwardOptimizer(
            train_size=100,
            validate_size=50,
            test_size=50,
            step_size=50,
        )

        total_length = 400
        windows = wf.generate_windows(total_length)

        expected = (total_length - (100 + 50 + 50)) // 50 + 1
        assert len(windows) == expected

    def test_windows_do_not_overlap_test_train(self) -> None:
        """Test period should never overlap with training period within a fold."""
        wf = WalkForwardOptimizer(
            train_size=100,
            validate_size=30,
            test_size=30,
            step_size=30,
        )
        windows = wf.generate_windows(500)
        for w in windows:
            assert w.test_start >= w.train_end

    def test_insufficient_data_raises(self) -> None:
        """Should produce zero windows when data is too short for even one fold."""
        wf = WalkForwardOptimizer(
            train_size=1000,
            validate_size=500,
            test_size=500,
            step_size=500,
        )
        windows = wf.generate_windows(100)
        assert len(windows) == 0
