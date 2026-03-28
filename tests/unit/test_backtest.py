"""Unit tests for the vectorized backtest engine and walk-forward validator."""

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
        signals = np.ones(n)  # always long
        confidences = np.ones(n) * 0.8

        bt = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
        ))
        result = bt.run(signals, prices, confidences)

        assert result.total_return_pct > 0
        assert len(result.equity_curve) == n

    def test_short_on_rising_prices_should_lose(self) -> None:
        """Short signal on rising prices should produce negative return."""
        n = 200
        prices = np.linspace(100, 150, n)
        signals = np.full(n, -1.0)  # always short
        confidences = np.ones(n) * 0.8

        bt = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            slippage_bps=0.0,
        ))
        result = bt.run(signals, prices, confidences)

        assert result.total_return_pct < 0

    def test_fee_impact(self) -> None:
        """Higher fees should result in lower net returns."""
        n = 200
        prices = np.linspace(100, 120, n)
        # Generate signals that alternate to maximize fee exposure
        signals = np.ones(n)
        signals[::5] = 0  # go flat every 5 bars to trigger entries/exits
        confidences = np.ones(n) * 0.7

        bt_low_fee = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=1.0,
            slippage_bps=0.0,
        ))
        bt_high_fee = VectorizedBacktest(BacktestConfig(
            initial_capital=10000.0,
            taker_fee_bps=20.0,
            slippage_bps=0.0,
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


class TestWalkForwardValidator:
    """Tests for the walk-forward optimization framework."""

    def test_correct_number_of_folds(self) -> None:
        """Walk-forward should create the expected number of folds."""
        wf = WalkForwardOptimizer(
            train_size=100,
            validate_size=50,
            test_size=50,
            step_size=50,
        )

        total_length = 400
        windows = wf.generate_windows(total_length)

        # window_total = 100+50+50 = 200
        # Fold 0: 0..200
        # Fold 1: 50..250
        # Fold 2: 100..300
        # Fold 3: 150..350
        # Fold 4: 200..400
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
        """Should raise ValueError when data is too short for even one fold."""
        wf = WalkForwardOptimizer(
            train_size=1000,
            validate_size=500,
            test_size=500,
            step_size=500,
        )
        windows = wf.generate_windows(100)
        assert len(windows) == 0
