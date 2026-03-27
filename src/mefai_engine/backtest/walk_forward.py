"""Walk-forward backtest optimizer for out-of-sample validation.

Splits data into sliding train/validate/test windows that move forward
in time. This prevents lookahead bias by ensuring the model never sees
future data during training or parameter selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Protocol

import numpy as np
import structlog

from mefai_engine.backtest.engine import BacktestConfig, VectorizedBacktest
from mefai_engine.types import BacktestResult

logger = structlog.get_logger()


@dataclass
class WalkForwardWindow:
    """A single walk-forward fold with its index boundaries."""
    fold_index: int
    train_start: int
    train_end: int
    validate_start: int
    validate_end: int
    test_start: int
    test_end: int


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""
    fold_index: int
    window: WalkForwardWindow
    train_result: BacktestResult
    validate_result: BacktestResult
    test_result: BacktestResult
    best_params: dict[str, Any] = dc_field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregate result from all walk-forward folds."""
    fold_results: list[FoldResult]
    aggregate_sharpe: float
    aggregate_return_pct: float
    aggregate_max_drawdown_pct: float
    aggregate_win_rate: float
    aggregate_profit_factor: float
    total_folds: int
    oos_consistency: float  # fraction of folds with positive OOS return


class SignalGenerator(Protocol):
    """Protocol for signal generation callables used in walk-forward."""

    def __call__(
        self,
        train_features: np.ndarray,
        train_prices: np.ndarray,
        test_features: np.ndarray,
        test_prices: np.ndarray,
        params: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train on training data and generate signals for test data.

        Returns:
            Tuple of (signals_array and confidences_array) for test period.
        """
        ...


class WalkForwardOptimizer:
    """Walk-forward validation engine.

    Splits historical data into rolling windows:
    [--- train ---|--- validate ---|--- test ---]
                  [--- train ---|--- validate ---|--- test ---]
                                [--- train ---|--- validate ---|--- test ---]

    Each fold trains on window N then validates on N+1 and tests on N+2.
    Prevents lookahead bias and measures true out-of-sample performance.
    """

    def __init__(
        self,
        train_size: int = 5000,
        validate_size: int = 1000,
        test_size: int = 1000,
        step_size: int = 1000,
        backtest_config: BacktestConfig | None = None,
    ) -> None:
        self._train_size = train_size
        self._validate_size = validate_size
        self._test_size = test_size
        self._step_size = step_size
        self._bt_config = backtest_config or BacktestConfig()

    def generate_windows(self, total_length: int) -> list[WalkForwardWindow]:
        """Generate all walk-forward windows for the given data length."""
        windows: list[WalkForwardWindow] = []
        fold = 0
        start = 0

        window_total = self._train_size + self._validate_size + self._test_size

        while start + window_total <= total_length:
            train_start = start
            train_end = start + self._train_size
            val_start = train_end
            val_end = val_start + self._validate_size
            test_start = val_end
            test_end = test_start + self._test_size

            windows.append(WalkForwardWindow(
                fold_index=fold,
                train_start=train_start,
                train_end=train_end,
                validate_start=val_start,
                validate_end=val_end,
                test_start=test_start,
                test_end=test_end,
            ))

            fold += 1
            start += self._step_size

        logger.info(
            "walk_forward.windows_generated",
            total_folds=len(windows),
            data_length=total_length,
        )
        return windows

    def run(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        signal_generator: SignalGenerator,
        param_grid: list[dict[str, Any]] | None = None,
        funding_rates: np.ndarray | None = None,
    ) -> WalkForwardResult:
        """Run full walk-forward optimization.

        Args:
            features: Feature matrix (n_samples x n_features)
            prices: Price array (n_samples)
            signal_generator: Callable that trains and generates signals
            param_grid: Optional parameter combinations to search
            funding_rates: Optional funding rate array

        Returns:
            WalkForwardResult with per-fold and aggregate metrics.
        """
        windows = self.generate_windows(len(prices))
        if not windows:
            raise ValueError(
                "Not enough data for walk-forward validation. "
                f"Need at least {self._train_size + self._validate_size + self._test_size} "
                f"samples but got {len(prices)}"
            )

        bt = VectorizedBacktest(self._bt_config)
        fold_results: list[FoldResult] = []

        for window in windows:
            logger.info(
                "walk_forward.fold_start",
                fold=window.fold_index,
                train=f"{window.train_start}:{window.train_end}",
                test=f"{window.test_start}:{window.test_end}",
            )

            # Slice data for this fold
            train_feat = features[window.train_start:window.train_end]
            train_prices = prices[window.train_start:window.train_end]
            val_feat = features[window.validate_start:window.validate_end]
            val_prices = prices[window.validate_start:window.validate_end]
            test_feat = features[window.test_start:window.test_end]
            test_prices = prices[window.test_start:window.test_end]

            train_funding = None
            val_funding = None
            test_funding = None
            if funding_rates is not None:
                train_funding = funding_rates[window.train_start:window.train_end]
                val_funding = funding_rates[window.validate_start:window.validate_end]
                test_funding = funding_rates[window.test_start:window.test_end]

            # Parameter optimization on validation set
            best_params: dict[str, Any] = {}
            best_val_sharpe = -float("inf")

            candidates = param_grid if param_grid else [{}]
            for params in candidates:
                val_signals, val_confs = signal_generator(
                    train_feat, train_prices, val_feat, val_prices, params
                )
                val_result = bt.run(val_signals, val_prices, val_confs, val_funding)
                if val_result.sharpe_ratio > best_val_sharpe:
                    best_val_sharpe = val_result.sharpe_ratio
                    best_params = params

            # Generate signals for all three windows with best params
            train_signals, train_confs = signal_generator(
                train_feat, train_prices, train_feat, train_prices, best_params
            )
            train_result = bt.run(train_signals, train_prices, train_confs, train_funding)

            val_signals, val_confs = signal_generator(
                train_feat, train_prices, val_feat, val_prices, best_params
            )
            validate_result = bt.run(val_signals, val_prices, val_confs, val_funding)

            test_signals, test_confs = signal_generator(
                train_feat, train_prices, test_feat, test_prices, best_params
            )
            test_result = bt.run(test_signals, test_prices, test_confs, test_funding)

            fold_results.append(FoldResult(
                fold_index=window.fold_index,
                window=window,
                train_result=train_result,
                validate_result=validate_result,
                test_result=test_result,
                best_params=best_params,
            ))

            logger.info(
                "walk_forward.fold_complete",
                fold=window.fold_index,
                train_sharpe=train_result.sharpe_ratio,
                val_sharpe=validate_result.sharpe_ratio,
                test_sharpe=test_result.sharpe_ratio,
                test_return=f"{test_result.total_return_pct:.2f}%",
            )

        return self._aggregate_results(fold_results)

    @staticmethod
    def _aggregate_results(fold_results: list[FoldResult]) -> WalkForwardResult:
        """Compute aggregate statistics across all folds."""
        test_sharpes = [f.test_result.sharpe_ratio for f in fold_results]
        test_returns = [f.test_result.total_return_pct for f in fold_results]
        test_drawdowns = [f.test_result.max_drawdown_pct for f in fold_results]
        test_win_rates = [f.test_result.win_rate for f in fold_results]
        test_profit_factors = [
            f.test_result.profit_factor
            for f in fold_results
            if f.test_result.profit_factor < float("inf")
        ]

        positive_folds = sum(1 for r in test_returns if r > 0)
        consistency = positive_folds / len(fold_results) if fold_results else 0.0

        agg_pf = float(np.mean(test_profit_factors)) if test_profit_factors else 0.0

        result = WalkForwardResult(
            fold_results=fold_results,
            aggregate_sharpe=round(float(np.mean(test_sharpes)), 3),
            aggregate_return_pct=round(float(np.mean(test_returns)), 2),
            aggregate_max_drawdown_pct=round(float(np.max(test_drawdowns)), 2),
            aggregate_win_rate=round(float(np.mean(test_win_rates)), 4),
            aggregate_profit_factor=round(agg_pf, 3),
            total_folds=len(fold_results),
            oos_consistency=round(consistency, 4),
        )

        logger.info(
            "walk_forward.complete",
            folds=result.total_folds,
            avg_sharpe=result.aggregate_sharpe,
            avg_return=f"{result.aggregate_return_pct:.2f}%",
            consistency=f"{result.oos_consistency:.1%}",
        )

        return result
