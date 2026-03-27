"""Vectorized backtesting engine for maximum speed.

Processes entire history as numpy arrays. Uses the same feature pipeline
and strategy logic as live trading for consistency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

from mefai_engine.constants import Direction
from mefai_engine.types import BacktestResult

logger = structlog.get_logger()


@dataclass
class BacktestConfig:
    """Backtest parameters."""
    initial_capital: float = 10000.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    slippage_bps: float = 3.0
    max_position_pct: float = 10.0
    leverage: int = 1
    funding_rate_interval: int = 480  # candles between funding payments (8h for 1m candles)


class VectorizedBacktest:
    """Ultra-fast vectorized backtest engine.

    All operations use numpy arrays. No Python loops in the hot path.
    Simulates: fees (maker/taker) + slippage + funding rates.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(
        self,
        signals: np.ndarray,
        prices: np.ndarray,
        confidences: np.ndarray | None = None,
        funding_rates: np.ndarray | None = None,
    ) -> BacktestResult:
        """Run backtest on signal array.

        Args:
            signals: Array of directions (1=long, -1=short, 0=flat)
            prices: Close price array (same length as signals)
            confidences: Optional confidence array for position sizing
            funding_rates: Optional funding rate array

        Returns:
            BacktestResult with full performance metrics.
        """
        n = len(signals)
        cfg = self._config

        if confidences is None:
            confidences = np.ones(n) * 0.7

        if funding_rates is None:
            funding_rates = np.zeros(n)

        # Position sizing: confidence * max_position_pct / 100
        position_sizes = signals * confidences * (cfg.max_position_pct / 100) * cfg.leverage

        # Detect position changes for fee calculation
        position_changes = np.diff(position_sizes, prepend=0)
        has_change = np.abs(position_changes) > 0.0001

        # Returns from holding position (divide by previous price not current)
        prev_prices = np.roll(prices, 1)
        prev_prices[0] = prices[0]
        price_returns = np.diff(prices, prepend=prices[0]) / np.where(prev_prices > 0, prev_prices, 1)

        # PnL per bar
        gross_pnl = position_sizes * price_returns * cfg.initial_capital

        # Fees on position changes
        fee_rate = cfg.taker_fee_bps / 10000
        slippage_rate = cfg.slippage_bps / 10000
        fees = np.abs(position_changes) * (fee_rate + slippage_rate) * cfg.initial_capital
        fees = np.where(has_change, fees, 0)

        # Funding costs
        funding_costs = np.zeros(n)
        for i in range(0, n, cfg.funding_rate_interval):
            if i < n:
                funding_costs[i] = np.abs(position_sizes[i]) * funding_rates[i] * cfg.initial_capital

        # Net PnL
        net_pnl = gross_pnl - fees - funding_costs

        # Equity curve
        equity = np.cumsum(net_pnl) + cfg.initial_capital

        # Drawdown calculation
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / np.where(running_max > 0, running_max, 1) * 100
        max_drawdown = float(np.max(drawdown))

        # Trade analysis
        trades = self._extract_trades(signals, prices, net_pnl)
        total_trades = len(trades)
        winners = [t for t in trades if t > 0]
        losers = [t for t in trades if t < 0]
        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_return = (equity[-1] - cfg.initial_capital) / cfg.initial_capital * 100

        # Sharpe ratio (annualized)
        daily_returns = np.diff(equity) / equity[:-1]
        sharpe = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365))

        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        sortino = 0.0
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = float(np.mean(daily_returns) / np.std(downside) * np.sqrt(365))

        # Calmar ratio
        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0

        # Recovery factor
        recovery = total_return / max_drawdown if max_drawdown > 0 else 0.0

        avg_winner = float(np.mean(winners)) if winners else 0.0
        avg_loser = float(np.mean(np.abs(losers))) if losers else 0.0
        expectancy = float(np.mean(trades)) if trades else 0.0

        result = BacktestResult(
            total_return_pct=round(total_return, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown_pct=round(max_drawdown, 2),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 3),
            total_trades=total_trades,
            avg_trade_duration_hours=0.0,  # Requires timestamp tracking
            avg_winner_pct=round(avg_winner / cfg.initial_capital * 100, 4) if avg_winner else 0.0,
            avg_loser_pct=round(avg_loser / cfg.initial_capital * 100, 4) if avg_loser else 0.0,
            expectancy=round(expectancy, 2),
            recovery_factor=round(recovery, 3),
            start_date=datetime.now(tz=timezone.utc),
            end_date=datetime.now(tz=timezone.utc),
            equity_curve=equity.tolist(),
        )

        logger.info(
            "backtest.complete",
            total_return=f"{total_return:.2f}%",
            sharpe=f"{sharpe:.3f}",
            max_dd=f"{max_drawdown:.2f}%",
            trades=total_trades,
            win_rate=f"{win_rate:.1%}",
            profit_factor=f"{profit_factor:.2f}",
        )

        return result

    @staticmethod
    def _extract_trades(
        signals: np.ndarray, prices: np.ndarray, pnl: np.ndarray
    ) -> list[float]:
        """Extract individual trade PnLs from signal array."""
        trades: list[float] = []
        in_trade = False
        trade_pnl = 0.0

        for i in range(1, len(signals)):
            # Direction change = close old trade BEFORE accumulating new pnl
            if (
                in_trade
                and signals[i] != 0
                and signals[i - 1] != 0
                and signals[i] != signals[i - 1]
            ):
                trades.append(trade_pnl)
                trade_pnl = pnl[i]
                continue

            if signals[i] != 0 and not in_trade:
                in_trade = True
                trade_pnl = pnl[i]
            elif signals[i] != 0 and in_trade:
                trade_pnl += pnl[i]
            elif signals[i] == 0 and in_trade:
                trades.append(trade_pnl)
                in_trade = False
                trade_pnl = 0.0

        if in_trade:
            trades.append(trade_pnl)

        return trades
