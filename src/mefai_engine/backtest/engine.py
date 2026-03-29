"""Vectorized backtesting engine with realistic market simulation.

Processes entire history as numpy arrays for speed while maintaining
accuracy through proper fee modeling and compounding equity and
liquidation checks and timeframe aware Sharpe calculation.

NOTE ON LOOKAHEAD BIAS: This engine processes pre-computed signal arrays.
The caller is responsible for ensuring signals at index i were generated
using only data available at or before index i. Use the WalkForwardOptimizer
to validate that your signal generation process is free of lookahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import structlog

from mefai_engine.types import BacktestResult

logger = structlog.get_logger()

# Annualization factors by timeframe (bars per year)
_BARS_PER_YEAR: dict[str, float] = {
    "1m": 365.25 * 24 * 60,
    "5m": 365.25 * 24 * 12,
    "15m": 365.25 * 24 * 4,
    "1h": 365.25 * 24,
    "4h": 365.25 * 6,
    "1d": 365.25,
}

# Funding intervals by timeframe (bars between 8h funding payments)
_FUNDING_BARS: dict[str, int] = {
    "1m": 480,
    "5m": 96,
    "15m": 32,
    "1h": 8,
    "4h": 2,
    "1d": 1,  # roughly 3x per day but applied once per bar
}


@dataclass
class BacktestConfig:
    """Backtest parameters."""
    initial_capital: float = 10000.0
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    slippage_base_bps: float = 3.0
    max_position_pct: float = 10.0
    leverage: int = 1
    timeframe: str = "1h"
    liquidation_threshold_pct: float = 90.0  # liquidate when margin loss exceeds this %
    compounding: bool = True  # use current equity for position sizing
    volatility_slippage: bool = True  # scale slippage with volatility


class VectorizedBacktest:
    """Production grade vectorized backtest engine.

    Realistic simulation features:
    1. Notional value based fees (not fixed capital)
    2. Compounding equity (profits grow position sizes)
    3. Liquidation detection (stops trading when margin blown)
    4. Timeframe aware Sharpe/Sortino annualization
    5. Funding rate only applied when position is open
    6. Volatility scaled slippage model
    7. Trade duration tracking
    8. Per bar equity tracking for accurate drawdown

    NOTE ON SURVIVORSHIP BIAS: This engine backtests symbols that exist today.
    Tokens that were delisted or projects that failed during the test period
    are not included. Real performance may differ due to this survivorship
    bias. Always validate with walk forward testing on recent data.
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
        """Run backtest with realistic market simulation.

        Args:
            signals: Array of directions (1=long and -1=short and 0=flat).
                     Signal at index i should be computed using only data up to i.
            prices: Close price array (same length as signals)
            confidences: Optional confidence array for position sizing (0 to 1)
            funding_rates: Optional per bar funding rate array

        Returns:
            BacktestResult with full performance metrics.
        """
        n = len(signals)
        cfg = self._config

        if confidences is None:
            confidences = np.ones(n) * 0.7

        if funding_rates is None:
            funding_rates = np.zeros(n)

        # Resolve timeframe settings
        bars_per_year = _BARS_PER_YEAR.get(cfg.timeframe, 365.25 * 24)
        funding_interval = _FUNDING_BARS.get(cfg.timeframe, 8)

        # Fee and slippage rates
        fee_rate = cfg.taker_fee_bps / 10000

        # Precompute rolling volatility for slippage scaling
        vol_multiplier = np.ones(n)
        if cfg.volatility_slippage and n > 20:
            log_returns = np.log(prices[1:] / prices[:-1])
            log_returns = np.insert(log_returns, 0, 0)
            rolling_vol = np.full(n, np.nan)
            for i in range(20, n):
                rolling_vol[i] = np.std(log_returns[i - 20: i + 1])
            median_vol = np.nanmedian(rolling_vol)
            if median_vol > 0:
                vol_multiplier = np.where(
                    np.isnan(rolling_vol), 1.0,
                    np.clip(rolling_vol / median_vol, 0.5, 5.0)
                )

        # Bar by bar simulation with compounding
        equity = np.zeros(n)
        equity[0] = cfg.initial_capital
        current_equity = cfg.initial_capital
        current_position = 0.0  # in base currency units (e.g. BTC)
        entry_price = 0.0

        total_fees = 0.0
        total_funding = 0.0
        liquidated = False
        liquidation_bar = n

        # Trade tracking
        trade_entries: list[int] = []
        trade_exits: list[int] = []
        trade_pnls: list[float] = []
        trade_pnl_accumulator = 0.0
        in_trade = False
        trade_start_bar = 0

        for i in range(1, n):
            if liquidated:
                equity[i] = equity[i - 1]
                continue

            price = prices[i]
            prev_price = prices[i - 1]

            # Target position in fraction of equity
            target_fraction = signals[i] * confidences[i] * (cfg.max_position_pct / 100) * cfg.leverage

            # Compounding: size based on current equity not initial
            sizing_capital = current_equity if cfg.compounding else cfg.initial_capital
            target_notional = target_fraction * sizing_capital
            target_quantity = target_notional / price if price > 0 else 0

            # Position change
            quantity_change = target_quantity - current_position

            # Fee on notional change (realistic: based on trade value not capital)
            bar_fee = 0.0
            if abs(quantity_change) > 1e-10:
                trade_notional = abs(quantity_change) * price
                slippage_bps = cfg.slippage_base_bps * vol_multiplier[i]
                slippage_rate = slippage_bps / 10000
                bar_fee = trade_notional * (fee_rate + slippage_rate)
                total_fees += bar_fee

            # Mark to market PnL from price movement on existing position
            price_pnl = current_position * (price - prev_price)

            # Funding cost: only when position is open and at funding interval
            bar_funding = 0.0
            if abs(current_position) > 1e-10 and i % funding_interval == 0:
                position_notional = abs(current_position) * price
                bar_funding = position_notional * funding_rates[i]
                total_funding += abs(bar_funding)

            # Update equity
            current_equity += price_pnl - bar_fee - abs(bar_funding)
            equity[i] = current_equity

            # Liquidation check
            if current_equity <= cfg.initial_capital * (1 - cfg.liquidation_threshold_pct / 100):
                liquidated = True
                liquidation_bar = i
                # Close any open trade
                if in_trade:
                    trade_pnls.append(trade_pnl_accumulator + price_pnl - bar_fee)
                    trade_exits.append(i)
                    in_trade = False
                logger.warning(
                    "backtest.liquidation",
                    bar=i,
                    equity=round(current_equity, 2),
                    threshold=cfg.initial_capital * (1 - cfg.liquidation_threshold_pct / 100),
                )
                continue

            # Trade tracking
            bar_net = price_pnl - bar_fee - abs(bar_funding)
            prev_signal = signals[i - 1] if i > 0 else 0

            # Direction change: close old trade and open new
            if in_trade and signals[i] != 0 and prev_signal != 0 and signals[i] != prev_signal:
                trade_pnls.append(trade_pnl_accumulator)
                trade_exits.append(i)
                in_trade = False
                trade_pnl_accumulator = 0.0

            if signals[i] != 0 and not in_trade:
                # Open new trade
                in_trade = True
                trade_start_bar = i
                trade_entries.append(i)
                trade_pnl_accumulator = bar_net
                entry_price = price
            elif signals[i] != 0 and in_trade:
                # Continue trade
                trade_pnl_accumulator += bar_net
            elif signals[i] == 0 and in_trade:
                # Close trade
                trade_pnls.append(trade_pnl_accumulator)
                trade_exits.append(i)
                in_trade = False
                trade_pnl_accumulator = 0.0

            # Update position
            current_position = target_quantity

        # Close any remaining open trade
        if in_trade:
            trade_pnls.append(trade_pnl_accumulator)
            trade_exits.append(n - 1)

        # Calculate trade durations in hours
        timeframe_hours = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "1h": 1, "4h": 4, "1d": 24}
        hours_per_bar = timeframe_hours.get(cfg.timeframe, 1)
        trade_durations = []
        for entry_idx, exit_idx in zip(trade_entries, trade_exits):
            duration = (exit_idx - entry_idx) * hours_per_bar
            trade_durations.append(duration)

        avg_duration = float(np.mean(trade_durations)) if trade_durations else 0.0

        # Metrics
        total_trades = len(trade_pnls)
        winners = [t for t in trade_pnls if t > 0]
        losers = [t for t in trade_pnls if t < 0]
        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 1e-10
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        total_return = (equity[-1] - cfg.initial_capital) / cfg.initial_capital * 100

        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdown = (running_max - equity) / np.where(running_max > 0, running_max, 1) * 100
        max_drawdown = float(np.max(drawdown))

        # Sharpe ratio (annualized with correct timeframe factor)
        bar_returns = np.diff(equity) / np.where(equity[:-1] > 0, equity[:-1], 1)
        annualization = np.sqrt(bars_per_year)

        sharpe = 0.0
        if len(bar_returns) > 1 and np.std(bar_returns) > 0:
            sharpe = float(np.mean(bar_returns) / np.std(bar_returns) * annualization)

        # Sortino ratio
        downside = bar_returns[bar_returns < 0]
        sortino = 0.0
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = float(np.mean(bar_returns) / np.std(downside) * annualization)

        calmar = total_return / max_drawdown if max_drawdown > 0 else 0.0
        recovery = total_return / max_drawdown if max_drawdown > 0 else 0.0

        avg_winner = float(np.mean(winners)) if winners else 0.0
        avg_loser = float(np.mean(np.abs(losers))) if losers else 0.0
        expectancy = float(np.mean(trade_pnls)) if trade_pnls else 0.0

        final_equity = equity[-1]
        avg_winner_pct = round(avg_winner / final_equity * 100, 4) if final_equity > 0 and avg_winner else 0.0
        avg_loser_pct = round(avg_loser / final_equity * 100, 4) if final_equity > 0 and avg_loser else 0.0

        result = BacktestResult(
            total_return_pct=round(total_return, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            calmar_ratio=round(calmar, 3),
            max_drawdown_pct=round(max_drawdown, 2),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 3),
            total_trades=total_trades,
            avg_trade_duration_hours=round(avg_duration, 2),
            avg_winner_pct=avg_winner_pct,
            avg_loser_pct=avg_loser_pct,
            expectancy=round(expectancy, 2),
            recovery_factor=round(recovery, 3),
            start_date=datetime.now(tz=UTC),
            end_date=datetime.now(tz=UTC),
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
            total_fees=f"{total_fees:.2f}",
            total_funding=f"{total_funding:.2f}",
            avg_duration_h=f"{avg_duration:.1f}",
            liquidated=liquidated,
            compounding=cfg.compounding,
            timeframe=cfg.timeframe,
        )

        return result
