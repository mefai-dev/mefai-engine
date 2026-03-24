"""Performance statistics and metrics calculation."""

from __future__ import annotations

import numpy as np

from mefai_engine.types import BacktestResult


def format_report(result: BacktestResult) -> str:
    """Format backtest result as readable text report."""
    pnl_prefix = "+" if result.total_return_pct >= 0 else ""

    return f"""
=====================================
  MEFAI Engine Backtest Report
=====================================

RETURNS
  Total Return:     {pnl_prefix}{result.total_return_pct:.2f}%
  Sharpe Ratio:     {result.sharpe_ratio:.3f}
  Sortino Ratio:    {result.sortino_ratio:.3f}
  Calmar Ratio:     {result.calmar_ratio:.3f}

RISK
  Max Drawdown:     {result.max_drawdown_pct:.2f}%
  Recovery Factor:  {result.recovery_factor:.3f}

TRADES
  Total Trades:     {result.total_trades}
  Win Rate:         {result.win_rate:.1%}
  Profit Factor:    {result.profit_factor:.3f}
  Expectancy:       ${result.expectancy:.2f}
  Avg Winner:       {result.avg_winner_pct:.4f}%
  Avg Loser:        {result.avg_loser_pct:.4f}%

=====================================
""".strip()


def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 365) -> float:
    """Calculate annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods
    if len(excess) < 2 or np.std(excess) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(excess) * np.sqrt(periods))


def calculate_sortino(returns: np.ndarray, risk_free_rate: float = 0.0, periods: int = 365) -> float:
    """Calculate annualized Sortino ratio."""
    excess = returns - risk_free_rate / periods
    downside = returns[returns < 0]
    if len(downside) < 1 or np.std(downside) == 0:
        return 0.0
    return float(np.mean(excess) / np.std(downside) * np.sqrt(periods))


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown percentage."""
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / np.where(running_max > 0, running_max, 1) * 100
    return float(np.max(drawdowns))
