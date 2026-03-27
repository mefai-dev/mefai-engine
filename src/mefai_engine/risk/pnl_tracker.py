"""Real-time P&L and equity curve tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TradeRecord:
    pnl: float
    timestamp: datetime


class PnLTracker:
    """Tracks realized P&L, equity curve, and drawdown metrics."""

    def __init__(self, initial_equity: float = 0.0) -> None:
        self._trades: list[TradeRecord] = []
        self._equity_curve: list[float] = [initial_equity] if initial_equity > 0 else []
        self._peak_equity: float = initial_equity
        self._total_pnl: float = 0.0
        self._win_count: int = 0
        self._loss_count: int = 0
        self._gross_profit: float = 0.0
        self._gross_loss: float = 0.0

    def record(self, pnl: float) -> None:
        """Record a closed trade's P&L."""
        self._trades.append(TradeRecord(pnl=pnl, timestamp=datetime.now(tz=timezone.utc)))
        self._total_pnl += pnl

        if pnl > 0:
            self._win_count += 1
            self._gross_profit += pnl
        elif pnl < 0:
            self._loss_count += 1
            self._gross_loss += abs(pnl)

    def update_equity(self, equity: float) -> None:
        """Update equity curve with current portfolio value."""
        self._equity_curve.append(equity)
        if equity > self._peak_equity:
            self._peak_equity = equity

    @property
    def total_pnl(self) -> float:
        return self._total_pnl

    @property
    def total_trades(self) -> int:
        return self._win_count + self._loss_count

    @property
    def win_rate(self) -> float:
        total = self.total_trades
        return self._win_count / total if total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        return self._gross_profit / self._gross_loss if self._gross_loss > 0 else float("inf")

    @property
    def current_drawdown_pct(self) -> float:
        if not self._equity_curve or self._peak_equity <= 0:
            return 0.0
        current = self._equity_curve[-1]
        return (self._peak_equity - current) / self._peak_equity * 100

    @property
    def max_drawdown_pct(self) -> float:
        if len(self._equity_curve) < 2:
            return 0.0
        peak = self._equity_curve[0]
        max_dd = 0.0
        for eq in self._equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def expectancy(self) -> float:
        """Average P&L per trade."""
        total = self.total_trades
        return self._total_pnl / total if total > 0 else 0.0

    @property
    def avg_winner(self) -> float:
        return self._gross_profit / self._win_count if self._win_count > 0 else 0.0

    @property
    def avg_loser(self) -> float:
        return self._gross_loss / self._loss_count if self._loss_count > 0 else 0.0

    @property
    def trade_history(self) -> list[float]:
        """Return list of all trade PnL values for Kelly calculation."""
        return [t.pnl for t in self._trades]

    def to_dict(self) -> dict[str, float]:
        """Export metrics as dictionary."""
        return {
            "total_pnl": self.total_pnl,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "expectancy": self.expectancy,
            "avg_winner": self.avg_winner,
            "avg_loser": self.avg_loser,
        }
