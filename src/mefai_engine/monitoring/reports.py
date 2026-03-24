"""Automated trading report generation.

Generates daily/weekly performance reports with key metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

from mefai_engine.risk.pnl_tracker import PnLTracker

logger = structlog.get_logger()


class ReportGenerator:
    """Generates structured trading performance reports."""

    def __init__(self, pnl_tracker: PnLTracker) -> None:
        self._tracker = pnl_tracker

    def daily_summary(self, equity: float, positions_count: int = 0) -> dict[str, Any]:
        """Generate daily trading summary."""
        metrics = self._tracker.to_dict()
        return {
            "report_type": "daily",
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "equity": round(equity, 2),
            "total_pnl": round(metrics["total_pnl"], 2),
            "total_trades": int(metrics["total_trades"]),
            "win_rate": round(metrics["win_rate"] * 100, 1),
            "profit_factor": round(metrics["profit_factor"], 2),
            "max_drawdown_pct": round(metrics["max_drawdown_pct"], 2),
            "current_drawdown_pct": round(metrics["current_drawdown_pct"], 2),
            "expectancy": round(metrics["expectancy"], 2),
            "avg_winner": round(metrics["avg_winner"], 2),
            "avg_loser": round(metrics["avg_loser"], 2),
            "open_positions": positions_count,
        }

    def format_telegram(self, report: dict[str, Any]) -> str:
        """Format report for Telegram notification."""
        pnl = report["total_pnl"]
        prefix = "+" if pnl >= 0 else ""

        lines = [
            f"<b>DAILY REPORT</b>",
            f"",
            f"Equity: <code>${report['equity']:,.2f}</code>",
            f"Total PnL: <code>{prefix}${pnl:,.2f}</code>",
            f"Trades: <code>{report['total_trades']}</code>",
            f"Win Rate: <code>{report['win_rate']}%</code>",
            f"Profit Factor: <code>{report['profit_factor']}</code>",
            f"",
            f"Max DD: <code>{report['max_drawdown_pct']}%</code>",
            f"Cur DD: <code>{report['current_drawdown_pct']}%</code>",
            f"Expectancy: <code>${report['expectancy']:,.2f}</code>",
            f"Avg Win: <code>${report['avg_winner']:,.2f}</code>",
            f"Avg Loss: <code>${report['avg_loser']:,.2f}</code>",
            f"Open Pos: <code>{report['open_positions']}</code>",
        ]
        return "\n".join(lines)

    def format_text(self, report: dict[str, Any]) -> str:
        """Format report as plain text."""
        pnl = report["total_pnl"]
        prefix = "+" if pnl >= 0 else ""

        return (
            f"=== MEFAI Engine Daily Report ===\n"
            f"Time:         {report['generated_at']}\n"
            f"Equity:       ${report['equity']:,.2f}\n"
            f"Total PnL:    {prefix}${pnl:,.2f}\n"
            f"Trades:       {report['total_trades']}\n"
            f"Win Rate:     {report['win_rate']}%\n"
            f"Profit Factor:{report['profit_factor']}\n"
            f"Max Drawdown: {report['max_drawdown_pct']}%\n"
            f"Expectancy:   ${report['expectancy']:,.2f}\n"
            f"================================="
        )
