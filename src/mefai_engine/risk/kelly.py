"""Kelly criterion position sizing.

Calculates optimal position size using the Kelly formula and its
fractional variants. Integrates with the risk manager to provide
mathematically optimal bet sizing based on edge and odds.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class KellyConfig:
    """Kelly criterion configuration."""
    fraction: float = 0.25  # Fractional Kelly (0.25 = quarter Kelly)
    max_position_pct: float = 10.0  # Hard cap on position size
    min_win_rate: float = 0.4  # Minimum win rate to allow any position
    min_trades: int = 30  # Minimum historical trades before trusting Kelly
    confidence_scaling: bool = True  # Scale by signal confidence


@dataclass
class KellyResult:
    """Result of Kelly criterion calculation."""
    full_kelly_pct: float
    fractional_kelly_pct: float
    capped_size_pct: float
    edge: float
    is_valid: bool
    reason: str


class KellyCriterion:
    """Kelly criterion position sizing calculator.

    The Kelly criterion determines the optimal fraction of capital to
    risk on a bet given the probability of winning and the payoff ratio.

    Full Kelly: f* = (p * b - q) / b
    where:
        p = probability of winning
        q = 1 - p (probability of losing)
        b = ratio of average win to average loss

    Fractional Kelly uses a fraction (e.g. 0.25 or 0.5) of full Kelly
    to reduce volatility at the cost of slightly lower returns.
    """

    def __init__(self, config: KellyConfig | None = None) -> None:
        self._config = config or KellyConfig()

    def calculate(
        self,
        win_rate: float,
        avg_winner: float,
        avg_loser: float,
        confidence: float = 1.0,
        n_trades: int = 0,
    ) -> KellyResult:
        """Calculate optimal position size using Kelly criterion.

        Args:
            win_rate: Historical win rate (0.0 to 1.0)
            avg_winner: Average winning trade amount (positive)
            avg_loser: Average losing trade amount (positive)
            confidence: Current signal confidence (0.0 to 1.0)
            n_trades: Number of historical trades used for statistics

        Returns:
            KellyResult with full and fractional Kelly sizes.
        """
        cfg = self._config

        # Validate inputs
        if n_trades < cfg.min_trades:
            return KellyResult(
                full_kelly_pct=0.0,
                fractional_kelly_pct=0.0,
                capped_size_pct=0.0,
                edge=0.0,
                is_valid=False,
                reason=f"Insufficient trades: {n_trades} < {cfg.min_trades}",
            )

        if win_rate < cfg.min_win_rate:
            return KellyResult(
                full_kelly_pct=0.0,
                fractional_kelly_pct=0.0,
                capped_size_pct=0.0,
                edge=0.0,
                is_valid=False,
                reason=f"Win rate too low: {win_rate:.2%} < {cfg.min_win_rate:.2%}",
            )

        if avg_loser <= 0:
            return KellyResult(
                full_kelly_pct=0.0,
                fractional_kelly_pct=0.0,
                capped_size_pct=0.0,
                edge=0.0,
                is_valid=False,
                reason="Average loser must be positive",
            )

        # Kelly formula
        p = win_rate
        q = 1.0 - p
        b = avg_winner / avg_loser  # Payoff ratio

        edge = p * b - q
        full_kelly = edge / b if b > 0 else 0.0

        if full_kelly <= 0:
            return KellyResult(
                full_kelly_pct=0.0,
                fractional_kelly_pct=0.0,
                capped_size_pct=0.0,
                edge=round(edge, 6),
                is_valid=False,
                reason="Negative edge: Kelly suggests no position",
            )

        full_kelly_pct = full_kelly * 100

        # Apply fractional Kelly
        fractional = full_kelly_pct * cfg.fraction

        # Apply confidence scaling
        if cfg.confidence_scaling:
            fractional *= confidence

        # Cap at maximum
        capped = min(fractional, cfg.max_position_pct)

        result = KellyResult(
            full_kelly_pct=round(full_kelly_pct, 4),
            fractional_kelly_pct=round(fractional, 4),
            capped_size_pct=round(capped, 4),
            edge=round(edge, 6),
            is_valid=True,
            reason="Valid Kelly position size",
        )

        logger.debug(
            "kelly.calculated",
            win_rate=f"{win_rate:.2%}",
            payoff_ratio=f"{b:.2f}",
            edge=f"{edge:.4f}",
            full_kelly=f"{full_kelly_pct:.2f}%",
            fractional=f"{fractional:.2f}%",
            capped=f"{capped:.2f}%",
        )

        return result

    def calculate_from_trades(
        self,
        trade_pnls: list[float],
        confidence: float = 1.0,
    ) -> KellyResult:
        """Calculate Kelly from a list of historical trade P&Ls.

        Args:
            trade_pnls: List of trade profit/loss values
            confidence: Current signal confidence

        Returns:
            KellyResult computed from trade history.
        """
        if not trade_pnls:
            return KellyResult(
                full_kelly_pct=0.0,
                fractional_kelly_pct=0.0,
                capped_size_pct=0.0,
                edge=0.0,
                is_valid=False,
                reason="No trade history",
            )

        winners = [t for t in trade_pnls if t > 0]
        losers = [t for t in trade_pnls if t < 0]

        n_trades = len(trade_pnls)
        win_rate = len(winners) / n_trades if n_trades > 0 else 0.0
        avg_winner = sum(winners) / len(winners) if winners else 0.0
        avg_loser = abs(sum(losers) / len(losers)) if losers else 0.0001

        return self.calculate(
            win_rate=win_rate,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            confidence=confidence,
            n_trades=n_trades,
        )
