"""Liquidity analysis and position size filtering.

Monitors order book depth and bid-ask spreads to ensure adequate
liquidity before entering positions. Estimates slippage based on
order size relative to book depth and auto-reduces position sizes
in low liquidity conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class LiquidityConfig:
    """Configuration for liquidity analysis."""
    min_depth_usdt: float = 50000.0  # Minimum order book depth (both sides)
    max_spread_bps: float = 20.0  # Maximum acceptable spread in basis points
    max_slippage_bps: float = 10.0  # Maximum acceptable estimated slippage
    depth_levels: int = 20  # Number of order book levels to analyze
    size_reduction_threshold_bps: float = 5.0  # Start reducing at this slippage
    min_volume_24h_usdt: float = 1000000.0  # Minimum 24h volume in USDT


@dataclass
class LiquiditySnapshot:
    """Point-in-time liquidity measurement for a symbol."""
    symbol: str
    bid_depth_usdt: float  # Total USDT depth on bid side
    ask_depth_usdt: float  # Total USDT depth on ask side
    spread_bps: float
    mid_price: float
    estimated_slippage_bps: float  # For the proposed order size
    book_imbalance: float  # (bid_depth - ask_depth) / (bid_depth + ask_depth)
    is_liquid: bool
    recommended_size_multiplier: float
    timestamp: datetime = dc_field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class LiquidityCheckResult:
    """Result of liquidity filter check."""
    is_acceptable: bool
    snapshot: LiquiditySnapshot
    original_size_pct: float
    adjusted_size_pct: float
    reason: str


class LiquidityFilter:
    """Order book liquidity analysis and position size filter.

    Analyzes real time order book depth to:
    1. Reject trades in illiquid markets
    2. Reduce position sizes when liquidity is thin
    3. Estimate slippage for a given order size
    4. Monitor bid-ask spread health
    """

    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self._config = config or LiquidityConfig()
        # Cache recent snapshots per symbol
        self._snapshots: dict[str, LiquiditySnapshot] = {}

    def analyze_orderbook(
        self,
        symbol: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        order_size_usdt: float = 0.0,
    ) -> LiquiditySnapshot:
        """Analyze an order book for liquidity quality.

        Args:
            symbol: Trading pair symbol
            bids: List of (price and quantity) tuples on bid side
            asks: List of (price and quantity) tuples on ask side
            order_size_usdt: Proposed order size for slippage estimation

        Returns:
            LiquiditySnapshot with full analysis.
        """
        cfg = self._config

        # Calculate depth in USDT
        bid_depth = sum(price * qty for price, qty in bids[:cfg.depth_levels])
        ask_depth = sum(price * qty for price, qty in asks[:cfg.depth_levels])

        # Mid price and spread
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        mid_price = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 0.0
        spread_bps = (
            (best_ask - best_bid) / mid_price * 10000
            if mid_price > 0 else 0.0
        )

        # Book imbalance
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0

        # Estimate slippage for proposed order size
        slippage_bps = self._estimate_slippage(
            asks if order_size_usdt > 0 else bids,
            abs(order_size_usdt),
            mid_price,
        )

        # Determine liquidity quality
        is_liquid = (
            (bid_depth + ask_depth) >= cfg.min_depth_usdt
            and spread_bps <= cfg.max_spread_bps
        )

        # Compute recommended size multiplier
        multiplier = 1.0
        if not is_liquid:
            depth_ratio = (bid_depth + ask_depth) / cfg.min_depth_usdt
            spread_ratio = cfg.max_spread_bps / max(spread_bps, 0.01)
            multiplier = min(depth_ratio, spread_ratio, 1.0)
            multiplier = max(multiplier, 0.0)

        if slippage_bps > cfg.size_reduction_threshold_bps:
            slip_mult = cfg.size_reduction_threshold_bps / max(slippage_bps, 0.01)
            multiplier = min(multiplier, slip_mult)

        snapshot = LiquiditySnapshot(
            symbol=symbol,
            bid_depth_usdt=round(bid_depth, 2),
            ask_depth_usdt=round(ask_depth, 2),
            spread_bps=round(spread_bps, 2),
            mid_price=round(mid_price, 4),
            estimated_slippage_bps=round(slippage_bps, 2),
            book_imbalance=round(imbalance, 4),
            is_liquid=is_liquid,
            recommended_size_multiplier=round(max(multiplier, 0.0), 4),
        )

        self._snapshots[symbol] = snapshot
        return snapshot

    def check(
        self,
        symbol: str,
        proposed_size_pct: float,
        portfolio_value: float,
        bids: list[tuple[float, float]] | None = None,
        asks: list[tuple[float, float]] | None = None,
    ) -> LiquidityCheckResult:
        """Check if proposed position size is acceptable given liquidity.

        Args:
            symbol: Trading pair
            proposed_size_pct: Proposed position size as pct of portfolio
            portfolio_value: Current portfolio value in USDT
            bids: Optional fresh order book bids
            asks: Optional fresh order book asks

        Returns:
            LiquidityCheckResult with adjusted size recommendation.
        """
        order_size_usdt = portfolio_value * proposed_size_pct / 100

        if bids is not None and asks is not None:
            snapshot = self.analyze_orderbook(symbol, bids, asks, order_size_usdt)
        elif symbol in self._snapshots:
            snapshot = self._snapshots[symbol]
        else:
            # No data available; allow with warning
            return LiquidityCheckResult(
                is_acceptable=True,
                snapshot=LiquiditySnapshot(
                    symbol=symbol,
                    bid_depth_usdt=0.0,
                    ask_depth_usdt=0.0,
                    spread_bps=0.0,
                    mid_price=0.0,
                    estimated_slippage_bps=0.0,
                    book_imbalance=0.0,
                    is_liquid=True,
                    recommended_size_multiplier=1.0,
                ),
                original_size_pct=proposed_size_pct,
                adjusted_size_pct=proposed_size_pct,
                reason="No order book data available",
            )

        adjusted_size = proposed_size_pct * snapshot.recommended_size_multiplier
        is_acceptable = snapshot.is_liquid and snapshot.estimated_slippage_bps <= self._config.max_slippage_bps

        reasons: list[str] = []
        if not snapshot.is_liquid:
            reasons.append(
                f"Low liquidity: depth={snapshot.bid_depth_usdt + snapshot.ask_depth_usdt:.0f} "
                f"USDT spread={snapshot.spread_bps:.1f}bps"
            )
        if snapshot.estimated_slippage_bps > self._config.max_slippage_bps:
            reasons.append(
                f"High slippage: {snapshot.estimated_slippage_bps:.1f}bps "
                f"exceeds {self._config.max_slippage_bps:.1f}bps limit"
            )

        reason = " | ".join(reasons) if reasons else "Liquidity adequate"

        if not is_acceptable:
            logger.warning(
                "liquidity.check_failed",
                symbol=symbol,
                spread_bps=snapshot.spread_bps,
                slippage_bps=snapshot.estimated_slippage_bps,
                multiplier=snapshot.recommended_size_multiplier,
            )

        return LiquidityCheckResult(
            is_acceptable=is_acceptable,
            snapshot=snapshot,
            original_size_pct=round(proposed_size_pct, 4),
            adjusted_size_pct=round(adjusted_size, 4),
            reason=reason,
        )

    @staticmethod
    def _estimate_slippage(
        book_side: list[tuple[float, float]],
        order_size_usdt: float,
        mid_price: float,
    ) -> float:
        """Estimate slippage for a given order size against one side of the book.

        Walks through the order book levels and calculates the
        volume-weighted average fill price vs mid price.

        Args:
            book_side: List of (price and qty) tuples
            order_size_usdt: Order size in USDT
            mid_price: Current mid price

        Returns:
            Estimated slippage in basis points.
        """
        if not book_side or mid_price <= 0 or order_size_usdt <= 0:
            return 0.0

        remaining = order_size_usdt
        total_filled_value = 0.0

        for price, qty in book_side:
            level_value = price * qty
            if remaining <= 0:
                break

            fill = min(remaining, level_value)
            total_filled_value += fill * (price / mid_price)
            remaining -= fill

        if remaining > order_size_usdt * 0.5:
            # Could not fill more than half the order in available levels
            return 100.0  # Very high slippage indicator

        filled = order_size_usdt - remaining
        if filled <= 0:
            return 0.0

        avg_price_ratio = total_filled_value / filled
        slippage_bps = abs(avg_price_ratio - 1.0) * 10000

        return slippage_bps

    def get_snapshot(self, symbol: str) -> LiquiditySnapshot | None:
        """Get the most recent liquidity snapshot for a symbol."""
        return self._snapshots.get(symbol)
