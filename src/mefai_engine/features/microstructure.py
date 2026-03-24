"""Market microstructure features - order flow, imbalance, CVD."""

from __future__ import annotations

import numpy as np

from mefai_engine.features.registry import feature


@feature(name="book_imbalance", depends_on=["bid_volume", "ask_volume"], category="microstructure")
def book_imbalance(bid_volume: np.ndarray, ask_volume: np.ndarray) -> np.ndarray:
    """Order book imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol).

    Range: -1 (all asks) to +1 (all bids).
    """
    total = bid_volume + ask_volume
    return np.where(total > 0, (bid_volume - ask_volume) / total, 0.0)


@feature(name="cvd", depends_on=["buy_volume", "sell_volume"], category="microstructure")
def cvd(buy_volume: np.ndarray, sell_volume: np.ndarray) -> np.ndarray:
    """Cumulative Volume Delta."""
    return np.cumsum(buy_volume - sell_volume)


@feature(name="trade_imbalance_50", depends_on=["buy_volume", "sell_volume"], params={"period": 50}, category="microstructure")
def trade_imbalance_50(buy_volume: np.ndarray, sell_volume: np.ndarray, period: int = 50) -> np.ndarray:
    """Rolling trade imbalance over N periods."""
    n = len(buy_volume)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        bv = np.sum(buy_volume[i - period + 1: i + 1])
        sv = np.sum(sell_volume[i - period + 1: i + 1])
        total = bv + sv
        out[i] = (bv - sv) / total if total > 0 else 0
    return out


@feature(name="large_trade_ratio", depends_on=["large_buy_volume", "large_sell_volume", "volume"], category="microstructure")
def large_trade_ratio(large_buy_volume: np.ndarray, large_sell_volume: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Ratio of large trades (whale activity) to total volume."""
    large_total = large_buy_volume + large_sell_volume
    return np.where(volume > 0, large_total / volume, 0.0)


@feature(name="spread_bps", depends_on=["best_bid", "best_ask"], category="microstructure")
def spread_bps(best_bid: np.ndarray, best_ask: np.ndarray) -> np.ndarray:
    """Bid-ask spread in basis points."""
    mid = (best_bid + best_ask) / 2
    return np.where(mid > 0, (best_ask - best_bid) / mid * 10000, 0.0)


@feature(name="weighted_mid_price", depends_on=["best_bid", "best_ask", "bid_volume", "ask_volume"], category="microstructure")
def weighted_mid_price(best_bid: np.ndarray, best_ask: np.ndarray, bid_volume: np.ndarray, ask_volume: np.ndarray) -> np.ndarray:
    """Volume-weighted mid price (microprice)."""
    total = bid_volume + ask_volume
    return np.where(
        total > 0,
        (best_bid * ask_volume + best_ask * bid_volume) / total,
        (best_bid + best_ask) / 2,
    )
