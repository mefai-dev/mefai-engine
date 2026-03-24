"""On-chain and perpetual futures specific features."""

from __future__ import annotations

import numpy as np

from mefai_engine.features.registry import feature


@feature(name="funding_rate", depends_on=["funding_rate_raw"], category="onchain")
def funding_rate(funding_rate_raw: np.ndarray) -> np.ndarray:
    """Funding rate as-is (already normalized)."""
    return funding_rate_raw


@feature(name="funding_premium", depends_on=["funding_rate_raw"], params={"period": 24}, category="onchain")
def funding_premium(funding_rate_raw: np.ndarray, period: int = 24) -> np.ndarray:
    """Annualized funding premium (rolling sum * 365)."""
    n = len(funding_rate_raw)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        out[i] = np.sum(funding_rate_raw[i - period + 1: i + 1]) * (365 / period * 3)
    return out


@feature(name="oi_change_pct", depends_on=["open_interest"], params={"period": 1}, category="onchain")
def oi_change_pct(open_interest: np.ndarray, period: int = 1) -> np.ndarray:
    """Open interest percentage change."""
    out = np.full_like(open_interest, np.nan)
    out[period:] = (open_interest[period:] - open_interest[:-period]) / np.where(
        open_interest[:-period] > 0, open_interest[:-period], 1
    ) * 100
    return out


@feature(name="long_short_ratio", depends_on=["long_accounts", "short_accounts"], category="onchain")
def long_short_ratio(long_accounts: np.ndarray, short_accounts: np.ndarray) -> np.ndarray:
    """Long/short account ratio."""
    return np.where(short_accounts > 0, long_accounts / short_accounts, 1.0)


@feature(name="liquidation_intensity", depends_on=["long_liquidations", "short_liquidations", "volume"], params={"period": 24}, category="onchain")
def liquidation_intensity(long_liquidations: np.ndarray, short_liquidations: np.ndarray, volume: np.ndarray, period: int = 24) -> np.ndarray:
    """Total liquidations as % of volume (rolling)."""
    total_liq = long_liquidations + short_liquidations
    n = len(volume)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        liq_sum = np.sum(total_liq[i - period + 1: i + 1])
        vol_sum = np.sum(volume[i - period + 1: i + 1])
        out[i] = liq_sum / vol_sum * 100 if vol_sum > 0 else 0
    return out
