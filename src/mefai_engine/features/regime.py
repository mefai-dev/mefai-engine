"""Market regime detection features."""

from __future__ import annotations

import numpy as np

from mefai_engine.features.registry import feature
from mefai_engine.features.technical import _ema, _sma, _wilder_smooth


@feature(name="trend_strength", depends_on=["close"], params={"short": 20, "long": 50}, category="regime")
def trend_strength(close: np.ndarray, short: int = 20, long: int = 50) -> np.ndarray:
    """Trend strength: normalized distance between short and long EMA.

    Positive = uptrend, negative = downtrend, magnitude = strength.
    """
    short_ema = _ema(close, short)
    long_ema = _ema(close, long)
    return (short_ema - long_ema) / np.where(long_ema > 0, long_ema, 1) * 100


@feature(name="volatility_regime", depends_on=["close"], params={"period": 20, "lookback": 252}, category="regime")
def volatility_regime(close: np.ndarray, period: int = 20, lookback: int = 252) -> np.ndarray:
    """Volatility regime: current vol percentile over lookback.

    0-25 = low vol, 25-75 = normal, 75-100 = high vol.
    """
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.insert(log_ret, 0, 0)
    n = len(close)
    out = np.full(n, np.nan)

    for i in range(max(period, lookback), n):
        current_vol = np.std(log_ret[i - period + 1: i + 1])
        hist_vols = np.array([
            np.std(log_ret[j - period + 1: j + 1])
            for j in range(i - lookback, i)
        ])
        out[i] = np.sum(hist_vols < current_vol) / len(hist_vols) * 100

    return out


@feature(name="mean_reversion_score", depends_on=["close"], params={"period": 20}, category="regime")
def mean_reversion_score(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Z-score of price vs SMA: how far from mean.

    |z| > 2 = extreme, likely mean reversion opportunity.
    """
    sma = _sma(close, period)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        std = np.std(window)
        out[i] = (close[i] - sma[i]) / std if std > 0 else 0
    return out


@feature(name="regime_label", depends_on=["close", "high", "low"], params={"adx_period": 14, "vol_period": 20}, category="regime")
def regime_label(close: np.ndarray, high: np.ndarray, low: np.ndarray, adx_period: int = 14, vol_period: int = 20) -> np.ndarray:
    """Market regime classification.

    Returns:
        0 = trending_up
        1 = trending_down
        2 = ranging
        3 = high_volatility
        4 = low_volatility
    """
    n = len(close)
    out = np.full(n, 2.0)  # default: ranging

    # ADX for trend detection
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))
    ))
    tr[0] = high[0] - low[0]
    atr = _wilder_smooth(tr, adx_period)
    plus_di = 100 * _wilder_smooth(plus_dm, adx_period) / np.where(atr > 0, atr, 1)
    minus_di = 100 * _wilder_smooth(minus_dm, adx_period) / np.where(atr > 0, atr, 1)
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1)
    adx = _wilder_smooth(dx, adx_period)

    # Volatility for vol regime
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.insert(log_ret, 0, 0)

    # EMA direction
    ema_short = _ema(close, 20)
    ema_long = _ema(close, 50)

    for i in range(max(50, adx_period * 2), n):
        vol = np.std(log_ret[i - vol_period + 1: i + 1])
        vol_mean = np.mean([
            np.std(log_ret[j - vol_period + 1: j + 1])
            for j in range(max(vol_period, i - 100), i)
        ]) if i > vol_period + 10 else vol

        if not np.isnan(adx[i]) and adx[i] > 25:
            # Strong trend
            out[i] = 0.0 if ema_short[i] > ema_long[i] else 1.0
        elif vol > vol_mean * 1.5:
            out[i] = 3.0  # high volatility
        elif vol < vol_mean * 0.5:
            out[i] = 4.0  # low volatility
        else:
            out[i] = 2.0  # ranging

    return out
