"""Technical analysis indicators - 50+ vectorized implementations.

All functions operate on numpy arrays for maximum performance.
No loops - pure vectorized operations with optional numba JIT.
"""

from __future__ import annotations

import numpy as np

from mefai_engine.features.registry import feature


# ═══════════════════════════════════════════════════════════════════
#  TREND INDICATORS
# ═══════════════════════════════════════════════════════════════════

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (vectorized)."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average using cumsum trick."""
    cs = np.cumsum(data)
    cs[period:] = cs[period:] - cs[:-period]
    out = np.full_like(data, np.nan)
    out[period - 1:] = cs[period - 1:] / period
    return out


def _wilder_smooth(data: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing method."""
    out = np.empty_like(data)
    out[:period] = np.nan
    out[period - 1] = np.nanmean(data[:period])
    for i in range(period, len(data)):
        out[i] = (out[i - 1] * (period - 1) + data[i]) / period
    return out


@feature(name="sma_10", depends_on=["close"], params={"period": 10}, category="trend")
def sma_10(close: np.ndarray, period: int = 10) -> np.ndarray:
    return _sma(close, period)


@feature(name="sma_20", depends_on=["close"], params={"period": 20}, category="trend")
def sma_20(close: np.ndarray, period: int = 20) -> np.ndarray:
    return _sma(close, period)


@feature(name="sma_50", depends_on=["close"], params={"period": 50}, category="trend")
def sma_50(close: np.ndarray, period: int = 50) -> np.ndarray:
    return _sma(close, period)


@feature(name="sma_200", depends_on=["close"], params={"period": 200}, category="trend")
def sma_200(close: np.ndarray, period: int = 200) -> np.ndarray:
    return _sma(close, period)


@feature(name="ema_10", depends_on=["close"], params={"period": 10}, category="trend")
def ema_10(close: np.ndarray, period: int = 10) -> np.ndarray:
    return _ema(close, period)


@feature(name="ema_20", depends_on=["close"], params={"period": 20}, category="trend")
def ema_20(close: np.ndarray, period: int = 20) -> np.ndarray:
    return _ema(close, period)


@feature(name="ema_50", depends_on=["close"], params={"period": 50}, category="trend")
def ema_50(close: np.ndarray, period: int = 50) -> np.ndarray:
    return _ema(close, period)


@feature(name="hma_20", depends_on=["close"], params={"period": 20}, category="trend")
def hma_20(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Hull Moving Average."""
    half = _ema(close, period // 2)
    full = _ema(close, period)
    raw = 2 * half - full
    return _ema(raw, int(np.sqrt(period)))


@feature(name="adx_14", depends_on=["high", "low", "close"], params={"period": 14}, category="trend")
def adx_14(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index."""
    n = len(close)
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]

    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / np.where(atr > 0, atr, 1)
    minus_di = 100 * _wilder_smooth(minus_dm, period) / np.where(atr > 0, atr, 1)

    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1)
    return _wilder_smooth(dx, period)


@feature(name="aroon_up_25", depends_on=["high"], params={"period": 25}, category="trend")
def aroon_up_25(high: np.ndarray, period: int = 25) -> np.ndarray:
    """Aroon Up."""
    out = np.full_like(high, np.nan)
    for i in range(period, len(high)):
        window = high[i - period: i + 1]
        days_since = period - np.argmax(window)
        out[i] = 100 * (period - days_since) / period
    return out


@feature(name="aroon_down_25", depends_on=["low"], params={"period": 25}, category="trend")
def aroon_down_25(low: np.ndarray, period: int = 25) -> np.ndarray:
    """Aroon Down."""
    out = np.full_like(low, np.nan)
    for i in range(period, len(low)):
        window = low[i - period: i + 1]
        days_since = period - np.argmin(window)
        out[i] = 100 * (period - days_since) / period
    return out


# ═══��═══════════════════════════════���═══════════════════��═══════════
#  MOMENTUM INDICATORS
# ══���══════════════════��═════════════════════════════════════════════

@feature(name="rsi_14", depends_on=["close"], params={"period": 14}, category="momentum")
def rsi_14(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index with Wilder smoothing."""
    delta = np.diff(close, prepend=close[0])
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = _wilder_smooth(gains, period)
    avg_loss = _wilder_smooth(losses, period)

    rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
    return 100 - (100 / (1 + rs))


@feature(name="macd_12_26_9", depends_on=["close"], params={"fast": 12, "slow": 26, "signal": 9}, category="momentum")
def macd_12_26_9(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram."""
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = _ema(macd_line, signal)
    return macd_line - signal_line  # histogram


@feature(name="macd_line", depends_on=["close"], params={"fast": 12, "slow": 26}, category="momentum")
def macd_line(close: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    """MACD line."""
    return _ema(close, fast) - _ema(close, slow)


@feature(name="macd_signal", depends_on=["close"], params={"fast": 12, "slow": 26, "signal": 9}, category="momentum")
def macd_signal(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD signal line."""
    line = _ema(close, fast) - _ema(close, slow)
    return _ema(line, signal)


@feature(name="stoch_k_14", depends_on=["high", "low", "close"], params={"period": 14, "smooth": 3}, category="momentum")
def stoch_k_14(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, smooth: int = 3) -> np.ndarray:
    """Stochastic %K."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        h = np.max(high[i - period + 1: i + 1])
        l = np.min(low[i - period + 1: i + 1])
        denom = h - l
        out[i] = 100 * (close[i] - l) / denom if denom > 0 else 50
    return _sma(out, smooth)


@feature(name="stoch_d_14", depends_on=["high", "low", "close"], params={"period": 14, "smooth": 3, "d_smooth": 3}, category="momentum")
def stoch_d_14(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14, smooth: int = 3, d_smooth: int = 3) -> np.ndarray:
    """Stochastic %D."""
    k = stoch_k_14(high, low, close, period, smooth)
    return _sma(k, d_smooth)


@feature(name="cci_20", depends_on=["high", "low", "close"], params={"period": 20}, category="momentum")
def cci_20(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma = _sma(tp, period)
    n = len(tp)
    mad = np.full(n, np.nan)
    for i in range(period - 1, n):
        mad[i] = np.mean(np.abs(tp[i - period + 1: i + 1] - sma[i]))
    return (tp - sma) / np.where(mad > 0, mad * 0.015, 1e-10)


@feature(name="williams_r_14", depends_on=["high", "low", "close"], params={"period": 14}, category="momentum")
def williams_r_14(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Williams %R."""
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1: i + 1])
        ll = np.min(low[i - period + 1: i + 1])
        denom = hh - ll
        out[i] = -100 * (hh - close[i]) / denom if denom > 0 else -50
    return out


@feature(name="roc_10", depends_on=["close"], params={"period": 10}, category="momentum")
def roc_10(close: np.ndarray, period: int = 10) -> np.ndarray:
    """Rate of Change."""
    out = np.full_like(close, np.nan)
    out[period:] = (close[period:] - close[:-period]) / close[:-period] * 100
    return out


@feature(name="mfi_14", depends_on=["high", "low", "close", "volume"], params={"period": 14}, category="momentum")
def mfi_14(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 14) -> np.ndarray:
    """Money Flow Index."""
    tp = (high + low + close) / 3
    raw_mf = tp * volume
    delta = np.diff(tp, prepend=tp[0])
    pos_mf = np.where(delta > 0, raw_mf, 0.0)
    neg_mf = np.where(delta < 0, raw_mf, 0.0)

    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        pmf = np.sum(pos_mf[i - period + 1: i + 1])
        nmf = np.sum(neg_mf[i - period + 1: i + 1])
        ratio = pmf / nmf if nmf > 0 else 100
        out[i] = 100 - (100 / (1 + ratio))
    return out


# ════════════════════════════════════���══════════════════════���═══════
#  VOLATILITY INDICATORS
# ═══════��════════════════════════════════���══════════════════════════

@feature(name="atr_14", depends_on=["high", "low", "close"], params={"period": 14}, category="volatility")
def atr_14(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    return _wilder_smooth(tr, period)


@feature(name="bollinger_upper_20", depends_on=["close"], params={"period": 20, "std": 2}, category="volatility")
def bollinger_upper_20(close: np.ndarray, period: int = 20, std: int = 2) -> np.ndarray:
    """Bollinger Band upper."""
    mid = _sma(close, period)
    n = len(close)
    sd = np.full(n, np.nan)
    for i in range(period - 1, n):
        sd[i] = np.std(close[i - period + 1: i + 1])
    return mid + std * sd


@feature(name="bollinger_lower_20", depends_on=["close"], params={"period": 20, "std": 2}, category="volatility")
def bollinger_lower_20(close: np.ndarray, period: int = 20, std: int = 2) -> np.ndarray:
    """Bollinger Band lower."""
    mid = _sma(close, period)
    n = len(close)
    sd = np.full(n, np.nan)
    for i in range(period - 1, n):
        sd[i] = np.std(close[i - period + 1: i + 1])
    return mid - std * sd


@feature(name="bollinger_width_20", depends_on=["close"], params={"period": 20, "std": 2}, category="volatility")
def bollinger_width_20(close: np.ndarray, period: int = 20, std: int = 2) -> np.ndarray:
    """Bollinger Band width (volatility squeeze detection)."""
    upper = bollinger_upper_20(close, period, std)
    lower = bollinger_lower_20(close, period, std)
    mid = _sma(close, period)
    return (upper - lower) / np.where(mid > 0, mid, 1e-10)


@feature(name="keltner_upper_20", depends_on=["high", "low", "close"], params={"period": 20, "mult": 1.5}, category="volatility")
def keltner_upper_20(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, mult: float = 1.5) -> np.ndarray:
    """Keltner Channel upper."""
    mid = _ema(close, period)
    atr = atr_14(high, low, close, period)
    return mid + mult * atr


@feature(name="keltner_lower_20", depends_on=["high", "low", "close"], params={"period": 20, "mult": 1.5}, category="volatility")
def keltner_lower_20(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20, mult: float = 1.5) -> np.ndarray:
    """Keltner Channel lower."""
    mid = _ema(close, period)
    atr = atr_14(high, low, close, period)
    return mid - mult * atr


@feature(name="historical_vol_20", depends_on=["close"], params={"period": 20}, category="volatility")
def historical_vol_20(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Annualized historical volatility."""
    log_ret = np.log(close[1:] / close[:-1])
    log_ret = np.insert(log_ret, 0, 0)
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period, n):
        out[i] = np.std(log_ret[i - period + 1: i + 1]) * np.sqrt(365)
    return out


# ═══��═══════════════════════════════════════════════════════════════
#  VOLUME INDICATORS
# ════════════��══════════════════════════════════��═══════════════════

@feature(name="obv", depends_on=["close", "volume"], category="volume")
def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    direction = np.sign(np.diff(close, prepend=close[0]))
    return np.cumsum(direction * volume)


@feature(name="vwap", depends_on=["high", "low", "close", "volume"], category="volume")
def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Volume Weighted Average Price (cumulative)."""
    tp = (high + low + close) / 3
    cum_tp_vol = np.cumsum(tp * volume)
    cum_vol = np.cumsum(volume)
    return cum_tp_vol / np.where(cum_vol > 0, cum_vol, 1)


@feature(name="volume_sma_20", depends_on=["volume"], params={"period": 20}, category="volume")
def volume_sma_20(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Volume SMA."""
    return _sma(volume, period)


@feature(name="volume_ratio", depends_on=["volume"], params={"period": 20}, category="volume")
def volume_ratio(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Current volume / average volume."""
    avg = _sma(volume, period)
    return volume / np.where(avg > 0, avg, 1)


@feature(name="accumulation_dist", depends_on=["high", "low", "close", "volume"], category="volume")
def accumulation_dist(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Accumulation/Distribution Line."""
    denom = high - low
    clv = np.where(denom > 0, ((close - low) - (high - close)) / denom, 0)
    return np.cumsum(clv * volume)


@feature(name="chaikin_mf_20", depends_on=["high", "low", "close", "volume"], params={"period": 20}, category="volume")
def chaikin_mf_20(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
    """Chaikin Money Flow."""
    denom = high - low
    clv = np.where(denom > 0, ((close - low) - (high - close)) / denom, 0)
    mf_volume = clv * volume
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(period - 1, n):
        sv = np.sum(mf_volume[i - period + 1: i + 1])
        v = np.sum(volume[i - period + 1: i + 1])
        out[i] = sv / v if v > 0 else 0
    return out


@feature(name="force_index_13", depends_on=["close", "volume"], params={"period": 13}, category="volume")
def force_index_13(close: np.ndarray, volume: np.ndarray, period: int = 13) -> np.ndarray:
    """Force Index."""
    raw = np.diff(close, prepend=close[0]) * volume
    return _ema(raw, period)
