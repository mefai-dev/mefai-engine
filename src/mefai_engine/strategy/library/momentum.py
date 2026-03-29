"""Momentum strategy - trend following with RSI and MACD confirmation.

A production-ready built-in strategy that developers can use as reference
or deploy directly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from mefai_engine.constants import Direction
from mefai_engine.strategy.base import BaseStrategy
from mefai_engine.types import Candle, Signal, Ticker

logger = structlog.get_logger()


class MomentumStrategy(BaseStrategy):
    """Trend-following momentum strategy.

    Entry conditions (LONG):
    - EMA 10 above EMA 50 (uptrend)
    - RSI between 40-70 (not overbought)
    - MACD histogram positive
    - ADX above 20 (trending market)

    Entry conditions (SHORT):
    - EMA 10 below EMA 50 (downtrend)
    - RSI between 30-60 (not oversold)
    - MACD histogram negative
    - ADX above 20 (trending market)

    Exit: Signal reversal or RSI extreme.
    """

    strategy_id = "momentum_v1"
    symbols = ["BTCUSDT"]
    timeframes = ["1h"]

    def __init__(
        self,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        adx_threshold: float = 20.0,
        min_confidence: float = 0.6,
    ) -> None:
        self._rsi_oversold = rsi_oversold
        self._rsi_overbought = rsi_overbought
        self._adx_threshold = adx_threshold
        self._min_confidence = min_confidence
        self._last_signal: dict[str, Direction] = {}

    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        candle: Candle,
        features: dict[str, float],
    ) -> Signal | None:
        """Evaluate candle close for momentum signal."""
        ema_10 = features.get("ema_10")
        ema_50 = features.get("ema_50")
        rsi = features.get("rsi_14")
        macd_hist = features.get("macd_12_26_9")
        adx = features.get("adx_14")
        trend_str = features.get("trend_strength", 0)

        # Need all indicators
        if any(v is None for v in [ema_10, ema_50, rsi, macd_hist, adx]):
            return None

        # Trending market filter
        if adx is not None and adx < self._adx_threshold:
            return None

        direction = Direction.FLAT
        confidence = 0.0

        # LONG conditions
        if (
            ema_10 is not None and ema_50 is not None
            and ema_10 > ema_50
            and rsi is not None and self._rsi_oversold < rsi < self._rsi_overbought
            and macd_hist is not None and macd_hist > 0
        ):
            direction = Direction.LONG
            # Confidence scales with indicator agreement
            conf_factors = []
            if adx is not None:
                conf_factors.append(min(adx / 40, 1.0))
            if rsi is not None:
                conf_factors.append(1.0 - abs(rsi - 50) / 50)
            if trend_str is not None:
                conf_factors.append(min(abs(trend_str) / 2, 1.0))
            confidence = sum(conf_factors) / len(conf_factors) if conf_factors else 0.5

        # SHORT conditions
        elif (
            ema_10 is not None and ema_50 is not None
            and ema_10 < ema_50
            and rsi is not None and self._rsi_oversold < rsi < self._rsi_overbought
            and macd_hist is not None and macd_hist < 0
        ):
            direction = Direction.SHORT
            conf_factors = []
            if adx is not None:
                conf_factors.append(min(adx / 40, 1.0))
            if rsi is not None:
                conf_factors.append(1.0 - abs(rsi - 50) / 50)
            if trend_str is not None:
                conf_factors.append(min(abs(trend_str) / 2, 1.0))
            confidence = sum(conf_factors) / len(conf_factors) if conf_factors else 0.5

        # RSI extreme exit
        elif rsi is not None and (rsi > 80 or rsi < 20):
            direction = Direction.FLAT
            confidence = 0.9

        if direction == Direction.FLAT:
            return None

        if confidence < self._min_confidence:
            return None

        # Avoid duplicate signals
        if self._last_signal.get(symbol) == direction:
            return None

        self._last_signal[symbol] = direction

        size_pct = 5.0 + (confidence - self._min_confidence) * 12.5

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            suggested_size_pct=min(size_pct, 10.0),
            strategy_id=self.strategy_id,
            features={
                "rsi": rsi if rsi is not None else 0,
                "macd": macd_hist if macd_hist is not None else 0,
                "adx": adx if adx is not None else 0,
                "ema_cross": (ema_10 - ema_50) / ema_50 * 100 if ema_50 and ema_50 > 0 else 0,
            },
            timestamp=datetime.now(tz=UTC),
        )

    async def on_tick(self, symbol: str, ticker: Ticker) -> Signal | None:
        """Not used in this strategy."""
        return None

    def get_required_features(self) -> list[str]:
        return [
            "ema_10", "ema_50", "rsi_14",
            "macd_12_26_9", "adx_14", "trend_strength",
        ]
