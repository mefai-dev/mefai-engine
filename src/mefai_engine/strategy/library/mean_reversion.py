"""Mean reversion strategy - Bollinger Band + RSI extreme entries.

Enters when price is at statistical extremes and likely to revert to mean.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from mefai_engine.constants import Direction
from mefai_engine.strategy.base import BaseStrategy
from mefai_engine.types import Candle, Signal, Ticker

logger = structlog.get_logger()


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI.

    Entry conditions (LONG):
    - Price below lower Bollinger Band
    - RSI below 25 (heavily oversold)
    - ADX below 25 (ranging market)
    - Positive funding rate (shorts paying = potential short squeeze)

    Entry conditions (SHORT):
    - Price above upper Bollinger Band
    - RSI above 75 (heavily overbought)
    - ADX below 25 (ranging market)
    - Negative funding rate (longs paying)

    Tight stop-loss as mean reversion has lower win rate but controlled risk.
    """

    strategy_id = "mean_reversion_v1"
    symbols = ["BTCUSDT"]
    timeframes = ["1h"]

    def __init__(
        self,
        rsi_extreme_low: float = 25.0,
        rsi_extreme_high: float = 75.0,
        adx_max: float = 25.0,
    ) -> None:
        self._rsi_low = rsi_extreme_low
        self._rsi_high = rsi_extreme_high
        self._adx_max = adx_max
        self._last_signal: dict[str, Direction] = {}

    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        candle: Candle,
        features: dict[str, float],
    ) -> Signal | None:
        """Check for mean reversion setup."""
        rsi = features.get("rsi_14")
        bb_upper = features.get("bollinger_upper_20")
        bb_lower = features.get("bollinger_lower_20")
        adx = features.get("adx_14")
        z_score = features.get("mean_reversion_score")
        funding = features.get("funding_rate", 0)

        if any(v is None for v in [rsi, bb_upper, bb_lower]):
            return None

        # Only in ranging markets
        if adx is not None and adx > self._adx_max:
            return None

        price = candle.close
        direction = Direction.FLAT
        confidence = 0.0

        # LONG: oversold bounce
        if (
            rsi is not None and rsi < self._rsi_low
            and bb_lower is not None and price <= bb_lower
        ):
            direction = Direction.LONG
            # More oversold = higher confidence
            confidence = min((self._rsi_low - rsi) / self._rsi_low + 0.5, 0.95)
            if funding is not None and funding > 0:
                confidence += 0.05  # Shorts paying = squeeze potential

        # SHORT: overbought rejection
        elif (
            rsi is not None and rsi > self._rsi_high
            and bb_upper is not None and price >= bb_upper
        ):
            direction = Direction.SHORT
            confidence = min((rsi - self._rsi_high) / (100 - self._rsi_high) + 0.5, 0.95)
            if funding is not None and funding < 0:
                confidence += 0.05

        if direction == Direction.FLAT:
            return None

        if self._last_signal.get(symbol) == direction:
            return None

        self._last_signal[symbol] = direction

        # Smaller size for mean reversion (higher risk)
        size_pct = 3.0 + confidence * 4.0

        return Signal(
            symbol=symbol,
            direction=direction,
            confidence=min(confidence, 1.0),
            suggested_size_pct=min(size_pct, 7.0),
            strategy_id=self.strategy_id,
            features={
                "rsi": rsi if rsi is not None else 0,
                "z_score": z_score if z_score is not None else 0,
                "bb_position": (price - (bb_lower or 0)) / ((bb_upper or 1) - (bb_lower or 0)) if bb_upper and bb_lower and bb_upper != bb_lower else 0.5,
                "funding": funding if funding is not None else 0,
            },
            timestamp=datetime.now(tz=UTC),
        )

    async def on_tick(self, symbol: str, ticker: Ticker) -> Signal | None:
        return None

    def get_required_features(self) -> list[str]:
        return [
            "rsi_14", "bollinger_upper_20", "bollinger_lower_20",
            "adx_14", "mean_reversion_score", "funding_rate",
        ]
