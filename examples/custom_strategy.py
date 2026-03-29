"""Custom strategy example: Dual EMA Crossover.

This script demonstrates how to create a custom trading strategy by
extending the BaseStrategy abstract class. The strategy implements a
simple dual EMA crossover with trend confirmation.

Each method is documented to explain its role in the strategy lifecycle.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import numpy as np

from mefai_engine.constants import Direction
from mefai_engine.features.technical import _ema
from mefai_engine.strategy.base import BaseStrategy
from mefai_engine.types import Candle, Signal, Ticker


class DualEMACrossover(BaseStrategy):
    """Dual EMA crossover strategy.

    Enters long when the fast EMA crosses above the slow EMA.
    Enters short when the fast EMA crosses below the slow EMA.
    Uses a minimum separation threshold to avoid whipsaw in tight ranges.

    This is a classic trend following approach that works best in
    strongly directional markets and struggles during consolidation.
    """

    strategy_id = "dual_ema_crossover"
    symbols = ["BTCUSDT"]
    timeframes = ["1h"]

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 50,
        min_separation_pct: float = 0.1,
    ) -> None:
        """Initialize the strategy with EMA parameters.

        Args:
            fast_period: Period for the fast EMA (shorter lookback).
            slow_period: Period for the slow EMA (longer lookback).
            min_separation_pct: Minimum percentage gap between EMAs
                to trigger a signal. Helps filter false crossovers.
        """
        self._fast_period = fast_period
        self._slow_period = slow_period
        self._min_separation = min_separation_pct
        self._close_history: list[float] = []
        self._last_direction: Direction = Direction.FLAT

    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        candle: Candle,
        features: dict[str, float],
    ) -> Signal | None:
        """Process a new candle and decide whether to emit a signal.

        This is the primary signal generation method. It is called once
        each time a candle closes on the configured timeframe.

        Args:
            symbol: The trading pair (e.g. BTCUSDT).
            timeframe: The candle timeframe (e.g. 1h).
            candle: The newly closed OHLCV candle.
            features: Pre-computed feature values for this bar.

        Returns:
            A Signal if a crossover is detected with sufficient
            separation. None if no trade is warranted.
        """
        self._close_history.append(candle.close)

        # Need enough data for the slow EMA
        if len(self._close_history) < self._slow_period + 5:
            return None

        closes = np.array(self._close_history)
        fast_ema = _ema(closes, self._fast_period)
        slow_ema = _ema(closes, self._slow_period)

        # Current values
        fast_now = fast_ema[-1]
        slow_now = slow_ema[-1]
        separation_pct = abs(fast_now - slow_now) / slow_now * 100

        # Check if separation is large enough
        if separation_pct < self._min_separation:
            return None

        # Determine direction
        if fast_now > slow_now:
            new_direction = Direction.LONG
        else:
            new_direction = Direction.SHORT

        # Only emit on direction change
        if new_direction == self._last_direction:
            return None

        self._last_direction = new_direction

        # Confidence scales with EMA separation
        confidence = min(0.6 + separation_pct * 0.1, 0.95)

        return Signal(
            symbol=symbol,
            direction=new_direction,
            confidence=confidence,
            suggested_size_pct=min(confidence * 10, 10.0),
            strategy_id=self.strategy_id,
            model_versions={"ema_crossover": "1.0"},
            timestamp=datetime.now(tz=UTC),
        )

    async def on_tick(self, symbol: str, ticker: Ticker) -> Signal | None:
        """Process a real-time tick update.

        This method is called on every tick. Most candle based strategies
        should return None here and only use on_candle for signal generation.
        Tick handlers are useful for strategies that need sub-candle precision
        such as scalping or market making.

        Args:
            symbol: The trading pair.
            ticker: The latest ticker data with bid/ask/last.

        Returns:
            None for this candle based strategy.
        """
        return None

    def get_required_features(self) -> list[str]:
        """Declare which pre-computed features this strategy needs.

        The feature pipeline will ensure these are available before
        on_candle is called. This strategy computes its own EMAs
        from raw close prices so it does not depend on external features.
        However it uses ADX for optional filtering.

        Returns:
            List of feature name strings.
        """
        return ["adx_14"]

    def get_required_timeframes(self) -> list[str]:
        """Declare which candle timeframes this strategy subscribes to.

        The data manager will ensure candle data for these timeframes
        is collected and delivered.

        Returns:
            List of timeframe strings.
        """
        return ["1h"]


async def demo() -> None:
    """Run a quick demonstration of the custom strategy."""
    strategy = DualEMACrossover(fast_period=10, slow_period=50)

    # Simulate candle feed with rising then falling prices
    rng = np.random.default_rng(123)
    prices_up = np.linspace(40000, 48000, 60) + rng.normal(0, 100, 60)
    prices_down = np.linspace(48000, 42000, 40) + rng.normal(0, 100, 40)
    all_prices = np.concatenate([prices_up, prices_down])

    print("=" * 60)
    print("  Dual EMA Crossover Strategy Demo")
    print("=" * 60)

    for i in range(len(all_prices)):
        p = float(all_prices[i])
        candle = Candle(
            timestamp=datetime.now(tz=UTC),
            open=p * 0.999,
            high=p * 1.005,
            low=p * 0.995,
            close=p,
            volume=1000.0,
        )

        signal = await strategy.on_candle(
            symbol="BTCUSDT",
            timeframe="1h",
            candle=candle,
            features={"adx_14": 30.0},
        )

        if signal is not None:
            print(
                f"  Bar {i:3d} | Price: {p:>10.2f} | "
                f"Direction: {signal.direction:>5s} | "
                f"Confidence: {signal.confidence:.3f}"
            )

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo())
