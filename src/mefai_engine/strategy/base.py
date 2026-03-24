"""Abstract base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mefai_engine.types import Candle, Signal, Ticker


class BaseStrategy(ABC):
    """Base class for all trading strategies.

    Strategies receive market data and produce trading signals.
    They should be stateless where possible - state lives in features.
    """

    strategy_id: str
    symbols: list[str]
    timeframes: list[str]

    @abstractmethod
    async def on_candle(
        self,
        symbol: str,
        timeframe: str,
        candle: Candle,
        features: dict[str, float],
    ) -> Signal | None:
        """Called when a new candle closes.

        This is the primary signal generation method.
        Returns a Signal if the strategy wants to trade, None otherwise.
        """
        ...

    @abstractmethod
    async def on_tick(self, symbol: str, ticker: Ticker) -> Signal | None:
        """Called on each new tick. For tick-sensitive strategies only.

        Most strategies should return None here and only use on_candle.
        """
        ...

    def get_required_features(self) -> list[str]:
        """Return list of feature names this strategy needs."""
        return []

    def get_required_timeframes(self) -> list[str]:
        """Return list of timeframes this strategy uses."""
        return self.timeframes
