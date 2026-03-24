"""Meta-learner: combines signals from multiple models with regime awareness."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

import structlog

from mefai_engine.constants import Direction, MarketRegime
from mefai_engine.strategy.signal import predictions_to_signal
from mefai_engine.types import Prediction, Signal

logger = structlog.get_logger()


class MetaLearner:
    """Ensemble meta-learner that combines multiple model predictions.

    Key responsibilities:
    1. Collect predictions from all active models
    2. Apply regime-based weighting (trust different models in different regimes)
    3. Filter based on minimum confidence and agreement
    4. Rate limit signal generation
    5. Produce final consolidated signal
    """

    # Regime-specific model trust weights
    # Higher weight = more trust in this model for this regime
    REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
        MarketRegime.TRENDING_UP: {
            "gradient_boost": 1.0,
            "transformer": 1.2,
            "rl": 0.8,
            "sentiment": 1.0,
        },
        MarketRegime.TRENDING_DOWN: {
            "gradient_boost": 1.0,
            "transformer": 1.2,
            "rl": 0.8,
            "sentiment": 1.1,
        },
        MarketRegime.RANGING: {
            "gradient_boost": 1.2,
            "transformer": 0.7,
            "rl": 1.0,
            "sentiment": 0.6,
        },
        MarketRegime.HIGH_VOLATILITY: {
            "gradient_boost": 0.8,
            "transformer": 0.6,
            "rl": 1.3,
            "sentiment": 1.2,
        },
        MarketRegime.LOW_VOLATILITY: {
            "gradient_boost": 1.0,
            "transformer": 0.9,
            "rl": 0.7,
            "sentiment": 0.5,
        },
    }

    def __init__(
        self,
        min_confidence: float = 0.65,
        regime_filter: bool = True,
        max_signals_per_hour: int = 4,
    ) -> None:
        self._min_confidence = min_confidence
        self._regime_filter = regime_filter
        self._max_signals_per_hour = max_signals_per_hour
        self._recent_signals: deque[datetime] = deque(maxlen=100)

    def evaluate(
        self,
        predictions: list[Prediction],
        regime: MarketRegime,
        symbol: str,
        strategy_id: str = "meta_learner",
    ) -> Signal | None:
        """Evaluate predictions and produce a consolidated signal.

        Args:
            predictions: Predictions from various models
            regime: Current market regime
            symbol: Trading pair
            strategy_id: Identifier for this strategy

        Returns:
            Signal if all conditions met, None otherwise
        """
        if not predictions:
            return None

        # Rate limit check
        if not self._check_rate_limit():
            logger.debug("meta_learner.rate_limited")
            return None

        # Apply regime-based weighting
        if self._regime_filter:
            predictions = self._apply_regime_weights(predictions, regime)

        # Generate signal
        signal = predictions_to_signal(
            predictions=predictions,
            symbol=symbol,
            strategy_id=strategy_id,
            min_confidence=self._min_confidence,
        )

        if signal is not None:
            self._recent_signals.append(datetime.now(tz=timezone.utc))
            logger.info(
                "meta_learner.signal",
                symbol=symbol,
                direction=signal.direction,
                confidence=f"{signal.confidence:.3f}",
                regime=regime,
            )

        return signal

    def _apply_regime_weights(
        self, predictions: list[Prediction], regime: MarketRegime
    ) -> list[Prediction]:
        """Adjust prediction confidence based on regime-model trust weights."""
        weights = self.REGIME_WEIGHTS.get(regime, {})
        adjusted: list[Prediction] = []

        for pred in predictions:
            model_key = pred.model_id.split("_")[0]  # e.g., "gradient_boost_v1" -> "gradient_boost"
            weight = weights.get(model_key, 1.0)
            adjusted_confidence = min(pred.confidence * weight, 1.0)

            adjusted.append(Prediction(
                direction=pred.direction,
                confidence=adjusted_confidence,
                magnitude=pred.magnitude,
                horizon_seconds=pred.horizon_seconds,
                model_id=pred.model_id,
                model_version=pred.model_version,
                timestamp=pred.timestamp,
            ))

        return adjusted

    def _check_rate_limit(self) -> bool:
        """Check if we're within signal rate limit."""
        now = datetime.now(tz=timezone.utc)
        cutoff = now.timestamp() - 3600
        recent = sum(1 for t in self._recent_signals if t.timestamp() > cutoff)
        return recent < self._max_signals_per_hour
