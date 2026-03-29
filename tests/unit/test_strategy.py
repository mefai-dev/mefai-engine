"""Unit tests for signal generation and meta-learner."""

from __future__ import annotations

from datetime import UTC, datetime

from mefai_engine.constants import Direction, MarketRegime
from mefai_engine.strategy.meta_learner import MetaLearner
from mefai_engine.strategy.signal import predictions_to_signal
from mefai_engine.types import Prediction


def _make_prediction(
    direction: Direction,
    confidence: float = 0.9,
    model_id: str = "test_model",
) -> Prediction:
    """Helper to create a Prediction instance."""
    return Prediction(
        direction=direction,
        confidence=confidence,
        magnitude=0.01,
        horizon_seconds=3600,
        model_id=model_id,
        model_version="1.0",
        timestamp=datetime.now(tz=UTC),
    )


class TestPredictionsToSignal:
    """Tests for predictions_to_signal consensus logic."""

    def test_unanimous_long(self) -> None:
        """All models predicting LONG should produce a LONG signal."""
        preds = [
            _make_prediction(Direction.LONG, 0.9, "model_a"),
            _make_prediction(Direction.LONG, 0.85, "model_b"),
            _make_prediction(Direction.LONG, 0.95, "model_c"),
        ]
        signal = predictions_to_signal(
            preds,
            symbol="BTCUSDT",
            strategy_id="test",
            min_confidence=0.65,
        )
        assert signal is not None
        assert signal.direction == Direction.LONG
        assert signal.confidence > 0.65

    def test_below_confidence_returns_none(self) -> None:
        """Signal should be None if confidence is below threshold."""
        preds = [
            _make_prediction(Direction.LONG, 0.5, "model_a"),
            _make_prediction(Direction.SHORT, 0.4, "model_b"),
            _make_prediction(Direction.FLAT, 0.3, "model_c"),
        ]
        signal = predictions_to_signal(
            preds,
            symbol="BTCUSDT",
            strategy_id="test",
            min_confidence=0.90,
        )
        assert signal is None

    def test_empty_predictions_returns_none(self) -> None:
        """Empty prediction list should return None."""
        signal = predictions_to_signal(
            [],
            symbol="BTCUSDT",
            strategy_id="test",
        )
        assert signal is None

    def test_mixed_predictions_no_consensus(self) -> None:
        """Evenly split predictions should return None (no agreement)."""
        preds = [
            _make_prediction(Direction.LONG, 0.5, "model_a"),
            _make_prediction(Direction.SHORT, 0.5, "model_b"),
        ]
        signal = predictions_to_signal(
            preds,
            symbol="BTCUSDT",
            strategy_id="test",
            min_confidence=0.65,
            min_agreement=0.6,
        )
        assert signal is None


class TestMetaLearner:
    """Tests for the meta-learner ensemble combiner."""

    def test_rate_limiting(self) -> None:
        """MetaLearner should stop producing signals after rate limit is hit."""
        ml = MetaLearner(
            min_confidence=0.5,
            regime_filter=False,
            max_signals_per_hour=2,
        )

        preds = [
            _make_prediction(Direction.LONG, 0.9, "model_a"),
            _make_prediction(Direction.LONG, 0.9, "model_b"),
        ]

        # First two should succeed
        s1 = ml.evaluate(preds, MarketRegime.TRENDING_UP, "BTCUSDT")
        s2 = ml.evaluate(preds, MarketRegime.TRENDING_UP, "BTCUSDT")
        assert s1 is not None
        assert s2 is not None

        # Third should be rate limited
        s3 = ml.evaluate(preds, MarketRegime.TRENDING_UP, "BTCUSDT")
        assert s3 is None

    def test_evaluate_with_regime_filter(self) -> None:
        """MetaLearner should apply regime weights when filter is enabled."""
        ml = MetaLearner(
            min_confidence=0.5,
            regime_filter=True,
            max_signals_per_hour=100,
        )

        preds = [
            _make_prediction(Direction.LONG, 0.8, "gradient_boost"),
            _make_prediction(Direction.LONG, 0.8, "transformer"),
        ]

        signal = ml.evaluate(preds, MarketRegime.TRENDING_UP, "BTCUSDT")
        assert signal is not None
        assert signal.direction == Direction.LONG
