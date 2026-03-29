"""Unit tests for core type definitions and data transfer objects."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mefai_engine.constants import Direction, RiskDecisionType
from mefai_engine.types import Candle, Prediction, RiskDecision, Signal


class TestSignalCreation:
    """Tests for Signal dataclass creation."""

    def test_signal_basic(self) -> None:
        """Signal should store all fields correctly."""
        sig = Signal(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            confidence=0.85,
            suggested_size_pct=5.0,
            strategy_id="test_strategy",
        )
        assert sig.symbol == "BTCUSDT"
        assert sig.direction == Direction.LONG
        assert sig.confidence == 0.85
        assert sig.suggested_size_pct == 5.0

    def test_signal_is_frozen(self) -> None:
        """Signal should be immutable (frozen dataclass)."""
        sig = Signal(
            symbol="ETHUSDT",
            direction=Direction.SHORT,
            confidence=0.7,
            suggested_size_pct=3.0,
            strategy_id="test",
        )
        with pytest.raises(AttributeError):
            sig.confidence = 0.9  # type: ignore[misc]


class TestPredictionCreation:
    """Tests for Prediction dataclass creation."""

    def test_prediction_basic(self) -> None:
        """Prediction should store model output fields."""
        pred = Prediction(
            direction=Direction.LONG,
            confidence=0.8,
            magnitude=0.02,
            horizon_seconds=3600,
            model_id="gradient_boost_v1",
            model_version="1.0.0",
        )
        assert pred.direction == Direction.LONG
        assert pred.confidence == 0.8
        assert pred.model_id == "gradient_boost_v1"

    def test_prediction_has_timestamp(self) -> None:
        """Prediction should auto-assign a timestamp."""
        pred = Prediction(
            direction=Direction.SHORT,
            confidence=0.6,
            magnitude=0.01,
            horizon_seconds=300,
            model_id="test",
            model_version="0.1",
        )
        assert isinstance(pred.timestamp, datetime)


class TestRiskDecisionDefaults:
    """Tests for RiskDecision default values."""

    def test_defaults(self) -> None:
        """RiskDecision should have empty lists as defaults."""
        rd = RiskDecision(
            decision=RiskDecisionType.APPROVED,
            approved_size_pct=5.0,
            reason="All checks passed",
        )
        assert rd.checks_passed == []
        assert rd.checks_failed == []
        assert rd.decision == RiskDecisionType.APPROVED

    def test_with_checks(self) -> None:
        """RiskDecision should accept check lists."""
        rd = RiskDecision(
            decision=RiskDecisionType.REDUCED,
            approved_size_pct=3.0,
            reason="Position size capped",
            checks_passed=["daily_loss", "drawdown"],
            checks_failed=["position_size"],
        )
        assert len(rd.checks_passed) == 2
        assert len(rd.checks_failed) == 1


class TestCandleFrozen:
    """Tests for Candle immutability."""

    def test_candle_creation(self) -> None:
        """Candle should store OHLCV data."""
        ts = datetime.now(tz=UTC)
        c = Candle(
            timestamp=ts,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )
        assert c.open == 100.0
        assert c.high == 105.0
        assert c.volume == 1000.0

    def test_candle_is_frozen(self) -> None:
        """Candle should be immutable."""
        ts = datetime.now(tz=UTC)
        c = Candle(
            timestamp=ts,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0,
        )
        with pytest.raises(AttributeError):
            c.close = 999.0  # type: ignore[misc]
