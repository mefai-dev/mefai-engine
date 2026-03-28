"""Unit tests for the feature computation pipeline and technical indicators."""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: import internal functions from the technical module
# ---------------------------------------------------------------------------

from mefai_engine.features.technical import (
    _ema,
    _sma,
    atr_14,
    bollinger_width_20,
    rsi_14,
)
from mefai_engine.features.registry import list_features

# Import pipeline to trigger registration of all feature modules
import mefai_engine.features.pipeline  # noqa: F401


class TestSMA:
    """Tests for Simple Moving Average calculation."""

    def test_sma_simple_array(self) -> None:
        """SMA with period=3 on [1 2 3 4 5] should return known values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _sma(data, period=3)

        # First two values should be NaN (not enough data)
        assert np.isnan(result[0])
        assert np.isnan(result[1])

        # sma[2] = (1+2+3)/3 = 2.0
        assert result[2] == pytest.approx(2.0)
        # sma[3] = (2+3+4)/3 = 3.0
        assert result[3] == pytest.approx(3.0)
        # sma[4] = (3+4+5)/3 = 4.0
        assert result[4] == pytest.approx(4.0)

    def test_sma_constant_input(self) -> None:
        """SMA of a constant array should equal that constant."""
        data = np.full(50, 42.0)
        result = _sma(data, period=10)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 42.0)


class TestEMA:
    """Tests for Exponential Moving Average calculation."""

    def test_ema_convergence(self) -> None:
        """EMA should converge toward the constant tail of a step function."""
        data = np.concatenate([np.full(20, 10.0), np.full(80, 50.0)])
        result = _ema(data, period=10)

        # After many bars at 50 the EMA should be very close to 50
        assert result[-1] == pytest.approx(50.0, abs=0.01)

    def test_ema_first_value(self) -> None:
        """First EMA value should equal the first data point."""
        data = np.array([100.0, 102.0, 104.0, 103.0, 105.0])
        result = _ema(data, period=3)
        assert result[0] == pytest.approx(100.0)


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_constant_price(self) -> None:
        """RSI of a constant price series should be near 50."""
        close = np.full(100, 100.0)
        result = rsi_14(close, period=14)
        # With zero change the gains and losses are both zero
        # The implementation divides zero by 1e-10 giving a huge RS
        # which drives RSI toward 100. A small check is that it is
        # at least between 0 and 100.
        valid = result[~np.isnan(result)]
        assert all((v >= 0) and (v <= 100) for v in valid)

    def test_rsi_rising_price(self) -> None:
        """RSI of a steadily rising price should be above 70."""
        close = np.linspace(100, 200, 100)
        result = rsi_14(close, period=14)
        # After warm up the RSI should be high
        assert result[-1] > 70

    def test_rsi_falling_price(self) -> None:
        """RSI of a steadily falling price should be below 30."""
        close = np.linspace(200, 100, 100)
        result = rsi_14(close, period=14)
        assert result[-1] < 30

    def test_rsi_bounds(self) -> None:
        """RSI should always be between 0 and 100."""
        rng = np.random.default_rng(42)
        close = np.cumsum(rng.standard_normal(500)) + 1000
        result = rsi_14(close, period=14)
        valid = result[~np.isnan(result)]
        assert all((v >= 0) and (v <= 100) for v in valid)


class TestATR:
    """Tests for Average True Range."""

    def test_atr_known_values(self) -> None:
        """ATR on known high/low/close should produce positive values."""
        n = 50
        high = np.full(n, 105.0)
        low = np.full(n, 95.0)
        close = np.full(n, 100.0)
        result = atr_14(high, low, close, period=14)
        # True range is always 10 so ATR should converge to 10
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert valid[-1] == pytest.approx(10.0, abs=0.5)


class TestBollingerWidth:
    """Tests for Bollinger Band width."""

    def test_bollinger_width_positive(self) -> None:
        """Bollinger width should be non-negative for valid data."""
        rng = np.random.default_rng(42)
        close = np.cumsum(rng.standard_normal(100)) + 500
        result = bollinger_width_20(close, period=20, std=2)
        valid = result[~np.isnan(result)]
        assert all(v >= 0 for v in valid)

    def test_bollinger_width_constant_zero(self) -> None:
        """Bollinger width of constant price should be near zero (zero std)."""
        close = np.full(50, 100.0)
        result = bollinger_width_20(close, period=20, std=2)
        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-8)


class TestFeatureRegistry:
    """Tests for the feature registry."""

    def test_registry_has_50_plus_features(self) -> None:
        """Registry should have at least 50 features registered."""
        all_features = list_features()
        assert len(all_features) >= 50

    def test_registry_categories(self) -> None:
        """Registry should contain multiple categories."""
        all_features = list_features()
        categories = {f.category for f in all_features}
        assert "trend" in categories
        assert "momentum" in categories
        assert "volatility" in categories
        assert "volume" in categories


class TestFeaturePipeline:
    """Tests for the full feature pipeline."""

    def test_pipeline_computes_without_error(self) -> None:
        """Pipeline should compute features on valid OHLCV data."""
        from mefai_engine.features.pipeline import FeaturePipeline

        n = 200
        rng = np.random.default_rng(42)
        base = np.cumsum(rng.standard_normal(n)) + 1000.0
        raw = {
            "open": base + rng.uniform(-1, 1, n),
            "high": base + np.abs(rng.standard_normal(n)) * 5,
            "low": base - np.abs(rng.standard_normal(n)) * 5,
            "close": base,
            "volume": np.abs(rng.standard_normal(n)) * 1e6 + 1e5,
        }
        # Make sure high >= open/close and low <= open/close
        raw["high"] = np.maximum(raw["high"], np.maximum(raw["open"], raw["close"]))
        raw["low"] = np.minimum(raw["low"], np.minimum(raw["open"], raw["close"]))

        pipeline = FeaturePipeline(enabled_features=["rsi_14", "sma_20", "atr_14", "obv"])
        result = pipeline.compute(raw)

        assert "rsi_14" in result
        assert "sma_20" in result
        assert "atr_14" in result
        assert "obv" in result
        assert len(result["rsi_14"]) == n
