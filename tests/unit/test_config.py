"""Unit tests for configuration loading and validation."""

from __future__ import annotations

import pytest

from mefai_engine.config import load_config, Settings


class TestLoadConfig:
    """Tests for loading config from default.yaml."""

    def test_load_default_config(self) -> None:
        """Loading default config should return a valid Settings object."""
        config = load_config("configs/default.yaml")
        assert isinstance(config, Settings)

    def test_all_sections_exist(self) -> None:
        """All expected config sections should be present."""
        config = load_config("configs/default.yaml")
        assert config.engine is not None
        assert config.exchanges is not None
        assert config.data is not None
        assert config.features is not None
        assert config.models is not None
        assert config.strategy is not None
        assert config.risk is not None
        assert config.execution is not None
        assert config.database is not None
        assert config.monitoring is not None
        assert config.hpo is not None
        assert config.drift is not None
        assert config.kelly is not None
        assert config.correlation is not None
        assert config.var is not None
        assert config.liquidity is not None

    def test_config_types_correct(self) -> None:
        """Config values should have the correct Python types."""
        config = load_config("configs/default.yaml")

        assert isinstance(config.engine.mode, str)
        assert isinstance(config.engine.symbols, list)
        assert isinstance(config.risk.max_position_pct, float)
        assert isinstance(config.risk.max_consecutive_losses, int)
        assert isinstance(config.data.timeframes, list)
        assert isinstance(config.kelly.fraction, float)
        assert isinstance(config.execution.taker_fee_bps, int)

    def test_risk_values_sensible(self) -> None:
        """Risk config values should be within sensible ranges."""
        config = load_config("configs/default.yaml")

        assert 0 < config.risk.max_position_pct <= 100
        assert 0 < config.risk.max_daily_loss_pct <= 100
        assert 0 < config.risk.max_drawdown_pct <= 100
        assert config.risk.max_consecutive_losses > 0

    def test_missing_file_returns_defaults(self) -> None:
        """Loading a missing config file should still return valid defaults."""
        config = load_config("nonexistent_file_that_does_not_exist.yaml")
        assert isinstance(config, Settings)
        assert config.engine.mode == "paper"
