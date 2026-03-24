"""Configuration management with YAML + environment variable support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from mefai_engine.constants import EngineMode, ExecutionAlgo


class ExchangeConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    secret: str = ""
    passphrase: str = ""
    testnet: bool = False
    rate_limit_per_second: int = 10


class ExchangesConfig(BaseModel):
    binance: ExchangeConfig = Field(default_factory=ExchangeConfig)
    okx: ExchangeConfig = Field(default_factory=ExchangeConfig)
    bybit: ExchangeConfig = Field(default_factory=ExchangeConfig)


class DataConfig(BaseModel):
    timeframes: list[str] = ["1m", "5m", "15m", "1h", "4h", "1d"]
    orderbook_depth: int = 20
    history_days: int = 365
    collection_interval_ms: int = 1000


class FeaturesConfig(BaseModel):
    enabled: list[str] = [
        "rsi_14", "macd_12_26_9", "bollinger_20_2", "atr_14",
        "obv", "vwap", "funding_rate", "oi_delta",
        "ema_10", "ema_20", "ema_50", "adx_14",
        "stoch_14_3", "cci_20", "williams_r_14",
        "volume_sma_20", "book_imbalance",
    ]


class GradientBoostConfig(BaseModel):
    enabled: bool = True
    retrain_interval_hours: int = 24
    lookback_window: int = 500
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05


class TransformerConfig(BaseModel):
    enabled: bool = True
    model_path: str = "models/tft_v1.pt"
    sequence_length: int = 128
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 64


class RLConfig(BaseModel):
    enabled: bool = True
    model_path: str = "models/ppo_v1.zip"
    retrain_interval_hours: int = 168
    total_timesteps: int = 500_000
    learning_rate: float = 0.0003


class SentimentConfig(BaseModel):
    enabled: bool = True
    model_name: str = "ProsusAI/finbert"
    news_sources: list[str] = ["cryptopanic", "rss"]
    update_interval_minutes: int = 15
    cache_ttl_minutes: int = 60


class ModelsConfig(BaseModel):
    gradient_boost: GradientBoostConfig = Field(default_factory=GradientBoostConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    sentiment: SentimentConfig = Field(default_factory=SentimentConfig)


class MetaLearnerConfig(BaseModel):
    min_confidence: float = 0.65
    regime_filter: bool = True
    max_signals_per_hour: int = 4


class StrategyConfig(BaseModel):
    meta_learner: MetaLearnerConfig = Field(default_factory=MetaLearnerConfig)


class RiskConfig(BaseModel):
    max_position_pct: float = 10.0
    max_total_exposure_pct: float = 30.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_consecutive_losses: int = 5
    circuit_breaker_cooldown_seconds: int = 3600
    funding_rate_cost_cap_pct: float = 0.1


class ExecutionConfig(BaseModel):
    default_algorithm: ExecutionAlgo = ExecutionAlgo.TWAP
    twap_duration_seconds: int = 120
    slippage_tolerance_bps: int = 10
    maker_fee_bps: int = 2
    taker_fee_bps: int = 5
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class DatabaseConfig(BaseModel):
    timescaledb_url: str = "postgresql+asyncpg://mefai:mefai@localhost:5432/mefai_engine"
    redis_url: str = "redis://localhost:6379/0"
    pool_size: int = 10
    pool_overflow: int = 20


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""


class MonitoringConfig(BaseModel):
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    metrics_port: int = 9090
    equity_snapshot_interval_seconds: int = 300
    health_check_interval_seconds: int = 30


class EngineConfig(BaseModel):
    mode: EngineMode = EngineMode.PAPER
    log_level: str = "INFO"
    symbols: list[str] = ["BTCUSDT"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEFAI_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    engine: EngineConfig = Field(default_factory=EngineConfig)
    exchanges: ExchangesConfig = Field(default_factory=ExchangesConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_config(config_path: str | Path | None = None) -> Settings:
    """Load configuration from YAML file with environment variable overrides."""
    if config_path is None:
        config_path = Path("configs/default.yaml")

    config_path = Path(config_path)
    yaml_data: dict[str, Any] = {}

    if config_path.exists():
        with open(config_path) as f:
            yaml_data = yaml.safe_load(f) or {}

    return Settings(**yaml_data)
