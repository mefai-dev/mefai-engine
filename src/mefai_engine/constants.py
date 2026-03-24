"""Global constants and enumerations."""

from enum import StrEnum


class EngineMode(StrEnum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


class Side(StrEnum):
    LONG = "long"
    SHORT = "short"


class Direction(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    TAKE_PROFIT_MARKET = "take_profit_market"


class OrderStatus(StrEnum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class ExchangeID(StrEnum):
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


class ExecutionAlgo(StrEnum):
    MARKET = "market"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    LIMIT_CHASE = "limit_chase"


class RiskDecisionType(StrEnum):
    APPROVED = "approved"
    REDUCED = "reduced"
    REJECTED = "rejected"


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class MarketRegime(StrEnum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


# Timeframe to seconds mapping
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

# Default fee rates (basis points)
DEFAULT_MAKER_FEE_BPS = 2
DEFAULT_TAKER_FEE_BPS = 5

# Risk defaults
DEFAULT_MAX_POSITION_PCT = 10.0
DEFAULT_MAX_DRAWDOWN_PCT = 10.0
DEFAULT_MAX_DAILY_LOSS_PCT = 3.0
DEFAULT_CIRCUIT_BREAKER_COOLDOWN = 3600
