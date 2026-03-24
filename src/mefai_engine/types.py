"""Core type definitions and data transfer objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mefai_engine.constants import (
    Direction,
    ExchangeID,
    ExecutionAlgo,
    MarketRegime,
    OrderStatus,
    OrderType,
    RiskDecisionType,
    Side,
)


@dataclass(slots=True, frozen=True)
class Candle:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float = 0.0
    trades: int = 0


@dataclass(slots=True, frozen=True)
class Ticker:
    """Real-time ticker data."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: float
    timestamp: datetime


@dataclass(slots=True, frozen=True)
class OrderBookLevel:
    """Single order book level."""
    price: float
    quantity: float


@dataclass(slots=True, frozen=True)
class OrderBook:
    """Order book snapshot."""
    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: datetime

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.asks[0].price + self.bids[0].price) / 2
        return 0.0


@dataclass(slots=True, frozen=True)
class FundingRate:
    """Perpetual futures funding rate."""
    symbol: str
    rate: float
    next_funding_time: datetime
    timestamp: datetime


@dataclass(slots=True, frozen=True)
class Balance:
    """Account balance."""
    total: float
    available: float
    unrealized_pnl: float
    margin_used: float
    currency: str = "USDT"


@dataclass(slots=True)
class Position:
    """Open position."""
    symbol: str
    side: Side
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    leverage: int
    liquidation_price: float
    margin: float
    exchange: ExchangeID
    timestamp: datetime


@dataclass(slots=True, frozen=True)
class OrderRequest:
    """Order placement request."""
    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    reduce_only: bool = False
    leverage: int = 1
    client_order_id: str | None = None


@dataclass(slots=True)
class OrderResult:
    """Order execution result."""
    order_id: str
    client_order_id: str
    symbol: str
    side: Side
    order_type: OrderType
    status: OrderStatus
    quantity: float
    filled_quantity: float
    average_price: float
    fee: float
    timestamp: datetime
    exchange: ExchangeID
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Fill:
    """Individual trade fill."""
    order_id: str
    symbol: str
    side: Side
    price: float
    quantity: float
    fee: float
    timestamp: datetime
    exchange: ExchangeID


@dataclass(slots=True, frozen=True)
class Signal:
    """Trading signal from strategy."""
    symbol: str
    direction: Direction
    confidence: float
    suggested_size_pct: float
    strategy_id: str
    model_versions: dict[str, str] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True, frozen=True)
class RiskDecision:
    """Risk manager decision on a signal."""
    decision: RiskDecisionType
    approved_size_pct: float
    reason: str
    checks_passed: list[str] = field(default_factory=list)
    checks_failed: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class Prediction:
    """Model prediction output."""
    direction: Direction
    confidence: float
    magnitude: float
    horizon_seconds: int
    model_id: str
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True, frozen=True)
class MarketState:
    """Current market state snapshot."""
    symbol: str
    ticker: Ticker
    regime: MarketRegime
    features: dict[str, float]
    predictions: list[Prediction]
    positions: list[Position]
    balance: Balance
    timestamp: datetime


@dataclass(slots=True, frozen=True)
class TradingDecision:
    """Final trading decision from agent orchestrator."""
    signal: Signal | None
    risk_decision: RiskDecision | None
    execution_algo: ExecutionAlgo
    reasoning: str
    agent_votes: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BacktestResult:
    """Backtest performance results."""
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration_hours: float
    avg_winner_pct: float
    avg_loser_pct: float
    expectancy: float
    recovery_factor: float
    start_date: datetime
    end_date: datetime
    equity_curve: list[float] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class EquitySnapshot:
    """Point-in-time equity snapshot."""
    timestamp: datetime
    total_equity: float
    unrealized_pnl: float
    realized_pnl: float
    drawdown_pct: float
    position_count: int
