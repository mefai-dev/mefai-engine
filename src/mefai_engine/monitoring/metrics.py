"""Prometheus metrics for the MEFAI trading engine.

Exposes counters and gauges and histograms for all critical trading
operations. Metrics are collected in process and scraped by Prometheus
via the /metrics endpoint.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

import structlog

logger = structlog.get_logger()

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class TradingMetrics:
    """Central metrics registry for the trading engine.

    All Prometheus metrics are defined here so they can be imported
    and updated from anywhere in the codebase.
    """

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        if not HAS_PROMETHEUS:
            logger.warning("monitoring.prometheus_not_installed")
            self._enabled = False
            return

        self._enabled = True
        self._registry = registry or CollectorRegistry()

        # Trade counters
        self.trades_total = Counter(
            "mefai_trades_total",
            "Total number of trades executed",
            labelnames=["symbol", "direction", "outcome"],
            registry=self._registry,
        )

        self.signals_total = Counter(
            "mefai_signals_total",
            "Total number of signals generated",
            labelnames=["symbol", "direction", "source"],
            registry=self._registry,
        )

        self.orders_total = Counter(
            "mefai_orders_total",
            "Total number of orders placed",
            labelnames=["symbol", "side", "order_type", "status"],
            registry=self._registry,
        )

        self.risk_rejections_total = Counter(
            "mefai_risk_rejections_total",
            "Total number of risk rejections",
            labelnames=["reason"],
            registry=self._registry,
        )

        # Latency histograms
        self.signal_latency_seconds = Histogram(
            "mefai_signal_latency_seconds",
            "Time from data receipt to signal generation",
            labelnames=["model"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
            registry=self._registry,
        )

        self.order_latency_seconds = Histogram(
            "mefai_order_latency_seconds",
            "Time from signal to order execution",
            labelnames=["exchange"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self._registry,
        )

        self.model_inference_seconds = Histogram(
            "mefai_model_inference_seconds",
            "Model inference duration",
            labelnames=["model"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self._registry,
        )

        # Portfolio gauges
        self.equity_usdt = Gauge(
            "mefai_equity_usdt",
            "Current portfolio equity in USDT",
            registry=self._registry,
        )

        self.drawdown_pct = Gauge(
            "mefai_drawdown_pct",
            "Current drawdown percentage from peak equity",
            registry=self._registry,
        )

        self.win_rate = Gauge(
            "mefai_win_rate",
            "Rolling win rate over recent trades",
            registry=self._registry,
        )

        self.open_positions_count = Gauge(
            "mefai_open_positions_count",
            "Number of currently open positions",
            registry=self._registry,
        )

        self.total_exposure_pct = Gauge(
            "mefai_total_exposure_pct",
            "Total portfolio exposure as percentage of equity",
            registry=self._registry,
        )

        self.unrealized_pnl_usdt = Gauge(
            "mefai_unrealized_pnl_usdt",
            "Total unrealized PnL across all positions",
            registry=self._registry,
        )

        self.daily_pnl_usdt = Gauge(
            "mefai_daily_pnl_usdt",
            "Realized PnL for the current day",
            registry=self._registry,
        )

        # Circuit breaker state (0 = closed / trading allowed | 1 = open / halted)
        self.circuit_breaker_state = Gauge(
            "mefai_circuit_breaker_state",
            "Circuit breaker state: 0=closed 1=open",
            registry=self._registry,
        )

        # Data freshness
        self.last_candle_timestamp = Gauge(
            "mefai_last_candle_timestamp",
            "Unix timestamp of the most recent candle received",
            labelnames=["symbol", "timeframe"],
            registry=self._registry,
        )

        self.websocket_connected = Gauge(
            "mefai_websocket_connected",
            "WebSocket connection state per exchange",
            labelnames=["exchange"],
            registry=self._registry,
        )

        # Regime
        self.current_regime = Gauge(
            "mefai_current_regime",
            "Current detected market regime (encoded as int)",
            labelnames=["symbol"],
            registry=self._registry,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled

    def generate_metrics(self) -> bytes:
        """Generate Prometheus exposition format output."""
        if not self._enabled:
            return b"# prometheus-client not installed\n"
        return generate_latest(self._registry)

    @contextmanager
    def measure_latency(
        self, histogram_name: str, **labels: str
    ) -> Generator[None, None, None]:
        """Context manager to measure and record latency.

        Usage:
            with metrics.measure_latency("signal_latency_seconds", model="xgboost"):
                result = model.predict(features)
        """
        if not self._enabled:
            yield
            return

        hist = getattr(self, histogram_name, None)
        if hist is None:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            hist.labels(**labels).observe(elapsed)

    def record_trade(
        self,
        symbol: str,
        direction: str,
        outcome: str,
    ) -> None:
        """Record a completed trade."""
        if not self._enabled:
            return
        self.trades_total.labels(
            symbol=symbol, direction=direction, outcome=outcome
        ).inc()

    def record_signal(
        self, symbol: str, direction: str, source: str
    ) -> None:
        """Record a generated signal."""
        if not self._enabled:
            return
        self.signals_total.labels(
            symbol=symbol, direction=direction, source=source
        ).inc()

    def update_equity(self, equity: float) -> None:
        """Update current equity gauge."""
        if not self._enabled:
            return
        self.equity_usdt.set(equity)

    def update_drawdown(self, dd_pct: float) -> None:
        """Update current drawdown gauge."""
        if not self._enabled:
            return
        self.drawdown_pct.set(dd_pct)

    def update_win_rate(self, rate: float) -> None:
        """Update rolling win rate gauge."""
        if not self._enabled:
            return
        self.win_rate.set(rate)

    def update_circuit_breaker(self, is_open: bool) -> None:
        """Update circuit breaker state."""
        if not self._enabled:
            return
        self.circuit_breaker_state.set(1.0 if is_open else 0.0)


# Singleton instance for global access
_metrics: TradingMetrics | None = None


def get_metrics() -> TradingMetrics:
    """Get the global metrics instance (lazy initialized)."""
    global _metrics
    if _metrics is None:
        _metrics = TradingMetrics()
    return _metrics


def reset_metrics() -> None:
    """Reset the global metrics instance (for testing)."""
    global _metrics
    _metrics = None
