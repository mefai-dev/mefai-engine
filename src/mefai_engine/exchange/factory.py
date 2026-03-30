"""Exchange factory with circuit breaker pattern."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import structlog

from mefai_engine.config import ExchangesConfig
from mefai_engine.constants import CircuitState, ExchangeID
from mefai_engine.exceptions import ExchangeConnectionError
from mefai_engine.exchange.base import BaseExchange
from mefai_engine.exchange.binance import BinanceExchange
from mefai_engine.exchange.models import ExchangeCredentials

logger = structlog.get_logger()

_REGISTRY: dict[ExchangeID, type[BaseExchange]] = {
    ExchangeID.BINANCE: BinanceExchange,
}


class CircuitBreaker:
    """Circuit breaker for exchange connections.

    CLOSED  -> normal operation
    OPEN    -> all calls blocked, auto-recovery after cooldown
    HALF_OPEN -> one test call allowed
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: int = 60) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._opened_at: datetime | None = None

    def record_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self._opened_at = datetime.now(tz=UTC)
            logger.warning(
                "circuit_breaker.opened",
                failures=self.failure_count,
                cooldown=self.cooldown_seconds,
            )

    def record_success(self) -> None:
        self.failure_count = 0
        if self.state != CircuitState.CLOSED:
            logger.info("circuit_breaker.closed")
        self.state = CircuitState.CLOSED

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN and self._opened_at:
            elapsed = (datetime.now(tz=UTC) - self._opened_at).total_seconds()
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                logger.info("circuit_breaker.half_open")
                return True
        if self.state == CircuitState.HALF_OPEN:
            return True
        return False


class ExchangeFactory:
    """Creates and manages exchange instances with health monitoring."""

    def __init__(self, config: ExchangesConfig) -> None:
        self._config = config
        self._instances: dict[ExchangeID, BaseExchange] = {}
        self._breakers: dict[ExchangeID, CircuitBreaker] = {}

    async def create(self, exchange_id: ExchangeID) -> BaseExchange:
        """Create and connect an exchange instance."""
        if exchange_id in self._instances:
            return self._instances[exchange_id]

        cls = _REGISTRY.get(exchange_id)
        if cls is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        cfg = getattr(self._config, exchange_id.value, None)
        if cfg is None or not cfg.enabled:
            raise ValueError(f"Exchange {exchange_id} is not enabled in config")

        credentials = ExchangeCredentials(
            api_key=cfg.api_key,
            secret=cfg.secret,
            passphrase=cfg.passphrase,
            testnet=cfg.testnet,
        )

        instance = cls(credentials)
        try:
            await instance.connect()
        except Exception as e:
            raise ExchangeConnectionError(
                f"Failed to connect to {exchange_id}: {e}"
            ) from e

        self._instances[exchange_id] = instance
        self._breakers[exchange_id] = CircuitBreaker()
        logger.info("exchange.created", exchange=exchange_id)
        return instance

    async def create_all_enabled(self) -> dict[ExchangeID, BaseExchange]:
        """Create instances for all enabled exchanges."""
        tasks = []
        for eid in ExchangeID:
            cfg = getattr(self._config, eid.value, None)
            if cfg and cfg.enabled:
                tasks.append(self.create(eid))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return dict(self._instances)

    def get(self, exchange_id: ExchangeID) -> BaseExchange | None:
        """Get an existing exchange instance."""
        return self._instances.get(exchange_id)

    def get_breaker(self, exchange_id: ExchangeID) -> CircuitBreaker | None:
        """Get circuit breaker for an exchange."""
        return self._breakers.get(exchange_id)

    async def shutdown(self) -> None:
        """Disconnect all exchanges."""
        for instance in self._instances.values():
            await instance.disconnect()
        self._instances.clear()
        self._breakers.clear()


MAX_RETRIES = 3
RETRY_DELAY = 1.5

def with_retry(func):
    """Decorator to retry exchange API calls on transient failures."""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                import time
                time.sleep(RETRY_DELAY * (attempt + 1))
    return wrapper
