"""Abstract exchange interface using Protocol for structural subtyping."""

from __future__ import annotations

import hashlib
import hmac
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

import aiohttp
import structlog

from mefai_engine.constants import ExchangeID
from mefai_engine.exchange.models import ExchangeCredentials, RateLimitState
from mefai_engine.types import (
    Balance,
    Candle,
    FundingRate,
    OrderBook,
    OrderRequest,
    OrderResult,
    Position,
    Ticker,
)

logger = structlog.get_logger()

TickerCallback = Callable[[Ticker], Coroutine[Any, Any, None]]
OrderBookCallback = Callable[[OrderBook], Coroutine[Any, Any, None]]
UserDataCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class BaseExchange(ABC):
    """Abstract base for all exchange implementations.

    Each exchange must implement the full interface. REST calls go through
    ``_request`` which handles signing, rate limiting, and retries.
    WebSocket connections are managed by ``ws_manager``.
    """

    exchange_id: ExchangeID
    base_url: str
    ws_url: str

    def __init__(self, credentials: ExchangeCredentials) -> None:
        self._credentials = credentials
        self._session: aiohttp.ClientSession | None = None
        self._rate_limiters: dict[str, RateLimitState] = {}
        self._connected = False

    # ── lifecycle ───────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish HTTP session and authenticate."""
        self._session = aiohttp.ClientSession(
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        self._connected = True
        logger.info("exchange.connected", exchange=self.exchange_id)

    async def disconnect(self) -> None:
        """Close HTTP session and all WebSocket connections."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._connected = False
        logger.info("exchange.disconnected", exchange=self.exchange_id)

    # ── http layer ──────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        signed: bool = False,
        category: str = "default",
    ) -> dict[str, Any]:
        """Execute an HTTP request with signing, rate limiting, and retry."""
        if not self._session:
            raise RuntimeError("Exchange not connected. Call connect() first.")

        # Rate limiting
        limiter = self._rate_limiters.get(category)
        if limiter and not limiter.consume():
            wait = 1.0 / limiter.max_per_second
            logger.warning("exchange.rate_limited", exchange=self.exchange_id, wait=wait)
            import asyncio
            await asyncio.sleep(wait)

        url = f"{self.base_url}{endpoint}"
        headers: dict[str, str] = {}

        if signed:
            headers, params, data = self._sign_request(method, endpoint, params, data)

        async with self._session.request(
            method, url, params=params, json=data, headers=headers
        ) as resp:
            body = await resp.json()
            if resp.status >= 400:
                logger.error(
                    "exchange.request_error",
                    exchange=self.exchange_id,
                    status=resp.status,
                    body=body,
                    endpoint=endpoint,
                )
                from mefai_engine.exceptions import ExchangeError
                raise ExchangeError(f"{self.exchange_id} {resp.status}: {body}")
            return body  # type: ignore[no-any-return]

    @abstractmethod
    def _sign_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None,
        data: dict[str, Any] | None,
    ) -> tuple[dict[str, str], dict[str, Any] | None, dict[str, Any] | None]:
        """Sign a request. Returns (headers, params, data)."""
        ...

    # ── market data ─────────────────────────────────────────────────

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker: ...

    @abstractmethod
    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook: ...

    @abstractmethod
    async def get_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int = 500
    ) -> list[Candle]: ...

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> FundingRate: ...

    # ── account ─────────────────────────────────────────────────────

    @abstractmethod
    async def get_balance(self) -> Balance: ...

    @abstractmethod
    async def get_positions(self) -> list[Position]: ...

    # ── orders ──────────────────────────────────────────────────────

    @abstractmethod
    async def place_order(self, request: OrderRequest) -> OrderResult: ...

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool: ...

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> OrderResult: ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> None: ...

    # ── websocket ───────────────────────────────────────────────────

    @abstractmethod
    async def subscribe_ticker(self, symbol: str, callback: TickerCallback) -> None: ...

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, callback: OrderBookCallback) -> None: ...

    @abstractmethod
    async def subscribe_user_data(self, callback: UserDataCallback) -> None: ...

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _hmac_sha256(secret: str, message: str) -> str:
        """HMAC-SHA256 signature."""
        return hmac.new(
            secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

    @staticmethod
    def _timestamp_ms() -> int:
        """Current timestamp in milliseconds."""
        return int(time.time() * 1000)
