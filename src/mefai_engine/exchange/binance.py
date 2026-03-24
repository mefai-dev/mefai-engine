"""Binance Futures USDT-M implementation."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import structlog

from mefai_engine.constants import ExchangeID, OrderStatus, OrderType, Side
from mefai_engine.exchange.base import (
    BaseExchange,
    OrderBookCallback,
    TickerCallback,
    UserDataCallback,
)
from mefai_engine.exchange.models import ExchangeCredentials, RateLimitState
from mefai_engine.types import (
    Balance,
    Candle,
    FundingRate,
    OrderBook,
    OrderBookLevel,
    OrderRequest,
    OrderResult,
    Position,
    Ticker,
)

logger = structlog.get_logger()

_TIMEFRAME_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}

_ORDER_TYPE_MAP = {
    OrderType.MARKET: "MARKET",
    OrderType.LIMIT: "LIMIT",
    OrderType.STOP_MARKET: "STOP_MARKET",
    OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT_MARKET",
}

_SIDE_MAP = {Side.LONG: "BUY", Side.SHORT: "SELL"}

_STATUS_MAP = {
    "NEW": OrderStatus.OPEN,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "EXPIRED": OrderStatus.EXPIRED,
}


class BinanceExchange(BaseExchange):
    """Binance USDT-M Futures exchange adapter."""

    exchange_id = ExchangeID.BINANCE

    def __init__(self, credentials: ExchangeCredentials) -> None:
        super().__init__(credentials)
        if credentials.testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com"

        self._rate_limiters = {
            "order": RateLimitState(category="order", max_per_second=10, tokens=10),
            "default": RateLimitState(category="default", max_per_second=20, tokens=20),
        }

    def _sign_request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None,
        data: dict[str, Any] | None,
    ) -> tuple[dict[str, str], dict[str, Any] | None, dict[str, Any] | None]:
        """Binance HMAC-SHA256 query string signing."""
        params = params or {}
        params["timestamp"] = self._timestamp_ms()
        params["recvWindow"] = 5000

        query = urlencode(params, doseq=True)
        signature = self._hmac_sha256(self._credentials.secret, query)
        params["signature"] = signature

        headers = {"X-MBX-APIKEY": self._credentials.api_key}
        return headers, params, data

    # ── market data ─────────────────────────────────────────────────

    async def get_ticker(self, symbol: str) -> Ticker:
        resp = await self._request("GET", "/fapi/v1/ticker/bookTicker", params={"symbol": symbol})
        price_resp = await self._request("GET", "/fapi/v1/ticker/price", params={"symbol": symbol})
        return Ticker(
            symbol=symbol,
            bid=float(resp["bidPrice"]),
            ask=float(resp["askPrice"]),
            last=float(price_resp["price"]),
            volume_24h=0.0,
            timestamp=datetime.now(tz=timezone.utc),
        )

    async def get_orderbook(self, symbol: str, depth: int = 20) -> OrderBook:
        resp = await self._request(
            "GET", "/fapi/v1/depth", params={"symbol": symbol, "limit": depth}
        )
        return OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(float(p), float(q)) for p, q in resp["bids"]],
            asks=[OrderBookLevel(float(p), float(q)) for p, q in resp["asks"]],
            timestamp=datetime.now(tz=timezone.utc),
        )

    async def get_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int = 500
    ) -> list[Candle]:
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": _TIMEFRAME_MAP.get(timeframe, timeframe),
            "limit": min(limit, 1500),
        }
        if since is not None:
            params["startTime"] = since

        resp = await self._request("GET", "/fapi/v1/klines", params=params)
        candles: list[Candle] = []
        for k in resp:
            candles.append(Candle(
                timestamp=datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                quote_volume=float(k[7]),
                trades=int(k[8]),
            ))
        return candles

    async def get_funding_rate(self, symbol: str) -> FundingRate:
        resp = await self._request(
            "GET", "/fapi/v1/premiumIndex", params={"symbol": symbol}
        )
        return FundingRate(
            symbol=symbol,
            rate=float(resp["lastFundingRate"]),
            next_funding_time=datetime.fromtimestamp(
                resp["nextFundingTime"] / 1000, tz=timezone.utc
            ),
            timestamp=datetime.now(tz=timezone.utc),
        )

    # ── account ─────────────────────────────────────────────────────

    async def get_balance(self) -> Balance:
        resp = await self._request("GET", "/fapi/v2/account", signed=True)
        return Balance(
            total=float(resp["totalWalletBalance"]),
            available=float(resp["availableBalance"]),
            unrealized_pnl=float(resp["totalUnrealizedProfit"]),
            margin_used=float(resp["totalInitialMargin"]),
        )

    async def get_positions(self) -> list[Position]:
        resp = await self._request("GET", "/fapi/v2/positionRisk", signed=True)
        positions: list[Position] = []
        for p in resp:
            size = float(p["positionAmt"])
            if size == 0:
                continue
            positions.append(Position(
                symbol=p["symbol"],
                side=Side.LONG if size > 0 else Side.SHORT,
                size=abs(size),
                entry_price=float(p["entryPrice"]),
                mark_price=float(p["markPrice"]),
                unrealized_pnl=float(p["unRealizedProfit"]),
                leverage=int(p["leverage"]),
                liquidation_price=float(p["liquidationPrice"]),
                margin=float(p["isolatedMargin"]) if p["marginType"] == "isolated" else 0.0,
                exchange=ExchangeID.BINANCE,
                timestamp=datetime.now(tz=timezone.utc),
            ))
        return positions

    # ── orders ──────────────────────────────────────────────────────

    async def place_order(self, request: OrderRequest) -> OrderResult:
        params: dict[str, Any] = {
            "symbol": request.symbol,
            "side": _SIDE_MAP[request.side],
            "type": _ORDER_TYPE_MAP[request.order_type],
            "quantity": str(request.quantity),
        }
        if request.price is not None:
            params["price"] = str(request.price)
            params["timeInForce"] = "GTC"
        if request.stop_price is not None:
            params["stopPrice"] = str(request.stop_price)
        if request.reduce_only:
            params["reduceOnly"] = "true"
        if request.client_order_id:
            params["newClientOrderId"] = request.client_order_id

        resp = await self._request(
            "POST", "/fapi/v1/order", params=params, signed=True, category="order"
        )
        return self._parse_order(resp)

    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        try:
            await self._request(
                "DELETE", "/fapi/v1/order",
                params={"symbol": symbol, "orderId": order_id},
                signed=True, category="order",
            )
            return True
        except Exception:
            logger.exception("exchange.cancel_failed", order_id=order_id)
            return False

    async def get_order(self, order_id: str, symbol: str) -> OrderResult:
        resp = await self._request(
            "GET", "/fapi/v1/order",
            params={"symbol": symbol, "orderId": order_id},
            signed=True,
        )
        return self._parse_order(resp)

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        await self._request(
            "POST", "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage},
            signed=True,
        )
        logger.info("exchange.leverage_set", symbol=symbol, leverage=leverage)

    # ── websocket ───────────────────────────────────────────────────

    async def subscribe_ticker(self, symbol: str, callback: TickerCallback) -> None:
        """Subscribe to real-time ticker via WebSocket."""
        from mefai_engine.exchange.ws_manager import BinanceWSManager
        if not hasattr(self, "_ws_manager"):
            self._ws_manager = BinanceWSManager(testnet=self._credentials.testnet)

        async def _handle(data: dict[str, Any]) -> None:
            ticker = Ticker(
                symbol=data.get("s", symbol),
                bid=float(data.get("b", 0)),
                ask=float(data.get("a", 0)),
                last=float(data.get("b", 0)),
                volume_24h=0.0,
                timestamp=datetime.now(tz=timezone.utc),
            )
            await callback(ticker)

        await self._ws_manager.subscribe_ticker(symbol, _handle)

    async def subscribe_orderbook(self, symbol: str, callback: OrderBookCallback) -> None:
        """Subscribe to real-time order book depth."""
        from mefai_engine.exchange.ws_manager import BinanceWSManager
        if not hasattr(self, "_ws_manager"):
            self._ws_manager = BinanceWSManager(testnet=self._credentials.testnet)

        async def _handle(data: dict[str, Any]) -> None:
            ob = OrderBook(
                symbol=symbol,
                bids=[OrderBookLevel(float(p), float(q)) for p, q in data.get("b", [])],
                asks=[OrderBookLevel(float(p), float(q)) for p, q in data.get("a", [])],
                timestamp=datetime.now(tz=timezone.utc),
            )
            await callback(ob)

        await self._ws_manager.subscribe_depth(symbol, _handle)

    async def subscribe_user_data(self, callback: UserDataCallback) -> None:
        """Subscribe to user order/position updates."""
        from mefai_engine.exchange.ws_manager import BinanceWSManager
        if not hasattr(self, "_ws_manager"):
            self._ws_manager = BinanceWSManager(testnet=self._credentials.testnet)

        # Get listen key for user data stream
        resp = await self._request("POST", "/fapi/v1/listenKey", signed=True)
        listen_key = resp.get("listenKey", "")
        if not listen_key:
            logger.error("exchange.no_listen_key")
            return

        await self._ws_manager.subscribe_user_data(listen_key, callback)

    # ── private helpers ─────────────────────────────────────────────

    def _parse_order(self, resp: dict[str, Any]) -> OrderResult:
        return OrderResult(
            order_id=str(resp["orderId"]),
            client_order_id=resp.get("clientOrderId", ""),
            symbol=resp["symbol"],
            side=Side.LONG if resp["side"] == "BUY" else Side.SHORT,
            order_type=OrderType.MARKET if resp["type"] == "MARKET" else OrderType.LIMIT,
            status=_STATUS_MAP.get(resp["status"], OrderStatus.PENDING),
            quantity=float(resp["origQty"]),
            filled_quantity=float(resp.get("executedQty", 0)),
            average_price=float(resp.get("avgPrice", 0)),
            fee=float(resp.get("commission", 0)),
            timestamp=datetime.fromtimestamp(resp["updateTime"] / 1000, tz=timezone.utc),
            exchange=ExchangeID.BINANCE,
            raw=resp,
        )
