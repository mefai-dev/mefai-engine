"""WebSocket connection manager for real-time exchange data streams."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger()

StreamCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class WebSocketStream:
    """Manages a single WebSocket connection with auto-reconnect."""

    def __init__(
        self,
        url: str,
        on_message: StreamCallback,
        ping_interval: float = 20.0,
        max_reconnects: int = 50,
        reconnect_delay: float = 3.0,
    ) -> None:
        self._url = url
        self._on_message = on_message
        self._ping_interval = ping_interval
        self._max_reconnects = max_reconnects
        self._reconnect_delay = reconnect_delay
        self._ws: Any = None
        self._running = False
        self._reconnect_count = 0
        self._last_message_time: datetime | None = None
        self._task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        """Start the WebSocket connection loop."""
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def disconnect(self) -> None:
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self) -> None:
        """Main connection loop with auto-reconnect."""
        import websockets

        while self._running and self._reconnect_count < self._max_reconnects:
            try:
                async with websockets.connect(
                    self._url,
                    ping_interval=self._ping_interval,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._reconnect_count = 0
                    logger.info("ws.connected", url=self._url[:80])

                    async for raw in ws:
                        if not self._running:
                            break
                        self._last_message_time = datetime.now(tz=UTC)
                        try:
                            data = json.loads(raw)
                            await self._on_message(data)
                        except json.JSONDecodeError:
                            logger.warning("ws.invalid_json")
                        except Exception:
                            logger.exception("ws.callback_error")

            except asyncio.CancelledError:
                break
            except Exception:
                self._reconnect_count += 1
                if self._running:
                    wait = min(self._reconnect_delay * self._reconnect_count, 60)
                    logger.warning(
                        "ws.reconnecting",
                        attempt=self._reconnect_count,
                        wait=wait,
                    )
                    await asyncio.sleep(wait)

        if self._reconnect_count >= self._max_reconnects:
            logger.error("ws.max_reconnects_reached", url=self._url[:80])

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._ws.closed if self._ws else False

    async def send(self, data: dict[str, Any]) -> None:
        """Send JSON message through WebSocket."""
        if self._ws and not self._ws.closed:
            await self._ws.send(json.dumps(data))


class BinanceWSManager:
    """Manages Binance Futures WebSocket streams.

    Handles multiple concurrent streams:
    - Ticker streams (bookTicker)
    - Kline/candlestick streams
    - Order book depth streams
    - User data streams (orders + positions)
    """

    BASE_WS = "wss://fstream.binance.com"
    TESTNET_WS = "wss://stream.binancefuture.com"

    def __init__(self, testnet: bool = False) -> None:
        self._base = self.TESTNET_WS if testnet else self.BASE_WS
        self._streams: dict[str, WebSocketStream] = {}
        self._callbacks: dict[str, list[StreamCallback]] = {}

    async def subscribe_ticker(self, symbol: str, callback: StreamCallback) -> None:
        """Subscribe to real-time ticker updates."""
        stream_name = f"{symbol.lower()}@bookTicker"
        url = f"{self._base}/ws/{stream_name}"
        await self._add_stream(stream_name, url, callback)

    async def subscribe_kline(self, symbol: str, interval: str, callback: StreamCallback) -> None:
        """Subscribe to candlestick updates."""
        stream_name = f"{symbol.lower()}@kline_{interval}"
        url = f"{self._base}/ws/{stream_name}"
        await self._add_stream(stream_name, url, callback)

    async def subscribe_depth(self, symbol: str, callback: StreamCallback, levels: int = 20) -> None:
        """Subscribe to order book depth updates."""
        stream_name = f"{symbol.lower()}@depth{levels}@100ms"
        url = f"{self._base}/ws/{stream_name}"
        await self._add_stream(stream_name, url, callback)

    async def subscribe_user_data(self, listen_key: str, callback: StreamCallback) -> None:
        """Subscribe to user data stream (orders and position updates)."""
        url = f"{self._base}/ws/{listen_key}"
        await self._add_stream(f"user_{listen_key[:8]}", url, callback)

    async def subscribe_combined(self, symbols: list[str], callback: StreamCallback) -> None:
        """Subscribe to multiple ticker streams via combined endpoint."""
        streams = "/".join(f"{s.lower()}@bookTicker" for s in symbols)
        url = f"{self._base}/stream?streams={streams}"

        async def _unwrap(data: dict[str, Any]) -> None:
            inner = data.get("data", data)
            await callback(inner)

        await self._add_stream(f"combined_{len(symbols)}", url, _unwrap)

    async def _add_stream(self, name: str, url: str, callback: StreamCallback) -> None:
        """Create and connect a new stream."""
        if name in self._streams:
            return

        stream = WebSocketStream(url=url, on_message=callback)
        self._streams[name] = stream
        await stream.connect()
        logger.info("ws_manager.subscribed", stream=name)

    async def unsubscribe_all(self) -> None:
        """Disconnect all streams."""
        for name, stream in self._streams.items():
            await stream.disconnect()
        self._streams.clear()
        logger.info("ws_manager.all_disconnected")

    def get_stream_status(self) -> dict[str, bool]:
        """Get connection status of all streams."""
        return {name: stream.is_connected for name, stream in self._streams.items()}
