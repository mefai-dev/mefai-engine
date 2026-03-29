"""WebSocket endpoint for real-time streaming to clients."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from mefai_engine.app import get_state
from mefai_engine.constants import ExchangeID

logger = structlog.get_logger()
router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket client connections."""

    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info("ws_client.connected", total=len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info("ws_client.disconnected", total=len(self._connections))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send data to all connected clients."""
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.remove(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


_manager = ConnectionManager()


@router.websocket("/ws/live")
async def live_stream(ws: WebSocket) -> None:
    """Real-time WebSocket stream for live data.

    Sends:
    - ticker updates (every second)
    - position updates (every 10 seconds)
    - signal updates (on new signal)
    - PnL snapshots (every 30 seconds)
    """
    await _manager.connect(ws)

    try:
        state = get_state()
        factory = state.get("exchange_factory")
        risk = state.get("risk_manager")
        config = state.get("config")
        symbols = config.engine.symbols if config else ["BTCUSDT"]

        while True:
            # Ticker data
            if factory:
                for eid in ExchangeID:
                    exchange = factory.get(eid)
                    if exchange:
                        for symbol in symbols:
                            try:
                                ticker = await exchange.get_ticker(symbol)
                                await ws.send_json({
                                    "type": "ticker",
                                    "symbol": ticker.symbol,
                                    "bid": ticker.bid,
                                    "ask": ticker.ask,
                                    "last": ticker.last,
                                    "timestamp": ticker.timestamp.isoformat(),
                                })
                            except Exception:
                                pass

                        # Positions (less frequent)
                        try:
                            positions = await exchange.get_positions()
                            if positions:
                                await ws.send_json({
                                    "type": "positions",
                                    "data": [
                                        {
                                            "symbol": p.symbol,
                                            "side": p.side.value,
                                            "size": p.size,
                                            "entry_price": p.entry_price,
                                            "unrealized_pnl": round(p.unrealized_pnl, 4),
                                        }
                                        for p in positions
                                    ],
                                })
                        except Exception:
                            pass
                        break

            # PnL metrics
            if risk:
                await ws.send_json({
                    "type": "pnl",
                    "data": risk.pnl_tracker.to_dict(),
                })

            # Signal tracker
            tracker = state.get("signal_tracker")
            if tracker and tracker.get_active_count() > 0:
                signals_data = []
                for key, tracked in tracker.get_all().items():
                    signals_data.append({
                        "symbol": tracked.original.symbol,
                        "direction": tracked.original.direction.value,
                        "confidence": tracked.current_confidence,
                        "evolution": tracked.evolution.value,
                    })
                await ws.send_json({"type": "signals", "data": signals_data})

            # Wait before next update cycle
            try:
                # Also listen for client messages (ping/subscribe)
                msg = await asyncio.wait_for(ws.receive_text(), timeout=3.0)
                try:
                    client_data = json.loads(msg)
                    if client_data.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                except json.JSONDecodeError:
                    pass
            except TimeoutError:
                pass

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("ws.stream_error")
    finally:
        _manager.disconnect(ws)
