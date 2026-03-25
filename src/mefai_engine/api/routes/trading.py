"""Trading endpoints - positions and order management and signals."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from mefai_engine.api.middleware import check_rate_limit, require_api_key
from mefai_engine.app import get_state
from mefai_engine.constants import Direction, ExchangeID, OrderType, Side

router = APIRouter(dependencies=[Depends(require_api_key), Depends(check_rate_limit)])


class ManualOrderRequest(BaseModel):
    """Manual order placement request."""
    symbol: str
    side: str  # "long" or "short"
    order_type: str = "market"
    quantity: float
    price: float | None = None
    leverage: int = 1
    reduce_only: bool = False


class ClosePositionRequest(BaseModel):
    """Close an open position."""
    symbol: str
    exchange: str = "binance"


# ── Positions ──────────────────────────────────────────────────

@router.get("/positions")
async def get_positions() -> dict[str, Any]:
    """Get all open positions across all exchanges."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        return {"positions": []}

    all_positions: list[dict[str, Any]] = []
    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                positions = await exchange.get_positions()
                for p in positions:
                    all_positions.append({
                        "symbol": p.symbol,
                        "side": p.side.value,
                        "size": p.size,
                        "entry_price": p.entry_price,
                        "mark_price": p.mark_price,
                        "unrealized_pnl": round(p.unrealized_pnl, 4),
                        "leverage": p.leverage,
                        "liquidation_price": p.liquidation_price,
                        "exchange": p.exchange.value,
                    })
            except Exception:
                pass

    return {"positions": all_positions, "count": len(all_positions)}


@router.get("/balance")
async def get_balance() -> dict[str, Any]:
    """Get account balance from primary exchange."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    # Get balance from first enabled exchange
    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                balance = await exchange.get_balance()
                return {
                    "total": round(balance.total, 4),
                    "available": round(balance.available, 4),
                    "unrealized_pnl": round(balance.unrealized_pnl, 4),
                    "margin_used": round(balance.margin_used, 4),
                    "currency": balance.currency,
                    "exchange": eid.value,
                }
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")


# ── Orders ─────────────────────────────────────────────────────

@router.post("/orders")
async def place_order(req: ManualOrderRequest) -> dict[str, Any]:
    """Place a manual order."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    exchange = factory.get(ExchangeID.BINANCE)
    if not exchange:
        raise HTTPException(status_code=503, detail="Binance not connected")

    from mefai_engine.types import OrderRequest

    side = Side.LONG if req.side.lower() in ("long", "buy") else Side.SHORT
    order_type = OrderType.LIMIT if req.price else OrderType.MARKET

    if req.leverage > 1:
        await exchange.set_leverage(req.symbol, req.leverage)

    order_req = OrderRequest(
        symbol=req.symbol,
        side=side,
        order_type=order_type,
        quantity=req.quantity,
        price=req.price,
        reduce_only=req.reduce_only,
        leverage=req.leverage,
    )

    try:
        result = await exchange.place_order(order_req)
        return {
            "order_id": result.order_id,
            "symbol": result.symbol,
            "side": result.side.value,
            "status": result.status.value,
            "quantity": result.quantity,
            "filled": result.filled_quantity,
            "average_price": result.average_price,
            "fee": result.fee,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/orders/{order_id}")
async def cancel_order(order_id: str, symbol: str) -> dict[str, Any]:
    """Cancel an open order."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    exchange = factory.get(ExchangeID.BINANCE)
    if not exchange:
        raise HTTPException(status_code=503, detail="Binance not connected")

    success = await exchange.cancel_order(order_id, symbol)
    return {"cancelled": success, "order_id": order_id}


@router.post("/positions/close")
async def close_position(req: ClosePositionRequest) -> dict[str, Any]:
    """Close an open position at market price."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    eid = ExchangeID(req.exchange)
    exchange = factory.get(eid)
    if not exchange:
        raise HTTPException(status_code=503, detail=f"{req.exchange} not connected")

    # Find current position
    positions = await exchange.get_positions()
    target = None
    for p in positions:
        if p.symbol == req.symbol:
            target = p
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"No open position for {req.symbol}")

    # Close with opposite side market order
    close_side = Side.SHORT if target.side == Side.LONG else Side.LONG
    from mefai_engine.types import OrderRequest
    close_req = OrderRequest(
        symbol=req.symbol,
        side=close_side,
        order_type=OrderType.MARKET,
        quantity=target.size,
        reduce_only=True,
    )

    result = await exchange.place_order(close_req)
    return {
        "closed": result.status.value in ("filled", "partially_filled"),
        "symbol": req.symbol,
        "size": target.size,
        "entry_price": target.entry_price,
        "close_price": result.average_price,
        "pnl": round(target.unrealized_pnl, 4),
    }


# ── Signals ────────────────────────────────────────────────────

@router.get("/signals")
async def get_signals() -> dict[str, Any]:
    """Get currently tracked signals and their evolution."""
    state = get_state()
    tracker = state.get("signal_tracker")
    if not tracker:
        return {"signals": []}

    signals = []
    for key, tracked in tracker.get_all().items():
        signals.append({
            "symbol": tracked.original.symbol,
            "direction": tracked.original.direction.value,
            "original_confidence": tracked.original.confidence,
            "current_confidence": tracked.current_confidence,
            "evolution": tracked.evolution.value,
            "updates": tracked.updates,
            "created_at": tracked.created_at.isoformat(),
            "last_update": tracked.last_update.isoformat(),
            "confidence_history": tracked.confidence_history[-20:],
            "notes": tracked.notes[-5:],
        })

    falsified = tracker.get_falsified()
    return {
        "signals": signals,
        "total_tracked": len(signals),
        "falsified_count": len(falsified),
    }


@router.get("/ticker/{symbol}")
async def get_ticker(symbol: str) -> dict[str, Any]:
    """Get real-time ticker for a symbol."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                ticker = await exchange.get_ticker(symbol.upper())
                return {
                    "symbol": ticker.symbol,
                    "bid": ticker.bid,
                    "ask": ticker.ask,
                    "last": ticker.last,
                    "spread": round(ticker.ask - ticker.bid, 4),
                    "spread_bps": round((ticker.ask - ticker.bid) / ticker.mid * 10000, 2) if (ticker.bid + ticker.ask) > 0 else 0,
                    "timestamp": ticker.timestamp.isoformat(),
                }
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")


@router.get("/orderbook/{symbol}")
async def get_orderbook(symbol: str, depth: int = 20) -> dict[str, Any]:
    """Get order book for a symbol."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                ob = await exchange.get_orderbook(symbol.upper(), depth)
                return {
                    "symbol": ob.symbol,
                    "bids": [{"price": l.price, "qty": l.quantity} for l in ob.bids[:depth]],
                    "asks": [{"price": l.price, "qty": l.quantity} for l in ob.asks[:depth]],
                    "spread": round(ob.spread, 4),
                    "mid_price": round(ob.mid_price, 4),
                    "timestamp": ob.timestamp.isoformat(),
                }
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")


@router.get("/funding/{symbol}")
async def get_funding_rate(symbol: str) -> dict[str, Any]:
    """Get current funding rate for a symbol."""
    state = get_state()
    factory = state.get("exchange_factory")
    if not factory:
        raise HTTPException(status_code=503, detail="Exchange not connected")

    for eid in ExchangeID:
        exchange = factory.get(eid)
        if exchange:
            try:
                fr = await exchange.get_funding_rate(symbol.upper())
                annual = fr.rate * 3 * 365 * 100
                return {
                    "symbol": fr.symbol,
                    "rate": fr.rate,
                    "annualized_pct": round(annual, 2),
                    "next_funding": fr.next_funding_time.isoformat(),
                    "direction": "longs pay" if fr.rate > 0 else "shorts pay",
                }
            except Exception as e:
                raise HTTPException(status_code=502, detail=str(e))

    raise HTTPException(status_code=503, detail="No exchange available")
