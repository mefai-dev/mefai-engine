"""Webhook endpoints - TradingView and external signal ingestion."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request

from mefai_engine.app import get_state

router = APIRouter()


@router.post("/webhook/tradingview")
async def tradingview_webhook(
    request: Request,
    x_signature: str | None = Header(default=None, alias="X-Signature"),
) -> dict[str, Any]:
    """Receive TradingView alert webhook.

    Expected JSON payload:
    {
        "symbol": "BTCUSDT",
        "action": "buy" | "sell" | "close",
        "timeframe": "1h",
        "price": 95000.50,
        "confidence": 0.75
    }
    """
    state = get_state()
    receiver = state.get("webhook_receiver")
    if not receiver:
        raise HTTPException(status_code=503, detail="Webhook receiver not initialized")

    body = await request.body()

    # Validate signature if configured
    if x_signature and not receiver.validate_signature(body, x_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    signal = receiver.process(data)
    if signal is None:
        return {"accepted": False, "reason": "Signal filtered (duplicate or invalid)"}

    # Track the signal
    tracker = state.get("signal_tracker")
    if tracker:
        tracked = tracker.track(signal)
        return {
            "accepted": True,
            "signal": {
                "symbol": signal.symbol,
                "direction": signal.direction.value,
                "confidence": signal.confidence,
            },
            "evolution": tracked.evolution.value,
        }

    return {
        "accepted": True,
        "signal": {
            "symbol": signal.symbol,
            "direction": signal.direction.value,
            "confidence": signal.confidence,
        },
    }


@router.post("/webhook/custom")
async def custom_webhook(request: Request) -> dict[str, Any]:
    """Generic webhook for custom signal sources.

    Accepts any JSON with at least 'symbol' and 'direction' fields.
    """
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    symbol = data.get("symbol", "").upper()
    direction = data.get("direction", "").lower()

    if not symbol or direction not in ("long", "short", "flat"):
        raise HTTPException(
            status_code=400,
            detail="Required fields: symbol (str) and direction (long|short|flat)",
        )

    state = get_state()
    receiver = state.get("webhook_receiver")
    if not receiver:
        raise HTTPException(status_code=503, detail="Webhook receiver not initialized")

    signal = receiver.process({
        "symbol": symbol,
        "action": direction,
        "confidence": data.get("confidence", 0.6),
        "price": data.get("price", 0),
    }, strategy_id="custom_webhook")

    if signal is None:
        return {"accepted": False, "reason": "Filtered"}

    return {
        "accepted": True,
        "symbol": signal.symbol,
        "direction": signal.direction.value,
        "confidence": signal.confidence,
    }
