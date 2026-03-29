"""System endpoints - health check and engine status and configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter

from mefai_engine import __version__
from mefai_engine.app import get_state

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint for load balancers and monitoring."""
    state = get_state()
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "components": {
            "database": state.get("store") is not None,
            "cache": state.get("cache") is not None,
            "exchange": state.get("exchange_factory") is not None,
        },
    }


@router.get("/status")
async def engine_status() -> dict[str, Any]:
    """Full engine status including all component states."""
    state = get_state()
    config = state.get("config")
    factory = state.get("exchange_factory")
    risk = state.get("risk_manager")
    registry = state.get("model_registry")
    tracker = state.get("signal_tracker")

    exchanges_status: dict[str, Any] = {}
    if factory:
        from mefai_engine.constants import ExchangeID
        for eid in ExchangeID:
            instance = factory.get(eid)
            breaker = factory.get_breaker(eid)
            if instance:
                exchanges_status[eid.value] = {
                    "connected": instance._connected,
                    "circuit_breaker": breaker.state.value if breaker else "unknown",
                }

    return {
        "engine": {
            "version": __version__,
            "mode": config.engine.mode.value if config else "unknown",
            "symbols": config.engine.symbols if config else [],
            "uptime": datetime.now(tz=UTC).isoformat(),
        },
        "exchanges": exchanges_status,
        "risk": {
            "circuit_breaker": risk.circuit_breaker.state.value if risk else "unknown",
            "metrics": risk.pnl_tracker.to_dict() if risk else {},
        },
        "models": registry.status() if registry else {},
        "signals": {
            "active_tracked": tracker.get_active_count() if tracker else 0,
            "falsified": len(tracker.get_falsified()) if tracker else 0,
        },
    }


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Get current engine configuration (secrets redacted)."""
    state = get_state()
    config = state.get("config")
    if not config:
        return {"error": "Config not loaded"}

    return {
        "engine": {
            "mode": config.engine.mode.value,
            "symbols": config.engine.symbols,
            "log_level": config.engine.log_level,
        },
        "data": {
            "timeframes": config.data.timeframes,
            "orderbook_depth": config.data.orderbook_depth,
            "history_days": config.data.history_days,
        },
        "features": {"enabled_count": len(config.features.enabled)},
        "models": {
            "gradient_boost": config.models.gradient_boost.enabled,
            "transformer": config.models.transformer.enabled,
            "rl": config.models.rl.enabled,
            "sentiment": config.models.sentiment.enabled,
        },
        "risk": {
            "max_position_pct": config.risk.max_position_pct,
            "max_total_exposure_pct": config.risk.max_total_exposure_pct,
            "max_daily_loss_pct": config.risk.max_daily_loss_pct,
            "max_drawdown_pct": config.risk.max_drawdown_pct,
        },
        "execution": {
            "default_algorithm": config.execution.default_algorithm.value,
            "maker_fee_bps": config.execution.maker_fee_bps,
            "taker_fee_bps": config.execution.taker_fee_bps,
        },
    }
