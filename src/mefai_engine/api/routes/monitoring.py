"""Monitoring endpoints - PnL metrics and reports and risk status."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from mefai_engine.app import get_state

router = APIRouter()


@router.get("/metrics")
async def get_metrics() -> dict[str, Any]:
    """Get current performance metrics."""
    state = get_state()
    risk = state.get("risk_manager")
    if not risk:
        return {"metrics": {}}

    return {
        "metrics": risk.pnl_tracker.to_dict(),
        "circuit_breaker": {
            "state": risk.circuit_breaker.state.value,
            "can_trade": risk.circuit_breaker.can_trade(),
        },
    }


@router.get("/report")
async def get_daily_report() -> dict[str, Any]:
    """Generate daily performance report."""
    state = get_state()
    generator = state.get("report_generator")
    if not generator:
        return {"error": "Report generator not available"}

    # Get current equity
    factory = state.get("exchange_factory")
    equity = 0.0
    positions_count = 0

    if factory:
        from mefai_engine.constants import ExchangeID
        for eid in ExchangeID:
            exchange = factory.get(eid)
            if exchange:
                try:
                    balance = await exchange.get_balance()
                    equity = balance.total + balance.unrealized_pnl
                    positions = await exchange.get_positions()
                    positions_count = len(positions)
                except Exception:
                    pass
                break

    report = generator.daily_summary(equity, positions_count)
    return {
        "report": report,
        "formatted": generator.format_text(report),
    }


@router.post("/report/send-telegram")
async def send_telegram_report() -> dict[str, Any]:
    """Send daily report via Telegram."""
    state = get_state()
    config = state.get("config")

    if not config or not config.monitoring.telegram.enabled:
        return {"sent": False, "reason": "Telegram not configured"}

    generator = state.get("report_generator")
    if not generator:
        return {"sent": False, "reason": "Report generator not available"}

    from mefai_engine.monitoring.telegram import TelegramNotifier
    notifier = TelegramNotifier(
        bot_token=config.monitoring.telegram.bot_token,
        chat_id=config.monitoring.telegram.chat_id,
    )

    factory = state.get("exchange_factory")
    equity = 0.0
    if factory:
        from mefai_engine.constants import ExchangeID
        for eid in ExchangeID:
            exchange = factory.get(eid)
            if exchange:
                try:
                    balance = await exchange.get_balance()
                    equity = balance.total + balance.unrealized_pnl
                except Exception:
                    pass
                break

    report = generator.daily_summary(equity)
    message = generator.format_telegram(report)
    success = await notifier.send(message)

    return {"sent": success}


@router.get("/risk")
async def get_risk_status() -> dict[str, Any]:
    """Get detailed risk management status."""
    state = get_state()
    risk = state.get("risk_manager")
    config = state.get("config")

    if not risk:
        return {"error": "Risk manager not available"}

    return {
        "circuit_breaker": {
            "state": risk.circuit_breaker.state.value,
            "can_trade": risk.circuit_breaker.can_trade(),
        },
        "pnl": risk.pnl_tracker.to_dict(),
        "limits": {
            "max_position_pct": config.risk.max_position_pct if config else 0,
            "max_total_exposure_pct": config.risk.max_total_exposure_pct if config else 0,
            "max_daily_loss_pct": config.risk.max_daily_loss_pct if config else 0,
            "max_drawdown_pct": config.risk.max_drawdown_pct if config else 0,
            "max_consecutive_losses": config.risk.max_consecutive_losses if config else 0,
        },
    }


@router.post("/risk/circuit-breaker/reset")
async def reset_circuit_breaker() -> dict[str, Any]:
    """Manually reset the circuit breaker to CLOSED state."""
    state = get_state()
    risk = state.get("risk_manager")
    if not risk:
        return {"error": "Risk manager not available"}

    risk.circuit_breaker.reset()
    return {"state": risk.circuit_breaker.state.value, "reset": True}
