"""FastAPI application factory for MEFAI Engine."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mefai_engine import __version__
from mefai_engine.config import Settings, load_config

logger = structlog.get_logger()

# Global state holder accessible by all routes
_state: dict[str, Any] = {}


def get_state() -> dict[str, Any]:
    """Get the shared application state."""
    return _state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    config = load_config()
    _state["config"] = config
    _state["started"] = True

    # Initialize data store
    from mefai_engine.data.store import TimeSeriesStore
    store = TimeSeriesStore(config.database.timescaledb_url)
    try:
        await store.connect()
        await store.init_schema()
        _state["store"] = store
    except Exception:
        logger.warning("app.db_unavailable")
        _state["store"] = None

    # Initialize cache
    from mefai_engine.data.cache import RedisCache
    cache = RedisCache(config.database.redis_url)
    try:
        await cache.connect()
        _state["cache"] = cache
    except Exception:
        logger.warning("app.cache_unavailable")
        _state["cache"] = None

    # Initialize exchange factory
    from mefai_engine.exchange.factory import ExchangeFactory
    factory = ExchangeFactory(config.exchanges)
    try:
        await factory.create_all_enabled()
        _state["exchange_factory"] = factory
    except Exception:
        logger.warning("app.exchange_unavailable")
        _state["exchange_factory"] = factory

    # Initialize risk manager
    from mefai_engine.risk.manager import RiskManager
    _state["risk_manager"] = RiskManager(config.risk)

    # Initialize model registry
    from mefai_engine.models.registry import ModelRegistry
    registry = ModelRegistry()
    _state["model_registry"] = registry

    # Initialize signal tracker
    from mefai_engine.agents.signal_tracker import SignalTracker
    _state["signal_tracker"] = SignalTracker()

    # Initialize webhook receiver
    from mefai_engine.data.webhook import WebhookReceiver
    _state["webhook_receiver"] = WebhookReceiver()

    # Initialize report generator
    from mefai_engine.monitoring.reports import ReportGenerator
    _state["report_generator"] = ReportGenerator(_state["risk_manager"].pnl_tracker)

    # Initialize Prometheus metrics
    from mefai_engine.monitoring.metrics import get_metrics
    _state["metrics"] = get_metrics()

    # Initialize audit log
    from mefai_engine.monitoring.audit_log import AuditConfig, AuditLog
    audit_cfg = AuditConfig(
        log_dir=getattr(config, "audit", {}).get("log_dir", "logs/audit")
        if isinstance(getattr(config, "audit", None), dict)
        else "logs/audit",
    )
    _state["audit_log"] = AuditLog(audit_cfg)

    # Initialize tenant manager
    from mefai_engine.api.tenants import TenantConfig, TenantManager
    _state["tenant_manager"] = TenantManager(TenantConfig())

    # Initialize usage tracker for billing
    from mefai_engine.api.routes.billing import UsageTracker
    _state["usage_tracker"] = UsageTracker()

    logger.info("app.started", version=__version__, mode=config.engine.mode)

    yield

    # Shutdown
    if _state.get("audit_log"):
        _state["audit_log"].close()
    if _state.get("exchange_factory"):
        await _state["exchange_factory"].shutdown()
    if _state.get("store"):
        await _state["store"].disconnect()
    if _state.get("cache"):
        await _state["cache"].disconnect()

    logger.info("app.shutdown")


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MEFAI Engine",
        description="Institutional-grade AI trading engine for crypto perpetual futures",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs" if os.environ.get("MEFAI__ENGINE__MODE") != "live" else None,
        redoc_url="/redoc" if os.environ.get("MEFAI__ENGINE__MODE") != "live" else None,
    )

    # CORS
    allowed_origins = os.environ.get("MEFAI_CORS_ORIGINS", "").split(",")
    if not allowed_origins or allowed_origins == [""]:
        allowed_origins = ["https://mefai.io", "http://localhost:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # Register routes
    from mefai_engine.api.routes.system import router as system_router
    from mefai_engine.api.routes.trading import router as trading_router
    from mefai_engine.api.routes.data import router as data_router
    from mefai_engine.api.routes.models import router as models_router
    from mefai_engine.api.routes.backtest import router as backtest_router
    from mefai_engine.api.routes.monitoring import router as monitoring_router
    from mefai_engine.api.routes.webhook import router as webhook_router
    from mefai_engine.api.routes.prometheus import router as prometheus_router
    from mefai_engine.api.routes.billing import router as billing_router
    from mefai_engine.api.websocket import router as ws_router

    app.include_router(system_router, prefix="/api/v1", tags=["System"])
    app.include_router(trading_router, prefix="/api/v1", tags=["Trading"])
    app.include_router(data_router, prefix="/api/v1", tags=["Data"])
    app.include_router(models_router, prefix="/api/v1", tags=["Models"])
    app.include_router(backtest_router, prefix="/api/v1", tags=["Backtest"])
    app.include_router(monitoring_router, prefix="/api/v1", tags=["Monitoring"])
    app.include_router(webhook_router, prefix="/api/v1", tags=["Webhook"])
    app.include_router(prometheus_router, tags=["Prometheus"])
    app.include_router(billing_router, prefix="/api/v1", tags=["Billing"])
    app.include_router(ws_router, tags=["WebSocket"])

    # Tenant extraction middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest
    from starlette.responses import Response as StarletteResponse

    class TenantMiddleware(BaseHTTPMiddleware):
        """Extract tenant from API key and attach to request state."""

        async def dispatch(
            self, request: StarletteRequest, call_next: Any
        ) -> StarletteResponse:
            api_key = request.headers.get("x-api-key", "")
            if api_key:
                from mefai_engine.api.tenants import tenant_middleware_extract
                tenant_mgr = _state.get("tenant_manager")
                if tenant_mgr:
                    tenant = tenant_middleware_extract(api_key, tenant_mgr)
                    if tenant:
                        request.state.tenant = tenant
                        request.state.tenant_id = tenant.tenant_id
            response = await call_next(request)
            return response

    app.add_middleware(TenantMiddleware)

    return app
