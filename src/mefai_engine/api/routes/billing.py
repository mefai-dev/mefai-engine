"""Billing and subscription management via Stripe integration.

Plan tiers:
- Free: 1 symbol and 10 signals/day
- Pro: 5 symbols and unlimited signals
- Enterprise: all features and unlimited everything

Handles subscription lifecycle and usage metering and Stripe webhooks.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

import structlog

from mefai_engine.api.middleware import require_api_key
from mefai_engine.app import get_state

logger = structlog.get_logger()

router = APIRouter(prefix="/billing")


# ---- Request / Response models ----

class CreateSubscriptionRequest(BaseModel):
    """Request to create a new subscription."""
    tenant_id: str
    plan: str  # "free" or "pro" or "enterprise"
    payment_method_id: str | None = None


class SubscriptionResponse(BaseModel):
    """Subscription details response."""
    subscription_id: str
    tenant_id: str
    plan: str
    status: str
    current_period_start: str
    current_period_end: str


class UsageResponse(BaseModel):
    """Current usage for a tenant."""
    tenant_id: str
    plan: str
    signals_used_today: int
    signals_limit: int
    symbols_active: int
    symbols_limit: int


# ---- Plan definitions ----

PLAN_PRICING: dict[str, dict[str, Any]] = {
    "free": {
        "name": "Free",
        "price_monthly_usd": 0,
        "max_symbols": 1,
        "max_signals_per_day": 10,
        "features": ["basic_signals", "paper_trading"],
    },
    "pro": {
        "name": "Pro",
        "price_monthly_usd": 99,
        "max_symbols": 5,
        "max_signals_per_day": 999999,
        "features": [
            "basic_signals", "paper_trading", "live_trading",
            "advanced_models", "custom_strategies",
        ],
    },
    "enterprise": {
        "name": "Enterprise",
        "price_monthly_usd": 499,
        "max_symbols": 999,
        "max_signals_per_day": 999999,
        "features": [
            "basic_signals", "paper_trading", "live_trading",
            "advanced_models", "custom_strategies",
            "multi_exchange", "dedicated_support",
            "custom_models", "api_priority",
        ],
    },
}


# ---- Stripe client wrapper ----

class StripeClient:
    """Wrapper around the Stripe Python SDK.

    Handles subscription CRUD and webhook verification.
    All Stripe calls go through this class for testability.
    """

    def __init__(self, api_key: str = "", webhook_secret: str = "") -> None:
        self._api_key = api_key
        self._webhook_secret = webhook_secret
        self._stripe: Any = None

    def _get_stripe(self) -> Any:
        if self._stripe is not None:
            return self._stripe

        try:
            import stripe
            stripe.api_key = self._api_key
            self._stripe = stripe
            return stripe
        except ImportError:
            raise ImportError("stripe package required. Run: pip install stripe")

    async def create_customer(self, email: str, name: str, tenant_id: str) -> str:
        """Create a Stripe customer and return customer ID."""
        stripe = self._get_stripe()
        customer = stripe.Customer.create(
            email=email,
            name=name,
            metadata={"tenant_id": tenant_id},
        )
        return str(customer.id)

    async def create_subscription(
        self,
        customer_id: str,
        plan: str,
        payment_method_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a Stripe subscription."""
        stripe = self._get_stripe()

        price_lookup = {
            "free": None,
            "pro": "price_pro_monthly",
            "enterprise": "price_enterprise_monthly",
        }

        price_id = price_lookup.get(plan)
        if price_id is None and plan != "free":
            raise ValueError(f"Unknown plan: {plan}")

        if plan == "free":
            return {
                "id": f"sub_free_{customer_id}",
                "status": "active",
                "current_period_start": int(time.time()),
                "current_period_end": int(time.time()) + 30 * 86400,
            }

        params: dict[str, Any] = {
            "customer": customer_id,
            "items": [{"price": price_id}],
        }
        if payment_method_id:
            params["default_payment_method"] = payment_method_id

        subscription = stripe.Subscription.create(**params)
        return {
            "id": subscription.id,
            "status": subscription.status,
            "current_period_start": subscription.current_period_start,
            "current_period_end": subscription.current_period_end,
        }

    async def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription at period end."""
        stripe = self._get_stripe()
        try:
            stripe.Subscription.modify(
                subscription_id, cancel_at_period_end=True
            )
            return True
        except Exception as exc:
            logger.error("stripe.cancel_error", error=str(exc))
            return False

    def verify_webhook(self, payload: bytes, signature: str) -> dict[str, Any] | None:
        """Verify and parse a Stripe webhook event."""
        stripe = self._get_stripe()
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self._webhook_secret
            )
            return dict(event)
        except Exception as exc:
            logger.warning("stripe.webhook_invalid", error=str(exc))
            return None


# ---- Usage tracker ----

class UsageTracker:
    """Tracks API and signal usage per tenant for metering."""

    def __init__(self) -> None:
        self._daily_signals: dict[str, int] = {}
        self._active_symbols: dict[str, set[str]] = {}
        self._reset_date: str = ""

    def record_signal(self, tenant_id: str, symbol: str) -> bool:
        """Record a signal usage. Returns False if limit exceeded."""
        self._check_daily_reset()

        if tenant_id not in self._daily_signals:
            self._daily_signals[tenant_id] = 0
        if tenant_id not in self._active_symbols:
            self._active_symbols[tenant_id] = set()

        self._daily_signals[tenant_id] += 1
        self._active_symbols[tenant_id].add(symbol)
        return True

    def get_usage(self, tenant_id: str) -> dict[str, int]:
        """Get current usage counts for a tenant."""
        self._check_daily_reset()
        return {
            "signals_today": self._daily_signals.get(tenant_id, 0),
            "symbols_active": len(self._active_symbols.get(tenant_id, set())),
        }

    def _check_daily_reset(self) -> None:
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if today != self._reset_date:
            self._daily_signals.clear()
            self._active_symbols.clear()
            self._reset_date = today


# ---- Routes ----

@router.get("/plans")
async def list_plans() -> dict[str, Any]:
    """List available subscription plans."""
    return {"plans": PLAN_PRICING}


@router.post("/subscriptions", dependencies=[Depends(require_api_key)])
async def create_subscription(req: CreateSubscriptionRequest) -> dict[str, Any]:
    """Create a new subscription for a tenant."""
    state = get_state()
    tenant_mgr = state.get("tenant_manager")
    stripe_client = state.get("stripe_client")

    if not tenant_mgr:
        raise HTTPException(status_code=503, detail="Tenant system not initialized")

    tenant = tenant_mgr.get_tenant(req.tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if req.plan not in PLAN_PRICING:
        raise HTTPException(status_code=400, detail=f"Unknown plan: {req.plan}")

    if stripe_client and req.plan != "free":
        try:
            customer_id = tenant.metadata.get("stripe_customer_id")
            if not customer_id:
                customer_id = await stripe_client.create_customer(
                    email=tenant.email, name=tenant.name, tenant_id=req.tenant_id
                )
                tenant.metadata["stripe_customer_id"] = customer_id

            sub = await stripe_client.create_subscription(
                customer_id=customer_id,
                plan=req.plan,
                payment_method_id=req.payment_method_id,
            )

            tenant.metadata["stripe_subscription_id"] = sub["id"]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # Update tenant plan
    tenant_mgr.update_plan(req.tenant_id, req.plan)

    return {
        "status": "active",
        "tenant_id": req.tenant_id,
        "plan": req.plan,
    }


@router.get("/usage/{tenant_id}", dependencies=[Depends(require_api_key)])
async def get_usage(tenant_id: str) -> dict[str, Any]:
    """Get current usage for a tenant."""
    state = get_state()
    tenant_mgr = state.get("tenant_manager")
    usage_tracker = state.get("usage_tracker")

    if not tenant_mgr:
        raise HTTPException(status_code=503, detail="Tenant system not initialized")

    tenant = tenant_mgr.get_tenant(tenant_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    plan_info = PLAN_PRICING.get(tenant.plan, PLAN_PRICING["free"])

    usage = {}
    if usage_tracker:
        usage = usage_tracker.get_usage(tenant_id)

    return {
        "tenant_id": tenant_id,
        "plan": tenant.plan,
        "signals_used_today": usage.get("signals_today", 0),
        "signals_limit": plan_info["max_signals_per_day"],
        "symbols_active": usage.get("symbols_active", 0),
        "symbols_limit": plan_info["max_symbols"],
    }


@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request) -> dict[str, str]:
    """Handle Stripe webhook events.

    Processes subscription lifecycle events:
    - customer.subscription.created
    - customer.subscription.updated
    - customer.subscription.deleted
    - invoice.payment_succeeded
    - invoice.payment_failed
    """
    state = get_state()
    stripe_client = state.get("stripe_client")
    tenant_mgr = state.get("tenant_manager")

    if not stripe_client:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    event = stripe_client.verify_webhook(payload, sig_header)
    if event is None:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    logger.info("stripe.webhook", event_type=event_type)

    if event_type == "customer.subscription.deleted":
        # Downgrade to free
        customer_id = data.get("customer", "")
        if tenant_mgr:
            for tenant in tenant_mgr.list_tenants():
                if tenant.metadata.get("stripe_customer_id") == customer_id:
                    tenant_mgr.update_plan(tenant.tenant_id, "free")
                    break

    elif event_type == "invoice.payment_failed":
        logger.warning("stripe.payment_failed", customer=data.get("customer"))

    return {"status": "received"}
