"""Multi-tenant system with isolated risk limits and API key management.

Each tenant has independent configuration and risk limits and PnL tracking.
Tenant identification happens via API key in the middleware layer.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from dataclasses import field as dc_field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class TenantConfig:
    """Configuration for the multi-tenant system."""
    max_tenants: int = 100
    api_key_prefix: str = "mefai_"
    api_key_length: int = 48
    default_plan: str = "free"


@dataclass
class TenantRiskLimits:
    """Per-tenant risk limits."""
    max_position_pct: float = 10.0
    max_total_exposure_pct: float = 30.0
    max_daily_loss_pct: float = 3.0
    max_drawdown_pct: float = 10.0
    max_symbols: int = 1
    max_signals_per_day: int = 10


@dataclass
class TenantPnL:
    """PnL tracking for a single tenant."""
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    daily_reset_date: str = ""

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades


@dataclass
class Tenant:
    """A single tenant in the system."""
    tenant_id: str
    name: str
    email: str
    plan: str = "free"
    is_active: bool = True
    api_keys: list[str] = dc_field(default_factory=list)
    risk_limits: TenantRiskLimits = dc_field(default_factory=TenantRiskLimits)
    pnl: TenantPnL = dc_field(default_factory=TenantPnL)
    metadata: dict[str, Any] = dc_field(default_factory=dict)
    created_at: datetime = dc_field(default_factory=lambda: datetime.now(tz=UTC))
    signals_today: int = 0


# Plan tier definitions
PLAN_LIMITS: dict[str, TenantRiskLimits] = {
    "free": TenantRiskLimits(
        max_position_pct=5.0,
        max_total_exposure_pct=10.0,
        max_daily_loss_pct=2.0,
        max_drawdown_pct=5.0,
        max_symbols=1,
        max_signals_per_day=10,
    ),
    "pro": TenantRiskLimits(
        max_position_pct=10.0,
        max_total_exposure_pct=30.0,
        max_daily_loss_pct=5.0,
        max_drawdown_pct=15.0,
        max_symbols=5,
        max_signals_per_day=999999,
    ),
    "enterprise": TenantRiskLimits(
        max_position_pct=20.0,
        max_total_exposure_pct=50.0,
        max_daily_loss_pct=10.0,
        max_drawdown_pct=25.0,
        max_symbols=999,
        max_signals_per_day=999999,
    ),
}


class TenantManager:
    """Manages multi-tenant operations.

    Provides tenant CRUD operations and API key management and
    per-tenant PnL tracking. Tenants are isolated from each other
    with independent risk limits based on their subscription plan.
    """

    def __init__(self, config: TenantConfig | None = None) -> None:
        self._config = config or TenantConfig()
        self._tenants: dict[str, Tenant] = {}
        self._api_key_to_tenant: dict[str, str] = {}

    def create_tenant(
        self,
        name: str,
        email: str,
        plan: str = "free",
        metadata: dict[str, Any] | None = None,
    ) -> Tenant:
        """Create a new tenant with an API key.

        Args:
            name: Tenant display name
            email: Tenant contact email
            plan: Subscription plan (free / pro / enterprise)
            metadata: Optional additional metadata

        Returns:
            Created Tenant with generated API key.
        """
        if len(self._tenants) >= self._config.max_tenants:
            raise ValueError(
                f"Maximum tenant limit reached: {self._config.max_tenants}"
            )

        tenant_id = self._generate_id(name, email)

        if tenant_id in self._tenants:
            raise ValueError(f"Tenant already exists: {tenant_id}")

        risk_limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

        api_key = self._generate_api_key()

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            email=email,
            plan=plan,
            risk_limits=risk_limits,
            api_keys=[api_key],
            metadata=metadata or {},
        )

        self._tenants[tenant_id] = tenant
        self._api_key_to_tenant[api_key] = tenant_id

        logger.info(
            "tenant.created",
            tenant_id=tenant_id,
            plan=plan,
        )

        return tenant

    def get_tenant(self, tenant_id: str) -> Tenant | None:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)

    def get_tenant_by_api_key(self, api_key: str) -> Tenant | None:
        """Look up tenant from API key (used by middleware)."""
        tenant_id = self._api_key_to_tenant.get(api_key)
        if tenant_id is None:
            return None
        return self._tenants.get(tenant_id)

    def rotate_api_key(self, tenant_id: str) -> str:
        """Generate a new API key for a tenant (old keys remain valid)."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant not found: {tenant_id}")

        new_key = self._generate_api_key()
        tenant.api_keys.append(new_key)
        self._api_key_to_tenant[new_key] = tenant_id

        logger.info("tenant.key_rotated", tenant_id=tenant_id)
        return new_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke a specific API key."""
        tenant_id = self._api_key_to_tenant.pop(api_key, None)
        if tenant_id is None:
            return False

        tenant = self._tenants.get(tenant_id)
        if tenant and api_key in tenant.api_keys:
            tenant.api_keys.remove(api_key)

        logger.info("tenant.key_revoked", tenant_id=tenant_id)
        return True

    def update_plan(self, tenant_id: str, plan: str) -> None:
        """Update a tenant's subscription plan."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            raise ValueError(f"Tenant not found: {tenant_id}")

        if plan not in PLAN_LIMITS:
            raise ValueError(f"Unknown plan: {plan}")

        tenant.plan = plan
        tenant.risk_limits = PLAN_LIMITS[plan]

        logger.info(
            "tenant.plan_updated",
            tenant_id=tenant_id,
            plan=plan,
        )

    def record_trade(self, tenant_id: str, pnl: float) -> None:
        """Record a trade result for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if tenant.pnl.daily_reset_date != today:
            tenant.pnl.daily_pnl = 0.0
            tenant.pnl.daily_reset_date = today
            tenant.signals_today = 0

        tenant.pnl.total_pnl += pnl
        tenant.pnl.daily_pnl += pnl
        tenant.pnl.total_trades += 1

        if pnl > 0:
            tenant.pnl.winning_trades += 1
        elif pnl < 0:
            tenant.pnl.losing_trades += 1

    def record_signal(self, tenant_id: str) -> bool:
        """Record a signal usage and check daily limit.

        Returns:
            True if within limit; False if limit exceeded.
        """
        tenant = self._tenants.get(tenant_id)
        if tenant is None:
            return False

        today = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        if tenant.pnl.daily_reset_date != today:
            tenant.pnl.daily_pnl = 0.0
            tenant.pnl.daily_reset_date = today
            tenant.signals_today = 0

        if tenant.signals_today >= tenant.risk_limits.max_signals_per_day:
            return False

        tenant.signals_today += 1
        return True

    def deactivate_tenant(self, tenant_id: str) -> None:
        """Deactivate a tenant (soft delete)."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            tenant.is_active = False
            # Revoke all API keys
            for key in list(tenant.api_keys):
                self._api_key_to_tenant.pop(key, None)
            tenant.api_keys.clear()
            logger.info("tenant.deactivated", tenant_id=tenant_id)

    def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        """List all tenants."""
        tenants = list(self._tenants.values())
        if active_only:
            tenants = [t for t in tenants if t.is_active]
        return tenants

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        raw = secrets.token_urlsafe(self._config.api_key_length)
        return f"{self._config.api_key_prefix}{raw}"

    @staticmethod
    def _generate_id(name: str, email: str) -> str:
        """Generate a deterministic tenant ID."""
        raw = f"{name.lower().strip()}:{email.lower().strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def tenant_middleware_extract(
    api_key: str, manager: TenantManager
) -> Tenant | None:
    """Extract tenant from API key for use in request middleware.

    Args:
        api_key: The API key from the request header
        manager: TenantManager instance

    Returns:
        Tenant if found and active; None otherwise.
    """
    tenant = manager.get_tenant_by_api_key(api_key)
    if tenant is None:
        return None
    if not tenant.is_active:
        return None
    return tenant
