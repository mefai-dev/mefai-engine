"""Exchange-specific data models and credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(slots=True)
class ExchangeCredentials:
    """Exchange API credentials."""
    api_key: str
    secret: str
    passphrase: str = ""
    testnet: bool = False


@dataclass(slots=True)
class RateLimitState:
    """Tracks rate limit consumption per endpoint category."""
    category: str
    max_per_second: int
    tokens: float
    last_refill: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def consume(self, cost: int = 1) -> bool:
        """Try to consume tokens. Returns True if allowed."""
        now = datetime.now(tz=UTC)
        elapsed = (now - self.last_refill).total_seconds()
        self.tokens = min(
            self.max_per_second,
            self.tokens + elapsed * self.max_per_second,
        )
        self.last_refill = now

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


@dataclass(slots=True)
class WebSocketState:
    """Tracks WebSocket connection state."""
    connected: bool = False
    last_message_time: datetime | None = None
    reconnect_count: int = 0
    subscriptions: set[str] = field(default_factory=set)
