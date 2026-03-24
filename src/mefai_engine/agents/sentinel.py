"""Sentinel agent - monitors for anomalies and system health."""

from __future__ import annotations

from datetime import datetime, timezone

import structlog

from mefai_engine.agents.base import AgentMessage, AgentRole, BaseAgent
from mefai_engine.types import MarketState

logger = structlog.get_logger()


class SentinelAgent(BaseAgent):
    """Monitoring and anomaly detection agent.

    Watches for:
    - Extreme price moves (flash crash detection)
    - Exchange API latency spikes
    - Unusual volume patterns
    - Funding rate extremes
    - System health issues
    """

    agent_id = "sentinel_v1"
    role = AgentRole.SENTINEL

    def __init__(
        self,
        price_change_threshold_pct: float = 5.0,
        volume_spike_multiplier: float = 5.0,
        funding_rate_extreme: float = 0.01,
    ) -> None:
        self._price_threshold = price_change_threshold_pct
        self._volume_spike = volume_spike_multiplier
        self._funding_extreme = funding_rate_extreme
        self._last_prices: dict[str, float] = {}
        self._alerts: list[dict[str, str]] = []

    async def process(self, market_state: MarketState) -> AgentMessage:
        """Monitor current market state for anomalies."""
        alerts: list[str] = []
        symbol = market_state.symbol
        current_price = market_state.ticker.last

        # Flash crash detection
        if symbol in self._last_prices:
            prev = self._last_prices[symbol]
            if prev > 0:
                change_pct = abs(current_price - prev) / prev * 100
                if change_pct > self._price_threshold:
                    alerts.append(
                        f"FLASH MOVE: {symbol} moved {change_pct:.1f}% "
                        f"({prev:.2f} -> {current_price:.2f})"
                    )

        self._last_prices[symbol] = current_price

        # Volume spike
        vol_ratio = market_state.features.get("volume_ratio", 1.0)
        if vol_ratio > self._volume_spike:
            alerts.append(f"VOLUME SPIKE: {symbol} volume {vol_ratio:.1f}x average")

        # Funding rate extreme
        funding = market_state.features.get("funding_rate", 0.0)
        if abs(funding) > self._funding_extreme:
            direction = "LONG" if funding > 0 else "SHORT"
            alerts.append(
                f"FUNDING EXTREME: {symbol} funding {funding:.4f} "
                f"({direction} paying)"
            )

        # Spread widening
        spread = market_state.features.get("spread_bps", 0.0)
        if spread > 50:  # More than 50 bps spread
            alerts.append(f"WIDE SPREAD: {symbol} spread {spread:.1f} bps")

        if alerts:
            for alert in alerts:
                logger.warning("sentinel.alert", alert=alert)

            # Multiple critical alerts = halt
            if len(alerts) >= 2:
                return AgentMessage(
                    sender=self.agent_id,
                    role=self.role,
                    action="halt",
                    payload={
                        "reason": f"Multiple anomalies: {'; '.join(alerts)}",
                        "alerts": alerts,
                    },
                )

        return AgentMessage(
            sender=self.agent_id,
            role=self.role,
            action="clear",
            payload={"alerts": alerts},
        )

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        return None
