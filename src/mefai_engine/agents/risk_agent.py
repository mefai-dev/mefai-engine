"""Risk agent - evaluates portfolio risk and approves/vetoes trades."""

from __future__ import annotations

import structlog

from mefai_engine.agents.base import AgentMessage, AgentRole, BaseAgent
from mefai_engine.risk.manager import RiskManager
from mefai_engine.types import Balance, MarketState, Position, RiskDecision, Signal

logger = structlog.get_logger()


class RiskAgent(BaseAgent):
    """Risk assessment agent with veto authority.

    Evaluates every trade proposal against risk limits.
    Can reduce position size or reject trades entirely.
    """

    agent_id = "risk_v1"
    role = AgentRole.RISK

    def __init__(self, risk_manager: RiskManager) -> None:
        self._risk_manager = risk_manager

    async def process(self, market_state: MarketState) -> AgentMessage:
        """Monitor current risk metrics."""
        metrics = self._risk_manager.pnl_tracker.to_dict()
        cb_state = self._risk_manager.circuit_breaker.state

        return AgentMessage(
            sender=self.agent_id,
            role=self.role,
            action="report",
            payload={
                "metrics": metrics,
                "circuit_breaker": cb_state.value,
            },
        )

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Evaluate a trade proposal from the analyst."""
        if message.action != "trade":
            return None

        signal = message.payload.get("signal")
        if not isinstance(signal, Signal):
            return AgentMessage(
                sender=self.agent_id,
                role=self.role,
                action="reject",
                payload={"reason": "Invalid signal payload"},
            )

        # Get current balance and positions from market state or payload
        balance = message.payload.get("balance")
        positions = message.payload.get("positions", [])

        if not isinstance(balance, Balance):
            balance = Balance(total=10000, available=10000, unrealized_pnl=0, margin_used=0)

        decision = await self._risk_manager.evaluate(
            signal=signal,
            balance=balance,
            positions=positions,
        )

        action = "approve" if decision.decision.value == "approved" else (
            "reduce" if decision.decision.value == "reduced" else "reject"
        )

        logger.info(
            "risk_agent.decision",
            action=action,
            reason=decision.reason,
            size=decision.approved_size_pct,
        )

        return AgentMessage(
            sender=self.agent_id,
            role=self.role,
            action=action,
            payload={"decision": decision},
        )
