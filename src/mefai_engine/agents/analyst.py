"""Analyst agent - analyzes market state and proposes trading signals."""

from __future__ import annotations

import structlog

from mefai_engine.agents.base import AgentMessage, AgentRole, BaseAgent
from mefai_engine.constants import Direction
from mefai_engine.strategy.meta_learner import MetaLearner
from mefai_engine.types import MarketState, Signal

logger = structlog.get_logger()


class AnalystAgent(BaseAgent):
    """Market analysis agent.

    Consumes features and model predictions, produces trade proposals.
    Considers market regime, confidence levels, and feature confluence.
    """

    agent_id = "analyst_v1"
    role = AgentRole.ANALYST

    def __init__(self, meta_learner: MetaLearner) -> None:
        self._meta_learner = meta_learner

    async def process(self, market_state: MarketState) -> AgentMessage:
        """Analyze market state and propose a trade or hold."""
        signal = self._meta_learner.evaluate(
            predictions=market_state.predictions,
            regime=market_state.regime,
            symbol=market_state.symbol,
        )

        if signal is None:
            return AgentMessage(
                sender=self.agent_id,
                role=self.role,
                action="hold",
                payload={"reason": "No confident signal from models"},
            )

        # Additional confluence checks
        features = market_state.features
        confluence_score = self._check_confluence(signal, features)

        if confluence_score < 0.5:
            return AgentMessage(
                sender=self.agent_id,
                role=self.role,
                action="hold",
                payload={
                    "reason": f"Low confluence: {confluence_score:.2f}",
                    "signal": signal,
                },
            )

        return AgentMessage(
            sender=self.agent_id,
            role=self.role,
            action="trade",
            payload={
                "signal": signal,
                "confluence": confluence_score,
                "regime": market_state.regime.value,
            },
        )

    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        return None

    def _check_confluence(self, signal: Signal, features: dict[str, float]) -> float:
        """Check how many technical indicators agree with the signal direction.

        Returns a confluence score between 0 (no agreement) and 1 (full agreement).
        """
        agreements = 0
        checks = 0

        rsi = features.get("rsi_14")
        if rsi is not None:
            checks += 1
            if signal.direction == Direction.LONG and rsi < 70:
                agreements += 1
            elif signal.direction == Direction.SHORT and rsi > 30:
                agreements += 1

        macd = features.get("macd_12_26_9")
        if macd is not None:
            checks += 1
            if signal.direction == Direction.LONG and macd > 0:
                agreements += 1
            elif signal.direction == Direction.SHORT and macd < 0:
                agreements += 1

        trend = features.get("trend_strength")
        if trend is not None:
            checks += 1
            if signal.direction == Direction.LONG and trend > 0:
                agreements += 1
            elif signal.direction == Direction.SHORT and trend < 0:
                agreements += 1

        adx = features.get("adx_14")
        if adx is not None:
            checks += 1
            if adx > 20:  # Trending market favors directional trades
                agreements += 1

        book_imbalance = features.get("book_imbalance")
        if book_imbalance is not None:
            checks += 1
            if signal.direction == Direction.LONG and book_imbalance > 0:
                agreements += 1
            elif signal.direction == Direction.SHORT and book_imbalance < 0:
                agreements += 1

        return agreements / checks if checks > 0 else 0.0
