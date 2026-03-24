"""Agent orchestrator - coordinates the multi-agent decision cycle."""

from __future__ import annotations

import asyncio

import structlog

from mefai_engine.agents.base import AgentMessage, AgentRole, BaseAgent
from mefai_engine.constants import Direction, ExecutionAlgo, RiskDecisionType
from mefai_engine.types import MarketState, RiskDecision, Signal, TradingDecision

logger = structlog.get_logger()


class AgentOrchestrator:
    """Manages agent lifecycle and coordinates the decision cycle.

    Flow:
    1. AnalystAgent analyzes market -> proposes signal
    2. RiskAgent evaluates risk -> approves/rejects
    3. ExecutorAgent executes -> manages orders
    4. SentinelAgent monitors -> can trigger emergency actions
    """

    def __init__(self) -> None:
        self._agents: dict[AgentRole, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator."""
        self._agents[agent.role] = agent
        logger.info("orchestrator.agent_registered", role=agent.role, id=agent.agent_id)

    async def run_decision_cycle(self, state: MarketState) -> TradingDecision:
        """Run a full decision cycle across all agents.

        Returns the final trading decision with reasoning.
        """
        votes: dict[str, str] = {}

        # Step 1: Analyst proposes
        analyst = self._agents.get(AgentRole.ANALYST)
        analyst_msg = None
        if analyst:
            analyst_msg = await analyst.process(state)
            votes["analyst"] = analyst_msg.action
            logger.info("orchestrator.analyst_vote", action=analyst_msg.action)

        if not analyst_msg or analyst_msg.action == "hold":
            return TradingDecision(
                signal=None,
                risk_decision=None,
                execution_algo=ExecutionAlgo.MARKET,
                reasoning="Analyst recommends holding - no trade",
                agent_votes=votes,
            )

        # Extract signal from analyst
        signal = analyst_msg.payload.get("signal")
        if not isinstance(signal, Signal):
            return TradingDecision(
                signal=None,
                risk_decision=None,
                execution_algo=ExecutionAlgo.MARKET,
                reasoning="No valid signal from analyst",
                agent_votes=votes,
            )

        # Step 2: Risk evaluates
        risk_agent = self._agents.get(AgentRole.RISK)
        if risk_agent:
            risk_msg = await risk_agent.handle_message(analyst_msg)
            if risk_msg:
                votes["risk"] = risk_msg.action
                risk_decision = risk_msg.payload.get("decision")
                if isinstance(risk_decision, RiskDecision):
                    if risk_decision.decision == RiskDecisionType.REJECTED:
                        return TradingDecision(
                            signal=signal,
                            risk_decision=risk_decision,
                            execution_algo=ExecutionAlgo.MARKET,
                            reasoning=f"Risk rejected: {risk_decision.reason}",
                            agent_votes=votes,
                        )
                else:
                    risk_decision = RiskDecision(
                        decision=RiskDecisionType.APPROVED,
                        approved_size_pct=signal.suggested_size_pct,
                        reason="No risk agent decision",
                    )
            else:
                risk_decision = RiskDecision(
                    decision=RiskDecisionType.APPROVED,
                    approved_size_pct=signal.suggested_size_pct,
                    reason="Risk agent returned no response",
                )
        else:
            risk_decision = RiskDecision(
                decision=RiskDecisionType.APPROVED,
                approved_size_pct=signal.suggested_size_pct,
                reason="No risk agent registered",
            )

        # Step 3: Sentinel check (can veto)
        sentinel = self._agents.get(AgentRole.SENTINEL)
        if sentinel:
            sentinel_msg = await sentinel.process(state)
            votes["sentinel"] = sentinel_msg.action
            if sentinel_msg.action == "halt":
                return TradingDecision(
                    signal=signal,
                    risk_decision=risk_decision,
                    execution_algo=ExecutionAlgo.MARKET,
                    reasoning=f"Sentinel halted: {sentinel_msg.payload.get('reason', 'anomaly detected')}",
                    agent_votes=votes,
                )

        # Step 4: Determine execution algorithm
        exec_algo = ExecutionAlgo.MARKET
        if risk_decision.approved_size_pct > 5.0:
            exec_algo = ExecutionAlgo.TWAP  # Larger orders use TWAP

        votes["executor"] = "execute"

        return TradingDecision(
            signal=signal,
            risk_decision=risk_decision,
            execution_algo=exec_algo,
            reasoning="All agents approved - executing trade",
            agent_votes=votes,
        )

    async def emergency_flatten(self, state: MarketState) -> None:
        """Emergency: close all positions immediately."""
        logger.critical("orchestrator.emergency_flatten")
        executor = self._agents.get(AgentRole.EXECUTOR)
        if executor:
            await executor.handle_message(AgentMessage(
                sender="orchestrator",
                role=AgentRole.EXECUTOR,
                action="flatten_all",
                payload={"state": state},
            ))
