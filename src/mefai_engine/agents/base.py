"""Base agent interface for the multi-agent trading system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


class AgentRole(StrEnum):
    ANALYST = "analyst"
    RISK = "risk"
    EXECUTOR = "executor"
    SENTINEL = "sentinel"


@dataclass(slots=True)
class AgentMessage:
    """Inter-agent communication message."""
    sender: str
    role: AgentRole
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


class BaseAgent(ABC):
    """Base class for all trading agents."""

    agent_id: str
    role: AgentRole

    @abstractmethod
    async def process(self, market_state: Any) -> AgentMessage:
        """Process current market state and produce a message."""
        ...

    @abstractmethod
    async def handle_message(self, message: AgentMessage) -> AgentMessage | None:
        """Handle a message from another agent. Returns response or None."""
        ...
