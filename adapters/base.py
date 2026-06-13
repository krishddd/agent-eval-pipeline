"""
adapters/base.py
AgentAdapter abstract base class + AgentResult model.
The eval harness only ever calls adapter.run(task) — zero framework knowledge.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Standardised result returned by every adapter regardless of framework."""

    output: str = Field(..., description="Final agent output text")
    success: bool = Field(..., description="Whether the agent completed the task")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    milestones: List[str] = Field(default_factory=list)
    policy_violations: List[str] = Field(default_factory=list)
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None  # RAG agents
    raw: Any = None  # Full framework-specific response

    class Config:
        arbitrary_types_allowed = True


class AgentAdapter(ABC):
    """
    Framework adapter interface.

    Every framework gets one subclass that implements run().
    Hooks are called synchronously during execution (real-time capture).
    """

    @abstractmethod
    def run(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ) -> AgentResult:
        """
        Execute the agent on the given task.

        Args:
            task: The task/prompt to execute.
            on_tool_call: Callback fired synchronously on each tool invocation.
                          Receives a ToolCallRecord.
            on_agent_msg: Callback fired on each inter-agent message.
                          Receives an AgentMessage.
            on_retrieval: Callback fired on each retrieval event.
                          Receives a dict with chunk data.

        Returns:
            AgentResult with output, tokens, cost, milestones, violations.
        """
        raise NotImplementedError
