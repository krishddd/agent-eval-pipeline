"""
adapters/adapter_factory.py
Dynamic adapter factory — maps AgentCard.framework → correct adapter class.

Used by the FastAPI eval endpoint to create the right adapter for any registered agent.
"""

from __future__ import annotations

from typing import Any

from adapters.base import AgentAdapter


class AdapterFactory:
    """
    Creates the correct AgentAdapter subclass based on the agent card's
    framework and remote_config.

    Priority:
    1. If remote_config is present → always use RemoteAgentAdapter
    2. Else, match framework name → native adapter
    3. Else → raise error
    """

    @staticmethod
    def create(card: Any) -> AgentAdapter:
        """
        Build an adapter for the given AgentCard.

        Args:
            card: AgentCard instance with framework and optional remote_config.

        Returns:
            Configured AgentAdapter ready to call .run(task).
        """
        remote_config = getattr(card, "remote_config", None)
        framework = card.framework.lower()

        # ── Odysseus autonomous agent (chat + agent modes) ───────
        # Routed when framework == "odysseus" OR remote_config opts in via
        # {"agent_kind": "odysseus"}.  OdysseusAdapter subclasses
        # RemoteAgentAdapter so the harness keeps its k=1 fast-path.
        if remote_config and (
            framework == "odysseus"
            or str(remote_config.get("agent_kind", "")).lower() == "odysseus"
        ):
            from adapters.odysseus_adapter import OdysseusAdapter
            return OdysseusAdapter(remote_config)

        # ── Remote / HTTP agents (highest priority) ──────────────
        if remote_config:
            from adapters.remote_adapter import RemoteAgentAdapter
            return RemoteAgentAdapter(remote_config)

        if framework in ("http", "remote"):
            raise ValueError(
                f"Framework '{framework}' requires remote_config with at least 'base_url'. "
                "Register with remote_config: {\"base_url\": \"https://...\"}"
            )

        # ── Native framework adapters ────────────────────────────
        if framework == "langchain":
            from adapters.langchain_adapter import LangChainAdapter
            return LangChainAdapter(card)

        if framework == "langgraph":
            from adapters.langgraph_adapter import LangGraphAdapter
            return LangGraphAdapter(card)

        if framework == "autogen":
            from adapters.autogen_adapter import AutoGenAdapter
            return AutoGenAdapter(card)

        if framework == "crewai":
            from adapters.crewai_adapter import CrewAIAdapter
            return CrewAIAdapter(card)

        # ── Fallback: unknown framework ──────────────────────────
        # If remote_config is available, use the generic HTTP adapter
        if remote_config:
            from adapters.remote_adapter import RemoteAgentAdapter
            return RemoteAgentAdapter(remote_config)

        raise NotImplementedError(
            f"No native adapter for framework '{framework}'. "
            "Provide remote_config with at least 'base_url' to use "
            "the generic HTTP adapter for any open-source agent."
        )
