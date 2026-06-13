"""
registry/registry.py
AgentRegistry singleton — register once, eval repeatedly.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from .agent_card import AgentCard


class AgentRegistry:
    """
    Thread-safe singleton registry.
    Register an AgentCard → auto-infer eval categories → store.
    """

    _instance: Optional["AgentRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "AgentRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._store: Dict[str, AgentCard] = {}
            return cls._instance

    # ── Public API ──────────────────────────────────────────────

    def register(self, card: AgentCard) -> str:
        """
        Register an agent card.  Auto-infers eval categories if not
        explicitly set.  Returns the agent_id.
        """
        # C7 fix: copy card to avoid mutating the caller's instance
        card = card.model_copy()

        if card.eval_categories is None:
            card.eval_categories = card.auto_infer_categories()

        self._store[card.agent_id] = card

        n = len(card.eval_categories)
        print(
            f"[Registry] {card.agent_type.value} '{card.name}' "
            f"→ {n} eval categories: {card.eval_categories}"
        )
        return card.agent_id

    def get(self, agent_id: str) -> AgentCard:
        """Retrieve a registered agent card by ID."""
        if agent_id not in self._store:
            raise KeyError(f"Agent '{agent_id}' not found in registry")
        return self._store[agent_id]

    def list_all(self) -> list:
        """Return all registered cards."""
        return list(self._store.values())

    def remove(self, agent_id: str) -> None:
        """Remove an agent by ID."""
        if agent_id in self._store:
            del self._store[agent_id]

    @classmethod
    def reset(cls):
        """S1 fix: Clear store for test isolation / hot-reload."""
        if cls._instance is not None:
            cls._instance._store.clear()

    def clear(self) -> None:
        """Remove all registered agents (useful for testing)."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, agent_id: str) -> bool:
        return agent_id in self._store


# Global singleton instance
registry = AgentRegistry()
