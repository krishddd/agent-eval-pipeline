"""
registry/agent_card.py
AgentCard universal schema — Pydantic V2.
Every field maps to one or more eval metrics.
Based on CLEAR (2025) & MultiAgentBench (ACL 2025).
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Agent Type Taxonomy (§1.1) ──────────────────────────────────────────

class AgentType(str, Enum):
    """Nine agent types covering every production pattern as of 2025-2026."""
    SINGLE_TASK   = "single_task"
    REACT         = "react"
    RAG           = "rag"
    SOCIAL_SIM    = "social_sim"
    FINANCIAL_ABM = "financial_abm"
    ORCHESTRATOR  = "orchestrator"
    WORKER        = "worker"
    SWARM         = "swarm"
    PIPELINE      = "pipeline"


class MemoryType(str, Enum):
    """Memory backend — drives conditional evaluator selection."""
    NONE      = "none"
    IN_MEMORY = "in_memory"
    VECTOR_DB = "vector_db"   # → RAG evaluator
    GRAPH_DB  = "graph_db"    # → GraphMemory evaluator
    HYBRID    = "hybrid"      # → Both vector + graph evaluators


# ── Tool Definition ─────────────────────────────────────────────────────

class ToolDef(BaseModel):
    """Tool schema entry for the agent's tool manifest."""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        frozen = True


# ── AgentCard ────────────────────────────────────────────────────────────

class AgentCard(BaseModel):
    """
    Universal schema for any AI agent.  Register once → evaluate indefinitely.

    Every field maps directly to one or more eval metrics.  Optional fields
    expand the eval scope — an agent without *tools_manifest* skips Tool Use;
    an agent without *subagents* skips Multi-Agent Coordination.
    """

    # ── Required ────────────────────────────────────────────────
    name: str = Field(..., min_length=1, description="Human-readable display name")
    agent_type: AgentType = Field(..., description="See taxonomy §1.1 — drives category auto-inference")
    framework: str = Field(
        ...,
        description="langchain | crewai | autogen | langgraph | oasis | mesa | bedrock | openai | custom",
    )
    model_backbone: str = Field(
        ...,
        description="Underlying LLM(s): gpt-4o | claude-3-5 | llama3 | qwen3 | gemini-2 | etc.",
    )

    # ── Auto-generated ──────────────────────────────────────────
    agent_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique agent identifier. Deterministic if set explicitly.",
    )
    version: str = "1.0.0"

    # ── Tool configuration ───────────────────────────────────────
    tools_manifest: List[ToolDef] = Field(default_factory=list)

    # ── Memory & source ─────────────────────────────────────────
    memory_type: MemoryType = MemoryType.NONE
    agent_source: str = "llm_generated"  # llm_generated | rule_based | hybrid | retrieved

    # ── Multi-agent ──────────────────────────────────────────────
    subagents: List[str] = Field(default_factory=list)

    # ── Eval configuration ───────────────────────────────────────
    eval_categories: Optional[List[str]] = None
    max_cost_usd: float = Field(default=5.0, ge=0)
    pass_k: int = Field(default=8, ge=1, le=50)
    # Reliability sampling for REMOTE agents: how many times to repeat a task
    # that is marked idempotent (chat / read-only) so pass@k & pass^k (τ-bench)
    # become computable. Side-effecting tasks always run once. 1 = disabled.
    reliability_k: int = Field(default=1, ge=1, le=20)
    golden_trajectory: Optional[List[str]] = None
    golden_milestones: Optional[List[str]] = None
    golden_sources: Optional[List[str]] = None  # For provenance comparator (silent failure detection)
    sla_latency_ms: int = Field(default=30_000, ge=100)

    # ── Type-specific extensions ─────────────────────────────────
    persona_spec: Optional[str] = None       # social_sim agents
    hk_params: Optional[Dict[str, Any]] = None  # financial_abm agents
    activity_pattern: Optional[Dict[str, Any]] = None  # simulation agents

    # ── Remote agent connection ──────────────────────────────────
    remote_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Connection config for hosted agents. Keys: "
            "base_url (required), chat_endpoint, health_endpoint, "
            "task_field, response_field, auth_headers, timeout_ms, extra_body"
        ),
    )

    # ── Metadata ─────────────────────────────────────────────────
    tags: Dict[str, str] = Field(default_factory=dict)

    class Config:
        use_enum_values = False
        validate_assignment = True

    # ── Validators ───────────────────────────────────────────────

    @field_validator("hk_params")
    @classmethod
    def validate_hk_params(cls, v: Optional[Dict], info) -> Optional[Dict]:
        """Ensure HK params contain required fields when provided."""
        if v is not None:
            required = {"epsilon", "n_agents"}
            missing = required - set(v.keys())
            if missing:
                raise ValueError(f"hk_params missing required fields: {missing}")
        return v

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        """Normalize framework name to lowercase. Accepts any framework."""
        v_lower = v.lower().replace("-", "_").replace(" ", "_")
        if not v_lower:
            raise ValueError("framework cannot be empty")
        return v_lower

    # ── Category Auto-Inference (§1.2 Matrix) ────────────────────

    def auto_infer_categories(self) -> List[str]:
        """
        Map agent_type + memory_type + subagents → applicable eval categories.
        Mirrors the Agent Type → Eval Category Matrix from §1.2.
        """
        # Base categories — apply to ALL agent types
        cats = [
            "task_completion",
            "trajectory",
            "reliability",
            "enterprise_cost",
            "safety",
        ]

        # Tool Use — if agent has tools
        if self.tools_manifest:
            cats.append("tool_use")

        # Multi-Agent Coordination — orchestrators, swarms, or agents with subagents
        if self.subagents or self.agent_type in (
            AgentType.ORCHESTRATOR,
            AgentType.SWARM,
        ):
            cats.append("multi_agent_coord")

        # RAG Quality — vector or hybrid memory
        if self.memory_type in (MemoryType.VECTOR_DB, MemoryType.HYBRID):
            cats.append("rag_quality")

        # Graph Memory — graph or hybrid memory
        if self.memory_type in (MemoryType.GRAPH_DB, MemoryType.HYBRID):
            cats.append("graph_memory")

        # Persona Consistency — social simulation agents
        if self.agent_type == AgentType.SOCIAL_SIM or self.persona_spec:
            cats.append("persona_consistency")

        # HK Contagion — financial ABM agents with HK configuration
        if self.agent_type == AgentType.FINANCIAL_ABM and self.hk_params:
            cats.append("hk_contagion")

        # Odysseus Metrics (M01-M33) — the autonomous Odysseus agent, routed by
        # framework name or a remote_config opt-in. Quality-graded metrics over
        # the chat + agent-run surface.
        if self.remote_config and (
            self.framework == "odysseus"
            or str(self.remote_config.get("agent_kind", "")).lower() == "odysseus"
        ):
            cats.append("odysseus_metrics")

        return cats
