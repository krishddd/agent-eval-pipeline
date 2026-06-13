"""
scripts/register_all_agents.py
Bulk agent registration for CI/CD pipeline.
Registers all agents defined in configuration and exits.
"""

from __future__ import annotations

import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registry.agent_card import AgentCard, AgentType, MemoryType, ToolDef
from registry.registry import registry


def register_example_agents():
    """Register the five example agents from §9."""

    # 9.1 — Customer Service RAG Agent (LangChain)
    rag_card = AgentCard(
        name="support-rag-agent-v3",
        agent_type=AgentType.RAG,
        framework="langchain",
        model_backbone="gpt-4o-mini",
        tools_manifest=[
            ToolDef(name="vector_search", description="Semantic search over support KB",
                    parameters={"query": "str", "top_k": "int"}),
            ToolDef(name="ticket_lookup", description="Fetch ticket by ID",
                    parameters={"ticket_id": "str"}),
            ToolDef(name="escalate", description="Escalate to human agent",
                    parameters={"reason": "str"}),
        ],
        memory_type=MemoryType.VECTOR_DB,
        pass_k=8,
        sla_latency_ms=5_000,
        max_cost_usd=0.05,
        golden_milestones=["kb_searched", "answer_grounded", "resolution_offered"],
        tags={"team": "support", "env": "prod", "region": "us-east"},
    )
    registry.register(rag_card)

    # 9.2 — Code Generation Agent (OpenAI Assistants)
    code_card = AgentCard(
        name="code-gen-agent-v1",
        agent_type=AgentType.REACT,
        framework="openai_assistants",
        model_backbone="gpt-4o",
        tools_manifest=[
            ToolDef(name="code_interpreter", description="Execute and test Python code",
                    parameters={"code": "str"}),
            ToolDef(name="file_search", description="Search codebase via embedding",
                    parameters={"query": "str"}),
            ToolDef(name="web_search", description="Search for documentation",
                    parameters={"query": "str"}),
        ],
        memory_type=MemoryType.VECTOR_DB,
        pass_k=5,
        max_cost_usd=0.50,
        golden_milestones=["requirements_understood", "code_written", "tests_pass", "docs_added"],
        golden_trajectory=["file_search", "code_interpreter", "code_interpreter"],
        sla_latency_ms=60_000,
    )
    registry.register(code_card)

    # 9.3 — Research Orchestrator (CrewAI)
    orch_card = AgentCard(
        name="research-orchestrator-v2",
        agent_type=AgentType.ORCHESTRATOR,
        framework="crewai",
        model_backbone="claude-sonnet-4-6",
        tools_manifest=[
            ToolDef(name="delegate_task", description="Assign sub-task to a worker agent",
                    parameters={"task": "str", "agent_id": "str"}),
            ToolDef(name="synthesize", description="Merge worker outputs into report",
                    parameters={"inputs": "list"}),
        ],
        subagents=["web-researcher-id", "data-analyst-id", "writer-id"],
        pass_k=5,
        max_cost_usd=2.0,
        golden_milestones=[
            "research_assigned", "data_collected", "analysis_done",
            "draft_written", "report_final",
        ],
    )
    registry.register(orch_card)

    # 9.4 — Bedrock Multi-Agent (AWS)
    bedrock_card = AgentCard(
        name="inventory-bedrock-agent-v1",
        agent_type=AgentType.REACT,
        framework="bedrock",
        model_backbone="amazon.nova-pro-v1:0",
        tools_manifest=[
            ToolDef(name="query_inventory_db", description="Run SQL against inventory DB",
                    parameters={"sql": "str"}),
            ToolDef(name="send_reorder_request", description="Trigger reorder via SNS",
                    parameters={"sku": "str", "qty": "int"}),
            ToolDef(name="get_supplier_info", description="Fetch supplier details from KB",
                    parameters={"supplier_id": "str"}),
        ],
        memory_type=MemoryType.VECTOR_DB,
        pass_k=8,
        max_cost_usd=0.30,
        sla_latency_ms=15_000,
        golden_milestones=["inventory_checked", "reorder_triggered", "confirmation_sent"],
        tags={"team": "supply-chain", "env": "prod", "region": "us-west-2"},
    )
    registry.register(bedrock_card)

    # 9.5 — Autonomous Browser Agent (Custom)
    browser_card = AgentCard(
        name="web-browser-agent-v1",
        agent_type=AgentType.REACT,
        framework="custom",
        model_backbone="claude-sonnet-4-6",
        tools_manifest=[
            ToolDef(name="navigate", description="Navigate browser to URL",
                    parameters={"url": "str"}),
            ToolDef(name="click", description="Click element by selector",
                    parameters={"selector": "str"}),
            ToolDef(name="type_text", description="Type text into a field",
                    parameters={"selector": "str", "text": "str"}),
            ToolDef(name="screenshot", description="Capture current page screenshot",
                    parameters={}),
            ToolDef(name="extract_text", description="Extract text from current page",
                    parameters={"selector": "str"}),
        ],
        pass_k=5,
        max_cost_usd=0.20,
        sla_latency_ms=120_000,
        golden_milestones=["page_loaded", "data_extracted", "form_submitted", "confirmation_seen"],
    )
    registry.register(browser_card)


if __name__ == "__main__":
    print("=" * 60)
    print("AGENT REGISTRATION")
    print("=" * 60)
    register_example_agents()
    print(f"\n✓ Registered {len(registry)} agents")
    for card in registry.list_all():
        print(f"  [{card.agent_type.value}] {card.name} → {len(card.eval_categories or [])} categories")
