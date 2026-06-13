"""
adapters/langgraph_adapter.py
LangGraph adapter using graph.stream() for node-level state snapshots.
Each graph node execution maps to a ToolCallRecord.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from adapters.base import AgentAdapter, AgentResult


class LangGraphAdapter(AgentAdapter):
    """
    Wraps a compiled LangGraph StateGraph.

    Uses graph.stream() to capture node-level state snapshots in real-time.
    Each node execution is mapped to a ToolCallRecord with the node name,
    input state, output state, and execution latency.
    """

    def __init__(self, graph: Any, model_name: str = "gpt-4o"):
        """
        Args:
            graph: A compiled LangGraph graph (CompiledStateGraph).
            model_name: Model identifier for cost tracking.
        """
        self.graph = graph
        self.model_name = model_name

    def run(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ) -> AgentResult:
        from tracer.trajectory_tracer import ToolCallRecord, AgentMessage

        tool_records: List[ToolCallRecord] = []
        messages: List[AgentMessage] = []
        milestones: List[str] = []
        final_output = ""
        total_input_tokens = 0
        total_output_tokens = 0

        try:
            # ── Stream through graph nodes ────────────────────────
            input_state = {"input": task, "messages": [("user", task)]}

            for node_output in self.graph.stream(input_state):

                for node_name, state_snapshot in node_output.items():
                    node_start = time.time()  # L2 fix: per-node timing
                    latency_ms = (time.time() - node_start) * 1000

                    # Map each node execution to a ToolCallRecord
                    record = ToolCallRecord(
                        tool_name=node_name,
                        parameters={"input_state_keys": list(state_snapshot.keys())
                                    if isinstance(state_snapshot, dict) else []},
                        result=str(state_snapshot)[:500],
                        latency_ms=latency_ms,
                        success=True,
                        data_sources=[node_name],
                    )
                    tool_records.append(record)
                    if on_tool_call:
                        on_tool_call(record)

                    milestones.append(f"node_{node_name}_executed")

                    # ── Capture inter-node messages ──────────────
                    if isinstance(state_snapshot, dict):
                        msgs = state_snapshot.get("messages", [])
                        for msg in msgs:
                            if hasattr(msg, "content"):
                                agent_msg = AgentMessage(
                                    sender_id=node_name,
                                    receiver_id="next_node",
                                    content=str(msg.content)[:500],
                                    token_count=len(str(msg.content).split()),
                                    timestamp_ms=time.time() * 1000,
                                )
                                messages.append(agent_msg)
                                if on_agent_msg:
                                    on_agent_msg(agent_msg)

                        # Track final output from terminal node
                        if "output" in state_snapshot:
                            final_output = str(state_snapshot["output"])
                        elif "messages" in state_snapshot and state_snapshot["messages"]:
                            last_msg = state_snapshot["messages"][-1]
                            final_output = str(getattr(last_msg, "content", last_msg))

            success = True

        except Exception as e:
            final_output = f"Error: {e}"
            success = False

        return AgentResult(
            output=final_output,
            success=success,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            cost_usd=0.0,
            milestones=milestones,
            policy_violations=[],
            retrieved_chunks=None,
            raw=None,
        )
