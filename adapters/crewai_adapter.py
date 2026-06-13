"""
adapters/crewai_adapter.py
CrewAI adapter using crew.kickoff() and task callback hooks.
Captures inter-agent messages via crew log access.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from adapters.base import AgentAdapter, AgentResult


class CrewAIAdapter(AgentAdapter):
    """
    Wraps a CrewAI Crew instance.

    Uses crew.kickoff() to execute, captures inter-agent messages
    via crew log inspection, and maps task callbacks to ToolCallRecords.
    """

    def __init__(self, crew: Any, task_descriptions: Optional[Dict[str, str]] = None):
        """
        Args:
            crew: A crewai.Crew instance.
            task_descriptions: Optional mapping of task names to descriptions.
        """
        self.crew = crew
        self.task_descriptions = task_descriptions or {}

    def run(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ) -> AgentResult:
        from tracer.trajectory_tracer import ToolCallRecord, AgentMessage

        milestones: List[str] = []
        policy_violations: List[str] = []
        tool_records: List[ToolCallRecord] = []
        messages: List[AgentMessage] = []

        t0 = time.time()

        try:
            # Execute the crew
            result = self.crew.kickoff(inputs={"task": task})

            # Extract output
            output = str(result)
            success = True

            # ── Capture inter-agent messages from crew logs ─────
            try:
                logs = self.crew.get_logs() if hasattr(self.crew, "get_logs") else []
                for i, log_entry in enumerate(logs):
                    sender = log_entry.get("agent", f"agent_{i}")
                    receiver = log_entry.get("delegated_to", "orchestrator")
                    content = log_entry.get("output", "")

                    msg = AgentMessage(
                        sender_id=str(sender),
                        receiver_id=str(receiver),
                        content=str(content)[:500],
                        token_count=len(str(content).split()),
                        timestamp_ms=time.time() * 1000,
                    )
                    messages.append(msg)
                    if on_agent_msg:
                        on_agent_msg(msg)
            except Exception:
                pass  # Graceful degradation if logs unavailable

            # ── Capture tool usage from tasks ───────────────────
            try:
                for crew_task in self.crew.tasks:
                    if hasattr(crew_task, "tools_used"):
                        for tool in crew_task.tools_used:
                            record = ToolCallRecord(
                                tool_name=getattr(tool, "name", str(tool)),
                                parameters={},
                                result=str(getattr(crew_task, "output", ""))[:300],
                                latency_ms=0.0,
                                success=True,
                            )
                            tool_records.append(record)
                            if on_tool_call:
                                on_tool_call(record)

                    milestones.append(f"task_{crew_task.description[:30]}_completed")
            except Exception:
                pass

        except Exception as e:
            output = f"Error: {e}"
            success = False

        elapsed_ms = (time.time() - t0) * 1000

        return AgentResult(
            output=output,
            success=success,
            input_tokens=0,   # CrewAI doesn't expose token counts natively
            output_tokens=0,
            cost_usd=0.0,
            milestones=milestones,
            policy_violations=policy_violations,
            retrieved_chunks=None,
            raw=result if success else None,
        )
