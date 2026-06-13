"""
adapters/autogen_adapter.py
AutoGen adapter using ConversableAgent.register_reply() for message interception.
Captures every message in GroupChat conversations.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from adapters.base import AgentAdapter, AgentResult


class AutoGenAdapter(AgentAdapter):
    """
    Wraps an AutoGen GroupChat / ConversableAgent setup.

    Uses register_reply() hooks to intercept every message in the
    GroupChat conversation for real-time capture.
    """

    def __init__(
        self,
        agents: List[Any],
        group_chat: Optional[Any] = None,
        manager: Optional[Any] = None,
    ):
        """
        Args:
            agents: List of AutoGen ConversableAgent instances.
            group_chat: Optional GroupChat instance.
            manager: Optional GroupChatManager instance.
        """
        self.agents = agents
        self.group_chat = group_chat
        self.manager = manager

    def run(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ) -> AgentResult:
        from tracer.trajectory_tracer import ToolCallRecord, AgentMessage

        messages: List[AgentMessage] = []
        tool_records: List[ToolCallRecord] = []
        milestones: List[str] = []

        # ── Register reply hooks on all agents ────────────────────
        def _make_message_hook(agent_name: str):
            """Create a closure that captures messages from this agent."""
            def _hook(recipient, messages_list, sender, config):
                if messages_list and on_agent_msg:
                    last_msg = messages_list[-1] if isinstance(messages_list, list) else messages_list
                    content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)

                    msg = AgentMessage(
                        sender_id=str(getattr(sender, "name", sender)),
                        receiver_id=str(getattr(recipient, "name", recipient)),
                        content=str(content)[:500],
                        token_count=len(str(content).split()),
                        timestamp_ms=time.time() * 1000,
                    )
                    messages.append(msg)
                    on_agent_msg(msg)

                # Return None, None to allow normal processing to continue
                return None, None
            return _hook

        # Install hooks (L3 fix: use run nonce to prevent hook accumulation)
        import uuid as _uuid
        run_nonce = str(_uuid.uuid4())
        self._current_run_nonce = run_nonce

        for agent in self.agents:
            try:
                hook = _make_message_hook(getattr(agent, "name", str(agent)))
                # Wrap hook to check run nonce
                _nonce = run_nonce
                _original_hook = hook
                def _guarded_hook(recipient, messages, sender, config, _h=_original_hook, _n=_nonce):
                    if getattr(self, '_current_run_nonce', None) != _n:
                        return None, None  # Stale hook from a previous run
                    return _h(recipient, messages, sender, config)

                agent.register_reply(
                    trigger=[type(a) for a in self.agents],
                    reply_func=_guarded_hook,
                    position=0,
                )
            except Exception:
                pass  # Graceful degradation

        # ── Execute ───────────────────────────────────────────────
        t0 = time.time()
        try:
            if self.manager and self.group_chat:
                # GroupChat mode
                initiator = self.agents[0]
                result = initiator.initiate_chat(
                    self.manager,
                    message=task,
                )
                # Extract output from chat history
                chat_history = getattr(result, "chat_history", [])
                output = chat_history[-1].get("content", "") if chat_history else str(result)
            else:
                # Two-agent mode
                result = self.agents[0].initiate_chat(
                    self.agents[1] if len(self.agents) > 1 else self.agents[0],
                    message=task,
                    max_turns=10,
                )
                chat_history = getattr(result, "chat_history", [])
                output = chat_history[-1].get("content", "") if chat_history else str(result)

            success = True
            milestones.append("chat_completed")

            # ── Extract tool calls from chat history ──────────────
            for entry in chat_history:
                if isinstance(entry, dict):
                    func_call = entry.get("function_call") or entry.get("tool_calls")
                    if func_call:
                        calls = func_call if isinstance(func_call, list) else [func_call]
                        for fc in calls:
                            name = fc.get("name", fc.get("function", {}).get("name", "unknown"))
                            record = ToolCallRecord(
                                tool_name=name,
                                parameters=fc.get("arguments", {}),
                                result="",
                                latency_ms=0.0,
                                success=True,
                            )
                            tool_records.append(record)
                            if on_tool_call:
                                on_tool_call(record)

        except Exception as e:
            output = f"Error: {e}"
            success = False

        return AgentResult(
            output=output,
            success=success,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            milestones=milestones,
            policy_violations=[],
            retrieved_chunks=None,
            raw=result if success else None,
        )
