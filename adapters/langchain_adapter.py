"""
adapters/langchain_adapter.py
LangChain adapter using BaseCallbackHandler for real-time hook capture.
Supports both LCEL chains and AgentExecutor.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from adapters.base import AgentAdapter, AgentResult


class LangChainAdapter(AgentAdapter):
    """
    Wraps any LangChain runnable (LCEL chain or AgentExecutor).

    Uses BaseCallbackHandler.on_tool_start / on_tool_end for real-time
    tool call capture.  Inter-agent messages are captured via
    on_chain_start for multi-step chains.
    """

    def __init__(self, runnable: Any, model_name: str = "gpt-4o"):
        """
        Args:
            runnable: A LangChain Runnable (RunnableSequence, AgentExecutor, etc.)
            model_name: Model identifier for cost tracking.
        """
        self.runnable = runnable
        self.model_name = model_name

    def run(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ) -> AgentResult:
        try:
            from langchain_core.callbacks import BaseCallbackHandler
        except ImportError:
            raise ImportError(
                "langchain-core is required for LangChainAdapter. "
                "Install with: pip install langchain-core"
            )

        # ── Build the real-time callback handler ─────────────────
        class _EvalCallbackHandler(BaseCallbackHandler):
            """Synchronous hooks — captures tool calls, retrievals, messages at trace time."""

            def __init__(self):
                self.tool_start_times: Dict[str, float] = {}  # keyed by run_id
                self.tool_names: Dict[str, str] = {}  # run_id → tool_name
                self.tool_call_records: list = []
                self.input_tokens: int = 0
                self.output_tokens: int = 0

            def on_tool_start(
                self,
                serialized: Dict[str, Any],
                input_str: str,
                *,
                run_id: Any = None,
                **kwargs,
            ):
                tool_name = serialized.get("name", "unknown_tool")
                key = str(run_id) if run_id else tool_name
                self.tool_start_times[key] = time.time()
                self.tool_names[key] = tool_name

            def on_tool_end(
                self,
                output: str,
                *,
                run_id: Any = None,
                **kwargs,
            ):
                # L1 fix: correlate by run_id, not by popping first dict entry
                from tracer.trajectory_tracer import ToolCallRecord

                key = str(run_id) if run_id else None
                tool_name = "unknown_tool"
                latency_ms = 0.0

                if key and key in self.tool_start_times:
                    tool_name = self.tool_names.pop(key, "unknown_tool")
                    latency_ms = (time.time() - self.tool_start_times.pop(key)) * 1000
                elif self.tool_start_times:
                    # Fallback: pop first entry (old behavior)
                    fallback_key = next(iter(self.tool_start_times))
                    tool_name = self.tool_names.pop(fallback_key, "unknown_tool")
                    latency_ms = (time.time() - self.tool_start_times.pop(fallback_key)) * 1000

                record = ToolCallRecord(
                    tool_name=tool_name,
                    parameters={"input": str(output)[:200]},
                    result=str(output)[:500],
                    latency_ms=latency_ms,
                    success=True,
                )
                self.tool_call_records.append(record)
                if on_tool_call:
                    on_tool_call(record)

            def on_retriever_end(
                self,
                documents: Any,
                *,
                run_id: Any = None,
                **kwargs,
            ):
                if on_retrieval and documents:
                    for doc in documents:
                        chunk = {
                            "content": getattr(doc, "page_content", str(doc)),
                            "metadata": getattr(doc, "metadata", {}),
                        }
                        on_retrieval(chunk)

            def on_llm_end(self, response: Any, **kwargs):
                # Extract token usage if available
                if hasattr(response, "llm_output") and response.llm_output:
                    usage = response.llm_output.get("token_usage", {})
                    self.input_tokens += usage.get("prompt_tokens", 0)
                    self.output_tokens += usage.get("completion_tokens", 0)

        # ── Execute ──────────────────────────────────────────────
        handler = _EvalCallbackHandler()
        config = {"callbacks": [handler]}

        result = None  # C5 fix: init before try to avoid UnboundLocalError
        try:
            result = self.runnable.invoke({"input": task}, config=config)
            output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
            success = True
        except Exception as e:
            output = f"Error: {e}"
            success = False

        return AgentResult(
            output=output,
            success=success,
            input_tokens=handler.input_tokens,
            output_tokens=handler.output_tokens,
            cost_usd=0.0,  # Calculated by cost evaluator
            milestones=[],
            policy_violations=[],
            retrieved_chunks=None,
            raw=result if success else None,
        )
