"""
adapters/remote_adapter.py
Generic HTTP adapter for any remotely-hosted agent.

Connects to agents exposed via REST API (ngrok, cloud, custom servers).
The adapter sends tasks via HTTP POST and maps the response to AgentResult.

Supports:
- Any base_url (ngrok, localhost, cloud endpoints)
- Configurable request/response field names
- Auth headers (Bearer, API key, custom)
- Timeout + retry with backoff
- Rich extraction of step_log, agents, token estimates, and retrieved chunks
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from adapters.base import AgentAdapter, AgentResult


# ── Token estimation heuristic ──────────────────────────────────────────
# Ollama local models don't expose token counts through the orchestrator API.
# We estimate based on step durations and known model throughput:
#   llama3.2:latest  ~10 tok/s
#   qwen3:4b         ~15 tok/s
#   qwen3:8b         ~12 tok/s
# Average: ~12 tok/s.  These are labelled as estimates, not actuals.
_ESTIMATED_TOKENS_PER_SECOND = 12


class RemoteAgentAdapter(AgentAdapter):
    """
    Framework-agnostic HTTP adapter for hosted agents.

    Enriches AgentResult with:
    - tool_calls extracted from step_log
    - agent_messages synthesised from agents_used + step_log
    - retrieved_chunks from research/financial previews
    - token estimates from step durations
    """

    # Common response field names across different agent frameworks
    RESPONSE_FIELD_CANDIDATES = [
        "output", "response", "result", "answer", "text",
        "content", "message", "reply", "data", "completion",
    ]

    # Common token usage field names
    TOKEN_FIELDS = {
        "input_tokens": ["input_tokens", "prompt_tokens", "tokens_in", "usage.prompt_tokens"],
        "output_tokens": ["output_tokens", "completion_tokens", "tokens_out", "usage.completion_tokens"],
        "cost_usd": ["cost", "cost_usd", "total_cost", "usage.cost"],
    }

    def __init__(self, remote_config: Dict[str, Any]):
        """
        Args:
            remote_config: Dict with keys:
                - base_url (str, required): Base URL of the hosted agent
                - chat_endpoint (str): POST endpoint for chat. Default: "/chat"
                - health_endpoint (str): GET endpoint for health. Default: "/health"
                - task_field (str): JSON key to send the task in. Default: "message"
                - response_field (str): JSON key to read the output from. Default: auto-detect
                - auth_headers (dict): Extra headers (e.g. {"Authorization": "Bearer xxx"})
                - timeout_ms (int): Request timeout in ms. Default: 30000
                - extra_body (dict): Extra fields to include in every request body
                - max_retries (int): Retry count on failure. Default: 2
        """
        if not remote_config or "base_url" not in remote_config:
            raise ValueError("remote_config must contain 'base_url'")

        self.base_url = remote_config["base_url"].rstrip("/")
        self.chat_endpoint = remote_config.get("chat_endpoint", "/chat")
        self.health_endpoint = remote_config.get("health_endpoint", "/health")
        self.task_field = remote_config.get("task_field", "message")
        self.response_field = remote_config.get("response_field", None)  # None = auto-detect
        self.auth_headers = remote_config.get("auth_headers", {})
        self.timeout_s = remote_config.get("timeout_ms", 30000) / 1000.0
        self.extra_body = remote_config.get("extra_body", {})
        self.max_retries = remote_config.get("max_retries", 2)

    def _build_headers(self) -> Dict[str, str]:
        """Build request headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            # ngrok free tier requires this header
            "ngrok-skip-browser-warning": "true",
        }
        headers.update(self.auth_headers)
        return headers

    def _build_payload(self, task: str, **overrides) -> Dict[str, Any]:
        """Build the request payload, with optional field overrides."""
        payload = {self.task_field: task}
        payload.update(self.extra_body)
        payload.update(overrides)
        return payload

    def _extract_output(self, data: Any) -> str:
        """Extract the agent's text output from the response."""
        # If response is a plain string
        if isinstance(data, str):
            return data

        # If response is a list, take the first item
        if isinstance(data, list):
            if data:
                return self._extract_output(data[0])
            return ""

        if not isinstance(data, dict):
            return str(data)

        # If response_field is explicitly configured, use it
        if self.response_field:
            # Support nested fields like "data.output"
            value = self._get_nested(data, self.response_field)
            if value is not None:
                return str(value)

        # Auto-detect: try common field names
        for field in self.RESPONSE_FIELD_CANDIDATES:
            value = self._get_nested(data, field)
            if value is not None:
                return str(value)

        # Last resort: return the full JSON as string
        import json
        return json.dumps(data, indent=2, default=str)

    def _extract_tokens(self, data: Dict) -> Dict[str, float]:
        """Extract token usage from response if available."""
        result = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
        if not isinstance(data, dict):
            return result

        for our_field, candidates in self.TOKEN_FIELDS.items():
            for candidate in candidates:
                value = self._get_nested(data, candidate)
                if value is not None:
                    try:
                        result[our_field] = float(value)
                        break
                    except (ValueError, TypeError):
                        pass
        return result

    # ── Pipeline-aware extraction (PipelineResult enrichment) ─────────

    @staticmethod
    def _extract_tool_calls_from_step_log(data: Dict) -> List[Dict]:
        """
        Convert PipelineResult.step_log into ToolCallRecord-compatible dicts.

        step_log entries: {step, total, agent, status, duration_s, detail}
        """
        step_log = data.get("step_log", [])
        if not isinstance(step_log, list):
            return []

        tool_calls = []
        for entry in step_log:
            if not isinstance(entry, dict):
                continue
            # status "OK" or "ok" → success=True
            status_raw = str(entry.get("status", "")).strip().upper()
            is_success = status_raw in ("OK", "SUCCESS", "DONE", "COMPLETE", "COMPLETED")
            tool_calls.append({
                "name": entry.get("agent", "unknown"),
                "parameters": {"step": entry.get("step", 0), "total": entry.get("total", 0)},
                "result": entry.get("detail", ""),
                "latency_ms": entry.get("duration_s", 0) * 1000,
                "success": is_success,
                "tool_name": entry.get("agent", "unknown"),  # alias for evaluators
            })
        return tool_calls

    @staticmethod
    def _extract_agent_messages_from_pipeline(data: Dict) -> List[Dict]:
        """
        Synthesise inter-agent messages from agents_used + step_log.

        Creates sequential hand-off messages: agent_A → agent_B with
        content derived from each step's detail field.
        """
        agents_used = data.get("agents_used", [])
        step_log = data.get("step_log", [])

        if not agents_used or len(agents_used) < 2:
            return []

        messages = []
        for i in range(len(agents_used) - 1):
            sender = agents_used[i]
            receiver = agents_used[i + 1]
            # Get detail from matching step_log entry
            detail = ""
            if i < len(step_log) and isinstance(step_log[i], dict):
                detail = step_log[i].get("detail", f"Step {i+1} complete")
            content = f"Handoff: {detail}" if detail else f"Pipeline step {i+1} → {i+2}"

            messages.append({
                "sender_id": sender,
                "receiver_id": receiver,
                "content": content,
                "token_count": len(content.split()),  # ~1 token/word heuristic
            })
        return messages

    @staticmethod
    def _extract_retrieved_chunks_from_pipeline(data: Dict) -> List[Dict]:
        """
        Build retrieved_chunks from PipelineResult previews and sources.

        Extracts from: research_preview, financial_preview, and sources_count.
        """
        chunks = []

        # Research preview → chunk with source info
        research = data.get("research_preview", "")
        if research:
            chunks.append({
                "content": research,
                "source": "deep_research",
                "type": "research",
            })

        # Financial preview → chunk
        financial = data.get("financial_preview", "")
        if financial:
            chunks.append({
                "content": financial,
                "source": "financial_analysis",
                "type": "financial",
            })

        # ABM report preview → chunk
        abm_report = data.get("abm_report_preview", "")
        if abm_report:
            chunks.append({
                "content": abm_report[:500],
                "source": "abm_simulation",
                "type": "abm_analysis",
            })

        # Investment report → chunk
        report = data.get("investment_report_preview", "")
        if report:
            chunks.append({
                "content": report,
                "source": "synthesis",
                "type": "synthesis_report",
            })

        # Sentiment → chunk
        sentiment = data.get("sentiment", "")
        if sentiment:
            chunks.append({
                "content": sentiment,
                "source": "sentiment_analysis",
                "type": "sentiment",
            })

        return chunks

    @staticmethod
    def _estimate_tokens_from_timings(data: Dict) -> Dict[str, int]:
        """
        Estimate input/output tokens from step durations when actual counts
        aren't available (e.g. local Ollama models).

        Returns: {"input_tokens": int, "output_tokens": int}
        Labelled as estimates in the report.
        """
        timings = data.get("timings", {})
        total_duration_ms = data.get("total_duration_ms", 0)

        if timings:
            total_s = sum(v for v in timings.values() if isinstance(v, (int, float)))
        elif total_duration_ms:
            total_s = total_duration_ms / 1000.0
        else:
            return {"input_tokens": 0, "output_tokens": 0}

        estimated_total = int(total_s * _ESTIMATED_TOKENS_PER_SECOND)
        # Rough split: 40% input (prompts), 60% output (generation)
        return {
            "input_tokens": int(estimated_total * 0.4),
            "output_tokens": int(estimated_total * 0.6),
        }

    @staticmethod
    def _extract_milestones_from_pipeline(data: Dict) -> List[str]:
        """
        Derive milestones from step_log statuses and pipeline results.
        """
        milestones = []
        step_log = data.get("step_log", [])

        for entry in step_log:
            if not isinstance(entry, dict):
                continue
            agent = entry.get("agent", "unknown")
            status = entry.get("status", "").upper()
            if status == "OK":
                milestones.append(f"{agent}_complete")

        # Check email/sheet status
        email_status = data.get("email_status", {})
        if email_status and email_status.get("message_id"):
            milestones.append("email_sent")

        return milestones

    @staticmethod
    def _get_nested(data: Dict, path: str) -> Any:
        """Get a nested value from a dict using dot notation (e.g. 'usage.tokens')."""
        keys = path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def run(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ) -> AgentResult:
        """
        Send task to the remote agent via HTTP POST and return AgentResult.
        Enriches result with step_log → tool_calls, agents → messages,
        timings → token estimates, previews → retrieved chunks.
        """
        url = f"{self.base_url}{self.chat_endpoint}"
        headers = self._build_headers()
        payload = self._build_payload(task)

        print(f"[RemoteAdapter] POST {url}")
        print(f"[RemoteAdapter] Payload: {payload}")
        print(f"[RemoteAdapter] Timeout: {self.timeout_s}s, Retries: {self.max_retries}")

        last_error = None
        response_data = None
        success = False
        start_time = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                print(f"[RemoteAdapter] Attempt {attempt + 1}/{self.max_retries + 1} ...")
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(url, json=payload, headers=headers)

                    print(f"[RemoteAdapter] Response: {resp.status_code} ({resp.elapsed.total_seconds():.1f}s)")

                    if resp.elapsed.total_seconds() > 30:
                        print(f"[RemoteAdapter] ⏳ Response took {resp.elapsed.total_seconds():.0f}s "
                              f"(agent is slow, not stuck)")

                    if resp.status_code >= 400:
                        last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                        print(f"[RemoteAdapter] ERROR: {last_error}")
                        if attempt < self.max_retries:
                            time.sleep(min(2 ** attempt, 8))
                            continue
                        break

                    # Try to parse as JSON
                    try:
                        response_data = resp.json()
                    except Exception:
                        response_data = resp.text

                    success = True
                    print(f"[RemoteAdapter] SUCCESS — response received")
                    break

            except httpx.TimeoutException:
                last_error = f"Timeout after {self.timeout_s}s on attempt {attempt + 1}"
                print(f"[RemoteAdapter] TIMEOUT: {last_error}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
            except httpx.ConnectError as e:
                last_error = f"Connection error: {e}"
                print(f"[RemoteAdapter] CONNECT ERROR: {last_error}")
                break  # Don't retry connection errors
            except Exception as e:
                last_error = f"Unexpected error: {e}"
                print(f"[RemoteAdapter] UNEXPECTED ERROR: {last_error}")
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue

        latency_ms = (time.time() - start_time) * 1000

        # Build result
        if not success or response_data is None:
            print(f"[RemoteAdapter] FAILED after {latency_ms:.0f}ms — {last_error}")
            return AgentResult(
                output=f"[REMOTE ERROR] {last_error}",
                success=False,
                raw={"error": last_error, "url": url, "latency_ms": latency_ms},
            )

        output_text = self._extract_output(response_data)

        # ── Enrich from PipelineResult if available ──────────────
        is_pipeline = isinstance(response_data, dict) and "step_log" in response_data

        if is_pipeline:
            # Step_log → tool calls
            tool_calls = self._extract_tool_calls_from_step_log(response_data)

            # Agents → messages
            agent_messages = self._extract_agent_messages_from_pipeline(response_data)

            # Timings → token estimates (when actual counts unavailable)
            tokens = self._extract_tokens(response_data)
            if tokens["input_tokens"] == 0 and tokens["output_tokens"] == 0:
                estimated = self._estimate_tokens_from_timings(response_data)
                tokens["input_tokens"] = estimated["input_tokens"]
                tokens["output_tokens"] = estimated["output_tokens"]
                print(f"[RemoteAdapter] Token estimate: ~{estimated['input_tokens']} in, ~{estimated['output_tokens']} out")

            # Previews → retrieved chunks
            retrieved_chunks = self._extract_retrieved_chunks_from_pipeline(response_data)

            # Milestones from step statuses
            milestones = self._extract_milestones_from_pipeline(response_data)

            print(f"[RemoteAdapter] Enriched: {len(tool_calls)} tool_calls, "
                  f"{len(agent_messages)} agent_msgs, {len(retrieved_chunks)} chunks, "
                  f"{len(milestones)} milestones")
        else:
            # Generic response — use legacy extraction
            tokens = self._extract_tokens(response_data) if isinstance(response_data, dict) else {}
            tool_calls = []
            agent_messages = []
            retrieved_chunks = None
            milestones = []

            # Legacy milestones extraction
            if isinstance(response_data, dict):
                for field in ["milestones", "steps", "stages", "progress"]:
                    value = response_data.get(field)
                    if isinstance(value, list):
                        milestones = [str(v) for v in value]
                        break

        # Fire tool call hooks
        if on_tool_call and tool_calls:
            for tc in tool_calls:
                try:
                    from tracer.trajectory_tracer import ToolCallRecord
                    record = ToolCallRecord(
                        tool_name=tc.get("name", "unknown"),
                        parameters=tc.get("parameters", {}),
                        result=str(tc.get("result", "")),
                        latency_ms=tc.get("latency_ms", 0),
                        success=tc.get("success", True),
                    )
                    on_tool_call(record)
                except Exception:
                    pass

        # Fire agent message hooks
        if on_agent_msg and agent_messages:
            for msg in agent_messages:
                try:
                    from tracer.trajectory_tracer import AgentMessage
                    am = AgentMessage(
                        sender_id=msg.get("sender_id", ""),
                        receiver_id=msg.get("receiver_id", ""),
                        content=msg.get("content", ""),
                        token_count=msg.get("token_count", 0),
                    )
                    on_agent_msg(am)
                except Exception:
                    pass

        # Fire retrieval hooks
        if on_retrieval and retrieved_chunks:
            for chunk in retrieved_chunks:
                try:
                    on_retrieval(chunk if isinstance(chunk, dict) else {"content": str(chunk)})
                except Exception:
                    pass

        # Legacy retrieved chunks extraction (non-pipeline responses)
        if retrieved_chunks is None and isinstance(response_data, dict):
            for field in ["retrieved_chunks", "sources", "context", "documents", "references"]:
                value = response_data.get(field)
                if isinstance(value, list):
                    retrieved_chunks = value
                    break

        return AgentResult(
            output=output_text,
            success=success and bool(output_text),
            input_tokens=int(tokens.get("input_tokens", 0)),
            output_tokens=int(tokens.get("output_tokens", 0)),
            cost_usd=float(tokens.get("cost_usd", 0.0)),
            milestones=milestones,
            retrieved_chunks=retrieved_chunks,
            raw=response_data,
        )

    def health_check(self) -> Dict[str, Any]:
        """Check if the remote agent is reachable."""
        url = f"{self.base_url}{self.health_endpoint}"
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, headers=self._build_headers())
                return {
                    "reachable": True,
                    "status_code": resp.status_code,
                    "latency_ms": resp.elapsed.total_seconds() * 1000,
                    "body": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text[:200],
                }
        except Exception as e:
            return {
                "reachable": False,
                "error": str(e),
            }
