"""
adapters/odysseus_adapter.py
Adapter for the Odysseus autonomous workspace agent (Docker, default :7000).

Subclasses RemoteAgentAdapter ON PURPOSE: the EvalHarness gates its k=1
fast-path on `isinstance(adapter, RemoteAgentAdapter)`, so a sibling class
would trigger pass_k (8) real executions per task — i.e. 8x shell/file side
effects.  Staying in the family keeps k=1.

Two execution modes (the eval suite picks per task):
  • chat  — POST /api/v1/chat   {"message": task}      (single reply, no trace)
  • agent — POST <agent_endpoint> {"task": task}        (autonomous, tool trace)

Mode resolution per task:
  1. a leading "AGENT:" / "CHAT:" directive in the task string (stripped before
     sending), else
  2. remote_config["mode"]  ("agent" | "chat" | "auto"; default "auto" → agent
     if an agent_endpoint is configured, else chat).

remote_config keys (beyond the base adapter's):
  base_url (req), auth_headers, health_endpoint (default /api/health)
  chat_endpoint   (default /api/v1/chat),  task_field   (default "message")
  agent_endpoint  (default /api/agent/run), agent_task_field (default "task")
  response_field  (None → auto-detect), timeout_ms, max_retries, extra_body
  model, session  (optional, sent with chat requests when set)

NOTE: the agent-run response shape is parsed defensively (many candidate keys)
because the live trace JSON is not yet pinned.  When confirmed, narrow
_TRACE_LIST_KEYS / _STEP_* below.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from adapters.base import AgentResult
from adapters.remote_adapter import RemoteAgentAdapter


# Candidate keys for the array of tool/step records in an agent-run response.
_TRACE_LIST_KEYS = ["tool_calls", "steps", "trace", "actions", "events", "history", "tools"]
# Candidate keys WITHIN one step record.
_STEP_NAME_KEYS = ["tool", "name", "tool_name", "action", "function", "tool_id"]
_STEP_PARAM_KEYS = ["parameters", "params", "input", "args", "arguments", "tool_input"]
_STEP_RESULT_KEYS = ["result", "output", "observation", "content", "stdout", "response"]
_STEP_STATUS_KEYS = ["status", "state", "success", "ok"]
_STEP_ERROR_KEYS = ["error", "stderr", "exception", "err"]
_STEP_EXIT_KEYS = ["exit_code", "returncode", "code", "exit_status"]


class OdysseusAdapter(RemoteAgentAdapter):
    """Chat + autonomous-agent adapter for Odysseus."""

    def __init__(self, remote_config: Dict[str, Any]):
        super().__init__(remote_config)
        self.health_endpoint = remote_config.get("health_endpoint", "/api/health")
        self.chat_endpoint = remote_config.get("chat_endpoint", "/api/v1/chat")
        self.task_field = remote_config.get("task_field", "message")
        self.agent_endpoint = remote_config.get("agent_endpoint", "/api/agent/run")
        self.agent_task_field = remote_config.get("agent_task_field", "task")
        self.mode = (remote_config.get("mode") or "auto").lower()
        self.model = remote_config.get("model")
        self.session = remote_config.get("session")
        self._agent_unavailable = False  # set once an agent endpoint 404/405s

    # ── Mode + payload ───────────────────────────────────────────────────
    def _resolve_mode(self, task: str) -> Tuple[str, str]:
        """Return (mode, clean_task), honouring AGENT:/CHAT: directives."""
        stripped = task.lstrip()
        upper = stripped[:7].upper()
        if upper.startswith("AGENT:"):
            return "agent", stripped[6:].lstrip()
        if upper.startswith("CHAT:"):
            return "chat", stripped[5:].lstrip()
        if self.mode == "auto":
            return ("agent" if self.agent_endpoint else "chat"), task
        return self.mode, task

    def _chat_payload(self, task: str) -> Dict[str, Any]:
        payload = {self.task_field: task}
        if self.model:
            payload["model"] = self.model
        if self.session:
            payload["session"] = self.session
        payload.update(self.extra_body)
        return payload

    def _agent_payload(self, task: str) -> Dict[str, Any]:
        payload = {self.agent_task_field: task}
        payload.update(self.extra_body)
        return payload

    # ── HTTP with retry (only 5xx/network are retried; 4xx is terminal) ──
    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Tuple[bool, Any, Optional[str], Optional[int]]:
        url = f"{self.base_url}{endpoint}"
        headers = self._build_headers()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(url, json=payload, headers=headers)
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:400]}"
                    # 4xx is a client error (missing endpoint / bad body) — do NOT
                    # retry; retrying just spams the server. Only retry 5xx.
                    if resp.status_code < 500 or attempt >= self.max_retries:
                        print(f"[Odysseus] {last_error}")
                        return False, None, last_error, resp.status_code
                    time.sleep(min(2 ** attempt, 8))
                    continue
                try:
                    return True, resp.json(), None, resp.status_code
                except Exception:
                    return True, resp.text, None, resp.status_code
            except httpx.ConnectError as e:
                return False, None, f"Connection error: {e}", None
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
        return False, None, last_error, None

    # ── Trace extraction (defensive) ─────────────────────────────────────
    @staticmethod
    def _first(d: Dict, keys: List[str], default=None):
        for k in keys:
            if isinstance(d, dict) and k in d and d[k] not in (None, ""):
                return d[k]
        return default

    @classmethod
    def _find_trace_list(cls, data: Any) -> List[Dict]:
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if not isinstance(data, dict):
            return []
        for key in _TRACE_LIST_KEYS:
            val = data.get(key)
            if isinstance(val, list) and val and isinstance(val[0], dict):
                return val
        # Nested one level (e.g. {"result": {"steps": [...]}})
        for v in data.values():
            if isinstance(v, dict):
                nested = cls._find_trace_list(v)
                if nested:
                    return nested
        return []

    @classmethod
    def _normalise_step(cls, step: Dict) -> Dict:
        name = cls._first(step, _STEP_NAME_KEYS, "unknown")
        params = cls._first(step, _STEP_PARAM_KEYS, {})
        if not isinstance(params, dict):
            params = {"value": params}
        result = cls._first(step, _STEP_RESULT_KEYS, "")
        error = cls._first(step, _STEP_ERROR_KEYS)
        exit_code = cls._first(step, _STEP_EXIT_KEYS)
        try:
            exit_code = int(exit_code) if exit_code is not None else None
        except (ValueError, TypeError):
            exit_code = None

        status_raw = cls._first(step, _STEP_STATUS_KEYS)
        if isinstance(status_raw, bool):
            success = status_raw
        elif isinstance(status_raw, str):
            success = status_raw.strip().upper() in ("OK", "SUCCESS", "DONE", "COMPLETED", "PASSED", "TRUE")
        else:
            success = error is None and (exit_code in (None, 0))

        dur = cls._first(step, ["duration_ms", "latency_ms"])
        if dur is None:
            ds = cls._first(step, ["duration_s", "duration", "elapsed_s"])
            dur = float(ds) * 1000 if ds is not None else 0.0

        return {
            "tool_name": str(name),
            "parameters": params,
            "result": str(result),
            "success": bool(success),
            "error": str(error) if error else None,
            "latency_ms": float(dur or 0.0),
        }

    # ── Run ──────────────────────────────────────────────────────────────
    def run(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ) -> AgentResult:
        mode, clean = self._resolve_mode(task)
        if mode == "agent" and self._agent_unavailable:
            mode = "chat"  # sticky fallback after a prior 404/405
        endpoint = self.agent_endpoint if mode == "agent" else self.chat_endpoint
        payload = self._agent_payload(clean) if mode == "agent" else self._chat_payload(clean)
        print(f"[Odysseus] mode={mode} POST {self.base_url}{endpoint}")

        t0 = time.time()
        ok, data, err, status = self._post(endpoint, payload)

        # Agent surface absent on this build (404/405) → fall back to chat,
        # once and stickily, so we don't spam the missing endpoint.
        if not ok and mode == "agent" and status in (404, 405):
            self._agent_unavailable = True
            print(f"[Odysseus] agent endpoint {endpoint} unavailable ({status}); "
                  f"falling back to chat ({self.chat_endpoint}) for this and future calls")
            mode, endpoint = "chat", self.chat_endpoint
            ok, data, err, status = self._post(endpoint, self._chat_payload(clean))

        latency_ms = (time.time() - t0) * 1000

        if not ok or data is None:
            print(f"[Odysseus] FAILED ({latency_ms:.0f}ms): {err}")
            return AgentResult(
                output=f"[ODYSSEUS ERROR] {err}",
                success=False,
                raw={"error": err, "endpoint": endpoint, "mode": mode},
            )

        output_text = self._extract_output(data)
        tokens = self._extract_tokens(data) if isinstance(data, dict) else {}

        # Extract + replay the tool trace (agent mode)
        steps = self._find_trace_list(data) if mode == "agent" else []
        tool_calls = [self._normalise_step(s) for s in steps]
        if on_tool_call:
            for tc in tool_calls:
                try:
                    from tracer.trajectory_tracer import ToolCallRecord
                    on_tool_call(ToolCallRecord(
                        tool_name=tc["tool_name"],
                        parameters=tc["parameters"],
                        result=tc["result"],
                        latency_ms=tc["latency_ms"],
                        success=tc["success"],
                        error=tc["error"],
                    ))
                except Exception:
                    pass

        milestones = [s["tool_name"] for s in tool_calls if s["success"]]

        return AgentResult(
            output=output_text,
            success=ok and bool(output_text) and not str(output_text).startswith("[ODYSSEUS ERROR]"),
            input_tokens=int(tokens.get("input_tokens", 0) or 0),
            output_tokens=int(tokens.get("output_tokens", 0) or 0),
            cost_usd=float(tokens.get("cost_usd", 0.0) or 0.0),
            milestones=milestones,
            raw=data if isinstance(data, dict) else {"text": data},
        )

    def health_check(self) -> Dict[str, Any]:
        url = f"{self.base_url}{self.health_endpoint}"
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(url, headers=self._build_headers())
                ctype = resp.headers.get("content-type", "")
                return {
                    "reachable": resp.status_code < 500,
                    "status_code": resp.status_code,
                    "latency_ms": resp.elapsed.total_seconds() * 1000,
                    "body": resp.json() if ctype.startswith("application/json") else resp.text[:200],
                }
        except Exception as e:
            return {"reachable": False, "error": str(e)}
