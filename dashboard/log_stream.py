"""
dashboard/log_stream.py
-----------------------
Real-time log broadcaster for the UI.

How it works
============
1. A `LogBroadcaster` keeps a ring buffer of the last N log lines (so a UI
   that connects mid-run sees context, not a blank screen) and a set of
   live asyncio queues — one per connected SSE client.
2. A `BroadcastLogHandler` (a stdlib `logging.Handler`) pushes every log
   record into the broadcaster.
3. A `StdoutTee` wraps `sys.stdout` / `sys.stderr` so that **plain
   `print(...)` lines** (which dominate this codebase) are *also*
   captured — without losing the original console output.
4. FastAPI exposes `GET /logs/stream` as a **Server-Sent Events** stream
   that the frontend opens with `new EventSource('/logs/stream')`.
5. Lines are tagged with an optional `job_id` so the UI can filter to one
   specific eval run.

Frontend usage
==============
    const es = new EventSource('http://localhost:8000/logs/stream');
    es.onmessage = (e) => {
        const { ts, level, msg, job_id } = JSON.parse(e.data);
        console.log(`[${ts}] ${level} ${msg}`);
    };

To filter to a single job:
    new EventSource('http://localhost:8000/logs/stream?job_id=' + folder);

To replay the buffered history first:
    new EventSource('http://localhost:8000/logs/stream?replay=1');
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Deque, Dict, Optional, Set

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse


# ── Context: current job_id (set by the eval background task) ────────────
# Allows logs emitted *inside* a job to be tagged so the UI can filter.
current_job_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_job_id", default=None
)


# ── The broadcaster ──────────────────────────────────────────────────────

class LogBroadcaster:
    def __init__(self, history_size: int = 500) -> None:
        self._history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._subscribers: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        # Fired whenever a new subscriber connects — used by
        # wait_for_subscriber() to gate eval start on the UI being ready.
        self._subscriber_event: Optional[asyncio.Event] = None
        # The asyncio loop the FastAPI server runs on. Captured lazily the
        # first time `subscribe()` runs (i.e. inside the SSE endpoint).
        # All cross-thread publishes are marshalled onto this loop via
        # call_soon_threadsafe so they don't silently no-op from worker
        # threads (FastAPI's threadpool, ABM threads, etc.).
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_event(self) -> asyncio.Event:
        if self._subscriber_event is None:
            self._subscriber_event = asyncio.Event()
        return self._subscriber_event

    def _fan_out(self, event: Dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def publish(self, event: Dict[str, Any]) -> None:
        """Push an event to every subscriber + the history ring buffer.

        Safe to call from ANY thread — if we're not on the event loop
        thread, the fan-out is scheduled via `call_soon_threadsafe`.
        Without this, events emitted from FastAPI's threadpool workers
        (sync endpoints like `POST /prompt` that run the orchestrator)
        would be lost.
        """
        self._history.append(event)
        loop = self._loop
        if loop is None:
            # No loop captured yet — fan out directly (happens during
            # very early startup before any SSE client has connected).
            self._fan_out(event)
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            self._fan_out(event)
        else:
            # Cross-thread (or no running loop in this thread): marshal
            # the put back onto the FastAPI event loop.
            try:
                loop.call_soon_threadsafe(self._fan_out, event)
            except RuntimeError:
                # Loop is closed (shutdown) — drop the event silently.
                pass

    def history(self) -> list:
        return list(self._history)

    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def subscribe(self) -> asyncio.Queue:
        # Capture the running loop the first time someone subscribes, so
        # cross-thread publishes can be marshalled onto it.
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        async with self._lock:
            self._subscribers.add(q)
            ev = self._ensure_event()
            ev.set()         # signal anyone waiting
            ev.clear()       # one-shot: next waiter will block again
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    async def wait_for_subscriber(self, timeout: float = 30.0) -> bool:
        """
        Block until at least one SSE client is connected, or until timeout.
        Returns True if a subscriber connected (or already was), False on timeout.
        """
        if self._subscribers:
            return True
        ev = self._ensure_event()
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


broadcaster = LogBroadcaster()


# ── Capture stdlib logging ───────────────────────────────────────────────

class BroadcastLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        broadcaster.publish({
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": msg,
            "job_id": current_job_id.get(),
            "source": "logging",
        })


# ── Capture `print(...)` output ──────────────────────────────────────────

class StdoutTee:
    """
    Wraps a stream so writes are mirrored to the broadcaster *and* the
    original console (so you don't lose the terminal view).
    """
    def __init__(self, original, level: str = "INFO") -> None:
        self._original = original
        self._level = level
        self._buf = ""

    def write(self, data: str) -> int:
        try:
            self._original.write(data)
        except Exception:
            pass
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                broadcaster.publish({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "level": self._level,
                    "logger": "stdout",
                    "msg": line,
                    "job_id": current_job_id.get(),
                    "source": "print",
                })
        return len(data)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        try:
            return self._original.isatty()
        except Exception:
            return False

    def __getattr__(self, name):
        return getattr(self._original, name)


def install(level: int = logging.INFO) -> None:
    """
    Idempotent: install the log handler on the root logger and tee stdout/stderr.
    Call once at app startup.
    """
    root = logging.getLogger()
    # Avoid double-install on reload
    if any(isinstance(h, BroadcastLogHandler) for h in root.handlers):
        return

    handler = BroadcastLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.setLevel(level)
    root.addHandler(handler)
    if root.level > level or root.level == logging.NOTSET:
        root.setLevel(level)

    # Some agent modules call `logging.basicConfig()` themselves and set
    # `propagate=False` on their loggers, which means our root handler
    # never sees their output. Force-propagate the known offenders so we
    # capture every line during a pipeline run.
    for name in (
        "orchestrator", "orchestrator_v6", "prompt_router", "llm_router",
        "src", "src.abm", "src.abm.simulation", "src.abm.agents",
        "src.abm.environment", "src.abm.config_agent", "src.abm.report_agent",
        "src.abm.trend_predictor", "src.abm.graphrag",
        "Crewai_module", "Marketing", "evaluation",
        "evaluation.judge_validator", "utils", "utils.cache",
    ):
        try:
            lg = logging.getLogger(name)
            lg.propagate = True
            if lg.level == logging.NOTSET or lg.level > level:
                lg.setLevel(level)
        except Exception:
            pass

    # Tee print() output
    if not isinstance(sys.stdout, StdoutTee):
        sys.stdout = StdoutTee(sys.stdout, level="INFO")
    if not isinstance(sys.stderr, StdoutTee):
        sys.stderr = StdoutTee(sys.stderr, level="ERROR")


# ── SSE endpoint ─────────────────────────────────────────────────────────

router = APIRouter(tags=["Live Logs"])


async def _event_stream(request: Request, job_id: Optional[str], replay: bool) -> AsyncIterator[str]:
    """
    SSE generator. We deliberately do NOT call request.is_disconnected()
    inside the loop — known Starlette behavior is for it to return True
    spuriously under load, which would close the stream just before the
    pipeline starts emitting events. We let the natural CancelledError
    from `yield` propagate when the client actually disconnects.
    """
    q = await broadcaster.subscribe()
    try:
        # Initial comment so the browser knows the stream is open
        yield ": connected\n\n"

        # Optional replay of buffered history (filtered)
        if replay:
            for ev in broadcaster.history():
                if job_id and ev.get("job_id") != job_id:
                    continue
                yield f"data: {json.dumps(ev)}\n\n"

        last_ping = time.time()
        while True:
            try:
                ev = await asyncio.wait_for(q.get(), timeout=15)
            except asyncio.TimeoutError:
                # Heartbeat — keeps the connection alive through proxies.
                # If the client is gone, this yield raises and we exit.
                yield f": ping {int(time.time() - last_ping)}\n\n"
                last_ping = time.time()
                continue

            if job_id and ev.get("job_id") != job_id:
                continue
            yield f"data: {json.dumps(ev)}\n\n"
    except asyncio.CancelledError:
        # Client disconnected — normal SSE termination
        raise
    except Exception as e:
        # Log but don't crash the whole stream — emit a final error event
        try:
            yield f"data: {json.dumps({'level': 'ERROR', 'msg': f'stream error: {e}'})}\n\n"
        except Exception:
            pass
    finally:
        await broadcaster.unsubscribe(q)


@router.get("/logs/stream")
async def logs_stream(
    request: Request,
    job_id: Optional[str] = None,
    replay: bool = False,
):
    """
    Server-Sent Events stream of live logs.

    Query params:
      job_id : filter to a single eval/run id (matches the folder name)
      replay : if 1/true, replay buffered history first (last ~500 lines)
    """
    return StreamingResponse(
        _event_stream(request, job_id, replay),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
            "Connection": "keep-alive",
        },
    )


@router.get("/logs/history")
async def logs_history(job_id: Optional[str] = None, limit: int = 500):
    """Snapshot of recent log lines (non-streaming). Useful for a one-shot fetch."""
    items = broadcaster.history()
    if job_id:
        items = [e for e in items if e.get("job_id") == job_id]
    return {"count": len(items), "items": items[-limit:]}


@router.post("/logs/test")
async def logs_test(message: str = "hello from /logs/test", n: int = 5):
    """
    Push N synthetic log lines into the broadcaster.
    Use this to verify the SSE stream is working end-to-end without having
    to trigger a 20-minute eval run.
    """
    import logging as _log
    logger = _log.getLogger("logs.test")
    for i in range(n):
        logger.info(f"[test {i+1}/{n}] {message}")
        # Also emit a print() to exercise the stdout tee
        print(f"[test-print {i+1}/{n}] {message}")
    return {
        "success": True,
        "emitted": n,
        "message": f"Pushed {n} test events. Open GET /logs/history to see them.",
    }


_VIEWER_HTML = """<!DOCTYPE html>
<html><head><title>Live Logs</title><style>
body{font-family:sans-serif;margin:0;padding:10px;background:#1e1e1e;color:#ddd}
.bar{display:flex;gap:10px;align-items:center;padding:6px;flex-wrap:wrap}
.chip{padding:3px 8px;background:#2a2a2a;border-radius:4px;font-size:12px}
.chip.ok{background:#0d4d0d}.chip.err{background:#5a1a1a}
.row{display:flex;gap:10px;height:calc(100vh - 70px)}
.panel{flex:1;display:flex;flex-direction:column;background:#111;border-radius:6px;overflow:hidden;min-width:0}
.panel h3{margin:0;padding:6px 10px;background:#2a2a2a;font-size:12px;font-weight:600}
.body{flex:1;overflow-y:auto;padding:6px 10px;font-family:Consolas,monospace;font-size:12px;line-height:1.4}
.line{white-space:pre-wrap;word-break:break-all}
.INFO{color:#88ccff}.WARNING{color:#ffcc55}.ERROR{color:#ff5555}.DEBUG{color:#777}
</style></head>
<body>
<div class="bar">
  <strong>Live Logs</strong>
  <span class="chip" id="s8000">eval :8000 — connecting…</span>
  <span class="chip" id="s8080">agent :8080 — connecting…</span>
  <span class="chip" id="filter">filter: ALL</span>
  <button onclick="document.getElementById('l8000').innerHTML='';document.getElementById('l8080').innerHTML=''">Clear</button>
</div>
<div class="row">
  <div class="panel"><h3>Eval Pipeline (port 8000)</h3><div class="body" id="l8000"></div></div>
  <div class="panel"><h3>Agent Orchestrator (port 8080)</h3><div class="body" id="l8080"></div></div>
</div>
<script>
const qs = new URLSearchParams(location.search);
const jobId = qs.get('job_id');
const evalPort = qs.get('eval_port') || '8000';
const agentPort = qs.get('agent_port') || '8080';
if (jobId) document.getElementById('filter').textContent = 'filter: ' + jobId;

function connect(port, panelId, chipId) {
  const box  = document.getElementById(panelId);
  const chip = document.getElementById(chipId);
  const params = new URLSearchParams({ replay: '1' });
  if (jobId) params.set('job_id', jobId);
  const es = new EventSource(`http://${location.hostname}:${port}/logs/stream?` + params);
  let count = 0;
  es.onopen    = () => { chip.textContent = `${chipId === 's8000' ? 'eval :8000' : 'agent :8080'} — connected`; chip.className = 'chip ok'; };
  es.onerror   = () => { chip.textContent = `${chipId === 's8000' ? 'eval :8000' : 'agent :8080'} — reconnecting…`; chip.className = 'chip err'; };
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    const t = (ev.ts || '').slice(11, 19);
    const d = document.createElement('div');
    d.className = 'line ' + (ev.level || 'INFO');
    d.textContent = `[${t}] ${(ev.level || '').padEnd(5)} ${ev.msg}`;
    box.appendChild(d);
    box.scrollTop = box.scrollHeight;
    chip.textContent = `${chipId === 's8000' ? 'eval :8000' : 'agent :8080'} — ${++count} events`;
  };
}
connect(evalPort,  'l8000', 's8000');
connect(agentPort, 'l8080', 's8080');
</script>
</body></html>"""


@router.get("/favicon.ico", include_in_schema=False)
async def _favicon():
    """Empty 1x1 transparent PNG — silences the browser's favicon 404."""
    from fastapi.responses import Response
    # 1x1 transparent PNG
    png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return Response(content=png, media_type="image/png")


@router.get("/logs/viewer")
async def logs_viewer():
    """
    Built-in HTML viewer for the live log stream.
    Open in browser:
      http://localhost:8000/logs/viewer
      http://localhost:8000/logs/viewer?job_id=<folder_name>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_VIEWER_HTML)


@router.get("/logs/debug")
async def logs_debug():
    """
    Diagnostic: returns the current state of the broadcaster.
    Use to confirm the log capture is actually installed.
    """
    import sys as _sys
    return {
        "subscribers": len(broadcaster._subscribers),
        "buffered_events": len(broadcaster._history),
        "stdout_teed": type(_sys.stdout).__name__ == "StdoutTee",
        "stderr_teed": type(_sys.stderr).__name__ == "StdoutTee",
        "root_logger_handlers": [
            type(h).__name__ for h in logging.getLogger().handlers
        ],
        "broadcast_handler_installed": any(
            isinstance(h, BroadcastLogHandler)
            for h in logging.getLogger().handlers
        ),
    }
