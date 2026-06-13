"""
dashboard/api.py
Unified FastAPI backend for the Universal Agent Evaluation Pipeline.

Endpoints:
- Registry:     POST /agents/register, GET /agents, GET /agents/{id}
- Eval:         POST /evals/run, GET /evals/status/{id}
- Reports:      GET /reports, GET /reports/{id}, GET /reports/{id}/scorecard
- Drift:        GET /drift/alerts, GET /metrics/trends/{agent_id}
- Dashboard:    GET /dashboard/scorecard, GET /dashboard/cost-explorer
- Health:       GET /health

APScheduler hourly drift detection job with Slack/PagerDuty webhook.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from dashboard.log_stream import (
    router as logs_router,
    install as install_log_stream,
    current_job_id as _current_job_id,
    broadcaster as _log_broadcaster,
)

# Install the log capture EARLY — at module import — so we catch logs
# emitted during app construction, before uvicorn finishes configuring.
install_log_stream()

from registry.agent_card import AgentCard, AgentType, MemoryType
from registry.registry import AgentRegistry, registry
from harness.eval_harness import EvalHarness, EvalReport, ThresholdGate, EvalFailureError
from storage.report_store import ReportStore
from storage.mlflow_logger import MLflowLogger
from storage.artifact_store import ArtifactStore, LocalBackend


# ── Background Services ──────────────────────────────────────────────────

report_store = ReportStore()
mlflow_logger = MLflowLogger()
# Save trajectories inside reports/evals/ so all artifacts are co-located
artifact_store = ArtifactStore(backend=LocalBackend(base_dir="./reports/evals"))

# In-memory eval job tracking (D3 fix: bounded dict, max 1000)
_eval_jobs: Dict[str, Dict[str, Any]] = {}
_EVAL_JOBS_MAX = 1000


# ── Drift Detection Background Job ──────────────────────────────────────

async def drift_detection_job():
    """
    Hourly drift detection job.
    Queries metric_drift view for |drift| > 10%.
    Fires Slack/PagerDuty alerts and promotes failing tasks to regression suite.
    """
    try:
        alerts = await report_store.get_drift_alerts(threshold=0.10)
        if not alerts:
            return

        for alert in alerts:
            print(
                f"[DRIFT ALERT] agent={alert['agent_id']} "
                f"metric={alert['metric']} "
                f"drift={alert.get('drift', 0):.4f} "
                f"recent={alert.get('recent_avg', 0):.4f} "
                f"baseline={alert.get('baseline_avg', 0):.4f}"
            )

            # Fire webhook (Slack/PagerDuty)
            await _fire_drift_webhook(alert)

            # Auto-promote to regression suite
            await _promote_to_regression(alert)

    except Exception as e:
        print(f"[DRIFT] Error in drift detection: {e}")


async def _fire_drift_webhook(alert: Dict):
    """Send drift alert to Slack or PagerDuty."""
    webhook_url = os.getenv("DRIFT_WEBHOOK_URL")
    if not webhook_url:
        return

    try:
        import httpx

        payload = {
            "text": (
                f"🚨 *Metric Drift Detected*\n"
                f"Agent: `{alert['agent_id']}`\n"
                f"Metric: `{alert['metric']}`\n"
                f"Drift: `{alert.get('drift', 0):.4f}`\n"
                f"Recent Avg: `{alert.get('recent_avg', 0):.4f}` | "
                f"Baseline: `{alert.get('baseline_avg', 0):.4f}`"
            ),
        }
        async with httpx.AsyncClient() as client:
            await client.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        print(f"[DRIFT] Webhook error: {e}")


async def _promote_to_regression(alert: Dict):
    """Promote drifting metric to regression suite (adaptive feedback loop)."""
    regression_path = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "regression_suite.json"
    )
    try:
        if os.path.exists(regression_path):
            with open(regression_path, "r") as f:
                suite = json.load(f)
        else:
            suite = {"tasks": [], "promoted_from_drift": []}

        entry = {
            "agent_id": str(alert.get("agent_id", "")),
            "metric": alert.get("metric", ""),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "drift_value": alert.get("drift", 0),
        }

        # Avoid duplicates
        existing = {
            (e["agent_id"], e["metric"])
            for e in suite.get("promoted_from_drift", [])
        }
        if (entry["agent_id"], entry["metric"]) not in existing:
            suite.setdefault("promoted_from_drift", []).append(entry)
            with open(regression_path, "w") as f:
                json.dump(suite, f, indent=2)
            print(f"[DRIFT] Promoted to regression suite: {entry['metric']}")
    except Exception as e:
        print(f"[DRIFT] Promotion error: {e}")


# ── Scheduler Setup ──────────────────────────────────────────────────────

def _setup_scheduler():
    """Setup APScheduler for hourly drift detection."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            drift_detection_job,
            "interval",
            hours=1,
            id="drift_detection",
            replace_existing=True,
        )
        scheduler.start()
        print("[Scheduler] Hourly drift detection job started")
        return scheduler
    except ImportError:
        print("[Scheduler] APScheduler not installed — drift detection disabled")
        return None


# ── App Lifecycle ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    install_log_stream()  # idempotent; safety net if module-level call was missed
    scheduler = _setup_scheduler()
    yield
    if scheduler:
        scheduler.shutdown()
    await report_store.close()


# ── FastAPI App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="Universal Agent Evaluation Pipeline",
    description=(
        "Framework-agnostic system for registering, instrumenting, "
        "and evaluating any AI agent or multi-agent system. "
        "Based on CLEAR (2025) & MultiAgentBench (ACL 2025)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

origins = [
    "http://localhost:3000",          # Local frontend (React/Vite/etc.)
    "http://127.0.0.1:3000",
    "https://proexecutive-myrtie-subsimple.ngrok-free.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies and auth headers
    allow_methods=["*"],              # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],              # Allow all headers
)


# ── Access-log filter ────────────────────────────────────────────────────
# Suppress the initial `POST /evals/run` access log so the ONLY 200 OK line
# the frontend team sees for an eval is the completion line emitted at the
# end of the background task.
import logging as _logging


class _SuppressEvalRunStartFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        # uvicorn.access format: '<client> - "POST /evals/run HTTP/1.1" 200 OK'
        # Drop the initial submit log; keep the completion log
        # ('/evals/run/complete') and every other route.
        if '"POST /evals/run HTTP' in msg:
            return False
        return True


_logging.getLogger("uvicorn.access").addFilter(_SuppressEvalRunStartFilter())

# Mount the live-log SSE router (/logs/stream, /logs/history)
app.include_router(logs_router)


# ── Request/Response Models ──────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    agent_type: str
    framework: str
    model_backbone: str
    memory_type: str = "none"
    tools_manifest: List[Dict[str, Any]] = Field(default_factory=list)
    subagents: List[str] = Field(default_factory=list)
    pass_k: int = 8
    max_cost_usd: float = 5.0
    sla_latency_ms: int = 30000
    golden_trajectory: Optional[List[str]] = None
    golden_milestones: Optional[List[str]] = None
    golden_sources: Optional[List[str]] = None
    persona_spec: Optional[str] = None
    hk_params: Optional[Dict[str, Any]] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    remote_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Connection config for hosted agents. Required keys: base_url. "
            "Optional: chat_endpoint, health_endpoint, task_field, response_field, "
            "auth_headers, timeout_ms, extra_body"
        ),
    )

    # V2 fix: validate agent_type is a recognized enum value
    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v: str) -> str:
        valid = {e.value for e in AgentType}
        if v not in valid:
            raise ValueError(f"agent_type must be one of {valid}, got '{v}'")
        return v


class EvalRequest(BaseModel):
    agent_id: str
    task_suite: Optional[List[str]] = None
    git_sha: Optional[str] = None
    trigger: str = "manual"
    fail_on_threshold: bool = True


class EvalJobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | completed | failed
    report: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ── Health ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agents_registered": len(registry),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Registry Endpoints ──────────────────────────────────────────────────

@app.post("/agents/register")
async def register_agent(req: RegisterRequest):
    """Register a new agent card."""
    from registry.agent_card import ToolDef

    tools = [ToolDef(**t) for t in req.tools_manifest] if req.tools_manifest else []

    card = AgentCard(
        name=req.name,
        agent_type=AgentType(req.agent_type),
        framework=req.framework,
        model_backbone=req.model_backbone,
        memory_type=MemoryType(req.memory_type),
        tools_manifest=tools,
        subagents=req.subagents,
        pass_k=req.pass_k,
        max_cost_usd=req.max_cost_usd,
        sla_latency_ms=req.sla_latency_ms,
        golden_trajectory=req.golden_trajectory,
        golden_milestones=req.golden_milestones,
        golden_sources=req.golden_sources,
        persona_spec=req.persona_spec,
        hk_params=req.hk_params,
        tags=req.tags,
        remote_config=req.remote_config,
    )

    agent_id = registry.register(card)
    await report_store.save_agent_card(card)

    eval_categories = card.eval_categories or card.auto_infer_categories()
    return {
        "status_code": 200,
        "success": True,
        "message": f"✅ Agent '{card.name}' registered successfully with {len(eval_categories)} eval categories.",
        "agent_id": agent_id,
        "name": card.name,
        "framework": card.framework,
        "remote": card.remote_config is not None,
        "eval_categories": eval_categories,
        "category_count": len(eval_categories),
    }


@app.get("/agents")
async def list_agents():
    """List all registered agents."""
    agents = registry.list_all()
    return {
        "count": len(agents),
        "agents": [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "agent_type": a.agent_type.value,
                "framework": a.framework,
                "version": a.version,
                "eval_categories": a.eval_categories,
            }
            for a in agents
        ],
    }


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent card details."""
    try:
        card = registry.get(agent_id)
        return card.model_dump(mode="json")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")


# ── Eval Endpoints ───────────────────────────────────────────────────────


class TestConnectionRequest(BaseModel):
    base_url: str
    health_endpoint: str = "/health"
    auth_headers: Dict[str, str] = Field(default_factory=dict)


@app.post("/agents/test-connection")
async def test_connection(req: TestConnectionRequest):
    """
    Verify that a remote agent is reachable before registering.
    Probes the health endpoint and returns connectivity status.
    """
    from adapters.remote_adapter import RemoteAgentAdapter

    adapter = RemoteAgentAdapter({
        "base_url": req.base_url,
        "health_endpoint": req.health_endpoint,
        "auth_headers": req.auth_headers,
    })
    result = adapter.health_check()

    # Also try to discover OpenAPI spec
    discovery = {}
    try:
        import httpx
        headers = {"ngrok-skip-browser-warning": "true"}
        headers.update(req.auth_headers)
        with httpx.Client(timeout=10) as client:
            for path in ["/openapi.json", "/docs"]:
                try:
                    resp = client.get(f"{req.base_url.rstrip('/')}{path}", headers=headers)
                    if resp.status_code == 200:
                        discovery["openapi_url"] = f"{req.base_url}{path}"
                        if path == "/openapi.json":
                            spec = resp.json()
                            discovery["title"] = spec.get("info", {}).get("title", "")
                            discovery["endpoints"] = list(spec.get("paths", {}).keys())
                        break
                except Exception:
                    pass
    except Exception:
        pass

    reachable = bool(result.get("reachable"))
    upstream_status = result.get("status_code")
    latency = result.get("latency_ms")
    if reachable:
        msg = (
            f"✅ Remote agent is reachable at {req.base_url} "
            f"(upstream status={upstream_status}, latency={latency} ms)."
        )
    else:
        msg = (
            f"❌ Remote agent at {req.base_url} is NOT reachable "
            f"(upstream status={upstream_status})."
        )

    return {
        "status_code": 200,
        "success": reachable,
        "message": msg,
        **result,
        "discovery": discovery,
    }


# ── Report Summary Helpers ───────────────────────────────────────────

def _print_eval_summary(report, job_id: str) -> None:
    """Print a formatted evaluation summary to the console."""
    w = 58  # box width
    name = report.agent_name or report.agent_id
    dur = report.duration_ms / 1000

    print()
    print(f"╔{'═' * w}╗")
    print(f"║  EVAL REPORT: {name[:w-16]:<{w-16}}║")
    print(f"║  Job: {job_id[:w-8]:<{w-8}}║")
    print(f"╠{'═' * w}╣")

    for r in report.results:
        icon = "✅" if r.passed else "❌"
        # Pick the primary metric for display
        primary = ""
        for k, v in r.metrics.items():
            if v is not None:
                primary = f"{k}={v}"
                break
        line = f"{icon} {r.category:<22} {primary}"
        print(f"║  {line:<{w-4}}║")
        for warn in r.warnings[:2]:
            wline = f"   ⚠ {warn[:w-10]}"
            print(f"║  {wline:<{w-4}}║")

    print(f"╠{'═' * w}╣")
    overall = "✅ PASS" if report.overall_passed else "❌ FAIL"
    footer = f"OVERALL: {overall}  │  Duration: {dur:.1f}s  │  Tasks: {report.tasks_evaluated}"
    print(f"║  {footer:<{w-4}}║")
    print(f"╚{'═' * w}╝")
    print()


def _save_summary_txt(report, job_id: str) -> None:
    """Save a human-readable summary.txt in the eval report folder."""
    eval_dir = os.path.join(".", "reports", "evals", job_id)
    os.makedirs(eval_dir, exist_ok=True)

    name = report.agent_name or report.agent_id
    dur = report.duration_ms / 1000
    lines = [
        f"EVAL REPORT: {name}",
        f"Job ID: {job_id}",
        f"Timestamp: {report.timestamp.isoformat()}",
        f"Duration: {dur:.1f}s",
        f"Tasks Evaluated: {report.tasks_evaluated}",
        f"Total Runs: {report.total_runs}",
        "",
        "RESULTS",
        "-" * 50,
    ]

    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.category}")
        for k, v in r.metrics.items():
            if v is not None:
                lines.append(f"  {k}: {v}")
        for w in r.warnings:
            lines.append(f"  WARNING: {w}")
        lines.append("")

    overall = "PASS" if report.overall_passed else "FAIL"
    lines.append("-" * 50)
    lines.append(f"OVERALL: {overall}")

    summary_path = os.path.join(eval_dir, "summary.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[ReportStore] Saved summary.txt to {eval_dir}/")
    except Exception as e:
        print(f"[ReportStore] Error saving summary.txt: {e}")


def _make_eval_folder_name(agent_name: str, task_suite: list) -> str:
    """Generate a human-readable eval folder name from agent name + task + timestamp."""
    safe_name = re.sub(r'[^\w\-]', '_', agent_name)[:40]
    first_task = task_suite[0] if task_suite else "unknown"
    task_short = re.sub(r'[^\w]', '_', first_task[:30]).strip('_')[:20]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{safe_name}_{task_short}_{ts}"


@app.post("/evals/run")
async def run_eval(req: EvalRequest, background_tasks: BackgroundTasks):
    """
    Trigger an evaluation run.
    Returns a job_id for status polling.
    """
    import uuid

    # Internal UUID for tracking; folder name is set later once we know agent + task
    job_id = str(uuid.uuid4())

    # D3 fix: evict oldest jobs when capacity exceeded
    if len(_eval_jobs) >= _EVAL_JOBS_MAX:
        oldest_key = next(iter(_eval_jobs))
        del _eval_jobs[oldest_key]

    _eval_jobs[job_id] = {"status": "pending", "report": None, "error": None}

    def _log_completion(folder: str, http_status: int = 200):
        """
        Emit a uvicorn-style access log line once the background eval finishes
        so the frontend team sees a `200 OK` signal for the completion event
        (the original POST /evals/run already returned earlier).

        We print directly instead of routing through the `uvicorn.access`
        logger because its custom formatter expects a structured 5-tuple
        (client, method, path, http_version, status_code) on the record's
        args — passing a plain message string raises
        "ValueError: not enough values to unpack (expected 5, got 2)".
        """
        status_text = "OK" if 200 <= http_status < 300 else (
            "Internal Server Error" if http_status >= 500 else "Error"
        )
        line = (
            f'INFO:     127.0.0.1:0 - '
            f'"POST /evals/run/complete?job={folder or job_id} HTTP/1.1" '
            f'{http_status} {status_text}'
        )
        print(line, flush=True)

    async def _run():
        # Tag every log line emitted inside this task with the eval's folder
        # name so the UI can filter the SSE stream to one run.
        _token = _current_job_id.set(job_id)
        _eval_jobs[job_id]["status"] = "waiting_for_viewer"
        try:
            # ── Open the live-log viewer in the user's default browser ──
            # then BLOCK until that viewer is actually connected to /logs/stream,
            # so the user sees logs from the very first line of the eval.
            try:
                import webbrowser
                viewer_url = (
                    f"http://localhost:8000/logs/viewer"
                    f"?eval_port=8000&agent_port=8080"
                )
                print(f"[EvalRun] 🌐 Opening live-log viewer: {viewer_url}")
                webbrowser.open(viewer_url, new=2)
            except Exception as e:
                print(f"[EvalRun] WARN: could not auto-open browser: {e}")

            print("[EvalRun] ⏳ Waiting up to 30s for the viewer to connect…")
            connected = await _log_broadcaster.wait_for_subscriber(timeout=30.0)
            if connected:
                print(
                    f"[EvalRun] ✅ Viewer connected "
                    f"({_log_broadcaster.subscriber_count()} subscriber(s)). "
                    f"Starting eval…"
                )
            else:
                print(
                    "[EvalRun] ⚠ No viewer connected within 30s — "
                    "starting eval anyway. You can still open "
                    "http://localhost:8000/logs/viewer to see live progress."
                )

            _eval_jobs[job_id]["status"] = "running"
            # ── Auto-load default task suite if not provided ──────
            task_suite = req.task_suite
            if not task_suite:
                default_suite = os.path.join(
                    os.path.dirname(__file__), "..", "tasks", "smoke_suite.json"
                )
                try:
                    with open(default_suite, "r", encoding="utf-8") as f:
                        task_suite = json.load(f).get("tasks", [])
                    print(f"[EvalRun] Auto-loaded {len(task_suite)} tasks from smoke_suite.json")
                except FileNotFoundError:
                    raise ValueError("No task_suite provided and default smoke_suite.json not found")

            # Dynamic adapter factory — creates the right adapter for any framework
            from adapters.adapter_factory import AdapterFactory
            def adapter_factory(card):
                adapter = AdapterFactory.create(card)
                print(f"[EvalRun] Adapter created: {type(adapter).__name__}")
                return adapter

            # ── Human-readable folder name: agent_name + task + timestamp ──
            card = registry.get(req.agent_id)
            agent_name = card.name if card else req.agent_id
            folder_name = _make_eval_folder_name(agent_name, task_suite)
            # Switch the context tag from internal UUID to the readable folder
            _current_job_id.set(folder_name)

            print(f"[EvalRun] Starting eval for agent {req.agent_id} ({len(task_suite)} tasks)")
            print(f"[EvalRun] Report folder: {folder_name}")
            harness = EvalHarness(registry, adapter_factory)
            report = await harness.run_eval(
                agent_id=req.agent_id,
                task_suite=task_suite,
                git_sha=req.git_sha,
                trigger=req.trigger,
                fail_on_threshold=False,  # We handle threshold gate after persistence
            )

            # ── Formatted console summary ────────────────────────
            _print_eval_summary(report, folder_name)

            report_dict = report.model_dump(mode="json")

            # ── ALWAYS persist report (even if threshold check will fail) ──
            try:
                artifact_url = await artifact_store.upload_trajectory(
                    f"{folder_name}/trajectory", report_dict
                )
                await report_store.save_report(report, artifact_url, job_id=folder_name)
                await report_store.save_metrics(report)
                await mlflow_logger.log(report)
                _save_summary_txt(report, folder_name)
                print(f"[EvalRun] Report persisted to reports/evals/{folder_name}/")
            except Exception as pe:
                print(f"[EvalRun] WARNING: Persistence error (report still in memory): {pe}")

            # ── Threshold gate (after persistence) ───────────────
            _eval_jobs[job_id]["folder_name"] = folder_name
            _eval_jobs[job_id]["report"] = report_dict
            if req.fail_on_threshold:
                from harness.eval_harness import ThresholdGate, EvalFailureError
                violations = ThresholdGate.check(report, raise_on_failure=False)
                if violations:
                    _eval_jobs[job_id]["status"] = "failed"
                    _eval_jobs[job_id]["error"] = str(violations)
                    print(f"[EvalRun] Threshold violations: {violations}")
                    # Eval ran end-to-end and report was persisted — surface 200 OK
                    # for the frontend even though threshold gate failed.
                    print(f"[EvalRun] ✅ Completed (threshold gate failed) — job={folder_name}")
                    _log_completion(folder_name, 200)
                    return

            _eval_jobs[job_id]["status"] = "completed"
            print(f"[EvalRun] ✅ Completed — job={folder_name}")
            _log_completion(folder_name, 200)

        except Exception as e:
            import traceback
            print(f"[EvalRun] ERROR: {e}")
            traceback.print_exc()
            _eval_jobs[job_id]["status"] = "failed"
            _eval_jobs[job_id]["error"] = str(e)
            _log_completion(_eval_jobs[job_id].get("folder_name", job_id), 500)
        finally:
            try:
                _current_job_id.reset(_token)
            except Exception:
                pass

    background_tasks.add_task(_run)

    viewer_url = "http://localhost:8000/logs/viewer?eval_port=8000&agent_port=8080"
    return {
        "status_code": 202,
        "success": True,
        "job_id": job_id,
        "status": "waiting_for_viewer",
        "viewer_url": viewer_url,
        "message": (
            f"🚀 Eval queued. The live-log viewer will open in your browser. "
            f"The eval will start as soon as the viewer connects (or after 30s). "
            f"Viewer: {viewer_url}  |  Status: GET /evals/status/{job_id}"
        ),
    }


@app.get("/evals/status/{job_id}")
async def eval_status(job_id: str):
    """Get the status of an evaluation job."""
    if job_id not in _eval_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"❌ Job '{job_id}' not found. It may have expired or never existed.",
        )
    job = _eval_jobs[job_id]
    status = job["status"]

    status_messages = {
        "pending":             "⏳ Eval is queued and waiting to start.",
        "waiting_for_viewer":  "🌐 Eval is waiting for the live-log viewer to connect before starting.",
        "running":             "⚙️  Eval is currently running — evaluators are scoring the agent.",
        "completed":           "✅ Eval completed successfully — report is ready.",
        "failed":              "❌ Eval finished but failed threshold gate or errored out — see 'error' field.",
    }

    return {
        "status_code": 200,
        "success": status in ("completed", "running", "pending", "waiting_for_viewer"),
        "message": status_messages.get(status, f"Unknown status: {status}"),
        "job_id": job_id,
        "status": status,
        "folder_name": job.get("folder_name"),
        "report": job["report"],
        "error": job["error"],
    }


# ── Report Endpoints ─────────────────────────────────────────────────────

@app.get("/reports")
async def list_reports(agent_id: Optional[str] = None, limit: int = 50):
    """List eval reports, optionally filtered by agent."""
    reports = await report_store.get_reports(agent_id=agent_id, limit=limit)
    return {"count": len(reports), "reports": reports}


@app.get("/reports/{job_id}")
async def get_report(job_id: str):
    """Get a specific eval report by job_id."""
    # Fast path: in-memory
    if job_id in _eval_jobs and _eval_jobs[job_id]["report"]:
        return _eval_jobs[job_id]["report"]

    # File store fallback: reports/evals/{job_id}/report.json
    import json as _json
    report_file = f"./reports/evals/{job_id}/report.json"
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass

    # Persistent store fallback
    try:
        report = await report_store.get_report(job_id)
        if report:
            return report
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Report not found")


@app.get("/reports/{job_id}/scorecard")
async def get_scorecard(job_id: str):
    """Get a formatted scorecard for a report."""
    # Try pre-built scorecard from file store
    import json as _json
    scorecard_file = f"./reports/evals/{job_id}/scorecard.json"
    if os.path.exists(scorecard_file):
        try:
            with open(scorecard_file, "r", encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass

    # Fall back to building from in-memory report
    report = None
    if job_id in _eval_jobs and _eval_jobs[job_id]["report"]:
        report = _eval_jobs[job_id]["report"]
    else:
        # Try report store
        try:
            report = await report_store.get_report(job_id)
        except Exception:
            pass

    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    scorecard = {
        "agent": report.get("agent_name", ""),
        "version": report.get("agent_version", ""),
        "overall": "✅ PASS" if report.get("overall_passed") else "❌ FAIL",
        "categories": [],
    }

    for r in report.get("results", []):
        icon = "✅" if r.get("passed") else "❌"
        scorecard["categories"].append({
            "category": r.get("category", ""),
            "status": f"{icon} {'PASS' if r.get('passed') else 'FAIL'}",
            "metrics": r.get("metrics", {}),
            "warnings": r.get("warnings", []),
        })

    return scorecard


# ── Drift & Observability ───────────────────────────────────────────────

@app.get("/drift/alerts")
async def get_drift_alerts(threshold: float = 0.10):
    """Get current metric drift alerts."""
    alerts = await report_store.get_drift_alerts(threshold)
    return {"count": len(alerts), "alerts": alerts}


@app.get("/metrics/trends/{agent_id}")
async def get_metric_trends(agent_id: str, metric: Optional[str] = None):
    """Get metric trend data for an agent."""
    # Query from report store
    reports = await report_store.get_reports(agent_id=agent_id, limit=100)
    return {"agent_id": agent_id, "reports_count": len(reports), "reports": reports}


# ── Dashboard Convenience ───────────────────────────────────────────────

@app.get("/dashboard/scorecard")
async def dashboard_scorecard():
    """Overview scorecard of all registered agents and their latest eval results."""
    agents = registry.list_all()
    overview = []
    for agent in agents:
        overview.append({
            "agent_id": agent.agent_id,
            "name": agent.name,
            "type": agent.agent_type.value,
            "framework": agent.framework,
            "categories": len(agent.eval_categories or []),
        })
    return {"total_agents": len(agents), "agents": overview}


@app.get("/dashboard/cost-explorer")
async def cost_explorer():
    """Cost analysis across all recent eval runs."""
    reports = await report_store.get_reports(limit=100)
    return {"reports_count": len(reports), "reports": reports}


# ── Latest Results (cross-pipeline) ──────────────────────────────────────
# Exposes the most recent run folder from BOTH pipelines so the frontend
# can fetch the latest generated outputs from a single endpoint.
#
#   eval pipeline   -> reports/evals/<run>/{report.json, scorecard.json, ...}
#   agentic backend -> agentic_backend/reports/output/<run>/{*.md, *.json}
#
# Override locations with env vars EVAL_REPORTS_DIR and AGENTIC_REPORTS_DIR.

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_EVAL_DIR = os.path.abspath(os.path.join(_HERE, "..", "reports", "evals"))
_DEFAULT_AGENTIC_DIR = os.path.abspath(
    os.path.join(_HERE, "..", "..", "agentic_backend", "reports", "output")
)
EVAL_REPORTS_DIR = os.getenv("EVAL_REPORTS_DIR", _DEFAULT_EVAL_DIR)
AGENTIC_REPORTS_DIR = os.getenv("AGENTIC_REPORTS_DIR", _DEFAULT_AGENTIC_DIR)

# Cap on returned text-file content to keep responses light. Frontend can
# fetch the raw file via /latest-results/{pipeline}/{run_id}/file?name=... if needed.
_MAX_TEXT_BYTES = 200_000


def _safe_listdir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return [n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n))]


def _latest_run_folder(base_dir: str) -> Optional[str]:
    """Return the most recently modified run subfolder, or None."""
    subs = _safe_listdir(base_dir)
    if not subs:
        return None
    return max(subs, key=lambda n: os.path.getmtime(os.path.join(base_dir, n)))


def _load_run(base_dir: str, run_id: str) -> Dict[str, Any]:
    """Load every file in a run folder. JSON is parsed; .md/.txt/.html returned as text."""
    run_path = os.path.join(base_dir, run_id)
    if not os.path.isdir(run_path):
        raise FileNotFoundError(run_path)

    files: Dict[str, Any] = {}
    for name in sorted(os.listdir(run_path)):
        fpath = os.path.join(run_path, name)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(name)[1].lower()
        try:
            if ext == ".json":
                with open(fpath, "r", encoding="utf-8") as f:
                    files[name] = json.load(f)
            elif ext in (".md", ".txt", ".html"):
                size = os.path.getsize(fpath)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    files[name] = f.read(_MAX_TEXT_BYTES)
                if size > _MAX_TEXT_BYTES:
                    files[name] += f"\n\n... [truncated, {size} bytes total]"
            else:
                files[name] = {"_skipped": True, "size_bytes": os.path.getsize(fpath)}
        except Exception as e:
            files[name] = {"_error": str(e)}

    return {
        "run_id": run_id,
        "path": run_path,
        "modified_at": datetime.fromtimestamp(
            os.path.getmtime(run_path), tz=timezone.utc
        ).isoformat(),
        "files": files,
    }


def _safe_join(base_dir: str, *parts: str) -> str:
    """Join paths and ensure result stays under base_dir (prevents path traversal)."""
    base_abs = os.path.abspath(base_dir)
    full = os.path.abspath(os.path.join(base_abs, *parts))
    if os.path.commonpath([base_abs, full]) != base_abs:
        raise HTTPException(status_code=400, detail="Invalid path")
    return full


_PIPELINE_DIRS = {
    "eval": EVAL_REPORTS_DIR,
    "agentic": AGENTIC_REPORTS_DIR,
}


@app.get("/latest-results", tags=["Latest Results"])
async def latest_results():
    """
    Return the most recent run from BOTH pipelines.

    Response:
      {
        "eval":    { run_id, modified_at, files: {...} } | null,
        "agentic": { run_id, modified_at, files: {...} } | null
      }
    """
    out: Dict[str, Any] = {}
    for key, base in _PIPELINE_DIRS.items():
        run = _latest_run_folder(base)
        out[key] = _load_run(base, run) if run else None
    return out


@app.get("/latest-results/list", tags=["Latest Results"])
async def list_runs(limit: int = 50):
    """List all run folders from both pipelines, newest first."""
    out: Dict[str, Any] = {}
    for key, base in _PIPELINE_DIRS.items():
        subs = _safe_listdir(base)
        subs.sort(key=lambda n: os.path.getmtime(os.path.join(base, n)), reverse=True)
        out[key] = {
            "base_dir": base,
            "count": len(subs),
            "runs": [
                {
                    "run_id": n,
                    "modified_at": datetime.fromtimestamp(
                        os.path.getmtime(os.path.join(base, n)), tz=timezone.utc
                    ).isoformat(),
                }
                for n in subs[:limit]
            ],
        }
    return out


@app.get("/latest-results/{pipeline}", tags=["Latest Results"])
async def latest_for_pipeline(pipeline: str):
    """Latest run for a single pipeline: 'eval' or 'agentic'."""
    if pipeline not in _PIPELINE_DIRS:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline '{pipeline}'")
    base = _PIPELINE_DIRS[pipeline]
    run = _latest_run_folder(base)
    if not run:
        raise HTTPException(status_code=404, detail=f"No runs found in {base}")
    return _load_run(base, run)


@app.get("/latest-results/{pipeline}/{run_id}", tags=["Latest Results"])
async def get_run(pipeline: str, run_id: str):
    """Fetch a specific run by id from the given pipeline."""
    if pipeline not in _PIPELINE_DIRS:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline '{pipeline}'")
    base = _PIPELINE_DIRS[pipeline]
    # Guard against traversal
    _safe_join(base, run_id)
    try:
        return _load_run(base, run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


@app.get("/latest-results/{pipeline}/{run_id}/file", tags=["Latest Results"])
async def get_run_file(pipeline: str, run_id: str, name: str):
    """Fetch a single raw file from a run (use ?name=report.json)."""
    if pipeline not in _PIPELINE_DIRS:
        raise HTTPException(status_code=404, detail=f"Unknown pipeline '{pipeline}'")
    base = _PIPELINE_DIRS[pipeline]
    fpath = _safe_join(base, run_id, name)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"File '{name}' not found")

    from fastapi.responses import FileResponse
    media_map = {
        ".json": "application/json",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".html": "text/html",
    }
    media = media_map.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
    return FileResponse(fpath, media_type=media, filename=name)


# ── Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "dashboard.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
