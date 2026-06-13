"""
scripts/run_odysseus_eval.py
One-shot driver: register Odysseus -> trigger eval -> poll -> print scorecard.

Requires the dashboard server to already be running (see module docstring of
dashboard/api.py). Reads .env for OPENAI_API_KEY / ODYSSEUS_TOKEN.

Usage:
  # Terminal 1 — start the server (once):
  python -m dashboard.api
  # Terminal 2 — run the eval:
  python scripts/run_odysseus_eval.py
  python scripts/run_odysseus_eval.py --dashboard http://localhost:8000 --mode auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evals._env import load_env

load_env()

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.register_odysseus_agent import build_request  # reuse the same card


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Odysseus eval end-to-end")
    ap.add_argument("--dashboard", default="http://localhost:8000")
    ap.add_argument("--base-url", default="http://127.0.0.1:7000")
    ap.add_argument("--mode", default="chat", choices=["auto", "agent", "chat"])
    ap.add_argument("--token", default=os.getenv("ODYSSEUS_TOKEN", ""))
    ap.add_argument("--suite", default=os.path.join(
        os.path.dirname(__file__), "..", "tasks", "odysseus_suite.json"))
    ap.add_argument("--poll-timeout", type=int, default=1800, help="seconds to wait for completion")
    args = ap.parse_args()

    import httpx

    if not args.token:
        print("WARN: ODYSSEUS_TOKEN not set — auth'd endpoints will 401.")

    with httpx.Client(timeout=60) as client:
        # 0) server reachable?
        try:
            client.get(f"{args.dashboard}/health")
        except Exception as e:
            print(f"ERROR: dashboard not reachable at {args.dashboard} — start it with "
                  f"`python -m dashboard.api`. ({e})")
            return 1

        # 1) register
        body = build_request(args.base_url, args.token, args.mode)
        r = client.post(f"{args.dashboard}/agents/register", json=body)
        r.raise_for_status()
        agent_id = r.json()["agent_id"]
        print(f"[register] agent_id={agent_id}  categories={r.json().get('eval_categories')}")

        # 2) load suite tasks
        with open(args.suite, "r", encoding="utf-8") as f:
            tasks = json.load(f).get("tasks", [])
        print(f"[suite] {len(tasks)} tasks from {os.path.basename(args.suite)}")

        # 3) trigger eval
        r = client.post(f"{args.dashboard}/evals/run", json={
            "agent_id": agent_id, "task_suite": tasks,
            "trigger": "manual", "fail_on_threshold": False,
        })
        r.raise_for_status()
        job_id = r.json()["job_id"]
        print(f"[run] job_id={job_id}  (a live-log viewer opens; eval starts within 30s)")

        # 4) poll — resilient to transient connection resets (uvicorn --reload,
        #    socket hiccups). A failed poll is retried, not fatal.
        deadline = time.time() + args.poll_timeout
        status, report = "pending", None
        while time.time() < deadline:
            time.sleep(5)
            try:
                s = client.get(f"{args.dashboard}/evals/status/{job_id}", timeout=15).json()
            except Exception as e:
                print(f"[poll] transient error, retrying: {type(e).__name__}")
                continue
            if s["status"] != status:
                status = s["status"]
                print(f"[status] {status}")
            if status in ("completed", "failed"):
                report = s.get("report")
                break

    # 5) print scorecard
    if not report:
        print("Timed out or no report. Check the server logs / the live-log viewer.")
        return 1

    print("\n" + "=" * 60)
    print(f"EVAL REPORT — overall {'PASS' if report.get('overall_passed') else 'FAIL'}")
    print("=" * 60)
    for res in report.get("results", []):
        if res.get("category") != "odysseus_metrics":
            print(f"[{'PASS' if res.get('passed') else 'FAIL'}] {res.get('category')}")
            continue
        print(f"[{'PASS' if res.get('passed') else 'FAIL'}] odysseus_metrics (M01-M33):")
        for k, v in sorted((res.get("metrics") or {}).items()):
            print(f"    {k:38} {v}")
        det = res.get("details", {}).get("odysseus_metrics", {})
        jm = det.get("_judge_methods", {})
        print(f"    judge backend: {jm.get('_status')}")
        cf = res.get("details", {}).get("critical_failures")
        if cf:
            print(f"    CRITICAL: {cf}")
    print("\nFull artifacts under: reports/evals/<run_folder>/ "
          "(report.json, scorecard.json, odysseus_metrics.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
