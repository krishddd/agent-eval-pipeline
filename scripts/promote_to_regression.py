"""
scripts/promote_to_regression.py
Adaptive Feedback Loop — Regression Promotion Logic (§8.1)

Monitors the metric_drift view and promotes failing tasks
to the regression suite automatically.

Run hourly via APScheduler (integrated in dashboard/api.py)
or standalone via: python scripts/promote_to_regression.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.report_store import ReportStore


REGRESSION_SUITE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tasks",
    "regression_suite.json",
)

DRIFT_THRESHOLD = 0.10  # |drift| > 10%


async def promote_drifting_metrics():
    """
    Detection → Promotion → Update cycle:

    1. Query metric_drift view for |drift| > 10%
    2. Extract the task and golden_trajectory from the failed run
    3. Append to tasks/regression_suite.json
    4. Next CI/CD run includes the regression case → loop closed
    """
    store = ReportStore()

    try:
        alerts = await store.get_drift_alerts(threshold=DRIFT_THRESHOLD)

        if not alerts:
            print("[Promote] No metric drift detected. All clear.")
            return

        # Load existing regression suite
        if os.path.exists(REGRESSION_SUITE_PATH):
            with open(REGRESSION_SUITE_PATH, "r") as f:
                suite = json.load(f)
        else:
            suite = {"tasks": [], "promoted_from_drift": []}

        existing_promotions = {
            (e["agent_id"], e["metric"])
            for e in suite.get("promoted_from_drift", [])
        }

        new_promotions = 0

        for alert in alerts:
            agent_id = str(alert.get("agent_id", ""))
            metric = alert.get("metric", "")
            drift = alert.get("drift", 0)
            recent_avg = alert.get("recent_avg", 0)
            baseline_avg = alert.get("baseline_avg", 0)

            # Skip already-promoted
            if (agent_id, metric) in existing_promotions:
                continue

            # Fetch the failing report to extract task context
            reports = await store.get_reports(agent_id=agent_id, limit=1)

            entry = {
                "agent_id": agent_id,
                "metric": metric,
                "drift_value": round(drift, 4) if drift else 0,
                "recent_avg": round(recent_avg, 4) if recent_avg else 0,
                "baseline_avg": round(baseline_avg, 4) if baseline_avg else 0,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "source_report": reports[0].get("report_id") if reports else None,
            }

            suite.setdefault("promoted_from_drift", []).append(entry)
            new_promotions += 1

            print(
                f"[Promote] DRIFT: agent={agent_id} "
                f"metric={metric} drift={drift:.4f} "
                f"(recent={recent_avg:.4f} baseline={baseline_avg:.4f})"
            )

        if new_promotions > 0:
            suite["metadata"] = suite.get("metadata", {})
            suite["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()

            with open(REGRESSION_SUITE_PATH, "w") as f:
                json.dump(suite, f, indent=2)

            print(f"\n✓ Promoted {new_promotions} new regression case(s)")
        else:
            print("[Promote] All drifting metrics already in regression suite")

    finally:
        await store.close()


if __name__ == "__main__":
    print("=" * 60)
    print("ADAPTIVE FEEDBACK LOOP — REGRESSION PROMOTION")
    print("=" * 60)
    asyncio.run(promote_drifting_metrics())
