"""
scripts/log_to_mlflow.py
Export eval results JSON to MLflow.
Usage: python scripts/log_to_mlflow.py eval_results.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.eval_harness import EvalReport
from storage.mlflow_logger import MLflowLogger


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/log_to_mlflow.py <eval_results.json>")
        sys.exit(1)

    results_path = sys.argv[1]
    if not os.path.exists(results_path):
        print(f"No eval results at {results_path} — nothing to log to MLflow")
        sys.exit(0)

    with open(results_path, "r") as f:
        data = json.load(f)

    # eval_results.json is a combined wrapper: {"overall_passed": ..., "reports": [...]}
    # Each element in "reports" is a serialised EvalReport.  Fall back to treating
    # the whole dict as a single report for backwards-compat (single-agent case where
    # harness/run.py spreads **all_reports[0] into the top-level dict).
    reports_data = data.get("reports")
    if not reports_data:
        # Older format or single-agent spread — try top-level dict directly.
        reports_data = [data] if "agent_id" in data else []

    if not reports_data:
        print("No individual EvalReport entries found in results — nothing to log")
        sys.exit(0)

    logger = MLflowLogger()
    logged = 0
    for report_dict in reports_data:
        try:
            report = EvalReport(**report_dict)
            run_id = await logger.log(report)
            if run_id:
                print(f"✓ Logged {report.agent_name} to MLflow: run_id={run_id}")
                logged += 1
            else:
                print(f"⚠ MLflow logging skipped for {report_dict.get('agent_name', '?')} — check configuration")
        except Exception as e:
            print(f"⚠ Could not log report: {e}")

    print(f"MLflow: {logged}/{len(reports_data)} reports logged")


if __name__ == "__main__":
    asyncio.run(main())
