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

    report = EvalReport(**data)
    logger = MLflowLogger()
    run_id = await logger.log(report)

    if run_id:
        print(f"✓ Logged to MLflow: run_id={run_id}")
    else:
        print("⚠ MLflow logging failed — check configuration")


if __name__ == "__main__":
    asyncio.run(main())
