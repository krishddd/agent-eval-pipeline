"""
storage/mlflow_logger.py
MLflow experiment/run logging with Git SHA linking.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class MLflowLogger:
    """Logs eval reports to MLflow for experiment versioning and comparison."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "agent-eval-pipeline",
    ):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "")
        self.experiment_name = experiment_name
        self._initialized = False

    def _init_mlflow(self):
        """Lazy initialization of MLflow."""
        if self._initialized:
            return True
        try:
            import mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._initialized = True
            return True
        except ImportError:
            print("[MLflow] mlflow not installed — logging disabled")
            return False
        except Exception as e:
            print(f"[MLflow] Init error: {e} — logging disabled")
            return False

    async def log(self, report) -> Optional[str]:
        """
        Log an EvalReport as an MLflow run.

        Returns the MLflow run_id or None if logging failed.
        """
        if not self._init_mlflow():
            return None

        try:
            import mlflow

            with mlflow.start_run(run_name=f"{report.agent_name}_v{report.agent_version}"):
                # ── Tags ─────────────────────────────────────────
                mlflow.set_tag("agent_id", report.agent_id)
                mlflow.set_tag("agent_name", report.agent_name)
                mlflow.set_tag("agent_version", report.agent_version)
                mlflow.set_tag("trigger", report.trigger)
                mlflow.set_tag("overall_passed", str(report.overall_passed))

                if report.git_sha:
                    mlflow.set_tag("git_sha", report.git_sha)
                    mlflow.set_tag("mlflow.source.git.commit", report.git_sha)

                # ── Metrics ──────────────────────────────────────
                mlflow.log_metric("overall_passed", 1.0 if report.overall_passed else 0.0)
                mlflow.log_metric("duration_ms", report.duration_ms)
                mlflow.log_metric("tasks_evaluated", report.tasks_evaluated)
                mlflow.log_metric("total_runs", report.total_runs)

                for result in report.results:
                    prefix = result.category
                    mlflow.log_metric(f"{prefix}.passed", 1.0 if result.passed else 0.0)

                    for metric_name, value in result.metrics.items():
                        if value is not None:
                            mlflow.log_metric(f"{prefix}.{metric_name}", float(value))

                # ── Artifacts ────────────────────────────────────
                import json
                import tempfile

                report_json = report.model_dump(mode="json")
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False
                ) as f:
                    json.dump(report_json, f, indent=2, default=str)
                    mlflow.log_artifact(f.name, "eval_reports")

                run_id = mlflow.active_run().info.run_id
                print(f"[MLflow] Logged run: {run_id}")
                return run_id

        except Exception as e:
            print(f"[MLflow] Logging error: {e}")
            return None
