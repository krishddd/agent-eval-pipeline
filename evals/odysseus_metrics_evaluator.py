"""
evals/odysseus_metrics_evaluator.py
OdysseusMetricsEvaluator — computes the 33 Odysseus agent metrics (M01-M33).

Integrates with EvalHarness via the standard BaseEvaluator interface.  It
normalises the captured TrajectoryRecords (across all tasks and k runs) into
the list-of-run dicts that odysseus_metrics.compute_all() expects, then stores:

  1. flat scalar metrics in EvalResult.metrics  (scorecard)
  2. the full nested report in EvalResult.details["odysseus_metrics"]

Category name: "odysseus_metrics"

Per-task expectations (expect_refusal, forbidden_tools, expected_tools,
expected_artifacts, golden_milestones, max_steps, mode) are read from the task
suite's optional "task_meta" map, matched by exact task string.  Override the
suite path with the ODYSSEUS_TASK_SUITE env var.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult
from evals.odysseus_metrics import EWMAMonitor, compute_all
from evals.odysseus_metrics_config import CRITICAL_METRICS, TOOL_CATEGORIES


_FLAT_KEYS = [
    "m01_goal_completion_rate", "m02_step_success_ratio", "m03_calibration_gap",
    "m04_autonomy_efficiency", "m05_tool_exec_success_rate", "m06_tool_selection_accuracy",
    "m07_parameter_f1_score", "m08_tool_hallucination_rate", "m09_shell_success_rate",
    "m10_command_recovery_rate", "m11_script_correctness", "m12_command_efficiency",
    "m13_file_op_success_rate", "m14_artifact_correctness", "m15_workspace_footprint",
    "m16_redundant_write_rate", "m17_web_fetch_success_rate", "m18_source_credibility_score",
    "m19_grounding_rate", "m20_mcp_selection_accuracy", "m21_mcp_invocation_success",
    "m22_mcp_tool_coverage", "m23_context_retention_score", "m24_memory_fidelity",
    "m25_cross_session_continuity", "m26_answer_faithfulness", "m27_evidence_traceability_score",
    "m28_refusal_fallback_quality", "m29_policy_adherence_score", "m30_sla_latency_compliance",
    "m31_budget_compliance", "m32_anomalies_detected", "m33_run_consistency_score",
]


class OdysseusMetricsEvaluator(BaseEvaluator):
    """Evaluates the 33 Odysseus quality metrics from captured trajectories."""

    category = "odysseus_metrics"

    def __init__(self):
        self._ewma = EWMAMonitor()
        self._meta_map = self._load_task_meta()

    # ── Task metadata ────────────────────────────────────────────────────
    @staticmethod
    def _load_task_meta() -> Dict[str, Dict]:
        path = os.getenv("ODYSSEUS_TASK_SUITE") or os.path.join(
            os.path.dirname(__file__), "..", "tasks", "odysseus_suite.json"
        )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("task_meta", {}) or {}
        except Exception:
            return {}

    # ── Normalisation ────────────────────────────────────────────────────
    def _record_to_run(self, task: str, record: Any, card: Any) -> Dict:
        meta = self._meta_map.get(task, {})

        tool_calls = []
        for tc in getattr(record, "tool_calls", []) or []:
            name = getattr(tc, "tool_name", None) or "unknown"
            params = getattr(tc, "parameters", {}) or {}
            result = getattr(tc, "result", "")
            # Heuristic exit code for shell calls: parse "exit code N" from result
            exit_code = None
            rtext = str(result or "")
            import re as _re
            m = _re.search(r"exit(?:\s+code|status)?[:\s]+(\d+)", rtext, _re.I)
            if m:
                exit_code = int(m.group(1))
            tool_calls.append({
                "name": name,
                "category": TOOL_CATEGORIES.get(name, "other"),
                "parameters": params,
                "result": rtext,
                "success": bool(getattr(tc, "success", True)),
                "error": getattr(tc, "error", None),
                "latency_ms": float(getattr(tc, "latency_ms", 0.0) or 0.0),
                "exit_code": exit_code,
            })

        mode = meta.get("mode") or ("agent" if tool_calls else "chat")

        return {
            "task": task,
            "task_meta": meta,
            "mode": mode,
            "final_output": getattr(record, "final_output", "") or "",
            "success": bool(getattr(record, "success", False)),
            "tool_calls": tool_calls,
            "milestones": list(getattr(record, "milestones_hit", []) or []),
            "retrieved_chunks": list(getattr(record, "retrieved_chunks", []) or []),
            "wall_latency_ms": float(getattr(record, "wall_latency_ms", 0.0) or 0.0),
            "input_tokens": int(getattr(record, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(record, "output_tokens", 0) or 0),
            "cost_usd": float(getattr(record, "total_cost_usd", 0.0) or 0.0),
            "model": getattr(card, "model_backbone", "") or "unknown",
            "sla_latency_ms": getattr(card, "sla_latency_ms", None),
            "max_cost_usd": getattr(card, "max_cost_usd", None),
        }

    # ── Evaluate ─────────────────────────────────────────────────────────
    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:
        runs: List[Dict] = []
        for task, records in all_runs.items():
            for record in records:
                if record is None:
                    continue
                runs.append(self._record_to_run(task, record, card))

        if not runs:
            return None

        full = compute_all(runs, self._ewma)

        flat: Dict[str, Optional[float]] = {}
        for k in _FLAT_KEYS:
            v = full.get(k)
            flat[k] = v if isinstance(v, (int, float)) else None

        # ── Pass/fail on critical metrics ───────────────────────────────
        critical_failures = []
        for key, (op, thr) in CRITICAL_METRICS.items():
            v = full.get(key)
            if not isinstance(v, (int, float)):
                continue
            if (op == "<" and v < thr) or (op == ">" and v > thr):
                critical_failures.append(f"{key}={v} {op} {thr}")

        warnings = full.get("_meta", {}).get("warnings", [])
        return EvalResult(
            category=self.category,
            passed=len(critical_failures) == 0,
            metrics=flat,
            details={
                "odysseus_metrics": full,
                "critical_failures": critical_failures,
                "runs_evaluated": len(runs),
                "modes": full.get("_meta", {}).get("modes", []),
            },
            warnings=warnings[:20],
        )
