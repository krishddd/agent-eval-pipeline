"""
evals/trajectory.py
Category 3 — Trajectory (§4.3)

Metrics:
- Exact match:            pred_traj == golden_traj                  regression gate
- In-order match:         golden tools in order; extras allowed     ≥ 0.75
- Any-order match:        all golden tools present                  ≥ 0.80
- Silent failure:         correct_output AND NOT valid_provenance   = 0
- Planning score:         LLM judge 1-5                             ≥ 3.5/5
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult, LLMJudge


class TrajectoryEvaluator(BaseEvaluator):
    """Evaluates execution path quality, not just output correctness."""

    category = "trajectory"

    def __init__(self):
        self.judge = LLMJudge(n_judges=3, reliability_threshold=0.7)

    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:

        all_records = self._all_records(all_runs)
        if not all_records:
            return None

        golden = card.golden_trajectory or []
        metrics: Dict[str, Optional[float]] = {}
        warnings: List[str] = []
        details: Dict[str, Any] = {}

        # ── Exact Match ──────────────────────────────────────────
        if golden:
            exact_scores = []
            for r in all_records:
                pred = [tc.tool_name for tc in r.tool_calls]
                exact_scores.append(1.0 if pred == golden else 0.0)
            metrics["exact_match"] = round(statistics.mean(exact_scores), 4)

        # ── In-Order Match ───────────────────────────────────────
        if golden:
            in_order_scores = []
            for r in all_records:
                pred = [tc.tool_name for tc in r.tool_calls]
                score = self._in_order_match(golden, pred)
                in_order_scores.append(score)
            metrics["in_order_match"] = round(statistics.mean(in_order_scores), 4)
            if metrics["in_order_match"] < 0.75:
                warnings.append(f"In-order match {metrics['in_order_match']:.3f} below 0.75")

        # ── Any-Order Match ──────────────────────────────────────
        if golden:
            any_order_scores = []
            for r in all_records:
                pred = set(tc.tool_name for tc in r.tool_calls)
                golden_set = set(golden)
                overlap = len(pred & golden_set) / len(golden_set) if golden_set else 1.0
                any_order_scores.append(overlap)
            metrics["any_order_match"] = round(statistics.mean(any_order_scores), 4)
            if metrics["any_order_match"] < 0.80:
                warnings.append(f"Any-order match {metrics['any_order_match']:.3f} below 0.80")

        # ── Silent Failure Detection ─────────────────────────────
        # Uses provenance comparator from TracingWrapper
        silent_failures = sum(
            1 for r in all_records
            if r.success and getattr(r, "silent_failure_detected", False)
        )
        metrics["silent_failure_count"] = float(silent_failures)
        details["silent_failure_details"] = []

        for r in all_records:
            if r.success and getattr(r, "silent_failure_detected", False):
                details["silent_failure_details"].append({
                    "run_id": r.run_id,
                    "task": r.task[:100],
                    "data_sources": r.data_sources,
                    "provenance_valid": r.provenance_valid,
                })

        if silent_failures > 0:
            warnings.append(
                f"CRITICAL: {silent_failures} silent failure(s) detected — "
                f"correct output from flawed execution path"
            )

        # ── Planning Score (LLM Judge) ───────────────────────────
        planning_scores = []
        # Bug fix: collect ALL kappas and average — overwriting each iteration
        # kept only the last kappa (same bug fixed in multi_agent.py)
        judge_kappas: List[float] = []

        for r in all_records[:5]:  # Cap at 5 runs for cost
            trajectory_text = " → ".join(tc.tool_name for tc in r.tool_calls)
            content = (
                f"Task: {r.task}\n"
                f"Trajectory: {trajectory_text}\n"
                f"Output: {r.final_output[:300]}\n"
                f"Tools available: {[t.name for t in card.tools_manifest]}"
            )

            rubric = self.judge.load_rubric("planning_rubric")

            score, kappa, reliable = await self.judge.judge(
                prompt=r.task,
                rubric=rubric,
                content=content,
                scale=(1, 5),
            )
            if score is not None:
                planning_scores.append(score)
            judge_kappas.append(kappa)

        planning_score = statistics.mean(planning_scores) if planning_scores else None
        if planning_score is not None:
            metrics["planning_score"] = round(planning_score, 4)
            if planning_score < 3.5:
                warnings.append(f"Planning score {planning_score:.2f} below 3.5/5")

        avg_kappa = statistics.mean(judge_kappas) if judge_kappas else None

        # ── Pass/Fail ────────────────────────────────────────────
        # Block on: silent failures > 0  (hard gate)
        # Block on: exact match regression (if golden exists)
        passed = silent_failures == 0

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics=metrics,
            details=details,
            warnings=warnings,
            judge_reliability=avg_kappa,
        )

    @staticmethod
    def _in_order_match(golden: List[str], predicted: List[str]) -> float:
        """Check if golden tools appear in order within predicted (extras allowed)."""
        if not golden:
            return 1.0

        matched = 0
        pred_idx = 0

        for g_tool in golden:
            while pred_idx < len(predicted):
                if predicted[pred_idx] == g_tool:
                    matched += 1
                    pred_idx += 1
                    break
                pred_idx += 1

        return matched / len(golden)
