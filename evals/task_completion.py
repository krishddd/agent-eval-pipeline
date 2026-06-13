"""
evals/task_completion.py
Category 1 — Task Completion (§4.1)

Metrics:
- Goal success rate (SR): |successful| / |total|               ≥ 0.80
- Pass@k consistency: P(success in all k runs)                  ≥ 0.80 at k=8
- Milestone KPI: Σ(w_i × achieved_i) / Σ(w_i)                  ≥ 0.75
- Step-wise progress: steps_completed / total × avg_quality     ≥ 0.60
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from math import comb
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult


class TaskCompletionEvaluator(BaseEvaluator):
    """Foundation evaluator — applies to every agent type."""

    category = "task_completion"

    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:

        all_records = self._all_records(all_runs)
        if not all_records:
            return None

        # ── Goal Success Rate (SR) ───────────────────────────────
        total = len(all_records)
        successes = sum(1 for r in all_records if r.success)
        sr = successes / total if total > 0 else 0.0

        # ── Pass@k Consistency ───────────────────────────────────
        # Standard unbiased estimator: 1 − C(n-c, k) / C(n, k)
        pass_at_k_scores = []
        for task, runs in all_runs.items():
            n = len(runs)
            if n == 0:
                continue
            c = sum(1 for r in runs if r.success)
            k = min(card.pass_k, n)  # Can't draw more than n samples
            if n - c < k:
                # More correct samples than needed → pass@k = 1.0
                pass_at_k_scores.append(1.0)
            else:
                denom = comb(n, k)
                if denom == 0:
                    pass_at_k_scores.append(0.0)
                else:
                    pass_at_k_scores.append(1.0 - comb(n - c, k) / denom)

        pass_at_k = (
            statistics.mean(pass_at_k_scores)
            if pass_at_k_scores else 0.0
        )

        # ── Milestone KPI ────────────────────────────────────────
        # Σ(w_i × achieved_i) / Σ(w_i) — weighted partial credit
        # FIX #6: None when not configured, not 0.0
        milestone_kpi = None
        golden_milestones = card.golden_milestones or []
        if golden_milestones:
            per_run_scores = []
            for r in all_records:
                hit = sum(1 for m in golden_milestones if m in r.milestones_hit)
                per_run_scores.append(hit / len(golden_milestones))
            milestone_kpi = statistics.mean(per_run_scores) if per_run_scores else 0.0

        # ── Step-wise Progress ───────────────────────────────────
        # steps_completed / total_steps × avg_step_quality
        step_progress_scores = []
        for r in all_records:
            n_tools = len(r.tool_calls)
            if n_tools == 0:
                step_progress_scores.append(0.0 if not r.success else 1.0)
                continue
            steps_ok = sum(1 for tc in r.tool_calls if tc.success)
            step_quality = steps_ok / n_tools
            step_progress_scores.append(step_quality)

        step_progress = (
            statistics.mean(step_progress_scores)
            if step_progress_scores else 0.0
        )

        # ── Pass/Fail ────────────────────────────────────────────
        passed = pass_at_k >= 0.80  # Primary gate: pass@k

        warnings = []
        if sr < 0.80:
            warnings.append(f"Goal SR {sr:.3f} below 0.80 threshold")
        # Bug fix: `> 0` silently suppressed warnings when kpi == 0.0 (all milestones missed)
        if milestone_kpi is not None and milestone_kpi < 0.75:
            warnings.append(f"Milestone KPI {milestone_kpi:.3f} below 0.75 threshold")
        if step_progress < 0.60:
            warnings.append(f"Step-wise progress {step_progress:.3f} below 0.60 threshold")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                "goal_success_rate": round(sr, 4),
                "pass_at_k": round(pass_at_k, 4),
                "milestone_kpi": round(milestone_kpi, 4) if milestone_kpi is not None else None,
                "step_wise_progress": round(step_progress, 4),
            },
            details={
                "k": card.pass_k,
                "total_runs": total,
                "successes": successes,
                "tasks_evaluated": len(all_runs),
            },
            warnings=warnings,
        )
