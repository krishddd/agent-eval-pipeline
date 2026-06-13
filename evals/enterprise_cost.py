"""
evals/enterprise_cost.py
Category 6 — Enterprise / Cost (§4.6)
Based on CLEAR 2025 framework.

Metrics:
- Cost-Normalized Accuracy (CNA):  accuracy / cost_per_task_USD   ≥ baseline × 0.90
- Token Efficiency Ratio (TER):    accuracy / tokens × 10,000     ≥ baseline × 0.90
- Budget compliance:               all runs within max_cost_usd   = 1.0
- Lab-to-production gap:           benchmark − live_prod_rate     Δ ≤ 15%
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult


class EnterpriseCostEvaluator(BaseEvaluator):
    """Ties agent performance to business economics."""

    category = "enterprise_cost"

    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:

        all_records = self._all_records(all_runs)
        if not all_records:
            return None

        warnings: List[str] = []

        # ── CNA: Cost-Normalized Accuracy ────────────────────────
        # accuracy / cost_per_task_USD (CLEAR 2025)
        accuracy = sum(1 for r in all_records if r.success) / len(all_records)
        total_cost = sum(r.total_cost_usd for r in all_records)
        avg_cost_per_task = total_cost / len(all_records) if all_records else 0.001

        cna = accuracy / max(avg_cost_per_task, 0.001)  # Avoid division by zero

        # ── TER: Token Efficiency Ratio ──────────────────────────
        # accuracy / (in + out + inter_agent_tokens) × 10,000
        total_tokens = sum(
            r.input_tokens + r.output_tokens + r.inter_agent_tokens
            for r in all_records
        )
        avg_tokens_per_task = total_tokens / len(all_records) if all_records else 1

        ter = (accuracy / max(avg_tokens_per_task, 1)) * 10_000

        # ── Budget Compliance ────────────────────────────────────
        # 1.0 iff all k runs within max_cost_usd
        over_budget = [
            r for r in all_records
            if r.total_cost_usd > card.max_cost_usd
        ]
        budget_compliance = 1.0 if len(over_budget) == 0 else 0.0

        if over_budget:
            warnings.append(
                f"{len(over_budget)} run(s) exceeded budget of ${card.max_cost_usd:.2f}: "
                f"max was ${max(r.total_cost_usd for r in over_budget):.4f}"
            )

        # ── Lab-to-Production Gap ────────────────────────────────
        # benchmark_score − live_production_success_rate (needs prod data)
        l2p_gap = None  # Requires external production baseline
        l2p_gap_details = "Requires production_baselines table data"

        # ── Per-run cost breakdown ───────────────────────────────
        cost_distribution = {
            "total_cost_usd": round(total_cost, 4),
            "avg_cost_per_task": round(avg_cost_per_task, 6),
            "max_cost_single_run": round(max(r.total_cost_usd for r in all_records), 6),
            "min_cost_single_run": round(min(r.total_cost_usd for r in all_records), 6),
            "total_tokens": total_tokens,
            "avg_tokens_per_run": int(avg_tokens_per_task),
        }

        # ── Pass/Fail ────────────────────────────────────────────
        passed = budget_compliance == 1.0  # Hard gate

        if cna < 1.0:
            warnings.append(f"CNA {cna:.2f} — check if cost is proportional to accuracy")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                "cost_normalized_accuracy": round(cna, 4),
                "token_efficiency_ratio": round(ter, 4),
                "budget_compliance": budget_compliance,
                "lab_to_production_gap": l2p_gap,
            },
            details={
                "cost_distribution": cost_distribution,
                "l2p_gap_note": l2p_gap_details,
                "accuracy": round(accuracy, 4),
            },
            warnings=warnings,
        )
