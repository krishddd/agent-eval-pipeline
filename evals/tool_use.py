"""
evals/tool_use.py
Category 2 — Tool Use (§4.2)

Metrics:
- Invocation accuracy:      correct_decisions / total_decision_points    ≥ 0.90
- Tool selection accuracy:  correct_selections / total_selections        ≥ 0.85
- Parameter F1 score:       2×(P×R)/(P+R)                               ≥ 0.80
- Redundant tool usage:     unnecessary / total × 100%                   ≤ 20%
- Tool retrieval MRR:       (1/|Q|) × Σ 1/rank_i                        ≥ 0.85
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional, Set

from evals.base_evaluator import BaseEvaluator, EvalResult


class ToolUseEvaluator(BaseEvaluator):
    """Evaluates tool invocation quality for agents with tools_manifest."""

    category = "tool_use"

    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:

        if not card.tools_manifest:
            return None  # No tools → skip

        all_records = self._all_records(all_runs)
        if not all_records:
            return None

        golden = card.golden_trajectory or []
        tool_names = {t.name for t in card.tools_manifest}

        # ── Invocation Accuracy ──────────────────────────────────
        # correct_invocation_decisions / total_decision_points
        total_decisions = 0
        correct_decisions = 0
        for r in all_records:
            for tc in r.tool_calls:
                total_decisions += 1
                if tc.tool_name in tool_names and tc.success:
                    correct_decisions += 1

        invocation_acc = (
            correct_decisions / total_decisions
            if total_decisions > 0 else 1.0
        )

        # ── Tool Selection Accuracy ──────────────────────────────
        # correct_tool_selections / total_selections (vs golden)
        selection_acc = 1.0
        if golden:
            # FIX #16: O(1) set lookup instead of O(n) list membership
            golden_set = set(golden)
            selection_scores = []
            for r in all_records:
                predicted = [tc.tool_name for tc in r.tool_calls]
                correct = sum(
                    1 for p in predicted if p in golden_set
                )
                selection_scores.append(
                    correct / len(predicted) if predicted else 0.0
                )
            selection_acc = statistics.mean(selection_scores) if selection_scores else 0.0

        # ── Parameter F1 Score ───────────────────────────────────
        # 2×(P×R)/(P+R) over parameter name+value slots
        param_f1 = self._compute_param_f1(all_records, card.tools_manifest)

        # ── Redundant Tool Usage Rate ────────────────────────────
        # unnecessary_calls / total_calls × 100%
        redundancy_scores = []
        for r in all_records:
            if not r.tool_calls:
                continue
            seen = set()
            redundant = 0
            for tc in r.tool_calls:
                key = (tc.tool_name, str(sorted(tc.parameters.items())))
                if key in seen:
                    redundant += 1
                seen.add(key)
            redundancy_scores.append(redundant / len(r.tool_calls))
        redundancy_rate = (
            statistics.mean(redundancy_scores)
            if redundancy_scores else 0.0
        )

        # ── Tool Retrieval MRR ───────────────────────────────────
        # (1/|Q|) × Σ 1/rank_i
        mrr = self._compute_mrr(all_records, golden) if golden else None

        # ── Pass/Fail ────────────────────────────────────────────
        passed = invocation_acc >= 0.90 and selection_acc >= 0.85

        warnings = []
        if param_f1 < 0.80:
            warnings.append(f"Parameter F1 {param_f1:.3f} below 0.80")
        if redundancy_rate > 0.20:
            warnings.append(f"Redundancy rate {redundancy_rate:.1%} exceeds 20%")
        if mrr is not None and mrr < 0.85:
            warnings.append(f"Tool MRR {mrr:.3f} below 0.85")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                "invocation_accuracy": round(invocation_acc, 4),
                "tool_selection_accuracy": round(selection_acc, 4),
                "parameter_f1": round(param_f1, 4),
                "redundant_usage_rate": round(redundancy_rate, 4),
                "tool_retrieval_mrr": round(mrr, 4) if mrr is not None else None,
            },
            warnings=warnings,
        )

    @staticmethod
    def _compute_param_f1(records: list, tools_manifest: list) -> float:
        """Compute F1 over parameter name+value slots."""
        expected_params: Dict[str, Set[str]] = {}
        for tool in tools_manifest:
            expected_params[tool.name] = set(tool.parameters.keys())

        precisions = []
        recalls = []

        for r in records:
            for tc in r.tool_calls:
                expected = expected_params.get(tc.tool_name, set())
                predicted = set(tc.parameters.keys())

                if not expected and not predicted:
                    continue

                tp = len(expected & predicted)
                precision = tp / len(predicted) if predicted else 0.0
                recall = tp / len(expected) if expected else 0.0

                precisions.append(precision)
                recalls.append(recall)

        if not precisions:
            return 1.0

        avg_p = statistics.mean(precisions)
        avg_r = statistics.mean(recalls)

        if avg_p + avg_r == 0:
            return 0.0

        return 2 * (avg_p * avg_r) / (avg_p + avg_r)

    @staticmethod
    def _compute_mrr(records: list, golden: list) -> float:
        """Mean Reciprocal Rank for tool retrieval (per-run, not per-golden-tool)."""
        rr_scores = []
        golden_set = set(golden)
        for r in records:
            predicted = [tc.tool_name for tc in r.tool_calls]
            # Find rank of the first golden tool in this run's predictions
            rr = 0.0
            for rank_idx, tool in enumerate(predicted, start=1):
                if tool in golden_set:
                    rr = 1.0 / rank_idx
                    break
            rr_scores.append(rr)

        return statistics.mean(rr_scores) if rr_scores else 0.0
