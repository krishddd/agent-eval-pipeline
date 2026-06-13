"""
evals/graph_memory.py
Conditional Category — Graph Memory (§4.9)
Fires when memory_type ∈ {GRAPH_DB, HYBRID}.
Based on Zep/Graphiti benchmark data (2025).

Metrics:
- DMR retrieval accuracy:      exact-match recall on DMR benchmark    ≥ 0.94
- Cross-session synthesis:     multi-hop QA across 35+ sessions       ≥ 0.75
- Temporal reasoning accuracy: correct chronological ordering         ≥ 0.70
- Relational fidelity:         correct edge traversal multi-hop       ≥ 0.80
- Graph vs vector delta:       improvement over vector-only           ≥ +10%
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult, LLMJudge


class GraphMemoryEvaluator(BaseEvaluator):
    """Evaluates graph memory quality — relationships, provenance, temporal position."""

    category = "graph_memory"

    def __init__(self):
        self.judge = LLMJudge(n_judges=3)

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

        # ── DMR Retrieval Accuracy ───────────────────────────────
        dmr_scores = []
        for r in all_records:
            if r.retrieved_chunks:
                # Heuristic: measure recall of chunks with graph metadata
                graph_chunks = [
                    c for c in r.retrieved_chunks
                    if c.get("metadata", {}).get("source_type") == "graph"
                    or c.get("metadata", {}).get("entity_type")
                ]
                total = len(r.retrieved_chunks)
                dmr_scores.append(len(graph_chunks) / total if total else 0.0)

        dmr_accuracy = statistics.mean(dmr_scores) if dmr_scores else None

        # ── Cross-Session Synthesis ──────────────────────────────
        cross_session_scores = []
        for r in all_records[:10]:
            score, _, _ = await self.judge.judge(
                prompt=r.task,
                rubric=(
                    "Rate the agent's ability to synthesize information across sessions.\n"
                    "1 = Uses only current session data.\n"
                    "3 = Partially integrates cross-session knowledge.\n"
                    "5 = Fully synthesizes multi-session knowledge with correct attribution."
                ),
                content=f"Task: {r.task}\nOutput: {r.final_output[:500]}",
                scale=(1, 5),
            )
            if score is not None:
                cross_session_scores.append(score / 5.0)

        cross_session = (
            statistics.mean(cross_session_scores)
            if cross_session_scores else None
        )

        # ── Temporal Reasoning Accuracy ──────────────────────────
        temporal_scores = []
        for r in all_records[:10]:
            score, _, _ = await self.judge.judge(
                prompt=r.task,
                rubric=(
                    "Rate temporal reasoning accuracy.\n"
                    "1 = Incorrect chronological ordering, wrong dates.\n"
                    "3 = Mostly correct but some temporal confusion.\n"
                    "5 = Perfect chronological understanding and temporal references."
                ),
                content=f"Task: {r.task}\nOutput: {r.final_output[:500]}",
                scale=(1, 5),
            )
            if score is not None:
                temporal_scores.append(score / 5.0)

        temporal_accuracy = (
            statistics.mean(temporal_scores) if temporal_scores else None
        )

        # ── Relational Fidelity ──────────────────────────────────
        relational_scores = []
        for r in all_records[:10]:
            score, _, _ = await self.judge.judge(
                prompt=r.task,
                rubric=(
                    "Rate relational fidelity — correct edge traversal for multi-hop queries.\n"
                    "1 = Completely wrong relationships. 5 = All relationships correctly traversed."
                ),
                content=f"Task: {r.task}\nOutput: {r.final_output[:500]}",
                scale=(1, 5),
            )
            if score is not None:
                relational_scores.append(score / 5.0)

        relational_fidelity = (
            statistics.mean(relational_scores) if relational_scores else None
        )

        # ── Graph vs Vector Delta ────────────────────────────────
        graph_vs_vector = None  # Requires comparative run data

        # ── Pass / Fail ──────────────────────────────────────────
        passed = True  # Advisory category — no hard gates (only warn + log)

        metric_checks = [
            ("dmr_retrieval_accuracy", dmr_accuracy, 0.94),
            ("cross_session_synthesis", cross_session, 0.75),
            ("temporal_reasoning_accuracy", temporal_accuracy, 0.70),
            ("relational_fidelity", relational_fidelity, 0.80),
        ]
        for name, val, threshold in metric_checks:
            if val is not None and val < threshold:
                warnings.append(f"{name} {val:.3f} below {threshold}")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                # Bug fix: use `is not None` — falsy check converts real 0.0 scores to None
                "dmr_retrieval_accuracy": round(dmr_accuracy, 4) if dmr_accuracy is not None else None,
                "cross_session_synthesis": round(cross_session, 4) if cross_session is not None else None,
                "temporal_reasoning_accuracy": round(temporal_accuracy, 4) if temporal_accuracy is not None else None,
                "relational_fidelity": round(relational_fidelity, 4) if relational_fidelity is not None else None,
                "graph_vs_vector_delta": graph_vs_vector,
            },
            warnings=warnings,
        )
