"""
evals/persona_consistency.py
Conditional Category — Persona Consistency (§4.10)
Fires when agent_type == SOCIAL_SIM or persona_spec is populated.
Based on PersonaGym (ACL 2025), ConsistencyAI (2025), LoCoMo (ACL 2024).

Metrics:
- PersonaScore:            decision-theory 3-axis [1-5]           ≥ 3.5/5
- Factual consistency:     cross-persona cosine similarity        ≥ 0.87
- Long-term memory:        cross-session QA accuracy              ≥ 0.70
- Behavioural drift rate:  embedding distance per 10 turns        ≤ 0.15
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult, LLMJudge


class PersonaEvaluator(BaseEvaluator):
    """Evaluates persona consistency for social simulation agents."""

    category = "persona_consistency"

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

        persona = card.persona_spec or "No persona specification provided"
        warnings: List[str] = []

        # ── PersonaScore (Decision Theory: normative + prescriptive + descriptive)
        persona_scores = []
        # Bug fix: collect all kappas and average — overwriting each iteration
        # kept only the last value (same bug fixed in multi_agent.py / trajectory.py)
        judge_kappas: List[float] = []
        for r in all_records[:10]:
            rubric = self.judge.load_rubric("persona_rubric")
            score, kappa, reliable = await self.judge.judge(
                prompt=f"Persona specification: {persona}\n\nTask: {r.task}",
                rubric=rubric,
                content=r.final_output[:500],
                scale=(1, 5),
            )
            # Bug fix: score can be None when LLM judge is unavailable.
            # Appending None to persona_scores causes statistics.mean() to crash.
            if score is not None:
                persona_scores.append(score)
            judge_kappas.append(kappa)

        persona_score = statistics.mean(persona_scores) if persona_scores else None

        # ── Factual Consistency (ConsistencyAI) ──────────────────
        # Cross-persona cosine similarity on outputs across runs
        consistency_scores = []
        outputs = [r.final_output for r in all_records if r.final_output]

        if len(outputs) >= 2:
            # Simplified: check output similarity across runs of same task
            for task, runs in all_runs.items():
                if len(runs) < 2:
                    continue
                task_outputs = [r.final_output for r in runs]
                # Use word overlap as a proxy for cosine similarity
                for i in range(len(task_outputs)):
                    for j in range(i + 1, len(task_outputs)):
                        sim = self._word_overlap_similarity(
                            task_outputs[i], task_outputs[j]
                        )
                        consistency_scores.append(sim)

        factual_consistency = (
            statistics.mean(consistency_scores)
            if consistency_scores else None
        )

        # ── Long-Term Memory (LoCoMo) ────────────────────────────
        # Cross-session QA accuracy across 35+ sessions
        ltm_score = None  # Requires multi-session test data
        ltm_note = "Requires structured multi-session test suite (35+ sessions, 300 turns)"

        # ── Behavioural Drift Rate ───────────────────────────────
        # Change in predicted responses per 10 turns
        drift_scores = []
        for task, runs in all_runs.items():
            if len(runs) < 2:
                continue
            # Measure output drift across sequential runs
            for i in range(1, len(runs)):
                sim = self._word_overlap_similarity(
                    runs[i - 1].final_output, runs[i].final_output
                )
                drift = 1.0 - sim  # Higher drift = more change
                drift_scores.append(drift)

        drift_rate = statistics.mean(drift_scores) if drift_scores else None

        # ── Pass / Fail ──────────────────────────────────────────
        passed = True  # Advisory category

        if persona_score is not None and persona_score < 3.5:
            warnings.append(f"PersonaScore {persona_score:.2f} below 3.5/5")
        if factual_consistency is not None and factual_consistency < 0.87:
            warnings.append(f"Factual consistency {factual_consistency:.3f} below 0.87")
        if drift_rate is not None and drift_rate > 0.15:
            warnings.append(f"Behavioural drift rate {drift_rate:.3f} exceeds 0.15")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                # Bug fix: use `is not None` — falsy check converts real 0.0 scores to None
                "persona_score": round(persona_score, 4) if persona_score is not None else None,
                "factual_consistency": round(factual_consistency, 4) if factual_consistency is not None else None,
                "long_term_memory": ltm_score,
                "behavioural_drift_rate": round(drift_rate, 4) if drift_rate is not None else None,
            },
            details={"ltm_note": ltm_note, "persona_spec": persona[:200]},
            warnings=warnings,
            judge_reliability=statistics.mean(judge_kappas) if judge_kappas else None,
        )

    @staticmethod
    def _word_overlap_similarity(text_a: str, text_b: str) -> float:
        """Simple word-overlap Jaccard similarity as a proxy for cosine."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
