"""
evals/rag_quality.py
Conditional Category — RAG Quality (§4.8)
Fires when memory_type ∈ {VECTOR_DB, HYBRID}.
Uses Ragas evaluation triad.

Metrics:
- Context precision:   relevant_chunks / total_chunks_retrieved    ≥ 0.80
- Context recall:      required_chunks_retrieved / total_required  ≥ 0.75
- Faithfulness:        claims_supported / total_claims              ≥ 0.85
- Answer relevancy:    cosine(reverse_questions, original)         ≥ 0.80
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult, LLMJudge


class RAGEvaluator(BaseEvaluator):
    """Evaluates the complete RAG triad — precision, recall, faithfulness."""

    category = "rag_quality"

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

        # ── Context Precision ────────────────────────────────────
        # G1 fix: use LLM judge for relevance instead of len>20 heuristic
        precision_scores = []
        for r in all_records[:10]:  # Cap for cost
            if not r.retrieved_chunks:
                continue
            total = len(r.retrieved_chunks)
            relevant_count = 0
            for chunk in r.retrieved_chunks[:5]:  # Cap chunks per record
                chunk_content = str(chunk.get("content", ""))
                if not chunk_content or len(chunk_content) < 5:
                    continue
                score, _, _ = await self.judge.judge(
                    prompt=r.task,
                    rubric=(
                        "Is this retrieved chunk relevant to answering the question?\n"
                        "1 = Completely irrelevant. 5 = Directly answers or supports the answer."
                    ),
                    content=f"Question: {r.task}\nChunk: {chunk_content[:500]}",
                    scale=(1, 5),
                )
                if score is not None and score >= 3.0:  # threshold: score 3+ = relevant
                    relevant_count += 1
            precision_scores.append(relevant_count / total if total > 0 else 0.0)

        context_precision = (
            statistics.mean(precision_scores) if precision_scores else None
        )

        # ── Context Recall ───────────────────────────────────
        # FIX #2: Use golden_sources (retrieval topics), not golden_milestones (task markers)
        context_recall = None
        golden_sources = getattr(card, 'golden_sources', None) or []
        if golden_sources:
            recall_scores = []
            for r in all_records:
                chunks_text = " ".join(
                    str(c.get("content", "")) for c in r.retrieved_chunks
                )
                covered = sum(
                    1 for s in golden_sources
                    if s.lower() in chunks_text.lower()
                )
                recall_scores.append(
                    covered / len(golden_sources)
                )
            context_recall = statistics.mean(recall_scores) if recall_scores else 0.0

        # ── Faithfulness (Groundedness) ──────────────────────────
        # Claims in output supported by retrieved context
        faithfulness_scores = []
        for r in all_records[:10]:  # Cap for cost
            if not r.retrieved_chunks:
                continue

            context_text = "\n".join(
                str(c.get("content", "")) for c in r.retrieved_chunks[:5]
            )

            score, kappa, reliable = await self.judge.judge(
                prompt=r.task,
                rubric=(
                    "Rate how faithfully the output is grounded in the provided context.\n"
                    "1 = Output contradicts or ignores the context entirely.\n"
                    "2 = Output mostly ignores context, relies on hallucination.\n"
                    "3 = Output partially uses context but adds unsupported claims.\n"
                    "4 = Output mostly grounded in context with minor unsupported additions.\n"
                    "5 = Output fully grounded in and supported by the provided context."
                ),
                content=f"Context:\n{context_text[:1000]}\n\nOutput:\n{r.final_output[:500]}",
                scale=(1, 5),
            )
            if score is not None:
                faithfulness_scores.append(score / 5.0)

        faithfulness = (
            statistics.mean(faithfulness_scores) if faithfulness_scores else None
        )

        # ── Answer Relevancy ─────────────────────────────────────
        relevancy_scores = []
        for r in all_records[:10]:
            score, _, _ = await self.judge.judge(
                prompt=r.task,
                rubric=(
                    "Rate how relevant the answer is to the original question.\n"
                    "1 = Completely irrelevant. 5 = Perfectly answers the question."
                ),
                content=f"Question: {r.task}\nAnswer: {r.final_output[:500]}",
                scale=(1, 5),
            )
            if score is not None:
                relevancy_scores.append(score / 5.0)

        answer_relevancy = (
            statistics.mean(relevancy_scores) if relevancy_scores else None
        )

        # ── Pass/Fail ────────────────────────────────────────────
        # Faithfulness is the hard gate (≥ 0.85 blocks PR merge)
        passed = faithfulness is None or faithfulness >= 0.85

        if context_precision is not None and context_precision < 0.80:
            warnings.append(f"Context precision {context_precision:.3f} below 0.80")
        if context_recall is not None and context_recall < 0.75:
            warnings.append(f"Context recall {context_recall:.3f} below 0.75")
        if answer_relevancy is not None and answer_relevancy < 0.80:
            warnings.append(f"Answer relevancy {answer_relevancy:.3f} below 0.80")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                # Bug fix: use `is not None` — `if x` evaluates 0.0 as falsy,
                # causing a real zero-score to be reported as None instead.
                "context_precision": round(context_precision, 4) if context_precision is not None else None,
                "context_recall": round(context_recall, 4) if context_recall is not None else None,
                "faithfulness": round(faithfulness, 4) if faithfulness is not None else None,
                "answer_relevancy": round(answer_relevancy, 4) if answer_relevancy is not None else None,
            },
            warnings=warnings,
        )
