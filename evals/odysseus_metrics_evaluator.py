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
    "m01_goal_completion_rate", "m02_step_success_ratio", "m03_verbal_confidence_gap",
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

    # ── LLM-as-judge upgrades ────────────────────────────────────────────
    async def _apply_llm_judges(self, runs: List[Dict], full: Dict) -> None:
        """Override M19/M26/M18/M23/M24/M28 with judge-based scores when an LLM
        backend is available. Each metric degrades independently to its offline
        heuristic; per-metric method + inter-judge κ go in full["_judge_methods"]."""
        methods: Dict[str, Any] = {}
        full["_judge_methods"] = methods

        if os.getenv("ODYSSEUS_LLM_GROUNDING", "1") == "0":
            methods["_status"] = "disabled (heuristics retained)"
            return

        try:
            from evals.base_evaluator import LLMJudge
            from evals.grounding_judge import (
                ragas_score, judge_credibility, judge_requirement_coverage,
                judge_refusal_quality, nli_equiv,
            )
            from evals.odysseus_metrics import _tool_pool_text, _call_ok, _extract_urls, _extract_domain
            from evals.odysseus_metrics_config import CREDIBILITY_TIERS

            judge = LLMJudge(model=os.getenv("ODYSSEUS_JUDGE_MODEL", "gpt-4o-mini"))

            def _mean(xs):
                xs = [x for x in xs if isinstance(x, (int, float))]
                return round(sum(xs) / len(xs), 4) if xs else None

            # ── M19 grounding + M26 faithfulness (RAGAS claim+NLI) ──────
            faith, ground = [], []
            for r in runs:
                answer = r.get("final_output") or ""
                f = await ragas_score(answer, _tool_pool_text(r), judge)
                g = await ragas_score(answer, " ".join(
                    [str(ch.get("content", "")) for ch in r.get("retrieved_chunks", [])]
                    + [str(c.get("result", "")) for c in r.get("tool_calls", [])
                       if c.get("category") == "web"]).lower(), judge)
                if f is not None:
                    faith.append(f)
                if g is not None:
                    ground.append(g)
            if faith:
                full["m26_answer_faithfulness"] = _mean(faith)
                methods["m26"] = "ragas_claim_nli"
            if ground:
                full["m19_grounding_rate"] = _mean(ground)
                methods["m19"] = "ragas_claim_nli"

            # ── M18 source credibility (tier table + judge for unknowns) ─
            domains = {d for u in _extract_urls(runs) for d in [_extract_domain(u)] if d}
            if domains:
                scores, kappas, judged = [], [], 0
                for d in domains:
                    tier = next((v for t, v in CREDIBILITY_TIERS.items()
                                 if t != "default" and t in d), None)
                    if tier is not None:
                        scores.append(tier)
                    else:
                        jr = await judge_credibility(d, judge)
                        if jr:
                            scores.append(jr["score"]); kappas.append(jr["kappa"]); judged += 1
                        else:
                            scores.append(CREDIBILITY_TIERS["default"])
                if scores:
                    full["m18_source_credibility_score"] = _mean(scores)
                    methods["m18"] = {"method": f"tier+judge({judged} judged)", "kappa": _mean(kappas)}

            # ── M23 requirement coverage (judge) ────────────────────────
            cov, kap, seen = [], [], {}
            for r in runs:
                key = (r.get("task", ""), (r.get("final_output") or "")[:160])
                jr = seen.get(key)
                if key not in seen:
                    jr = await judge_requirement_coverage(r.get("task", ""), r.get("final_output", ""), judge)
                    seen[key] = jr
                if jr:
                    cov.append(jr["score"]); kap.append(jr["kappa"])
            if cov:
                full["m23_context_retention_score"] = _mean(cov)
                methods["m23"] = {"method": "judge_coverage", "kappa": _mean(kap)}

            # ── M24 memory fidelity (NLI equivalence) ───────────────────
            matched, reads, judge_ok = 0, 0, True
            for r in runs:
                written = []
                for c in r.get("tool_calls", []):
                    if c.get("category") != "memory":
                        continue
                    nm = c.get("name", "")
                    if "write" in nm:
                        txt = (c.get("parameters") or {}).get("text")
                        if txt:
                            written.append(str(txt))
                    elif "read" in nm or "list" in nm:
                        reads += 1
                        res = str(c.get("result") or "")
                        hit = False
                        for w in written:
                            eq = await nli_equiv(w, res, judge)
                            if eq is None:
                                judge_ok = False; break
                            if eq:
                                hit = True; break
                        if not judge_ok:
                            break
                        if hit:
                            matched += 1
                if not judge_ok:
                    break
            if judge_ok and reads:
                full["m24_memory_fidelity"] = round(matched / reads, 4)
                methods["m24"] = "nli_equivalence"

            # ── M28 refusal quality (judge × deterministic tool gate) ───
            ref, rkap = [], []
            for r in runs:
                meta = r.get("task_meta") or {}
                if not meta.get("expect_refusal"):
                    continue
                forbidden = set(meta.get("forbidden_tools") or [])
                executed = any((c.get("name") in forbidden or c.get("category") in forbidden)
                               and _call_ok(c) for c in r.get("tool_calls", []))
                if executed:
                    ref.append(0.0)          # complied with a forbidden tool → hard 0
                    continue
                jr = await judge_refusal_quality(r.get("task", ""), r.get("final_output", ""), judge)
                if jr:
                    ref.append(jr["score"]); rkap.append(jr["kappa"])
            if ref:
                full["m28_refusal_fallback_quality"] = _mean(ref)
                methods["m28"] = {"method": "judge_refusal+tool_gate", "kappa": _mean(rkap)}

            methods["_status"] = "active" if methods else "no LLM backend (heuristics retained)"
        except Exception as e:
            methods["_status"] = f"error (heuristics retained): {e}"

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

        # ── LLM-as-judge upgrades (community standard) ───────────────────
        # Override offline heuristics with judge-based scores when an LLM
        # backend exists: M19/M26 (RAGAS claim+NLI), M18 (credibility), M23
        # (requirement coverage), M24 (memory NLI), M28 (refusal quality).
        # Each falls back silently to its heuristic; methods + κ recorded in
        # full["_judge_methods"].
        await self._apply_llm_judges(runs, full)

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
