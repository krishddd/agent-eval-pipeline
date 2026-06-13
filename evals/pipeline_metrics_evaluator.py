"""
evals/pipeline_metrics_evaluator.py
PipelineMetricsEvaluator — Computes all 33 pipeline-level metrics (M01–M33).

This evaluator integrates with the EvalHarness via the standard
BaseEvaluator interface.  It extracts the full PipelineResult dict
from TrajectoryRecord.raw → AgentResult.raw and passes it through
the pipeline_metrics.compute_all() function.

The resulting metrics are:
  1. Stored in the EvalResult.metrics dict (flat keys for scorecard)
  2. Stored in EvalResult.details["pipeline_metrics"] (full nested JSON)
  3. Written as a separate pipeline_metrics.json by the report store

Category name: "pipeline_metrics"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult
from evals.pipeline_metrics import EWMAMonitor, compute_all
from evals.pipeline_metrics_config import METRIC_THRESHOLDS


class PipelineMetricsEvaluator(BaseEvaluator):
    """
    Evaluates pipeline-level cross-step metrics (M01–M33) for agents
    that return a PipelineResult JSON via the RemoteAgentAdapter.

    Applicable when: AgentResult.raw contains 'step_log' (i.e. the
    agent is a multi-step pipeline exposed over HTTP).
    """

    category = "pipeline_metrics"

    def __init__(self):
        self._ewma_monitor = EWMAMonitor()

    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:
        """
        Compute all 33 pipeline metrics from the raw PipelineResult.

        For remote pipeline agents, there is typically k=1 run.
        We extract pipeline_data from the first successful run's raw field.
        """
        all_records = self._all_records(all_runs)
        if not all_records:
            return None

        # Find the first record with pipeline data
        pipeline_data = None
        for record in all_records:
            # Primary: pipeline_data field on TrajectoryRecord
            raw = getattr(record, "pipeline_data", None)
            if isinstance(raw, dict) and "step_log" in raw:
                pipeline_data = raw
                break
            # Fallback: check raw attribute
            raw = getattr(record, "raw", None)
            if isinstance(raw, dict) and "step_log" in raw:
                pipeline_data = raw
                break

        if pipeline_data is None:
            # No pipeline data found — this evaluator is not applicable
            return None

        # ── Compute all 33 metrics ──────────────────────────────────
        full_metrics = compute_all(pipeline_data, self._ewma_monitor)

        # ── Extract flat metrics for scorecard ──────────────────────
        flat = self._flatten_for_scorecard(full_metrics)

        # ── Determine pass/fail ─────────────────────────────────────
        warnings = full_metrics.get("_meta", {}).get("warnings", [])
        critical_failures = []

        # Check critical metrics
        calibration_gap = full_metrics.get("m01_calibration_gap")
        if calibration_gap is not None and calibration_gap > 60:
            critical_failures.append(f"M01 calibration gap {calibration_gap}% > 60%")

        alignment = full_metrics.get("m02_sentiment_alignment_score")
        if alignment is not None and alignment < 0.4:
            critical_failures.append(f"M02 sentiment alignment {alignment} < 0.4")

        tool_success = full_metrics.get("m23_tool_execution_success_rate")
        if tool_success is not None and tool_success < 0.7:
            critical_failures.append(f"M23 tool success rate {tool_success} < 0.7")

        judge_rel = full_metrics.get("m32_judge_reliability_score")
        if judge_rel is not None and judge_rel < 0.4:
            critical_failures.append(f"M32 judge reliability {judge_rel} < 0.4")

        passed = len(critical_failures) == 0

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics=flat,
            details={
                "pipeline_metrics": full_metrics,
                "critical_failures": critical_failures,
            },
            warnings=warnings[:20],  # Cap warnings
            judge_reliability=full_metrics.get("m32_judge_reliability_score"),
        )

    @staticmethod
    def _flatten_for_scorecard(metrics: Dict) -> Dict[str, Optional[float]]:
        """
        Extract the most important scalar metrics for the flat scorecard.
        Only includes numeric values (float/int/None).
        """
        keys_to_include = [
            # Section 1
            "m01_calibration_gap",
            "m02_sentiment_alignment_score",
            "m03_sla_compliance_rate",
            "m04_pipeline_throughput_chars_per_sec",
            # Section 2
            "m05_source_credibility_score",
            "m06_freshness_score_research",
            "m07_query_coverage_score",
            "m08_claim_density_research",
            "m08_claim_density_synthesis",
            "m09_news_overlap_ratio",
            # Section 3
            "m10_data_source_completeness",
            "m11_kpi_coverage_score",
            "m12_xbrl_parse_success_rate",
            "m13_risk_count",
            # Section 4
            "m14_sentiment_confidence",
            "m15_sentiment_abm_gap",
            "m16_marketing_depth_score",
            # Section 5
            "m17_mc_path_variance",
            "m19_agent_consensus_score",
            "m20_contagion_propagation_rate",
            "m22_abm_compute_efficiency",
            # Section 6
            "m23_tool_execution_success_rate",
            "m24_tool_hallucination_rate",
            "m25_parameter_f1_score",
            # Section 7
            "m27_evidence_traceability_score",
            "m28_fallback_quality_score",
            # Section 8
            "m29_context_retention_score",
            "m30_kg_utilisation_rate",
            # Section 9
            "m31_anomalies_detected",
            "m32_judge_reliability_score",
        ]

        flat = {}
        for key in keys_to_include:
            val = metrics.get(key)
            if isinstance(val, (int, float)):
                flat[key] = val
            else:
                flat[key] = None

        return flat
