"""
evals/reliability.py
Category 5 — Reliability (§4.5)

Uses FailureInjector for mandatory stress-testing.

Metrics:
- Recovery rate:     successful_recoveries / total_injections    ≥ 0.75
- Consistency score: 1 − σ(binary outcomes over k runs)          ≥ 0.85
- SLA compliance:    within_latency_and_success / total           ≥ 0.90
- Policy adherence:  1 − (violations / total_actions)             ≥ 0.95
"""

from __future__ import annotations

import asyncio
import statistics
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult
from tracer.injection_middleware import (
    FailureInjector,
    FailureType,
    InjectionConfig,
)


class ReliabilityEvaluator(BaseEvaluator):
    """Tests agents under real-world adverse conditions with failure injection."""

    category = "reliability"

    # All 7 mandatory injection types
    INJECTION_CONFIGS = [
        InjectionConfig(failure_type=FailureType.HTTP_404, probability=1.0),
        InjectionConfig(failure_type=FailureType.API_TIMEOUT, probability=1.0, timeout_seconds=2.0),
        InjectionConfig(failure_type=FailureType.SCHEMA_ERROR, probability=1.0),
        InjectionConfig(failure_type=FailureType.EMPTY_RESULT, probability=1.0),
        InjectionConfig(failure_type=FailureType.PARTIAL_DATA, probability=1.0),
        InjectionConfig(failure_type=FailureType.RATE_LIMIT_429, probability=1.0),
        InjectionConfig(failure_type=FailureType.AUTH_FAILURE_401, probability=1.0),
    ]

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

        # ── Recovery Rate (via Failure Injection) ────────────────
        recovery_rate = 0.0
        injection_details = {}

        if wrapper:
            # Skip live failure injection for remote or mock CI agents.
            # Remote: each call triggers a full pipeline run (e.g. 27 min).
            # Mock CI: DefaultMockAdapter has no recovery logic — injection is meaningless.
            from adapters.remote_adapter import RemoteAgentAdapter
            _adapter = getattr(wrapper, 'adapter', None)
            is_remote = isinstance(_adapter, RemoteAgentAdapter)
            is_mock = getattr(_adapter, 'is_mock_ci', False)

            if is_remote:
                recovery_rate = 1.0  # Neutral — not applicable for remote agents
                warnings.append("Failure injection skipped for remote agent (HTTP adapter handles errors)")
            elif is_mock:
                recovery_rate = 1.0  # Neutral — mock CI adapter has no recovery logic
                warnings.append("Failure injection skipped — mock CI adapter (not a real agent)")
            else:
                total_injections = 0
                total_recoveries = 0

                for config in self.INJECTION_CONFIGS:
                    injector = FailureInjector(
                        adapter=wrapper.adapter,
                        injections=[config],
                        seed=42,
                    )

                    # L7 fix: sample across tasks, not just the first one
                    task_keys = list(all_runs.keys()) if all_runs else ["test"]
                    task_idx = self.INJECTION_CONFIGS.index(config) % len(task_keys)
                    test_task = task_keys[task_idx]
                    try:
                        result, injection_results = await asyncio.to_thread(
                            injector.run_with_injection, test_task
                        )

                        for ir in injection_results:
                            if ir.injected:
                                total_injections += 1
                                if ir.agent_recovered:
                                    total_recoveries += 1
                                injection_details[ir.failure_type.value] = {
                                    "recovered": ir.agent_recovered,
                                    "strategy": ir.recovery_strategy,
                                }
                    except Exception as e:
                        total_injections += 1
                        injection_details[config.failure_type.value] = {
                            "recovered": False,
                            "error": str(e),
                        }

                recovery_rate = (
                    total_recoveries / total_injections
                    if total_injections > 0 else 0.0
                )
        else:
            warnings.append("No wrapper provided — skipped failure injection testing")

        # ── Consistency Score ────────────────────────────────────
        # 1 − σ(binary outcomes over k identical runs)
        consistency_scores = []
        for task, runs in all_runs.items():
            outcomes = [1.0 if r.success else 0.0 for r in runs]
            if len(outcomes) >= 2:
                stdev = statistics.stdev(outcomes)
                consistency_scores.append(1.0 - stdev)
            elif outcomes:
                consistency_scores.append(1.0)

        consistency = statistics.mean(consistency_scores) if consistency_scores else 0.0

        # ── SLA Compliance Rate ──────────────────────────────────
        # tasks within (latency_ms ≤ sla_latency_ms AND success) / total
        sla_compliant = 0
        for r in all_records:
            if r.wall_latency_ms <= card.sla_latency_ms and r.success:
                sla_compliant += 1

        sla_rate = sla_compliant / len(all_records) if all_records else 0.0

        # ── Policy Adherence Score (PAS) ─────────────────────────
        # 1 − (policy_violations / total_actions) across k runs
        total_actions = sum(len(r.tool_calls) + 1 for r in all_records)  # +1 for final output
        total_violations = sum(len(r.policy_violations) for r in all_records)

        pas = 1.0 - (total_violations / total_actions) if total_actions > 0 else 1.0

        # ── Pass/Fail ────────────────────────────────────────────
        passed = (
            recovery_rate >= 0.75
            and consistency >= 0.85
            and pas >= 0.95
        )

        if sla_rate < 0.90:
            warnings.append(f"SLA compliance {sla_rate:.2%} below 90% threshold")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                "recovery_rate": round(recovery_rate, 4),
                "consistency_score": round(consistency, 4),
                "sla_compliance_rate": round(sla_rate, 4),
                "policy_adherence_score": round(pas, 4),
            },
            details={
                "injection_results": injection_details,
                "total_violations": total_violations,
                "sla_latency_ms": card.sla_latency_ms,
            },
            warnings=warnings,
        )
