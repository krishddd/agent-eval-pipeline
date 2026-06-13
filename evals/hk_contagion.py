"""
evals/hk_contagion.py
Conditional Category — HK Contagion (§4.11)
Fires when agent_type == FINANCIAL_ABM and hk_params is populated.
Validates Hegselmann-Krause bounded confidence dynamics.

Metrics:
- Convergence speed τ:       steps until σ < 0.01, vs O(N log N)    anomaly detection
- Clustering accuracy:       cluster count vs theoretical HK          ±1 cluster
- ε-polarisation boundary:   critical ε vs theoretical                within 5%
- Contagion detection rate:   synthetic shocks correctly flagged      ≥ 0.90
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult


class HKContagionEvaluator(BaseEvaluator):
    """Validates HK bounded confidence dynamics for ABM/financial agents."""

    category = "hk_contagion"

    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:

        if not card.hk_params:
            return None

        all_records = self._all_records(all_runs)
        if not all_records:
            return None

        hk = card.hk_params
        epsilon = hk.get("epsilon", 0.2)
        n_agents = hk.get("n_agents", 100)
        warnings: List[str] = []

        # ── Convergence Speed τ ──────────────────────────────────
        # Should grow O(N log N) — detect O(N²) regression
        expected_tau = n_agents * math.log(n_agents) if n_agents > 1 else 1
        convergence_scores = []

        for r in all_records:
            # Extract convergence steps from agent output
            tau = self._extract_convergence_tau(r)
            if tau is not None:
                # Check if τ is within reasonable bounds of O(N log N)
                ratio = tau / expected_tau if expected_tau > 0 else float("inf")
                convergence_scores.append(ratio)

        tau_ratio = statistics.mean(convergence_scores) if convergence_scores else None

        # Bug fix: `if tau_ratio` is False when tau_ratio == 0.0 — use `is not None`
        if tau_ratio is not None and tau_ratio > 2.0:
            warnings.append(
                f"Convergence speed ratio {tau_ratio:.2f}× — possible O(N²) regression"
            )

        # ── Clustering Accuracy ──────────────────────────────────
        # Count of clusters at convergence vs theoretical HK prediction
        theoretical_clusters = self._theoretical_cluster_count(epsilon, n_agents)
        clustering_scores = []

        for r in all_records:
            observed = self._extract_cluster_count(r)
            if observed is not None:
                diff = abs(observed - theoretical_clusters)
                clustering_scores.append(1.0 if diff <= 1 else 0.0)

        clustering_accuracy = (
            statistics.mean(clustering_scores) if clustering_scores else None
        )

        # ── ε-Polarisation Boundary ──────────────────────────────
        # Critical ε below which fragmentation occurs
        theoretical_boundary = 1.0 / (2 * n_agents) if n_agents > 0 else 0.0
        epsilon_scores = []

        for r in all_records:
            observed_boundary = self._extract_epsilon_boundary(r)
            if observed_boundary is not None and theoretical_boundary > 0:
                error = abs(observed_boundary - theoretical_boundary) / theoretical_boundary
                epsilon_scores.append(1.0 if error <= 0.05 else 0.0)

        epsilon_accuracy = (
            statistics.mean(epsilon_scores) if epsilon_scores else None
        )

        # ── Contagion Detection Rate ─────────────────────────────
        # Fraction of synthetic sentiment shocks flagged
        detection_scores = []
        for r in all_records:
            detection = self._extract_contagion_detection(r)
            if detection is not None:
                detection_scores.append(detection)

        contagion_detection = (
            statistics.mean(detection_scores) if detection_scores else None
        )

        # ── Pass/Fail ────────────────────────────────────────────
        # FIX #11: Fail when no metrics could be computed, not silently pass
        all_none = all(v is None for v in [tau_ratio, clustering_accuracy, epsilon_accuracy, contagion_detection])
        if all_none:
            warnings.append("WARN: No HK metrics could be extracted from agent output — check output format")
            passed = False
        elif contagion_detection is not None:
            passed = contagion_detection >= 0.90
        else:
            passed = True  # Some metrics computed but contagion_detection wasn't applicable

        if contagion_detection is not None and contagion_detection < 0.90:
            warnings.append(
                f"Contagion detection {contagion_detection:.2%} below 0.90 — blocks merge"
            )
        if clustering_accuracy is not None and clustering_accuracy < 1.0:
            warnings.append("Clustering count deviates from HK theory by > ±1")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                # Bug fix: use `is not None` — falsy check converts real 0.0 scores to None
                "convergence_speed_ratio": round(tau_ratio, 4) if tau_ratio is not None else None,
                "clustering_accuracy": round(clustering_accuracy, 4) if clustering_accuracy is not None else None,
                "epsilon_polarisation_accuracy": round(epsilon_accuracy, 4) if epsilon_accuracy is not None else None,
                "contagion_detection_rate": round(contagion_detection, 4) if contagion_detection is not None else None,
            },
            details={
                "hk_params": card.hk_params,
                "theoretical_clusters": theoretical_clusters,
                "expected_tau_O_NlogN": round(expected_tau, 2),
            },
            warnings=warnings,
        )

    # ── Helper Methods ───────────────────────────────────────────
    # G2 fix: prefer structured output fields, fallback to regex

    @staticmethod
    def _theoretical_cluster_count(epsilon: float, n: int) -> int:
        """Estimate theoretical cluster count for HK model."""
        if epsilon >= 0.5:
            return 1  # Consensus
        elif epsilon >= 0.25:
            return 2
        elif epsilon >= 0.15:
            return max(1, int(1.0 / (2 * epsilon)))
        else:
            return max(1, int(0.5 / epsilon))

    @staticmethod
    def _try_parse_structured(output: str) -> Optional[dict]:
        """Try to parse JSON from agent output."""
        import json as _json
        import re
        # Look for JSON block in output
        json_match = re.search(r'\{[^{}]*\}', output)
        if json_match:
            try:
                return _json.loads(json_match.group())
            except (ValueError, _json.JSONDecodeError):
                pass
        return None

    @staticmethod
    def _extract_convergence_tau(record) -> Optional[float]:
        """Extract convergence steps from agent output."""
        # G2: try structured fields first
        parsed = HKContagionEvaluator._try_parse_structured(record.final_output)
        if parsed:
            for key in ["convergence_steps", "tau", "convergence_tau", "steps_to_converge"]:
                if key in parsed and parsed[key] is not None:
                    try:
                        return float(parsed[key])
                    except (ValueError, TypeError):
                        pass

        # Fallback: regex on prose
        output = record.final_output.lower()
        for keyword in ["convergence", "steps", "tau", "iterations"]:
            if keyword in output:
                import re
                numbers = re.findall(r"(\d+\.?\d*)", output)
                if numbers:
                    return float(numbers[0])
        return None

    @staticmethod
    def _extract_cluster_count(record) -> Optional[int]:
        """Extract observed cluster count from agent output."""
        parsed = HKContagionEvaluator._try_parse_structured(record.final_output)
        if parsed:
            for key in ["cluster_count", "clusters", "n_clusters", "num_clusters"]:
                if key in parsed and parsed[key] is not None:
                    try:
                        return int(parsed[key])
                    except (ValueError, TypeError):
                        pass

        output = record.final_output.lower()
        for keyword in ["cluster", "group"]:
            if keyword in output:
                import re
                numbers = re.findall(r"(\d+)", output)
                if numbers:
                    return int(numbers[0])
        return None

    @staticmethod
    def _extract_epsilon_boundary(record) -> Optional[float]:
        """Extract observed ε boundary from agent output."""
        parsed = HKContagionEvaluator._try_parse_structured(record.final_output)
        if parsed:
            for key in ["epsilon_boundary", "critical_epsilon", "epsilon_critical"]:
                if key in parsed and parsed[key] is not None:
                    try:
                        return float(parsed[key])
                    except (ValueError, TypeError):
                        pass

        output = record.final_output.lower()
        if "epsilon" in output or "boundary" in output:
            import re
            numbers = re.findall(r"(\d+\.?\d*)", output)
            if numbers:
                return float(numbers[0])
        return None

    @staticmethod
    def _extract_contagion_detection(record) -> Optional[float]:
        """Extract contagion detection rate from agent output."""
        parsed = HKContagionEvaluator._try_parse_structured(record.final_output)
        if parsed:
            for key in ["contagion_detected", "detection_rate", "contagion_rate"]:
                if key in parsed and parsed[key] is not None:
                    val = parsed[key]
                    if isinstance(val, bool):
                        return 1.0 if val else 0.0
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass

        # Fallback: milestones and keywords
        for milestone in record.milestones_hit:
            if "contagion" in milestone.lower() or "detection" in milestone.lower():
                return 1.0

        output = record.final_output.lower()
        if "detected" in output and "contagion" in output:
            return 1.0
        elif "shock" in output:
            return 0.5

        return None
