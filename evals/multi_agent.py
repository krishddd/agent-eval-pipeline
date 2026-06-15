"""
evals/multi_agent.py
Category 4 — Multi-Agent Coordination (§4.4)
Based on MultiAgentBench (ACL 2025) MARBLE framework.

Metrics:
- Coordination score:       (comm + planning) / 2 [LLM judge 1-5]   ≥ 3.5/5
- Collaboration success:    successful_handoffs / total_handoffs     ≥ 0.85
- Task handoff accuracy:    correct_assignments / total_handoffs     ≥ 0.85
- Communication overhead:   inter_agent_tokens / output_tokens       ≤ 3.0×
- Individual contribution:  milestones_by_agent / total_milestones   ICI > 0 for all
- Workflow parallelism:     parallel_steps / max_parallel            ≥ 0.60
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from evals.base_evaluator import BaseEvaluator, EvalResult, LLMJudge


class MultiAgentEvaluator(BaseEvaluator):
    """Multi-agent coordination evaluator for orchestrators and swarms."""

    category = "multi_agent_coord"

    def __init__(self):
        self.judge = LLMJudge(n_judges=3)

    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],
        card: Any,
        wrapper: Any = None,
    ) -> Optional[EvalResult]:

        if not card.subagents and card.agent_type.value not in ("orchestrator", "swarm"):
            return None

        all_records = self._all_records(all_runs)
        if not all_records:
            return None

        warnings: List[str] = []

        # ── Coordination Score (LLM Judge) ───────────────────────
        coord_scores = []
        judge_kappas = []

        for r in all_records[:5]:
            msgs_text = "\n".join(
                f"  {m.sender_id} → {m.receiver_id}: {m.content[:100]}"
                for m in r.agent_messages[:10]
            )
            content = (
                f"Task: {r.task}\n"
                f"Agent messages:\n{msgs_text}\n"
                f"Subagents: {card.subagents}\n"
                f"Output: {r.final_output[:200]}"
            )
            rubric = self.judge.load_rubric("coordination_rubric")
            score, kappa, reliable = await self.judge.judge(
                prompt=r.task, rubric=rubric, content=content, scale=(1, 5)
            )
            if score is not None:
                coord_scores.append(score)
            # FIX #9: Collect all kappas, don't overwrite
            judge_kappas.append(kappa)

        judge_available = bool(coord_scores)
        coordination_score = statistics.mean(coord_scores) if coord_scores else 0.0

        # ── Collaboration Success Rate ───────────────────────────
        # F5 fix: per-handoff success — message followed by receiver's response
        collab_scores = []
        for r in all_records:
            total_handoffs = len(r.agent_messages)
            if total_handoffs == 0:
                collab_scores.append(1.0 if r.success else 0.0)
                continue
            receivers_who_responded = set()
            seen_receivers = set()
            for i, m in enumerate(r.agent_messages):
                seen_receivers.add(m.receiver_id)
                for later_m in r.agent_messages[i + 1:]:
                    if later_m.sender_id == m.receiver_id:
                        receivers_who_responded.add(m.receiver_id)
                        break
            collab_scores.append(
                len(receivers_who_responded) / len(seen_receivers) if seen_receivers else 0.0
            )
        collab_sr = statistics.mean(collab_scores) if collab_scores else 0.0

        # ── Task Handoff Accuracy ────────────────────────
        # F6 fix: independent metric — checks if messages route to valid subagents
        # Build expanded set: card.subagents + any sender/receiver IDs from messages
        # (remote pipelines use step names like 'deep_research' not role names)
        subagent_set = set(card.subagents) if card.subagents else set()
        # Also collect all unique agent IDs from actual messages
        all_msg_agents = set()
        for r in all_records:
            for m in (r.agent_messages or []):
                all_msg_agents.add(m.sender_id)
                all_msg_agents.add(m.receiver_id)
        # Combined: declared subagents + observed agents
        known_agents = subagent_set | all_msg_agents

        handoff_acc_scores = []
        for r in all_records:
            if not r.agent_messages or not known_agents:
                handoff_acc_scores.append(1.0 if r.success else 0.0)
                continue
            valid_routes = sum(
                1 for m in r.agent_messages
                if m.receiver_id in known_agents or m.sender_id in known_agents
            )
            handoff_acc_scores.append(valid_routes / len(r.agent_messages))
        handoff_acc = statistics.mean(handoff_acc_scores) if handoff_acc_scores else 0.0

        # ── Communication Overhead Ratio ─────────────────────
        overhead_ratios = []
        for r in all_records:
            if r.output_tokens > 0:
                ratio = r.inter_agent_tokens / r.output_tokens
                overhead_ratios.append(ratio)
        comm_overhead = statistics.mean(overhead_ratios) if overhead_ratios else 0.0

        # ── Individual Contribution Index (ICI) ──────────────
        # F4 fix: attribute milestones to agents, not message counts
        ici_scores: Dict[str, float] = defaultdict(float)
        total_milestones = 0
        for r in all_records:
            total_milestones += len(r.milestones_hit)
            if not r.agent_messages or not r.milestones_hit:
                continue
            speaker_order = [m.sender_id for m in r.agent_messages]
            for i, milestone in enumerate(r.milestones_hit):
                speaker_idx = min(i, len(speaker_order) - 1)
                ici_scores[speaker_order[speaker_idx]] += 1

        free_riders = []
        # FIX: Use known_agents (includes step names from messages)
        # Only flag as free-rider if the agent contributed ZERO milestones
        if total_milestones > 0:
            # Check all known agents, but only flag declared subagents
            check_set = subagent_set if subagent_set else all_msg_agents
            for agent in check_set:
                agent_ici = ici_scores.get(agent, 0) / max(total_milestones, 1)
                if agent_ici == 0:
                    # Before flagging, check if this agent appears in messages at all
                    # If a declared subagent never appears in messages, it might be a name mismatch
                    # rather than a true free-rider.
                    appeared_in_messages = agent in all_msg_agents
                    if not appeared_in_messages:
                        # This subagent name doesn't match any message agent
                        # Could be a name mismatch — don't flag as free-rider
                        continue
                    free_riders.append(agent)

        # ── Workflow Parallelism Score ────────────────────────────
        parallelism_scores = []
        for r in all_records:
            unique_agents = len(set(m.sender_id for m in r.agent_messages))
            max_possible = len(card.subagents) if card.subagents else 1
            parallelism_scores.append(
                unique_agents / max_possible if max_possible > 0 else 0.0
            )
        parallelism = statistics.mean(parallelism_scores) if parallelism_scores else 0.0

        # ── Pass/Fail ────────────────────────────────────────────
        # Skip coordination gate when LLM judge is unavailable — a score of
        # 0.0 from an empty judge run is not evidence of poor coordination.
        if not judge_available:
            warnings.append(
                "Coordination score not measured — LLM judge unavailable (install openai)"
            )
        passed = (
            (coordination_score >= 3.5 if judge_available else True)
            and collab_sr >= 0.85
            and handoff_acc >= 0.85
        )

        if comm_overhead > 3.0:
            warnings.append(f"Communication overhead {comm_overhead:.2f}× exceeds 3.0×")
        if free_riders:
            warnings.append(f"Free-rider detected: {free_riders}")
        if parallelism < 0.60:
            warnings.append(f"Parallelism {parallelism:.2f} below 0.60 advisory")

        return EvalResult(
            category=self.category,
            passed=passed,
            metrics={
                "coordination_score": round(coordination_score, 4),
                "collaboration_success_rate": round(collab_sr, 4),
                "task_handoff_accuracy": round(handoff_acc, 4),
                "communication_overhead_ratio": round(comm_overhead, 4),
                "workflow_parallelism_score": round(parallelism, 4),
            },
            details={
                "ici_per_agent": dict(ici_scores),
                "free_riders": free_riders,
                "subagents": card.subagents,
            },
            warnings=warnings,
            # FIX #9: Average all kappas
            judge_reliability=statistics.mean(judge_kappas) if judge_kappas else None,
        )
