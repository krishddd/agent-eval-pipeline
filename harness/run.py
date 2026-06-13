"""
harness/run.py
CLI entry point for running the eval harness from command line or CI/CD.

Usage:
    python -m harness.run --task-suite tasks/smoke_suite.json --output eval_results.json --fail-on-threshold
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.eval_harness import EvalHarness, EvalFailureError
from registry.registry import registry
from adapters.base import AgentAdapter, AgentResult


class DefaultMockAdapter(AgentAdapter):
    """Default mock adapter for testing — replace with actual adapter factory."""

    def run(self, task, on_tool_call=None, on_agent_msg=None, on_retrieval=None):
        return AgentResult(
            output=f"Mock response to: {task[:50]}",
            success=True,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            milestones=["mock_completed"],
        )


def adapter_factory(card):
    """
    Create the appropriate adapter for the given agent card.
    Override this factory to use real framework adapters.
    """
    return DefaultMockAdapter()


async def main():
    parser = argparse.ArgumentParser(description="Run the Agent Eval Harness")
    parser.add_argument("--task-suite", required=True, help="Path to task suite JSON")
    parser.add_argument("--output", default="eval_results.json", help="Output file path")
    parser.add_argument("--agent-id", help="Specific agent ID to eval (default: all)")
    parser.add_argument("--fail-on-threshold", action="store_true", help="Exit non-zero on threshold violation")
    parser.add_argument("--max-concurrent", type=int, default=4, help="Max concurrent runs")

    args = parser.parse_args()

    # Load task suite
    with open(args.task_suite, "r") as f:
        suite_data = json.load(f)
    task_suite = suite_data.get("tasks", [])

    if not task_suite:
        print("Error: No tasks in suite")
        sys.exit(1)

    # Get agents to evaluate
    agents = registry.list_all()
    if not agents:
        print("Warning: No agents registered. Run scripts/register_all_agents.py first.")
        sys.exit(0)

    if args.agent_id:
        agents = [a for a in agents if a.agent_id == args.agent_id]

    harness = EvalHarness(registry, adapter_factory, max_concurrent_runs=args.max_concurrent)

    git_sha = os.getenv("GIT_SHA")
    trigger = os.getenv("TRIGGER", "manual")

    all_reports = []
    overall_pass = True

    for card in agents:
        print(f"\n{'='*60}")
        print(f"Evaluating: {card.name} ({card.agent_type.value})")
        print(f"Categories: {card.eval_categories}")
        print(f"{'='*60}")

        try:
            report = await harness.run_eval(
                agent_id=card.agent_id,
                task_suite=task_suite,
                git_sha=git_sha,
                trigger=trigger,
                fail_on_threshold=False,  # Check manually below
            )

            all_reports.append(report.model_dump(mode="json"))

            # Print results
            for r in report.results:
                status = "✅ PASS" if r.passed else "❌ FAIL"
                print(f"  [{status}] {r.category}")
                for k, v in r.metrics.items():
                    if v is not None:
                        print(f"    {k}: {v}")
                for w in r.warnings:
                    print(f"    ⚠️  {w}")

            print(f"\n  Overall: {'✅ PASS' if report.overall_passed else '❌ FAIL'}")

            if not report.overall_passed:
                overall_pass = False

        except EvalFailureError as e:
            print(f"  ❌ THRESHOLD VIOLATION: {e.violations}")
            overall_pass = False
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            overall_pass = False

    # Save results
    combined = {
        "overall_passed": overall_pass,
        "agent_count": len(all_reports),
        "reports": all_reports,
        **(all_reports[0] if len(all_reports) == 1 else {}),
    }

    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n✓ Results saved to {args.output}")

    if args.fail_on_threshold and not overall_pass:
        print("\n❌ Eval failed — threshold violations detected")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
