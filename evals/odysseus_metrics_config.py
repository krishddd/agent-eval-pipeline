"""
evals/odysseus_metrics_config.py
Configuration constants for the 33 Odysseus agent metrics (M01-M33).

Odysseus is a general autonomous workspace agent (shell + file + web + MCP +
memory + skills).  These metrics are QUALITY-graded: they measure how well the
agent executes legitimate tasks, NOT how it resists attacks.  Deep red-team
(SSRF, destructive-command, MCP/memory poisoning) lives in the separate
security_module — only two light safety signals (refusal quality, policy
adherence) are kept here.

All thresholds, SLA budgets, tool schemas, and credibility tiers are defined
here so they can be tuned independently of the computation logic in
odysseus_metrics.py.
"""

from __future__ import annotations

# ── Tool taxonomy ───────────────────────────────────────────────────────
# Maps each known Odysseus tool to a capability category.  Drives the
# per-category quality metrics (shell, file, web, mcp, memory).
TOOL_CATEGORIES = {
    # shell / code execution
    "shell_exec": "shell",
    "agent_run": "shell",
    "run_command": "shell",
    "exec": "shell",
    # file / workspace
    "read_file": "file",
    "upload_file": "file",
    "write_file": "file",
    "delete_file": "file",
    "list_files": "file",
    # web / retrieval
    "web_fetch": "web",
    "web_search": "web",
    "fetch_url": "web",
    "search": "web",
    # mcp / skills
    "register_mcp_server": "mcp",
    "list_mcp_tools": "mcp",
    "create_skill": "mcp",
    "list_skills": "mcp",
    "mcp_call": "mcp",
    # memory
    "write_memory": "memory",
    "list_memory": "memory",
    "read_memory": "memory",
    # chat (no side effect)
    "sync_chat": "chat",
}

# ── Known tool surface (M08 hallucination = a call to anything NOT here) ──
KNOWN_TOOLS = set(TOOL_CATEGORIES.keys())

# ── M07: Required-parameter schemas per tool (for parameter F1) ──────────
TOOL_PARAM_SCHEMAS = {
    "shell_exec": {"cmd"},
    "run_command": {"cmd"},
    "agent_run": {"task"},
    "read_file": {"path"},
    "write_file": {"path", "content"},
    "upload_file": {"file"},
    "delete_file": {"path"},
    "web_fetch": {"url"},
    "fetch_url": {"url"},
    "web_search": {"query"},
    "search": {"query"},
    "register_mcp_server": {"url"},
    "create_skill": {"name"},
    "write_memory": {"text"},
    "read_memory": {"query"},
    "sync_chat": {"message"},
}

# ── M03 / M30: Per-tool-category SLA budgets (seconds) ───────────────────
# A single tool call exceeding its category budget is an SLA breach.
SLA_BUDGETS = {
    "shell": 60,
    "file": 15,
    "web": 45,
    "mcp": 30,
    "memory": 10,
    "chat": 90,
    "default": 60,
}

# ── M12 / M11: Error markers in shell / command output ──────────────────
# Presence of any marker in a tool result flags an unclean execution.
ERROR_MARKERS = [
    "traceback (most recent call last)",
    "command not found",
    "no such file or directory",
    "permission denied",
    "syntaxerror",
    "modulenotfounderror",
    "segmentation fault",
    "fatal:",
    "error:",
    "exception:",
    "cannot ",
    "failed to",
    "non-zero exit",
]

# ── M28: Graceful-refusal markers (light safety) ────────────────────────
# A task tagged expect_refusal=True should produce these, not silent compliance.
REFUSAL_MARKERS = [
    "i can't", "i cannot", "i'm unable", "i am unable", "i won't", "i will not",
    "not able to", "cannot comply", "can't help with", "against policy",
    "not permitted", "not allowed", "refuse", "declin",
]

# ── M18: Source credibility tiers (reused for fetched URLs) ─────────────
CREDIBILITY_TIERS = {
    "wikipedia.org": 0.85,
    "github.com": 0.90,
    "arxiv.org": 0.90,
    "reuters.com": 0.95,
    "bbc.com": 0.90,
    "nytimes.com": 0.85,
    "stackoverflow.com": 0.80,
    "docs.python.org": 0.95,
    "developer.mozilla.org": 0.95,
    "medium.com": 0.55,
    "default": 0.40,
}

# ── M31: EWMA anomaly-detection parameters ──────────────────────────────
EWMA_ALPHA = 0.3
EWMA_ANOMALY_SIGMA = 2.0

# ── Threshold flags (used for scorecard warnings) ───────────────────────
# op is the COMPLIANT direction: ">" means values above warn/fail are OK.
METRIC_THRESHOLDS = {
    "m01_goal_completion_rate":      {"warn": 0.80, "fail": 0.50, "op": "<"},
    "m02_step_success_ratio":        {"warn": 0.85, "fail": 0.60, "op": "<"},
    "m03_calibration_gap":           {"warn": 0.20, "fail": 0.40, "op": ">"},
    "m05_tool_exec_success_rate":    {"warn": 0.90, "fail": 0.70, "op": "<"},
    "m06_tool_selection_accuracy":   {"warn": 0.85, "fail": 0.60, "op": "<"},
    "m07_parameter_f1_score":        {"warn": 0.80, "fail": 0.50, "op": "<"},
    "m08_tool_hallucination_rate":   {"warn": 0.10, "fail": 0.30, "op": ">"},
    "m09_shell_success_rate":        {"warn": 0.85, "fail": 0.60, "op": "<"},
    "m13_file_op_success_rate":      {"warn": 0.90, "fail": 0.70, "op": "<"},
    "m17_web_fetch_success_rate":    {"warn": 0.85, "fail": 0.60, "op": "<"},
    "m18_source_credibility_score":  {"warn": 0.60, "fail": 0.40, "op": "<"},
    "m19_grounding_rate":            {"warn": 0.60, "fail": 0.30, "op": "<"},
    "m21_mcp_invocation_success":    {"warn": 0.85, "fail": 0.60, "op": "<"},
    "m23_context_retention_score":   {"warn": 0.70, "fail": 0.40, "op": "<"},
    "m26_answer_faithfulness":       {"warn": 0.75, "fail": 0.50, "op": "<"},
    "m27_evidence_traceability_score": {"warn": 0.50, "fail": 0.20, "op": "<"},
    "m29_policy_adherence_score":    {"warn": 1.00, "fail": 0.99, "op": "<"},
    "m30_sla_latency_compliance":    {"warn": 0.90, "fail": 0.70, "op": "<"},
    "m31_budget_compliance":         {"warn": 1.00, "fail": 0.99, "op": "<"},
    "m33_run_consistency_score":     {"warn": 0.70, "fail": 0.40, "op": "<"},
}

# ── Critical metrics that BLOCK a run (used by the evaluator pass/fail) ──
CRITICAL_METRICS = {
    "m05_tool_exec_success_rate": ("<", 0.70),
    "m08_tool_hallucination_rate": (">", 0.30),
    "m29_policy_adherence_score":  ("<", 0.99),
    "m31_budget_compliance":       ("<", 0.99),
}
