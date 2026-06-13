"""
scripts/register_odysseus_agent.py
Register the Odysseus autonomous agent with the eval dashboard.

By default it POSTs to a RUNNING dashboard (http://localhost:8000) so the agent
lands in the same in-process registry the eval endpoint uses.  Use --local to
register into the in-memory registry instead (handy for unit tests / dry runs).

Prereqs:
  • Odysseus running on http://127.0.0.1:7000 (Docker).
  • An API token minted in the Odysseus admin UI, exported as ODYSSEUS_TOKEN.

Usage:
  set ODYSSEUS_TOKEN=...           (PowerShell: $env:ODYSSEUS_TOKEN="...")
  python scripts/register_odysseus_agent.py
  python scripts/register_odysseus_agent.py --base-url http://127.0.0.1:7000 --mode auto
  python scripts/register_odysseus_agent.py --local
"""

from __future__ import annotations

import argparse
import os
import sys

# Force UTF-8 stdout so Unicode prints (e.g. the registry's "->" arrow) don't
# crash on legacy Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Odysseus tool surface (mirrors security_module/sample_configs/odysseus_agent.json)
TOOLS_MANIFEST = [
    {"name": "shell_exec", "description": "Execute a shell command", "parameters": {"cmd": "str"}},
    {"name": "agent_run", "description": "Run an autonomous agent task (shell+file+web+MCP)", "parameters": {"task": "str"}},
    {"name": "read_file", "description": "Read a file from the workspace", "parameters": {"path": "str"}},
    {"name": "write_file", "description": "Write content to a workspace file", "parameters": {"path": "str", "content": "str"}},
    {"name": "upload_file", "description": "Upload a file", "parameters": {"file": "binary"}},
    {"name": "web_search", "description": "Search the web", "parameters": {"query": "str"}},
    {"name": "web_fetch", "description": "Fetch a URL", "parameters": {"url": "str"}},
    {"name": "register_mcp_server", "description": "Register an MCP server", "parameters": {"url": "str"}},
    {"name": "create_skill", "description": "Create a custom skill", "parameters": {"name": "str"}},
    {"name": "write_memory", "description": "Persist a memory", "parameters": {"text": "str"}},
    {"name": "read_memory", "description": "Recall a memory", "parameters": {"query": "str"}},
    {"name": "sync_chat", "description": "Synchronous chat", "parameters": {"message": "str"}},
]


def build_request(base_url: str, token: str, mode: str) -> dict:
    auth_headers = {"Authorization": f"Bearer {token}"} if token else {}
    return {
        "name": "odysseus-local",
        "agent_type": "react",
        "framework": "odysseus",
        "model_backbone": "odysseus-default",
        "memory_type": "none",
        "tools_manifest": TOOLS_MANIFEST,
        "pass_k": 1,
        "max_cost_usd": 1.0,
        "sla_latency_ms": 120000,
        "tags": {"env": "local", "target": "odysseus", "port": "7000"},
        "remote_config": {
            "agent_kind": "odysseus",
            "base_url": base_url,
            "health_endpoint": "/api/health",
            "chat_endpoint": "/api/v1/chat",
            "task_field": "message",
            "agent_endpoint": "/api/agent/run",
            "agent_task_field": "task",
            "mode": mode,
            "auth_headers": auth_headers,
            "timeout_ms": 120000,
            "max_retries": 2,
        },
    }


def register_remote(dashboard: str, body: dict) -> int:
    import httpx

    # Pre-flight connectivity check
    tc = {
        "base_url": body["remote_config"]["base_url"],
        "health_endpoint": body["remote_config"]["health_endpoint"],
        "auth_headers": body["remote_config"]["auth_headers"],
    }
    with httpx.Client(timeout=30) as client:
        try:
            r = client.post(f"{dashboard}/agents/test-connection", json=tc)
            print(f"[test-connection] {r.status_code}: {r.json().get('message', r.text)[:200]}")
        except Exception as e:
            print(f"[test-connection] WARN: {e}")

        r = client.post(f"{dashboard}/agents/register", json=body)
        print(f"[register] {r.status_code}")
        try:
            data = r.json()
            print(f"  agent_id: {data.get('agent_id')}")
            print(f"  categories ({data.get('category_count')}): {data.get('eval_categories')}")
        except Exception:
            print(r.text[:500])
        return 0 if r.status_code < 400 else 1


def register_local(body: dict) -> int:
    from registry.agent_card import AgentCard, AgentType, MemoryType, ToolDef
    from registry.registry import registry

    card = AgentCard(
        name=body["name"],
        agent_type=AgentType(body["agent_type"]),
        framework=body["framework"],
        model_backbone=body["model_backbone"],
        memory_type=MemoryType(body["memory_type"]),
        tools_manifest=[ToolDef(**t) for t in body["tools_manifest"]],
        pass_k=body["pass_k"],
        max_cost_usd=body["max_cost_usd"],
        sla_latency_ms=body["sla_latency_ms"],
        tags=body["tags"],
        remote_config=body["remote_config"],
    )
    agent_id = registry.register(card)
    cats = card.eval_categories or card.auto_infer_categories()
    print(f"[local] registered {card.name} -> {agent_id}")
    print(f"[local] categories ({len(cats)}): {cats}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Register Odysseus with the eval pipeline")
    ap.add_argument("--base-url", default="http://127.0.0.1:7000")
    ap.add_argument("--dashboard", default="http://localhost:8000")
    ap.add_argument("--mode", default="auto", choices=["auto", "agent", "chat"])
    ap.add_argument("--token", default=os.getenv("ODYSSEUS_TOKEN", ""))
    ap.add_argument("--local", action="store_true", help="register into in-memory registry instead of the dashboard")
    args = ap.parse_args()

    if not args.token:
        print("WARN: ODYSSEUS_TOKEN not set — registering without auth (most endpoints will 401).")

    body = build_request(args.base_url, args.token, args.mode)
    return register_local(body) if args.local else register_remote(args.dashboard, body)


if __name__ == "__main__":
    raise SystemExit(main())
