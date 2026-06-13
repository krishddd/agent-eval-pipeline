"""
evals/_env.py
Zero-dependency .env loader (no python-dotenv requirement).

Reads KEY=VALUE lines from the project-root .env into os.environ WITHOUT
overwriting variables already set in the real environment (real env wins).
Called at import of the judge layer and the dashboard so OPENAI_API_KEY etc.
are available wherever the eval runs.
"""

from __future__ import annotations

import os

_LOADED = False


def load_env() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip('"').strip("'")
                if key and key not in os.environ:   # real env takes precedence
                    os.environ[key] = val
    except FileNotFoundError:
        pass
