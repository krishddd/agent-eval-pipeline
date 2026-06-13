"""
storage/report_store.py
Async persistence for eval reports and metric time-series.

Supports:
- PostgreSQL + TimescaleDB (production)
- Local JSON file store (fallback when DB is offline)
- In-memory cache for fast API access

Reports are ALWAYS saved to local files (./reports/) so they
survive server restarts even without a database.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ReportStore:
    """
    Persists eval reports with dual storage:
    1. Local JSON files (always available, ./reports/)
    2. PostgreSQL (optional, for dashboard and drift detection)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        reports_dir: str = "./reports",
    ):
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://localhost:5432/agent_eval"
        )
        self._pool = None
        self._db_failed = False  # Cache DB failure to avoid repeated warnings
        self.reports_dir = reports_dir
        # Dedicated evals subfolder
        self.evals_dir = os.path.join(reports_dir, "evals")
        os.makedirs(self.evals_dir, exist_ok=True)
        # In-memory index: job_id → folder path
        self._index: Dict[str, str] = {}
        # Load existing reports into index on startup
        self._load_index()

    def _load_index(self):
        """Scan reports/evals/ directory and rebuild the in-memory index."""
        try:
            for entry in os.listdir(self.evals_dir):
                entry_path = os.path.join(self.evals_dir, entry)
                if os.path.isdir(entry_path):
                    report_file = os.path.join(entry_path, "report.json")
                    if os.path.exists(report_file):
                        self._index[entry] = entry_path
        except Exception:
            pass

    async def _get_pool(self):
        """Lazy-initialize the connection pool. Caches failure to avoid repeated warnings."""
        if self._pool is not None:
            return self._pool
        if self._db_failed:
            return None  # Already failed — don't retry or re-print warnings
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
            )
        except Exception as e:
            self._db_failed = True  # Cache the failure
            print(f"[ReportStore] Warning: DB connection failed: {e}")
            print("[ReportStore] Running in memory-only mode (local file store active)")
        return self._pool

    async def save_report(
        self,
        report,
        artifact_url: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Save an EvalReport to reports/evals/{job_id}/ folder.
        Creates structured files: report.json, scorecard.json, metrics.json, metadata.json.
        Returns the job_id.
        """
        report_data = report.model_dump(mode="json")
        report_id = job_id or str(uuid.uuid4())

        # ── Create dedicated eval folder ──────────────────────────
        eval_dir = os.path.join(self.evals_dir, report_id)
        os.makedirs(eval_dir, exist_ok=True)

        try:
            # 1. Full report
            with open(os.path.join(eval_dir, "report.json"), "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=str)

            # 2. Scorecard (pass/fail summary for quick reading)
            scorecard = {
                "agent_name": report_data.get("agent_name", "unknown"),
                "agent_version": report_data.get("agent_version", "1.0.0"),
                "overall_passed": report_data.get("overall_passed", False),
                "timestamp": report_data.get("timestamp"),
                "duration_ms": report_data.get("duration_ms"),
                "categories": [],
            }
            for result in report_data.get("results", []):
                scorecard["categories"].append({
                    "category": result.get("category"),
                    "passed": result.get("passed"),
                    "metrics": result.get("metrics", {}),
                    "warnings": result.get("warnings", []),
                })
            with open(os.path.join(eval_dir, "scorecard.json"), "w", encoding="utf-8") as f:
                json.dump(scorecard, f, indent=2, default=str)

            # 3. Flat metrics (for dashboards / time-series)
            flat_metrics = {}
            for result in report_data.get("results", []):
                cat = result.get("category", "unknown")
                for metric_name, value in result.get("metrics", {}).items():
                    flat_metrics[f"{cat}.{metric_name}"] = value
            with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(flat_metrics, f, indent=2, default=str)

            # 4. Odysseus metrics (M01–M33) — extracted from odysseus_metrics evaluator
            for result in report_data.get("results", []):
                if result.get("category") == "odysseus_metrics":
                    pm = result.get("details", {}).get("odysseus_metrics", {})
                    if pm:
                        with open(os.path.join(eval_dir, "odysseus_metrics.json"), "w", encoding="utf-8") as f:
                            json.dump(pm, f, indent=2, default=str)
                        print(f"[ReportStore]   ├── odysseus_metrics.json (M01–M33)")
                    break

            # 5. Metadata
            metadata = {
                "job_id": report_id,
                "agent_id": report_data.get("agent_id"),
                "agent_name": report_data.get("agent_name"),
                "git_sha": report_data.get("git_sha"),
                "trigger": report_data.get("trigger"),
                "artifact_url": artifact_url,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "tasks_evaluated": report_data.get("tasks_evaluated"),
                "total_runs": report_data.get("total_runs"),
            }
            with open(os.path.join(eval_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)

            self._index[report_id] = eval_dir
            print(f"[ReportStore] Saved eval report to {eval_dir}/")
            print(f"[ReportStore]   ├── report.json           (full report)")
            print(f"[ReportStore]   ├── scorecard.json        (pass/fail summary)")
            print(f"[ReportStore]   ├── metrics.json          (flat metrics)")
            print(f"[ReportStore]   ├── pipeline_metrics.json (M01-M33, if applicable)")
            print(f"[ReportStore]   └── metadata.json         (agent info)")

        except Exception as e:
            print(f"[ReportStore] Error saving to file: {e}")

        # ── Save to PostgreSQL if available ────────────────────────
        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO eval_reports
                            (agent_id, agent_version, overall_pass, metrics,
                             git_sha, trigger, artifact_url, duration_ms,
                             tasks_evaluated, total_runs)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                        """,
                        report.agent_id,
                        report.agent_version,
                        report.overall_passed,
                        json.dumps(report_data),
                        report.git_sha,
                        report.trigger,
                        artifact_url,
                        report.duration_ms,
                        report.tasks_evaluated,
                        report.total_runs,
                    )
            except Exception as e:
                print(f"[ReportStore] DB save failed (file store OK): {e}")

        return report_id

    async def get_report(self, report_id: str) -> Optional[Dict]:
        """
        Retrieve a report by ID.
        Tries: in-memory index → evals folder → PostgreSQL.
        """
        # Try in-memory index (points to eval folder)
        if report_id in self._index:
            report_file = os.path.join(self._index[report_id], "report.json")
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Try evals folder directly
        eval_dir = os.path.join(self.evals_dir, report_id)
        report_file = os.path.join(eval_dir, "report.json")
        if os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._index[report_id] = eval_dir
                    return data
            except Exception:
                pass

        # Bug fix: the DB schema has no job_id column — the previous code queried
        # `WHERE agent_id = $1` with the job UUID, which would never match.
        # The file store (above) is the correct source of truth for per-job lookup.
        # Skip the broken SQL path to avoid silently returning wrong data.
        # (To enable DB lookup by job_id, add a `job_id` column to eval_reports
        #  and include it in the INSERT in save_report().)

        return None

    async def get_scorecard(self, report_id: str) -> Optional[Dict]:
        """Retrieve the scorecard for a report."""
        eval_dir = self._index.get(report_id) or os.path.join(self.evals_dir, report_id)
        scorecard_file = os.path.join(eval_dir, "scorecard.json")
        if os.path.exists(scorecard_file):
            try:
                with open(scorecard_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    async def save_metrics(self, report) -> None:
        """
        Explode eval results into per-metric time-series rows
        in the metric_timeseries table.
        """
        pool = await self._get_pool()
        if not pool:
            return

        rows = []
        now = datetime.now(timezone.utc)

        for result in report.results:
            for metric_name, value in result.metrics.items():
                if value is not None:
                    rows.append((
                        now,
                        report.agent_id,
                        result.category,
                        metric_name,
                        float(value),
                    ))

        if rows:
            try:
                async with pool.acquire() as conn:
                    await conn.executemany(
                        """
                        INSERT INTO metric_timeseries (ts, agent_id, category, metric, value)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        rows,
                    )
            except Exception as e:
                print(f"[ReportStore] Error saving metrics: {e}")

    async def save_agent_card(self, card) -> None:
        """Persist an AgentCard to the agent_cards table."""
        pool = await self._get_pool()
        if not pool:
            return

        try:
            card_json = card.model_dump(mode="json")
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO agent_cards
                        (agent_id, name, agent_type, framework, model_backbone,
                         memory_type, version, card_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (agent_id) DO UPDATE SET
                        card_json = EXCLUDED.card_json,
                        version = EXCLUDED.version,
                        updated_at = NOW()
                    """,
                    card.agent_id,
                    card.name,
                    card.agent_type.value,
                    card.framework,
                    card.model_backbone,
                    card.memory_type.value,
                    card.version,
                    json.dumps(card_json),
                )
        except Exception as e:
            print(f"[ReportStore] Error saving agent card: {e}")

    async def get_drift_alerts(self, threshold: float = 0.10) -> List[Dict]:
        """Query the metric_drift view for metrics exceeding the threshold."""
        pool = await self._get_pool()
        if not pool:
            return []

        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT agent_id, metric, recent_avg, baseline_avg, drift
                    FROM metric_drift
                    WHERE ABS(drift) > $1
                    ORDER BY ABS(drift) DESC
                    """,
                    threshold,
                )
                return [dict(r) for r in rows]
        except Exception as e:
            print(f"[ReportStore] Error querying drift: {e}")
            return []

    async def get_reports(
        self,
        agent_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Retrieve recent eval reports.
        Tries PostgreSQL first, falls back to local file store.
        """
        # Try database first
        pool = await self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    if agent_id:
                        rows = await conn.fetch(
                            """
                            SELECT * FROM eval_reports
                            WHERE agent_id = $1
                            ORDER BY created_at DESC LIMIT $2
                            """,
                            agent_id, limit,
                        )
                    else:
                        rows = await conn.fetch(
                            "SELECT * FROM eval_reports ORDER BY created_at DESC LIMIT $1",
                            limit,
                        )
                    return [dict(r) for r in rows]
            except Exception as e:
                print(f"[ReportStore] DB query failed, using file store: {e}")

        # Fall back to file store
        # Bug fix: reports live in self.evals_dir/{job_id}/report.json, NOT in
        # self.reports_dir/*.json — the previous code scanned the wrong folder
        # and never found any reports.
        reports = []
        try:
            job_dirs = sorted(
                [
                    os.path.join(self.evals_dir, d)
                    for d in os.listdir(self.evals_dir)
                    if os.path.isdir(os.path.join(self.evals_dir, d))
                    and d != "by_agent"  # skip the by_agent index directory
                ],
                key=os.path.getmtime,
                reverse=True,
            )
            for job_dir in job_dirs:
                report_file = os.path.join(job_dir, "report.json")
                if not os.path.exists(report_file):
                    continue
                try:
                    with open(report_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if agent_id and data.get("agent_id") != agent_id:
                        continue
                    reports.append(data)
                    if len(reports) >= limit:
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"[ReportStore] Error reading file store: {e}")

        return reports

    async def close(self):
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
