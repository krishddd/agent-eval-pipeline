-- schema/tables.sql
-- Universal Agent Evaluation Pipeline — PostgreSQL + TimescaleDB DDL
-- Based on CLEAR (2025) & MultiAgentBench (ACL 2025)

-- ═══════════════════════════════════════════════════════════════════
-- Agent Registry
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS agent_cards (
    agent_id       UUID PRIMARY KEY,
    name           TEXT NOT NULL,
    agent_type     TEXT NOT NULL,
    framework      TEXT NOT NULL,
    model_backbone TEXT NOT NULL,
    memory_type    TEXT DEFAULT 'none',
    version        TEXT DEFAULT '1.0.0',
    card_json      JSONB NOT NULL,              -- Full AgentCard as JSON
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    updated_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_cards_type ON agent_cards(agent_type);
CREATE INDEX IF NOT EXISTS idx_agent_cards_framework ON agent_cards(framework);


-- ═══════════════════════════════════════════════════════════════════
-- Eval Reports — with artifact URL for large trajectory offloading
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS eval_reports (
    report_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id       UUID REFERENCES agent_cards(agent_id),
    agent_version  TEXT,
    overall_pass   BOOLEAN NOT NULL,
    metrics        JSONB NOT NULL,              -- Full EvalReport as JSON blob
    git_sha        TEXT,
    trigger        TEXT DEFAULT 'manual',       -- ci_cd | manual | scheduled
    artifact_url   TEXT,                        -- S3/GCS URL for large trajectories
    duration_ms    DOUBLE PRECISION,
    tasks_evaluated INTEGER,
    total_runs     INTEGER,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_reports_agent_time
    ON eval_reports (agent_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_eval_reports_git_sha
    ON eval_reports (git_sha);


-- ═══════════════════════════════════════════════════════════════════
-- TimescaleDB Hypertable — Per-metric time-series tracking
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS metric_timeseries (
    ts         TIMESTAMPTZ NOT NULL,
    agent_id   UUID NOT NULL,
    category   TEXT NOT NULL,
    metric     TEXT NOT NULL,
    value      DOUBLE PRECISION NOT NULL,
    run_id     UUID
);

-- Convert to hypertable (TimescaleDB extension required)
-- SELECT create_hypertable('metric_timeseries', 'ts', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_metric_ts_agent_metric
    ON metric_timeseries (agent_id, metric, ts DESC);


-- ═══════════════════════════════════════════════════════════════════
-- Drift Detection — 7-day rolling window vs historical baseline
-- ═══════════════════════════════════════════════════════════════════

CREATE OR REPLACE VIEW metric_drift AS
SELECT
    agent_id,
    metric,
    AVG(value) FILTER (WHERE ts >= NOW() - INTERVAL '7 days')  AS recent_avg,
    AVG(value) FILTER (WHERE ts  < NOW() - INTERVAL '7 days')  AS baseline_avg,
    AVG(value) FILTER (WHERE ts >= NOW() - INTERVAL '7 days')
      - AVG(value) FILTER (WHERE ts < NOW() - INTERVAL '7 days') AS drift,
    COUNT(*) FILTER (WHERE ts >= NOW() - INTERVAL '7 days')    AS recent_count,
    COUNT(*) FILTER (WHERE ts  < NOW() - INTERVAL '7 days')    AS baseline_count
FROM metric_timeseries
GROUP BY agent_id, metric
HAVING
    COUNT(*) FILTER (WHERE ts >= NOW() - INTERVAL '7 days') > 0
    AND COUNT(*) FILTER (WHERE ts < NOW() - INTERVAL '7 days') > 0;


-- ═══════════════════════════════════════════════════════════════════
-- Production Baselines — for lab-to-production gap tracking
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS production_baselines (
    id             SERIAL PRIMARY KEY,
    agent_id       UUID REFERENCES agent_cards(agent_id),
    metric         TEXT NOT NULL,
    baseline_value DOUBLE PRECISION NOT NULL,
    recorded_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_prod_baselines_agent
    ON production_baselines (agent_id, metric, recorded_at DESC);


-- ═══════════════════════════════════════════════════════════════════
-- Drift Alerts Log — tracks fired alerts for deduplication
-- ═══════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS drift_alerts (
    alert_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id       UUID REFERENCES agent_cards(agent_id),
    metric         TEXT NOT NULL,
    recent_avg     DOUBLE PRECISION,
    baseline_avg   DOUBLE PRECISION,
    drift_pct      DOUBLE PRECISION,
    alerted_at     TIMESTAMPTZ DEFAULT NOW(),
    resolved       BOOLEAN DEFAULT FALSE
);
