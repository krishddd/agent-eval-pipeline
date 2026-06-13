"""
evals/pipeline_metrics_config.py
Configuration constants for the 33 pipeline-level metrics (M01–M33).

All thresholds, SLA budgets, credibility tiers, KPI lists, and
due-diligence dimensions are defined here so they can be tuned
independently of the computation logic.
"""

from __future__ import annotations

# ── M03: Per-step SLA budgets (seconds) ─────────────────────────────────
SLA_BUDGETS = {
    "deep_research": 60,
    "news": 30,
    "financial": 90,
    "sec_risk": 90,
    "marketing": 90,
    "sentiment": 45,
    "abm_simulation": 600,
    "abm_report": 120,
    "synthesize": 120,
    "adversarial_validation": 180,
    "save+email+sheet": 60,
    # Aliases (some steps log with different names)
    "save_email_sheet": 60,
}

# ── M05: Source credibility tiers ────────────────────────────────────────
CREDIBILITY_TIERS = {
    "sec.gov": 1.0,
    "reuters.com": 0.95,
    "bloomberg.com": 0.95,
    "wsj.com": 0.90,
    "ft.com": 0.90,
    "cnbc.com": 0.85,
    "marketwatch.com": 0.80,
    "yahoo.com": 0.75,
    "investopedia.com": 0.70,
    "fool.com": 0.65,
    "seekingalpha.com": 0.65,
    "stockanalysis.com": 0.60,
    "default": 0.40,
}

# ── M07: Due diligence dimensions ───────────────────────────────────────
DD_DIMENSIONS = [
    "business_model",
    "competitive_moat",
    "financials_summary",
    "management_quality",
    "risks",
    "recent_news",
    "valuation",
    "growth_outlook",
]

# ── M11: Financial KPI list ─────────────────────────────────────────────
REQUIRED_KPIS = [
    "pe_ratio",
    "eps",
    "revenue_growth",
    "gross_margin",
    "ebitda",
    "debt_to_equity",
    "free_cash_flow",
    "dividend_yield",
    "market_cap",
    "52_week_range",
    "rsi",
    "macd",
]

# KPI keyword patterns for detection in financial output
KPI_PATTERNS = {
    "pe_ratio": ["p/e", "pe ratio", "trailing pe", "forward pe", "trailingpe", "forwardpe"],
    "eps": ["eps", "earnings per share", "trailingeps", "forwardeps"],
    "revenue_growth": ["revenue growth", "revenue grew", "yoy revenue", "revenue increase"],
    "gross_margin": ["gross margin", "grossmargins"],
    "ebitda": ["ebitda"],
    "debt_to_equity": ["debt to equity", "debt/equity", "debttoequity", "d/e ratio"],
    "free_cash_flow": ["free cash flow", "fcf", "freecashflow"],
    "dividend_yield": ["dividend yield", "dividendyield"],
    "market_cap": ["market cap", "marketcap", "market capitalization"],
    "52_week_range": ["52 week", "52wk", "52-week", "52w high", "52w low"],
    "rsi": ["rsi", "relative strength index"],
    "macd": ["macd", "moving average convergence"],
}

# ── M16: Marketing moat dimensions ──────────────────────────────────────
MOAT_DIMENSIONS = [
    "pricing_power",
    "switching_costs",
    "network_effects",
    "scale_advantages",
    "intangible_assets",
    "competitive_threats",
]

MOAT_PATTERNS = {
    "pricing_power": ["pricing power", "price premium", "pricing advantage"],
    "switching_costs": ["switching cost", "lock-in", "customer retention", "sticky"],
    "network_effects": ["network effect", "network value", "platform effect", "ecosystem"],
    "scale_advantages": ["scale advantage", "economies of scale", "cost advantage", "market share"],
    "intangible_assets": ["brand", "patent", "intellectual property", "ip portfolio", "intangible"],
    "competitive_threats": ["threat", "competitor", "disrupt", "challenge", "risk"],
}

# ── M10: Expected financial data sources ────────────────────────────────
EXPECTED_FINANCIAL_SOURCES = ["yfinance", "xbrl", "alpha_vantage", "technicals"]

# ── M22: ABM compute budget per path ────────────────────────────────────
ABM_PATH_TIME_BUDGET_S = 84
ABM_PATH_CALL_BUDGET = 41

# ── M25: Tool parameter schemas ─────────────────────────────────────────
TOOL_PARAM_SCHEMAS = {
    "tavily_search": {"query", "max_results", "search_depth"},
    "tavily_news": {"query", "max_results"},
    "yfinance_ticker": {"ticker", "period", "interval"},
    "xbrl_fetch": {"cik", "form_type", "year"},
    "sec_filings": {"ticker"},
    "alpha_vantage_earnings": {"ticker"},
    "alpha_vantage_overview": {"ticker"},
    "technical_indicators": {"ticker"},
    "gmail_send": {"to", "subject", "body"},
    "sheets_append": {"spreadsheet_id", "data"},
}

# ── M26: Claim failure categories ───────────────────────────────────────
CLAIM_FAILURE_TYPES = ["no_evidence", "hallucinated", "contradicted", "unverifiable"]

# ── M31: EWMA parameters ────────────────────────────────────────────────
EWMA_ALPHA = 0.3
EWMA_ANOMALY_SIGMA = 2.0

# ── Threshold flags ─────────────────────────────────────────────────────
# Used to flag metrics in scorecard warnings
METRIC_THRESHOLDS = {
    "m01_calibration_gap": {"warn": 30, "fail": 60, "op": ">"},
    "m02_sentiment_alignment": {"warn": 0.7, "fail": 0.4, "op": "<"},
    "m03_sla_compliance_rate": {"warn": 0.9, "fail": 0.7, "op": "<"},
    "m05_source_credibility": {"warn": 0.65, "fail": 0.4, "op": "<"},
    "m06_freshness_research": {"warn": 0.5, "fail": 0.1, "op": "<"},
    "m07_query_coverage": {"warn": 0.75, "fail": 0.5, "op": "<"},
    "m08_claim_density": {"warn": 5.0, "fail": 2.0, "op": "<"},
    "m09_news_overlap_ratio": {"warn": 0.6, "fail": 0.8, "op": ">"},
    "m10_data_source_completeness": {"warn": 1.0, "fail": 0.5, "op": "<"},
    "m11_kpi_coverage": {"warn": 0.8, "fail": 0.5, "op": "<"},
    "m13_risk_count": {"warn": 3, "fail": 1, "op": "<"},
    "m14_sentiment_confidence": {"warn": 0.65, "fail": 0.4, "op": "<"},
    "m15_sentiment_abm_gap": {"warn": 0.4, "fail": 0.7, "op": ">"},
    "m16_marketing_depth": {"warn": 0.67, "fail": 0.33, "op": "<"},
    "m17_mc_path_variance": {"warn": 0.15, "fail": 0.3, "op": ">"},
    "m19_agent_consensus": {"warn": 0.7, "fail": 0.5, "op": "<"},
    "m23_tool_success_rate": {"warn": 0.9, "fail": 0.7, "op": "<"},
    "m24_tool_hallucination_rate": {"warn": 0.2, "fail": 0.5, "op": ">"},
    "m27_evidence_traceability": {"warn": 0.5, "fail": 0.2, "op": "<"},
    "m29_context_retention": {"warn": 0.7, "fail": 0.4, "op": "<"},
    "m32_judge_reliability": {"warn": 0.7, "fail": 0.4, "op": "<"},
}

# ── Model assignments (used for M33 attribution) ────────────────────────
MODEL_STEP_MAP = {
    "llama3.2:latest": ["deep_research", "news"],
    "qwen3:4b": ["financial", "sec_risk"],
    "qwen3:8b": ["marketing", "sentiment", "abm_report", "synthesize", "adversarial_validation"],
}

STEP_MODEL_MAP = {}
for model, steps in MODEL_STEP_MAP.items():
    for step in steps:
        STEP_MODEL_MAP[step] = model
