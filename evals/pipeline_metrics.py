"""
evals/pipeline_metrics.py
Core computation logic for 33 pipeline-level metrics (M01–M33).

Each metric is implemented as a standalone function that takes the
PipelineResult dict (available via AgentResult.raw) and returns a
dict with the metric key → value.  The top-level `compute_all()`
function calls every metric and assembles the full JSON report.

This module has ZERO framework dependencies — it operates purely
on the JSON dict returned by the remote agent HTTP endpoint.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from evals.pipeline_metrics_config import (
    ABM_PATH_CALL_BUDGET,
    ABM_PATH_TIME_BUDGET_S,
    CLAIM_FAILURE_TYPES,
    CREDIBILITY_TIERS,
    DD_DIMENSIONS,
    EWMA_ALPHA,
    EWMA_ANOMALY_SIGMA,
    EXPECTED_FINANCIAL_SOURCES,
    KPI_PATTERNS,
    METRIC_THRESHOLDS,
    MOAT_PATTERNS,
    MODEL_STEP_MAP,
    REQUIRED_KPIS,
    SLA_BUDGETS,
    STEP_MODEL_MAP,
    TOOL_PARAM_SCHEMAS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helper utilities
# ═══════════════════════════════════════════════════════════════════════════

def _safe_float(val, default=0.0) -> float:
    """Convert a value to float safely."""
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _extract_domain(url: str) -> str:
    """Extract bare domain from a URL string."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().lstrip("www.")
        return domain
    except Exception:
        return ""


def _get_step_entry(step_log: list, agent_name: str) -> Optional[Dict]:
    """Find the first step_log entry matching an agent name."""
    for entry in step_log:
        if isinstance(entry, dict) and entry.get("agent") == agent_name:
            return entry
    return None


def _get_step_detail(step_log: list, agent_name: str) -> str:
    """Get the detail string from a step_log entry."""
    entry = _get_step_entry(step_log, agent_name)
    return entry.get("detail", "") if entry else ""


def _text_contains_any(text: str, patterns: list) -> bool:
    """Check if text contains any of the given patterns (case-insensitive)."""
    lower = text.lower()
    return any(p.lower() in lower for p in patterns)


def _count_chars_from_step_details(step_log: list) -> int:
    """Sum character counts from step detail fields like '4649 chars'."""
    total = 0
    for entry in step_log:
        if not isinstance(entry, dict):
            continue
        detail = entry.get("detail", "")
        m = re.search(r"(\d+)\s*chars", detail)
        if m:
            total += int(m.group(1))
    return total


def _parse_sentiment_label(text: str) -> str:
    """Extract BULLISH/BEARISH/NEUTRAL from sentiment text."""
    upper = text.upper()
    if "BULLISH" in upper:
        return "BULLISH"
    if "BEARISH" in upper:
        return "BEARISH"
    return "NEUTRAL"


def _sentiment_to_numeric(label: str) -> float:
    """Map sentiment label to numeric: BULLISH=+1, NEUTRAL=0, BEARISH=-1."""
    m = {"BULLISH": 1.0, "NEUTRAL": 0.0, "BEARISH": -1.0}
    return m.get(label.upper(), 0.0)


def _parse_sentiment_score(text: str) -> Optional[float]:
    """Extract numeric 'Score: X.X' from sentiment text."""
    m = re.search(r"Score:\s*([-\d.]+)", text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def _parse_sentiment_confidence(text: str) -> Optional[float]:
    """Extract confidence percentage from sentiment text."""
    m = re.search(r"Confidence:\s*(\d+)", text)
    if m:
        try:
            return int(m.group(1)) / 100.0
        except ValueError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: Pipeline-Level Cross-Step Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m01_calibration_gap(data: Dict) -> Dict:
    """M01 — Self-Score vs. Adversarial Calibration Gap."""
    quality = _safe_float(data.get("quality_score"), None)
    validation = data.get("validation_result") if isinstance(data.get("validation_result"), dict) else {}
    if not validation:
        # Try to extract from the raw pipeline data
        validation = {}

    survival_rate = _safe_float(validation.get("claim_survival_rate"), None)
    total_claims = validation.get("total_claims", 0)
    supported = validation.get("supported_claims", 0)

    # If survival_rate not directly available, compute from counts
    if survival_rate is None and total_claims and total_claims > 0:
        survival_rate = supported / total_claims

    result = {
        "m01_self_quality_score": quality,
        "m01_adversarial_survival_rate": survival_rate,
        "m01_total_claims": total_claims,
        "m01_supported_claims": supported,
    }

    if quality is not None and survival_rate is not None:
        gap = abs(quality / 10.0 - survival_rate) * 100
        result["m01_calibration_gap"] = round(gap, 2)
        result["m01_calibration_flag"] = gap > 30
    else:
        result["m01_calibration_gap"] = None
        result["m01_calibration_flag"] = None

    return result


def m02_sentiment_alignment(data: Dict) -> Dict:
    """M02 — Cross-Agent Sentiment Alignment Score."""
    sentiment_text = data.get("sentiment", "")
    abm_majority = data.get("abm_majority", "")

    llm_label = _parse_sentiment_label(sentiment_text) if sentiment_text else None
    abm_label = abm_majority.upper() if abm_majority else None

    result = {
        "m02_llm_sentiment": llm_label,
        "m02_abm_majority": abm_label,
    }

    if llm_label and abm_label:
        llm_score = _sentiment_to_numeric(llm_label)
        abm_score = _sentiment_to_numeric(abm_label)
        alignment = 1.0 - (abs(llm_score - abm_score) / 2.0)
        result["m02_sentiment_alignment_score"] = round(alignment, 4)
        result["m02_alignment_flag"] = alignment < 0.7
    else:
        result["m02_sentiment_alignment_score"] = None
        result["m02_alignment_flag"] = None

    return result


def m03_sla_compliance(data: Dict) -> Dict:
    """M03 — Step SLA Compliance Rate."""
    step_log = data.get("step_log", [])
    timings = data.get("timings", {})

    per_step = {}
    breaches = []

    for entry in step_log:
        if not isinstance(entry, dict):
            continue
        agent = entry.get("agent", "")
        duration = _safe_float(entry.get("duration_s"))
        budget = SLA_BUDGETS.get(agent)

        if budget is not None:
            breached = duration > budget
            per_step[agent] = {
                "actual_s": round(duration, 1),
                "budget_s": budget,
                "breached": breached,
            }
            if breached:
                breaches.append(agent)

    # Also check timings dict for steps not in step_log
    for key, duration in timings.items():
        # timings keys may be like "deep_research_TSLA" or "sentiment"
        base_agent = key.split("_")[0] if "_" in key else key
        # Try to find matching SLA budget
        matched_budget = None
        for sla_key in SLA_BUDGETS:
            if key.startswith(sla_key) or key == sla_key:
                matched_budget = SLA_BUDGETS[sla_key]
                if sla_key not in per_step:
                    breached = _safe_float(duration) > matched_budget
                    per_step[sla_key] = {
                        "actual_s": round(_safe_float(duration), 1),
                        "budget_s": matched_budget,
                        "breached": breached,
                    }
                    if breached and sla_key not in breaches:
                        breaches.append(sla_key)
                break

    total = len(per_step)
    within = sum(1 for v in per_step.values() if not v["breached"])

    return {
        "m03_sla_compliance_rate": round(within / total, 4) if total > 0 else None,
        "m03_total_steps_checked": total,
        "m03_breaches": breaches,
        "m03_breach_count": len(breaches),
        "m03_per_step_sla": per_step,
        "m03_sla_flag": "FAIL" if len(breaches) > 2 else ("WARN" if breaches else "PASS"),
    }


def m04_pipeline_throughput(data: Dict) -> Dict:
    """M04 — Pipeline Throughput Score (chars/sec)."""
    step_log = data.get("step_log", [])
    total_duration_ms = _safe_float(data.get("total_duration_ms", 0))
    total_duration_s = total_duration_ms / 1000.0 if total_duration_ms else 0

    # Sum output chars from step details
    total_chars = _count_chars_from_step_details(step_log)

    # If no chars found in details, estimate from previews
    if total_chars == 0:
        for field in ["research_preview", "financial_preview", "investment_report_preview",
                       "sentiment", "abm_report_preview"]:
            val = data.get(field, "")
            if val:
                total_chars += len(val)

    throughput = round(total_chars / total_duration_s, 2) if total_duration_s > 0 else None

    return {
        "m04_total_output_chars": total_chars,
        "m04_total_duration_s": round(total_duration_s, 2),
        "m04_pipeline_throughput_chars_per_sec": throughput,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: Research & News Step Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m05_source_credibility(data: Dict) -> Dict:
    """M05 — Source Credibility Score from research URLs."""
    # Try to extract URLs from research_preview or raw research output
    research = data.get("research_preview", "")
    # Also look in step_log detail or raw data
    all_text = research
    for entry in data.get("step_log", []):
        if isinstance(entry, dict) and entry.get("agent") in ("deep_research", "news"):
            all_text += " " + entry.get("detail", "")

    # Extract URLs from text
    urls = re.findall(r'https?://[^\s<>"]+', all_text)
    if not urls:
        # No URLs found — can't compute credibility
        return {
            "m05_source_credibility_score": None,
            "m05_sources_found": 0,
            "m05_source_domains": [],
        }

    scores = []
    domains = []
    for url in urls:
        domain = _extract_domain(url)
        if not domain:
            continue
        domains.append(domain)
        # Match domain against tiers
        score = CREDIBILITY_TIERS.get("default", 0.4)
        for tier_domain, tier_score in CREDIBILITY_TIERS.items():
            if tier_domain != "default" and tier_domain in domain:
                score = tier_score
                break
        scores.append(score)

    avg_score = statistics.mean(scores) if scores else None

    return {
        "m05_source_credibility_score": round(avg_score, 4) if avg_score is not None else None,
        "m05_sources_found": len(scores),
        "m05_source_domains": list(set(domains)),
        "m05_credibility_flag": avg_score is not None and avg_score < 0.65,
    }


def m06_information_freshness(data: Dict) -> Dict:
    """M06 — Information Freshness Score.
    Estimates source age from any dates found in research/news text.
    """
    now = datetime.now(timezone.utc)
    research = data.get("research_preview", "")
    # Use step outputs if available in previews
    all_text = research + " " + data.get("sentiment", "")

    # Extract dates like "2025-01-15", "January 15, 2025", "Q4 2025"
    date_patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{4}/\d{2}/\d{2})\b",
    ]
    dates = []
    for pattern in date_patterns:
        for match in re.findall(pattern, all_text):
            try:
                dt = datetime.strptime(match.replace("/", "-"), "%Y-%m-%d")
                dt = dt.replace(tzinfo=timezone.utc)
                dates.append(dt)
            except ValueError:
                pass

    if not dates:
        return {
            "m06_freshness_score_research": None,
            "m06_avg_source_age_days": None,
            "m06_dates_found": 0,
        }

    ages = [(now - d).days for d in dates if (now - d).days >= 0]
    avg_age = statistics.mean(ages) if ages else 0
    freshness = 1.0 / (1.0 + avg_age) if avg_age >= 0 else None

    return {
        "m06_freshness_score_research": round(freshness, 4) if freshness is not None else None,
        "m06_avg_source_age_days": round(avg_age, 1),
        "m06_dates_found": len(ages),
        "m06_freshness_flag": avg_age > 90,
    }


def m07_query_coverage(data: Dict) -> Dict:
    """M07 — Query Coverage Score for deep_research output."""
    research = data.get("research_preview", "")
    if not research:
        return {"m07_query_coverage_score": None, "m07_covered_dimensions": [], "m07_missing_dimensions": DD_DIMENSIONS[:]}

    # Use keyword matching to check coverage
    dimension_keywords = {
        "business_model": ["business model", "revenue stream", "how they make money", "revenue model"],
        "competitive_moat": ["moat", "competitive advantage", "barrier to entry", "differentiation"],
        "financials_summary": ["revenue", "profit", "margin", "earnings", "financial"],
        "management_quality": ["management", "ceo", "leadership", "executive", "board"],
        "risks": ["risk", "threat", "concern", "challenge", "vulnerability"],
        "recent_news": ["news", "recent", "latest", "announcement", "update"],
        "valuation": ["valuation", "p/e", "price to", "undervalued", "overvalued", "fair value"],
        "growth_outlook": ["growth", "outlook", "forecast", "future", "opportunity", "catalyst"],
    }

    covered = []
    missing = []
    lower_text = research.lower()
    for dim, keywords in dimension_keywords.items():
        if any(kw in lower_text for kw in keywords):
            covered.append(dim)
        else:
            missing.append(dim)

    score = len(covered) / len(DD_DIMENSIONS) if DD_DIMENSIONS else 0

    return {
        "m07_query_coverage_score": round(score, 4),
        "m07_covered_dimensions": covered,
        "m07_missing_dimensions": missing,
        "m07_coverage_flag": score < 0.75,
    }


def m08_claim_density(data: Dict) -> Dict:
    """M08 — Claim Density Score (verifiable claims per 1000 chars).
    Uses heuristic: counts sentences with numbers, proper nouns, or
    specific factual patterns as 'claims'.
    """
    results = {}
    fields = {
        "research": data.get("research_preview", ""),
        "synthesis": data.get("investment_report_preview", ""),
    }

    for label, text in fields.items():
        if not text:
            results[f"m08_claim_density_{label}"] = None
            results[f"m08_claim_count_{label}"] = None
            continue

        # Heuristic: a claim contains a number, dollar amount, percentage, or date
        sentences = re.split(r'[.!?\n]', text)
        claim_count = 0
        for s in sentences:
            s = s.strip()
            if len(s) < 15:
                continue
            # Sentence has quantitative info → likely a verifiable claim
            if re.search(r'\$[\d,.]+|[\d,.]+%|\b\d{4}\b|\b\d+\.\d+\b|\b\d+[BMK]\b', s):
                claim_count += 1

        char_count = len(text)
        density = (claim_count / (char_count / 1000.0)) if char_count > 0 else 0

        results[f"m08_claim_density_{label}"] = round(density, 2)
        results[f"m08_claim_count_{label}"] = claim_count

    return results


def m09_news_overlap(data: Dict) -> Dict:
    """M09 — News-to-Research Overlap Ratio.
    Uses simple word-level Jaccard similarity as a lightweight proxy
    for semantic overlap (no embedding model required).
    """
    research = data.get("research_preview", "")
    # News not directly in PipelineResult — use sentiment or step detail
    # The news output is not in the standard preview fields
    # We approximate using all available text
    news = ""
    for entry in data.get("step_log", []):
        if isinstance(entry, dict) and entry.get("agent") == "news":
            news = entry.get("detail", "")
            break

    if not research or not news or len(news) < 50:
        return {"m09_news_overlap_ratio": None}

    # Tokenize into word sets
    def _words(t):
        return set(re.findall(r'\b\w{3,}\b', t.lower()))

    r_words = _words(research)
    n_words = _words(news)

    if not n_words:
        return {"m09_news_overlap_ratio": None}

    overlap = len(r_words & n_words) / len(n_words)
    return {
        "m09_news_overlap_ratio": round(overlap, 4),
        "m09_overlap_flag": overlap > 0.60,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: Financial & SEC Data Step Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m10_data_source_completeness(data: Dict) -> Dict:
    """M10 — Data Source Completeness Score for financial step."""
    financial = data.get("financial_preview", "")
    step_log = data.get("step_log", [])

    # Detect which sources contributed data
    all_text = financial
    for entry in step_log:
        if isinstance(entry, dict) and entry.get("agent") == "financial":
            all_text += " " + entry.get("detail", "")

    detected = []
    failed = []
    source_patterns = {
        "yfinance": ["financial metrics for", "current price", "market cap", "p/e"],
        "xbrl": ["xbrl", "sec xbrl", "companyfacts"],
        "alpha_vantage": ["alpha vantage", "quarterly eps", "annual eps"],
        "technicals": ["technical analysis", "rsi", "macd", "bollinger"],
    }

    lower = all_text.lower()
    for source, patterns in source_patterns.items():
        if any(p in lower for p in patterns):
            detected.append(source)
        else:
            failed.append(source)

    score = len(detected) / len(EXPECTED_FINANCIAL_SOURCES) if EXPECTED_FINANCIAL_SOURCES else 1.0

    return {
        "m10_data_source_completeness": round(score, 4),
        "m10_detected_sources": detected,
        "m10_failed_sources": failed,
        "m10_completeness_flag": score < 1.0,
    }


def m11_kpi_coverage(data: Dict) -> Dict:
    """M11 — Financial KPI Coverage Score."""
    financial = data.get("financial_preview", "")
    if not financial:
        return {"m11_kpi_coverage_score": None, "m11_found_kpis": [], "m11_missing_kpis": REQUIRED_KPIS[:]}

    lower = financial.lower()
    found = []
    missing = []

    for kpi, patterns in KPI_PATTERNS.items():
        if any(p in lower for p in patterns):
            found.append(kpi)
        else:
            missing.append(kpi)

    score = len(found) / len(REQUIRED_KPIS) if REQUIRED_KPIS else 0

    return {
        "m11_kpi_coverage_score": round(score, 4),
        "m11_found_kpis": found,
        "m11_missing_kpis": missing,
        "m11_kpi_flag": score < 0.80,
    }


def m12_xbrl_parse_rate(data: Dict) -> Dict:
    """M12 — XBRL Parse Success Rate.
    Estimates based on presence of key XBRL concepts in output.
    """
    financial = data.get("financial_preview", "")
    lower = financial.lower() if financial else ""

    xbrl_concepts = [
        "revenue", "net income", "eps", "total assets", "total liabilities",
        "stockholders equity", "operating income", "cash & equivalents",
        "long-term debt",
    ]

    if "xbrl" not in lower:
        return {"m12_xbrl_parse_success_rate": None, "m12_xbrl_fields_found": 0}

    found = sum(1 for c in xbrl_concepts if c.lower() in lower)
    rate = found / len(xbrl_concepts)

    return {
        "m12_xbrl_parse_success_rate": round(rate, 4),
        "m12_xbrl_fields_found": found,
        "m12_xbrl_total_expected": len(xbrl_concepts),
    }


def m13_risk_factors(data: Dict) -> Dict:
    """M13 — Risk Factor Count and Severity Distribution.
    Counts risk-related statements and categorizes by severity keywords.
    """
    # Look for sec_analysis in raw data or step detail
    sec_text = ""
    step_log = data.get("step_log", [])
    for entry in step_log:
        if isinstance(entry, dict) and entry.get("agent") == "sec_risk":
            sec_text = entry.get("detail", "")
            break

    # Also look in the PipelineResult raw output
    # sec_analysis is not directly in preview fields, so we look at available text
    all_text = sec_text + " " + data.get("financial_preview", "")

    # Count risk factor mentions
    risk_patterns = re.findall(r'(?i)\b(?:risk|threat|concern|vulnerability|exposure|liability)\b', all_text)
    risk_count = len(risk_patterns)

    # Severity distribution heuristic
    high_kw = ["critical", "major", "severe", "material", "significant", "high risk"]
    medium_kw = ["moderate", "medium", "notable", "potential"]
    low_kw = ["minor", "low", "limited", "minimal"]

    lower = all_text.lower()
    severity = {
        "high": sum(1 for kw in high_kw if kw in lower),
        "medium": sum(1 for kw in medium_kw if kw in lower),
        "low": sum(1 for kw in low_kw if kw in lower),
    }

    return {
        "m13_risk_count": risk_count,
        "m13_severity_distribution": severity,
        "m13_risk_flag": risk_count < 3,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: Sentiment & Analysis Step Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m14_sentiment_confidence(data: Dict) -> Dict:
    """M14 — Sentiment Confidence Score."""
    sentiment_text = data.get("sentiment", "")
    confidence = _parse_sentiment_confidence(sentiment_text)
    label = _parse_sentiment_label(sentiment_text)

    # Detect mixed signals
    mixed_signals = []
    lower = sentiment_text.lower()
    if "downgrade" in lower:
        mixed_signals.append("analyst_downgrade")
    if "bearish" in lower and label == "BULLISH":
        mixed_signals.append("contradictory_bearish_note")
    if "risk" in lower or "concern" in lower:
        mixed_signals.append("risk_factors_mentioned")

    return {
        "m14_sentiment_label": label,
        "m14_sentiment_confidence": round(confidence, 4) if confidence is not None else None,
        "m14_mixed_signals": mixed_signals,
        "m14_confidence_flag": confidence is not None and confidence < 0.65,
    }


def m15_sentiment_polarity(data: Dict) -> Dict:
    """M15 — Sentiment Polarity Score (Continuous) + ABM gap."""
    sentiment_text = data.get("sentiment", "")
    polarity = _parse_sentiment_score(sentiment_text)
    label = _parse_sentiment_label(sentiment_text)

    # Fallback: if no numeric score parsed, use label-based mapping
    if polarity is None:
        polarity = _sentiment_to_numeric(label)

    # ABM average from path data
    abm_sentiment = data.get("abm_sentiment", {})
    abm_agreement = _safe_float(data.get("abm_agreement", 0))
    abm_majority = data.get("abm_majority", "")

    abm_avg = None
    if abm_majority:
        abm_avg = _sentiment_to_numeric(abm_majority)
        # Weight by agreement level
        abm_avg = abm_avg * abm_agreement if abm_agreement else abm_avg

    gap = abs(polarity - abm_avg) if polarity is not None and abm_avg is not None else None

    return {
        "m15_llm_polarity_score": round(polarity, 4) if polarity is not None else None,
        "m15_abm_avg_score": round(abm_avg, 4) if abm_avg is not None else None,
        "m15_sentiment_abm_gap": round(gap, 4) if gap is not None else None,
        "m15_gap_flag": gap is not None and gap > 0.40,
    }


def m16_marketing_depth(data: Dict) -> Dict:
    """M16 — Marketing Output Depth Score."""
    # Marketing output is not in standard preview fields
    # Check step_log for marketing agent detail
    step_log = data.get("step_log", [])
    marketing_chars = 0
    for entry in step_log:
        if isinstance(entry, dict) and entry.get("agent") == "marketing":
            detail = entry.get("detail", "")
            m = re.search(r"(\d+)\s*chars", detail)
            if m:
                marketing_chars = int(m.group(1))
            break

    # Use research_preview as proxy text for dimension checking
    # In reality, marketing output would need to be passed through
    text = data.get("research_preview", "")  # best available proxy

    covered = []
    missing = []
    lower = text.lower()
    for dim, patterns in MOAT_PATTERNS.items():
        if any(p in lower for p in patterns):
            covered.append(dim)
        else:
            missing.append(dim)

    score = len(covered) / len(MOAT_PATTERNS) if MOAT_PATTERNS else 0

    # Check for length anomaly
    avg_chars = 4000  # approximate pipeline step average
    length_anomaly = marketing_chars > 0 and marketing_chars < avg_chars * 0.5

    return {
        "m16_marketing_depth_score": round(score, 4),
        "m16_covered_moat_dimensions": covered,
        "m16_missing_moat_dimensions": missing,
        "m16_marketing_chars": marketing_chars,
        "m16_length_anomaly": length_anomaly,
        "m16_depth_flag": score < 0.67,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: ABM Simulation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m17_mc_path_variance(data: Dict) -> Dict:
    """M17 — Monte Carlo Path Variance."""
    abm_paths = data.get("abm_paths", 0)
    path_variance = _safe_float(data.get("abm_path_variance", None), None)
    confidence_interval = data.get("abm_confidence_interval", {})

    # If path_variance available from ABM analytics
    result = {
        "m17_mc_paths": abm_paths,
        "m17_mc_path_variance": round(path_variance, 4) if path_variance is not None else None,
        "m17_confidence_interval": confidence_interval,
    }

    if path_variance is not None:
        if path_variance < 0.05:
            result["m17_variance_interpretation"] = "stable"
        elif path_variance < 0.15:
            result["m17_variance_interpretation"] = "moderate_uncertainty"
        else:
            result["m17_variance_interpretation"] = "high_noise"
        result["m17_variance_flag"] = path_variance > 0.15
    else:
        result["m17_variance_interpretation"] = None
        result["m17_variance_flag"] = None

    return result


def m18_catalyst_impact(data: Dict) -> Dict:
    """M18 — Catalyst Impact Score (per type)."""
    catalyst_shocks = data.get("abm_catalyst_shocks", [])
    if not catalyst_shocks:
        return {"m18_catalyst_impact_scores": {}, "m18_catalysts_found": 0}

    impacts_by_type = {}
    for shock in catalyst_shocks:
        if not isinstance(shock, dict):
            continue
        event_type = shock.get("event_type", "unknown")
        magnitude = _safe_float(shock.get("magnitude", 0))
        if event_type not in impacts_by_type:
            impacts_by_type[event_type] = []
        impacts_by_type[event_type].append(magnitude)

    avg_impacts = {}
    for ct, magnitudes in impacts_by_type.items():
        avg_impacts[ct] = round(statistics.mean(magnitudes), 4)

    return {
        "m18_catalyst_impact_scores": avg_impacts,
        "m18_catalysts_found": len(catalyst_shocks),
        "m18_catalyst_types": list(impacts_by_type.keys()),
    }


def m19_agent_consensus(data: Dict) -> Dict:
    """M19 — Agent Consensus Score from ABM."""
    abm_agreement = _safe_float(data.get("abm_agreement", None), None)
    abm_total_agents = data.get("abm_total_agents", 0)
    abm_sentiment = data.get("abm_sentiment", {})

    # Agreement is already a consensus metric (fraction of paths that agree)
    if abm_agreement is not None:
        consensus = abm_agreement
        if consensus > 0.80:
            interpretation = "strong_convergence"
        elif consensus > 0.70:
            interpretation = "moderate_convergence"
        else:
            interpretation = "polarised"
    else:
        consensus = None
        interpretation = None

    return {
        "m19_agent_consensus_score": round(consensus, 4) if consensus is not None else None,
        "m19_total_agents": abm_total_agents,
        "m19_sentiment_distribution": abm_sentiment,
        "m19_consensus_interpretation": interpretation,
        "m19_consensus_flag": consensus is not None and consensus < 0.70,
    }


def m20_contagion_propagation(data: Dict) -> Dict:
    """M20 — Contagion Propagation Rate."""
    contagion_events = data.get("abm_contagion_events", [])
    total_agents = data.get("abm_total_agents", 50)

    if not contagion_events:
        return {
            "m20_contagion_propagation_rate": None,
            "m20_contagion_events_count": 0,
        }

    rates = []
    for event in contagion_events:
        if not isinstance(event, dict):
            continue
        # Estimate propagation from shift magnitude
        shift = _safe_float(event.get("shift", event.get("magnitude", 0)))
        # Approximate agents shifted as proportion of total
        if total_agents > 0:
            rate = min(1.0, abs(shift) * total_agents / total_agents)
            rates.append(rate)

    avg_rate = statistics.mean(rates) if rates else None

    return {
        "m20_contagion_propagation_rate": round(avg_rate, 4) if avg_rate is not None else None,
        "m20_contagion_events_count": len(contagion_events),
        "m20_propagation_flag": avg_rate is not None and avg_rate > 0.30,
    }


def m21_kol_influence(data: Dict) -> Dict:
    """M21 — KOL Influence Amplification Score.
    Computed from ABM coalition data if available.
    """
    coalitions = data.get("abm_coalitions", [])
    abm_sentiment = data.get("abm_sentiment", {})

    if not coalitions and not abm_sentiment:
        return {"m21_kol_amplification_score": None}

    # If coalition data available, look for KOL vs rule-based differential
    kol_score = None
    for coalition in coalitions:
        if isinstance(coalition, dict):
            # Coalition data might contain type-level info
            if "kol" in str(coalition).lower():
                kol_score = _safe_float(coalition.get("influence", coalition.get("score")), None)
                break

    return {
        "m21_kol_amplification_score": round(kol_score, 4) if kol_score is not None else None,
        "m21_coalitions_found": len(coalitions),
    }


def m22_abm_compute_efficiency(data: Dict) -> Dict:
    """M22 — ABM Compute Efficiency Ratio."""
    timings = data.get("timings", {})
    abm_duration = _safe_float(timings.get("abm_simulation", 0))
    abm_paths = data.get("abm_paths", 0)

    if abm_paths <= 0 or abm_duration <= 0:
        return {"m22_abm_compute_efficiency": None}

    per_path_duration = abm_duration / abm_paths
    time_efficiency = ABM_PATH_TIME_BUDGET_S / per_path_duration if per_path_duration > 0 else 0

    return {
        "m22_abm_compute_efficiency": round(time_efficiency, 4),
        "m22_abm_total_duration_s": round(abm_duration, 1),
        "m22_abm_per_path_duration_s": round(per_path_duration, 1),
        "m22_abm_paths": abm_paths,
        "m22_efficiency_flag": time_efficiency < 1.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: Tool Use Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m23_tool_execution_success(data: Dict) -> Dict:
    """M23 — Tool Execution Success Rate from step_log."""
    step_log = data.get("step_log", [])

    total = 0
    success = 0
    per_tool = {}

    for entry in step_log:
        if not isinstance(entry, dict):
            continue
        agent = entry.get("agent", "unknown")
        status = str(entry.get("status", "")).upper()
        is_success = status in ("OK", "DONE", "COMPLETED", "SUCCESS")
        total += 1
        if is_success:
            success += 1

        if agent not in per_tool:
            per_tool[agent] = {"total": 0, "success": 0}
        per_tool[agent]["total"] += 1
        if is_success:
            per_tool[agent]["success"] += 1

    rate = success / total if total > 0 else None

    return {
        "m23_tool_execution_success_rate": round(rate, 4) if rate is not None else None,
        "m23_total_tool_calls": total,
        "m23_successful_calls": success,
        "m23_per_tool_success": per_tool,
        "m23_success_flag": rate is not None and rate < 0.90,
    }


def m24_tool_hallucination_rate(data: Dict) -> Dict:
    """M24 — Tool Hallucination Rate.
    Compares stated financial figures in synthesis against data source outputs.
    This is an approximation — full implementation requires ground-truth API logs.
    """
    synthesis = data.get("investment_report_preview", "")
    financial = data.get("financial_preview", "")

    if not synthesis or not financial:
        return {"m24_tool_hallucination_rate": None}

    # Extract dollar amounts from both texts
    def _extract_numbers(text):
        nums = {}
        # Match patterns like $123B, $45.6M, $1,234
        for m in re.finditer(r'\$[\d,.]+[BMK]?', text):
            nums[m.group()] = True
        return set(nums.keys())

    synth_nums = _extract_numbers(synthesis)
    fin_nums = _extract_numbers(financial)

    if not synth_nums:
        return {"m24_tool_hallucination_rate": 0.0, "m24_synth_numbers": 0, "m24_matched_numbers": 0}

    matched = len(synth_nums & fin_nums)
    hallucination_rate = 1.0 - (matched / len(synth_nums)) if synth_nums else 0.0

    return {
        "m24_tool_hallucination_rate": round(hallucination_rate, 4),
        "m24_synth_numbers": len(synth_nums),
        "m24_matched_numbers": matched,
        "m24_hallucination_flag": hallucination_rate > 0.20,
    }


def m25_parameter_f1(data: Dict) -> Dict:
    """M25 — Parameter F1 Score per tool per step.
    Evaluated from step_log structure — checks if required fields are present.
    """
    step_log = data.get("step_log", [])
    f1_scores = []

    for entry in step_log:
        if not isinstance(entry, dict):
            continue
        agent = entry.get("agent", "")
        # Each step_log entry has: step, total, agent, status, duration_s, detail
        required_fields = {"step", "total", "agent", "status", "duration_s"}
        provided_fields = set(entry.keys())

        correct = required_fields & provided_fields
        precision = len(correct) / len(provided_fields) if provided_fields else 0
        recall = len(correct) / len(required_fields) if required_fields else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    avg_f1 = statistics.mean(f1_scores) if f1_scores else None

    return {
        "m25_parameter_f1_score": round(avg_f1, 4) if avg_f1 is not None else None,
        "m25_steps_evaluated": len(f1_scores),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: Adversarial Validation Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m26_claim_failure_categorisation(data: Dict) -> Dict:
    """M26 — Per-Claim Failure Categorisation."""
    validation = data.get("validation_result", {})
    if not isinstance(validation, dict):
        return {"m26_failure_type_distribution": None, "m26_flagged_claims": []}

    flagged = validation.get("flagged_claims", [])
    total_claims = validation.get("total_claims", 0)

    # Categorize each flagged claim
    distribution = {t: 0 for t in CLAIM_FAILURE_TYPES}
    categorized_claims = []

    for claim in flagged:
        if not isinstance(claim, dict):
            continue
        reason = str(claim.get("reason", "")).lower()
        verdict = str(claim.get("verdict", "")).lower()

        # Classify failure type
        if "hallucin" in reason or "fabricat" in reason:
            ftype = "hallucinated"
        elif "contradict" in reason or "opposite" in reason:
            ftype = "contradicted"
        elif "no evidence" in reason or "unsupported" in reason or "not found" in reason:
            ftype = "no_evidence"
        else:
            ftype = "unverifiable"

        distribution[ftype] += 1
        categorized_claims.append({
            "claim": claim.get("claim", ""),
            "verdict": verdict,
            "failure_type": ftype,
            "reason": claim.get("reason", ""),
        })

    return {
        "m26_failure_type_distribution": distribution,
        "m26_total_flagged": len(flagged),
        "m26_total_claims": total_claims,
        "m26_categorized_claims": categorized_claims,
    }


def m27_evidence_traceability(data: Dict) -> Dict:
    """M27 — Evidence Traceability Score."""
    synthesis = data.get("investment_report_preview", "")
    validation = data.get("validation_result", {})

    if not isinstance(validation, dict) or not synthesis:
        return {"m27_evidence_traceability_score": None}

    total_claims = validation.get("total_claims", 0)
    supported = validation.get("supported_claims", 0)

    if total_claims > 0:
        score = supported / total_claims
    else:
        score = None

    return {
        "m27_evidence_traceability_score": round(score, 4) if score is not None else None,
        "m27_total_claims": total_claims,
        "m27_traceable_claims": supported,
        "m27_traceability_flag": score is not None and score < 0.5,
    }


def m28_fallback_synthesis_quality(data: Dict) -> Dict:
    """M28 — Fallback Synthesis Quality Score."""
    validation = data.get("validation_result", {})
    if not isinstance(validation, dict):
        return {"m28_fallback_quality_score": None}

    should_replace = validation.get("should_replace", False)
    survival = _safe_float(validation.get("claim_survival_rate", 1.0))
    quality = _safe_float(data.get("quality_score", 0))

    result = {
        "m28_replacement_triggered": should_replace,
        "m28_original_self_quality": quality,
        "m28_survival_rate": round(survival, 4),
    }

    if should_replace:
        corrected = validation.get("corrected_report", "")
        # Estimate quality of corrected report by length and structure
        has_sections = len(re.findall(r'##\s+\d', corrected)) if corrected else 0
        corrected_len = len(corrected)
        # Simple heuristic: length * section structure
        fallback_quality = min(10.0, (corrected_len / 1000) + has_sections)
        result["m28_fallback_quality_score"] = round(fallback_quality, 2)
        result["m28_fallback_length"] = corrected_len
        result["m28_quality_flag"] = fallback_quality < 5.0
    else:
        result["m28_fallback_quality_score"] = None
        result["m28_quality_flag"] = None

    return result


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: Memory & Context Metrics
# ═══════════════════════════════════════════════════════════════════════════

def m29_context_retention(data: Dict) -> Dict:
    """M29 — Cross-Step Context Retention Score."""
    synthesis = data.get("investment_report_preview", "")
    if not synthesis:
        return {"m29_context_retention_score": None}

    # Build ground-truth facts from pipeline data
    facts = {}
    sentiment_text = data.get("sentiment", "")
    if sentiment_text:
        facts["sentiment_label"] = _parse_sentiment_label(sentiment_text)

    abm_majority = data.get("abm_majority", "")
    if abm_majority:
        facts["abm_majority"] = abm_majority

    abm_paths = data.get("abm_paths", 0)
    if abm_paths:
        facts["abm_paths"] = str(abm_paths)

    # Check which facts are reflected in synthesis
    lower = synthesis.lower()
    retained = 0
    total = len(facts)

    for key, value in facts.items():
        if str(value).lower() in lower:
            retained += 1

    score = retained / total if total > 0 else None

    return {
        "m29_context_retention_score": round(score, 4) if score is not None else None,
        "m29_facts_checked": total,
        "m29_facts_retained": retained,
        "m29_retention_flag": score is not None and score < 0.70,
    }


def m30_kg_utilisation(data: Dict) -> Dict:
    """M30 — Knowledge Graph Utilisation Rate.
    Uses available ABM data to check entity coverage.
    """
    abm_report = data.get("abm_report_preview", "")
    synthesis = data.get("investment_report_preview", "")
    combined = (abm_report + " " + synthesis).lower()

    # Without direct KG access, we check for entity-like references
    # from the pipeline data
    company = data.get("intent", {})
    tickers = []
    if isinstance(company, dict):
        tickers = company.get("tickers", [])

    if not combined or not tickers:
        return {"m30_kg_utilisation_rate": None}

    # Check basic entity presence (tickers, common financial entities)
    entities_checked = list(set(tickers))
    entities_found = [t for t in entities_checked if t.lower() in combined]

    rate = len(entities_found) / len(entities_checked) if entities_checked else None

    return {
        "m30_kg_utilisation_rate": round(rate, 4) if rate is not None else None,
        "m30_entities_checked": len(entities_checked),
        "m30_entities_found": len(entities_found),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 9: Observability & Monitoring Metrics
# ═══════════════════════════════════════════════════════════════════════════

class EWMAMonitor:
    """M31 — Anomaly detection using EWMA (Exponentially Weighted Moving Average)."""

    def __init__(self, alpha: float = EWMA_ALPHA, sigma_threshold: float = EWMA_ANOMALY_SIGMA):
        self.alpha = alpha
        self.sigma_threshold = sigma_threshold
        self.state: Dict[str, Dict[str, float]] = {}

    def check(self, metric_name: str, value: float) -> Tuple[bool, float]:
        """Check if value is anomalous. Returns (is_anomaly, detection_latency_ms)."""
        t0 = time.time()

        if metric_name not in self.state:
            self.state[metric_name] = {"mean": value, "var": 0.0}
            latency = (time.time() - t0) * 1000
            return False, latency

        prev_mean = self.state[metric_name]["mean"]
        prev_var = self.state[metric_name]["var"]

        new_mean = self.alpha * value + (1 - self.alpha) * prev_mean
        new_var = self.alpha * (value - prev_mean) ** 2 + (1 - self.alpha) * prev_var

        sigma = math.sqrt(new_var) if new_var > 0 else 0
        is_anomaly = sigma > 0 and abs(value - new_mean) > self.sigma_threshold * sigma

        self.state[metric_name] = {"mean": new_mean, "var": new_var}
        latency = (time.time() - t0) * 1000

        return is_anomaly, latency


def m31_anomaly_detection(data: Dict, monitor: Optional[EWMAMonitor] = None) -> Dict:
    """M31 — Anomaly Detection Latency."""
    if monitor is None:
        monitor = EWMAMonitor()

    # Check key metrics for anomalies
    metrics_to_check = {
        "quality_score": _safe_float(data.get("quality_score")),
        "total_duration_ms": _safe_float(data.get("total_duration_ms")),
    }

    # Add step durations from timings
    for key, val in data.get("timings", {}).items():
        metrics_to_check[f"timing_{key}"] = _safe_float(val)

    anomalies = []
    total_latency = 0

    for name, value in metrics_to_check.items():
        if value is None or value == 0:
            continue
        is_anomaly, latency = monitor.check(name, value)
        total_latency += latency
        if is_anomaly:
            anomalies.append({"metric": name, "value": value, "detection_latency_ms": round(latency, 3)})

    return {
        "m31_anomalies_detected": len(anomalies),
        "m31_anomaly_flags": anomalies,
        "m31_total_detection_latency_ms": round(total_latency, 3),
    }


def m32_judge_reliability(data: Dict) -> Dict:
    """M32 — LLM-as-Judge Reliability Score.
    Based on adversarial validation consistency signals.
    """
    validation = data.get("validation_result", {})
    if not isinstance(validation, dict):
        return {"m32_judge_reliability_score": None}

    total_claims = validation.get("total_claims", 0)
    flagged = validation.get("flagged_claims", [])
    survival = _safe_float(validation.get("claim_survival_rate", None), None)

    # Reliability heuristic: if all claims fail (0%) or all pass (100%),
    # judge may be unreliable (either too harsh or too lenient)
    if total_claims == 0:
        reliability = None
    elif survival == 0.0 and total_claims > 3:
        # 0% survival with many claims → possible false negatives
        reliability = 0.3
    elif survival == 1.0 and total_claims > 3:
        # 100% survival → possibly too lenient
        reliability = 0.5
    else:
        # Normal range → higher reliability
        reliability = min(1.0, 0.6 + survival * 0.4)

    return {
        "m32_judge_reliability_score": round(reliability, 4) if reliability is not None else None,
        "m32_total_claims_judged": total_claims,
        "m32_reliability_flag": reliability is not None and reliability < 0.70,
        "m32_reliability_note": (
            "adversarial validation may be producing false negatives"
            if reliability is not None and reliability < 0.70
            else None
        ),
    }


def m33_per_model_attribution(data: Dict) -> Dict:
    """M33 — Per-Model Performance Attribution."""
    step_log = data.get("step_log", [])
    timings = data.get("timings", {})

    model_stats = {}
    for model, steps in MODEL_STEP_MAP.items():
        model_stats[model] = {
            "steps": steps,
            "total_duration_s": 0,
            "total_chars": 0,
            "step_count": 0,
        }

    for entry in step_log:
        if not isinstance(entry, dict):
            continue
        agent = entry.get("agent", "")
        model = STEP_MODEL_MAP.get(agent)
        if not model:
            continue

        duration = _safe_float(entry.get("duration_s"))
        detail = entry.get("detail", "")
        chars = 0
        m = re.search(r"(\d+)\s*chars", detail)
        if m:
            chars = int(m.group(1))

        model_stats[model]["total_duration_s"] += duration
        model_stats[model]["total_chars"] += chars
        model_stats[model]["step_count"] += 1

    # Compute per-model chars/sec
    for model, stats in model_stats.items():
        dur = stats["total_duration_s"]
        chars = stats["total_chars"]
        stats["avg_chars_per_second"] = round(chars / dur, 2) if dur > 0 else None

    return {
        "m33_per_model_performance": model_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Master aggregation
# ═══════════════════════════════════════════════════════════════════════════

def compute_all(pipeline_data: Dict, ewma_monitor: Optional[EWMAMonitor] = None) -> Dict:
    """
    Compute all 33 metrics from a PipelineResult dict.

    Args:
        pipeline_data: The full JSON response from the remote agent endpoint
                       (available as AgentResult.raw in the eval pipeline).
        ewma_monitor:  Optional shared EWMA monitor for anomaly detection
                       across multiple runs.

    Returns:
        Dict with all metric keys and their values, plus summary metadata.
    """
    if not isinstance(pipeline_data, dict):
        return {"error": "pipeline_data must be a dict", "metrics_computed": 0}

    all_metrics = {}
    warnings = []
    t0 = time.time()

    # Compute all metrics
    metric_funcs = [
        ("M01", m01_calibration_gap),
        ("M02", m02_sentiment_alignment),
        ("M03", m03_sla_compliance),
        ("M04", m04_pipeline_throughput),
        ("M05", m05_source_credibility),
        ("M06", m06_information_freshness),
        ("M07", m07_query_coverage),
        ("M08", m08_claim_density),
        ("M09", m09_news_overlap),
        ("M10", m10_data_source_completeness),
        ("M11", m11_kpi_coverage),
        ("M12", m12_xbrl_parse_rate),
        ("M13", m13_risk_factors),
        ("M14", m14_sentiment_confidence),
        ("M15", m15_sentiment_polarity),
        ("M16", m16_marketing_depth),
        ("M17", m17_mc_path_variance),
        ("M18", m18_catalyst_impact),
        ("M19", m19_agent_consensus),
        ("M20", m20_contagion_propagation),
        ("M21", m21_kol_influence),
        ("M22", m22_abm_compute_efficiency),
        ("M23", m23_tool_execution_success),
        ("M24", m24_tool_hallucination_rate),
        ("M25", m25_parameter_f1),
        ("M26", m26_claim_failure_categorisation),
        ("M27", m27_evidence_traceability),
        ("M28", m28_fallback_synthesis_quality),
        ("M29", m29_context_retention),
        ("M30", m30_kg_utilisation),
        ("M32", m32_judge_reliability),
        ("M33", m33_per_model_attribution),
    ]

    for label, func in metric_funcs:
        try:
            result = func(pipeline_data)
            all_metrics.update(result)
        except Exception as e:
            warnings.append(f"{label}: computation failed — {e}")

    # M31 needs the monitor
    try:
        result = m31_anomaly_detection(pipeline_data, ewma_monitor)
        all_metrics.update(result)
    except Exception as e:
        warnings.append(f"M31: computation failed — {e}")

    # ── Generate threshold warnings ──────────────────────────────
    for metric_key, cfg in METRIC_THRESHOLDS.items():
        value = all_metrics.get(metric_key)
        if value is None:
            continue
        op = cfg["op"]
        warn_thresh = cfg["warn"]
        if op == ">" and value > warn_thresh:
            warnings.append(f"{metric_key} = {value} exceeds warn threshold {warn_thresh}")
        elif op == "<" and value < warn_thresh:
            warnings.append(f"{metric_key} = {value} below warn threshold {warn_thresh}")

    computation_ms = (time.time() - t0) * 1000

    all_metrics["_meta"] = {
        "metrics_computed": len(metric_funcs) + 1,  # +1 for M31
        "computation_ms": round(computation_ms, 2),
        "warnings": warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return all_metrics
