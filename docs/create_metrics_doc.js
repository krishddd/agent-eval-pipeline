const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  TableOfContents
} = require("docx");

// ── Color palette ───────────────────────────────────────────────
const C = {
  primary:    "1B3A5C",  // dark navy
  accent:     "2E75B6",  // blue
  success:    "28A745",  // green
  warning:    "F0AD4E",  // amber
  danger:     "DC3545",  // red
  lightBg:    "F0F5FA",  // very light blue
  headerBg:   "1B3A5C",  // dark navy for table headers
  headerText: "FFFFFF",  // white text on headers
  border:     "C0C8D0",  // light gray border
  muted:      "6C757D",  // muted gray text
};

// ── Helpers ─────────────────────────────────────────────────────
const border = { style: BorderStyle.SINGLE, size: 1, color: C.border };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function headerCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    margins: cellMargins,
    shading: { fill: C.headerBg, type: ShadingType.CLEAR },
    verticalAlign: "center",
    children: [new Paragraph({ alignment: AlignmentType.LEFT, children: [
      new TextRun({ text, bold: true, color: C.headerText, font: "Arial", size: 18 })
    ]})],
  });
}

function dataCell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    margins: cellMargins,
    shading: opts.shade ? { fill: C.lightBg, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ alignment: AlignmentType.LEFT, children: [
      new TextRun({ text: String(text), font: "Arial", size: 18, bold: opts.bold || false, color: opts.color || "000000" })
    ]})],
  });
}

function priorityCell(priority, width) {
  const colorMap = { CRITICAL: C.danger, HIGH: C.warning, MEDIUM: C.accent };
  return dataCell(priority, width, { bold: true, color: colorMap[priority] || "000000" });
}

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.spaceBefore || 80, after: opts.spaceAfter || 80 },
    alignment: opts.align || AlignmentType.LEFT,
    children: [new TextRun({
      text, font: "Arial", size: opts.size || 21, bold: opts.bold || false,
      italics: opts.italics || false, color: opts.color || "333333",
    })],
  });
}

function heading(text, level) {
  return new Paragraph({
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 360 : 240, after: 120 },
    children: [new TextRun({ text, font: "Arial", bold: true, color: C.primary,
      size: level === HeadingLevel.HEADING_1 ? 32 : level === HeadingLevel.HEADING_2 ? 26 : 22 })],
  });
}

function bullet(text, ref = "bullets", level = 0) {
  return new Paragraph({
    numbering: { reference: ref, level },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, font: "Arial", size: 20 })],
  });
}

function richBullet(runs, ref = "bullets", level = 0) {
  return new Paragraph({
    numbering: { reference: ref, level },
    spacing: { before: 40, after: 40 },
    children: runs,
  });
}

function boldLabel(label, value) {
  return [
    new TextRun({ text: label, font: "Arial", size: 20, bold: true }),
    new TextRun({ text: value, font: "Arial", size: 20 }),
  ];
}

function codeBlock(text) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    indent: { left: 360 },
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: "Consolas", size: 17, color: "333333" })],
  });
}

function divider() {
  return new Paragraph({
    spacing: { before: 200, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 3, color: C.accent, space: 1 } },
    children: [],
  });
}

// ── Metric definitions ──────────────────────────────────────────
const metrics = [
  // Section 1
  { id: "M01", name: "Self-Score vs. Adversarial Calibration Gap", section: 1, priority: "CRITICAL",
    step: "synthesize + adversarial_validation",
    what: "The absolute difference between the synthesize agent's self-reported quality score and the adversarial validation's claim survival rate, expressed as a percentage.",
    formula: "calibration_gap = abs(self_quality_score / 10 - adversarial_survival_rate) * 100",
    why: "A 10/10 self-score with 0% adversarial survival is the clearest signal of LLM overconfidence. The dashboard currently shows a passing quality score while the actual report was invalidated.",
    threshold: "Flag if gap > 30%",
  },
  { id: "M02", name: "Cross-Agent Sentiment Alignment Score", section: 1, priority: "CRITICAL",
    step: "sentiment + abm_simulation",
    what: "A 0-1 score measuring whether the LLM sentiment classifier and the ABM simulation majority outcome agree on directional market sentiment.",
    formula: "alignment_score = 1 - (abs(llm_score - abm_majority_score) / 2)",
    why: "BULLISH vs. neutral is a direct contradiction currently invisible in the pipeline. The synthesize agent receives conflicting inputs and has no way to know they conflict.",
    threshold: "Flag if alignment_score < 0.7",
  },
  { id: "M03", name: "Step SLA Compliance Rate", section: 1, priority: "HIGH",
    step: "All 11 steps",
    what: "The percentage of pipeline steps that complete within a predefined maximum latency budget per step.",
    formula: "sla_compliance_rate = steps_within_budget / total_steps",
    why: "abm_simulation used 68.6% of total pipeline time. No SLA threshold is tracked so this bottleneck is invisible to CI/CD gates.",
    threshold: "WARN if any step breaches, FAIL if > 2 steps breach",
  },
  { id: "M04", name: "Pipeline Throughput Score", section: 1, priority: "MEDIUM",
    step: "Pipeline level",
    what: "Total useful output characters generated per second of wall-clock pipeline time.",
    formula: "pipeline_throughput = total_output_chars / total_duration_seconds",
    why: "Comparing pipeline efficiency across model versions and prompt changes. Justifying compute cost of long-running steps.",
    threshold: "Track across runs for regression detection",
  },
  // Section 2
  { id: "M05", name: "Source Credibility Score", section: 2, priority: "HIGH",
    step: "deep_research",
    what: "Average domain authority score of the Tavily search sources, weighted by their contribution to the output.",
    formula: "source_credibility_score = weighted_average(scores, weights=char_contributions)",
    why: "Ensures reports are grounded in authoritative sources. Low credibility triggers re-search.",
    threshold: "Flag if score < 0.65",
  },
  { id: "M06", name: "Information Freshness Score", section: 2, priority: "HIGH",
    step: "deep_research + news",
    what: "Average age of retrieved sources in days, converted to a 0-1 freshness score.",
    formula: "freshness_score = 1 / (1 + avg_source_age_days)",
    why: "Detecting stale research that could lead to outdated investment conclusions.",
    threshold: "Flag if avg_age > 90 days",
  },
  { id: "M07", name: "Query Coverage Score", section: 2, priority: "HIGH",
    step: "deep_research",
    what: "Percentage of standard due diligence dimensions substantively addressed in the deep_research output.",
    formula: "query_coverage_score = covered_dimensions / total_dimensions",
    why: "Ensuring research does not miss critical DD components before downstream agents build on it.",
    threshold: "Flag if < 0.75",
  },
  { id: "M08", name: "Claim Density Score", section: 2, priority: "MEDIUM",
    step: "deep_research + synthesize",
    what: "Number of verifiable, specific factual claims per 1,000 output characters.",
    formula: "claim_density = claim_count / (char_count / 1000)",
    why: "Comparing output richness across model versions. Low claim density flags shallow analysis.",
    threshold: "deep_research >= 5 claims/1000 chars",
  },
  { id: "M09", name: "News-to-Research Overlap Ratio", section: 2, priority: "MEDIUM",
    step: "news",
    what: "Fraction of news digest content that duplicates information already in the deep_research output.",
    formula: "overlap_ratio = duplicate_sentences / total_news_sentences",
    why: "Optimizing pipeline efficiency. If consistently > 60% overlap, merge or replace the news step.",
    threshold: "Flag if overlap_ratio > 0.60",
  },
  // Section 3
  { id: "M10", name: "Data Source Completeness Score", section: 3, priority: "HIGH",
    step: "financial",
    what: "Fraction of expected financial data sources (yfinance, XBRL, Alpha Vantage, technicals) that successfully returned usable data.",
    formula: "data_source_completeness = successful_sources / expected_sources",
    why: "Detecting partial data failures that produce misleadingly complete-looking financial outputs.",
    threshold: "Flag if < 1.0 (any source failed)",
  },
  { id: "M11", name: "Financial KPI Coverage Score", section: 3, priority: "HIGH",
    step: "financial",
    what: "Percentage of standard equity due diligence KPIs (P/E, EPS, revenue growth, margins, EBITDA, etc.) present in the output.",
    formula: "kpi_coverage_score = found_kpis / required_kpis",
    why: "Ensuring investment recommendations are based on a complete financial picture.",
    threshold: "Flag if < 0.80",
  },
  { id: "M12", name: "XBRL Parse Success Rate", section: 3, priority: "MEDIUM",
    step: "financial + sec_risk",
    what: "Ratio of XBRL taxonomy fields successfully extracted to total fields expected from the target filing.",
    formula: "xbrl_parse_success_rate = successfully_parsed_fields / expected_fields",
    why: "Monitoring data pipeline health. XBRL failures are silent -- financial output can look complete while missing data.",
    threshold: "Flag if < 0.70",
  },
  { id: "M13", name: "Risk Factor Count and Severity Distribution", section: 3, priority: "HIGH",
    step: "sec_risk",
    what: "Total distinct risk factors extracted and their distribution across severity tiers (high/medium/low).",
    formula: "Categorical: {high: N, medium: N, low: N}",
    why: "Detecting material risk escalation between filings. Cross-checking if synthesize adequately addresses identified risks.",
    threshold: "Flag if risk_count < 3",
  },
  // Section 4
  { id: "M14", name: "Sentiment Confidence Score", section: 4, priority: "CRITICAL",
    step: "sentiment",
    what: "A 0-1 probability score attached to the sentiment label reflecting classifier certainty.",
    formula: "confidence = model_confidence / 10 (from LLM self-assessment)",
    why: "Gating the ABM simulation. Low confidence should trigger wider prior distribution in the ABM step.",
    threshold: "Flag if confidence < 0.65",
  },
  { id: "M15", name: "Sentiment Polarity Score (Continuous)", section: 4, priority: "HIGH",
    step: "sentiment",
    what: "Scalar value in [-1.0, +1.0] representing degree of bullishness, enabling numeric comparison with ABM output averages.",
    formula: "sentiment_abm_gap = abs(llm_polarity - abm_avg)",
    why: "Quantitative cross-validation between LLM sentiment and ABM simulation. Bridges categorical label vs. numeric simulation output.",
    threshold: "Flag if gap > 0.40",
  },
  { id: "M16", name: "Marketing Output Depth Score", section: 4, priority: "MEDIUM",
    step: "marketing",
    what: "Whether the brand and moat analysis covers a minimum set of required analytical dimensions.",
    formula: "marketing_depth_score = covered_dimensions / total_moat_dimensions",
    why: "Flagging under-analysis. 1900 chars over 49.6s may indicate prompt truncation or early stopping.",
    threshold: "Flag if < 0.67 (fewer than 4 of 6 dimensions)",
  },
  // Section 5
  { id: "M17", name: "Monte Carlo Path Variance", section: 5, priority: "HIGH",
    step: "abm_simulation",
    what: "Standard deviation of the final-tick average sentiment across all Monte Carlo paths. Measures simulation consistency.",
    formula: "mc_path_variance = std(final_tick_avgs)",
    why: "Quantifying simulation confidence. High variance = ABM outcome should carry lower weight in synthesis.",
    threshold: "std < 0.05 = stable; 0.05-0.15 = moderate; > 0.15 = high noise",
  },
  { id: "M18", name: "Catalyst Impact Score (per type)", section: 5, priority: "HIGH",
    step: "abm_simulation",
    what: "Average sentiment change at the tick immediately following each catalyst injection, computed per catalyst type.",
    formula: "catalyst_impact = mean(delta_avg_per_catalyst_type)",
    why: "Calibrating the ABM model. Validates that catalyst magnitudes produce proportional market responses.",
    threshold: "If realised impact < 0.05, agents are insensitive",
  },
  { id: "M19", name: "Agent Consensus Score", section: 5, priority: "HIGH",
    step: "abm_simulation",
    what: "Per-path score measuring how much the 50-agent population converged by simulation end.",
    formula: "consensus_score = 1 - std_at_final_tick",
    why: "Distinguishing stable neutral signal from divided market. Both can produce 'majority=neutral' but carry different risk implications.",
    threshold: "> 0.80 strong; 0.70-0.80 moderate; < 0.70 polarised",
  },
  { id: "M20", name: "Contagion Propagation Rate", section: 5, priority: "MEDIUM",
    step: "abm_simulation",
    what: "When a contagion event is detected, the fraction of agents that changed sentiment direction within 1 tick.",
    formula: "contagion_rate = agents_shifted / total_agents",
    why: "Quantifying viral sentiment dynamics. Makes contagion events measurable and comparable across runs.",
    threshold: "rate > 0.30 = social influence dominates",
  },
  { id: "M21", name: "KOL Influence Amplification Score", section: 5, priority: "MEDIUM",
    step: "abm_simulation",
    what: "Ratio of sentiment change attributable to the 5 KOL agents vs. the 35 rule-based agents.",
    formula: "kol_amplification = (kol_avg - rule_avg) / (abs(rule_avg) + 0.001)",
    why: "Empirically validating the 5 KOL + 10 LLM + 35 rule-based agent split. Guides agent count tuning.",
    threshold: "If amplification near 0: KOL mechanic adds no signal",
  },
  { id: "M22", name: "ABM Compute Efficiency Ratio", section: 5, priority: "HIGH",
    step: "abm_simulation",
    what: "Ratio of actual LLM calls and duration per path vs. the estimated budget (~41 calls, ~84s per path).",
    formula: "efficiency = budget_time / actual_time (ratio < 1.0 = over-budget)",
    why: "Monitoring ABM compute costs across runs. Enables ABM-specific SLA enforcement.",
    threshold: "Ratio < 1.0 = over-budget",
  },
  // Section 6
  { id: "M23", name: "Tool Execution Success Rate", section: 6, priority: "CRITICAL",
    step: "All tool-using steps",
    what: "Fraction of tool invocations that produce a valid, non-empty, parseable result.",
    formula: "tool_success_rate = successful_calls / total_calls",
    why: "Root-cause analysis for failing tool_use category. Foundation of all data quality. Fixes current invocation_accuracy = 0.0 problem.",
    threshold: "Flag if < 0.90",
  },
  { id: "M24", name: "Tool Hallucination Rate", section: 6, priority: "CRITICAL",
    step: "financial + sec_risk",
    what: "Fraction of tool invocations where the LLM fabricates financial figures rather than using real API responses.",
    formula: "hallucination_rate = 1 - mean(field_matches)",
    why: "Detecting cases where financial data was generated by LLM rather than retrieved from APIs. 0/6 adversarial claims survived -- possible cause is hallucinated financial figures.",
    threshold: "Flag if hallucination_rate > 0.20",
  },
  { id: "M25", name: "Parameter F1 Score (per tool, per step)", section: 6, priority: "CRITICAL",
    step: "All tool-using steps",
    what: "For each tool call: precision = correct_params / all_params_provided; recall = correct_params / all_params_required. F1 = harmonic mean.",
    formula: "parameter_f1 = 2 * P * R / (P + R)",
    why: "Fixes current parameter_f1 = 0.0 by logging actual tool calls at the HTTP adapter layer.",
    threshold: "Flag if < 0.80",
  },
  // Section 7
  { id: "M26", name: "Per-Claim Failure Categorisation", section: 7, priority: "CRITICAL",
    step: "adversarial_validation",
    what: "Typed breakdown of why each claim failed: no_evidence, hallucinated, contradicted, or unverifiable.",
    formula: "Categorical distribution: {no_evidence: N, hallucinated: N, contradicted: N, unverifiable: N}",
    why: "Diagnosing the 0% survival rate. Each failure type requires a different fix: hallucinated -> fix synthesis prompt; no_evidence -> fix research coverage.",
    threshold: "Route fix recommendations based on dominant failure type",
  },
  { id: "M27", name: "Evidence Traceability Score", section: 7, priority: "HIGH",
    step: "adversarial_validation",
    what: "Fraction of synthesize claims that can be traced back to a specific source from steps 1-4.",
    formula: "traceability = traceable_claims / total_claims",
    why: "Building an auditable due diligence pipeline. In regulated financial contexts, every claim must trace to a primary source.",
    threshold: "Score of 0.0 = confirmed hallucination mode",
  },
  { id: "M28", name: "Fallback Synthesis Quality Score", section: 7, priority: "HIGH",
    step: "adversarial_validation",
    what: "Quality score for the replacement synthesis produced when the original fails (survival = 0%).",
    formula: "LLM-as-judge quality assessment on replacement output",
    why: "Validating that the replacement mechanism produces superior outputs. Without this, replacement could silently degrade quality.",
    threshold: "If fallback_quality < 5.0, escalate for human review",
  },
  // Section 8
  { id: "M29", name: "Cross-Step Context Retention Score", section: 8, priority: "HIGH",
    step: "synthesize",
    what: "Fraction of key facts from steps 1-8 that are accurately reflected in the step 9 synthesis output.",
    formula: "context_retention = facts_retained / facts_checked",
    why: "Verifying synthesize actually integrated all 8 upstream outputs. The 0% adversarial survival suggests possible context window truncation.",
    threshold: "Flag if < 0.70",
  },
  { id: "M30", name: "Knowledge Graph Utilisation Rate", section: 8, priority: "MEDIUM",
    step: "abm_simulation -> abm_report -> synthesize",
    what: "Fraction of the knowledge graph nodes that are referenced in downstream step outputs.",
    formula: "kg_utilisation = nodes_referenced / total_nodes",
    why: "If < 0.20 consistently, remove KG generation from pipeline. If < 0.10 across 5 runs, flag KG as non-contributing.",
    threshold: "Flag if < 0.20",
  },
  // Section 9
  { id: "M31", name: "Anomaly Detection Latency", section: 9, priority: "HIGH",
    step: "Pipeline monitoring layer",
    what: "Time elapsed between when an anomalous metric value occurs and when the system flags it. Uses EWMA (Exponentially Weighted Moving Average) with 2-sigma threshold.",
    formula: "EWMA check: is_anomaly = abs(value - mean) > 2 * sigma",
    why: "Production pipeline monitoring. abm_simulation SLA breach and 0% survival are anomalies that currently proceed silently.",
    threshold: "Target detection latency < 5.6s (AMDM benchmark)",
  },
  { id: "M32", name: "LLM-as-Judge Reliability Score", section: 9, priority: "HIGH",
    step: "adversarial_validation",
    what: "Consistency of the LLM judge across repeated evaluations. Current report shows judge_reliability = 0.0.",
    formula: "Consistency-based: 3 repeat evaluations, check agreement",
    why: "Validating that the adversarial validator itself is trustworthy before trusting its 0/6 survival finding.",
    threshold: "If < 0.70, add warning: adversarial validation may be producing false negatives",
  },
  { id: "M33", name: "Per-Model Performance Attribution", section: 9, priority: "MEDIUM",
    step: "All LLM steps (3 models)",
    what: "Breakdown of output quality, latency, and claim survival attributed separately to each LLM: llama3.2, qwen3:4b, qwen3:8b.",
    formula: "Per-model aggregation: avg_chars/sec, claim_density, total_duration",
    why: "Model selection optimisation. Enables evaluation of whether upgrading a model improves research quality.",
    threshold: "Expose in UI as model comparison table",
  },
];

const sectionNames = {
  1: "Pipeline-Level Cross-Step Metrics",
  2: "Research & News Step Metrics",
  3: "Financial & SEC Data Step Metrics",
  4: "Sentiment & Analysis Step Metrics",
  5: "ABM Simulation Metrics",
  6: "Tool Use Metrics",
  7: "Adversarial Validation Metrics",
  8: "Memory & Context Metrics",
  9: "Observability & Monitoring Metrics",
};

// ── Build document ──────────────────────────────────────────────
const children = [];

// ── Title page ──────────────────────────────────────────────────
children.push(new Paragraph({ spacing: { before: 2400 }, children: [] }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [new TextRun({ text: "Pipeline Metrics Reference", font: "Arial", size: 52, bold: true, color: C.primary })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({ text: "33 Metrics for Multi-Agent Pipeline Evaluation", font: "Arial", size: 28, color: C.accent })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
  children: [new TextRun({ text: "M01 \u2013 M33  |  Agent Eval Pipeline v6.0", font: "Arial", size: 22, color: C.muted })],
}));
children.push(divider());
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 400, after: 100 },
  children: [new TextRun({ text: "April 2026", font: "Arial", size: 22, color: C.muted })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Confidential \u2013 Internal Use Only", font: "Arial", size: 20, italics: true, color: C.muted })],
}));

// ── Page break + TOC ────────────────────────────────────────────
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("Table of Contents", HeadingLevel.HEADING_1));
children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 1: Executive Summary ────────────────────────────────
children.push(heading("1. Executive Summary", HeadingLevel.HEADING_1));
children.push(p("This document provides a comprehensive reference for the 33 pipeline-level metrics (M01\u2013M33) implemented in the Agent Eval Pipeline. These metrics are designed to provide deep observability into a multi-agent financial due diligence pipeline that orchestrates 11 sequential steps across 3 LLM models (llama3.2, qwen3:4b, qwen3:8b), multiple data sources (Tavily, yfinance, SEC EDGAR, Alpha Vantage), and an Agent-Based Model (ABM) simulation engine."));

children.push(p("The metrics are organized into 9 sections covering every aspect of pipeline health:", { spaceBefore: 160 }));

const sectionSummaries = [
  ["Section 1 (M01\u2013M04):", " Cross-step pipeline metrics including calibration gap, sentiment alignment, SLA compliance, and throughput."],
  ["Section 2 (M05\u2013M09):", " Research and news quality metrics including source credibility, freshness, coverage, claim density, and overlap."],
  ["Section 3 (M10\u2013M13):", " Financial and SEC data completeness metrics including data source availability, KPI coverage, XBRL parsing, and risk factor analysis."],
  ["Section 4 (M14\u2013M16):", " Sentiment and marketing analysis metrics including confidence scoring, polarity alignment, and moat analysis depth."],
  ["Section 5 (M17\u2013M22):", " ABM simulation metrics including Monte Carlo variance, catalyst impact, agent consensus, contagion propagation, KOL influence, and compute efficiency."],
  ["Section 6 (M23\u2013M25):", " Tool use metrics that fix the current 0.0 scores by instrumenting the HTTP adapter layer."],
  ["Section 7 (M26\u2013M28):", " Adversarial validation metrics including failure categorization, evidence traceability, and fallback synthesis quality."],
  ["Section 8 (M29\u2013M30):", " Memory and context metrics measuring cross-step information retention and knowledge graph utilization."],
  ["Section 9 (M31\u2013M33):", " Observability metrics including anomaly detection, judge reliability, and per-model performance attribution."],
];

for (const [label, desc] of sectionSummaries) {
  children.push(richBullet(boldLabel(label, desc), "bullets"));
}

children.push(p(""));
children.push(p("Why These Metrics Matter", { bold: true, size: 23, color: C.primary, spaceBefore: 200 }));
children.push(p("The previous evaluation framework captured 7 categories of metrics (task completion, tool use, trajectory, multi-agent coordination, reliability, enterprise cost, and safety) but had significant blind spots:"));

children.push(bullet("Calibration blindness: The synthesize agent self-reported 10/10 quality while 0/6 adversarial claims survived."));
children.push(bullet("Silent contradictions: BULLISH sentiment from the LLM vs. NEUTRAL from ABM simulation went undetected."));
children.push(bullet("Tool use opacity: invocation_accuracy = 0.0 and parameter_F1 = 0.0 because the eval harness could not observe remote agent tool calls."));
children.push(bullet("No SLA enforcement: The ABM simulation consumed 68.6% of pipeline time with no threshold check."));
children.push(bullet("No financial data validation: The pipeline had no way to detect hallucinated financial figures."));

children.push(p("These 33 metrics close every one of these gaps.", { spaceBefore: 160 }));

// ── Page break ──────────────────────────────────────────────────
children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 2: Architecture Overview ────────────────────────────
children.push(heading("2. Architecture Overview", HeadingLevel.HEADING_1));
children.push(p("The pipeline metrics system integrates with the existing eval harness through 3 new files and 4 modified files:", { spaceBefore: 120 }));

children.push(p("New Files", { bold: true, size: 23, color: C.primary, spaceBefore: 200 }));

const archTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [3200, 6160],
  rows: [
    new TableRow({ children: [headerCell("File", 3200), headerCell("Purpose", 6160)] }),
    new TableRow({ children: [
      dataCell("pipeline_metrics_config.py", 3200, { bold: true }),
      dataCell("SLA budgets, credibility tiers, KPI lists, thresholds, model-step mappings", 6160),
    ]}),
    new TableRow({ children: [
      dataCell("pipeline_metrics.py", 3200, { bold: true, shade: true }),
      dataCell("Core computation logic for all 33 metrics. Pure functions operating on PipelineResult dict.", 6160, { shade: true }),
    ]}),
    new TableRow({ children: [
      dataCell("pipeline_metrics_evaluator.py", 3200, { bold: true }),
      dataCell("PipelineMetricsEvaluator class integrating with the eval harness via BaseEvaluator interface.", 6160),
    ]}),
  ],
});
children.push(archTable);

children.push(p("Modified Files", { bold: true, size: 23, color: C.primary, spaceBefore: 300 }));

const modTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [3200, 6160],
  rows: [
    new TableRow({ children: [headerCell("File", 3200), headerCell("Change", 6160)] }),
    new TableRow({ children: [
      dataCell("eval_harness.py", 3200, { bold: true }),
      dataCell("Registered PipelineMetricsEvaluator + blocking/warning thresholds", 6160),
    ]}),
    new TableRow({ children: [
      dataCell("trajectory_tracer.py", 3200, { bold: true, shade: true }),
      dataCell("Added pipeline_data field to TrajectoryRecord for raw PipelineResult passthrough", 6160, { shade: true }),
    ]}),
    new TableRow({ children: [
      dataCell("report_store.py", 3200, { bold: true }),
      dataCell("Saves pipeline_metrics.json alongside other report files", 6160),
    ]}),
    new TableRow({ children: [
      dataCell("evals/__init__.py", 3200, { bold: true, shade: true }),
      dataCell("Exports PipelineMetricsEvaluator", 6160, { shade: true }),
    ]}),
  ],
});
children.push(modTable);

children.push(p("Data Flow", { bold: true, size: 23, color: C.primary, spaceBefore: 300 }));
children.push(p("1. Remote agent pipeline runs and returns PipelineResult JSON via HTTP POST.", { spaceBefore: 80 }));
children.push(p("2. RemoteAgentAdapter captures the full response in AgentResult.raw."));
children.push(p("3. TracingWrapper passes AgentResult.raw into TrajectoryRecord.pipeline_data."));
children.push(p("4. PipelineMetricsEvaluator.evaluate_suite() extracts pipeline_data and calls compute_all()."));
children.push(p("5. All 33 metrics computed from the single PipelineResult dict (zero additional API calls)."));
children.push(p("6. Results stored in EvalResult.metrics (flat scorecard) + EvalResult.details (full nested JSON)."));
children.push(p("7. ReportStore writes pipeline_metrics.json to the eval report folder."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 3: Priority Summary Table ───────────────────────────
children.push(heading("3. Metric Priority Summary", HeadingLevel.HEADING_1));
children.push(p("All 33 metrics at a glance, sorted by priority level. CRITICAL metrics block CI/CD merges; HIGH metrics trigger warnings; MEDIUM metrics are tracked for trend analysis."));

const priorityCounts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0 };
metrics.forEach(m => priorityCounts[m.priority]++);

children.push(p(`Distribution: ${priorityCounts.CRITICAL} CRITICAL, ${priorityCounts.HIGH} HIGH, ${priorityCounts.MEDIUM} MEDIUM`, { bold: true, spaceBefore: 160 }));

const summaryRows = [
  new TableRow({ children: [
    headerCell("ID", 700), headerCell("Metric Name", 4100), headerCell("Priority", 1100),
    headerCell("Pipeline Step", 1800), headerCell("Threshold", 1660),
  ] }),
];

for (const m of metrics) {
  const shade = metrics.indexOf(m) % 2 === 1;
  summaryRows.push(new TableRow({ children: [
    dataCell(m.id, 700, { bold: true, shade }),
    dataCell(m.name, 4100, { shade }),
    priorityCell(m.priority, 1100),
    dataCell(m.step.length > 25 ? m.step.substring(0, 25) + "..." : m.step, 1800, { shade }),
    dataCell(m.threshold.length > 22 ? m.threshold.substring(0, 22) + "..." : m.threshold, 1660, { shade }),
  ]}));
}

children.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [700, 4100, 1100, 1800, 1660],
  rows: summaryRows,
}));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 4: Detailed Metric Reference ────────────────────────
children.push(heading("4. Detailed Metric Reference", HeadingLevel.HEADING_1));
children.push(p("This section provides the complete specification for each of the 33 metrics including what it measures, how it is computed, why it matters, and the threshold configuration."));

let currentSection = 0;
for (const m of metrics) {
  if (m.section !== currentSection) {
    currentSection = m.section;
    children.push(new Paragraph({ children: [new PageBreak()] }));
    children.push(heading(`4.${currentSection}. Section ${currentSection}: ${sectionNames[currentSection]}`, HeadingLevel.HEADING_2));
  }

  // Metric header
  children.push(new Paragraph({
    spacing: { before: 280, after: 80 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: C.accent, space: 4 } },
    children: [
      new TextRun({ text: `${m.id}  `, font: "Arial", size: 24, bold: true, color: C.accent }),
      new TextRun({ text: m.name, font: "Arial", size: 24, bold: true, color: C.primary }),
    ],
  }));

  // Info table
  const infoTable = new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2000, 7360],
    rows: [
      new TableRow({ children: [
        new TableCell({ width: { size: 2000, type: WidthType.DXA }, borders: noBorders, margins: cellMargins,
          shading: { fill: C.lightBg, type: ShadingType.CLEAR },
          children: [new Paragraph({ children: [new TextRun({ text: "Priority", font: "Arial", size: 18, bold: true, color: C.primary })] })] }),
        new TableCell({ width: { size: 7360, type: WidthType.DXA }, borders: noBorders, margins: cellMargins,
          children: [new Paragraph({ children: [new TextRun({ text: m.priority, font: "Arial", size: 18, bold: true,
            color: m.priority === "CRITICAL" ? C.danger : m.priority === "HIGH" ? C.warning : C.accent })] })] }),
      ]}),
      new TableRow({ children: [
        new TableCell({ width: { size: 2000, type: WidthType.DXA }, borders: noBorders, margins: cellMargins,
          shading: { fill: C.lightBg, type: ShadingType.CLEAR },
          children: [new Paragraph({ children: [new TextRun({ text: "Pipeline Step", font: "Arial", size: 18, bold: true, color: C.primary })] })] }),
        new TableCell({ width: { size: 7360, type: WidthType.DXA }, borders: noBorders, margins: cellMargins,
          children: [new Paragraph({ children: [new TextRun({ text: m.step, font: "Arial", size: 18 })] })] }),
      ]}),
      new TableRow({ children: [
        new TableCell({ width: { size: 2000, type: WidthType.DXA }, borders: noBorders, margins: cellMargins,
          shading: { fill: C.lightBg, type: ShadingType.CLEAR },
          children: [new Paragraph({ children: [new TextRun({ text: "Threshold", font: "Arial", size: 18, bold: true, color: C.primary })] })] }),
        new TableCell({ width: { size: 7360, type: WidthType.DXA }, borders: noBorders, margins: cellMargins,
          children: [new Paragraph({ children: [new TextRun({ text: m.threshold, font: "Arial", size: 18 })] })] }),
      ]}),
    ],
  });
  children.push(infoTable);

  // What it is
  children.push(p("What It Measures", { bold: true, size: 21, color: C.primary, spaceBefore: 160 }));
  children.push(p(m.what));

  // Formula
  children.push(p("How It Is Computed", { bold: true, size: 21, color: C.primary, spaceBefore: 160 }));
  children.push(codeBlock(m.formula));

  // Why it matters
  children.push(p("Why It Matters", { bold: true, size: 21, color: C.primary, spaceBefore: 160 }));
  children.push(p(m.why));
}

children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 5: Threshold Configuration ──────────────────────────
children.push(heading("5. Threshold Configuration", HeadingLevel.HEADING_1));
children.push(p("The pipeline metrics system uses a two-tier threshold gate integrated with the existing ThresholdGate class in eval_harness.py:"));

children.push(p("Blocking Thresholds (block CI/CD merge)", { bold: true, size: 23, color: C.danger, spaceBefore: 200 }));
children.push(p("These thresholds must pass for a pipeline run to be considered successful:"));

const blockTable = new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [4680, 1560, 3120],
  rows: [
    new TableRow({ children: [headerCell("Metric", 4680), headerCell("Operator", 1560), headerCell("Threshold", 3120)] }),
    new TableRow({ children: [dataCell("m23_tool_execution_success_rate", 4680), dataCell(">=", 1560), dataCell("0.70", 3120)] }),
    new TableRow({ children: [dataCell("m02_sentiment_alignment_score", 4680, { shade: true }), dataCell(">=", 1560, { shade: true }), dataCell("0.40", 3120, { shade: true })] }),
  ],
});
children.push(blockTable);

children.push(p("Warning Thresholds (warn + log, do not block)", { bold: true, size: 23, color: C.warning, spaceBefore: 300 }));

const warnRows = [
  new TableRow({ children: [headerCell("Metric", 4680), headerCell("Operator", 1560), headerCell("Threshold", 3120)] }),
];
const warnEntries = [
  ["m01_calibration_gap", "<=", "30.0"],
  ["m03_sla_compliance_rate", ">=", "0.90"],
  ["m05_source_credibility_score", ">=", "0.65"],
  ["m07_query_coverage_score", ">=", "0.75"],
  ["m11_kpi_coverage_score", ">=", "0.80"],
  ["m14_sentiment_confidence", ">=", "0.65"],
  ["m27_evidence_traceability_score", ">=", "0.50"],
  ["m29_context_retention_score", ">=", "0.70"],
  ["m32_judge_reliability_score", ">=", "0.70"],
];
warnEntries.forEach(([metric, op, thresh], i) => {
  const shade = i % 2 === 1;
  warnRows.push(new TableRow({ children: [dataCell(metric, 4680, { shade }), dataCell(op, 1560, { shade }), dataCell(thresh, 3120, { shade })] }));
});
children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [4680, 1560, 3120], rows: warnRows }));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 6: Output Format ────────────────────────────────────
children.push(heading("6. Output Format", HeadingLevel.HEADING_1));
children.push(p("When the pipeline runs and completes, the metrics are saved in multiple formats for different consumers:"));

children.push(p("Report Directory Structure", { bold: true, size: 23, color: C.primary, spaceBefore: 200 }));
children.push(codeBlock("reports/evals/{job_id}/"));
children.push(codeBlock("  \u251C\u2500\u2500 report.json              (full eval report with all categories)"));
children.push(codeBlock("  \u251C\u2500\u2500 scorecard.json           (pass/fail summary per category)"));
children.push(codeBlock("  \u251C\u2500\u2500 metrics.json             (flat key-value metrics for dashboards)"));
children.push(codeBlock("  \u251C\u2500\u2500 pipeline_metrics.json    (full M01-M33 nested JSON)"));
children.push(codeBlock("  \u251C\u2500\u2500 metadata.json            (agent info, git SHA, trigger)"));
children.push(codeBlock("  \u2514\u2500\u2500 trajectory.json          (full trajectory record)"));

children.push(p("Flat Metrics (metrics.json)", { bold: true, size: 23, color: C.primary, spaceBefore: 240 }));
children.push(p("The flat metrics file includes all 33 pipeline metrics alongside the existing 7 categories. Each metric is prefixed with pipeline_metrics. for namespace isolation. This format is optimized for time-series dashboards and trend analysis."));

children.push(p("Full Pipeline Metrics (pipeline_metrics.json)", { bold: true, size: 23, color: C.primary, spaceBefore: 240 }));
children.push(p("The full pipeline metrics file contains the complete nested JSON output from compute_all(), including per-step breakdowns, flagged items, categorized claims, severity distributions, and the _meta block with computation timing and warnings."));

children.push(p("Scorecard Integration", { bold: true, size: 23, color: C.primary, spaceBefore: 240 }));
children.push(p("The pipeline_metrics category appears alongside existing categories (task_completion, tool_use, etc.) in the scorecard. It shows pass/fail status based on the blocking thresholds and lists any warning-level violations."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 7: Interpreting Results ─────────────────────────────
children.push(heading("7. Interpreting Results", HeadingLevel.HEADING_1));
children.push(p("This section provides guidance on interpreting common metric patterns and taking corrective action."));

children.push(heading("7.1 High Calibration Gap (M01 > 30%)", HeadingLevel.HEADING_2));
children.push(p("A high calibration gap means the synthesis agent is overconfident. The self-reported quality score is significantly higher than the adversarial survival rate."));
children.push(richBullet(boldLabel("Root cause: ", "The synthesis LLM generates plausible-sounding claims that are not grounded in upstream data."), "bullets"));
children.push(richBullet(boldLabel("Fix: ", "Improve the synthesis prompt to require explicit citation of data sources. Add grounding constraints."), "bullets"));

children.push(heading("7.2 Low Sentiment Alignment (M02 < 0.7)", HeadingLevel.HEADING_2));
children.push(p("The LLM sentiment classifier and ABM simulation disagree on market direction."));
children.push(richBullet(boldLabel("Root cause: ", "The sentiment step may be using different data signals than the ABM agents."), "bullets"));
children.push(richBullet(boldLabel("Fix: ", "Inject the alignment warning into the synthesize prompt context so the synthesis agent knows about the contradiction."), "bullets"));

children.push(heading("7.3 SLA Breach (M03 < 1.0)", HeadingLevel.HEADING_2));
children.push(p("One or more pipeline steps exceeded their latency budget."));
children.push(richBullet(boldLabel("Common offender: ", "abm_simulation (budget: 600s, typical: 1017s). This step runs 5 Monte Carlo paths with 50 agents each."), "bullets"));
children.push(richBullet(boldLabel("Fix: ", "Reduce MC paths to 3 (standard mode), reduce agents to 30, or switch to a faster LLM for agent deliberation."), "bullets"));

children.push(heading("7.4 Zero Tool Success (M23 = 0.0)", HeadingLevel.HEADING_2));
children.push(p("All tool calls appear to have failed. This usually indicates a logging issue rather than actual failures."));
children.push(richBullet(boldLabel("Root cause: ", "The HTTP adapter layer does not log tool calls at sufficient granularity."), "bullets"));
children.push(richBullet(boldLabel("Fix: ", "Instrument the adapter layer to log tool_name, params, response_size, success flag, and latency for each call."), "bullets"));

children.push(heading("7.5 High Hallucination Rate (M24 > 0.20)", HeadingLevel.HEADING_2));
children.push(p("More than 20% of financial figures in the synthesis cannot be traced to API responses."));
children.push(richBullet(boldLabel("Root cause: ", "The LLM is generating plausible financial data from parametric memory rather than using retrieved data."), "bullets"));
children.push(richBullet(boldLabel("Fix: ", "Log actual API responses separately. Compare key numeric fields after steps 3 and 4."), "bullets"));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ── Chapter 8: SLA Budget Reference ─────────────────────────────
children.push(heading("8. SLA Budget Reference", HeadingLevel.HEADING_1));
children.push(p("Per-step latency budgets used by M03 (Step SLA Compliance Rate). These are configurable in pipeline_metrics_config.py."));

const slaRows = [
  new TableRow({ children: [headerCell("Pipeline Step", 3120), headerCell("Budget (seconds)", 2000), headerCell("LLM Model", 2120), headerCell("Role", 2120)] }),
];
const slaData = [
  ["deep_research", "60", "llama3.2:latest", "Web search + analysis"],
  ["news", "30", "llama3.2:latest", "News digest"],
  ["financial", "90", "qwen3:4b", "yfinance + XBRL + AV"],
  ["sec_risk", "90", "qwen3:4b", "SEC EDGAR risk analysis"],
  ["marketing", "90", "qwen3:8b", "Brand & moat analysis"],
  ["sentiment", "45", "qwen3:8b", "Sentiment classification"],
  ["abm_simulation", "600", "Mesa ABM", "Monte Carlo simulation"],
  ["abm_report", "120", "qwen3:8b", "Post-simulation narrative"],
  ["synthesize", "120", "qwen3:8b", "Investment report"],
  ["adversarial_validation", "180", "qwen3:8b", "Claim verification"],
  ["save+email+sheet", "60", "APIs", "Gmail + Sheets + file save"],
];
slaData.forEach(([step, budget, model, role], i) => {
  const shade = i % 2 === 1;
  slaRows.push(new TableRow({ children: [
    dataCell(step, 3120, { bold: true, shade }),
    dataCell(budget + "s", 2000, { shade }),
    dataCell(model, 2120, { shade }),
    dataCell(role, 2120, { shade }),
  ]}));
});
children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 2000, 2120, 2120], rows: slaRows }));

// ── Final page ──────────────────────────────────────────────────
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading("9. Appendix: Source Credibility Tiers", HeadingLevel.HEADING_1));
children.push(p("Domain authority scores used by M05 (Source Credibility Score). Configurable in pipeline_metrics_config.py."));

const credRows = [
  new TableRow({ children: [headerCell("Domain", 4680), headerCell("Credibility Score", 4680)] }),
];
const credData = [
  ["sec.gov", "1.00"], ["reuters.com", "0.95"], ["bloomberg.com", "0.95"],
  ["wsj.com", "0.90"], ["ft.com", "0.90"], ["cnbc.com", "0.85"],
  ["marketwatch.com", "0.80"], ["yahoo.com", "0.75"], ["investopedia.com", "0.70"],
  ["fool.com", "0.65"], ["seekingalpha.com", "0.65"], ["stockanalysis.com", "0.60"],
  ["(default/unknown)", "0.40"],
];
credData.forEach(([domain, score], i) => {
  const shade = i % 2 === 1;
  credRows.push(new TableRow({ children: [dataCell(domain, 4680, { shade }), dataCell(score, 4680, { shade })] }));
});
children.push(new Table({ width: { size: 9360, type: WidthType.DXA }, columnWidths: [4680, 4680], rows: credRows }));

// ── Assemble document ───────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Arial", color: C.primary },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Arial", color: C.primary },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial", color: C.accent },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
          { level: 1, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
        ] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: C.accent, space: 4 } },
          children: [new TextRun({ text: "Pipeline Metrics Reference  |  M01\u2013M33", font: "Arial", size: 16, color: C.muted, italics: true })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          border: { top: { style: BorderStyle.SINGLE, size: 1, color: C.border, space: 4 } },
          children: [
            new TextRun({ text: "Page ", font: "Arial", size: 16, color: C.muted }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 16, color: C.muted }),
            new TextRun({ text: "  |  Agent Eval Pipeline v6.0  |  Confidential", font: "Arial", size: 16, color: C.muted }),
          ],
        })],
      }),
    },
    children,
  }],
});

// ── Write file ──────────────────────────────────────────────────
const outPath = process.argv[2] || "Pipeline_Metrics_Reference.docx";
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(outPath, buffer);
  console.log(`Document written to ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
});
