# Universal Agent Evaluation Pipeline — Revised Implementation Plan

Production-ready, framework-agnostic agent evaluation pipeline based on CLEAR (2025) and MultiAgentBench (ACL 2025). **FastAPI serves as the unified backend** for agent registration, eval execution, report generation, and drift monitoring.

---

## Proposed Changes

### Registry Layer (`/registry`) — Pydantic V2

#### [NEW] [agent_card.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/registry/agent_card.py)
- All models use **Pydantic V2 `BaseModel`** with strict validators for `hk_params`, `tools_manifest`, `persona_spec`
- `AgentType` enum (9 types) · `MemoryType` enum (5 types) · `ToolDef` model · `AgentCard` model
- `auto_infer_categories()` method on `AgentCard`

#### [NEW] [registry.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/registry/registry.py)
- `AgentRegistry` singleton: `register()`, `get()`, `list_all()`

---

### Adapter Pattern (`/adapters`) — 5 Adapters

#### [NEW] [base.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/adapters/base.py)
- `AgentResult` Pydantic model · `AgentAdapter` abstract base
- Hook signature: `run(task, on_tool_call, on_agent_msg, on_retrieval) → AgentResult`

#### [NEW] [langchain_adapter.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/adapters/langchain_adapter.py)
- `BaseCallbackHandler.on_tool_start/on_tool_end` hooks

#### [NEW] [crewai_adapter.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/adapters/crewai_adapter.py)
- `crew.kickoff()` + `crew.get_logs()` for inter-agent messages

#### [NEW] [langgraph_adapter.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/adapters/langgraph_adapter.py)
- `graph.stream()` with **node-level state snapshots** → each node maps to `ToolCallRecord`

#### [NEW] [autogen_adapter.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/adapters/autogen_adapter.py)
- `ConversableAgent.register_reply()` hook for **message interception** in GroupChat

---

### Instrumentation Layer (`/tracer`) — With Injection & PII Masking

#### [NEW] [trajectory_tracer.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/tracer/trajectory_tracer.py)
- `ToolCallRecord`, `AgentMessage`, `TrajectoryRecord` (Pydantic V2)
- `TracingWrapper` with OpenTelemetry spans, **PII masking middleware** (regex + spaCy NER on captured data before persistence)
- **Provenance Comparator** logic: compares `ToolCallRecord.data_sources` against `AgentCard.golden_sources` list to flag silent failures

#### [NEW] [injection_middleware.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/tracer/injection_middleware.py)
- `FailureInjector` class implementing all **7 mandatory failure types**:
  `HTTP_404`, `API_TIMEOUT`, `SCHEMA_ERROR`, `EMPTY_RESULT`, `PARTIAL_DATA`, `RATE_LIMIT_429`, `AUTH_FAILURE_401`
- Wraps the adapter's tool execution layer as a middleware, injecting faults at configurable probability
- Used by `ReliabilityEvaluator` to measure recovery rate

---

### Eval Engine (`/evals`) — 11 Evaluators + Judge Framework

#### [NEW] [base_evaluator.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/evals/base_evaluator.py)
- `EvalResult` model · `BaseEvaluator` abstract class
- **`LLMJudge`** utility with **judge meta-evaluation**: runs each LLM judgment N times, computes inter-judge agreement (Cohen's κ), flags `judge_reliability < 0.7` as unreliable

#### Core evaluators (7):

| File | Metrics | Key Formula |
|------|---------|-------------|
| `task_completion.py` | SR, pass@k, milestone KPI, step-wise progress | `pass@k = P(success in all k runs)` |
| `tool_use.py` | invocation acc, selection acc, param F1, redundancy, MRR | `F1 = 2×(P×R)/(P+R)` |
| `trajectory.py` | exact/in-order/any-order match, **silent failure via provenance comparator**, planning score | `silent_failure = correct_output AND NOT valid_provenance` |
| `multi_agent.py` | coordination, collab SR, handoff, overhead, ICI, parallelism | `ICI = milestones_by_agent_i / total` |
| `reliability.py` | recovery, consistency, SLA, PAS | Uses `FailureInjector` for the 7 fault types |
| `enterprise_cost.py` | CNA, TER, budget compliance, L2P gap | `CNA = accuracy / cost_per_task_USD` |
| `safety.py` | injection resistance, harm rate, fairness gap | 5 injection payload categories |

#### Conditional evaluators (4):

| File | Trigger | Metrics |
|------|---------|---------|
| `rag_quality.py` | `memory_type ∈ {VECTOR_DB, HYBRID}` | context precision/recall, faithfulness, relevancy |
| `graph_memory.py` | `memory_type ∈ {GRAPH_DB, HYBRID}` | DMR, cross-session, temporal, relational, delta |
| `persona_consistency.py` | `agent_type == SOCIAL_SIM` | PersonaScore, ConsistencyAI, LoCoMo, drift |
| `hk_contagion.py` | `agent_type == FINANCIAL_ABM` | convergence τ, clustering, ε-boundary, detection |

#### [NEW] [judge_prompts/](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/evals/judge_prompts/)
- `planning_rubric.txt` — 1–5 rubric for plan completeness, sequence logic, efficiency
- `coordination_rubric.txt` — 1–5 rubric for communication + planning quality
- `persona_rubric.txt` — normative + prescriptive + descriptive axes rubric

---

### Eval Harness (`/harness`) — Rate-Limit Aware

#### [NEW] [eval_harness.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/harness/eval_harness.py)
- `EvalHarness` with **rate-limit aware batching**: `asyncio.Semaphore` to cap concurrent API calls; configurable `max_concurrent_runs`
- `EvalReport` Pydantic model · `ThresholdGate` with all 18 CI/CD rules from §7.1

---

### Storage Layer (`/storage` & `/schema`) — With Artifact Store

#### [NEW] [tables.sql](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/schema/tables.sql)
- `agent_cards`, `eval_reports` (metrics as JSONB + `artifact_url` column), `metric_timeseries` (TimescaleDB hypertable), `production_baselines`, `metric_drift` view

#### [NEW] [report_store.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/storage/report_store.py)
- Async PostgreSQL via `asyncpg`
- **Artifact offloading**: large `TrajectoryRecord` JSON → S3/GCS, only `artifact_url` stored in Postgres

#### [NEW] [mlflow_logger.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/storage/mlflow_logger.py)
- MLflow experiment/run logging with Git SHA linking

#### [NEW] [artifact_store.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/storage/artifact_store.py)
- Pluggable backend: S3 (boto3) / GCS / local filesystem fallback
- `upload_trajectory()`, `download_trajectory()` keyed by `report_id`

---

### FastAPI Backend (`/dashboard`) — Unified API

#### [NEW] [api.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/dashboard/api.py)

Serves as the **unified backend** for all operations:

| Endpoint Group | Routes |
|---|---|
| **Registry** | `POST /agents/register`, `GET /agents`, `GET /agents/{id}` |
| **Eval Execution** | `POST /evals/run`, `GET /evals/status/{id}` |
| **Reports** | `GET /reports`, `GET /reports/{id}`, `GET /reports/{id}/scorecard` |
| **Drift & Observability** | `GET /drift/alerts`, `GET /metrics/trends/{agent_id}` |
| **Dashboard** | `GET /dashboard/scorecard`, `GET /dashboard/cost-explorer` |

- APScheduler hourly drift job + Slack/PagerDuty webhook
- CORS middleware, health check, OpenAPI docs

---

### Support Files

#### [NEW] [scripts/promote_to_regression.py](file:///c:/Users/hp/Downloads/AI_Testing_Suites/Agent%20eval%20pipeline/scripts/promote_to_regression.py)
- **Adaptive Feedback Loop** (§8.1): monitors `metric_drift` view, extracts failed task + trajectory, appends to `tasks/regression_suite.json`

#### Other support files:
- `tasks/smoke_suite.json`, `full_suite.json`, `regression_suite.json`
- `scripts/register_all_agents.py`, `scripts/log_to_mlflow.py`
- `.github/workflows/agent-eval.yml`
- `requirements.txt`

---

## Verification Plan

### Automated
1. `python -m py_compile` on every `.py` file
2. Import chain test: `from registry.agent_card import AgentCard` through all packages
3. Pydantic validation test: invalid `AgentCard` fields should raise `ValidationError`

### Manual
- Verify all 27+15 metric formulas match spec tables
- Confirm CI/CD YAML syntax is valid
- Verify FastAPI endpoints via OpenAPI docs at `/docs`
