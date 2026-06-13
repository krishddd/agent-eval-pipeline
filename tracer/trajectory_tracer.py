"""
tracer/trajectory_tracer.py
TracingWrapper — Universal instrumentation layer.

Captures every tool call, retrieval, LLM invocation, inter-agent message,
token count, cost, and policy violation in a single TrajectoryRecord.

Design principles:
- Real-time capture via synchronous hooks (non-negotiable)
- PII masking middleware before persistence
- Provenance comparator for silent failure detection
"""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

# FIX #15: Graceful no-op fallback when opentelemetry is not installed
try:
    from opentelemetry import trace
except ImportError:
    import contextlib

    class _NoOpSpan:
        def set_attribute(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class _NoOpTracer:
        def start_as_current_span(self, *a, **kw): return _NoOpSpan()

    class _NoOpTrace:
        @staticmethod
        def get_tracer(name): return _NoOpTracer()

    trace = _NoOpTrace()

from pydantic import BaseModel, Field


# ── Data Models ──────────────────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    """Single tool invocation captured at trace time."""
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    latency_ms: float = 0.0
    success: bool = True
    data_sources: List[str] = Field(default_factory=list)
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class AgentMessage(BaseModel):
    """Inter-agent message captured at trace time."""
    sender_id: str
    receiver_id: str
    content: str
    token_count: int = 0
    timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)


class TrajectoryRecord(BaseModel):
    """
    Central data object consumed by ALL evaluators.
    Every field is captured at trace time — never reconstructed post-hoc.
    """
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    task: str
    final_output: str = ""

    # ── Real-time captured data ──────────────────────────────────
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    agent_messages: List[AgentMessage] = Field(default_factory=list)

    # ── Token & cost tracking ────────────────────────────────────
    input_tokens: int = 0
    output_tokens: int = 0
    inter_agent_tokens: int = 0
    total_cost_usd: float = 0.0

    # ── Performance ──────────────────────────────────────────────
    wall_latency_ms: float = 0.0

    # ── Quality signals ──────────────────────────────────────────
    milestones_hit: List[str] = Field(default_factory=list)
    policy_violations: List[str] = Field(default_factory=list)
    data_sources: List[str] = Field(default_factory=list)
    success: bool = False

    # ── Provenance tracking ──────────────────────────────────────
    silent_failure_detected: bool = False
    provenance_valid: bool = True

    # ── Raw pipeline data (for pipeline_metrics evaluator) ──────
    pipeline_data: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


# ── PII Masking Middleware ───────────────────────────────────────────────

class PIIMasker:
    """
    Masks personally identifiable information from trace data
    BEFORE it is persisted to storage.

    Uses regex patterns for common PII types.  For production,
    extend with spaCy NER for entity-level detection.
    """

    # Common PII patterns
    PATTERNS = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "phone": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
        "ip_v4": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    }

    @classmethod
    def mask_text(cls, text: str) -> str:
        """Replace PII patterns with redacted placeholders."""
        if not text:
            return text
        result = text
        for pii_type, pattern in cls.PATTERNS.items():
            result = pattern.sub(f"[REDACTED_{pii_type.upper()}]", result)
        return result

    @classmethod
    def mask_record(cls, record: TrajectoryRecord) -> TrajectoryRecord:
        """Apply PII masking to all string fields in a TrajectoryRecord."""
        # L4 fix: deep-copy to prevent mutating original data
        import copy
        masked = copy.deepcopy(record)

        masked.final_output = cls.mask_text(masked.final_output)
        masked.task = cls.mask_text(masked.task)

        for tc in masked.tool_calls:
            tc.result = cls.mask_text(str(tc.result)) if tc.result else tc.result
            tc.error = cls.mask_text(tc.error) if tc.error else None

        for msg in masked.agent_messages:
            msg.content = cls.mask_text(msg.content)

        for chunk in masked.retrieved_chunks:
            if "content" in chunk:
                chunk["content"] = cls.mask_text(str(chunk["content"]))

        return masked


# ── Provenance Comparator ────────────────────────────────────────────────

class ProvenanceComparator:
    """
    Detects silent failures by comparing per-tool-call data_sources
    against a "Golden Source" list from the AgentCard.

    A silent failure = correct-looking output derived from a provably
    flawed execution path (stale cache, wrong DB, hallucinated data).
    """

    @staticmethod
    def check_provenance(
        record: TrajectoryRecord,
        golden_sources: Optional[List[str]] = None,
    ) -> bool:
        """
        Returns True if provenance is valid, False if silent failure detected.

        Checks:
        1. Every tool call has at least one data_source
        2. All data_sources are in the golden_sources list (if provided)
        3. No tool returned success=True with empty data_sources
        """
        if not golden_sources:
            # No golden sources configured — can't validate provenance
            return True

        all_sources = []
        has_successful_calls = False
        for tc in record.tool_calls:
            if tc.success and not tc.data_sources:
                # Tool succeeded but has no provenance — suspicious
                return False
            if tc.success:
                has_successful_calls = True
            all_sources.extend(tc.data_sources)

        # L6 fix: if golden_sources configured but agent reported no sources, flag it
        if not all_sources and has_successful_calls:
            return False  # Silent failure: agent used no attributed sources

        # Check all captured sources are in the golden set
        if all_sources:
            unauthorized_sources = set(all_sources) - set(golden_sources)
            if unauthorized_sources:
                return False

        return True


# ── TracingWrapper ───────────────────────────────────────────────────────

class TracingWrapper:
    """
    Universal instrumentation wrapper.  Works with ANY framework adapter.
    Zero framework-specific code.

    Wraps adapter.run() with:
    - OpenTelemetry span tree
    - Synchronous hook callbacks (real-time capture)
    - PII masking before output
    - Provenance validation for silent failure detection
    """

    def __init__(self, adapter, card, enable_pii_masking: bool = True):
        """
        Args:
            adapter: Any AgentAdapter subclass.
            card: AgentCard for this agent.
            enable_pii_masking: Whether to mask PII in captured data.
        """
        self.adapter = adapter
        self.card = card
        self.enable_pii_masking = enable_pii_masking
        self.tracer = trace.get_tracer(f"agent.eval.{card.agent_id}")

    def run(self, task: str) -> TrajectoryRecord:
        """
        Execute the agent and capture a complete TrajectoryRecord.

        All hooks fire synchronously during execution — never post-hoc.
        """
        run_id = str(uuid.uuid4())
        tool_calls: List[ToolCallRecord] = []
        messages: List[AgentMessage] = []
        chunks: List[Dict[str, Any]] = []

        t0 = time.time()

        with self.tracer.start_as_current_span(f"run.{self.card.name}") as span:
            span.set_attribute("agent.id", self.card.agent_id)
            span.set_attribute("agent.type", self.card.agent_type.value)
            span.set_attribute("agent.framework", self.card.framework)
            span.set_attribute("task", task[:200])
            span.set_attribute("run_id", run_id)

            # ── Execute with real-time hooks ─────────────────────
            def _on_tool_call(tc: ToolCallRecord):
                with self.tracer.start_as_current_span(f"tool.{tc.tool_name}") as tool_span:
                    tool_span.set_attribute("tool.name", tc.tool_name)
                    tool_span.set_attribute("tool.success", tc.success)
                    tool_span.set_attribute("tool.latency_ms", tc.latency_ms)
                tool_calls.append(tc)

            def _on_agent_msg(msg: AgentMessage):
                # FIX #3: Estimate token_count from content when not set
                if msg.token_count == 0 and msg.content:
                    msg.token_count = len(msg.content.split())  # ~1 token/word heuristic
                with self.tracer.start_as_current_span("agent.message") as msg_span:
                    msg_span.set_attribute("sender", msg.sender_id)
                    msg_span.set_attribute("receiver", msg.receiver_id)
                messages.append(msg)

            def _on_retrieval(chunk: Dict[str, Any]):
                with self.tracer.start_as_current_span("retrieval") as ret_span:
                    ret_span.set_attribute("chunk.keys", str(list(chunk.keys())))
                chunks.append(chunk)

            result = self.adapter.run(
                task,
                on_tool_call=_on_tool_call,
                on_agent_msg=_on_agent_msg,
                on_retrieval=_on_retrieval,
            )

        wall_latency_ms = (time.time() - t0) * 1000

        # ── Build the TrajectoryRecord ───────────────────────────
        # Capture raw pipeline data for pipeline_metrics evaluator
        raw_pipeline = None
        if hasattr(result, "raw") and isinstance(result.raw, dict):
            raw_pipeline = result.raw

        record = TrajectoryRecord(
            run_id=run_id,
            agent_id=self.card.agent_id,
            task=task,
            final_output=result.output,
            tool_calls=tool_calls,
            retrieved_chunks=result.retrieved_chunks or chunks,
            agent_messages=messages,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            inter_agent_tokens=sum(m.token_count for m in messages),
            total_cost_usd=result.cost_usd,
            wall_latency_ms=wall_latency_ms,
            milestones_hit=result.milestones,
            policy_violations=result.policy_violations,
            data_sources=[s for tc in tool_calls for s in tc.data_sources],
            success=result.success,
            pipeline_data=raw_pipeline,
        )

        # ── Provenance validation (silent failure detection) ─────
        record.provenance_valid = ProvenanceComparator.check_provenance(
            record, self.card.golden_sources
        )
        record.silent_failure_detected = (
            record.success and not record.provenance_valid
        )

        # ── PII masking ──────────────────────────────────────────
        if self.enable_pii_masking:
            record = PIIMasker.mask_record(record)

        return record
