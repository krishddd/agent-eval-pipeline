"""Tracer package — Instrumentation, PII masking, and failure injection."""

from .trajectory_tracer import (
    TracingWrapper,
    TrajectoryRecord,
    ToolCallRecord,
    AgentMessage,
    PIIMasker,
    ProvenanceComparator,
)

__all__ = [
    "TracingWrapper", "TrajectoryRecord", "ToolCallRecord",
    "AgentMessage", "PIIMasker", "ProvenanceComparator",
]
