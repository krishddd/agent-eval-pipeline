"""
tracer/injection_middleware.py
Failure Injection Engine — simulates 7 mandatory fault types mid-execution.

Used by ReliabilityEvaluator to measure Recovery Rate.
Wraps the adapter's tool execution layer as middleware, injecting
faults at configurable probability or deterministic schedule.
"""

from __future__ import annotations

import random
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class FailureType(str, Enum):
    """Seven mandatory failure injection types from §4.5."""
    HTTP_404         = "http_404"         # Remote API returns not-found
    API_TIMEOUT      = "api_timeout"      # 30-second hard timeout
    SCHEMA_ERROR     = "schema_error"     # Tool returns malformed JSON
    EMPTY_RESULT     = "empty_result"     # Tool returns null / empty list
    PARTIAL_DATA     = "partial_data"     # Incomplete result set
    RATE_LIMIT_429   = "rate_limit_429"   # Provider throttling
    AUTH_FAILURE_401  = "auth_failure_401" # Credential expiry


class InjectionConfig(BaseModel):
    """Configuration for a failure injection run."""
    failure_type: FailureType
    probability: float = Field(default=1.0, ge=0.0, le=1.0)
    target_tool: Optional[str] = None  # Inject on specific tool, or any
    inject_at_call_index: Optional[int] = None  # Inject at Nth tool call
    timeout_seconds: float = 30.0  # For API_TIMEOUT type


class InjectionResult(BaseModel):
    """Outcome of a failure injection test."""
    failure_type: FailureType
    injected: bool = False
    agent_recovered: bool = False
    recovery_strategy: str = ""  # retry | skip | fallback | fail
    error_message: str = ""


class FailureInjectionError(Exception):
    """Raised when a failure is injected into a tool call."""
    def __init__(self, failure_type: FailureType, message: str):
        self.failure_type = failure_type
        super().__init__(message)


# ── Failure Generators ───────────────────────────────────────────────────

def _generate_http_404() -> Dict[str, Any]:
    """Simulate a 404 Not Found response."""
    raise FailureInjectionError(
        FailureType.HTTP_404,
        '{"status": 404, "error": "Not Found", "message": "The requested resource does not exist"}'
    )


def _generate_api_timeout(timeout_seconds: float = 30.0):
    """Simulate a hard timeout."""
    time.sleep(min(timeout_seconds, 2.0))  # Capped at 2s for eval speed
    raise FailureInjectionError(
        FailureType.API_TIMEOUT,
        f"TimeoutError: Operation timed out after {timeout_seconds}s"
    )


def _generate_schema_error() -> Dict[str, Any]:
    """Return malformed JSON that fails validation."""
    raise FailureInjectionError(
        FailureType.SCHEMA_ERROR,
        '{"result": [INVALID_JSON, missing_quotes: true, }}'
    )


def _generate_empty_result() -> Dict[str, Any]:
    """Return null or empty list."""
    return {"result": None, "data": [], "count": 0}


def _generate_partial_data() -> Dict[str, Any]:
    """Return incomplete result set (3 of 10 requested records)."""
    return {
        "result": [
            {"id": 1, "value": "partial_1"},
            {"id": 2, "value": "partial_2"},
            {"id": 3, "value": "partial_3"},
        ],
        "total_expected": 10,
        "total_returned": 3,
        "warning": "Incomplete result set",
    }


def _generate_rate_limit_429():
    """Simulate provider throttling."""
    raise FailureInjectionError(
        FailureType.RATE_LIMIT_429,
        '{"status": 429, "error": "Rate Limit Exceeded", '
        '"retry_after": 60, "message": "Too many requests. Please retry after 60 seconds."}'
    )


def _generate_auth_failure_401():
    """Simulate credential expiry."""
    raise FailureInjectionError(
        FailureType.AUTH_FAILURE_401,
        '{"status": 401, "error": "Unauthorized", '
        '"message": "Authentication token expired. Please re-authenticate."}'
    )


# Map failure types to generators
FAILURE_GENERATORS = {
    FailureType.HTTP_404:        _generate_http_404,
    FailureType.API_TIMEOUT:     _generate_api_timeout,
    FailureType.SCHEMA_ERROR:    _generate_schema_error,
    FailureType.EMPTY_RESULT:    _generate_empty_result,
    FailureType.PARTIAL_DATA:    _generate_partial_data,
    FailureType.RATE_LIMIT_429:  _generate_rate_limit_429,
    FailureType.AUTH_FAILURE_401: _generate_auth_failure_401,
}


# ── FailureInjector ──────────────────────────────────────────────────────

class FailureInjector:
    """
    Middleware that wraps an adapter to inject failures mid-execution.

    Usage:
        injector = FailureInjector(adapter, [
            InjectionConfig(failure_type=FailureType.HTTP_404),
            InjectionConfig(failure_type=FailureType.API_TIMEOUT),
        ])
        result, injection_results = injector.run_with_injection(task)
    """

    def __init__(
        self,
        adapter: Any,  # AgentAdapter
        injections: List[InjectionConfig],
        seed: Optional[int] = None,
    ):
        self.adapter = adapter
        self.injections = injections
        self.injection_results: List[InjectionResult] = []
        self._call_count = 0
        self._rng = random.Random(seed)

    def _should_inject(self, config: InjectionConfig, tool_name: str) -> bool:
        """Determine if this tool call should be injected with a failure."""
        # Check tool name filter
        if config.target_tool and config.target_tool != tool_name:
            return False

        # Check call index filter
        if config.inject_at_call_index is not None:
            return self._call_count == config.inject_at_call_index

        # Probabilistic injection
        return self._rng.random() < config.probability

    def _create_injecting_hook(
        self,
        original_hook: Optional[Callable],
    ) -> Callable:
        """Create a tool call hook that injects failures."""
        import threading
        _lock = threading.Lock()

        def _hook(tool_call_record):
            with _lock:
                self._call_count += 1

            for config in self.injections:
                if self._should_inject(config, tool_call_record.tool_name):
                    generator = FAILURE_GENERATORS[config.failure_type]

                    injection_result = InjectionResult(
                        failure_type=config.failure_type,
                        injected=True,
                    )

                    try:
                        if config.failure_type == FailureType.API_TIMEOUT:
                            generator(config.timeout_seconds)
                        elif config.failure_type in (
                            FailureType.EMPTY_RESULT,
                            FailureType.PARTIAL_DATA,
                        ):
                            # Non-exception failures: modify the result
                            injected_data = generator()
                            tool_call_record.result = injected_data
                            tool_call_record.data_sources = ["injected_failure"]
                            injection_result.agent_recovered = True
                            injection_result.recovery_strategy = "received_degraded_data"
                        else:
                            generator()

                    except FailureInjectionError as e:
                        tool_call_record.success = False
                        tool_call_record.error = str(e)
                        tool_call_record.result = str(e)
                        injection_result.error_message = str(e)
                        # C6 fix: store result then re-raise so adapter sees the failure
                        self.injection_results.append(injection_result)
                        if original_hook:
                            original_hook(tool_call_record)
                        raise  # Agent must handle this for real recovery testing

                    self.injection_results.append(injection_result)
                    break  # Only inject one failure per tool call

            if original_hook:
                original_hook(tool_call_record)

        return _hook

    def run_with_injection(
        self,
        task: str,
        on_tool_call: Optional[Callable] = None,
        on_agent_msg: Optional[Callable] = None,
        on_retrieval: Optional[Callable] = None,
    ):
        """
        Run the adapter with failure injection enabled.

        Returns:
            Tuple of (AgentResult, List[InjectionResult])
        """
        self._call_count = 0
        self.injection_results = []

        injecting_hook = self._create_injecting_hook(on_tool_call)

        result = self.adapter.run(
            task,
            on_tool_call=injecting_hook,
            on_agent_msg=on_agent_msg,
            on_retrieval=on_retrieval,
        )

        # Determine recovery for each injection
        for ir in self.injection_results:
            if ir.injected and result.success:
                ir.agent_recovered = True
                if not ir.recovery_strategy:
                    ir.recovery_strategy = "task_completed_despite_failure"

        return result, self.injection_results
