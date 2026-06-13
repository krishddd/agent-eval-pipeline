"""
evals/base_evaluator.py
BaseEvaluator interface, EvalResult model, and LLMJudge with meta-evaluation.

All evaluators inherit from BaseEvaluator and return EvalResult.
LLMJudge includes Cohen's κ inter-judge agreement to detect "judge hallucination."
"""

from __future__ import annotations

import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from evals._env import load_env

load_env()  # make OPENAI_API_KEY / judge config available wherever the judge runs


# ── Eval Result ──────────────────────────────────────────────────────────

class EvalResult(BaseModel):
    """Standardised result from any evaluator."""
    category: str
    passed: bool
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    details: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    judge_reliability: Optional[float] = None  # Cohen's κ if LLM judge used


# ── Base Evaluator ───────────────────────────────────────────────────────

class BaseEvaluator(ABC):
    """
    Abstract base for all evaluators.

    Each evaluator:
    - Receives List[TrajectoryRecord] (k runs) + AgentCard
    - Returns EvalResult or None if not applicable
    - Is stateless — no shared state between evaluator instances
    - Runs concurrently via asyncio.gather()
    """

    category: str = "base"

    @abstractmethod
    async def evaluate_suite(
        self,
        all_runs: Dict[str, list],  # task → List[TrajectoryRecord]
        card: Any,                   # AgentCard
        wrapper: Any = None,         # TracingWrapper (for injection tests)
    ) -> Optional[EvalResult]:
        """
        Evaluate all k runs across all tasks.

        Args:
            all_runs: Mapping of task string → list of TrajectoryRecords.
            card: The AgentCard for the agent being evaluated.
            wrapper: TracingWrapper (needed by reliability evaluator for injection).

        Returns:
            EvalResult with category, pass/fail, and metric values.
            None if this evaluator is not applicable.
        """
        raise NotImplementedError

    def _all_records(self, all_runs: Dict[str, list]) -> list:
        """Flatten all runs across all tasks into a single list."""
        return [r for runs in all_runs.values() for r in runs]


# ── LLM Judge with Meta-Evaluation ──────────────────────────────────────

class LLMJudge:
    """
    LLM-as-a-Judge utility with built-in judge reliability checking.

    For metrics requiring subjective judgment (Planning Score, Coordination
    Score, PersonaScore), the judge runs N times and computes inter-judge
    agreement using Cohen's κ.  Flags judge_reliability < 0.7 as unreliable.

    This prevents "judge hallucination" from corrupting eval results.

    Backend priority:
    1. OpenAI (if openai package installed + OPENAI_API_KEY set)
    2. Ollama local (http://localhost:11434) — uses qwen3:14b by default
    """

    _import_warned = False  # Class-level flag: warn once about missing OpenAI
    _backend = None          # "openai" | "ollama" | None (auto-detect on first call)

    # Ollama config — override via env vars if needed
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "qwen3:14b"  # Best available local model for judging

    def __init__(
        self,
        model: str = "gpt-4o",
        n_judges: int = 3,
        temperature: float = 0.0,
        reliability_threshold: float = 0.7,
    ):
        self.model = model
        self.n_judges = n_judges
        self.temperature = temperature
        self.reliability_threshold = reliability_threshold
        self._client = None  # Lazy-initialized OpenAI client

    async def judge(
        self,
        prompt: str,
        rubric: str,
        content: str,
        scale: Tuple[int, int] = (1, 5),
    ) -> Tuple[Optional[float], float, bool]:
        """
        Run the LLM judge N times and return the aggregated score.

        Args:
            prompt: Task/context description.
            rubric: Evaluation rubric text (from judge_prompts/).
            content: The content to evaluate.
            scale: Min/max score range (default 1-5).

        Returns:
            Tuple of (aggregated_score, cohen_kappa, is_reliable).
            aggregated_score is None if OpenAI SDK is not installed.
        """
        scores = []

        for _ in range(self.n_judges):
            score = await self._single_judgment(prompt, rubric, content, scale)
            if score is None:
                # FIX #4: OpenAI not installed — return None, don't fake scores
                return None, 0.0, False
            scores.append(score)

        aggregated = statistics.mean(scores)
        kappa = self._compute_cohen_kappa(scores, scale)
        is_reliable = kappa >= self.reliability_threshold

        return aggregated, kappa, is_reliable

    def _build_messages(
        self, prompt: str, rubric: str, content: str, scale: Tuple[int, int]
    ) -> list:
        """Build the chat messages for the judge call."""
        return [
            {
                "role": "system",
                "content": (
                    f"You are an expert evaluator. Score the following content "
                    f"on a scale of {scale[0]} to {scale[1]}.\n\n"
                    f"RUBRIC:\n{rubric}\n\n"
                    f"Respond with ONLY a single integer score, nothing else."
                ),
            },
            {
                "role": "user",
                "content": f"TASK: {prompt}\n\nCONTENT TO EVALUATE:\n{content}",
            },
        ]

    @staticmethod
    def _extract_score(raw: str, scale: Tuple[int, int]) -> Optional[float]:
        """Extract a numeric score from LLM output text."""
        for token in raw.split():
            try:
                score = float(token)
                return max(scale[0], min(scale[1], score))
            except ValueError:
                continue
        return None

    async def complete(self, system: str, user: str) -> Optional[str]:
        """
        Raw free-text completion via the detected backend (OpenAI → Ollama).

        Returns None when no backend is available, so callers can fall back to
        an offline heuristic.  Used by the RAGAS-style grounding/faithfulness
        judge (claim extraction + NLI verification).
        """
        if LLMJudge._backend is None:
            LLMJudge._backend = await self._detect_backend()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if LLMJudge._backend == "openai":
            try:
                import openai
                if self._client is None:
                    self._client = openai.AsyncOpenAI()
                resp = await self._client.chat.completions.create(
                    model=self.model, temperature=self.temperature, messages=messages,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception:
                LLMJudge._backend = "ollama"  # fall through
        if LLMJudge._backend == "ollama":
            try:
                import httpx
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{self.OLLAMA_BASE_URL}/api/chat",
                        json={"model": self.OLLAMA_MODEL, "messages": messages,
                              "stream": False, "options": {"temperature": self.temperature}},
                    )
                    if resp.status_code == 200:
                        return (resp.json().get("message", {}).get("content", "") or "").strip()
            except Exception:
                return None
        return None

    async def _single_judgment(
        self,
        prompt: str,
        rubric: str,
        content: str,
        scale: Tuple[int, int],
    ) -> Optional[float]:
        """
        Execute a single LLM judgment call.

        Tries OpenAI first; falls back to local Ollama if unavailable.
        Returns None only when no backend works.
        """
        # Auto-detect backend on first call
        if LLMJudge._backend is None:
            LLMJudge._backend = await self._detect_backend()

        messages = self._build_messages(prompt, rubric, content, scale)

        if LLMJudge._backend == "openai":
            return await self._judge_openai(messages, scale)
        elif LLMJudge._backend == "ollama":
            return await self._judge_ollama(messages, scale)
        else:
            return None  # No backend available

    async def _detect_backend(self) -> str:
        """Auto-detect the best available LLM backend."""
        import os

        # 1. Try OpenAI if package installed AND key is set
        if os.environ.get("OPENAI_API_KEY"):
            try:
                import openai  # noqa: F401
                print("[LLMJudge] Using OpenAI backend")
                return "openai"
            except ImportError:
                pass

        # 2. Try Ollama local
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.OLLAMA_BASE_URL}/api/tags")
                if resp.status_code == 200:
                    models = [m["name"] for m in resp.json().get("models", [])]
                    # Check if our preferred model is available
                    if any(self.OLLAMA_MODEL in m for m in models):
                        print(f"[LLMJudge] Using Ollama backend (model: {self.OLLAMA_MODEL})")
                        return "ollama"
                    # Fall back to any available qwen model
                    for m in models:
                        if "qwen" in m or "llama" in m:
                            LLMJudge.OLLAMA_MODEL = m
                            print(f"[LLMJudge] Using Ollama backend (model: {m})")
                            return "ollama"
        except Exception:
            pass

        print("[LLMJudge] WARNING: No LLM backend available. "
              "LLM-judged metrics will be skipped. "
              "Set OPENAI_API_KEY or start Ollama.")
        return "none"

    async def _judge_openai(
        self, messages: list, scale: Tuple[int, int]
    ) -> Optional[float]:
        """Call OpenAI for judgment."""
        try:
            import openai
            if self._client is None:
                self._client = openai.AsyncOpenAI()
            response = await self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
            )
            raw = response.choices[0].message.content.strip()
            return self._extract_score(raw, scale)
        except Exception as e:
            # OpenAI failed — try Ollama as fallback
            if not LLMJudge._import_warned:
                print(f"[LLMJudge] OpenAI failed ({type(e).__name__}), falling back to Ollama")
                LLMJudge._import_warned = True
            LLMJudge._backend = "ollama"
            return await self._judge_ollama(messages, scale)

    async def _judge_ollama(
        self, messages: list, scale: Tuple[int, int]
    ) -> Optional[float]:
        """Call local Ollama for judgment."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{self.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": self.OLLAMA_MODEL,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                )
                if resp.status_code != 200:
                    return None
                raw = resp.json().get("message", {}).get("content", "").strip()
                return self._extract_score(raw, scale)
        except Exception:
            return None

    @staticmethod
    def _compute_cohen_kappa(scores: List[float], scale: Tuple[int, int]) -> float:
        """
        Compute Randolph's free-marginal multi-rater kappa for one subject
        (one piece of content) rated by N judges.

        NOTE on naming: this is NOT Cohen's κ. Cohen's κ is defined only for
        EXACTLY TWO raters. For N>2 raters the correct coefficients are Fleiss' κ
        (nominal) or Krippendorff's α (any measurement level). We use Randolph's
        free-marginal variant here because the judge scale is small-ordinal and
        marginals are not fixed in advance. The method name is retained for
        backward compatibility; treat the returned value as a multi-rater
        agreement coefficient.

        Discretises continuous scores to integer bins on the scale,
        then computes observed vs chance agreement.
        """
        if len(scores) < 2:
            return 1.0

        n_raters = len(scores)
        # Discretise scores to integer categories within scale
        categories = list(range(scale[0], scale[1] + 1))
        n_categories = len(categories)

        # Map each score to the nearest integer category
        rounded = [max(scale[0], min(scale[1], round(s))) for s in scores]

        # Count assignments per category (single "subject" with N raters)
        counts = [rounded.count(cat) for cat in categories]

        # Observed agreement: P_o = (1 / (n*(n-1))) * (Σ n_j² - n)
        sum_sq = sum(c * c for c in counts)
        p_observed = (sum_sq - n_raters) / (n_raters * (n_raters - 1)) if n_raters > 1 else 1.0

        # Expected agreement: P_e = Σ (p_j)² where p_j = n_j / n
        p_expected = sum((c / n_raters) ** 2 for c in counts)

        if abs(1.0 - p_expected) < 1e-10:
            return 1.0  # Perfect agreement or degenerate case

        kappa = (p_observed - p_expected) / (1.0 - p_expected)
        return max(0.0, min(1.0, kappa))

    @staticmethod
    def load_rubric(rubric_name: str) -> str:
        """Load a rubric from the judge_prompts directory."""
        import os

        rubric_dir = os.path.join(os.path.dirname(__file__), "judge_prompts")
        rubric_path = os.path.join(rubric_dir, f"{rubric_name}.txt")

        if os.path.exists(rubric_path):
            with open(rubric_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return f"Evaluate quality on a 1-5 scale for: {rubric_name}"
