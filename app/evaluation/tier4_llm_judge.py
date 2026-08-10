"""Stage 6, Tier 4 — LLM-judge for explanation quality and patch minimality.

This tier uses an LLM as a judge to evaluate two qualitative dimensions that
are hard to measure automatically:

1. **Explanation quality** — is the model's ``rationale`` clear, accurate,
   and actionable?
2. **Patch minimality** — does the suggested patch fix the vulnerability
   without introducing unnecessary or risky changes?

The LLM judge is behind a ``LlmJudgeBackend`` Protocol so tests can inject
a mock. When no LLM API key is configured, the evaluator can run in
"rubber-stamp" mode (returns 0.5 for everything) — useful for dry-run.

The judge prompt is minimal to keep cost low; a typical invocation sends
the CVE description + vulnerable code + patch + rationale and asks for
two float scores (0.0–1.0) and a brief rationale.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Protocol

from app.schemas.prediction_eval import LlmJudgeScore, ModelPrediction
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


# Default judge prompt — minimal and cost-conscious.
JUDGE_PROMPT = """\
You are evaluating a vulnerability-fix suggestion from a security LLM.

VULNERABILITY:
CWE: {cwe_id}
Description: {description}
Language: {language}

VULNERABLE CODE:
{vulnerable_code}

SUGGESTED PATCH (unified diff):
{patch_diff}

RATIONALE:
{rationale}

Rate two dimensions from 0.0 (worst) to 1.0 (best), then explain briefly.

1. Explanation quality — is the rationale clear, technically accurate, and
   does it correctly identify the root cause and fix?
2. Patch minimality — does the patch fix the vulnerability without
   unnecessary changes, dead code, or risky modifications?

Respond as JSON:
{{"explanation_quality": <float>, "patch_minimality": <float>, "rationale": "<brief explanation>"}}
"""


class LlmJudgeBackend(Protocol):
    """Anything that can send a prompt to an LLM and return its text output."""

    def invoke(self, prompt: str, model: str, max_tokens: int = 512) -> str:
        """Send *prompt* to the LLM identified by *model* and return text."""
        ...


@dataclass
class MockLlmJudgeBackend:
    """Deterministic LLM judge backend for testing.

    Returns a canned ``LlmJudgeScore`` based on the ``default_scores`` mapping
    keyed by ``prediction_id``. All requests not in the mapping receive
    the ``fallback`` score.
    """

    fallback_explanation_quality: float = 0.5
    fallback_patch_minimality: float = 0.5
    default_model: str = "mock-judge"

    def invoke(self, prompt: str, model: str, max_tokens: int = 512) -> str:
        import json
        return json.dumps({
            "explanation_quality": self.fallback_explanation_quality,
            "patch_minimality": self.fallback_patch_minimality,
            "rationale": "mock-judge default response",
        })


def _parse_judge_response(text: str) -> tuple[float, float, str]:
    """Parse the LLM judge's JSON response.

    Falls back to (0.5, 0.5, text[:200]) if parsing fails.
    """
    import json
    try:
        data = json.loads(text.strip())
        eq = float(data.get("explanation_quality", 0.5))
        pm = float(data.get("patch_minimality", 0.5))
        rationale = str(data.get("rationale", ""))
        return max(0.0, min(1.0, eq)), max(0.0, min(1.0, pm)), rationale
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Failed to parse LLM judge response, using fallback: %s", exc)
        return 0.5, 0.5, text[:200]


class LlmJudge:
    """Tier 4 evaluator — uses an LLM to score explanation quality and patch minimality.

    Parameters
    ----------
    backend:
        A ``LlmJudgeBackend`` implementation. Inject ``MockLlmJudgeBackend``
        for tests; use a real backend (OpenAI/Claude) in production.
    model:
        The model name to send to the backend (e.g. ``"gpt-4o-mini"``).
    max_tokens:
        Max tokens for the judge response.
    """

    def __init__(
        self,
        backend: LlmJudgeBackend | None = None,
        model: str | None = None,
        max_tokens: int = 512,
    ):
        if backend is not None:
            self._backend = backend
        else:
            # Try to construct a real backend from the environment.
            self._backend = self._build_default_backend()

        self._model = model or os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
        self._max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        """Send a prompt to the LLM backend (sync wrapper around async)."""
        if asyncio.get_event_loop().is_running():
            # We're inside an event loop — can't use asyncio.run.
            return self._backend.invoke(prompt, self._model, self._max_tokens)
        return self._backend.invoke(prompt, self._model, self._max_tokens)

    @staticmethod
    def _build_default_backend() -> LlmJudgeBackend:
        """Construct a backend from the environment.

        Supports OpenAI-compatible endpoints via ``OPENAI_API_KEY`` /
        ``OPENAI_BASE_URL`` env vars. If neither is set, returns a
        ``MockLlmJudgeBackend`` so the system degrades gracefully.
        """
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            return _OpenAiBackend(api_key=api_key, base_url=base_url)
        logger.info("No LLM judge backend configured — using mock backend")
        return MockLlmJudgeBackend()

    def evaluate(
        self,
        sample: VulnSample,
        prediction: ModelPrediction,
    ) -> LlmJudgeScore:
        """Score a single prediction's explanation quality and patch minimality.

        Parameters
        ----------
        sample:
            The gold-eval ``VulnSample``.
        prediction:
            The model's ``ModelPrediction``.
        """
        prompt = JUDGE_PROMPT.format(
            cwe_id=sample.cwe_id,
            description=sample.description,
            language=sample.language,
            vulnerable_code=sample.vulnerable_code,
            patch_diff=prediction.suggested_patch_diff,
            rationale=prediction.rationale,
        )
        response = self.invoke(prompt)
        eq, pm, rationale = _parse_judge_response(response)
        return LlmJudgeScore(
            prediction_id=prediction.sample_id,
            explanation_quality=round(eq, 4),
            patch_minimality=round(pm, 4),
            evaluator_model=self._model,
            rationale=rationale,
        )

    def evaluate_all(
        self,
        samples: list[VulnSample],
        predictions: list[ModelPrediction],
    ) -> list[LlmJudgeScore]:
        """Score all predictions in a batch.

        Predictions are matched to samples by ``sample_id``.
        """
        pred_by_sample: dict[str, ModelPrediction] = {
            p.sample_id: p for p in predictions
        }
        results: list[LlmJudgeScore] = []
        for sample in samples:
            pred = pred_by_sample.get(sample.id)
            if pred is None:
                logger.warning("No prediction for sample %s — skipping", sample.id)
                continue
            results.append(self.evaluate(sample, pred))
        return results


class _OpenAiBackend:
    """Minimal OpenAI-compatible backend using ``openai`` client.

    Falls back gracefully if the ``openai`` package is not installed.
    """

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        except ImportError:
            logger.error(
                "openai package not installed — LLM judge disabled; "
                "using MockLlmJudgeBackend fallback."
            )
            self._client = None

    def invoke(self, prompt: str, model: str, max_tokens: int = 512) -> str:
        if self._client is None:
            return MockLlmJudgeBackend().invoke(prompt, model, max_tokens)
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
