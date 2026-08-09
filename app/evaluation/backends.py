"""Model backend abstraction for Stage 4 (pre-fine-tuning baseline) and
Stage 6 (four-tier evaluation).

The baseline evaluation needs to call a code LLM and get back a text response.
Rather than hard-code a single inference path (``transformers``, ``vLLM``,
``Ollama``, etc.), we define a ``ModelBackend`` Protocol with a single
``generate`` method. This makes every downstream component (prompt builder,
parser, metrics) trivially testable with a mock — the same pattern used by
``TokenCounter`` in Stage 3 and ``EmbeddingBackend`` in Stage 2.

The default production backend (``QwenBackend``) uses ``transformers`` to load
the project's base model (Qwen2.5-Coder-7B-Instruct). If ``transformers`` is
not installed or the model can't be downloaded, the backend raises a clear
``RuntimeError`` — callers should fall back to ``MockBackend`` for tests.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)

# The project's base model, as stated in the README tech stack.
DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Generation defaults — conservative so we don't blow past context windows.
DEFAULT_MAX_NEW_TOKENS: int = 2048
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TOP_P: float = 0.95


class ModelBackend(Protocol):
    """Anything that can take a prompt string and return a generated string.

    The ``generate`` method is intentionally minimal: a single prompt in, one
    text response out. All decoding parameters are captured at construction time
    so they're consistent across a run.
    """

    def generate(self, prompt: str) -> str: ...


class QwenBackend:
    """Production backend using ``transformers`` to run Qwen2.5-Coder.

    Loads the model lazily on first ``generate`` call, so simply constructing
    the backend is cheap (no model download until actually used).

    Parameters
    ----------
    model_name:
        HuggingFace model ID. Defaults to the project's base model.
    max_new_tokens:
        Maximum tokens to generate in the response.
    temperature:
        Sampling temperature — low for deterministic classification.
    top_p:
        Nucleus sampling threshold.
    device:
        Device to load the model on (``"auto"`` detects GPU).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BASE_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self._pipeline = None

    def _load(self):
        """Lazy-load the HuggingFace text-generation pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Run "
                "`pip install -e '.[ml]'` to use the QwenBackend."
            ) from exc

        logger.info("Loading model %s on device=%s", self.model_name, self.device)
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map=self.device,
            framework="pt",
        )
        return self._pipeline

    def generate(self, prompt: str) -> str:
        """Generate a response from the model for the given prompt."""
        pipe = self._load()
        result = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            return_full_text=False,
        )
        # transformers pipeline returns a list of dicts; each dict has
        # either "generated_text" (single str) or "text" (older versions).
        if isinstance(result, list) and len(result) > 0:
            entry = result[0]
            text = entry.get("generated_text", entry.get("text", ""))
        else:
            text = str(result)
        return text.strip()


class MockBackend:
    """Deterministic backend for testing — returns a canned response.

    The ``responses`` dict maps a key (or callable) to a canned output string.
    In its simplest form, pass a single string and every ``generate`` call
    returns it. For per-sample control, pass a dict keyed by a substring of
    the prompt.

    Parameters
    ----------
    responses:
        A dict mapping prompt-substring -> response string. If a prompt
        contains any key as a substring, that response is returned.
        If none match, ``default`` is returned.
    default:
        Fallback response when no key matches.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "{}",
    ):
        self._responses = responses or {}
        self._default = default
        self.call_count = 0
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.calls.append(prompt[:80] + "..." if len(prompt) > 80 else prompt)
        for key, response in self._responses.items():
            if key in prompt:
                return response
        return self._default
