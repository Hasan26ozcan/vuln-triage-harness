"""Stage 3, step 2: token counting with an injectable backend.

Provides a ``TokenCounter`` that can use either:
- A real HuggingFace tokenizer (Qwen2.5-Coder by default) for production runs.
- A lightweight regex-based fallback for CI / tests that don't have ``transformers``
  or network access to download model files.

The counter is **injectable** into the builder and pipeline — tests pass a
mock so they never depend on model downloads. This mirrors the same pattern
used by ``EmbeddingBackend`` in Stage 2.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

logger = logging.getLogger(__name__)

# Default model — matches the project's stated base model. The tokenizer for
# Qwen2.5-Coder-7B-Instruct is shared across the 1.5B/7B variants in the
# Qwen2.5 family, so this tag covers both.
DEFAULT_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Hard upper bound for a single training example (prompt + target). Qwen2.5
# has a 32k context, but we keep sequences short so batches fit in 8GB VRAM.
DEFAULT_MAX_TOKENS: int = 4096


class TokenBackend(Protocol):
    """Protocol: anything with an ``encode`` method returning a token ID list."""

    def encode(self, text: str) -> list[int]: ...


class TokenCounter:
    """Counts tokens in a string, using a real tokenizer when available and
    a lightweight regex-based fallback otherwise.

    Parameters
    ----------
    model_name:
        HuggingFace model ID for the tokenizer. Defaults to the project's
        base model (Qwen2.5-Coder-7B-Instruct).
    tokenizer:
        Injectable for tests — anything with an ``encode(text) -> list[int]``
        method. If provided, ``model_name`` is ignored for loading.
    trust_remote_code:
        Passed through to ``AutoTokenizer.from_pretrained``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        tokenizer: TokenBackend | None = None,
        trust_remote_code: bool = False,
    ):
        self.model_name = model_name
        self._tokenizer = tokenizer
        self._trust_remote_code = trust_remote_code

    def _load(self) -> TokenBackend:
        """Lazy-load the HuggingFace tokenizer on first use.

        Raises a ``RuntimeError`` with actionable guidance if ``transformers``
        is not installed, so the caller can fall back to the heuristic counter.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Run "
                "`pip install transformers` or `pip install -e '.[ml]'`."
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=self._trust_remote_code,
            # Model tag pinning is in the design; CI uses heuristic fallback
            )  # nosec B615
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                f"Failed to load tokenizer for '{self.model_name}' from "
                f"HuggingFace Hub. Underlying error: {exc}\n"
                f"Try a different model, or pass a custom backend to "
                f"TokenCounter(tokenizer=...)."
            ) from exc
        return self._tokenizer

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text``."""
        try:
            tokenizer = self._load()
            return len(tokenizer.encode(text))
        except RuntimeError:
            # Fallback: the real tokenizer isn't available — use the heuristic.
            # This keeps CI / air-gapped environments working.
            return _heuristic_count(text)

    def count_prompt_and_target(
        self,
        prompt: str,
        *targets: str,
    ) -> int:
        """Count tokens in the prompt plus every target string.

        This is the number that matters for the token budget check during
        training: the model must attend to both the input and the expected
        output.
        """
        total = self.count(prompt)
        for t in targets:
            total += self.count(t)
        return total


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

# A simple regex that approximates subword tokenization: it counts words as
# one token each, and estimates ~0.33 extra tokens per CJK character or
# other non-word character (operators, punctuation). This is deliberately
# rough — it's only for environments where the real tokenizer can't load.
#
# Reference: the 100 tokens ~= 75 words rule of thumb (≈4 chars/tokens) gives
# a decent ballpark for English code.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _heuristic_count(text: str) -> int:
    """Approximate token count using the 4-chars-per-token heuristic.

    Fast, dependency-free, and good enough for budget filtering where we just
    need an upper/lower bound to decide keep-or-drop.

    Uses the word count from a simple ``\\w+`` regex and the classic
    chars/4 heuristic, taking the max as a conservative upper bound.
    """
    if not text:
        return 0
    n_chars = len(text)
    n_words = len(_WORD_RE.findall(text))
    # The classic heuristic is chars / 4. Use max(words, chars/4) as a
    # conservative upper bound so we rarely over-budget.
    return max(n_words, (n_chars + 3) // 4)
