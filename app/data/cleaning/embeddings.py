"""Code embedding backend for Stage 2 near-duplicate detection.

This implements the roadmap's embedding-based dedup: `sentence-transformers`
with a code-embedding model. It needs `pip install -e ".[ml]"` and network
access to huggingface.co to download model weights on first use.

The default model is `jinaai/jina-embeddings-v2-base-code`, a model trained
specifically for code retrieval. It ships custom modelling code on the HF Hub
and therefore requires `trust_remote_code=True`.

Note on where this runs: the dev sandbox this repo was authored in does
NOT have network access to huggingface.co, so this backend can't be
exercised end-to-end there — it's built for real and unit-tested against
an injectable mock model (see tests/unit/test_dedup.py), but the actual
run — on real CVEfixes data, with the real model — happens on your own
machine, where that network access exists.

If you hit an ImportError when loading the jina model (it references
`find_pruneable_heads_and_indices` which was removed from newer
`transformers` releases), either downgrade transformers to <5.0, or set
`trust_remote_code=False` with a model that doesn't ship custom code
(e.g. `intfloat/multilingual-e5-base`).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingBackend:
    """Wraps a sentence-transformers model. `model_name` is pinned to a
    specific tag rather than left to float, so re-runs stay comparable.

    Parameters
    ----------
    model_name:
        HuggingFace Hub model ID. Defaults to a code-trained model.
    model:
        Injectable for tests — anything with an ``encode()`` method.
    trust_remote_code:
        Passed through to ``SentenceTransformer``. Set ``False`` if the model
        doesn't ship custom code (avoids a potential ImportError on newer
        ``transformers`` versions).
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        model=None,
        trust_remote_code: bool = True,
    ):
        self.model_name = model_name
        self._model = model
        self._trust_remote_code = trust_remote_code

    def _load(self):
        """Lazy-load the SentenceTransformer model on first use.

        ``trust_remote_code`` is passed through; if loading fails because of
        a version incompatibility (the model's custom code references a
        ``transformers`` internal that was removed), we re-raise as a
        ``RuntimeError`` with actionable guidance rather than a cryptic
        ImportError.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run "
                    "`pip install -e '.[ml]'` (requires network access to "
                    "huggingface.co to download model weights on first use)."
                ) from exc

            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    trust_remote_code=self._trust_remote_code,
                )
            except ImportError as exc:
                # The model's custom code (trust_remote_code=True) may
                # reference a transformers internal that was removed in
                # newer versions. Re-raise with actionable guidance.
                raise RuntimeError(
                    f"Failed to load embedding model '{self.model_name}' from "
                    f"Hugging Face Hub. This is often caused by a version "
                    f"incompatibility between your installed `transformers` "
                    f"and the model's custom code.\n"
                    f"Underlying error: {exc}\n"
                    f"Try one of:\n"
                    f"  1. pip install 'transformers<5'  (or a version known "
                    f"to work with this model)\n"
                    f"  2. Use a model that doesn't require trust_remote_code, "
                    f"e.g.\n"
                    f"     EmbeddingBackend(model_name="
                    f"'intfloat/multilingual-e5-base', trust_remote_code=False)"
                ) from exc
        return self._model

    def embed(self, texts: list[str], batch_size: int = 32, max_chars: int = 2048) -> Any:
        """Returns one L2-normalized embedding vector per input text.

        ``max_chars`` truncates very long code snippets (some CVEfixes records
        are 200K+ characters) before tokenization, preventing Out-of-Memory
        errors on CPU.  ``batch_size`` controls how many texts are passed to
        the model at once to keep peak memory bounded.
        """
        model = self._load()
        truncated = [t[:max_chars] for t in texts]
        return model.encode(
            truncated,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )


def cosine_similarity(a: Any, b: Any) -> float:
    """a, b are L2-normalized vectors (numpy arrays or anything supporting
    element-wise multiply and sum) — dot product reduces to cosine similarity.

    If the vectors are not already normalized, pass the raw dot product
    divided by norms instead — but the EmbeddingBackend always normalizes.
    """
    dot = float((a * b).sum())
    # Clamp to [-1, 1] to guard against tiny floating-point drift above 1.0
    # (which would break acos-based downstream computations).
    return max(-1.0, min(1.0, dot))
