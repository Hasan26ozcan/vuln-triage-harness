"""Code embedding backend for Stage 2 near-duplicate detection.

This is the roadmap's actual tool: `sentence-transformers` with a
code-embedding model (`jinaai/jina-embeddings-v2-base-code`). It needs
`pip install -e ".[ml]"` and network access to huggingface.co to download
model weights on first use.

Note on where this runs: the dev sandbox this repo was authored in does
NOT have network access to huggingface.co, so this backend can't be
exercised end-to-end there — it's built for real and unit-tested against
an injectable mock model (see tests/unit/test_dedup.py), but the actual
run — on real CVEfixes data, with the real model — happens on your own
machine, where that network access exists.
"""

from __future__ import annotations


class EmbeddingBackend:
    """Wraps a sentence-transformers model. `model_name` is pinned to a
    specific tag rather than left to float, so re-runs stay comparable.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        model=None,  # injectable for tests — anything with an .encode() method
    ):
        self.model_name = model_name
        self._model = model

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is not installed. Run "
                    "`pip install -e '.[ml]'` (requires network access to "
                    "huggingface.co to download model weights on first use)."
                ) from exc
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def embed(self, texts: list[str]):
        """Returns one L2-normalized embedding vector per input text."""
        model = self._load()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def cosine_similarity(a, b) -> float:
    """a, b are L2-normalized vectors (numpy arrays or anything supporting
    @ and sum) — dot product reduces to cosine similarity.
    """
    return float((a * b).sum())
