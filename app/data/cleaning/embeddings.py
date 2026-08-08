"""Pluggable code-embedding backends for Stage 2 near-duplicate detection.

The roadmap's stated tool is `sentence-transformers` with a code-embedding
model (`jinaai/jina-embeddings-v2-base-code`). That's implemented below as
`SentenceTransformerEmbeddingBackend` — but it downloads model weights
from huggingface.co on first use, and huggingface.co is not in this
project's dev-sandbox network allowlist. So it's written and unit-tested
against an injectable mock model, meant to actually run on your own
machine (`pip install -e ".[ml]"`), not inside the sandbox this was
authored in.

For CI and fast local dev, `HashedNgramBackend` is a dependency-free
fallback: token n-gram hashing + TF weighting, cosine similarity on sparse
vectors. It is NOT a semantic embedding — it catches lexical near-dupes
(the common real case: the same fix mirrored across forked repos, or
copy-pasted vulnerable code with variables renamed) but will miss
duplicates that are semantically identical yet lexically very different.
That trade-off is intentional and documented, not hidden.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

SparseVector = dict[int, float]


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[SparseVector]:
        """Return one L2-normalized sparse vector per input text, so that
        cosine similarity reduces to a plain dot product (see
        `cosine_similarity` below).
        """


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S", text)


class HashedNgramBackend(EmbeddingBackend):
    """Default backend: hashed token n-grams, TF-weighted, L2-normalized."""

    def __init__(self, n: int = 3, num_buckets: int = 2**16):
        self.n = n
        self.num_buckets = num_buckets

    def embed(self, texts: list[str]) -> list[SparseVector]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> SparseVector:
        tokens = _tokenize(text)
        if len(tokens) >= self.n:
            grams = ["\u241f".join(tokens[i : i + self.n]) for i in range(len(tokens) - self.n + 1)]
        else:
            grams = tokens or [""]

        counts = Counter(grams)
        vec: SparseVector = {}
        for gram, count in counts.items():
            bucket = int(hashlib.blake2b(gram.encode(), digest_size=4).hexdigest(), 16)
            bucket %= self.num_buckets
            vec[bucket] = vec.get(bucket, 0.0) + count

        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}


class SentenceTransformerEmbeddingBackend(EmbeddingBackend):
    """Production backend per the roadmap. Requires `pip install -e ".[ml]"`
    and network access to huggingface.co on first run to fetch weights.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        model=None,  # injectable for tests — a mock with an .encode() method
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

    def embed(self, texts: list[str]) -> list[SparseVector]:
        model = self._load()
        rows = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [{i: float(v) for i, v in enumerate(row) if v != 0.0} for row in rows]


def cosine_similarity(a: SparseVector, b: SparseVector) -> float:
    """Dot product of two L2-normalized sparse vectors == cosine similarity."""
    if len(a) > len(b):
        a, b = b, a
    return sum(value * b.get(key, 0.0) for key, value in a.items())
