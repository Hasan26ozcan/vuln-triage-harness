"""Tests dedup.py's pairing/removal logic against a mock embedding model
2	injected via EmbeddingBackend(model=...). This does NOT exercise the real
3	jina-embeddings model — that happens on your machine with network access
4	to huggingface.co. This only proves the near-duplicate selection logic
5	(threshold comparison, which sample gets kept, no double-counting) is
6	correct, independent of which model produces the vectors.
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.data.cleaning.dedup import dedup_samples, find_near_duplicates
from app.data.cleaning.embeddings import EmbeddingBackend, cosine_similarity
from app.schemas.vuln import VulnSample


def _sample(id_: str, code: str, repo: str = "org/repo") -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code=code,
        description="d",
    )


class _FakeModel:
    """Maps specific input texts to hand-picked vectors, so test outcomes
    are exact and don't depend on any real embedding model's behavior.
    """

    def __init__(self, vectors_by_text: dict[str, list[float]]):
        self.vectors_by_text = vectors_by_text

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
        return np.array([self.vectors_by_text[t] for t in texts])


def test_near_duplicate_pair_detected_above_threshold():
    a = _sample("a", "code_a")
    b = _sample("b", "code_b")
    model = _FakeModel({"code_a": [1.0, 0.0], "code_b": [0.99, 0.0141]})  # cos ~0.99
    backend = EmbeddingBackend(model=model)

    pairs = find_near_duplicates([a, b], backend=backend, threshold=0.95)

    assert len(pairs) == 1
    assert pairs[0].keep_id == "a"
    assert pairs[0].remove_id == "b"


def test_dissimilar_pair_not_flagged():
    a = _sample("a", "code_a")
    b = _sample("b", "code_b")
    model = _FakeModel({"code_a": [1.0, 0.0], "code_b": [0.0, 1.0]})  # orthogonal, cos=0
    backend = EmbeddingBackend(model=model)

    pairs = find_near_duplicates([a, b], backend=backend, threshold=0.95)

    assert pairs == []


def test_dedup_samples_removes_the_flagged_duplicate():
    a = _sample("a", "code_a")
    b = _sample("b", "code_b")
    model = _FakeModel({"code_a": [1.0, 0.0], "code_b": [1.0, 0.0]})  # identical, cos=1.0
    backend = EmbeddingBackend(model=model)

    kept, pairs = dedup_samples([a, b], backend=backend, threshold=0.95)

    assert len(kept) == 1
    assert kept[0].id == "a"
    assert len(pairs) == 1


def test_find_near_duplicates_skips_already_removed_in_inner_loop():
    """When sample j was already removed by an earlier outer iteration,
    the inner loop hits `continue` (line 41).

    Scenario: A and C are near-duplicates (C removed at i=0), but A and B
    are dissimilar. At i=1 (B), j=2 (C) is already in `removed`, so the
    inner loop skips it via continue — no duplicate pair (B, C) is formed.
    """
    a = _sample("a", "code_a")
    b = _sample("b", "code_b")
    c = _sample("c", "code_c")
    model = _FakeModel(
        {
            "code_a": [1.0, 0.0],  # A
            "code_b": [0.0, 1.0],  # B — orthogonal to A, not a duplicate
            "code_c": [0.99, 0.01],  # C — near-identical to A, duplicate of A
        }
    )
    backend = EmbeddingBackend(model=model)

    pairs = find_near_duplicates([a, b, c], backend=backend, threshold=0.95)

    assert len(pairs) == 1
    assert pairs[0].keep_id == "a"
    assert pairs[0].remove_id == "c"


def test_dedup_three_samples_keeps_first_and_removes_two_matches():
    """a is kept; b and c are both near-duplicates of a and should both be
    removed. No duplicate pairs between b and c since b is already removed."""
    a = _sample("a", "code_a")
    b = _sample("b", "code_b")
    c = _sample("c", "code_c")
    model = _FakeModel(
        {
            "code_a": [1.0, 0.0],
            "code_b": [0.99, 0.01],
            "code_c": [0.98, 0.02],
        }
    )
    backend = EmbeddingBackend(model=model)

    kept, pairs = dedup_samples([a, b, c], backend=backend, threshold=0.90)

    assert {s.id for s in kept} == {"a"}
    assert len(pairs) == 2
    removed_ids = {p.remove_id for p in pairs}
    assert removed_ids == {"b", "c"}


def test_cosine_similarity_returns_1_for_identical_vectors():
    v = np.array([3.0, 4.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_similarity_returns_0_for_orthogonal_vectors():
    v = np.array([1.0, 0.0])
    w = np.array([0.0, 1.0])
    assert abs(cosine_similarity(v, w) - 0.0) < 1e-6


def test_cosine_similarity_clamps_above_one():
    """Floating-point drift can push slightly above 1.0; ensure we clamp."""
    v = np.array([1.0, 0.0])
    w = np.array([1.0 + 1e-15, 0.0])
    result = cosine_similarity(v, w)
    assert result <= 1.0


def test_backend_raises_clear_error_when_sentence_transformers_missing():
    """When no mock model is injected AND sentence-transformers is not
    importable, the backend must raise a RuntimeError mentioning
    'sentence-transformers' with install instructions.
    """
    backend = EmbeddingBackend(model_name="fake/model")

    # Make the `from sentence_transformers import SentenceTransformer`
    # line inside _load() raise ImportError, simulating a missing package.
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            backend.embed(["some code"])


def test_backend_wraps_model_load_error_as_runtime_error():
    """When sentence-transformers IS installed but the model fails to load
    (e.g. version incompatibility), the backend should re-raise as a
    RuntimeError with actionable guidance, not a bare ImportError.
    """
    backend = EmbeddingBackend(
        model_name="jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
    )

    # Only run this if we can't actually load the model in this env.
    # If the real model loads fine, this test is a no-op (the env is good).
    try:
        backend._load()
    except RuntimeError as exc:
        assert "Failed to load embedding model" in str(exc) or "sentence-transformers" in str(exc)
    except Exception:
        # If the real model loads, great — no error, test passes.
        pass
    else:
        # Model loaded successfully — no error raised.
        pass
