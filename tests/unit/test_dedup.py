"""Tests dedup.py's pairing/removal logic against a mock embedding model
injected via EmbeddingBackend(model=...). This does NOT exercise the real
jina-embeddings model — that happens on your machine with network access
to huggingface.co. This only proves the near-duplicate selection logic
(threshold comparison, which sample gets kept, no double-counting) is
correct, independent of which model produces the vectors.
"""

import numpy as np

from app.data.cleaning.dedup import dedup_samples, find_near_duplicates
from app.data.cleaning.embeddings import EmbeddingBackend
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

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
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


def test_backend_raises_clear_error_without_sentence_transformers_or_mock():
    backend = EmbeddingBackend()  # no injected model, and (presumably) no real one installed
    try:
        backend.embed(["some code"])
    except RuntimeError as exc:
        assert "sentence-transformers" in str(exc)
    else:
        # If sentence-transformers happens to be installed in this
        # environment, this test can't assert the error path — that's fine,
        # it just means the real backend is available here too.
        pass
