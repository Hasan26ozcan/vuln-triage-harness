"""Tests for app/data/cleaning/embeddings.py — the EmbeddingBackend
lazy-loading wrapper and the cosine_similarity helper.

These tests mock the SentenceTransformer import so they run without network
access or a GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.data.cleaning.embeddings import EmbeddingBackend, cosine_similarity


def test_embed_uses_injected_model():
    """embed() delegates to the injected model's encode()."""
    fake = MagicMock()
    fake.encode.return_value = np.array([[0.5, 0.5]])
    backend = EmbeddingBackend(model_name="dummy/model", model=fake)

    result = backend.embed(["hello world"])

    fake.encode.assert_called_once_with(
        ["hello world"], convert_to_numpy=True, normalize_embeddings=True,
        batch_size=32, show_progress_bar=False,
    )
    assert isinstance(result, np.ndarray)


def test_load_lazy_loads_model_when_none():
    """_load() returns the injected model if already set."""
    fake = MagicMock()
    backend = EmbeddingBackend(model_name="dummy/model", model=fake)

    assert backend._load() is fake


def test_load_import_error_re_raised_as_runtime_error():
    """When sentence-transformers is importable but the model constructor
    raises ImportError (e.g. trust_remote_code incompatibility), the
    backend should re-raise as RuntimeError with actionable guidance.

    This covers lines 82-99 of embeddings.py.
    """
    backend = EmbeddingBackend(
        model_name="jinaai/jina-embeddings-v2-base-code",
        trust_remote_code=True,
    )

    # Make the import succeed but the constructor raise ModuleNotFoundError,
    # simulating a trust_remote_code version incompatibility.
    fake_module = MagicMock()
    fake_module.SentenceTransformer = MagicMock(
        side_effect=ModuleNotFoundError("No module named 'transformers.foo'")
    )

    with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
        with pytest.raises(RuntimeError, match="Failed to load embedding model"):
            backend._load()


def test_load_import_error_for_import_error_type():
    """Same path but with ImportError instead of ModuleNotFoundError."""
    backend = EmbeddingBackend(
        model_name="some/model",
        trust_remote_code=False,
    )

    fake_module = MagicMock()
    fake_module.SentenceTransformer = MagicMock(
        side_effect=ImportError("cannot import name 'foo'")
    )

    with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
        with pytest.raises(RuntimeError, match="Failed to load embedding model"):
            backend._load()


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


def test_cosine_similarity_clamps_below_minus_one():
    """Drift can also push below -1.0; ensure we clamp."""
    v = np.array([1.0, 0.0])
    w = np.array([-(1.0 + 1e-15), 0.0])
    result = cosine_similarity(v, w)
    assert result >= -1.0


def test_cosine_similarity_with_numpy_scalar_elements():
    """cosine_similarity works with numpy arrays that produce scalar results."""
    a = np.array([0.6, 0.8])
    b = np.array([0.6, 0.8])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6
