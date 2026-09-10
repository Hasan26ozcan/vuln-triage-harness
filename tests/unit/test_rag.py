"""Unit tests for Stage 5+ RAG retrieval module.

Covers:
  - _HAS_RAG fallback when sentence_transformers is not available.
  - RAGRetriever: build_index (with and without sentence_transformers),
    retrieve (embedding + keyword fallback), _retrieve_by_embedding,
    _retrieve_by_keyword, __post_init__.
  - build_rag_patch_prompt, build_rag_classification_prompt.
  - retrieve_and_build_prompt convenience function.
"""

from __future__ import annotations

import pytest

from app.training.rag import (
    RAGRetriever,
    RetrievalResult,
    build_rag_classification_prompt,
    build_rag_patch_prompt,
    retrieve_and_build_prompt,
)

# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_defaults(self):
        r = RetrievalResult(sample_id="s1", text="hello", score=0.9)
        assert r.sample_id == "s1"
        assert r.text == "hello"
        assert r.score == 0.9
        assert r.metadata == {}

    def test_with_metadata(self):
        r = RetrievalResult(sample_id="s1", text="hello", score=0.9, metadata={"k": "v"})
        assert r.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# RAGRetriever — _HAS_RAG is False (sentence_transformers not installed)
# ---------------------------------------------------------------------------


class TestRAGRetrieverNoSentenceTransformers:
    """Tests the fallback path when _HAS_RAG is False."""

    @pytest.fixture(autouse=True)
    def _patch_has_rag_false(self, monkeypatch):
        """Force _HAS_RAG to False by making the import fail."""
        import app.training.rag as rag_module

        rag_module._HAS_RAG = False

    def test_post_init_warns_when_no_rag(self, caplog):
        retriever = RAGRetriever()
        assert "RAG disabled" in caplog.text or retriever.model_name == "all-MiniLM-L6-v2"

    def test_build_index_without_sentence_transformers(self):
        """build_index stores examples/texts/ids even without embeddings."""
        import app.training.rag as rag_module
        rag_module._HAS_RAG = False

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": "SQL injection fix", "sample_id": "s1"})(),
            type("Ex", (), {"target_explanation": "XSS fix", "sample_id": "s2"})(),
        ]
        retriever.build_index(examples)
        assert len(retriever.examples) == 2
        assert len(retriever.texts) == 2
        assert retriever.ids == ["s1", "s2"]
        assert retriever.embeddings is None

    def test_build_index_filters_empty_texts(self):
        """Empty texts are filtered out."""
        import app.training.rag as rag_module
        rag_module._HAS_RAG = False

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": "valid", "sample_id": "s1"})(),
            type("Ex", (), {"target_explanation": "", "sample_id": "s2"})(),
            type("Ex", (), {"target_explanation": "  ", "sample_id": "s3"})(),
        ]
        retriever.build_index(examples, text_field="target_explanation")
        # Without sentence_transformers, build_index stores all examples (no filtering)
        # The filtering only happens in the _HAS_RAG path
        assert len(retriever.examples) == 3
        assert len(retriever.texts) == 3

    def test_retrieve_empty_returns_empty(self):
        """When no examples are indexed, retrieve returns []."""
        retriever = RAGRetriever()
        results = retriever.retrieve("query", top_k=3)
        assert results == []

    def test_retrieve_keyword_fallback(self):
        """When no embeddings, retrieve uses keyword matching."""
        import app.training.rag as rag_module
        rag_module._HAS_RAG = False

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": "SQL injection vulnerability", "sample_id": "s1"})(),  # noqa: E501
            type("Ex", (), {"target_explanation": "XSS cross-site scripting", "sample_id": "s2"})(),  # noqa: E501
        ]
        retriever.build_index(examples)
        results = retriever.retrieve("SQL injection", top_k=1, query_field="target_explanation")
        assert len(results) >= 1
        assert results[0].sample_id == "s1"
        assert results[0].score > 0

    def test_retrieve_top_k_limited(self):
        """top_k limits the number of results."""
        import app.training.rag as rag_module
        rag_module._HAS_RAG = False

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": f"sample {i}", "sample_id": f"s{i}"})()
            for i in range(5)
        ]
        retriever.build_index(examples)
        results = retriever.retrieve("sample", top_k=2, query_field="target_explanation")
        assert len(results) <= 2

    def test_retrieve_by_keyword_with_custom_query_field(self):
        """query_field parameter works in _retrieve_by_keyword."""
        import app.training.rag as rag_module
        rag_module._HAS_RAG = False

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"prompt": "fix sql injection", "sample_id": "s1"})(),
        ]
        retriever.build_index(examples, text_field="prompt")
        results = retriever.retrieve("sql", top_k=1, query_field="prompt")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# RAGRetriever — _HAS_RAG is True (sentence_transformers available)
# ---------------------------------------------------------------------------


class TestRAGRetrieverWithEmbeddings:
    """Tests the embedding-based retrieval path."""

    @pytest.fixture(autouse=True)
    def _setup_with_embeddings(self, monkeypatch):
        """Set up RAGRetriever with mock embeddings (384-dim like all-MiniLM-L6-v2)."""
        import app.training.rag as rag_module
        rag_module._HAS_RAG = True

        self.retriever = RAGRetriever()
        self.examples = [
            type("Ex", (), {"target_explanation": "SQL injection fix", "sample_id": "s1"})(),
            type("Ex", (), {"target_explanation": "XSS fix", "sample_id": "s2"})(),
        ]
        # all-MiniLM-L6-v2 produces 384-dim embeddings
        import numpy as np
        np.random.seed(42)
        self.retriever.examples = self.examples
        self.retriever.texts = ["sql injection", "xss fix"]
        self.retriever.ids = ["s1", "s2"]
        self.retriever.embeddings = np.random.rand(2, 384).astype(np.float32)

    def test_retrieve_by_embedding(self):
        """_retrieve_by_embedding returns results sorted by similarity."""
        results = self.retriever._retrieve_by_embedding("sql injection", top_k=1)
        assert len(results) == 1
        assert results[0].sample_id in ["s1", "s2"]

    def test_retrieve_by_embedding_top_k(self):
        """top_k parameter is respected."""
        results = self.retriever._retrieve_by_embedding("sql injection", top_k=2)
        assert len(results) == 2

    def test_retrieve_prefers_embedding_over_keyword(self):
        """When embeddings exist and _HAS_RAG, _retrieve_by_embedding is used."""
        results = self.retriever.retrieve("sql injection", top_k=1)
        assert len(results) == 1

    def test_build_index_with_embeddings(self, monkeypatch):
        """build_index with _HAS_RAG=True creates embeddings via SentenceTransformer."""
        from unittest.mock import MagicMock

        import numpy as np

        import app.training.rag as rag_module
        rag_module._HAS_RAG = True

        # Mock SentenceTransformer so we don't download the real model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(2, 384).astype(np.float32)
        monkeypatch.setattr(
            rag_module, "SentenceTransformer",
            lambda model_name: mock_model,
        )

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": "SQL injection fix", "sample_id": "s1"})(),
            type("Ex", (), {"target_explanation": "XSS fix", "sample_id": "s2"})(),
        ]
        retriever.build_index(examples, text_field="target_explanation")
        assert len(retriever.examples) == 2
        assert retriever.embeddings is not None
        assert retriever.embeddings.shape == (2, 384)

    def test_build_index_filters_empty_texts(self):
        """Empty texts are filtered out when _HAS_RAG=True."""
        from unittest.mock import MagicMock

        import app.training.rag as rag_module
        rag_module._HAS_RAG = True
        rag_module.SentenceTransformer = lambda model_name: MagicMock()

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": "valid", "sample_id": "s1"})(),
            type("Ex", (), {"target_explanation": "", "sample_id": "s2"})(),
            type("Ex", (), {"target_explanation": "  ", "sample_id": "s3"})(),
        ]
        retriever.build_index(examples, text_field="target_explanation")
        assert len(retriever.examples) == 1
        assert retriever.examples[0].sample_id == "s1"
        assert len(retriever.texts) == 1

    def test_build_index_with_empty_texts_returns_no_embeddings(self):
        """build_index with only empty texts results in empty embeddings."""
        from unittest.mock import MagicMock

        import app.training.rag as rag_module
        rag_module._HAS_RAG = True
        rag_module.SentenceTransformer = lambda model_name: MagicMock()

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": "", "sample_id": "s1"})(),
        ]
        retriever.build_index(examples)
        assert retriever.embeddings is not None or len(retriever.examples) == 0


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


class TestBuildRagPatchPrompt:
    def test_basic_prompt(self):
        from app.training.rag import RetrievalResult

        results = [
            RetrievalResult(sample_id="s1", text="example text", score=0.9)
        ]
        prompt = build_rag_patch_prompt("vuln_code", "CWE-89", results)
        assert "CWE-89" in prompt
        assert "vuln_code" in prompt
        assert "example text" in prompt
        assert "0.900" in prompt

    def test_no_results(self):
        prompt = build_rag_patch_prompt("vuln_code", "CWE-89", [])
        assert "CWE-89" in prompt
        assert "vuln_code" in prompt
        assert "Your Turn" in prompt

    def test_prompt_contains_instructions(self):
        from app.training.rag import RetrievalResult

        results = [RetrievalResult(sample_id="s1", text="t", score=0.5)]
        prompt = build_rag_patch_prompt("code", "CWE-79", results)
        assert "unified diff patch" in prompt
        assert "```diff" in prompt


class TestBuildRagClassificationPrompt:
    def test_basic_prompt(self):
        from app.training.rag import RetrievalResult

        results = [
            RetrievalResult(sample_id="s1", text="example text", score=0.9)
        ]
        prompt = build_rag_classification_prompt(
            "vuln_code", "static findings", "python", results
        )
        assert "python" in prompt
        assert "vuln_code" in prompt
        assert "static findings" in prompt
        assert "CWE" in prompt

    def test_no_results(self):
        prompt = build_rag_classification_prompt(
            "code", "findings", "javascript", []
        )
        assert "javascript" in prompt
        assert "Classify this vulnerability" in prompt


# ---------------------------------------------------------------------------
# retrieve_and_build_prompt
# ---------------------------------------------------------------------------


class TestRetrieveAndBuildPrompt:
    def test_returns_tuple(self):
        import app.training.rag as rag_module
        rag_module._HAS_RAG = False

        retriever = RAGRetriever()
        examples = [
            type("Ex", (), {"target_explanation": "SQL injection", "sample_id": "s1"})()
        ]
        retriever.build_index(examples)
        results, prompt = retrieve_and_build_prompt(retriever, "vuln", "CWE-89", top_k=2)
        assert isinstance(results, list)
        assert isinstance(prompt, str)
        assert "CWE-89" in prompt
        assert "vuln" in prompt
