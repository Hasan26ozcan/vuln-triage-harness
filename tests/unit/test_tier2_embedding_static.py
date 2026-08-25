"""Unit tests for Stage 6, Tier 2 — static signal + embedding similarity."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.tier2_embedding_static import (
    DEFAULT_RULE_TO_CWE,
    EmbeddingBackend,
    StaticSignalEvaluator,
    _cosine_similarity,
)
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample


def _make_sample(rule_id: str = "python.sqli-string-concat") -> VulnSample:
    return VulnSample(
        id="test_001",
        source="cve_real",
        repo_name="test/repo",
        cwe_id="CWE-89",
        severity="medium",
        language="python",
        vulnerable_code="def foo(): pass",
        description="test",
        static_findings=[],  # populated by caller
    )


class TestDefaultRuleToCWE:
    def test_all_six_cwes_covered(self):
        cwes = set(DEFAULT_RULE_TO_CWE.values())
        expected = {"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"}
        assert cwes == expected

    def test_sqli_rule_mapped(self):
        assert DEFAULT_RULE_TO_CWE["python.sqli-string-concat"] == "CWE-89"
        assert DEFAULT_RULE_TO_CWE["python.sqli-f-string"] == "CWE-89"

    def test_path_traversal_unsafe_open_mapped(self):
        assert DEFAULT_RULE_TO_CWE["python.path-traversal-unsafe-open"] == "CWE-22"

    def test_yaml_deser_mapped(self):
        assert DEFAULT_RULE_TO_CWE["python.deserialization-yaml"] == "CWE-502"


class TestStaticSignalEvaluator:
    def test_default_config(self):
        evaluator = StaticSignalEvaluator()
        assert evaluator.uses_embeddings is False
        assert len(evaluator.rule_to_cwe) == len(DEFAULT_RULE_TO_CWE)

    def test_custom_rule_mapping(self):
        custom = {"my.custom.rule": "CWE-89"}
        evaluator = StaticSignalEvaluator(rule_to_cwe=custom)
        assert evaluator.rule_to_cwe["my.custom.rule"] == "CWE-89"

    def test_no_embeddings_no_crash(self):
        """Static-only mode must not require sentence-transformers."""
        evaluator = StaticSignalEvaluator()
        assert evaluator.uses_embeddings is False

        sample = _make_sample()
        sample.static_findings = []
        result = evaluator.evaluate(sample)
        assert result.predicted_cwe is None
        assert result.confidence == 0.0
        assert result.embedding_similarity is None

    def test_single_finding_voted_correctly(self):
        from app.schemas.vuln import StaticFinding

        evaluator = StaticSignalEvaluator()
        sample = _make_sample()
        sample.static_findings = [
            StaticFinding(
                tool="semgrep",
                rule_id="python.sqli-string-concat",
                message="SQLi detected",
                line_range=(1, 2),
            )
        ]
        result = evaluator.evaluate(sample)
        assert result.predicted_cwe == "CWE-89"
        assert result.confidence == 1.0
        assert "semgrep:python.sqli-string-concat" in result.signal_sources

    def test_multiple_findings_vote(self):
        from app.schemas.vuln import StaticFinding

        evaluator = StaticSignalEvaluator()
        sample = _make_sample()
        sample.static_findings = [
            StaticFinding(
                tool="semgrep",
                rule_id="python.sqli-string-concat",
                message="sqli",
                line_range=(1, 2),
            ),
            StaticFinding(
                tool="semgrep", rule_id="python.sqli-f-string", message="sqli2", line_range=(3, 4)
            ),
            StaticFinding(
                tool="semgrep", rule_id="python.command-injection", message="cmd", line_range=(5, 6)
            ),
        ]
        result = evaluator.evaluate(sample)
        # CWE-89 has 2 votes, CWE-78 has 1 → winner is CWE-89
        assert result.predicted_cwe == "CWE-89"
        assert result.confidence == round(2 / 3, 4)

    def test_unknown_rule_id_no_vote(self):
        from app.schemas.vuln import StaticFinding

        evaluator = StaticSignalEvaluator()
        sample = _make_sample()
        sample.static_findings = [
            StaticFinding(
                tool="semgrep", rule_id="unknown.rule", message="unknown", line_range=(1, 2)
            ),
        ]
        result = evaluator.evaluate(sample)
        assert result.predicted_cwe is None
        assert result.confidence == 0.0
        assert "semgrep:unknown.rule" in result.signal_sources

    def test_evaluate_all_batch(self):
        from app.schemas.vuln import StaticFinding

        evaluator = StaticSignalEvaluator()
        s1 = _make_sample("python.sqli-string-concat")
        s1.static_findings = [
            StaticFinding(
                tool="semgrep", rule_id="python.sqli-string-concat", message="", line_range=(1, 2)
            )
        ]
        s2 = _make_sample("python.deserialization-pickle")
        s2.static_findings = [
            StaticFinding(
                tool="semgrep",
                rule_id="python.deserialization-pickle",
                message="",
                line_range=(1, 2),
            )
        ]
        results = evaluator.evaluate_all([s1, s2])
        assert results[0].predicted_cwe == "CWE-89"
        assert results[1].predicted_cwe == "CWE-502"

    def test_embedding_similarity_none_without_model(self):
        evaluator = StaticSignalEvaluator()  # no embedding_model
        sample = _make_sample()
        sample.static_findings = []
        sample.fixed_code = "fixed code"
        pred = ModelPrediction(
            sample_id="test_001",
            run_id="run_001",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            suggested_patch_diff="some patch",
            rationale="test rationale",
        )
        result = evaluator.evaluate(sample, prediction=pred)
        assert result.embedding_similarity is None


class TestEmbeddingBackend:
    def test_encode_raises_without_sentence_transformers(self):
        """If sentence-transformers isn't installed, encode() raises RuntimeError."""
        backend = EmbeddingBackend(model_name="test-model")
        # Force the inner import to fail regardless of whether the package
        # is actually installed.
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                backend.encode("hello")

    def test_encode_forces_import_error_raises_runtime_error(self):
        """Lines 91-92: when the import fails (forced via sys.modules),
        encode() raises RuntimeError with the install hint."""
        backend = EmbeddingBackend(model_name="test-model")
        # Force the inner import to fail regardless of whether the package
        # is actually installed.
        with patch.dict(sys.modules, {"sentence_transformers": None}):
            with pytest.raises(RuntimeError, match="sentence-transformers"):
                backend.encode("hello")

    def test_encode_successful_import_loads_model(self):
        """Lines 97-98: when sentence_transformers is importable, encode()
        instantiates SentenceTransformer and caches it."""
        backend = EmbeddingBackend(model_name="test-model")

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_model.encode.return_value = MagicMock()
        mock_model.encode.return_value.tolist = lambda: [0.1, 0.2, 0.3]
        mock_st.SentenceTransformer.return_value = mock_model

        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            result = backend.encode("hello")

        assert result == [0.1, 0.2, 0.3]
        # SentenceTransformer was called with the model name (line 98)
        mock_st.SentenceTransformer.assert_called_once_with("test-model")
        # Model was cached
        assert backend._model is mock_model


# ---------------------------------------------------------------------------
# _cosine_similarity — edge cases (lines 104-111)
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Tests for the _cosine_similarity helper function."""

    def test_cosine_similarity_empty_vectors(self):
        """Empty vectors → 0.0."""
        assert _cosine_similarity([], [1.0, 2.0]) == 0.0

    def test_cosine_similarity_mismatched_lengths(self):
        """Vectors of different lengths → 0.0."""
        assert _cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_cosine_similarity_zero_vector(self):
        """Zero vector (norm == 0) → 0.0."""
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert _cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_cosine_similarity_identical_vectors(self):
        """Identical vectors → 1.0."""
        assert _cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Orthogonal vectors → 0.0."""
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# _compute_embedding_similarity — branch coverage (lines 229-241)
# ---------------------------------------------------------------------------


class TestComputeEmbeddingSimilarity:
    """Tests for StaticSignalEvaluator._compute_embedding_similarity."""

    def test_no_fixed_code_returns_none(self):
        """When sample.fixed_code is None → return None (line 229-230)."""
        evaluator = StaticSignalEvaluator(embedding_model="dummy-model")
        # Replace the lazy embedder with a mock so encode() doesn't try to
        # load a real model.
        evaluator._embedder = MagicMock()
        sample = _make_sample()
        sample.static_findings = []
        sample.fixed_code = None  # no gold patch
        pred = ModelPrediction(
            sample_id="test_001",
            run_id="run_001",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            suggested_patch_diff="some patch",
            rationale="test",
        )
        result = evaluator._compute_embedding_similarity(sample, pred)
        assert result is None

    def test_no_prediction_returns_none(self):
        """When prediction is None → return None (line 231)."""
        evaluator = StaticSignalEvaluator(embedding_model="dummy-model")
        evaluator._embedder = MagicMock()
        sample = _make_sample()
        sample.static_findings = []
        sample.fixed_code = "fixed code"
        result = evaluator._compute_embedding_similarity(sample, None)
        assert result is None

    def test_empty_patch_returns_none(self):
        """When prediction.patch is empty → return None (line 231)."""
        evaluator = StaticSignalEvaluator(embedding_model="dummy-model")
        evaluator._embedder = MagicMock()
        sample = _make_sample()
        sample.static_findings = []
        sample.fixed_code = "fixed code"
        pred = ModelPrediction(
            sample_id="test_001",
            run_id="run_001",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            suggested_patch_diff="   ",  # whitespace-only
            rationale="test",
        )
        result = evaluator._compute_embedding_similarity(sample, pred)
        assert result is None

    def test_runtime_error_returns_none(self):
        """When embedder.encode raises RuntimeError → return None (lines 239-241)."""
        evaluator = StaticSignalEvaluator(embedding_model="dummy-model")
        mock_embedder = MagicMock()
        mock_embedder.encode.side_effect = RuntimeError("model not loaded")
        evaluator._embedder = mock_embedder
        sample = _make_sample()
        sample.static_findings = []
        sample.fixed_code = "fixed code"
        pred = ModelPrediction(
            sample_id="test_001",
            run_id="run_001",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            suggested_patch_diff="some patch",
            rationale="test",
        )
        result = evaluator._compute_embedding_similarity(sample, pred)
        assert result is None

    def test_valid_similarity_returns_float(self):
        """Valid inputs → cosine similarity float."""
        evaluator = StaticSignalEvaluator(embedding_model="dummy-model")
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [0.5, 0.5]
        evaluator._embedder = mock_embedder
        sample = _make_sample()
        sample.static_findings = []
        sample.fixed_code = "fixed code"
        pred = ModelPrediction(
            sample_id="test_001",
            run_id="run_001",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            suggested_patch_diff="some patch",
            rationale="test",
        )
        result = evaluator._compute_embedding_similarity(sample, pred)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
