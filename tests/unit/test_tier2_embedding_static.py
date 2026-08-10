"""Unit tests for Stage 6, Tier 2 — static signal + embedding similarity."""

from app.evaluation.tier2_embedding_static import (
    DEFAULT_RULE_TO_CWE,
    EmbeddingBackend,
    StaticSignalEvaluator,
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
            StaticFinding(tool="semgrep", rule_id="python.sqli-string-concat",
                          message="sqli", line_range=(1, 2)),
            StaticFinding(tool="semgrep", rule_id="python.sqli-f-string",
                          message="sqli2", line_range=(3, 4)),
            StaticFinding(tool="semgrep", rule_id="python.command-injection",
                          message="cmd", line_range=(5, 6)),
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
            StaticFinding(tool="semgrep", rule_id="unknown.rule",
                          message="unknown", line_range=(1, 2)),
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
            StaticFinding(tool="semgrep", rule_id="python.sqli-string-concat",
                          message="", line_range=(1, 2))
        ]
        s2 = _make_sample("python.deserialization-pickle")
        s2.static_findings = [
            StaticFinding(tool="semgrep", rule_id="python.deserialization-pickle",
                          message="", line_range=(1, 2))
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
        import importlib
        if importlib.util.find_spec("sentence_transformers"):
            # sentence-transformers IS installed in this env — skip the
            # "not installed" path. We just verify encode() works.
            backend = EmbeddingBackend()
            result = backend.encode("hello")
            assert isinstance(result, list)
            assert len(result) > 0
        else:
            backend = EmbeddingBackend()
            try:
                backend.encode("hello")
                raise AssertionError("Should have raised RuntimeError")
            except RuntimeError as exc:
                assert "sentence-transformers" in str(exc)
