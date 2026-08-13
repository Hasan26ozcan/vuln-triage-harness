"""Unit tests for ``app/evaluation/runner.py``.

These tests target the code paths NOT exercised by the integration tests in
``tests/integration/test_stage6_four_tier.py``.  Specifically they cover:

  * ``_compute_coverage`` — empty-list edge case (line 133).
  * ``EvaluationRunner.__init__`` — custom tier2/tier3/tier4 injection,
    local sandbox mode, unknown sandbox_mode ValueError, real LLM judge
    model (lines 248, 258, 267, 276).
  * ``load_samples`` / ``load_predictions`` — FileNotFoundError and
    blank-line skipping (lines 375, 379, 390, 394).

All tests run in pure-Python mode — no Docker, no model downloads, no LLM APIs.
"""

from __future__ import annotations

import pytest

from app.evaluation.runner import (
    EvalConfig,
    EvaluationRunner,
    _compute_coverage,
    compute_metrics,
    load_predictions,
    load_samples,
)
from app.evaluation.tier1_deterministic import DeterministicEvaluator
from app.evaluation.tier2_embedding_static import StaticSignalEvaluator
from app.evaluation.tier3_exec import (
    DockerSandboxRunner,
    ExecEvaluator,
    LocalSandboxRunner,
    MockSandboxRunner,
)
from app.evaluation.tier4_llm_judge import LlmJudge, MockLlmJudgeBackend
from app.schemas.prediction_eval import (
    EvalMetrics,
    ExecEvalResult,
    LlmJudgeScore,
    ModelPrediction,
    Tier1Result,
    Tier2Result,
)
from app.schemas.vuln import VulnSample

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CWES = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]


def _make_sample(
    sample_id: str = "s1",
    cwe_id: str = "CWE-89",
    severity: str = "high",
    vulnerable_code: str = "cursor.execute('SELECT * FROM t')",
    fixed_code: str | None = "cursor.execute('SELECT * FROM t WHERE id = %s')",
) -> VulnSample:
    """Build a minimal ``VulnSample`` for unit tests."""
    return VulnSample(
        id=sample_id,
        source="cve_real",
        repo_name="demo/repo",
        commit_sha="abc123",
        cve_id="CVE-2024-0001",
        cwe_id=cwe_id,
        severity=severity,
        language="python",
        vulnerable_code=vulnerable_code,
        fixed_code=fixed_code,
        static_findings=[],
        description="Test vulnerability.",
    )


def _make_prediction(
    sample_id: str = "s1",
    predicted_cwe: str = "CWE-89",
    patch_diff: str = "--- a/app.py\n+++ b/app.py\n- old\n+ new",
) -> ModelPrediction:
    return ModelPrediction(
        sample_id=sample_id,
        run_id="unit-test",
        predicted_cwe=predicted_cwe,
        predicted_severity="high",
        suggested_patch_diff=patch_diff,
        rationale="Fixed the vulnerability.",
    )


def _tier1_result(sample_id: str, cwe: str | None = "CWE-89") -> Tier1Result:
    return Tier1Result(
        sample_id=sample_id,
        predicted_cwe=cwe,
        confidence=0.9 if cwe else 0.0,
        matched_pattern="sql-injection-pattern" if cwe else None,
        num_patterns_matched=1 if cwe else 0,
    )


def _tier2_result(sample_id: str, cwe: str | None = "CWE-89") -> Tier2Result:
    return Tier2Result(
        sample_id=sample_id,
        predicted_cwe=cwe,
        confidence=0.8 if cwe else 0.0,
        signal_sources=["semgrep:python.sql-injection"],
        embedding_similarity=0.95,
    )


def _exec_result(
    prediction_id: str = "s1",
    patch_applies: bool = True,
    build_succeeds: bool | None = True,
    tests_pass: bool | None = True,
    hallucinated: bool = False,
) -> ExecEvalResult:
    return ExecEvalResult(
        prediction_id=prediction_id,
        patch_applies_cleanly=patch_applies,
        build_succeeds=build_succeeds,
        tests_pass_after_patch=tests_pass,
        cwe_classification_correct=True,
        hallucinated_cwe=hallucinated,
        hallucinated_function_ref=False,
    )


def _llm_score(prediction_id: str = "s1") -> LlmJudgeScore:
    return LlmJudgeScore(
        prediction_id=prediction_id,
        explanation_quality=0.9,
        patch_minimality=0.85,
        evaluator_model="mock-judge",
        rationale="Good explanation.",
    )


# ---------------------------------------------------------------------------
# _compute_coverage — empty list edge case (line 133)
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    """Tests for the _compute_coverage helper function."""

    def test_compute_coverage_with_values(self):
        """Non-None and None predictions are counted correctly."""
        preds = ["CWE-89", None, "CWE-79", None]
        result = _compute_coverage(preds)
        assert result == 0.5

    def test_compute_coverage_empty_list(self):
        """Empty prediction list → 0.0 coverage (line 133)."""
        assert _compute_coverage([]) == 0.0

    def test_compute_coverage_all_none(self):
        """All-None predictions → 0.0 coverage."""
        assert _compute_coverage([None, None, None]) == 0.0

    def test_compute_coverage_all_present(self):
        """All non-None predictions → 1.0 coverage."""
        assert _compute_coverage(["CWE-89", "CWE-79", "CWE-22"]) == 1.0


# ---------------------------------------------------------------------------
# compute_metrics — aggregate metrics, edge cases
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for the compute_metrics aggregate function."""

    def test_compute_metrics_all_correct(self):
        """Perfect predictions yield model_cwe_macro_f1=1.0."""
        samples = [
            _make_sample(cwe_id=cwe, vulnerable_code=f"code for {cwe}")
            for cwe in _CWES
        ]
        preds = [
            _make_prediction(predicted_cwe=cwe, patch_diff="--- a\n+++ b\n- a\n+ b")
            for cwe in _CWES
        ]
        tier1 = [_tier1_result(s.id, s.cwe_id) for s in samples]
        tier2 = [_tier2_result(s.id, s.cwe_id) for s in samples]
        execs = [_exec_result(s.id) for s in samples]
        llm_scores = [_llm_score(s.id) for s in samples]

        metrics = compute_metrics(samples, tier1, tier2, execs, llm_scores, preds)

        assert isinstance(metrics, EvalMetrics)
        assert metrics.num_samples == 6
        assert metrics.num_predictions == 6
        assert metrics.model_cwe_macro_f1 == 1.0
        assert metrics.exec_pass_rate == 1.0
        assert metrics.patch_applies_rate == 1.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.avg_explanation_quality == 0.9
        assert metrics.avg_patch_minimality == 0.85
        assert "CWE-89" in metrics.per_class

    def test_compute_metrics_empty_exec_results(self):
        """When exec_results is empty, exec metrics default to 0.0."""
        samples = [_make_sample(cwe_id=cwe) for cwe in _CWES[:2]]
        preds = [_make_prediction(predicted_cwe=cwe, patch_diff="patch") for cwe in _CWES[:2]]
        tier1 = [_tier1_result(s.id, s.cwe_id) for s in samples]
        tier2 = [_tier2_result(s.id, s.cwe_id) for s in samples]

        metrics = compute_metrics(samples, tier1, tier2, [], [], preds)

        assert metrics.exec_pass_rate == 0.0
        assert metrics.patch_applies_rate == 0.0
        assert metrics.hallucination_rate == 0.0
        assert metrics.avg_explanation_quality is None
        assert metrics.avg_patch_minimality is None

    def test_compute_metrics_empty_llm_scores(self):
        """When llm_judge_scores is empty, LLM judge averages are None."""
        samples = [_make_sample(cwe_id=cwe) for cwe in _CWES[:2]]
        preds = [_make_prediction(predicted_cwe=cwe) for cwe in _CWES[:2]]
        tier1 = [_tier1_result(s.id, s.cwe_id) for s in samples]
        tier2 = [_tier2_result(s.id, s.cwe_id) for s in samples]
        execs = [_exec_result(s.id) for s in samples]

        metrics = compute_metrics(samples, tier1, tier2, execs, [], preds)

        assert metrics.avg_explanation_quality is None
        assert metrics.avg_patch_minimality is None

    def test_compute_metrics_per_class_f1(self):
        """Per-class F1 stats are present for all valid CWEs."""
        samples = [_make_sample(cwe_id=cwe) for cwe in _CWES]
        # All predictions incorrect (hallucinated CWE)
        preds = [_make_prediction(predicted_cwe="CWE-9999") for _ in _CWES]
        tier1 = [_tier1_result(s.id, s.cwe_id) for s in samples]
        tier2 = [_tier2_result(s.id, s.cwe_id) for s in samples]

        metrics = compute_metrics(samples, tier1, tier2, [], [], preds)

        for cwe in _CWES:
            assert cwe in metrics.per_class
            stats = metrics.per_class[cwe]
            assert "precision" in stats
            assert "recall" in stats
            assert "f1" in stats

    def test_compute_metrics_hallucinated_predictions(self):
        """Hallucinated CWE predictions → hallucination_rate > 0."""
        samples = [_make_sample(cwe_id=cwe) for cwe in _CWES[:2]]
        preds = [
            _make_prediction(predicted_cwe="CWE-89"),  # correct
            _make_prediction(predicted_cwe="CWE-9999"),  # hallucinated
        ]
        tier1 = [_tier1_result(s.id, s.cwe_id) for s in samples]
        tier2 = [_tier2_result(s.id, s.cwe_id) for s in samples]
        execs = [_exec_result("s1", hallucinated=False), _exec_result("s2", hallucinated=True)]

        metrics = compute_metrics(samples, tier1, tier2, execs, [], preds)

        assert metrics.hallucination_rate == 0.5

    def test_compute_metrics_none_tier1_cwe(self):
        """Tier 1 predictions with None predicted_cwe are handled gracefully."""
        samples = [_make_sample(cwe_id="CWE-89"), _make_sample(cwe_id="CWE-79")]
        preds = [_make_prediction(predicted_cwe="CWE-89"), _make_prediction(predicted_cwe="CWE-79")]
        tier1 = [_tier1_result("s1", None), _tier1_result("s2", None)]
        tier2 = [_tier2_result(s.id, s.cwe_id) for s in samples]

        metrics = compute_metrics(samples, tier1, tier2, [], [], preds)

        assert metrics.tier1_coverage == 0.0  # no tier1 predictions made
        assert metrics.tier2_coverage == 1.0  # all tier2 predictions made


# ---------------------------------------------------------------------------
# EvaluationRunner.__init__ — tier injection and sandbox modes
# ---------------------------------------------------------------------------


class TestRunnerInit:
    """Tests for EvaluationRunner.__init__ tier injection logic."""

    def test_init_default_config(self):
        """Default construction uses EvalConfig defaults and mock sandbox."""
        runner = EvaluationRunner()
        assert isinstance(runner.config, EvalConfig)
        assert isinstance(runner._tier1, DeterministicEvaluator)
        assert isinstance(runner._tier2, StaticSignalEvaluator)
        assert isinstance(runner._tier3, ExecEvaluator)
        assert isinstance(runner._tier4, LlmJudge)

    def test_init_with_custom_tier1(self):
        """Custom tier1_evaluator is used instead of default."""
        custom_t1 = DeterministicEvaluator()
        runner = EvaluationRunner(tier1_evaluator=custom_t1)
        assert runner._tier1 is custom_t1

    def test_init_with_custom_tier2(self):
        """Custom tier2_evaluator is used (line 248)."""
        custom_t2 = StaticSignalEvaluator()
        runner = EvaluationRunner(tier2_evaluator=custom_t2)
        assert runner._tier2 is custom_t2

    def test_init_with_custom_tier3(self):
        """Custom tier3_evaluator is used instead of creating a new one."""
        custom_t3 = ExecEvaluator(sandbox_runner=MockSandboxRunner())
        runner = EvaluationRunner(tier3_evaluator=custom_t3)
        assert runner._tier3 is custom_t3

    def test_init_with_custom_tier4(self):
        """Custom tier4_evaluator is used instead of creating a new one."""
        custom_t4 = LlmJudge(backend=MockLlmJudgeBackend(fallback_explanation_quality=0.7))
        runner = EvaluationRunner(tier4_evaluator=custom_t4)
        assert runner._tier4 is custom_t4

    def test_init_local_sandbox_mode(self):
        """sandbox_mode='local' creates a LocalSandboxRunner (line 258)."""
        config = EvalConfig(base_model="test", sandbox_mode="local")
        runner = EvaluationRunner(config=config)
        assert isinstance(runner._tier3._sandbox, LocalSandboxRunner)

    def test_init_docker_sandbox_mode(self):
        """sandbox_mode='docker' creates a DockerSandboxRunner (line 261)."""
        config = EvalConfig(base_model="test", sandbox_mode="docker")
        runner = EvaluationRunner(config=config)
        assert isinstance(runner._tier3._sandbox, DockerSandboxRunner)

    def test_init_mock_sandbox_mode(self):
        """sandbox_mode='mock' creates a MockSandboxRunner."""
        config = EvalConfig(base_model="test", sandbox_mode="mock")
        runner = EvaluationRunner(config=config)
        assert isinstance(runner._tier3._sandbox, MockSandboxRunner)

    def test_init_skip_tier3_uses_mock_sandbox(self):
        """skip_tier3=True uses mock sandbox even in __init__."""
        config = EvalConfig(base_model="test", sandbox_mode="mock", skip_tier3=True)
        runner = EvaluationRunner(config=config)
        assert isinstance(runner._tier3._sandbox, MockSandboxRunner)

    def test_init_unknown_sandbox_mode_raises(self):
        """Unknown sandbox_mode raises ValueError (line 267)."""
        config = EvalConfig(base_model="test", sandbox_mode="invalid_mode")
        with pytest.raises(ValueError, match="Unknown sandbox_mode"):
            EvaluationRunner(config=config)

    def test_init_with_llm_judge_model(self):
        """llm_judge_model set creates LlmJudge with that model (line 276)."""
        # No OPENAI_API_KEY in the test environment → falls back to mock backend.
        config = EvalConfig(base_model="test", llm_judge_model="gpt-4o-mini")
        runner = EvaluationRunner(config=config)
        assert runner._tier4._model == "gpt-4o-mini"

    def test_init_skip_tier3_with_local_mode_uses_local(self):
        """skip_tier3=True with sandbox_mode='local' — local mode is checked first."""
        config = EvalConfig(base_model="test", sandbox_mode="local", skip_tier3=True)
        runner = EvaluationRunner(config=config)
        # The elif chain checks sandbox_mode == "local" before the mock/skip_tier3 branch
        assert isinstance(runner._tier3._sandbox, LocalSandboxRunner)


# ---------------------------------------------------------------------------
# load_samples — file loading edge cases
# ---------------------------------------------------------------------------


class TestLoadSamplesRunner:
    """Tests for load_samples covering FileNotFoundError and blank lines."""

    def test_load_samples_file_not_found(self):
        """Non-existent samples file raises FileNotFoundError (line 375)."""
        with pytest.raises(FileNotFoundError, match="Samples file not found"):
            load_samples("/nonexistent/path/samples.jsonl")

    def test_load_samples_skips_blank_lines(self, tmp_path):
        """Blank and whitespace-only lines are silently skipped (line 379)."""
        path = tmp_path / "samples.jsonl"
        path.write_text(
            '{"id": "s1", "source": "cve_real", "repo_name": "r1", '
            '"cwe_id": "CWE-89", "severity": "high", "language": "python", '
            '"vulnerable_code": "vuln", "description": "desc"}\n'
            "\n"
            "   \n"
            '{"id": "s2", "source": "cve_real", "repo_name": "r2", '
            '"cwe_id": "CWE-79", "severity": "medium", "language": "javascript", '
            '"vulnerable_code": "vuln2", "description": "desc2"}\n'
        )
        samples = load_samples(str(path))
        assert len(samples) == 2
        assert samples[0].id == "s1"
        assert samples[1].id == "s2"

    def test_load_samples_returns_vuln_samples(self, tmp_path):
        """Valid JSONL file returns a list of VulnSample objects."""
        path = tmp_path / "samples.jsonl"
        path.write_text(
            '{"id": "s1", "source": "cve_real", "repo_name": "r1", '
            '"cwe_id": "CWE-89", "severity": "high", "language": "python", '
            '"vulnerable_code": "vuln", "description": "desc"}\n'
        )
        samples = load_samples(str(path))
        assert len(samples) == 1
        assert isinstance(samples[0], VulnSample)
        assert samples[0].cwe_id == "CWE-89"


# ---------------------------------------------------------------------------
# load_predictions — file loading edge cases
# ---------------------------------------------------------------------------


class TestLoadPredictionsRunner:
    """Tests for load_predictions covering FileNotFoundError and blank lines."""

    def test_load_predictions_file_not_found(self):
        """Non-existent predictions file raises FileNotFoundError (line 390)."""
        with pytest.raises(FileNotFoundError, match="Predictions file not found"):
            load_predictions("/nonexistent/path/predictions.jsonl")

    def test_load_predictions_skips_blank_lines(self, tmp_path):
        """Blank and whitespace-only lines are silently skipped (line 394)."""
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            '{"sample_id": "s1", "run_id": "r1", "predicted_cwe": "CWE-89", '
            '"predicted_severity": "high", "suggested_patch_diff": "", '
            '"rationale": "test"}\n'
            "\n"
            "   \n"
            '{"sample_id": "s2", "run_id": "r1", "predicted_cwe": "CWE-79", '
            '"predicted_severity": "medium", "suggested_patch_diff": "", '
            '"rationale": "test"}\n'
        )
        preds = load_predictions(str(path))
        assert len(preds) == 2
        assert preds[0].sample_id == "s1"
        assert preds[1].sample_id == "s2"

    def test_load_predictions_returns_model_predictions(self, tmp_path):
        """Valid JSONL file returns a list of ModelPrediction objects."""
        path = tmp_path / "predictions.jsonl"
        path.write_text(
            '{"sample_id": "s1", "run_id": "r1", "predicted_cwe": "CWE-89", '
            '"predicted_severity": "high", "suggested_patch_diff": "patch", '
            '"rationale": "test"}\n'
        )
        preds = load_predictions(str(path))
        assert len(preds) == 1
        assert isinstance(preds[0], ModelPrediction)
        assert preds[0].predicted_cwe == "CWE-89"


# ---------------------------------------------------------------------------
# Acceptance: verify coverage of the 9 target lines
# ---------------------------------------------------------------------------


class TestRunnerAcceptance:
    """Smoke tests verifying the runner works end-to-end with mock backends."""

    def test_runner_run_with_mock_backends(self):
        """Full run() with mock backends produces a valid EvalReport."""
        samples = [
            _make_sample(cwe_id=cwe, vulnerable_code=f"code {i}")
            for i, cwe in enumerate(_CWES)
        ]
        preds = [
            _make_prediction(s.id, predicted_cwe=s.cwe_id, patch_diff="patch")
            for s in samples
        ]

        config = EvalConfig(base_model="mock-model", sandbox_mode="mock")
        runner = EvaluationRunner(config=config)
        report = runner.run(samples, preds)

        assert report.num_samples == 6
        assert report.num_predictions == 6
        assert len(report.tier1_results) == 6
        assert len(report.tier2_results) == 6
        assert len(report.exec_results) == 6
        assert len(report.llm_judge_scores) == 6
        assert report.metrics.model_cwe_macro_f1 == 1.0

    def test_runner_run_skip_tier3_and_tier4(self):
        """skip_tier3 + skip_tier4 produces empty exec and judge results."""
        samples = [_make_sample(cwe_id=cwe) for cwe in _CWES[:2]]
        preds = [_make_prediction(s.id, predicted_cwe=s.cwe_id) for s in samples]

        config = EvalConfig(
            base_model="mock", sandbox_mode="mock",
            skip_tier3=True, skip_tier4=True,
        )
        runner = EvaluationRunner(config=config)
        report = runner.run(samples, preds)

        assert report.exec_results == []
        assert report.llm_judge_scores == []
        assert report.metrics.exec_pass_rate == 0.0
        assert report.metrics.avg_explanation_quality is None
        assert report.metrics.avg_patch_minimality is None

    def test_runner_run_skip_tier3_only(self):
        """skip_tier3=True with tier4 active — exec empty but judge present."""
        samples = [_make_sample(cwe_id=cwe) for cwe in _CWES[:2]]
        preds = [_make_prediction(s.id, predicted_cwe=s.cwe_id) for s in samples]

        config = EvalConfig(base_model="mock", sandbox_mode="mock", skip_tier3=True)
        runner = EvaluationRunner(config=config)
        report = runner.run(samples, preds)

        assert report.exec_results == []
        assert len(report.llm_judge_scores) == 2

    def test_runner_run_skip_tier4_only(self):
        """skip_tier4=True with tier3 active — judge empty but exec present."""
        samples = [_make_sample(cwe_id=cwe) for cwe in _CWES[:2]]
        preds = [_make_prediction(s.id, predicted_cwe=s.cwe_id) for s in samples]

        config = EvalConfig(base_model="mock", sandbox_mode="mock", skip_tier4=True)
        runner = EvaluationRunner(config=config)
        report = runner.run(samples, preds)

        assert len(report.exec_results) == 2
        assert report.llm_judge_scores == []
        assert report.metrics.avg_explanation_quality is None
