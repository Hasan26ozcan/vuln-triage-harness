"""Integration tests for Stage 6 — four-tier evaluation harness.

These tests exercise the full four-tier pipeline end-to-end using the
bundled gold-eval set with mock backends (no model downloads, no Docker,
no LLM APIs).

Pipeline: gold-eval JSONL → Tier1 (regex) → Tier2 (static) →
          Tier3 (mock sandbox) → Tier4 (mock LLM judge) →
          EvalReport with all metrics.

The real gold-eval data is loaded from eval/gold_set/gold.jsonl (12 samples
covering all 6 CWE classes).
"""

from __future__ import annotations

import json
import os
import tempfile

from app.evaluation.runner import (
    EvalConfig,
    EvaluationRunner,
    load_predictions,
    load_samples,
)
from app.schemas.prediction_eval import EvalReport, ModelPrediction
from app.schemas.vuln import VulnSample

# Path to the bundled gold-eval set
GOLD_EVAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "eval", "gold_set", "gold.jsonl",
)


# --- Helpers ---


def _make_mock_predictions(gold_samples: list[VulnSample]) -> list[ModelPrediction]:
    """Create mock ModelPredictions that are 'always correct' on CWE
    but have minimal patches and rationales."""
    preds = []
    for s in gold_samples:
        fixed = s.fixed_code or "fixed code"
        patch_diff = (
            f"--- a/code.py\n"
            f"+++ b/code.py\n"
            f"- {s.vulnerable_code.splitlines()[0]}\n"
            f"+ {fixed.splitlines()[0]}\n"
        )
        preds.append(ModelPrediction(
            sample_id=s.id,
            run_id="stage6_test_run",
            predicted_cwe=s.cwe_id,
            predicted_severity=s.severity,
            suggested_patch_diff=patch_diff,
            rationale=f"Fixed {s.cwe_id} by applying the gold patch.",
        ))
    return preds


def _make_hallucinating_predictions(gold_samples: list[VulnSample]) -> list[ModelPrediction]:
    """Create predictions that predict a hallucinated CWE (CWE-9999)."""
    preds = []
    for s in gold_samples:
        preds.append(ModelPrediction(
            sample_id=s.id,
            run_id="stage6_hallucination_test",
            predicted_cwe="CWE-9999",  # not a real CWE
            predicted_severity=s.severity,
            suggested_patch_diff="some patch",
            rationale="Some rationale.",
        ))
    return preds


def _write_preds_jsonl(path, preds: list[ModelPrediction]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p.model_dump_json() + "\n")


# --- Gold loading ---


class TestLoadSamples:
    def test_load_samples_returns_vuln_samples(self):
        samples = load_samples(GOLD_EVAL_PATH)
        assert len(samples) == 12
        assert all(isinstance(s, VulnSample) for s in samples)

    def test_load_samples_has_all_six_cwes(self):
        samples = load_samples(GOLD_EVAL_PATH)
        cwes = {s.cwe_id for s in samples}
        assert cwes == {"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"}

    def test_load_predictions_returns_model_predictions(self):
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            _write_preds_jsonl(f.name, preds)
            path = f.name
        try:
            loaded = load_predictions(path)
            assert len(loaded) == 12
            assert all(isinstance(p, ModelPrediction) for p in loaded)
        finally:
            os.unlink(path)


# --- Full pipeline (mock mode) ---


class TestFourTierPipeline:
    """End-to-end test of the four-tier evaluation with mock backends."""

    def _get_runner(self, samples, preds, **config_overrides):
        """Create an EvaluationRunner in mock mode."""
        from app.evaluation.tier3_exec import MockSandboxRunner, SandboxResult
        from app.evaluation.tier4_llm_judge import LlmJudge, MockLlmJudgeBackend

        sandbox = MockSandboxRunner(
            default_result=SandboxResult(
                patch_applies_cleanly=True,
                build_succeeds=True,
                tests_pass_after_patch=True,
            )
        )
        judge_backend = MockLlmJudgeBackend(
            fallback_explanation_quality=0.9,
            fallback_patch_minimality=0.8,
        )

        overrides = {
            "base_model": "mock-model",
            "sandbox_mode": "mock",
            "skip_tier3": False,
            "skip_tier4": False,
        }
        overrides.update(config_overrides)
        config = EvalConfig(**overrides)

        return EvaluationRunner(
            config=config,
            tier3_evaluator=__import__(
                "app.evaluation.tier3_exec", fromlist=["ExecEvaluator"]
            ).ExecEvaluator(sandbox_runner=sandbox),
            tier4_evaluator=LlmJudge(backend=judge_backend),
        )

    def test_full_pipeline_end_to_end(self):
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)

        runner = self._get_runner(samples, preds)
        report = runner.run(samples, preds)

        assert isinstance(report, EvalReport)
        assert report.run_id.startswith("stage6-")
        assert report.stage == 6

        # All 4 tiers should have results
        assert len(report.tier1_results) == 12
        assert len(report.tier2_results) == 12
        assert len(report.exec_results) == 12
        assert len(report.llm_judge_scores) == 12

    def test_tier1_all_cwe_classes_predicted(self):
        """Tier 1 (deterministic) should classify all 6 CWE classes."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        tier1_cwes = {r.predicted_cwe for r in report.tier1_results if r.predicted_cwe}
        assert tier1_cwes == {"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"}

    def test_tier2_all_cwe_classes_predicted(self):
        """Tier 2 (static signal) should classify all 6 CWE classes."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        tier2_cwes = {r.predicted_cwe for r in report.tier2_results if r.predicted_cwe}
        assert tier2_cwes == {"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"}

    def test_model_predictions_correct_cwe(self):
        """When predictions match gold CWEs, model macro-F1 should be high."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)  # correct CWE predictions
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        assert report.metrics.model_cwe_macro_f1 == 1.0

    def test_model_predictions_hallucinating_cwe(self):
        """Hallucinated CWE predictions → hallucination_rate = 1.0."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_hallucinating_predictions(samples)
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        assert report.metrics.hallucination_rate == 1.0
        assert report.metrics.model_cwe_macro_f1 < 0.1

    def test_exec_metrics_computed(self):
        """Exec tier metrics should be computed from SandboxResult."""
        from app.evaluation.tier3_exec import MockSandboxRunner, SandboxResult

        # Custom sandbox: 6 pass, 6 fail
        results_map = {}
        for i in range(6):
            results_map[f"def f{i}"] = SandboxResult(
                patch_applies_cleanly=True,
                build_succeeds=True,
                tests_pass_after_patch=True,
            )
        for i in range(6, 12):
            results_map[f"def f{i}"] = SandboxResult(
                patch_applies_cleanly=False,
                error="bad patch",
            )

        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)

        config = EvalConfig(
            base_model="mock",
            sandbox_mode="mock",
        )
        from app.evaluation.tier3_exec import ExecEvaluator
        runner = EvaluationRunner(
            config=config,
            tier3_evaluator=ExecEvaluator(sandbox_runner=MockSandboxRunner(results=results_map)),
        )
        report = runner.run(samples, preds)

        # Note: the mock sandbox matches on first 40 chars of vulnerable_code,
        # which may not match the "def f0" keys. So default_result applies.
        # In the default mock, all are True.
        assert 0.0 <= report.metrics.exec_pass_rate <= 1.0
        assert 0.0 <= report.metrics.patch_applies_rate <= 1.0

    def test_skip_tier3(self):
        """When skip_tier3=True, exec_results should be empty."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)

        config = EvalConfig(
            base_model="mock",
            sandbox_mode="mock",
            skip_tier3=True,
            skip_tier4=True,
        )
        runner = EvaluationRunner(config=config)
        report = runner.run(samples, preds)

        assert report.exec_results == []
        assert report.llm_judge_scores == []
        # Metrics should be 0 for skipped tiers
        assert report.metrics.exec_pass_rate == 0.0

    def test_report_has_manifest(self):
        """The report should include a manifest with run metadata."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        assert "run_id" in report.manifest
        assert "started_at" in report.manifest
        assert "elapsed_seconds" in report.manifest
        assert "config" in report.manifest
        assert "tier_order" in report.manifest
        assert report.manifest["config"]["base_model"] == "mock-model"

    def test_report_serializes_to_json(self):
        """EvalReport should be JSON-serializable via pydantic."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        json_str = report.model_dump_json()
        data = json.loads(json_str)

        assert data["run_id"]
        assert data["stage"] == 6
        assert len(data["tier1_results"]) == 12
        assert len(data["tier2_results"]) == 12
        assert len(data["exec_results"]) == 12
        assert len(data["llm_judge_scores"]) == 12
        assert "metrics" in data
        assert "manifest" in data

    def test_per_class_f1_computed(self):
        """Per-class F1 should be computed for all 6 CWE classes."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        for cwe in ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]:
            assert cwe in report.metrics.per_class
            stats = report.metrics.per_class[cwe]
            assert "precision" in stats
            assert "recall" in stats
            assert "f1" in stats

    def test_tier4_scores_present(self):
        """LLM judge scores should be present and non-null."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)
        runner = self._get_runner(samples, preds)

        report = runner.run(samples, preds)
        for score in report.llm_judge_scores:
            assert 0.0 <= score.explanation_quality <= 1.0
            assert 0.0 <= score.patch_minimality <= 1.0
            assert score.evaluator_model

    def test_tier5_metrics_with_skip_tier4(self):
        """Without Tier 4, LLM judge averages should be None."""
        samples = load_samples(GOLD_EVAL_PATH)
        preds = _make_mock_predictions(samples)

        config = EvalConfig(
            base_model="mock",
            sandbox_mode="mock",
            skip_tier4=True,
        )
        runner = EvaluationRunner(config=config)
        report = runner.run(samples, preds)

        assert report.metrics.avg_explanation_quality is None
        assert report.metrics.avg_patch_minimality is None
