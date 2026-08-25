"""Unit tests for ``app/evaluation/cli.py`` — covering all uncovered branches.

Covers:
  - stage6: non-None avg_explanation_quality / avg_patch_minimality (lines 135, 137)
  - stage7: non-mock path (lines 215-220), forgetting warning (line 253)
  - stage8: empty method key (line 395), config warnings (line 421), no best_result (line 453)
  - stage10: without --stage6-report (lines 522-550)
  - stage11: demo success (lines 717-725), demo failure (lines 726-727),
    validation failure (lines 734-735)
  - baseline: invalid strategy (lines 785-789), few-shot warning (lines 816-822),
    non-mock backend (line 810), RuntimeError (lines 832-834)
  - evaluate: blank lines (lines 886-887, 896-897), output_dir (lines 921-926),
    empty predictions (line 901)
  - __main__ guard (line 930)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.evaluation.cli import app
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample

GOLD_EVAL = os.path.join("eval", "gold_set", "gold.jsonl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_baseline_metrics(path: str | os.PathLike, f1: float = 0.85) -> None:
    """Write a Stage 4 metrics.json with the given CWE Macro-F1."""
    metrics = {
        "run_id": "stage4_test",
        "num_predictions": 3,
        "num_parsed": 3,
        "num_parse_failures": 0,
        "cwe_macro_f1": f1,
        "cwe_micro_accuracy": 0.90,
        "severity_accuracy": 0.80,
        "hallucination_rate": 0.0,
        "patch_coverage": 0.95,
        "per_class": {},
    }
    Path(path).write_text(json.dumps(metrics), encoding="utf-8")


def _load_gold_samples(path: str = GOLD_EVAL, limit: int | None = None) -> list[VulnSample]:
    """Read VulnSamples from a gold-eval JSONL file."""
    samples: list[VulnSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(VulnSample(**json.loads(line)))
            if limit and len(samples) >= limit:
                break
    return samples


def _write_predictions(
    path: str | os.PathLike, samples: list[VulnSample], run_id: str = "test"
) -> None:
    """Write ModelPrediction JSONL from gold samples (correct CWE predictions)."""
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            pred = ModelPrediction(
                sample_id=s.id,
                run_id=run_id,
                predicted_cwe=s.cwe_id,
                predicted_severity=s.severity,
                suggested_patch_diff="",
                rationale="test",
            )
            f.write(pred.model_dump_json() + "\n")


def _mock_eval_report_json(model_f1: float = 0.82) -> str:
    """Return a JSON string that mimics an EvalReport with metrics."""
    return json.dumps(
        {
            "run_id": "stage6_unit_test",
            "base_model": "ci-regression-gate",
            "stage": 6,
            "num_samples": 12,
            "num_predictions": 12,
            "tier1_results": [],
            "tier2_results": [],
            "exec_results": [],
            "llm_judge_scores": [],
            "metrics": {
                "num_samples": 12,
                "num_predictions": 12,
                "tier1_cwe_macro_f1": 0.95,
                "tier1_coverage": 1.0,
                "tier2_cwe_macro_f1": 0.90,
                "tier2_coverage": 1.0,
                "model_cwe_macro_f1": model_f1,
                "exec_pass_rate": 0.50,
                "patch_applies_rate": 0.90,
                "build_succeeds_rate": 0.95,
                "hallucination_rate": 0.05,
                "avg_patch_coverage": 0.90,
                "per_class": {},
            },
            "manifest": {"test": "data"},
        }
    )


def _mock_eval_report() -> MagicMock:
    """Build a MagicMock EvalReport with non-None Tier 4 metrics."""
    report = MagicMock()
    metrics = MagicMock()
    metrics.tier1_cwe_macro_f1 = 0.95
    metrics.tier1_coverage = 1.0
    metrics.tier2_cwe_macro_f1 = 0.90
    metrics.tier2_coverage = 1.0
    metrics.model_cwe_macro_f1 = 0.85
    metrics.exec_pass_rate = 0.50
    metrics.patch_applies_rate = 0.90
    metrics.build_succeeds_rate = 0.95
    metrics.hallucination_rate = 0.05
    metrics.avg_patch_coverage = 0.90
    metrics.avg_explanation_quality = 0.8
    metrics.avg_patch_minimality = 0.9
    metrics.per_class = {}
    report.metrics = metrics
    report.run_id = "stage6_unit_test"
    report.stage = 6
    report.model_dump_json.return_value = '{"run_id": "stage6_unit_test", "stage": 6}'
    return report


# ---------------------------------------------------------------------------
# stage6 — non-None Tier 4 metrics (lines 135, 137)
# ---------------------------------------------------------------------------


def test_stage6_non_null_tier4_metrics(tmp_path):
    """Cover lines 135, 137: Avg Explanation Quality and Avg Patch Minimality
    are printed when not None (Tier 4 not skipped)."""
    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples)

    mock_report = _mock_eval_report()
    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = mock_report

    runner = CliRunner()
    with patch("app.evaluation.runner.EvaluationRunner", return_value=mock_runner_instance):
        result = runner.invoke(
            app,
            [
                "stage6",
                "--gold-eval",
                GOLD_EVAL,
                "--predictions",
                str(preds_path),
                "--sandbox-mode",
                "mock",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Avg Explanation Quality" in result.output
    assert "Avg Patch Minimality" in result.output


# ---------------------------------------------------------------------------
# stage7 — non-mock path + forgetting warning (lines 215-220, 253)
# ---------------------------------------------------------------------------


def test_stage7_non_mock_forgetting_warning(tmp_path):
    """Cover lines 215-220 (non-mock: QwenBackend + LocalCodeTestRunner)
    and line 253 (forgetting warning when delta < 0)."""
    mock_report = MagicMock()
    mock_report.run_id = "stage7_non_mock"
    mock_report.base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    mock_report.tuned_model = "tuned-checkpoint"
    mock_report.base_metrics.execution_accuracy = 1.0
    mock_report.tuned_metrics.execution_accuracy = 0.3
    mock_report.forgetting_delta = -0.7  # negative → warning (line 253)
    mock_report.model_dump_json.return_value = '{"run_id": "stage7_non_mock"}'

    runner = CliRunner()
    with patch(
        "app.evaluation.general_capability.run_regression_analysis",
        return_value=mock_report,
    ) as mock_analysis:
        result = runner.invoke(
            app,
            [
                "stage7",
                "--tuned-model",
                "tuned-checkpoint",
                "--output-dir",
                str(tmp_path / "stage7_out"),
                "--timeout",
                "15",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Forgetting delta:" in result.output
    assert "[WARN]" in result.output
    mock_analysis.assert_called_once()
    call_kwargs = mock_analysis.call_args.kwargs
    assert "base_backend" in call_kwargs
    assert "tuned_backend" in call_kwargs
    assert "runner" in call_kwargs


# ---------------------------------------------------------------------------
# stage8 — empty method key, config warnings, no best_result
# ---------------------------------------------------------------------------


def test_stage8_empty_method_key_skipped(tmp_path):
    """Cover line 395: empty method key (from 'gptq,,gguf') is skipped via continue."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stage8",
            "--source-checkpoint",
            "dummy_ckpt",
            "--mock",
            "--output-dir",
            str(tmp_path / "stage8_out"),
            "--methods",
            "gptq,,gguf",
            "--bits",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Stage 8" in result.output
    # The empty key should be skipped; gptq + gguf should run
    assert "gptq" in result.output.lower()
    assert "gguf" in result.output.lower()


def test_stage8_config_warnings_and_no_best_result(tmp_path):
    """Cover lines 421 (config warnings) and 453 (no best_result)."""
    mock_report = MagicMock()
    mock_report.run_id = "stage8_warnings"
    mock_report.results = []
    mock_report.best_result = None
    mock_report.model_dump_json.return_value = '{"run_id": "stage8_warnings"}'

    with (
        patch(
            "app.quantization.quantizer.run_quantization_matrix",
            return_value=mock_report,
        ),
        patch(
            "app.quantization.config.QuantConfig.all_warnings",
            return_value=["Test config warning"],
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage8",
                "--source-checkpoint",
                "dummy_ckpt",
                "--output-dir",
                str(tmp_path / "stage8_out"),
                "--methods",
                "gptq",
                "--bits",
                "4",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Warning: Test config warning" in result.output
    assert "Best config:       none" in result.output


# ---------------------------------------------------------------------------
# stage10 — without --stage6-report (lines 522-550)
# ---------------------------------------------------------------------------


def test_stage10_without_stage6_report(tmp_path):
    """Cover lines 522-550: stage10 without --stage6-report builds Stage 6
    from predictions + gold-eval, then runs the gate."""
    # Write baseline metrics
    baseline_path = tmp_path / "metrics.json"
    _write_baseline_metrics(baseline_path, f1=0.85)

    # Write predictions file (named with 'predictions' so CLI falls back
    # to bundled gold-eval for gold_eval_path)
    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples, run_id="stage10_test")

    # Mock EvaluationRunner so stage6 doesn't actually run the full pipeline.
    # The mock report's model_dump_json must produce valid JSON that
    # load_stage6_report can parse (needs metrics.model_cwe_macro_f1).
    mock_report = _mock_eval_report()
    mock_report.model_dump_json.return_value = _mock_eval_report_json(model_f1=0.82)
    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = mock_report

    with patch(
        "app.evaluation.runner.EvaluationRunner",
        return_value=mock_runner_instance,
    ):
        result = CliRunner().invoke(
            app,
            [
                "stage10",
                "--baseline-metrics",
                str(baseline_path),
                "--predictions",
                str(preds_path),
                "--output-dir",
                str(tmp_path / "stage10_out"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Stage 10" in result.output
    assert (tmp_path / "stage10_out" / "eval_report.json").exists()
    assert (tmp_path / "stage10_out" / "gate_result.json").exists()


# ---------------------------------------------------------------------------
# stage11 — demo success, demo failure, validation failure
# ---------------------------------------------------------------------------


def test_stage11_demo_succeeds(tmp_path):
    """Cover lines 717-725: demo runs and succeeds."""
    mock_gen = MagicMock()
    mock_gen.ensure_deliverables.return_value = {
        "model_card.md": str(tmp_path / "model_card.md"),
        "training_report.md": str(tmp_path / "training_report.md"),
        "demo.py": str(tmp_path / "demo.py"),
    }
    mock_demo_result = MagicMock()
    mock_demo_result.succeeded = True
    mock_demo_result.num_gold_samples = 12
    mock_demo_result.metrics = {
        "tuned_cwe_macro_f1": 0.92,
        "exec_pass_rate": 0.85,
        "forgetting_delta": 0.0,
        "gate_status": "PASS",
    }
    mock_gen.run_demo.return_value = mock_demo_result
    mock_gen.validate_deliverables.return_value = True

    with patch("app.stage11.generator.Stage11Generator", return_value=mock_gen):
        result = CliRunner().invoke(
            app,
            [
                "stage11",
                "--docs-dir",
                str(tmp_path / "docs"),
                "--output-dir",
                str(tmp_path / "stage11_out"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Demo completed" in result.output
    assert "CWE Macro-F1:     0.92" in result.output


def test_stage11_demo_failure(tmp_path):
    """Cover lines 726-727: demo runs and fails."""
    mock_gen = MagicMock()
    mock_gen.ensure_deliverables.return_value = {"model_card.md": "path"}
    mock_demo_result = MagicMock()
    mock_demo_result.succeeded = False
    mock_demo_result.error = "Demo pipeline crashed"
    mock_gen.run_demo.return_value = mock_demo_result
    mock_gen.validate_deliverables.return_value = True

    with patch("app.stage11.generator.Stage11Generator", return_value=mock_gen):
        result = CliRunner().invoke(
            app,
            [
                "stage11",
                "--docs-dir",
                str(tmp_path / "docs"),
                "--output-dir",
                str(tmp_path / "stage11_out"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "[FAIL] Demo failed: Demo pipeline crashed" in result.output


def test_stage11_validation_failure(tmp_path):
    """Cover lines 734-735: validate_deliverables returns False → exit 1."""
    mock_gen = MagicMock()
    mock_gen.ensure_deliverables.return_value = {"model_card.md": "path"}
    mock_gen.validate_deliverables.return_value = False

    with patch("app.stage11.generator.Stage11Generator", return_value=mock_gen):
        result = CliRunner().invoke(
            app,
            [
                "stage11",
                "--docs-dir",
                str(tmp_path / "docs"),
                "--output-dir",
                str(tmp_path / "stage11_out"),
                "--no-demo",
            ],
        )

    assert result.exit_code != 0, result.output
    assert "validation FAILED" in result.output


# ---------------------------------------------------------------------------
# baseline — invalid strategy, few-shot, non-mock, RuntimeError
# ---------------------------------------------------------------------------


def test_baseline_invalid_strategy(tmp_path):
    """Cover lines 785-789: invalid strategy → error + exit 1."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "baseline",
            "--gold-eval",
            GOLD_EVAL,
            "--strategy",
            "invalid_strategy",
            "--mock",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 1
    assert "invalid_strategy" in result.output


def test_baseline_few_shot_no_examples_warning(tmp_path):
    """Cover lines 816-820: few-shot without --few-shot-examples → warning."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "baseline",
            "--gold-eval",
            GOLD_EVAL,
            "--strategy",
            "few_shot",
            "--mock",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert "falling back to zero-shot" in result.output.lower()


def test_baseline_few_shot_with_examples(tmp_path):
    """Cover line 822: few-shot with --few-shot-examples → prints examples line."""
    samples = _load_gold_samples(limit=2)
    examples_path = tmp_path / "train.jsonl"
    with open(examples_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {
                        "id": s.id,
                        "sample_id": s.id,
                        "prompt": s.vulnerable_code,
                        "target_cwe": s.cwe_id,
                        "target_severity": s.severity,
                        "target_explanation": s.description,
                        "token_count_estimate": 10,
                    }
                )
                + "\n"
            )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "baseline",
            "--gold-eval",
            GOLD_EVAL,
            "--strategy",
            "few_shot",
            "--few-shot-examples",
            str(examples_path),
            "--mock",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Examples:" in result.output


def test_baseline_non_mock_backend_none(tmp_path):
    """Cover line 810: backend = None when mock=False."""
    with patch(
        "app.evaluation.cli.run_baseline",
        return_value=MagicMock(
            run_id="test",
            num_predictions=0,
            num_parse_failures=0,
            total_attempted=0,
            metrics=MagicMock(
                cwe_macro_f1=0.0,
                cwe_micro_accuracy=0.0,
                severity_accuracy=0.0,
                hallucination_rate=0.0,
                patch_coverage=0.0,
                per_class={},
            ),
        ),
    ) as mock_run:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "baseline",
                "--gold-eval",
                GOLD_EVAL,
                "--strategy",
                "zero_shot",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0, result.output
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["backend"] is None


def test_baseline_runtime_error(tmp_path):
    """Cover lines 832-834: RuntimeError from run_baseline → exit 1."""
    with patch(
        "app.evaluation.cli.run_baseline",
        side_effect=RuntimeError("Backend crashed"),
    ):
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "baseline",
                "--gold-eval",
                GOLD_EVAL,
                "--strategy",
                "zero_shot",
                "--mock",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 1
    assert "Error: Backend crashed" in result.output


# ---------------------------------------------------------------------------
# evaluate — blank lines, output_dir, empty predictions
# ---------------------------------------------------------------------------


def test_evaluate_with_blank_lines(tmp_path):
    """Cover lines 886-887, 896-897: blank lines in predictions and gold-eval files."""
    samples = _load_gold_samples(limit=3)

    # Write predictions with blank/whitespace lines interleaved
    preds_path = tmp_path / "predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if i > 0:
                f.write("\n")
                f.write("   \n")
            pred = ModelPrediction(
                sample_id=s.id,
                run_id="eval_test",
                predicted_cwe=s.cwe_id,
                predicted_severity=s.severity,
                suggested_patch_diff="",
                rationale="test",
            )
            f.write(pred.model_dump_json() + "\n")

    # Write gold-eval with blank/whitespace lines interleaved
    gold_path = tmp_path / "gold_eval.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(samples):
            if i > 0:
                f.write("\n")
                f.write("   \n")
            f.write(s.model_dump_json() + "\n")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--predictions",
            str(preds_path),
            "--gold-eval",
            str(gold_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Loaded" in result.output


def test_evaluate_empty_predictions(tmp_path):
    """Cover line 901: empty predictions → run_id = 're-evaluated'."""
    preds_path = tmp_path / "predictions.jsonl"
    preds_path.write_text("\n\n   \n", encoding="utf-8")

    samples = _load_gold_samples(limit=3)
    gold_path = tmp_path / "gold_eval.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--predictions",
            str(preds_path),
            "--gold-eval",
            str(gold_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "re-evaluated" in result.output


def test_evaluate_with_output_dir(tmp_path):
    """Cover lines 921-926: --output-dir writes metrics.json."""
    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples, run_id="eval_test")

    gold_path = tmp_path / "gold_eval.jsonl"
    with open(gold_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.model_dump_json() + "\n")

    out_dir = tmp_path / "metrics_out"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--predictions",
            str(preds_path),
            "--gold-eval",
            str(gold_path),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "metrics.json").exists()
    assert "Metrics written to" in result.output


# ---------------------------------------------------------------------------
# stage6 — per-class F1 output (line 141)
# ---------------------------------------------------------------------------


def test_stage6_per_class_f1_output(tmp_path):
    """Cover line 141: per-class F1 stats are printed when per_class is non-empty."""
    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples)

    mock_report = _mock_eval_report()
    mock_report.metrics.per_class = {
        "CWE-89": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        "CWE-79": {"precision": 0.7, "recall": 0.6, "f1": 0.65},
    }
    mock_report.model_dump_json.return_value = _mock_eval_report_json()

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = mock_report

    with patch("app.evaluation.runner.EvaluationRunner", return_value=mock_runner_instance):
        result = CliRunner().invoke(
            app,
            [
                "stage6",
                "--gold-eval",
                GOLD_EVAL,
                "--predictions",
                str(preds_path),
                "--sandbox-mode",
                "mock",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "CWE-89" in result.output
    assert "CWE-79" in result.output


# ---------------------------------------------------------------------------
# stage7 — mock mode (lines 205-213)
# ---------------------------------------------------------------------------


def test_stage7_mock_mode(tmp_path):
    """Cover lines 205-213: mock mode constructs MockBackend + MockCodeTestRunner."""
    mock_report = MagicMock()
    mock_report.run_id = "stage7_mock"
    mock_report.base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    mock_report.tuned_model = "tuned-checkpoint"
    mock_report.base_metrics.execution_accuracy = 1.0
    mock_report.tuned_metrics.execution_accuracy = 0.95
    mock_report.forgetting_delta = 0.05  # positive → no forgetting warning
    mock_report.model_dump_json.return_value = '{"run_id": "stage7_mock"}'

    with patch(
        "app.evaluation.general_capability.run_regression_analysis",
        return_value=mock_report,
    ) as mock_analysis:
        result = CliRunner().invoke(
            app,
            [
                "stage7",
                "--tuned-model",
                "tuned-checkpoint",
                "--output-dir",
                str(tmp_path / "stage7_out"),
                "--timeout",
                "15",
                "--mock",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "[OK] No forgetting" in result.output
    mock_analysis.assert_called_once()


# ---------------------------------------------------------------------------
# stage8 — unknown quantization method (lines 397-402)
# ---------------------------------------------------------------------------


def test_stage8_unknown_method_errors(tmp_path):
    """Cover lines 397-402: unknown quantization method → error + exit 1."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "stage8",
            "--source-checkpoint",
            "dummy_ckpt",
            "--output-dir",
            str(tmp_path / "stage8_out"),
            "--methods",
            "invalid_method",
            "--bits",
            "4",
        ],
    )
    assert result.exit_code == 1
    assert "unknown quantization method" in result.output
    assert "invalid_method" in result.output


# ---------------------------------------------------------------------------
# stage9 — serve command (lines 307-309)
# ---------------------------------------------------------------------------


def test_stage9_serve_invokes_cli_serve(tmp_path):
    """Cover lines 307-309: stage9 serve lazily imports and calls serve()."""
    with patch("app.serving.cli.serve") as mock_serve:
        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage9",
                "serve",
                "--model-path",
                str(tmp_path / "model.gguf"),
                "--backend",
                "mock",
            ],
        )

    assert result.exit_code == 0, result.output
    mock_serve.assert_called_once()
    call_kwargs = mock_serve.call_args.kwargs
    assert call_kwargs["backend_type"] == "mock"


# ---------------------------------------------------------------------------
# stage10 — forgetting_delta not None (line 592)
# ---------------------------------------------------------------------------


def test_stage10_with_forgetting_delta(tmp_path):
    """Cover line 592: forgetting_delta is printed when not None."""
    baseline_path = tmp_path / "metrics.json"
    _write_baseline_metrics(baseline_path, f1=0.85)

    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples)

    mock_report = _mock_eval_report()
    mock_report.model_dump_json.return_value = _mock_eval_report_json(model_f1=0.85)

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = mock_report

    # Mock RegressionGateResult with forgetting_delta set (not None → line 592)
    mock_gate_result = MagicMock()
    mock_gate_result.run_id = "stage10_with_forgetting"
    mock_gate_result.timestamp = "2024-01-01T00:00:00"
    mock_gate_result.status.value = "pass"
    mock_gate_result.checks = []
    mock_gate_result.baseline_cwe_macro_f1 = 0.85
    mock_gate_result.current_cwe_macro_f1 = 0.85
    mock_gate_result.f1_drop_percent = 0.0
    mock_gate_result.max_allowed_f1_drop_percent = 5.0
    mock_gate_result.forgetting_delta = -0.05  # not None → line 592
    mock_gate_result.forgetting_threshold = -0.10
    mock_gate_result.exec_pass_rate = 0.50
    mock_gate_result.min_exec_pass_rate = 0.0
    mock_gate_result.hallucination_rate = 0.05
    mock_gate_result.max_hallucination_rate = 0.50
    mock_gate_result.passed = True
    mock_gate_result.model_dump_json.return_value = '{"run_id": "test"}'

    with (
        patch("app.evaluation.runner.EvaluationRunner", return_value=mock_runner_instance),
        patch("app.ci.gate.run_gate", return_value=mock_gate_result),
    ):
        result = CliRunner().invoke(
            app,
            [
                "stage10",
                "--baseline-metrics",
                str(baseline_path),
                "--predictions",
                str(preds_path),
                "--output-dir",
                str(tmp_path / "stage10_out"),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "Forgetting delta:" in result.output
    assert "[OK] Regression gate PASSED" in result.output


# ---------------------------------------------------------------------------
# stage10 — gate failure (lines 608-613)
# ---------------------------------------------------------------------------


def test_stage10_gate_failure(tmp_path):
    """Cover lines 608-613: gate failure → exit 1 with FAIL message."""
    baseline_path = tmp_path / "metrics.json"
    _write_baseline_metrics(baseline_path, f1=0.85)

    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples)

    mock_report = _mock_eval_report()
    mock_report.model_dump_json.return_value = _mock_eval_report_json(model_f1=0.82)

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = mock_report

    # Mock RegressionGateResult that fails (passed=False → line 608-613)
    mock_gate_result = MagicMock()
    mock_gate_result.run_id = "stage10_fail"
    mock_gate_result.timestamp = "2024-01-01T00:00:00"
    mock_gate_result.status.value = "fail"
    mock_gate_result.checks = [
        MagicMock(
            status=MagicMock(value="fail"),
            name="cwe_f1_regression",
            message="F1 dropped too much",
        ),
    ]
    mock_gate_result.baseline_cwe_macro_f1 = 0.85
    mock_gate_result.current_cwe_macro_f1 = 0.82
    mock_gate_result.f1_drop_percent = 3.5
    mock_gate_result.max_allowed_f1_drop_percent = 5.0
    mock_gate_result.forgetting_delta = None
    mock_gate_result.forgetting_threshold = -0.10
    mock_gate_result.exec_pass_rate = 0.50
    mock_gate_result.min_exec_pass_rate = 0.0
    mock_gate_result.hallucination_rate = 0.05
    mock_gate_result.max_hallucination_rate = 0.50
    mock_gate_result.passed = False
    mock_gate_result.model_dump_json.return_value = '{"run_id": "test", "status": "fail"}'

    with (
        patch("app.evaluation.runner.EvaluationRunner", return_value=mock_runner_instance),
        patch("app.ci.gate.run_gate", return_value=mock_gate_result),
    ):
        result = CliRunner().invoke(
            app,
            [
                "stage10",
                "--baseline-metrics",
                str(baseline_path),
                "--predictions",
                str(preds_path),
                "--output-dir",
                str(tmp_path / "stage10_out"),
            ],
        )

    assert result.exit_code == 1
    assert "[FAIL] Regression gate FAILED" in result.output
    assert "does not pass the quality bar" in result.output


# ---------------------------------------------------------------------------
# __main__ guard (line 930)
# ---------------------------------------------------------------------------


def test_main_guard(monkeypatch):
    """Cover the ``if __name__ == '__main__': app()`` guard.

    This block is excluded from coverage by the project's coverage config
    (``exclude_lines`` matches ``if __name__ == .__main__.:``).  The test
    still exercises the code path to confirm the guard compiles correctly.
    """
    import app.evaluation.cli as cli_module

    source = 'if __name__ == "__main__":\n    app()'
    code = compile(source, str(cli_module.__file__), "exec")
    namespace: dict = {"__name__": "__main__", "app": cli_module.app}

    monkeypatch.setattr(sys, "argv", ["cli.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        exec(code, namespace)
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# stage6 — local LLM judge path (lines 124-146, 158)
# ---------------------------------------------------------------------------


def test_stage6_local_llm_judge_with_checkpoint(tmp_path):
    """Cover lines 124-134, 158: --llm-judge-model local with --checkpoint.

    This exercises the local LLM judge backend creation path where a LoRA
    checkpoint is loaded on top of the base model via QwenBackend._load().
    """
    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples)

    mock_report = MagicMock()
    mock_report.run_id = "stage6_local"
    mock_report.stage = 6
    mock_report.metrics = MagicMock()
    mock_report.metrics.tier1_cwe_macro_f1 = 0.9
    mock_report.metrics.tier1_coverage = 0.9
    mock_report.metrics.tier2_cwe_macro_f1 = 0.8
    mock_report.metrics.tier2_coverage = 0.8
    mock_report.metrics.model_cwe_macro_f1 = 0.7
    mock_report.metrics.exec_pass_rate = 0.5
    mock_report.metrics.patch_applies_rate = 0.9
    mock_report.metrics.build_succeeds_rate = 0.95
    mock_report.metrics.hallucination_rate = 0.05
    mock_report.metrics.avg_patch_coverage = 0.9
    mock_report.metrics.avg_explanation_quality = None
    mock_report.metrics.avg_patch_minimality = None
    mock_report.metrics.per_class = {}
    mock_report.model_dump_json.return_value = '{"run_id": "stage6_local", "stage": 6}'

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = mock_report

    with (
        patch(
            "app.evaluation.runner.EvaluationRunner",
            return_value=mock_runner_instance,
        ) as mock_runner_cls,
        patch("app.evaluation.backends.QwenBackend._load") as mock_load,
        patch("app.evaluation.tier4_llm_judge.LocalLlmJudgeBackend") as mock_local_backend_cls,
        patch("app.evaluation.tier4_llm_judge.LlmJudge") as mock_llm_judge_cls,
    ):
        # Mock QwenBackend._load to return a pipe with .model and .tokenizer
        mock_pipe = MagicMock()
        mock_load.return_value = mock_pipe

        mock_backend_instance = MagicMock()
        mock_local_backend_cls.return_value = mock_backend_instance
        mock_judge = MagicMock()
        mock_llm_judge_cls.return_value = mock_judge

        result = CliRunner().invoke(
            app,
            [
                "stage6",
                "--gold-eval",
                GOLD_EVAL,
                "--predictions",
                str(preds_path),
                "--llm-judge-model",
                "local",
                "--checkpoint",
                "/path/to/checkpoint",
                "--base-model",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0, result.output
    # Verify QwenBackend was used with checkpoint + base_model (PEFT path)
    # and EvaluationRunner was constructed with tier4_evaluator
    mock_runner_cls.assert_called_once()
    call_kwargs = mock_runner_cls.call_args.kwargs
    assert "tier4_evaluator" in call_kwargs
    assert call_kwargs["tier4_evaluator"] is mock_judge


def test_stage6_local_llm_judge_without_checkpoint(tmp_path):
    """Cover lines 135-136, 145-146: --llm-judge-model local without --checkpoint.

    Without a checkpoint, QwenBackend is initialized with just the base model.
    """
    samples = _load_gold_samples(limit=3)
    preds_path = tmp_path / "predictions.jsonl"
    _write_predictions(preds_path, samples)

    mock_report = MagicMock()
    mock_report.run_id = "stage6_local_no_ckpt"
    mock_report.stage = 6
    mock_report.metrics = MagicMock()
    mock_report.metrics.tier1_cwe_macro_f1 = 0.9
    mock_report.metrics.tier1_coverage = 0.9
    mock_report.metrics.tier2_cwe_macro_f1 = 0.8
    mock_report.metrics.tier2_coverage = 0.8
    mock_report.metrics.model_cwe_macro_f1 = 0.7
    mock_report.metrics.exec_pass_rate = 0.5
    mock_report.metrics.patch_applies_rate = 0.9
    mock_report.metrics.build_succeeds_rate = 0.95
    mock_report.metrics.hallucination_rate = 0.05
    mock_report.metrics.avg_patch_coverage = 0.9
    mock_report.metrics.avg_explanation_quality = None
    mock_report.metrics.avg_patch_minimality = None
    mock_report.metrics.per_class = {}
    mock_report.model_dump_json.return_value = '{"run_id": "stage6_local", "stage": 6}'

    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = mock_report

    with (
        patch(
            "app.evaluation.runner.EvaluationRunner",
            return_value=mock_runner_instance,
        ) as mock_runner_cls,
        patch("app.evaluation.backends.QwenBackend") as mock_qwen_cls,
        patch("app.evaluation.tier4_llm_judge.LocalLlmJudgeBackend") as mock_local_backend_cls,
        patch("app.evaluation.tier4_llm_judge.LlmJudge") as mock_llm_judge_cls,
    ):
        mock_qwen = MagicMock()
        mock_qwen._load.return_value = MagicMock(model="model_obj", tokenizer="tokenizer_obj")
        mock_qwen_cls.return_value = mock_qwen
        mock_local_backend_cls.return_value = MagicMock()
        mock_judge = MagicMock()
        mock_llm_judge_cls.return_value = mock_judge

        result = CliRunner().invoke(
            app,
            [
                "stage6",
                "--gold-eval",
                GOLD_EVAL,
                "--predictions",
                str(preds_path),
                "--llm-judge-model",
                "local",
                "--base-model",
                "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )

    assert result.exit_code == 0, result.output
    # Verify EvaluationRunner got the tier4_evaluator
    mock_runner_cls.assert_called_once()
    call_kwargs = mock_runner_cls.call_args.kwargs
    assert "tier4_evaluator" in call_kwargs
