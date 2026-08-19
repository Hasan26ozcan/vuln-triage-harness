"""Integration tests for Stage 10 — CI/CD regression gate pipeline.

These tests exercise the full CI pipeline end-to-end using mock backends
(no GPU, no Docker, no network):

  1. Generate Stage 4 baseline metrics (mock backend).
  2. Generate Stage 6 eval report (mock sandbox).
  3. Generate Stage 7 regression report (mock mode).
  4. Run the Stage 10 regression gate on the artifacts.
  5. Test the ``stage10`` Typer CLI subcommand via CliRunner.
  6. Write a gate result to disk and verify it round-trips.
  7. Verify that a failing gate produces a non-zero exit code from the CLI.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.ci.config import RegressionGateConfig
from app.ci.gate import run_gate
from app.schemas.ci import GateStatus

# ---------------------------------------------------------------------------
# Fixtures — write synthetic Stage 4/6/7 artifacts to a temp directory
# ---------------------------------------------------------------------------


def _write_baseline_metrics(path: Path, f1: float = 0.85) -> None:
    """Write a Stage 4 metrics.json file.

    Mirrors the flat structure produced by ``run_baseline`` (which serializes the
    ``BaselineMetrics`` dataclass via ``asdict``), so ``cwe_macro_f1`` lives at
    the top level — exactly what ``load_baseline_metrics`` expects.
    """
    metrics = {
        "run_id": "stage4_mock",
        "num_predictions": 12,
        "num_parsed": 12,
        "num_parse_failures": 0,
        "cwe_macro_f1": f1,
        "cwe_micro_accuracy": 0.90,
        "severity_accuracy": 0.80,
        "hallucination_rate": 0.0,
        "patch_coverage": 0.95,
        "per_class": {},
    }
    path.write_text(json.dumps(metrics), encoding="utf-8")


def _write_stage6_report(path: Path, f1: float = 0.82, exec_rate: float = 0.50) -> None:
    """Write a Stage 6 eval_report.json file."""
    report = {
        "run_id": "stage6_mock",
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
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
            "model_cwe_macro_f1": f1,
            "exec_pass_rate": exec_rate,
            "patch_applies_rate": 0.90,
            "build_succeeds_rate": 0.95,
            "hallucination_rate": 0.05,
            "avg_patch_coverage": 0.95,
            "per_class": {},
        },
        "manifest": {"test": "data"},
    }
    path.write_text(json.dumps(report), encoding="utf-8")


def _write_stage7_report(path: Path, delta: float = -0.02) -> None:
    """Write a Stage 7 regression_report.json file."""
    report = {
        "run_id": "stage7_mock",
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "tuned_model": "ci-checkpoint",
        "base_metrics": {
            "num_tasks": 12,
            "num_passed": 10,
            "execution_accuracy": 0.8333,
            "task_results": [],
        },
        "tuned_metrics": {
            "num_tasks": 12,
            "num_passed": 10,
            "execution_accuracy": 0.8333,
            "task_results": [],
        },
        "forgetting_delta": delta,
        "manifest": {"test": "data"},
    }
    path.write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def ci_artifacts(tmp_path: Path) -> dict:
    """Create all three Stage 4/6/7 artifact files in a temp directory.

    Returns a dict of paths.
    """
    baseline_path = tmp_path / "metrics.json"
    stage6_path = tmp_path / "eval_report.json"
    stage7_path = tmp_path / "regression_report.json"

    _write_baseline_metrics(baseline_path, f1=0.85)
    _write_stage6_report(stage6_path, f1=0.82, exec_rate=0.50)
    _write_stage7_report(stage7_path, delta=-0.02)

    return {
        "baseline": str(baseline_path),
        "stage6": str(stage6_path),
        "stage7": str(stage7_path),
        "output_dir": str(tmp_path / "stage10_output"),
    }


# ---------------------------------------------------------------------------
# 1. End-to-end gate run from synthetic artifacts
# ---------------------------------------------------------------------------


class TestGateEndToEnd:
    """Full gate run with pre-written artifact files (no CLI)."""

    def test_gate_passes_on_synthetic_artifacts(self, ci_artifacts):
        """Baseline 0.85 → current 0.82 (3.5% drop) should pass the 5% gate."""
        config = RegressionGateConfig(
            baseline_metrics_path=ci_artifacts["baseline"],
            stage6_report_path=ci_artifacts["stage6"],
            stage7_report_path=ci_artifacts["stage7"],
        )
        result = run_gate(config)

        assert result.status == GateStatus.PASS
        assert result.passed is True
        assert result.baseline_cwe_macro_f1 == 0.85
        assert result.current_cwe_macro_f1 == 0.82
        # (0.85-0.82)/0.85*100 ≈ 3.53%
        assert 3.0 < result.f1_drop_percent < 4.0

        # All 4 checks present.
        assert len(result.checks) == 4
        check_names = {c.name for c in result.checks}
        assert "cwe_f1_regression" in check_names
        assert "forgetting_check" in check_names
        assert "exec_pass_rate" in check_names
        assert "hallucination_rate" in check_names

    def test_gate_result_written_to_disk(self, ci_artifacts):
        """The gate result can be serialized to JSON and written to a file."""
        config = RegressionGateConfig(
            baseline_metrics_path=ci_artifacts["baseline"],
            stage6_report_path=ci_artifacts["stage6"],
            stage7_report_path=ci_artifacts["stage7"],
        )
        result = run_gate(config)

        output_path = Path(ci_artifacts["output_dir"])
        output_path.mkdir(parents=True, exist_ok=True)
        gate_path = output_path / "gate_result.json"
        gate_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        loaded = json.loads(gate_path.read_text(encoding="utf-8"))
        assert loaded["status"] == "pass"
        assert loaded["run_id"].startswith("stage10-")
        assert loaded["baseline_cwe_macro_f1"] == 0.85
        assert loaded["current_cwe_macro_f1"] == 0.82

    def test_gate_fails_on_large_f1_drop(self, ci_artifacts):
        """If F1 drops more than 5%, the gate should fail."""
        # Overwrite stage6 report with a much lower F1.
        _write_stage6_report(Path(ci_artifacts["stage6"]), f1=0.50, exec_rate=0.50)

        config = RegressionGateConfig(
            baseline_metrics_path=ci_artifacts["baseline"],
            stage6_report_path=ci_artifacts["stage6"],
            stage7_report_path=ci_artifacts["stage7"],
        )
        result = run_gate(config)

        assert result.status == GateStatus.FAIL
        assert result.passed is False
        f1_check = next(c for c in result.checks if c.name == "cwe_f1_regression")
        assert f1_check.status == GateStatus.FAIL
        # Drop = (0.85 - 0.50) / 0.85 * 100 ≈ 41.2%
        assert f1_check.details["f1_drop_percent"] > 5.0


# ---------------------------------------------------------------------------
# 2. CLI integration — stage10 subcommand
# ---------------------------------------------------------------------------


class TestCLIStage10:
    """Test the ``stage10`` Typer CLI subcommand via CliRunner."""

    def test_cli_passes(self, ci_artifacts):
        """CLI stage10 with passing artifacts should exit 0."""
        from app.evaluation.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage10",
                "--baseline-metrics",
                ci_artifacts["baseline"],
                "--predictions",
                ci_artifacts["stage6"],  # dummy path; gate loads stage6-report
                "--stage6-report",
                ci_artifacts["stage6"],
                "--stage7-report",
                ci_artifacts["stage7"],
                "--output-dir",
                ci_artifacts["output_dir"],
            ],
        )
        assert result.exit_code == 0, result.output
        assert "PASSED" in result.output
        assert "stage10" in result.output.lower()

    def test_cli_fails_on_f1_drop(self, ci_artifacts):
        """CLI stage10 with a failing gate should exit non-zero."""
        from app.evaluation.cli import app

        # Overwrite stage6 report with a much lower F1.
        _write_stage6_report(Path(ci_artifacts["stage6"]), f1=0.40, exec_rate=0.50)

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage10",
                "--baseline-metrics",
                ci_artifacts["baseline"],
                "--predictions",
                ci_artifacts["stage6"],
                "--stage6-report",
                ci_artifacts["stage6"],
                "--stage7-report",
                ci_artifacts["stage7"],
                "--output-dir",
                ci_artifacts["output_dir"],
            ],
        )
        assert result.exit_code != 0, result.output
        assert "FAILED" in result.output

    def test_cli_without_stage7(self, ci_artifacts):
        """CLI stage10 without --stage7-report should skip forgetting check
        and still pass if F1 is within threshold."""
        from app.evaluation.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage10",
                "--baseline-metrics",
                ci_artifacts["baseline"],
                "--predictions",
                ci_artifacts["stage6"],
                "--stage6-report",
                ci_artifacts["stage6"],
                "--output-dir",
                ci_artifacts["output_dir"],
            ],
        )
        assert result.exit_code == 0, result.output
        assert "SKIP" in result.output.lower() or "PASSED" in result.output


# ---------------------------------------------------------------------------
# 3. End-to-end from real Stage 4 baseline in mock mode
# ---------------------------------------------------------------------------


class TestFullMockPipeline:
    """Generate real (mock-mode) Stage 4 baseline, then run Stage 6 + 7 + gate.

    This mirrors what the CI eval-gate job does: it calls the actual
    baseline, stage6, and stage7 CLI commands in mock mode, then feeds the
    resulting artifacts into the gate.
    """

    GOLD_EVAL = os.path.join("eval", "gold_set", "gold.jsonl")

    def test_full_mock_pipeline_passes(self, tmp_path: Path):
        """The full mock pipeline should produce artifacts that pass the gate.

        Steps:
          1. Stage 4 baseline (mock, zero-shot) → metrics.json
          2. Stage 6 eval (mock sandbox, predictions from step 1) → eval_report.json
          3. Stage 7 regression (mock) → regression_report.json
          4. Stage 10 gate → gate_result.json (should PASS)
        """
        from app.evaluation.cli import app

        cli = CliRunner()

        # Step 1: Stage 4 baseline (mock)
        stage4_dir = tmp_path / "stage4_baseline"
        result = cli.invoke(
            app,
            [
                "baseline",
                "--gold-eval",
                self.GOLD_EVAL,
                "--strategy",
                "zero_shot",
                "--mock",
                "--output-dir",
                str(stage4_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (stage4_dir / "metrics.json").exists()

        # Step 2: Stage 6 eval (mock sandbox) using Stage 4 predictions
        stage6_dir = tmp_path / "stage6"
        result = cli.invoke(
            app,
            [
                "stage6",
                "--gold-eval",
                self.GOLD_EVAL,
                "--predictions",
                str(stage4_dir / "predictions.jsonl"),
                "--sandbox-mode",
                "mock",
                "--skip-tier4",
                "--output-dir",
                str(stage6_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (stage6_dir / "eval_report.json").exists()

        # Step 3: Stage 7 regression (mock)
        stage7_dir = tmp_path / "stage7"
        result = cli.invoke(
            app,
            [
                "stage7",
                "--mock",
                "--tuned-model",
                "ci-checkpoint",
                "--output-dir",
                str(stage7_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (stage7_dir / "regression_report.json").exists()

        # Step 4: Stage 10 gate
        stage10_dir = tmp_path / "stage10"
        result = cli.invoke(
            app,
            [
                "stage10",
                "--baseline-metrics",
                str(stage4_dir / "metrics.json"),
                "--predictions",
                str(stage4_dir / "predictions.jsonl"),
                "--stage6-report",
                str(stage6_dir / "eval_report.json"),
                "--stage7-report",
                str(stage7_dir / "regression_report.json"),
                "--output-dir",
                str(stage10_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (stage10_dir / "gate_result.json").exists()

        # Verify the gate result.
        gate_data = json.loads((stage10_dir / "gate_result.json").read_text())
        assert gate_data["status"] == "pass"
        # Mock mode: baseline and current F1 should be very close.
        assert abs(gate_data["baseline_cwe_macro_f1"] - gate_data["current_cwe_macro_f1"]) < 0.1


# ---------------------------------------------------------------------------
# 4. Artifact loading through the public API
# ---------------------------------------------------------------------------


class TestArtifactIntegration:
    """Verify the loaders can read artifacts written by Stages 4-7."""

    def test_loaders_read_mock_artifacts(self, tmp_path: Path):
        """Loaders should read files written by the actual Stage 4/6/7 mock pipeline."""
        from typer.testing import CliRunner

        from app.ci.gate import load_baseline_metrics, load_stage6_report, load_stage7_report
        from app.evaluation.cli import app

        cli = CliRunner()

        # Generate real mock artifacts.
        stage4_dir = tmp_path / "stage4"
        cli.invoke(
            app,
            [
                "baseline",
                "--gold-eval",
                "eval/gold_set/gold.jsonl",
                "--mock",
                "--strategy",
                "zero_shot",
                "--output-dir",
                str(stage4_dir),
            ],
        )

        stage6_dir = tmp_path / "stage6"
        cli.invoke(
            app,
            [
                "stage6",
                "--gold-eval",
                "eval/gold_set/gold.jsonl",
                "--predictions",
                str(stage4_dir / "predictions.jsonl"),
                "--sandbox-mode",
                "mock",
                "--skip-tier4",
                "--output-dir",
                str(stage6_dir),
            ],
        )

        stage7_dir = tmp_path / "stage7"
        cli.invoke(
            app,
            [
                "stage7",
                "--mock",
                "--tuned-model",
                "test",
                "--output-dir",
                str(stage7_dir),
            ],
        )

        # Load through the public API.
        baseline = load_baseline_metrics(stage4_dir / "metrics.json")
        assert "cwe_macro_f1" in baseline
        assert baseline["cwe_macro_f1"] > 0

        report6 = load_stage6_report(stage6_dir / "eval_report.json")
        assert report6["metrics"]["model_cwe_macro_f1"] > 0

        report7 = load_stage7_report(stage7_dir / "regression_report.json")
        assert "forgetting_delta" in report7
