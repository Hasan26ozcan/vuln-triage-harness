"""Integration tests for scripts/run_stage7_only.py (real-mode Stage 7 script).

Since the script normally requires a trained checkpoint and GPU/CUDA for
model loading, these tests exercise the pipeline by patching ``QwenBackend``
with a ``MockBackend`` wrapper and ``LocalCodeTestRunner`` with
``MockCodeTestRunner``. This validates:

  1. ``_load_stage6_metrics`` — loads metrics from various Stage 6 JSON shapes.
  2. ``run_stage7_real`` — end-to-end with mocked backends: report writing,
     manifest, RegressionSummary generation, file output.
  3. CLI argument parsing (argparse) with ``--help`` and checkpoint validation.

No model download or subprocess model execution is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from app.evaluation.backends import MockBackend
from app.evaluation.general_capability import MockCodeTestRunner

# ---------------------------------------------------------------------------
# Make scripts/ importable as a module
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_stage7_only  # noqa: E402  (import after sys.path tweak)

# ---------------------------------------------------------------------------
# Autouse fixture: bypass the real filesystem checkpoint check.
#
# run_stage7_real() now calls scripts.verify_checkpoint.verify_checkpoint()
# as a hard pre-flight gate (see the Stage 7 adapter-weights regression
# fix). These tests use fake paths like "/fake/checkpoint" that don't exist
# on disk, so we stub verify_checkpoint to return a fake-but-valid LoRA
# fingerprint, keeping these tests focused on report/manifest plumbing
# rather than real filesystem state. The guard itself is covered separately
# by tests/unit/test_verify_checkpoint.py and
# tests/unit/test_evaluation_backends.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_verify_checkpoint():
    fake_fingerprint = {
        "checkpoint_dir": "/fake/checkpoint",
        "checkpoint_type": "lora",
        "adapter_weight_file": "adapter_model.safetensors",
        "adapter_size_bytes": 12345,
        "adapter_sha256": "0" * 64,
    }
    with patch(
        "scripts.verify_checkpoint.verify_checkpoint",
        return_value=fake_fingerprint,
    ):
        yield fake_fingerprint

# ---------------------------------------------------------------------------
# Mock wrappers — accept production constructor signatures
# ---------------------------------------------------------------------------


class _MockQwenBackend(MockBackend):
    """MockBackend that accepts ``QwenBackend.__init__`` keyword arguments.

    The real ``QwenBackend`` takes ``model_name`` and ``base_model`` etc.
    We ignore those and forward ``default``/``responses`` to MockBackend.
    """

    def __init__(
        self,
        model_name: str = "mock",
        base_model: str | None = None,
        **kwargs,
    ):
        kwargs.pop("allow_base_fallback", None)
        default = kwargs.pop("default", "pass")
        responses = kwargs.pop("responses", None)
        super().__init__(responses=responses, default=default)
        self.model_name = model_name
        self.base_model = base_model
        # Real QwenBackend sets this once a LoRA adapter is actually merged.
        # The mock pretends it always "applied" successfully so
        # run_stage7_real's manifest-writing code has something to read.
        self.adapter_applied = True

    def _load(self):
        """No-op: MockBackend.generate() doesn't need a lazy-loaded pipeline,
        but run_stage7_real calls _load() eagerly to surface loading errors
        before running the full task suite."""
        return None


class _MockRunner(MockCodeTestRunner):
    """MockCodeTestRunner that accepts ``LocalCodeTestRunner(timeout_seconds=...)``."""

    def __init__(self, timeout_seconds: int = 30, default_passed: bool = True, **kwargs):
        super().__init__(default_passed=default_passed)


# Patch targets used throughout the tests.
_PATCH_TARGETS = {
    "QwenBackend": "app.evaluation.backends.QwenBackend",
    "LocalCodeTestRunner": "app.evaluation.general_capability.LocalCodeTestRunner",
}


def _patch_models(default_passed: bool = True):
    """Return a context manager that patches both QwenBackend and LocalCodeTestRunner."""
    return patch.multiple(
        "app.evaluation.general_capability",
        LocalCodeTestRunner=lambda **kw: _MockRunner(
            default_passed=default_passed, **kw
        ),
    ),


def _run_with_mocks(**kwargs):
    """Helper: call run_stage7_real with QwenBackend and LocalCodeTestRunner mocked."""
    default_passed = kwargs.pop("default_passed", True)
    with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
        "app.evaluation.general_capability.LocalCodeTestRunner",
        lambda timeout_seconds=30: _MockRunner(
            default_passed=default_passed, timeout_seconds=timeout_seconds,
        ),
    ):
        return run_stage7_only.run_stage7_real(
            default_passed=default_passed, **_kwargs_scrub(kwargs)
        )


def _kwargs_scrub(d):
    """Remove internal-only keys from kwargs before passing to run_stage7_real."""
    d.pop("default_passed", None)
    return d


# ---------------------------------------------------------------------------
# 1. _load_stage6_metrics
# ---------------------------------------------------------------------------


class TestLoadStage6Metrics:
    """Tests for the Stage 6 metrics loading helper."""

    def test_load_from_eval_report(self, tmp_path):
        """Loads metrics from a Stage 6 eval_report.json (nested 'metrics' key)."""
        report_data = {
            "run_id": "r1",
            "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "metrics": {
                "num_samples": 100,
                "num_predictions": 100,
                "tier1_cwe_macro_f1": 0.95,
                "tier1_coverage": 1.0,
                "tier2_cwe_macro_f1": 0.9,
                "tier2_coverage": 1.0,
                "model_cwe_macro_f1": 0.8,
                "exec_pass_rate": 0.5,
                "patch_applies_rate": 0.9,
                "build_succeeds_rate": 0.95,
                "hallucination_rate": 0.1,
                "avg_patch_coverage": 0.95,
            },
        }
        report_path = tmp_path / "eval_report.json"
        report_path.write_text(json.dumps(report_data), encoding="utf-8")

        result = run_stage7_only._load_stage6_metrics(str(report_path), None)
        assert result is not None
        assert result["num_samples"] == 100
        assert result["model_cwe_macro_f1"] == 0.8

    def test_load_from_eval_results(self, tmp_path):
        """Loads metrics from output/stage5/eval_results.json (nested 'stage6_metrics')."""
        results_data = {
            "stage6_metrics": {
                "num_samples": 50,
                "num_predictions": 50,
                "tier1_cwe_macro_f1": 0.9,
                "tier1_coverage": 1.0,
                "tier2_cwe_macro_f1": 0.85,
                "tier2_coverage": 0.98,
                "model_cwe_macro_f1": 0.75,
                "exec_pass_rate": 0.4,
                "patch_applies_rate": 0.88,
                "build_succeeds_rate": 0.92,
                "hallucination_rate": 0.15,
                "avg_patch_coverage": 0.9,
            },
        }
        metrics_path = tmp_path / "eval_results.json"
        metrics_path.write_text(json.dumps(results_data), encoding="utf-8")

        result = run_stage7_only._load_stage6_metrics(None, str(metrics_path))
        assert result is not None
        assert result["num_samples"] == 50
        assert result["exec_pass_rate"] == 0.4

    def test_load_from_flat_metrics(self, tmp_path):
        """Loads metrics from a flat metrics JSON (no nesting)."""
        flat_data = {
            "num_samples": 30,
            "num_predictions": 30,
            "tier1_cwe_macro_f1": 0.92,
            "tier1_coverage": 1.0,
            "tier2_cwe_macro_f1": 0.87,
            "tier2_coverage": 0.99,
            "model_cwe_macro_f1": 0.78,
            "exec_pass_rate": 0.6,
            "patch_applies_rate": 0.91,
            "build_succeeds_rate": 0.93,
            "hallucination_rate": 0.12,
            "avg_patch_coverage": 0.94,
        }
        metrics_path = tmp_path / "flat_metrics.json"
        metrics_path.write_text(json.dumps(flat_data), encoding="utf-8")

        result = run_stage7_only._load_stage6_metrics(None, str(metrics_path))
        assert result is not None
        assert result["num_samples"] == 30
        assert result["model_cwe_macro_f1"] == 0.78

    def test_no_paths_returns_none(self):
        """When neither path is provided, returns None."""
        assert run_stage7_only._load_stage6_metrics(None, None) is None

    def test_nonexistent_files_return_none(self, tmp_path):
        """When files don't exist, returns None."""
        result = run_stage7_only._load_stage6_metrics(
            str(tmp_path / "missing_report.json"),
            str(tmp_path / "missing_metrics.json"),
        )
        assert result is None

    def test_report_path_takes_precedence(self, tmp_path):
        """When both paths are given, the report path is used."""
        report_data = {
            "metrics": {
                "num_samples": 100,
                "num_predictions": 100,
                "tier1_cwe_macro_f1": 0.95,
                "tier1_coverage": 1.0,
                "tier2_cwe_macro_f1": 0.9,
                "tier2_coverage": 1.0,
                "model_cwe_macro_f1": 0.8,
                "exec_pass_rate": 0.5,
                "patch_applies_rate": 0.9,
                "build_succeeds_rate": 0.95,
                "hallucination_rate": 0.1,
                "avg_patch_coverage": 0.95,
            },
        }
        report_path = tmp_path / "eval_report.json"
        report_path.write_text(json.dumps(report_data), encoding="utf-8")

        result = run_stage7_only._load_stage6_metrics(
            str(report_path), str(tmp_path / "missing.json")
        )
        assert result["num_samples"] == 100


# ---------------------------------------------------------------------------
# 2. run_stage7_real with mocked backends
# ---------------------------------------------------------------------------


class TestRunStage7RealMocked:
    """End-to-end tests for run_stage7_real with QwenBackend patched out."""

    def test_writes_regression_report(self, tmp_path):
        """The regression report JSON is written to the output directory."""
        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
            )

        report_path = Path(result["report_path"])
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["run_id"].startswith("stage7-")
        assert data["base_model"] == "mock-base"
        assert data["tuned_model"] == "/fake/checkpoint"
        assert "forgetting_delta" in data
        assert "base_metrics" in data
        assert "tuned_metrics" in data
        assert "manifest" in data

    def test_writes_manifest(self, tmp_path):
        """A manifest.json with provenance info is written."""
        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
            )

        manifest_path = Path(result["manifest_path"])
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["script"] == "scripts/run_stage7_only.py"
        assert manifest["base_model"] == "mock-base"
        assert manifest["checkpoint"] == "/fake/checkpoint"
        assert manifest["timeout_seconds"] == 10

    def test_no_forgetting_delta_zero(self, tmp_path):
        """Mock all-pass → both models score 1.0 → delta = 0, no forgetting."""
        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
            )

        assert result["forgetting_delta"] == 0.0
        assert result["base_exec_accuracy"] == 1.0
        assert result["tuned_exec_accuracy"] == 1.0

    def test_all_fail_delta_zero(self, tmp_path):
        """Mock all-fail → both models score 0.0 → delta = 0 (no relative change)."""
        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=False, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
            )

        assert result["forgetting_delta"] == 0.0
        assert result["base_exec_accuracy"] == 0.0
        assert result["tuned_exec_accuracy"] == 0.0

    def test_summary_not_built_without_stage6(self, tmp_path):
        """Without Stage 6 inputs, RegressionSummary is not created."""
        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
            )

        assert result["summary_path"] is None
        assert not (tmp_path / "stage7_out" / "regression_summary.json").exists()

    def test_summary_built_with_stage6_report(self, tmp_path):
        """With a Stage 6 eval_report.json, RegressionSummary is built and written."""
        # Create a fake Stage 6 eval_report.json
        stage6_dir = tmp_path / "stage6"
        stage6_dir.mkdir()
        eval_report_path = stage6_dir / "eval_report.json"
        eval_report_data = {
            "run_id": "stage6_test",
            "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "metrics": {
                "num_samples": 12,
                "num_predictions": 12,
                "tier1_cwe_macro_f1": 0.95,
                "tier1_coverage": 1.0,
                "tier2_cwe_macro_f1": 0.9,
                "tier2_coverage": 1.0,
                "model_cwe_macro_f1": 0.8,
                "exec_pass_rate": 0.5,
                "patch_applies_rate": 0.9,
                "build_succeeds_rate": 0.95,
                "hallucination_rate": 0.1,
                "avg_patch_coverage": 0.95,
            },
        }
        eval_report_path.write_text(json.dumps(eval_report_data), encoding="utf-8")

        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
                stage6_report_path=str(eval_report_path),
                inference_cost_usd=10.0,
                training_cost_usd=20.0,
            )

        assert result["summary_path"] is not None
        summary_path = Path(result["summary_path"])
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["run_id"].startswith("stage7-")
        assert summary["cwe_macro_f1"] == 0.8
        assert summary["exec_pass_rate"] == 0.5
        assert summary["general_capability_delta"] == 0.0  # mock: no forgetting
        # accepted = 12 * 0.5 = 6; total = 30; cost = 5.0
        assert summary["cost_per_accepted_patch_usd"] == round(30.0 / 6, 4)

    def test_summary_built_with_stage6_metrics(self, tmp_path):
        """With a Stage 6 metrics JSON (nested), RegressionSummary is built."""
        stage6_dir = tmp_path / "stage6"
        stage6_dir.mkdir()
        metrics_path = stage6_dir / "eval_results.json"
        metrics_data = {
            "stage6_metrics": {
                "num_samples": 10,
                "num_predictions": 10,
                "tier1_cwe_macro_f1": 0.9,
                "tier1_coverage": 1.0,
                "tier2_cwe_macro_f1": 0.85,
                "tier2_coverage": 0.98,
                "model_cwe_macro_f1": 0.75,
                "exec_pass_rate": 0.4,
                "patch_applies_rate": 0.88,
                "build_succeeds_rate": 0.92,
                "hallucination_rate": 0.15,
                "avg_patch_coverage": 0.9,
            },
        }
        metrics_path.write_text(json.dumps(metrics_data), encoding="utf-8")

        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
                stage6_metrics_path=str(metrics_path),
            )

        assert result["summary_path"] is not None
        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        assert summary["exec_pass_rate"] == 0.4
        assert summary["cwe_macro_f1"] == 0.75

    def test_returns_summary_dict(self, tmp_path):
        """The return value is a dict with expected keys."""
        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
            )

        assert "run_id" in result
        assert "base_model" in result
        assert "tuned_model" in result
        assert "base_exec_accuracy" in result
        assert "tuned_exec_accuracy" in result
        assert "forgetting_delta" in result
        assert "report_path" in result
        assert "summary_path" in result
        assert "manifest_path" in result

    def test_cost_params_propagate(self, tmp_path):
        """Inference and training costs are passed through to the summary."""
        stage6_dir = tmp_path / "stage6"
        stage6_dir.mkdir()
        eval_report_path = stage6_dir / "eval_report.json"
        eval_report_data = {
            "metrics": {
                "num_samples": 10,
                "num_predictions": 10,
                "tier1_cwe_macro_f1": 0.9,
                "tier1_coverage": 1.0,
                "tier2_cwe_macro_f1": 0.85,
                "tier2_coverage": 0.98,
                "model_cwe_macro_f1": 0.75,
                "exec_pass_rate": 0.5,
                "patch_applies_rate": 0.9,
                "build_succeeds_rate": 0.95,
                "hallucination_rate": 0.1,
                "avg_patch_coverage": 0.95,
            },
        }
        eval_report_path.write_text(json.dumps(eval_report_data), encoding="utf-8")

        output_dir = str(tmp_path / "stage7_out")
        with patch("app.evaluation.backends.QwenBackend", _MockQwenBackend), patch(
            "app.evaluation.general_capability.LocalCodeTestRunner",
            lambda timeout_seconds=30: _MockRunner(
                default_passed=True, timeout_seconds=timeout_seconds,
            ),
        ):
            result = run_stage7_only.run_stage7_real(
                base_model="mock-base",
                checkpoint="/fake/checkpoint",
                output_dir=output_dir,
                timeout_seconds=10,
                stage6_report_path=str(eval_report_path),
                inference_cost_usd=50.0,
                training_cost_usd=50.0,
            )

        summary = json.loads(Path(result["summary_path"]).read_text(encoding="utf-8"))
        # accepted = 10 * 0.5 = 5; total = 100; cost = 20.0
        assert summary["cost_per_accepted_patch_usd"] == round(100.0 / 5, 4)


# ---------------------------------------------------------------------------
# 3. CLI argument parsing
# ---------------------------------------------------------------------------


class TestCLIArgumentParsing:
    """Test that argparse can parse the CLI arguments correctly."""

    def test_help_message(self):
        """--help should print usage and exit 0."""
        import subprocess

        result = subprocess.run(
            [sys.executable, str(_PROJECT_ROOT / "scripts" / "run_stage7_only.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "Stage 7" in result.stdout

    def test_checkpoint_not_found_exits_nonzero(self, tmp_path):
        """When the checkpoint doesn't exist, the script exits with code 1."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_stage7_only.py"),
                "--checkpoint",
                str(tmp_path / "nonexistent_checkpoint"),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 1
        output = result.stdout + result.stderr
        assert "Checkpoint not found" in output

    def test_custom_arguments_accepted(self, tmp_path):
        """All custom CLI arguments are accepted by argparse."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                str(_PROJECT_ROOT / "scripts" / "run_stage7_only.py"),
                "--base-model", "Qwen/Qwen2.5-Coder-7B-Instruct",
                "--checkpoint", "/nonexistent",
                "--output-dir", str(tmp_path / "out"),
                "--timeout", "30",
                "--stage6-report", str(tmp_path / "report.json"),
                "--stage6-metrics", str(tmp_path / "metrics.json"),
                "--inference-cost-usd", "5.0",
                "--training-cost-usd", "10.0",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        # Should fail at checkpoint existence check, not at argument parsing
        assert result.returncode == 1
        assert "Checkpoint not found" in (result.stdout + result.stderr)
