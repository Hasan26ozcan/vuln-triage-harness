"""Integration tests for Stage 7 — regression / forgetting analysis.

These tests exercise the full Stage 7 pipeline end-to-end:

  1. Mock mode (MockBackend + MockCodeTestRunner) — no model download,
     no subprocess, fully deterministic.
  2. Local mode (MockBackend + LocalCodeTestRunner) — actually runs
     Python tests in a subprocess to detect forgetting/improvement.
  3. RegressionSummary construction from synthesized Stage 6 metrics +
     Stage 7 regression report.
  4. CLI ``stage7 --mock`` subcommand via Typer's CliRunner.

The bundled ``DEFAULT_GENERAL_TASKS`` (12 algorithm problems) are used
in tests that don't require actual code execution. For tests that need
real pass/fail discrimination, small single-task configs with
LocalCodeTestRunner are used.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from app.evaluation.backends import MockBackend
from app.evaluation.general_capability import (
    DEFAULT_GENERAL_TASKS,
    GeneralCapabilityTask,
    LocalCodeTestRunner,
    MockCodeTestRunner,
    RegressionConfig,
    build_regression_summary,
    run_regression_analysis,
)
from app.schemas.prediction_eval import (
    EvalMetrics,
    GeneralCapabilityMetrics,
    RegressionReport,
)

# ---------------------------------------------------------------------------
# Constants for subprocess-based tests
# ---------------------------------------------------------------------------

CORRECT_ADD = "def add(a, b):\n    return a + b\n"
WRONG_ADD = "def add(a, b):\n    return a - b\n"


def _make_add_task() -> GeneralCapabilityTask:
    """A simple add-two-numbers task for tests that need real execution."""
    return GeneralCapabilityTask(
        task_id="add_task",
        name="add",
        description="Add two integers and return the sum.",
        prompt_code="def add(a, b):",
        test_code="from solution import add\n\ndef test_solution():\n    assert add(2, 3) == 5\n",
        timeout_seconds=10,
    )


# ---------------------------------------------------------------------------
# 1. Mock mode end-to-end (no subprocess)
# ---------------------------------------------------------------------------


class TestMockModeEndToEnd:
    """Full regression analysis with MockBackend + MockCodeTestRunner."""

    def test_mock_all_pass_delta_zero(self):
        """Both models pass all tasks → delta = 0, no forgetting."""
        config = RegressionConfig(
            base_model="mock-base",
            tuned_model="mock-tuned",
            tasks=DEFAULT_GENERAL_TASKS,
            timeout_seconds=10,
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=True)

        report = run_regression_analysis(config, backend, backend, runner)

        assert isinstance(report, RegressionReport)
        assert report.run_id.startswith("stage7-")
        assert report.base_model == "mock-base"
        assert report.tuned_model == "mock-tuned"
        assert report.base_metrics.num_tasks == len(DEFAULT_GENERAL_TASKS)
        assert report.tuned_metrics.num_tasks == len(DEFAULT_GENERAL_TASKS)
        assert report.base_metrics.execution_accuracy == 1.0
        assert report.tuned_metrics.execution_accuracy == 1.0
        assert report.forgetting_delta == 0.0

    def test_mock_all_fail_delta_zero(self):
        """Both models fail all tasks → delta = 0 (no relative change)."""
        config = RegressionConfig(
            base_model="mock-base",
            tuned_model="mock-tuned",
            tasks=[_make_add_task()],
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=False)

        report = run_regression_analysis(config, backend, backend, runner)

        assert report.forgetting_delta == 0.0
        assert report.base_metrics.execution_accuracy == 0.0
        assert report.tuned_metrics.execution_accuracy == 0.0

    def test_mock_report_serializable(self):
        """The report should round-trip through JSON."""
        config = RegressionConfig(
            base_model="mock-base",
            tuned_model="mock-tuned",
            tasks=[_make_add_task()],
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=True)

        report = run_regression_analysis(config, backend, backend, runner)
        json_str = report.model_dump_json(indent=2)
        data = json.loads(json_str)

        assert data["run_id"]
        assert data["base_model"] == "mock-base"
        assert data["tuned_model"] == "mock-tuned"
        assert "forgetting_delta" in data
        assert "base_metrics" in data
        assert "tuned_metrics" in data
        assert "manifest" in data

    def test_mock_manifest_has_task_ids(self):
        """Manifest should list all task IDs."""
        config = RegressionConfig(
            base_model="mock-base",
            tuned_model="mock-tuned",
            tasks=[_make_add_task()],
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=True)

        report = run_regression_analysis(config, backend, backend, runner)

        manifest = report.manifest
        assert "task_ids" in manifest
        assert manifest["task_ids"] == ["add_task"]
        assert manifest["num_tasks"] == 1

    def test_default_tasks_bundled(self):
        """DEFAULT_GENERAL_TASKS should be a meaningful set for evaluation."""
        assert len(DEFAULT_GENERAL_TASKS) >= 10  # at least 10 problems
        names = {t.name for t in DEFAULT_GENERAL_TASKS}
        # Should cover a range of algorithm categories
        assert "factorial" in names
        assert "is_palindrome" in names
        assert "fibonacci" in names
        assert "binary_search" in names


# ---------------------------------------------------------------------------
# 2. Local runner end-to-end (real subprocess)
# ---------------------------------------------------------------------------


class TestLocalRunnerEndToEnd:
    """Regression analysis with MockBackend + LocalCodeTestRunner.

    These tests actually spawn Python subprocesses to run test code.
    They use MockBackend (no model download) with carefully crafted
    responses to simulate correct / incorrect code generation.
    """

    def test_no_forgetting_base_and_tuned_both_correct(self):
        """Both models generate correct code → delta = 0."""
        task = _make_add_task()
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=[task], timeout_seconds=10,
        )
        backend = MockBackend(
            responses={"def add": CORRECT_ADD},
            default=CORRECT_ADD,
        )
        runner = LocalCodeTestRunner(timeout_seconds=10)

        report = run_regression_analysis(config, backend, backend, runner)

        assert report.forgetting_delta == 0.0
        assert report.base_metrics.execution_accuracy == 1.0
        assert report.tuned_metrics.execution_accuracy == 1.0

    def test_forgetting_detected(self):
        """Base model writes correct code, tuned writes wrong code → negative delta."""
        task = _make_add_task()
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=[task], timeout_seconds=10,
        )
        base_backend = MockBackend(
            responses={"def add": CORRECT_ADD},
            default=CORRECT_ADD,
        )
        tuned_backend = MockBackend(
            responses={"def add": WRONG_ADD},
            default=WRONG_ADD,
        )
        runner = LocalCodeTestRunner(timeout_seconds=10)

        report = run_regression_analysis(
            config, base_backend, tuned_backend, runner,
        )

        assert report.forgetting_delta < 0
        assert report.base_metrics.num_passed == 1
        assert report.tuned_metrics.num_passed == 0
        # delta = 0.0 - 1.0 = -1.0
        assert report.forgetting_delta == -1.0

    def test_improvement_detected(self):
        """Base model writes wrong code, tuned writes correct code → positive delta."""
        task = _make_add_task()
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=[task], timeout_seconds=10,
        )
        base_backend = MockBackend(
            responses={"def add": WRONG_ADD},
            default=WRONG_ADD,
        )
        tuned_backend = MockBackend(
            responses={"def add": CORRECT_ADD},
            default=CORRECT_ADD,
        )
        runner = LocalCodeTestRunner(timeout_seconds=10)

        report = run_regression_analysis(
            config, base_backend, tuned_backend, runner,
        )

        assert report.forgetting_delta > 0
        assert report.base_metrics.num_passed == 0
        assert report.tuned_metrics.num_passed == 1
        assert report.forgetting_delta == 1.0

    def test_multi_task_partial_forgetting(self):
        """Two tasks: base passes both, tuned passes one → delta = -0.5."""
        task1 = _make_add_task()
        task2 = GeneralCapabilityTask(
            task_id="sub_task",
            name="subtract",
            description="Subtract b from a.",
            prompt_code="def subtract(a, b):",
            test_code=(
                "from solution import subtract\n"
                "\n"
                "def test_solution():\n"
                "    assert subtract(10, 3) == 7\n"
            ),
            timeout_seconds=10,
        )
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=[task1, task2], timeout_seconds=10,
        )
        correct_add = "def add(a, b):\n    return a + b\n"
        wrong_add = "def add(a, b):\n    return a - b\n"
        correct_sub = "def subtract(a, b):\n    return a - b\n"

        base_backend = MockBackend(
            responses={
                "def add": correct_add,
                "def subtract": correct_sub,
            },
            default=correct_add,
        )
        # Tuned: correct on subtract, wrong on add.
        tuned_backend = MockBackend(
            responses={
                "def add": wrong_add,
                "def subtract": correct_sub,
            },
            default=wrong_add,
        )
        runner = LocalCodeTestRunner(timeout_seconds=10)

        report = run_regression_analysis(
            config, base_backend, tuned_backend, runner,
        )

        assert report.base_metrics.num_passed == 2
        assert report.tuned_metrics.num_passed == 1
        assert report.forgetting_delta == round(0.5 - 1.0, 4)
        assert report.forgetting_delta < 0


# ---------------------------------------------------------------------------
# 3. RegressionSummary integration (Stage 6 + Stage 7)
# ---------------------------------------------------------------------------


def _make_eval_metrics(
    cwe_macro_f1: float = 0.8,
    exec_pass_rate: float = 0.5,
    hallucination_rate: float = 0.1,
    num_predictions: int = 12,
) -> EvalMetrics:
    """Create a realistic EvalMetrics for testing."""
    return EvalMetrics(
        num_samples=12,
        num_predictions=num_predictions,
        tier1_cwe_macro_f1=0.95,
        tier1_coverage=1.0,
        tier2_cwe_macro_f1=0.9,
        tier2_coverage=1.0,
        model_cwe_macro_f1=cwe_macro_f1,
        exec_pass_rate=exec_pass_rate,
        patch_applies_rate=0.9,
        build_succeeds_rate=0.95,
        hallucination_rate=hallucination_rate,
        avg_patch_coverage=0.95,
    )


def _make_regression_report(
    delta: float = -0.15,
) -> RegressionReport:
    """Create a RegressionReport for testing."""
    return RegressionReport(
        run_id="stage7_test",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        tuned_model="sft_qlora_r8",
        base_metrics=GeneralCapabilityMetrics(
            num_tasks=12, num_passed=10,
            execution_accuracy=0.8333,
            task_results=[],
        ),
        tuned_metrics=GeneralCapabilityMetrics(
            num_tasks=12, num_passed=8,
            execution_accuracy=0.6667,
            task_results=[],
        ),
        forgetting_delta=delta,
        manifest={"test": "data"},
    )


class TestRegressionSummaryIntegration:
    """Combine Stage 6 metrics + Stage 7 report → RegressionSummary."""

    def test_summary_with_forgetting(self):
        """Negative delta (forgetting) propagates to the summary."""
        metrics = _make_eval_metrics()
        report = _make_regression_report(delta=-0.20)

        summary = build_regression_summary(
            run_id="checkpoint_001",
            stage6_metrics=metrics,
            regression_report=report,
            inference_cost_usd=10.0,
            training_cost_usd=20.0,
        )

        assert summary.run_id == "checkpoint_001"
        assert summary.cwe_macro_f1 == 0.8
        assert summary.exec_pass_rate == 0.5
        assert summary.hallucination_rate == 0.1
        assert summary.general_capability_delta == -0.20
        # accepted = 12 * 0.5 = 6; total = 30; cost = 5.0
        assert summary.cost_per_accepted_patch_usd == round(30.0 / 6, 4)

    def test_summary_with_improvement(self):
        """Positive delta (improvement) propagates to the summary."""
        metrics = _make_eval_metrics()
        report = _make_regression_report(delta=0.10)

        summary = build_regression_summary(
            run_id="checkpoint_002",
            stage6_metrics=metrics,
            regression_report=report,
        )

        assert summary.general_capability_delta == 0.10

    def test_summary_json_serializable(self):
        """RegressionSummary should round-trip through JSON."""
        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=_make_eval_metrics(),
            regression_report=_make_regression_report(delta=-0.1),
            inference_cost_usd=5.0,
            training_cost_usd=5.0,
        )
        data = json.loads(summary.model_dump_json())
        assert data["run_id"] == "r1"
        assert data["general_capability_delta"] == -0.1
        assert data["cost_per_accepted_patch_usd"] > 0

    def test_summary_written_to_file(self, tmp_path):
        """RegressionSummary can be written to a JSON file and read back."""
        summary = build_regression_summary(
            run_id="file_test",
            stage6_metrics=_make_eval_metrics(),
            regression_report=_make_regression_report(delta=0.0),
        )
        path = tmp_path / "summary.json"
        path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")

        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["run_id"] == "file_test"
        assert loaded["cwe_macro_f1"] == 0.8
        assert loaded["general_capability_delta"] == 0.0

    def test_zero_passes_no_cost(self):
        """When no patches pass exec-eval, cost = 0.0."""
        metrics = _make_eval_metrics(exec_pass_rate=0.0)
        report = _make_regression_report(delta=-0.1)

        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=metrics,
            regression_report=report,
            inference_cost_usd=100.0,
            training_cost_usd=200.0,
        )

        assert summary.cost_per_accepted_patch_usd == 0.0


# ---------------------------------------------------------------------------
# 4. End-to-end mock pipeline (full DEFAULT_GENERAL_TASKS)
# ---------------------------------------------------------------------------


class TestFullPipelineMock:
    """Run the entire default task set in mock mode and validate the report."""

    def test_full_pipeline_mock(self):
        """All 12 default tasks, mock runner, all pass → delta = 0."""
        config = RegressionConfig(
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            tuned_model="sft_qlora_r8",
            tasks=DEFAULT_GENERAL_TASKS,
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=True)

        report = run_regression_analysis(config, backend, backend, runner)

        # Report structure
        assert report.run_id.startswith("stage7-")
        assert report.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert report.tuned_model == "sft_qlora_r8"

        # Metrics
        assert report.base_metrics.num_tasks == 12
        assert report.tuned_metrics.num_tasks == 12
        assert report.base_metrics.num_passed == 12
        assert report.tuned_metrics.num_passed == 12
        assert report.base_metrics.execution_accuracy == 1.0
        assert report.tuned_metrics.execution_accuracy == 1.0
        assert report.forgetting_delta == 0.0

        # Manifest
        assert report.manifest["num_tasks"] == 12
        assert len(report.manifest["task_ids"]) == 12

    def test_full_pipeline_all_fail(self):
        """All tasks fail for both models → delta = 0, accuracy = 0."""
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=DEFAULT_GENERAL_TASKS,
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=False)

        report = run_regression_analysis(config, backend, backend, runner)

        assert report.base_metrics.execution_accuracy == 0.0
        assert report.tuned_metrics.execution_accuracy == 0.0
        assert report.forgetting_delta == 0.0

    def test_full_pipeline_task_results_populated(self):
        """Each task result should be populated with task_id and name."""
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=DEFAULT_GENERAL_TASKS[:3],
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=True)

        report = run_regression_analysis(config, backend, backend, runner)

        results = report.base_metrics.task_results
        assert len(results) == 3
        for r in results:
            assert r.task_id
            assert r.name
            assert r.passed is True
            assert r.model_response  # non-empty

    def test_full_pipeline_json_dump_to_file(self, tmp_path):
        """The report can be written to a file and read back."""
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=DEFAULT_GENERAL_TASKS[:3],
        )
        backend = MockBackend(default="pass")
        runner = MockCodeTestRunner(default_passed=True)

        report = run_regression_analysis(config, backend, backend, runner)

        out = tmp_path / "regression_report.json"
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["run_id"]
        assert data["forgetting_delta"] == 0.0
        assert len(data["base_metrics"]["task_results"]) == 3


# ---------------------------------------------------------------------------
# 5. CLI integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Test the ``stage7`` Typer CLI subcommand."""

    def test_cli_mock_mode(self, tmp_path):
        """CLI stage7 --mock should run and write a report."""
        from app.evaluation.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage7",
                "--mock",
                "--tuned-model", "mock-tuned-checkpoint",
                "--output-dir", str(tmp_path / "stage7_out"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Stage 7" in result.output
        assert "Forgetting delta:" in result.output
        assert "No forgetting" in result.output  # mock: all pass → delta=0

        report_path = tmp_path / "stage7_out" / "regression_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["run_id"].startswith("stage7-")
        assert data["base_model"]
        assert data["tuned_model"] == "mock-tuned-checkpoint"

    def test_cli_missing_tuned_model(self, tmp_path):
        """CLI stage7 without --tuned-model should fail."""
        from app.evaluation.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["stage7", "--mock"],
        )
        assert result.exit_code != 0

    def test_cli_custom_models(self, tmp_path):
        """CLI with custom base and tuned model names."""
        from app.evaluation.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage7",
                "--mock",
                "--base-model", "my/base",
                "--tuned-model", "my/tuned",
                "--output-dir", str(tmp_path / "stage7_out"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "my/base" in result.output
        assert "my/tuned" in result.output
