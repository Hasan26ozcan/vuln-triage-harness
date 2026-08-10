"""Unit tests for Stage 7 — regression / forgetting analysis.

Covers: task definitions, prompt builder, code extraction,
CodeTestRunner (mock + local), GeneralCapabilityEvaluator,
RegressionConfig, run_regression_analysis, cost estimation,
and build_regression_summary.
"""

from __future__ import annotations

import json as _json
import re as _re

from app.evaluation.backends import MockBackend
from app.evaluation.general_capability import (
    DEFAULT_GENERAL_TASKS,
    CodeTestResult,
    GeneralCapabilityEvaluator,
    GeneralCapabilityTask,
    LocalCodeTestRunner,
    MockCodeTestRunner,
    RegressionConfig,
    _extract_code,
    build_capability_prompt,
    build_regression_summary,
    estimate_cost_per_accepted_patch_usd,
    run_regression_analysis,
)
from app.schemas.prediction_eval import (
    EvalMetrics,
    GeneralCapabilityMetrics,
    RegressionReport,
    RegressionSummary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal ``ModelBackend`` stand-in for testing.

    Returns a fixed response for every prompt, optionally tracking calls.
    """

    def __init__(self, response: str = "pass", responses: dict | None = None):
        self._default = response
        self._responses = responses or {}
        self.call_count = 0
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.calls.append(prompt)
        for key, val in self._responses.items():
            if key in prompt:
                return val
        return self._default


def _make_task(
    task_id: str = "test_001",
    name: str = "test_fn",
    description: str = "Test problem.",
    prompt_code: str = "def test_fn(x):",
    test_code: str = (
        "from solution import test_fn\n"
        "\n"
        "def test_solution():\n"
        "    assert test_fn(1) == 1\n"
    ),
    timeout_seconds: int = 30,
) -> GeneralCapabilityTask:
    """Create a ``GeneralCapabilityTask`` for testing."""
    return GeneralCapabilityTask(
        task_id=task_id,
        name=name,
        description=description,
        prompt_code=prompt_code,
        test_code=test_code,
        timeout_seconds=timeout_seconds,
    )


def _make_add_task() -> GeneralCapabilityTask:
    """Create a simple 'add two numbers' task for subprocess-based tests.

    The correct implementation returns ``a + b``; the incorrect one
    returns ``a - b``. The test asserts ``add(2, 3) == 5``.
    """
    return GeneralCapabilityTask(
        task_id="add_task",
        name="add",
        description="Add two integers and return the sum.",
        prompt_code="def add(a, b):",
        test_code="from solution import add\n\ndef test_solution():\n    assert add(2, 3) == 5\n",
        timeout_seconds=10,
    )


CORRECT_ADD = "def add(a, b):\n    return a + b\n"
WRONG_ADD = "def add(a, b):\n    return a - b\n"


# ---------------------------------------------------------------------------
# Default task set
# ---------------------------------------------------------------------------


class TestDefaultGeneralTasks:
    def test_default_tasks_loaded(self):
        assert len(DEFAULT_GENERAL_TASKS) > 0

    def test_task_ids_unique(self):
        ids = [t.task_id for t in DEFAULT_GENERAL_TASKS]
        assert len(ids) == len(set(ids))

    def test_all_tasks_have_descriptions(self):
        for task in DEFAULT_GENERAL_TASKS:
            assert task.description
            assert len(task.description) > 10

    def test_all_tasks_have_prompt_code(self):
        for task in DEFAULT_GENERAL_TASKS:
            assert "def " in task.prompt_code

    def test_test_code_imports_solution(self):
        """Every task's test_code should import from the solution module."""
        for task in DEFAULT_GENERAL_TASKS:
            assert "from solution import" in task.test_code

    def test_test_code_compiles(self):
        """Every task's test_code should be valid Python syntax."""
        for task in DEFAULT_GENERAL_TASKS:
            compile(task.test_code, f"<{task.task_id}>", "exec")

    def test_test_code_uses_assert(self):
        """Every task should use assert statements for validation."""
        for task in DEFAULT_GENERAL_TASKS:
            assert "assert" in task.test_code

    def test_task_ids_follow_convention(self):
        for task in DEFAULT_GENERAL_TASKS:
            assert _re.match(r"^gc_\d{3}$", task.task_id), (
                f"task_id {task.task_id} doesn't match gc_NNN pattern"
            )

    def test_no_security_keywords_in_tasks(self):
        """Tasks should be general coding, not security-related."""
        security_keywords = [
            "sql", "injection", "xss", "deserializ", "pickle",
            "vulnerab", "exploit", "attack", "cwe", "owasp",
        ]
        for task in DEFAULT_GENERAL_TASKS:
            desc_lower = task.description.lower()
            for kw in security_keywords:
                assert kw not in desc_lower, (
                    f"Task {task.task_id} description contains security keyword '{kw}'"
                )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class TestBuildCapabilityPrompt:
    def test_prompt_contains_description(self):
        task = _make_task(description="Reverse a string.")
        prompt = build_capability_prompt(task)
        assert "Reverse a string" in prompt

    def test_prompt_contains_function_signature(self):
        task = _make_task(prompt_code="def reverse_string(s):")
        prompt = build_capability_prompt(task)
        assert "def reverse_string(s):" in prompt

    def test_prompt_instructs_only_code(self):
        task = _make_task()
        prompt = build_capability_prompt(task)
        assert "only" in prompt.lower()

    def test_prompt_no_explanations_requested(self):
        task = _make_task()
        prompt = build_capability_prompt(task)
        assert "explan" not in prompt.lower()


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_plain_code_returned_as_is(self):
        code = "def factorial(n):\n    return 1\n"
        assert _extract_code(code) == code.strip()

    def test_strips_python_fences(self):
        text = "```python\ndef factorial(n):\n    return 1\n```"
        result = _extract_code(text)
        assert "def factorial" in result
        assert "```" not in result

    def test_strips_plain_fences(self):
        text = "```\ndef factorial(n):\n    return 1\n```"
        result = _extract_code(text)
        assert "def factorial" in result
        assert "```" not in result

    def test_strips_with_leading_text(self):
        text = "Here is the code:\n```python\ndef foo(): pass\n```\nDone!"
        result = _extract_code(text)
        assert "def foo" in result
        assert "```" not in result
        assert "Here" not in result

    def test_empty_response(self):
        assert _extract_code("") == ""
        assert _extract_code("   ") == ""


# ---------------------------------------------------------------------------
# CodeTestResult
# ---------------------------------------------------------------------------


class TestCodeTestResult:
    def test_defaults(self):
        result = CodeTestResult(passed=True)
        assert result.passed is True
        assert result.output is None
        assert result.error is None

    def test_with_output(self):
        result = CodeTestResult(passed=False, output="stdout", error="traceback")
        assert result.passed is False
        assert result.output == "stdout"
        assert result.error == "traceback"


# ---------------------------------------------------------------------------
# MockCodeTestRunner
# ---------------------------------------------------------------------------


class TestMockCodeTestRunner:
    def test_default_all_pass(self):
        runner = MockCodeTestRunner()
        result = runner.run_code_test("code", "test", task_id="gc_001")
        assert result.passed is True
        assert result.error is None

    def test_default_all_fail(self):
        runner = MockCodeTestRunner(default_passed=False)
        result = runner.run_code_test("code", "test", task_id="gc_001")
        assert result.passed is False
        assert result.error is not None

    def test_per_task_results(self):
        runner = MockCodeTestRunner(
            default_passed=False,
            results={"gc_001": True, "gc_002": False},
        )
        assert runner.run_code_test("code", "test", task_id="gc_001").passed is True
        assert runner.run_code_test("code", "test", task_id="gc_002").passed is False
        # Default applies for unknown tasks
        assert runner.run_code_test("code", "test", task_id="gc_999").passed is False

    def test_per_task_no_task_id(self):
        runner = MockCodeTestRunner(default_passed=True)
        result = runner.run_code_test("code", "test")
        assert result.passed is True

    def test_call_count(self):
        runner = MockCodeTestRunner()
        runner.run_code_test("a", "b")
        runner.run_code_test("c", "d")
        assert runner.call_count == 2

    def test_last_task_id(self):
        runner = MockCodeTestRunner()
        runner.run_code_test("a", "b", task_id="gc_001")
        assert runner.last_task_id == "gc_001"


# ---------------------------------------------------------------------------
# LocalCodeTestRunner
# ---------------------------------------------------------------------------


class TestLocalCodeTestRunner:
    def test_passing_code(self):
        runner = LocalCodeTestRunner(timeout_seconds=10)
        code = "def add(a, b):\n    return a + b\n"
        test = "from solution import add\n\ndef test_solution():\n    assert add(2, 3) == 5\n"
        result = runner.run_code_test(code, test, task_id="test")
        assert result.passed is True

    def test_failing_test(self):
        runner = LocalCodeTestRunner(timeout_seconds=10)
        code = "def add(a, b):\n    return a - b\n"  # wrong impl
        test = "from solution import add\n\ndef test_solution():\n    assert add(2, 3) == 5\n"
        result = runner.run_code_test(code, test, task_id="test")
        assert result.passed is False

    def test_syntax_error_in_code(self):
        runner = LocalCodeTestRunner(timeout_seconds=10)
        code = "def broken(:\n    pass\n"
        test = "from solution import broken\n"
        result = runner.run_code_test(code, test, task_id="test")
        assert result.passed is False

    def test_no_import(self):
        """Generated code that doesn't define the expected function → test fails."""
        runner = LocalCodeTestRunner(timeout_seconds=10)
        code = "def wrong_name(x):\n    return x\n"
        test = "from solution import expected_name\n"
        result = runner.run_code_test(code, test, task_id="test")
        assert result.passed is False

    def test_extracts_code_from_fences(self):
        """Model output with markdown fences should still be tested correctly."""
        runner = LocalCodeTestRunner(timeout_seconds=10)
        code = "```python\ndef add(a, b):\n    return a + b\n```"
        test = "from solution import add\n\ndef test_solution():\n    assert add(2, 3) == 5\n"
        result = runner.run_code_test(code, test, task_id="test")
        assert result.passed is True


# ---------------------------------------------------------------------------
# GeneralCapabilityEvaluator
# ---------------------------------------------------------------------------


class TestGeneralCapabilityEvaluator:
    def test_default_tasks_used(self):
        """When tasks=None, DEFAULT_GENERAL_TASKS is used."""
        backend = _StubBackend("pass")
        runner = MockCodeTestRunner(default_passed=True)
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner)
        assert len(evaluator.tasks) == len(DEFAULT_GENERAL_TASKS)

    def test_custom_tasks_used(self):
        tasks = [_make_task("custom_1"), _make_task("custom_2")]
        backend = _StubBackend("pass")
        runner = MockCodeTestRunner(default_passed=True)
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=tasks)
        assert len(evaluator.tasks) == 2

    def test_evaluate_single_task(self):
        task = _make_task(
            "t1",
            test_code=(
                "from solution import test_fn\n"
                "\n"
                "def test_solution():\n"
                "    assert test_fn(1) == 1\n"
            ),
        )
        backend = _StubBackend("def test_fn(x):\n    return x\n")
        runner = MockCodeTestRunner(default_passed=True)
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=[task])
        result = evaluator.evaluate(task)
        assert result.task_id == "t1"
        assert result.name == "test_fn"
        assert result.passed is True
        assert result.model_response

    def test_evaluate_all_metrics(self):
        tasks = [
            _make_task("t1"),
            _make_task("t2"),
            _make_task("t3"),
        ]
        backend = _StubBackend("def test_fn(x):\n    return x\n")
        runner = MockCodeTestRunner(
            default_passed=True,
            results={"t1": True, "t2": True, "t3": False},
        )
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=tasks)
        metrics = evaluator.evaluate_all()
        assert metrics.num_tasks == 3
        assert metrics.num_passed == 2
        assert metrics.execution_accuracy == round(2 / 3, 4)

    def test_evaluate_all_all_pass(self):
        tasks = [_make_task("t1"), _make_task("t2")]
        backend = _StubBackend("pass")
        runner = MockCodeTestRunner(default_passed=True)
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=tasks)
        metrics = evaluator.evaluate_all()
        assert metrics.num_passed == 2
        assert metrics.execution_accuracy == 1.0

    def test_evaluate_all_all_fail(self):
        tasks = [_make_task("t1"), _make_task("t2")]
        backend = _StubBackend("pass")
        runner = MockCodeTestRunner(default_passed=False)
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=tasks)
        metrics = evaluator.evaluate_all()
        assert metrics.num_passed == 0
        assert metrics.execution_accuracy == 0.0

    def test_evaluate_all_empty_tasks(self):
        backend = _StubBackend("pass")
        runner = MockCodeTestRunner()
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=[])
        metrics = evaluator.evaluate_all()
        assert metrics.num_tasks == 0
        assert metrics.num_passed == 0
        assert metrics.execution_accuracy == 0.0

    def test_evaluate_all_task_results_populated(self):
        tasks = [_make_task("t1"), _make_task("t2")]
        backend = _StubBackend("pass")
        runner = MockCodeTestRunner(default_passed=True)
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=tasks)
        metrics = evaluator.evaluate_all()
        assert len(metrics.task_results) == 2
        for r in metrics.task_results:
            assert r.task_id in ("t1", "t2")

    def test_backend_called_for_each_task(self):
        tasks = [_make_task("t1"), _make_task("t2"), _make_task("t3")]
        backend = _StubBackend("pass")
        runner = MockCodeTestRunner(default_passed=True)
        evaluator = GeneralCapabilityEvaluator(backend=backend, runner=runner, tasks=tasks)
        evaluator.evaluate_all()
        assert backend.call_count == 3


# ---------------------------------------------------------------------------
# RegressionConfig
# ---------------------------------------------------------------------------


class TestRegressionConfig:
    def test_defaults(self):
        config = RegressionConfig()
        assert config.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert config.tuned_model == "tuned"
        assert config.tasks is None
        assert config.timeout_seconds == 30

    def test_custom_values(self):
        custom_tasks = [_make_task("custom")]
        config = RegressionConfig(
            base_model="my/base",
            tuned_model="my/tuned",
            tasks=custom_tasks,
            timeout_seconds=60,
        )
        assert config.base_model == "my/base"
        assert config.tuned_model == "my/tuned"
        assert config.tasks == custom_tasks
        assert config.timeout_seconds == 60


# ---------------------------------------------------------------------------
# run_regression_analysis
# ---------------------------------------------------------------------------


class TestRunRegressionAnalysis:
    def test_no_forgetting_delta_zero(self):
        """Base and tuned both pass all tasks → delta = 0.

        Uses MockCodeTestRunner (returns True for all tasks).
        Both backends get the same code-runner results.
        """
        tasks = [_make_task("t1"), _make_task("t2")]
        config = RegressionConfig(base_model="base", tuned_model="tuned", tasks=tasks)
        runner = MockCodeTestRunner(default_passed=True)
        backend = MockBackend(default="pass")

        report = run_regression_analysis(config, backend, backend, runner)

        assert report.forgetting_delta == 0.0
        assert report.base_metrics.execution_accuracy == 1.0
        assert report.tuned_metrics.execution_accuracy == 1.0
        assert report.base_metrics.num_passed == 2
        assert report.tuned_metrics.num_passed == 2

    def test_forgetting_detected_negative_delta(self):
        """Tuned model's code is wrong → lower accuracy → negative delta.

        Uses MockBackend returning different code per model +
        LocalCodeTestRunner (actually runs the tests).
        """
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

        report = run_regression_analysis(config, base_backend, tuned_backend, runner)

        assert report.forgetting_delta < 0
        assert report.base_metrics.execution_accuracy == 1.0
        assert report.tuned_metrics.execution_accuracy == 0.0

    def test_improvement_positive_delta(self):
        """Tuned model's code is correct, base model's is wrong → positive delta."""
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

        report = run_regression_analysis(config, base_backend, tuned_backend, runner)

        assert report.forgetting_delta > 0
        assert report.base_metrics.execution_accuracy == 0.0
        assert report.tuned_metrics.execution_accuracy == 1.0

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

        report = run_regression_analysis(config, base_backend, tuned_backend, runner)

        assert report.base_metrics.num_passed == 2
        assert report.tuned_metrics.num_passed == 1
        assert report.forgetting_delta == round(0.5 - 1.0, 4)
        assert report.forgetting_delta < 0

    def test_empty_tasks_no_forgetting(self):
        """With 0 tasks, accuracy=0 and delta=0."""
        config = RegressionConfig(base_model="base", tuned_model="tuned", tasks=[])
        runner = MockCodeTestRunner()
        backend = MockBackend(default="pass")

        report = run_regression_analysis(config, backend, backend, runner)
        assert report.base_metrics.num_tasks == 0
        assert report.tuned_metrics.num_tasks == 0
        assert report.forgetting_delta == 0.0
        assert report.base_metrics.execution_accuracy == 0.0
        assert report.tuned_metrics.execution_accuracy == 0.0

    def test_manifest_contents(self):
        tasks = [_make_task("t1"), _make_task("t2")]
        config = RegressionConfig(
            base_model="my/base",
            tuned_model="my/tuned",
            tasks=tasks,
            timeout_seconds=15,
        )
        runner = MockCodeTestRunner(default_passed=True)
        backend = MockBackend(default="pass")

        report = run_regression_analysis(config, backend, backend, runner)

        assert report.run_id.startswith("stage7-")
        assert "started_at" in report.manifest
        assert "elapsed_seconds" in report.manifest
        assert report.manifest["base_model"] == "my/base"
        assert report.manifest["tuned_model"] == "my/tuned"
        assert report.manifest["num_tasks"] == 2
        assert report.manifest["timeout_seconds"] == 15

    def test_default_runner_is_mock(self):
        """When runner=None, defaults to MockCodeTestRunner (all pass)."""
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=[_make_task("t1")],
        )
        backend = MockBackend(default="pass")
        report = run_regression_analysis(config, backend, backend)
        assert report.base_metrics.execution_accuracy == 1.0
        assert report.tuned_metrics.execution_accuracy == 1.0
        assert report.forgetting_delta == 0.0

    def test_report_is_serializable(self):
        """RegressionReport should be JSON-serializable via pydantic."""
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=[_make_task("t1"), _make_task("t2")],
        )
        runner = MockCodeTestRunner(default_passed=True)
        backend = MockBackend(default="pass")

        report = run_regression_analysis(config, backend, backend, runner)
        json_str = report.model_dump_json()
        data = _json.loads(json_str)
        assert "run_id" in data
        assert "forgetting_delta" in data
        assert "base_metrics" in data
        assert "tuned_metrics" in data
        assert "manifest" in data

    def test_default_tasks_used_when_none(self):
        """When config.tasks is None, DEFAULT_GENERAL_TASKS is used."""
        config = RegressionConfig(
            base_model="base", tuned_model="tuned",
            tasks=None,
        )
        runner = MockCodeTestRunner(default_passed=True)
        backend = MockBackend(default="pass")
        report = run_regression_analysis(config, backend, backend, runner)
        assert report.base_metrics.num_tasks == len(DEFAULT_GENERAL_TASKS)
        assert report.manifest["num_tasks"] == len(DEFAULT_GENERAL_TASKS)


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_normal_case(self):
        cost = estimate_cost_per_accepted_patch_usd(
            model_cwe_macro_f1=0.8,
            exec_pass_rate=0.5,
            num_predictions=12,
            inference_cost_usd=6.0,
            training_cost_usd=4.0,
        )
        # accepted = 12 * 0.5 = 6; total = 10; cost = 10/6
        assert cost == round(10.0 / 6, 4)

    def test_zero_accepted_returns_zero(self):
        cost = estimate_cost_per_accepted_patch_usd(
            model_cwe_macro_f1=0.0,
            exec_pass_rate=0.0,
            num_predictions=0,
            inference_cost_usd=10.0,
            training_cost_usd=5.0,
        )
        assert cost == 0.0

    def test_zero_cost_returns_zero(self):
        cost = estimate_cost_per_accepted_patch_usd(
            model_cwe_macro_f1=1.0,
            exec_pass_rate=1.0,
            num_predictions=10,
            inference_cost_usd=0.0,
            training_cost_usd=0.0,
        )
        assert cost == 0.0

    def test_full_cost(self):
        cost = estimate_cost_per_accepted_patch_usd(
            model_cwe_macro_f1=1.0,
            exec_pass_rate=1.0,
            num_predictions=4,
            inference_cost_usd=0.0,
            training_cost_usd=8.0,
        )
        # accepted = 4 * 1.0 = 4; total = 8; cost = 2.0
        assert cost == 2.0


# ---------------------------------------------------------------------------
# build_regression_summary
# ---------------------------------------------------------------------------


class TestBuildRegressionSummary:
    def _make_metrics(self) -> EvalMetrics:
        return EvalMetrics(
            num_samples=12,
            num_predictions=12,
            tier1_cwe_macro_f1=0.9,
            tier1_coverage=1.0,
            tier2_cwe_macro_f1=0.85,
            tier2_coverage=1.0,
            model_cwe_macro_f1=0.8,
            exec_pass_rate=0.5,
            patch_applies_rate=0.9,
            build_succeeds_rate=0.95,
            hallucination_rate=0.1,
            avg_patch_coverage=0.95,
        )

    def _make_regression_report(self, delta: float = -0.15) -> RegressionReport:
        return RegressionReport(
            run_id="test_run_1",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            tuned_model="sft_checkpoint_001",
            base_metrics=GeneralCapabilityMetrics(
                num_tasks=12, num_passed=10, execution_accuracy=0.8333,
                task_results=[],
            ),
            tuned_metrics=GeneralCapabilityMetrics(
                num_tasks=12, num_passed=8, execution_accuracy=0.6667,
                task_results=[],
            ),
            forgetting_delta=delta,
            manifest={"some": "data"},
        )

    def test_summary_fields_extracted(self):
        metrics = self._make_metrics()
        report = self._make_regression_report(delta=-0.1666)
        summary = build_regression_summary(
            run_id="checkpoint_001",
            stage6_metrics=metrics,
            regression_report=report,
        )
        assert summary.run_id == "checkpoint_001"
        assert summary.cwe_macro_f1 == 0.8
        assert summary.exec_pass_rate == 0.5
        assert summary.hallucination_rate == 0.1
        assert summary.general_capability_delta == -0.1666

    def test_summary_with_cost(self):
        metrics = self._make_metrics()
        report = self._make_regression_report(delta=0.0)
        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=metrics,
            regression_report=report,
            inference_cost_usd=6.0,
            training_cost_usd=4.0,
        )
        # accepted = 12 * 0.5 = 6; total = 10; cost = 10/6 ≈ 1.6667
        assert summary.cost_per_accepted_patch_usd == round(10.0 / 6, 4)

    def test_summary_no_cost_when_no_accepted(self):
        metrics = self._make_metrics()
        metrics.exec_pass_rate = 0.0  # no patches pass
        report = self._make_regression_report(delta=-0.1)
        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=metrics,
            regression_report=report,
        )
        assert summary.cost_per_accepted_patch_usd == 0.0

    def test_summary_is_pydantic_model(self):
        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=self._make_metrics(),
            regression_report=self._make_regression_report(),
        )
        assert isinstance(summary, RegressionSummary)
        data = _json.loads(summary.model_dump_json())
        assert data["run_id"] == "r1"

    def test_positive_delta(self):
        """When model improved, delta is positive."""
        metrics = self._make_metrics()
        report = self._make_regression_report(delta=0.15)
        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=metrics,
            regression_report=report,
        )
        assert summary.general_capability_delta == 0.15

    def test_negative_delta(self):
        """When model forgot, delta is negative."""
        metrics = self._make_metrics()
        report = self._make_regression_report(delta=-0.25)
        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=metrics,
            regression_report=report,
        )
        assert summary.general_capability_delta == -0.25

    def test_no_cost_defaults_to_zero(self):
        """When no cost params given, cost should be 0 (no accepted patches)."""
        # Actually: if exec_pass_rate=0.5 and num_predictions=12, accepted=6
        # but cost=0 because no cost params are given (defaults to 0.0)
        metrics = self._make_metrics()
        report = self._make_regression_report(delta=0.0)
        summary = build_regression_summary(
            run_id="r1",
            stage6_metrics=metrics,
            regression_report=report,
        )
        # accepted = 6, cost = 0/6 = 0
        assert summary.cost_per_accepted_patch_usd == 0.0
