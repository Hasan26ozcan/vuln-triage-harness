"""Stage 7 — regression / forgetting analysis.

After fine-tuning the base code-LLM on vulnerability classification +
patching tasks (Stages 5-6), a common failure mode is **catastrophic
forgetting**: the model gets good at security tasks but loses general
code-generation / reasoning ability.

This module measures that *general code-capability delta* by running
a small set of self-contained Python coding tasks (HumanEval-style
algorithm problems — deliberately NOT security-related) through the
model via the injectable ``ModelBackend`` Protocol, exec-testing each
generated solution in an isolated subprocess via ``CodeTestRunner``.

The forgetting delta is::

    delta = tuned_metrics.execution_accuracy
          - base_metrics.execution_accuracy

A *negative* delta means the fine-tuned model forgot general coding
ability relative to the base model.

Both ``ModelBackend`` and ``CodeTestRunner`` are injectable Protocols,
so every code path is testable with mocks — no model download, no GPU,
no subprocess. ``LocalCodeTestRunner`` uses ``subprocess`` (same
suppression approach as ``tier3_exec.py``) and ``MockCodeTestRunner``
returns canned results for unit tests.
"""

from __future__ import annotations

import logging
import re
import subprocess  # nosec B404
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.evaluation.backends import DEFAULT_BASE_MODEL, ModelBackend
from app.schemas.prediction_eval import (
    EvalMetrics,
    GeneralCapabilityMetrics,
    GeneralCapabilityResult,
    RegressionReport,
    RegressionSummary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# General-capability tasks — HumanEval-style algorithm problems
# (intentionally NOT security-related)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneralCapabilityTask:
    """A single general-code-capability task (NOT a vulnerability task).

    Attributes
    ----------
    task_id:
        Stable identifier used for reporting and mock keying.
    name:
        Short function name (e.g. ``"factorial"``).
    description:
        Problem statement shown to the model.
    prompt_code:
        The function signature the model should complete.
    test_code:
        Python test code that imports from ``solution`` and asserts correctness.
        Tests MUST be wrapped in ``def test_solution():`` so pytest collects
        them (bare module-level asserts are NOT collected by pytest).
    timeout_seconds:
        Per-task execution timeout.
    """

    task_id: str
    name: str
    description: str
    prompt_code: str
    test_code: str
    timeout_seconds: int = 30


def _gc_task(
    task_id: str,
    name: str,
    description: str,
    fn_sig: str,
    test_code: str,
    timeout_seconds: int = 30,
) -> GeneralCapabilityTask:
    """Convenience constructor for ``GeneralCapabilityTask``."""
    return GeneralCapabilityTask(
        task_id=task_id,
        name=name,
        description=description,
        prompt_code=fn_sig,
        test_code=test_code,
        timeout_seconds=timeout_seconds,
    )


# A curated set of 12 classic Python algorithm problems spanning:
#   - recursion / math (factorial, fibonacci)
#   - string manipulation (palindrome, anagrams, vowel counting)
#   - array / list operations (binary search, two-sum, two-pointer, max subarray)
#   - stack / data-structure simulation (valid parentheses)
#   - integer math (reverse int)
#   - string algorithms (longest common prefix)
#
# All use only the Python standard library — no external imports needed
# so the sandbox subprocess never hits network or missing packages.

DEFAULT_GENERAL_TASKS: list[GeneralCapabilityTask] = [
    _gc_task(
        "gc_001",
        "factorial",
        "Write a function that returns the factorial of a non-negative integer n "
        "(n! = 1 * 2 * ... * n, and 0! = 1).",
        "def factorial(n):",
        """
from solution import factorial

def test_solution():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(5) == 120
    assert factorial(10) == 3628800
""",
    ),
    _gc_task(
        "gc_002",
        "is_palindrome",
        "Write a function that checks whether a string is a palindrome, "
        "considering only alphanumeric characters and ignoring case.",
        "def is_palindrome(s):",
        """
from solution import is_palindrome

def test_solution():
    assert is_palindrome("A man, a plan, a canal: Panama") is True
    assert is_palindrome("race a car") is False
    assert is_palindrome("") is True
    assert is_palindrome("racecar") is True
    assert is_palindrome("hello") is False
""",
    ),
    _gc_task(
        "gc_003",
        "fibonacci",
        "Write a function that returns the n-th Fibonacci number. "
        "fibonacci(0) = 0, fibonacci(1) = 1, fibonacci(2) = 1, etc.",
        "def fibonacci(n):",
        """
from solution import fibonacci

def test_solution():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765
""",
    ),
    _gc_task(
        "gc_004",
        "binary_search",
        "Write a function that performs binary search on a sorted list. "
        "Return the index of target if found, or -1 if not found.",
        "def binary_search(arr, target):",
        """
from solution import binary_search

def test_solution():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2
    assert binary_search([1, 3, 5, 7, 9], 1) == 0
    assert binary_search([1, 3, 5, 7, 9], 9) == 4
    assert binary_search([1, 3, 5, 7, 9], 4) == -1
    assert binary_search([], 5) == -1
""",
    ),
    _gc_task(
        "gc_005",
        "two_sum",
        "Write a function that, given a list of integers and a target sum, "
        "returns the indices of the two numbers that add up to the target. "
        "Assume exactly one solution exists, or return [] if none.",
        "def two_sum(nums, target):",
        """
from solution import two_sum

def test_solution():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 2]
    assert two_sum([3, 3], 6) == [0, 1]
    assert two_sum([1, 2, 3], 10) == []
""",
    ),
    _gc_task(
        "gc_006",
        "count_vowels",
        "Write a function that counts the number of vowels (a, e, i, o, u) "
        "in a string, case-insensitive.",
        "def count_vowels(s):",
        """
from solution import count_vowels

def test_solution():
    assert count_vowels("hello") == 2
    assert count_vowels("HELLO") == 2
    assert count_vowels("") == 0
    assert count_vowels("xyz") == 0
    assert count_vowels("aeiou") == 5
""",
    ),
    _gc_task(
        "gc_007",
        "reverse_int",
        "Write a function that reverses the digits of a 32-bit signed integer. "
        "If the reversed integer overflows 32-bit range, return 0. "
        "Ignore leading zeros.",
        "def reverse_int(x):",
        """
from solution import reverse_int

def test_solution():
    assert reverse_int(123) == 321
    assert reverse_int(-123) == -321
    assert reverse_int(120) == 21
    assert reverse_int(0) == 0
""",
    ),
    _gc_task(
        "gc_008",
        "is_anagram",
        "Write a function that determines if two strings are anagrams "
        "(contain the same characters with the same frequency, ignoring case).",
        "def is_anagram(s, t):",
        """
from solution import is_anagram

def test_solution():
    assert is_anagram("anagram", "nagaram") is True
    assert is_anagram("rat", "car") is False
    assert is_anagram("", "") is True
    assert is_anagram("Listen", "Silent") is True
""",
    ),
    _gc_task(
        "gc_009",
        "longest_common_prefix",
        "Write a function that finds the longest common prefix string "
        "among an array of strings. Return an empty string if none.",
        "def longest_common_prefix(strs):",
        """
from solution import longest_common_prefix

def test_solution():
    assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
    assert longest_common_prefix(["dog", "racecar", "car"]) == ""
    assert longest_common_prefix(["interspecies", "interstellar", "interstate"]) == "inters"
    assert longest_common_prefix([""]) == ""
""",
    ),
    _gc_task(
        "gc_010",
        "valid_parentheses",
        "Write a function that determines if the input string of parentheses "
        "is valid. Valid means open brackets are closed by the same type "
        "and in the correct order. Only '()[]{}' characters.",
        "def valid_parentheses(s):",
        """
from solution import valid_parentheses

def test_solution():
    assert valid_parentheses("()") is True
    assert valid_parentheses("()[]{}") is True
    assert valid_parentheses("(]") is False
    assert valid_parentheses("([)]") is False
    assert valid_parentheses("{[]}") is True
    assert valid_parentheses("") is True
""",
    ),
    _gc_task(
        "gc_011",
        "remove_duplicates",
        "Write a function that removes duplicates from a sorted list in-place "
        "and returns the new length. The first part of the list should contain "
        "unique values in their original order.",
        "def remove_duplicates(nums):",
        """
from solution import remove_duplicates

def test_solution():
    assert remove_duplicates([1, 1, 2]) == 2
    assert remove_duplicates([0, 0, 1, 1, 1, 2, 2, 3, 3, 4]) == 5
    assert remove_duplicates([]) == 0
    assert remove_duplicates([1]) == 1
""",
    ),
    _gc_task(
        "gc_012",
        "max_subarray_sum",
        "Write a function that finds the contiguous subarray with the largest "
        "sum and returns the sum. At least one number is non-negative.",
        "def max_subarray_sum(nums):",
        """
from solution import max_subarray_sum

def test_solution():
    assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
    assert max_subarray_sum([1]) == 1
    assert max_subarray_sum([5, 4, -1, 7, 2]) == 17
    assert max_subarray_sum([-2, -1, -3]) == -1
""",
    ),
]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


# Regex to extract the contents of a fenced code block (with optional
# language tag). Works anywhere in the text — not just at the start.
_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9]+)?\n(.*?)\n```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Extract Python code from model output, stripping markdown fences.

    If the text contains a fenced code block (`` ```python ... ``` `` or
    `` ``` ... ``` ``), the fences and language tag are removed and only
    the code inside is returned. Leading/trailing whitespace is stripped.

    If no fences are present, the text is returned as-is (stripped).
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Look for a fenced code block anywhere in the text.
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1).strip()

    return text


def build_capability_prompt(task: GeneralCapabilityTask) -> str:
    """Build a code-generation prompt for a general-capability task.

    The prompt includes the task description and the function signature
    stub. The model is instructed to return *only* the completed
    function in a Python code block — no explanatory prose.
    """
    prompt = (
        f"You are a Python coding assistant. Solve the following problem "
        f"by completing the function signature provided.\n\n"
        f"Problem:\n{task.description}\n\n"
        f"Complete the following function:\n\n"
        f"```python\n{task.prompt_code}\n```\n\n"
        f"Return ONLY the completed function in a Python code block. "
        f"Do not include any comments, prose, or additional text."
    )
    return prompt


# ---------------------------------------------------------------------------
# CodeTestRunner Protocol and implementations
# ---------------------------------------------------------------------------


@dataclass
class CodeTestResult:
    """Result of running a code test.

    Attributes
    ----------
    passed:
        True if all test assertions passed.
    output:
        Captured stdout from the test run (may be None).
    error:
        Captured stderr / error message (may be None).
    """

    passed: bool
    output: str | None = None
    error: str | None = None


@runtime_checkable
class CodeTestRunner(Protocol):
    """Protocol for running generated code against test code.

    Implementations may use a real subprocess (LocalCodeTestRunner)
    or return canned results (MockCodeTestRunner) for testing.
    """

    def run_code_test(
        self,
        code: str,
        test_code: str,
        task_id: str | None = None,
        timeout_seconds: int = 30,
    ) -> CodeTestResult:
        """Run the given code against test_code and return the result.

        Parameters
        ----------
        code:
            The model-generated solution code.
        test_code:
            The test code (must contain ``def test_*():`` functions
            for pytest collection).
        task_id:
            Optional task identifier (used by mocks for per-task results).
        timeout_seconds:
            Maximum execution time before killing the test.
        """
        ...


class MockCodeTestRunner:
    """Deterministic code runner for testing — no subprocess.

    Returns ``default_passed`` for every task unless a per-task
    override is provided in ``results``.
    """

    def __init__(
        self,
        default_passed: bool = True,
        results: dict[str, bool] | None = None,
    ):
        self._default = default_passed
        self._results = results or {}
        self.call_count = 0
        self.last_task_id: str | None = None

    def run_code_test(
        self,
        code: str,  # NOSONAR
        test_code: str,  # NOSONAR
        task_id: str | None = None,
        timeout_seconds: int = 30,  # NOSONAR
    ) -> CodeTestResult:
        self.call_count += 1
        self.last_task_id = task_id
        if task_id is not None and task_id in self._results:
            passed = self._results[task_id]
        else:
            passed = self._default
        return CodeTestResult(
            passed=passed,
            output=f"mock test result for task {task_id}" if passed else None,
            error=None if passed else "mock failure",
        )


def _sanitize_paths(text: str | None) -> str | None:
    """Replace absolute local paths in *text* with relative equivalents.

    Pytest output on Windows embeds absolute paths like
    ``C:\\Users\\<user>\\.clone\\vuln-triage-harness\\.venv\\Scripts\\python.exe``
    and ``C:\\Users\\<user>\\AppData\\Local\\Temp\\tmpXXXXXX``.  These are
    machine-specific and should not appear in committed artefacts.
    """
    if not text:
        return text
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    # Normalise backslashes for consistent replacement.
    normalised = text.replace("\\", "/")
    project_norm = project_root.replace("\\", "/")
    # Project-root paths → relative (``./...``).
    normalised = normalised.replace(project_norm, ".")
    # Remaining ``C:/Users/<user>/AppData/Local/Temp/tmpXXXXXX`` → ``<tmp>/<last>``.
    normalised = re.sub(
        r"/Users/[^/]+/AppData/Local/Temp/(tmp\w+)",
        r"<tmp>/\1",
        normalised,
    )
    # Unix temp dirs. This is a text-redaction regex, not a path we read from
    # or write to, so bandit's hardcoded-tmp-directory check does not apply.
    normalised = re.sub(
        r"/tmp/(tmp\w+)",  # nosec B108 # NOSONAR - text-redaction regex, not a filesystem path; not a "publicly writable directory" use
        r"<tmp>/\1",
        normalised,
    )
    # Strip any remaining drive-letter prefixes (e.g. "C:" left behind after
    # the ``/`` after it was consumed by an earlier regex, or any remaining
    # drive-letter paths that didn't match project-root replacement).
    normalised = re.sub(r"^[A-Za-z]:", "", normalised)
    return normalised


class LocalCodeTestRunner:
    """Runs generated code against tests in an isolated subprocess.

    Writes ``solution.py`` (the model output) and ``test_solution.py``
    (the task's test code) to a temporary directory, then runs
    ``python -m pytest`` on it. No Docker needed — the temp dir is
    cleaned up automatically.

    For production CI with untrusted code, use ``DockerCodeTestRunner``
    for stronger isolation (read-only fs, no network, non-root user,
    memory limits).
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds

    def run_code_test(
        self,
        code: str,
        test_code: str,
        task_id: str | None = None,  # NOSONAR
        timeout_seconds: int | None = None,
    ) -> CodeTestResult:
        # Step 1: Extract clean code from the model response.
        clean_code = _extract_code(code)

        # Step 2: Write solution + test to a temp directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            solution_file = tmp / "solution.py"
            test_file = tmp / "test_solution.py"

            solution_file.write_text(clean_code, encoding="utf-8")
            test_file.write_text(test_code, encoding="utf-8")

            # Step 3: Run pytest in the temp directory.
            timeout = timeout_seconds if timeout_seconds is not None else self.timeout
            result = self._run_pytest(test_file, timeout)

            output = _sanitize_paths(result.stdout.strip()) if result.stdout else None
            error = (
                _sanitize_paths(result.stderr.strip())
                if result.returncode != 0 and result.stderr
                else None
            )
            return CodeTestResult(
                passed=result.returncode == 0,
                output=output,
                error=error,
            )

    def _run_pytest(self, test_file: Path, timeout: int) -> subprocess.CompletedProcess:
        """Run pytest on the test file and return the completed process."""
        # Inputs are trusted: sys.executable (system Python) and a temp
        # file path created by this method itself.
        return subprocess.run(  # nosec B603
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(test_file.parent),
        )


class DockerCodeTestRunner:
    """Runs generated code tests inside an isolated Docker container.

    Mirrors ``LocalCodeTestRunner`` but executes pytest inside the
    ``vuln-triage-sandbox`` container (the same image used by
    ``DockerSandboxRunner`` in ``tier3_exec.py``). The container has:

    - **Read-only root filesystem** (``--read-only``)
    - A **writable tmpfs** at ``/tmp`` only
    - **No network access** (``--network none``)
    - **Non-root user** (UID 1000) inside the container
    - A **memory limit** (default 512 MB)

    This provides a stronger isolation boundary than the local
    subprocess runner and is the recommended choice for CI / untrusted
    model output.

    Requires the ``docker`` Python package and a running Docker daemon.
    If Docker is unavailable, :meth:`run_code_test` returns a failed
    ``CodeTestResult`` with a descriptive error (it does **not** raise,
    so CI pipelines can fall back to ``LocalCodeTestRunner``).
    """

    # Reuse the same image as DockerSandboxRunner for consistency.
    DEFAULT_IMAGE: str = "vuln-triage-sandbox:python3.11"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        timeout_seconds: int = 60,
        memory_limit: str = "512m",
    ):
        self.image = image
        self.timeout = timeout_seconds
        self.memory_limit = memory_limit

    def run_code_test(
        self,
        code: str,
        test_code: str,
        task_id: str | None = None,  # NOSONAR — Protocol method
        timeout_seconds: int | None = None,
    ) -> CodeTestResult:
        """Run the given code against tests inside a Docker container.

        Delegates container creation to the lazy ``_docker_client`` so
        that importing this module never requires the ``docker`` package.
        """
        clean_code = _extract_code(code)
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout

        # Lazily import the docker package.
        try:
            from docker import from_env
        except ImportError as exc:
            return CodeTestResult(
                passed=False,
                output=None,
                error=(
                    "The 'docker' package is required for DockerCodeTestRunner. "
                    "Install it with: pip install docker. "
                    f"Original error: {exc}"
                ),
            )

        try:
            client = from_env()
        except Exception as exc:  # noqa: BLE001
            return CodeTestResult(
                passed=False,
                output=None,
                error=f"Could not connect to Docker daemon: {exc}",
            )

        # Ensure the sandbox image exists.
        try:
            client.images.get(self.image)
        except Exception:  # noqa: BLE001
            return CodeTestResult(
                passed=False,
                output=None,
                error=(
                    f"Docker image '{self.image}' not found. Build it with:\n"
                    f"  docker build -t {self.image} -f sandbox/Dockerfile ."
                ),
            )

        # Write solution + test to a temp directory (host-side).
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            solution_file = tmp / "solution.py"
            test_file = tmp / "test_solution.py"
            solution_file.write_text(clean_code, encoding="utf-8")
            test_file.write_text(test_code, encoding="utf-8")

            # Copy the files into the container via mounts and run pytest.
            try:
                output = client.containers.run(
                    self.image,
                    [
                        "python",
                        "-m",
                        "pytest",
                        "/test/test_solution.py",
                        "-v",
                        "--tb=short",
                    ],
                    mounts=[
                        {
                            "type": "bind",
                            "source": str(solution_file),
                            "target": "/test/solution.py",
                            "read_only": True,
                        },
                        {
                            "type": "bind",
                            "source": str(test_file),
                            "target": "/test/test_solution.py",
                            "read_only": True,
                        },
                    ],
                    read_only=True,
                    network_disabled=True,
                    mem_limit=self.memory_limit,
                    auto_remove=True,
                    stdout=True,
                    stderr=True,
                    timeout=timeout,
                    user="1000:1000",
                )
                # ``containers.run`` returns bytes in stdout/stderr.
                if isinstance(output, dict):
                    stdout_text = output.get(b"stdout", b"").decode("utf-8", errors="replace")
                elif isinstance(output, bytes):
                    stdout_text = output.decode("utf-8", errors="replace")
                else:
                    stdout_text = str(output)
            except Exception as exc:  # noqa: BLE001
                return CodeTestResult(
                    passed=False,
                    output=None,
                    error=f"Docker container execution failed: {exc}",
                )

        passed = "passed" in stdout_text.lower() and "failed" not in stdout_text.lower()
        error_text = _sanitize_paths(stdout_text.strip()) if stdout_text else None

        return CodeTestResult(
            passed=passed,
            output=error_text,
            error=None if passed else "Docker pytest run reported failures",
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class GeneralCapabilityEvaluator:
    """Evaluates a model's general code-generation ability.

    Runs a set of ``GeneralCapabilityTask`` items through an injectable
    ``ModelBackend`` (which generates code) and an injectable
    ``CodeTestRunner`` (which exec-tests the generated code).

    Parameters
    ----------
    backend:
        Any object implementing ``ModelBackend`` (e.g. ``QwenBackend``,
        ``MockBackend``).
    runner:
        Any object implementing ``CodeTestRunner`` (e.g.
        ``LocalCodeTestRunner``, ``MockCodeTestRunner``).
    tasks:
        General-capability tasks to evaluate. Defaults to
        ``DEFAULT_GENERAL_TASKS`` (12 algorithm problems).
    """

    def __init__(
        self,
        backend: ModelBackend,
        runner: CodeTestRunner,
        tasks: list[GeneralCapabilityTask] | None = None,
    ):
        self._backend = backend
        self._runner = runner
        self._tasks = tasks if tasks is not None else list(DEFAULT_GENERAL_TASKS)

    def evaluate(self, task: GeneralCapabilityTask) -> GeneralCapabilityResult:
        """Evaluate the model on a single general-capability task.

        Returns a ``GeneralCapabilityResult`` with ``passed=True`` if the
        generated code passes all test assertions.
        """
        prompt = build_capability_prompt(task)
        model_response = self._backend.generate(prompt)

        result = self._runner.run_code_test(
            code=model_response,
            test_code=task.test_code,
            task_id=task.task_id,
            timeout_seconds=task.timeout_seconds,
        )

        return GeneralCapabilityResult(
            task_id=task.task_id,
            name=task.name,
            passed=result.passed,
            model_response=model_response,
            exec_output=result.output,
            exec_error=result.error,
        )

    def evaluate_all(self) -> GeneralCapabilityMetrics:
        """Evaluate the model on all tasks and return aggregate metrics.

        Returns
        -------
        GeneralCapabilityMetrics
            Aggregate results with ``execution_accuracy`` = passed / total.
        """
        task_results: list[GeneralCapabilityResult] = []
        for task in self._tasks:
            logger.info(" Evaluating general-capability task %s", task.task_id)
            res = self.evaluate(task)
            task_results.append(res)

        num_tasks = len(task_results)
        num_passed = sum(1 for r in task_results if r.passed)
        exec_acc = num_passed / num_tasks if num_tasks > 0 else 0.0

        return GeneralCapabilityMetrics(
            num_tasks=num_tasks,
            num_passed=num_passed,
            execution_accuracy=round(exec_acc, 4),
            task_results=task_results,
        )

    @property
    def tasks(self) -> list[GeneralCapabilityTask]:
        """Return the task set this evaluator uses."""
        return list(self._tasks)


# ---------------------------------------------------------------------------
# Regression analysis — "before" vs "after"
# ---------------------------------------------------------------------------


@dataclass
class RegressionConfig:
    """Configuration for the Stage 7 regression / forgetting analysis.

    Attributes
    ----------
    base_model:
        The pre-fine-tuning model name (e.g. ``"Qwen/Qwen2.5-Coder-7B-Instruct"``).
    tuned_model:
        The fine-tuned checkpoint identifier (e.g. a run_id or path).
    tasks:
        General-capability tasks to evaluate. ``None`` uses
        ``DEFAULT_GENERAL_TASKS``.
    timeout_seconds:
        Per-task execution timeout (for the local runner).
    """

    base_model: str = DEFAULT_BASE_MODEL
    tuned_model: str = "tuned"
    tasks: list[GeneralCapabilityTask] | None = None
    timeout_seconds: int = 30


def run_regression_analysis(
    config: RegressionConfig,
    base_backend: ModelBackend,
    tuned_backend: ModelBackend,
    runner: CodeTestRunner | None = None,
) -> RegressionReport:
    """Run forgetting analysis: compare general-capability scores.

    Evaluates both the base model and the fine-tuned model on the same set
    of general-capability tasks, then computes the forgetting delta:

        delta = tuned_execution_accuracy - base_execution_accuracy

    A negative delta means the fine-tuned model forgot general coding
    ability.

    Parameters
    ----------
    config:
        ``RegressionConfig`` with model names and task set.
    base_backend:
        ``ModelBackend`` for the **base** (pre-fine-tuning) model.
    tuned_backend:
        ``ModelBackend`` for the **tuned** (post-fine-tuning) model.
    runner:
        ``CodeTestRunner`` to use (e.g. ``LocalCodeTestRunner`` or
        ``MockCodeTestRunner``). Defaults to ``LocalCodeTestRunner``.

    Returns
    -------
    RegressionReport
        Report with base/tuned metrics, forgetting delta, and manifest.
    """
    tasks = config.tasks if config.tasks is not None else list(DEFAULT_GENERAL_TASKS)
    code_runner = runner if runner is not None else LocalCodeTestRunner()  # type: ignore[call-arg]

    start_time = time.time()
    run_id = f"stage7-{uuid.uuid4().hex[:8]}"

    # Evaluate the base (pre-fine-tuning) model.
    logger.info(
        " Stage 7: evaluating base model %s on %d tasks",
        config.base_model,
        len(tasks),
    )
    base_evaluator = GeneralCapabilityEvaluator(
        backend=base_backend,
        runner=code_runner,
        tasks=tasks,
    )
    base_metrics = base_evaluator.evaluate_all()

    # Evaluate the tuned (post-fine-tuning) model.
    logger.info(
        " Stage 7: evaluating tuned model %s on %d tasks",
        config.tuned_model,
        len(tasks),
    )
    tuned_evaluator = GeneralCapabilityEvaluator(
        backend=tuned_backend,
        runner=code_runner,
        tasks=tasks,
    )
    tuned_metrics = tuned_evaluator.evaluate_all()

    # Forgetting delta: positive = improvement, negative = forgetting.
    forgetting_delta = round(tuned_metrics.execution_accuracy - base_metrics.execution_accuracy, 4)

    elapsed = time.time() - start_time
    manifest = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "elapsed_seconds": round(elapsed, 2),
        "base_model": config.base_model,
        "tuned_model": config.tuned_model,
        "num_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "timeout_seconds": config.timeout_seconds,
    }

    logger.info(
        " Stage 7 complete: base_acc=%.4f tuned_acc=%.4f delta=%.4f",
        base_metrics.execution_accuracy,
        tuned_metrics.execution_accuracy,
        forgetting_delta,
    )

    return RegressionReport(
        run_id=run_id,
        base_model=config.base_model,
        tuned_model=config.tuned_model,
        base_metrics=base_metrics,
        tuned_metrics=tuned_metrics,
        forgetting_delta=forgetting_delta,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# RegressionSummary — combine Stage 6 metrics + Stage 7 delta + cost
# ---------------------------------------------------------------------------


def estimate_cost_per_accepted_patch_usd(
    exec_pass_rate: float,
    num_predictions: int,
    inference_cost_usd: float = 0.0,
    training_cost_usd: float = 0.0,
) -> float:
    """Estimate the cost per accepted (passing) patch.

    ``cost = (inference_cost + amortized_training_cost) / accepted_patches``

    where ``accepted_patches = num_predictions * exec_pass_rate``.

    Returns ``0.0`` when no patches pass (avoids division by zero).
    """
    accepted = num_predictions * exec_pass_rate
    if accepted <= 0:
        return 0.0
    total_cost = inference_cost_usd + training_cost_usd
    return round(total_cost / accepted, 4)


def build_regression_summary(
    run_id: str,
    stage6_metrics: EvalMetrics,
    regression_report: RegressionReport,
    inference_cost_usd: float = 0.0,
    training_cost_usd: float = 0.0,
) -> RegressionSummary:
    """Build a ``RegressionSummary`` from Stage 6 + Stage 7 outputs.

    Combines Stage 6 evaluation metrics (``EvalMetrics``) with the Stage 7
    forgetting delta and a cost-per-patch estimate into a single summary
    row — the primary output consumed by the Stage 10 regression gate.

    Parameters
    ----------
    run_id:
        Unique identifier for this checkpoint evaluation.
    stage6_metrics:
        ``EvalMetrics`` from the Stage 6 four-tier harness.
    regression_report:
        ``RegressionReport`` from Stage 7 forgetting analysis.
    inference_cost_usd:
        Total inference cost in USD (optional).
    training_cost_usd:
        Amortized training cost in USD (optional).

    Returns
    -------
    RegressionSummary
        A single summary row for the regression gate.
    """
    cost = estimate_cost_per_accepted_patch_usd(
        exec_pass_rate=stage6_metrics.exec_pass_rate,
        num_predictions=stage6_metrics.num_predictions,
        inference_cost_usd=inference_cost_usd,
        training_cost_usd=training_cost_usd,
    )

    return RegressionSummary(
        run_id=run_id,
        cwe_macro_f1=stage6_metrics.model_cwe_macro_f1,
        exec_pass_rate=stage6_metrics.exec_pass_rate,
        hallucination_rate=stage6_metrics.hallucination_rate,
        general_capability_delta=regression_report.forgetting_delta,
        cost_per_accepted_patch_usd=cost,
    )
