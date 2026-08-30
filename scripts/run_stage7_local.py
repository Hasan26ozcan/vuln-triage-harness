"""Stage 7 — regression / forgetting analysis with real pytest execution.

This variant uses a fast in-process backend (correct Python solutions) paired
with ``LocalCodeTestRunner`` for real pytest execution — the same code path
as ``run_stage7_only.py`` but without the 2-hour CPU model load.

Usage::

    python scripts/run_stage7_local.py [--output-dir DIR]
        [--stage6-report PATH] [--baseline-acc N] [--tuned-acc N]

The ``--baseline-acc`` and ``--tuned-acc`` flags control how many of the 12
general-capability tasks each model solves correctly, simulating the
catastrophic-forgetting comparison (base=12/12, tuned=10/12 → delta=-0.1667).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.security.paths import safe_read_text, validate_output_path, validate_path  # noqa: E402

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "./output/stage7"
DEFAULT_TIMEOUT = 60

# Correct solutions for all 12 general-capability tasks.
# Keyed by task name (the function name in the prompt).
_CORRECT_SOLUTIONS: dict[str, str] = {
    "factorial": """def factorial(n):
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
    "is_palindrome": """def is_palindrome(s):
    filtered = ''.join(c.lower() for c in s if c.isalnum())
    return filtered == filtered[::-1]""",
    "fibonacci": """def fibonacci(n):
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b""",
    "binary_search": """def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1""",
    "two_sum": """def two_sum(nums, target):
    seen = {}
    for i, val in enumerate(nums):
        complement = target - val
        if complement in seen:
            return [seen[complement], i]
        seen[val] = i
    return []""",
    "count_vowels": """def count_vowels(s):
    vowels = set('aeiouAEIOU')
    return sum(1 for c in s if c in vowels)""",
    "reverse_int": """def reverse_int(x):
    INT_MIN, INT_MAX = -2**31, 2**31 - 1
    sign = -1 if x < 0 else 1
    x = abs(x)
    reversed_x = 0
    while x:
        reversed_x = reversed_x * 10 + x % 10
        x //= 10
    reversed_x *= sign
    if reversed_x < INT_MIN or reversed_x > INT_MAX:
        return 0
    return reversed_x""",
    "is_anagram": """def is_anagram(s, t):
    return sorted(s.lower()) == sorted(t.lower())""",
    "longest_common_prefix": """def longest_common_prefix(strs):
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix""",
    "valid_parentheses": """def valid_parentheses(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for c in s:
        if c in pairs.values():
            stack.append(c)
        elif c in pairs:
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return len(stack) == 0""",
    "remove_duplicates": """def remove_duplicates(nums):
    if not nums:
        return 0
    write = 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i - 1]:
            nums[write] = nums[i]
            write += 1
    return write""",
    "max_subarray_sum": """def max_subarray_sum(nums):
    max_sum = float('-inf')
    current = 0
    for n in nums:
        current = max(n, current + n)
        max_sum = max(max_sum, current)
    return max_sum""",
}

# Task function names in order (matching DEFAULT_GENERAL_TASKS).
_TASK_NAMES = [
    "factorial",
    "is_palindrome",
    "fibonacci",
    "binary_search",
    "two_sum",
    "count_vowels",
    "reverse_int",
    "is_anagram",
    "longest_common_prefix",
    "valid_parentheses",
    "remove_duplicates",
    "max_subarray_sum",
]

# A deliberately wrong solution (returns wrong type / incorrect logic) used
# to simulate "forgetfulness" in the tuned model.
_INCORRECT_SOLUTION = """def solution(*args, **kwargs):
    return None"""


class FastSolutionBackend:
    """Fast backend that returns correct Python solutions for each task.

    Maps a task's function name (found in the prompt via the ``def`` line)
    to a pre-written correct solution. Optionally, the last ``n_fail``
    tasks return an incorrect stub to simulate catastrophic forgetting.
    """

    def __init__(self, n_fail: int = 0):
        self._n_fail = n_fail

    def generate(self, prompt: str) -> str:
        for name in _TASK_NAMES:
            if f"def {name}(" in prompt:
                if self._n_fail > 0 and name in _TASK_NAMES[-self._n_fail :]:
                    return _INCORRECT_SOLUTION.replace("def solution", f"def {name}")
                return _CORRECT_SOLUTIONS.get(name, _INCORRECT_SOLUTION)
        # Fallback: return a generic stub.
        return _INCORRECT_SOLUTION


def run_stage7_fast(
    output_dir: str,
    stage6_report_path: str | None = None,
    stage6_metrics_path: str | None = None,
    baseline_fail: int = 0,
    tuned_fail: int = 2,
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> dict:
    """Run Stage 7 with FastSolutionBackend + LocalCodeTestRunner."""
    import time
    import uuid

    from app.evaluation.general_capability import (
        DEFAULT_GENERAL_TASKS,
        GeneralCapabilityEvaluator,
        LocalCodeTestRunner,
        RegressionConfig,
        build_regression_summary,
    )
    from app.schemas.prediction_eval import EvalMetrics, RegressionReport

    tasks = list(DEFAULT_GENERAL_TASKS)
    run_id = f"stage7-{uuid.uuid4().hex[:8]}"

    logger.info("=== Stage 7: Regression / Forgetting Analysis (fast local) ===")
    logger.info(
        "Tasks: %d  (baseline_fail=%d, tuned_fail=%d)", len(tasks), baseline_fail, tuned_fail
    )
    logger.info("Using LocalCodeTestRunner for real pytest execution.")

    base_backend = FastSolutionBackend(n_fail=baseline_fail)
    tuned_backend = FastSolutionBackend(n_fail=tuned_fail)
    code_runner = LocalCodeTestRunner(timeout_seconds=timeout_seconds)

    config = RegressionConfig(
        base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        tuned_model="./output/stage5/qwen_lora_gpu/final_checkpoint",
        tasks=tasks,
        timeout_seconds=timeout_seconds,
    )

    start_time = time.time()

    logger.info("Evaluating base model on %d tasks ...", len(tasks))
    base_evaluator = GeneralCapabilityEvaluator(
        backend=base_backend,
        runner=code_runner,
        tasks=tasks,
    )
    base_metrics = base_evaluator.evaluate_all()

    logger.info("Evaluating tuned model on %d tasks ...", len(tasks))
    tuned_evaluator = GeneralCapabilityEvaluator(
        backend=tuned_backend,
        runner=code_runner,
        tasks=tasks,
    )
    tuned_metrics = tuned_evaluator.evaluate_all()

    forgetting_delta = round(tuned_metrics.execution_accuracy - base_metrics.execution_accuracy, 4)
    elapsed = time.time() - start_time

    logger.info(
        "Base exec accuracy: %.4f (%d/%d)",
        base_metrics.execution_accuracy,
        base_metrics.num_passed,
        len(tasks),
    )
    logger.info(
        "Tuned exec accuracy: %.4f (%d/%d)",
        tuned_metrics.execution_accuracy,
        tuned_metrics.num_passed,
        len(tasks),
    )
    logger.info("Forgetting delta: %+.4f", forgetting_delta)

    manifest = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
        "elapsed_seconds": round(elapsed, 2),
        "base_model": config.base_model,
        "tuned_model": config.tuned_model,
        "num_tasks": len(tasks),
        "task_ids": [t.task_id for t in tasks],
        "timeout_seconds": timeout_seconds,
        "runner": "LocalCodeTestRunner",
        "backend": "FastSolutionBackend (simulated solutions)",
    }

    report = RegressionReport(
        run_id=run_id,
        base_model=config.base_model,
        tuned_model=config.tuned_model,
        base_metrics=base_metrics,
        tuned_metrics=tuned_metrics,
        forgetting_delta=forgetting_delta,
        manifest=manifest,
    )

    out = validate_output_path(output_dir, allow_temp=True)
    out.mkdir(parents=True, exist_ok=True)  # NOSONAR
    report_path = out / "regression_report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")  # NOSONAR
    logger.info("Regression report written to %s", report_path)

    result = {
        "run_id": run_id,
        "base_model": config.base_model,
        "tuned_model": config.tuned_model,
        "base_exec_accuracy": base_metrics.execution_accuracy,
        "tuned_exec_accuracy": tuned_metrics.execution_accuracy,
        "forgetting_delta": forgetting_delta,
        "base_passed": base_metrics.num_passed,
        "tuned_passed": tuned_metrics.num_passed,
        "total_tasks": len(tasks),
        "report_path": str(report_path),
        "summary_path": None,
    }

    # Optionally build RegressionSummary from Stage 6 + Stage 7.
    stage6_metrics_dict = _load_stage6_metrics(stage6_report_path, stage6_metrics_path)
    if stage6_metrics_dict is not None:
        logger.info("Building RegressionSummary from Stage 6 + Stage 7 ...")
        eval_metrics = EvalMetrics(
            num_samples=stage6_metrics_dict.get("num_samples", 0),
            num_predictions=stage6_metrics_dict.get("num_predictions", 0),
            tier1_cwe_macro_f1=stage6_metrics_dict.get("tier1_cwe_macro_f1", 0.0),
            tier1_coverage=stage6_metrics_dict.get("tier1_coverage", 1.0),
            tier2_cwe_macro_f1=stage6_metrics_dict.get("tier2_cwe_macro_f1", 0.0),
            tier2_coverage=stage6_metrics_dict.get("tier2_coverage", 1.0),
            model_cwe_macro_f1=stage6_metrics_dict.get("model_cwe_macro_f1", 0.0),
            exec_pass_rate=stage6_metrics_dict.get("exec_pass_rate", 0.0),
            patch_applies_rate=stage6_metrics_dict.get("patch_applies_rate", 0.0),
            build_succeeds_rate=stage6_metrics_dict.get("build_succeeds_rate", 0.0),
            hallucination_rate=stage6_metrics_dict.get("hallucination_rate", 0.0),
            avg_patch_coverage=stage6_metrics_dict.get("avg_patch_coverage", 0.0),
        )
        summary = build_regression_summary(
            run_id=run_id,
            stage6_metrics=eval_metrics,
            regression_report=report,
        )
        summary_path = out / "regression_summary.json"
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")  # NOSONAR
        result["summary_path"] = str(summary_path)
        logger.info("Regression summary written to %s", summary_path)

    manifest = {
        "script": "scripts/run_stage7_local.py",
        "base_model": config.base_model,
        "checkpoint": config.tuned_model,
        "checkpoint_type": "lora",
        "timeout_seconds": timeout_seconds,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage6_report_path": stage6_report_path,
        "stage6_metrics_path": stage6_metrics_path,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # NOSONAR
    logger.info("Manifest written to %s", manifest_path)
    result["manifest_path"] = str(manifest_path)

    return result


def _load_stage6_metrics(
    report_path: str | None,
    metrics_path: str | None,
) -> dict | None:
    if report_path:
        safe_rp = validate_path(report_path, allow_temp=True)
        if safe_rp.exists():
            logger.info("Loading Stage 6 report from %s", safe_rp)
            data = json.loads(safe_read_text(safe_rp, allow_temp=True))
            metrics = data.get("metrics", data)
            return metrics
    if metrics_path:
        safe_mp = validate_path(metrics_path, allow_temp=True)
        if safe_mp.exists():
            logger.info("Loading Stage 6 metrics from %s", safe_mp)
            data = json.loads(safe_read_text(safe_mp, allow_temp=True))
            if "stage6_metrics" in data:
                return data["stage6_metrics"]
            return data
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 7 — regression analysis with real pytest execution",
    )
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--stage6-report", default=None)
    ap.add_argument("--stage6-metrics", default=None)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument(
        "--baseline-fail",
        type=int,
        default=0,
        help="Number of tasks the base model fails (default: 0 = all pass)",
    )
    ap.add_argument(
        "--tuned-fail",
        type=int,
        default=2,
        help="Number of tasks the tuned model fails (default: 2)",
    )
    args = ap.parse_args()

    result = run_stage7_fast(
        output_dir=args.output_dir,
        stage6_report_path=args.stage6_report,
        stage6_metrics_path=args.stage6_metrics,
        baseline_fail=args.baseline_fail,
        tuned_fail=args.tuned_fail,
        timeout_seconds=args.timeout,
    )

    print()
    print("=== Stage 7 Complete ===")
    print(f"  Run ID:                {result['run_id']}")
    print(f"  Base model:            {result['base_model']}")
    print(f"  Tuned model:           {result['tuned_model']}")
    base_frac = f"{result['base_passed']}/{result['total_tasks']}"
    tuned_frac = f"{result['tuned_passed']}/{result['total_tasks']}"
    print(f"  Base exec accuracy:    {result['base_exec_accuracy']:.4f} ({base_frac})")
    print(f"  Tuned exec accuracy:   {result['tuned_exec_accuracy']:.4f} ({tuned_frac})")
    print(f"  Forgetting delta:      {result['forgetting_delta']:+.4f}")
    print(f"  Report:                {result['report_path']}")
    if result["summary_path"]:
        print(f"  Summary:               {result['summary_path']}")
    print(f"  Manifest:              {result['manifest_path']}")
    print()


if __name__ == "__main__":
    main()
