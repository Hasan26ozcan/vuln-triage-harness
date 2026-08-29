"""Stage 7 — regression / forgetting analysis on a real trained checkpoint.

After fine-tuning (Stage 5) and four-tier evaluation (Stage 6), the tuned model
may have suffered **catastrophic forgetting** — it gets good at vulnerability
tasks but loses general code-generation ability.  Stage 7 measures this by
running a set of HumanEval-style algorithm problems (12 tasks, bundled in
``DEFAULT_GENERAL_TASKS``) through both the **base** model and the **tuned**
checkpoint, then computing::

    forgetting_delta = tuned_exec_accuracy - base_exec_accuracy

A *negative* delta means the fine-tuned model forgot general coding ability.

This script loads a Stage 5 LoRA checkpoint (base model + PEFT adapter) the
same way ``scripts/run_stage6_only.py`` does, creates two ``QwenBackend``
instances (one for the base, one for the tuned LoRA), and runs the regression
analysis with ``LocalCodeTestRunner`` (subprocess-based pytest, no Docker needed).

Usage::

    python scripts/run_stage7_only.py [--base-model MODEL] [--checkpoint PATH]
        [--output-dir DIR] [--stage6-report PATH] [--stage6-metrics PATH]
        [--timeout SECONDS] [--no-cost]

The ``--stage6-report`` / ``--stage6-metrics`` options are optional: when
provided, a ``RegressionSummary`` (combining Stage 6 eval metrics + Stage 7
forgetting delta + cost-per-patch) is also written to ``output/stage7/``,
ready for the Stage 10 regression gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

# Ensure the project root is on sys.path when run as a standalone script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.security.paths import safe_read_text, validate_output_path, validate_path  # noqa: E402

# Unbuffered output so progress is visible in background runs.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_CHECKPOINT = "./output/stage5/sft_qlora/final_checkpoint"
DEFAULT_OUTPUT_DIR = "./output/stage7"
DEFAULT_TIMEOUT = 60


def run_stage7_real(
    base_model: str,
    checkpoint: str,
    output_dir: str,
    timeout_seconds: int = DEFAULT_TIMEOUT,
    stage6_report_path: str | None = None,
    stage6_metrics_path: str | None = None,
    inference_cost_usd: float = 0.0,
    training_cost_usd: float = 0.0,
    max_new_tokens: int | None = None,
) -> dict:
    """Run Stage 7 regression analysis with real model backends.

    Parameters
    ----------
    base_model:
        The pre-fine-tuning model ID (e.g. ``"Qwen/Qwen2.5-Coder-1.5B-Instruct"``).
    checkpoint:
        Path to the Stage 5 trained checkpoint (LoRA adapter dir or full model).
    output_dir:
        Directory to write ``regression_report.json`` and optionally
        ``regression_summary.json``.
    timeout_seconds:
        Per-task test execution timeout for ``LocalCodeTestRunner``.
    stage6_report_path:
        Optional path to Stage 6 ``eval_report.json``. When provided, a
        ``RegressionSummary`` is built from Stage 6 metrics + Stage 7 delta.
    stage6_metrics_path:
        Optional path to a flat metrics JSON (``output/stage5/eval_results.json``
        style with ``stage6_metrics`` block). Used as a fallback when
        ``stage6_report_path`` is not given.
    inference_cost_usd / training_cost_usd:
        Cost inputs for the cost-per-accepted-patch estimate in
        ``RegressionSummary``.
    max_new_tokens:
        Maximum generation tokens per model call. When ``None`` (default),
        the QwenBackend default (2048) is used. Lower values (e.g. 256)
        speed up CPU inference significantly.

    Returns
    -------
    dict
        A summary dict with run metadata, forgetting delta, and file paths.
    """
    from app.evaluation.backends import MissingAdapterWeightsError, QwenBackend
    from app.evaluation.general_capability import (
        DEFAULT_GENERAL_TASKS,
        LocalCodeTestRunner,
        RegressionConfig,
        build_regression_summary,
        run_regression_analysis,
    )
    from app.schemas.prediction_eval import EvalMetrics, RegressionSummary
    from scripts.verify_checkpoint import verify_checkpoint

    # ------------------------------------------------------------------
    # Step 1: Verify the checkpoint is complete, then load models
    # ------------------------------------------------------------------
    logger.info("=== Stage 7: Regression / Forgetting Analysis (real mode) ===")
    logger.info("Base model:  %s", base_model)
    logger.info("Checkpoint:  %s", checkpoint)
    logger.info("Tasks:       %d", len(DEFAULT_GENERAL_TASKS))

    # Hard pre-flight check: this raises with a clear message if the
    # checkpoint directory exists but the adapter weights don't (the exact
    # state that previously let Stage 7 silently evaluate the base model
    # twice and report a meaningless forgetting_delta of 0.0). No
    # allow_base_fallback escape hatch here — if you want to benchmark the
    # base model on purpose, pass the base model as --checkpoint directly,
    # don't point it at a broken LoRA dir.
    checkpoint_fingerprint = verify_checkpoint(checkpoint)
    logger.info("Checkpoint verified: %s", checkpoint_fingerprint)

    is_lora = checkpoint_fingerprint["checkpoint_type"] == "lora"

    logger.info("Loading base model backend ...")
    base_backend = QwenBackend(
        model_name=base_model,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else 2048,
    )

    logger.info("Loading tuned model backend ...")
    if is_lora:
        tuned_backend = QwenBackend(
            model_name=checkpoint,
            base_model=base_model,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 2048,
            allow_base_fallback=False,
        )
    else:
        tuned_backend = QwenBackend(
            model_name=checkpoint,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 2048,
        )

    # Force the tuned backend to actually load now (rather than lazily on
    # first generate() call) so a MissingAdapterWeightsError surfaces here,
    # before we spend time evaluating the base model on 12 tasks.
    try:
        tuned_backend._load()
    except MissingAdapterWeightsError:
        logger.error(
            "Tuned backend refused to load an incomplete checkpoint. "
            "This should be unreachable since verify_checkpoint() already "
            "passed — investigate a race condition or a second checkpoint "
            "path mismatch."
        )
        raise

    code_runner = LocalCodeTestRunner(timeout_seconds=timeout_seconds)

    # ------------------------------------------------------------------
    # Step 2: Run regression analysis
    # ------------------------------------------------------------------
    config = RegressionConfig(
        base_model=base_model,
        tuned_model=checkpoint,
        tasks=list(DEFAULT_GENERAL_TASKS),
        timeout_seconds=timeout_seconds,
    )

    logger.info("Evaluating base model on %d general-capability tasks ...", len(config.tasks))
    report = run_regression_analysis(
        config=config,
        base_backend=base_backend,
        tuned_backend=tuned_backend,
        runner=code_runner,
    )

    # ------------------------------------------------------------------
    # Step 3: Write regression report
    # ------------------------------------------------------------------
    out = validate_output_path(output_dir, allow_temp=True)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "regression_report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Regression report written to %s", report_path)

    # Print summary
    logger.info("")
    logger.info("Results:")
    logger.info("  Base exec accuracy:        %.4f", report.base_metrics.execution_accuracy)
    logger.info("  Tuned exec accuracy:       %.4f", report.tuned_metrics.execution_accuracy)
    logger.info("  Forgetting delta:          %+.4f", report.forgetting_delta)
    logger.info("  Elapsed:                   %.2fs", report.manifest.get("elapsed_seconds", "?"))

    if report.forgetting_delta >= 0:
        logger.info("  [OK] No forgetting — tuned model maintains or improves general capability.")
    else:
        logger.warning("  [WARN] Forgetting detected — tuned model lost general coding ability.")

    # ------------------------------------------------------------------
    # Step 4: Optionally build RegressionSummary (Stage 6 + Stage 7)
    # ------------------------------------------------------------------
    summary: RegressionSummary | None = None
    summary_path = out / "regression_summary.json"

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
            run_id=report.run_id,
            stage6_metrics=eval_metrics,
            regression_report=report,
            inference_cost_usd=inference_cost_usd,
            training_cost_usd=training_cost_usd,
        )
        summary_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Regression summary written to %s", summary_path)
        logger.info("  Cost per accepted patch (USD): %.4f", summary.cost_per_accepted_patch_usd)

    # ------------------------------------------------------------------
    # Step 5: Write manifest with run provenance
    # ------------------------------------------------------------------
    manifest = {
        "script": "scripts/run_stage7_only.py",
        "base_model": base_model,
        "checkpoint": checkpoint,
        "checkpoint_type": checkpoint_fingerprint["checkpoint_type"],
        # Pre-load check (file exists) — kept for backwards compatibility.
        "adapter_weights_available": is_lora,
        # Post-load check (actually merged into the pipeline) — this is the
        # field that should be trusted. If this is False for a LoRA
        # checkpoint, the run would have raised before getting here, so in
        # practice it's always True by the time we write this manifest.
        "adapter_applied": tuned_backend.adapter_applied,
        # Fingerprint of exactly which weights were used, so a "which
        # checkpoint was this?" question is never ambiguous again.
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "timeout_seconds": timeout_seconds,
        "max_new_tokens": max_new_tokens if max_new_tokens is not None else 2048,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage6_report_path": stage6_report_path,
        "stage6_metrics_path": stage6_metrics_path,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Manifest written to %s", manifest_path)

    return {
        "run_id": report.run_id,
        "base_model": base_model,
        "tuned_model": checkpoint,
        "base_exec_accuracy": report.base_metrics.execution_accuracy,
        "tuned_exec_accuracy": report.tuned_metrics.execution_accuracy,
        "forgetting_delta": report.forgetting_delta,
        "report_path": str(report_path),
        "summary_path": str(summary_path) if summary else None,
        "manifest_path": str(manifest_path),
    }


def _load_stage6_metrics(
    report_path: str | None,
    metrics_path: str | None,
) -> dict | None:
    """Load Stage 6 metrics from an eval_report.json or a flat metrics dict.

    Tries ``report_path`` first (expects ``{"metrics": {...}}`` structure),
    then falls back to ``metrics_path`` (expects a flat dict or a dict with a
    ``stage6_metrics`` key).  Returns ``None`` if neither path is provided or
    the files don't exist.
    """
    # 1. Try the Stage 6 eval_report.json (has a nested "metrics" block).
    if report_path:
        safe_rp = validate_path(report_path, allow_temp=True)
        if safe_rp.exists():
            logger.info("Loading Stage 6 report from %s", safe_rp)
            data = json.loads(safe_read_text(safe_rp, allow_temp=True))
            metrics = data.get("metrics", data)
            return metrics

    # 2. Try a flat metrics file (e.g. output/stage5/eval_results.json).
    if metrics_path:
        safe_mp = validate_path(metrics_path, allow_temp=True)
        if safe_mp.exists():
            logger.info("Loading Stage 6 metrics from %s", safe_mp)
            data = json.loads(safe_read_text(safe_mp, allow_temp=True))
            # eval_results.json nests under "stage6_metrics".
            if "stage6_metrics" in data:
                return data["stage6_metrics"]
            # Otherwise assume the top-level dict IS the metrics.
            return data

    return None


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 7 — regression / forgetting analysis on a trained checkpoint",
    )
    ap.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base (pre-fine-tuning) model name (default: {DEFAULT_BASE_MODEL})",
    )
    ap.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to Stage 5 trained checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    ap.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-task test execution timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    ap.add_argument(
        "--stage6-report",
        default=None,
        help="Path to Stage 6 eval_report.json (enables RegressionSummary output)",
    )
    ap.add_argument(
        "--stage6-metrics",
        default=None,
        help="Path to Stage 6 metrics JSON (flat or eval_results.json; enables RegressionSummary)",
    )
    ap.add_argument(
        "--inference-cost-usd",
        type=float,
        default=0.0,
        help="Total inference cost in USD for cost-per-accepted-patch estimate",
    )
    ap.add_argument(
        "--training-cost-usd",
        type=float,
        default=0.0,
        help="Amortized training cost in USD for cost-per-accepted-patch estimate",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for generation (default: 2048; lower e.g. 256 for CPU)",
    )
    args = ap.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        logger.error("Run Stage 5 training first, e.g.:")
        logger.error("  python -m app.training.cli sft --train-jsonl ./output/stage3/train.jsonl "
                     "--val-jsonl ./output/stage3/val.jsonl --output-dir ./output/stage5/sft_qlora")
        sys.exit(1)

    result = run_stage7_real(
        base_model=args.base_model,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout,
        stage6_report_path=args.stage6_report,
        stage6_metrics_path=args.stage6_metrics,
        inference_cost_usd=args.inference_cost_usd,
        training_cost_usd=args.training_cost_usd,
        max_new_tokens=args.max_new_tokens,
    )

    print()
    print("=== Stage 7 Complete ===")
    print(f"  Run ID:                {result['run_id']}")
    print(f"  Base model:            {result['base_model']}")
    print(f"  Tuned model:           {result['tuned_model']}")
    print(f"  Base exec accuracy:    {result['base_exec_accuracy']:.4f}")
    print(f"  Tuned exec accuracy:    {result['tuned_exec_accuracy']:.4f}")
    print(f"  Forgetting delta:      {result['forgetting_delta']:+.4f}")
    print(f"  Report:                {result['report_path']}")
    if result["summary_path"]:
        print(f"  Summary:               {result['summary_path']}")
    print(f"  Manifest:              {result['manifest_path']}")
    print()


if __name__ == "__main__":
    main()
