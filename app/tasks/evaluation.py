"""Celery tasks for four-tier evaluation (Stage 6).

These tasks run the evaluation pipeline asynchronously:

    Tier 1 (deterministic) → Tier 2 (static + embedding) → Tier 3 (exec) → Tier 4 (LLM judge)

The exec tier (Tier 3) applies patches to real code and runs pytest
inside a Docker sandbox — this is the most computationally expensive
step and benefits most from being backgrounded.

Usage::

    # Enqueue a full evaluation
    result = run_evaluation_task.delay(samples_json, predictions_json)

    # Check progress
    result.status   # "PENDING", "PROGRESS", "SUCCESS", "FAILURE"
    result.info     # Current stage description
    result.result   # Final EvalReport dict when complete
"""

from __future__ import annotations

import json
import logging

from app.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.evaluation.run_evaluation_task")
def run_evaluation_task(
    self,
    samples_json: str,
    predictions_json: str,
    sandbox_mode: str = "docker",
    skip_tier3: bool = False,
    skip_tier4: bool = False,
) -> dict:
    """Run the full four-tier evaluation pipeline asynchronously.

    Parameters
    ----------
    samples_json:
        JSON string of ``VulnSample`` objects.
    predictions_json:
        JSON string of ``ModelPrediction`` objects.
    sandbox_mode:
        Sandbox runner mode: ``"local"``, ``"docker"``, or ``"mock"``.
    skip_tier3:
        If ``True``, skip the exec (Tier 3) sandbox tests.
    skip_tier4:
        If ``True``, skip the LLM judge (Tier 4).

    Returns
    -------
    dict
        Complete evaluation report as a JSON-serializable dict,
        including per-tier metrics, the aggregate EvalMetrics,
        and the run manifest.
    """
    logger.info(
        "[run_evaluation_task] Starting: samples=%s predictions=%s sandbox=%s",
        len(samples_json),
        len(predictions_json),
        sandbox_mode,
    )

    try:
        # --- Parse inputs ---
        samples = json.loads(samples_json)
        predictions = json.loads(predictions_json)

        logger.info(
            "[run_evaluation_task] Loaded %d samples, %d predictions",
            len(samples),
            len(predictions),
        )

        from app.evaluation.runner import (
            EvaluationRunner,
            EvalConfig,
            load_samples,
            load_predictions,
        )
        from app.schemas.vuln import VulnSample
        from app.schemas.prediction_eval import ModelPrediction

        # Build VulnSample and ModelPrediction objects from JSON.
        sample_objs = [VulnSample(**s) for s in samples]
        pred_objs = [ModelPrediction(**p) for p in predictions]

        # --- Configure the evaluation runner ---
        eval_config = EvalConfig(
            base_model="Qwen2.5-Coder-7B-Instruct",
            sandbox_mode=sandbox_mode,
            skip_tier3=skip_tier3,
            skip_tier4=skip_tier4,
        )

        runner = EvaluationRunner(config=eval_config)

        # --- Update task status during execution ---
        self.update_state(state="PROGRESS", meta={"stage": "tier1_deterministic"})

        # --- Tier 1: Deterministic ---
        logger.info("[run_evaluation_task] Tier 1: deterministic")
        from app.evaluation.tier1_deterministic import DeterministicEvaluator
        tier1_evaluator = DeterministicEvaluator()
        tier1_results = tier1_evaluator.evaluate_all(sample_objs)

        self.update_state(state="PROGRESS", meta={"stage": "tier2_embedding_static"})

        # --- Tier 2: Static + embedding ---
        logger.info("[run_evaluation_task] Tier 2: static + embedding")
        from app.evaluation.tier2_embedding_static import StaticSignalEvaluator
        tier2_evaluator = StaticSignalEvaluator()
        pred_map = {p.sample_id: p for p in pred_objs}
        tier2_results = tier2_evaluator.evaluate_all(sample_objs, predictions=pred_map)

        # --- Tier 3: Exec (sandbox) ---
        tier3_results = []
        if not skip_tier3:
            self.update_state(state="PROGRESS", meta={"stage": "tier3_exec"})
            logger.info("[run_evaluation_task] Tier 3: exec sandbox")
            from app.evaluation.tier3_exec import ExecEvaluator
            if sandbox_mode == "docker":
                from app.evaluation.tier3_exec import DockerSandboxRunner
                tier3_evaluator = ExecEvaluator(sandbox_runner=DockerSandboxRunner())
            elif sandbox_mode == "local":
                from app.evaluation.tier3_exec import LocalSandboxRunner
                tier3_evaluator = ExecEvaluator(sandbox_runner=LocalSandboxRunner())
            else:
                from app.evaluation.tier3_exec import MockSandboxRunner
                tier3_evaluator = ExecEvaluator(sandbox_runner=MockSandboxRunner())
            tier3_results = tier3_evaluator.evaluate_all(sample_objs, pred_objs)

        # --- Tier 4: LLM Judge ---
        llm_judge_scores = []
        if not skip_tier4:
            self.update_state(state="PROGRESS", meta={"stage": "tier4_llm_judge"})
            logger.info("[run_evaluation_task] Tier 4: LLM judge")
            from app.evaluation.tier4_llm_judge import LlmJudge
            tier4_evaluator = LlmJudge(backend=None)  # Uses mock backend by default
            llm_judge_scores = tier4_evaluator.evaluate_all(sample_objs, pred_objs)

        # --- Compute metrics ---
        self.update_state(state="PROGRESS", meta={"stage": "compute_metrics"})
        logger.info("[run_evaluation_task] Computing metrics...")

        eval_report = runner.run(sample_objs, pred_objs)

        # Convert report to JSON-serializable dict.
        report_dict = eval_report.model_dump()

        logger.info(
            "[run_evaluation_task] Complete: model_cwe_macro_f1=%s",
            report_dict.get("metrics", {}).get("model_cwe_macro_f1"),
        )

        return {
            **report_dict,
            "task_id": self.request.id,
            "status": "completed",
        }

    except Exception as exc:
        logger.exception("[run_evaluation_task] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=120, max_retries=2)


@celery_app.task(bind=True, name="app.tasks.evaluation.run_baseline_task")
def run_baseline_task(
    self,
    gold_eval_path: str,
    strategy: str = "zero_shot",
    mock: bool = True,
) -> dict:
    """Run Stage 4 baseline evaluation asynchronously.

    Parameters
    ----------
    gold_eval_path:
        Path to the gold-eval JSONL file.
    strategy:
        Prompt strategy: ``"zero_shot"``, ``"few_shot"``, etc.
    mock:
        If ``True``, use mock backend (no model download needed).

    Returns
    -------
    dict
        Baseline metrics and predictions as a JSON-serializable dict.
    """
    logger.info(
        "[run_baseline_task] Starting: path=%s strategy=%s mock=%s",
        gold_eval_path,
        strategy,
        mock,
    )

    try:
        from importlib import import_module
        from app.evaluation.baseline import BaselineConfig, run_baseline

        config = BaselineConfig(
            strategy=strategy,
            base_model="Qwen2.5-Coder-7B-Instruct",
        )

        backend_mod = import_module("app.evaluation.backends")
        if mock:
            backend = backend_mod.MockBackend(
                responses={},
                default='{"cwe_id": "CWE-89", "severity": "high", '
                       '"explanation": "Baseline mock response.", "patch_diff": ""}',
            )
        else:
            raise ValueError("Non-mock baseline requires a real model; use mock=True for CI.")

        result = run_baseline(
            gold_eval_path=gold_eval_path,
            output_dir=f"output/baseline_task_{self.request.id[:8]}",
            config=config,
            backend=backend,
        )

        return {
            "metrics": result.metrics.model_dump(),
            "num_predictions": len(result.predictions),
            "task_id": self.request.id,
            "status": "completed",
        }

    except Exception as exc:
        logger.exception("[run_baseline_task] Failed: %s", exc)
        raise self.retry(exc=exc, countdown=60, max_retries=3)
