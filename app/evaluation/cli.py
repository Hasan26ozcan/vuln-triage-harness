"""Stage 4 CLI: pre-fine-tuning baseline evaluation.

Usage:
    # Run zero-shot baseline on the gold-eval set
    python -m app.evaluation.cli baseline --gold-eval eval/gold_set/gold.jsonl

    # Run few-shot baseline with 3 in-context examples from Stage 3 output
    python -m app.evaluation.cli baseline --gold-eval eval/gold_set/gold.jsonl \\
        --strategy few-shot --num-shots 3 --few-shot-examples output/stage3/train.jsonl

    # Re-evaluate saved predictions without re-running inference
    python -m app.evaluation.cli evaluate --predictions output/stage4/predictions.jsonl \\
        --gold-eval eval/gold_set/gold.jsonl

    # Use a smaller model for fast iteration
    python -m app.evaluation.cli baseline --gold-eval eval/gold_set/gold.jsonl \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

from __future__ import annotations

import json
import logging
import os

import typer

from app.evaluation.backends import MockBackend
from app.evaluation.baseline import (
    BaselineConfig,
    run_baseline,
)
from app.evaluation.metrics import compute_metrics
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample

app = typer.Typer(help="Evaluation tools for the vuln-triage-harness.")

# Stage 6 subcommands — lazy-import to keep Stage 4 CLI lightweight.
_stage6_app = typer.Typer(help="Stage 6: four-tier evaluation harness.")


@app.command(name="stage6")
def stage6(
    gold_eval: str = typer.Option(
        ..., "--gold-eval", "-g",
        help="Path to gold-eval JSONL file (one VulnSample per line).",
    ),
    predictions: str = typer.Option(
        ..., "--predictions", "-p",
        help="Path to ModelPrediction JSONL file.",
    ),
    output_dir: str = typer.Option(
        "./output/stage6", "--output-dir", "-o",
        help="Directory to write the EvalReport JSON.",
    ),
    base_model: str = typer.Option(
        "unknown", "--base-model", "-m",
        help="Model name being evaluated.",
    ),
    embedding_model: str = typer.Option(
        None, "--embedding-model", "-e",
        help="Sentence-transformers model for Tier 2 embedding similarity (optional).",
    ),
    sandbox_mode: str = typer.Option(
        "mock", "--sandbox-mode",
        help="Sandbox mode: mock | local | docker.",
    ),
    llm_judge_model: str = typer.Option(
        None, "--llm-judge-model",
        help="LLM model for Tier 4 judge (optional; requires OPENAI_API_KEY or similar).",
    ),
    skip_tier3: bool = typer.Option(
        False, "--skip-tier3",
        help="Skip exec-based evaluation (Tier 3).",
    ),
    skip_tier4: bool = typer.Option(
        False, "--skip-tier4",
        help="Skip LLM judge evaluation (Tier 4). Saves LLM cost.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the Stage 6 four-tier evaluation harness."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.evaluation.runner import EvalConfig, EvaluationRunner, load_predictions, load_samples

    config = EvalConfig(
        base_model=base_model,
        embedding_model=embedding_model,
        sandbox_mode=sandbox_mode,
        llm_judge_model=llm_judge_model,
        skip_tier3=skip_tier3,
        skip_tier4=skip_tier4,
    )

    runner = EvaluationRunner(config=config)
    samples = load_samples(gold_eval)
    preds = load_predictions(predictions)

    typer.echo("Running Stage 6 evaluation")
    typer.echo(f"Samples:     {len(samples)}")
    typer.echo(f"Predictions: {len(preds)}")
    typer.echo(f"Sandbox:     {sandbox_mode}")
    typer.echo(f"Skip Tier 3: {skip_tier3}")
    typer.echo(f"Skip Tier 4: {skip_tier4}")

    report = runner.run(samples, preds)

    # Write report
    import os
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Print summary
    typer.echo("")
    typer.echo(f"Run ID: {report.run_id}")
    typer.echo(f"Stage:  {report.stage}")
    typer.echo("")
    m = report.metrics
    typer.echo("Metrics:")
    typer.echo(f"  Tier1 CWE Macro-F1:     {m.tier1_cwe_macro_f1:.4f}")
    typer.echo(f"  Tier1 Coverage:         {m.tier1_coverage:.4f}")
    typer.echo(f"  Tier2 CWE Macro-F1:     {m.tier2_cwe_macro_f1:.4f}")
    typer.echo(f"  Tier2 Coverage:         {m.tier2_coverage:.4f}")
    typer.echo(f"  Model CWE Macro-F1:     {m.model_cwe_macro_f1:.4f}")
    typer.echo(f"  Exec Pass Rate:         {m.exec_pass_rate:.4f}")
    typer.echo(f"  Patch Applies:          {m.patch_applies_rate:.4f}")
    typer.echo(f"  Build Succeeds:         {m.build_succeeds_rate:.4f}")
    typer.echo(f"  Hallucination Rate:     {m.hallucination_rate:.4f}")
    typer.echo(f"  Patch Coverage:         {m.avg_patch_coverage:.4f}")
    if m.avg_explanation_quality is not None:
        typer.echo(f"  Avg Explanation Quality: {m.avg_explanation_quality:.4f}")
    if m.avg_patch_minimality is not None:
        typer.echo(f"  Avg Patch Minimality:   {m.avg_patch_minimality:.4f}")
    typer.echo("")
    typer.echo("Per-class F1 (model):")
    for cwe, stats in sorted(m.per_class.items()):
        typer.echo(
            f"  {cwe:10s}  P={stats['precision']:.4f}  "
            f"R={stats['recall']:.4f}  F1={stats['f1']:.4f}"
        )
    typer.echo("")
    typer.echo(f"Report written to: {report_path}")


# Stage 7 subcommands


@app.command(name="stage7")
def stage7(
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-Coder-7B-Instruct", "--base-model", "-b",
        help="Base (pre-fine-tuning) model name or HuggingFace path.",
    ),
    tuned_model: str = typer.Option(
        ..., "--tuned-model", "-t",
        help="Tuned (post-fine-tuning) checkpoint name or path.",
    ),
    output_dir: str = typer.Option(
        "./output/stage7", "--output-dir", "-o",
        help="Directory to write the RegressionReport JSON.",
    ),
    mock: bool = typer.Option(
        False, "--mock",
        help="Use MockBackend + MockCodeTestRunner (no model download, no subprocess).",
    ),
    timeout_seconds: int = typer.Option(
        30, "--timeout",
        help="Per-task test execution timeout in seconds (local runner only).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run Stage 7 regression / forgetting analysis.

    Evaluates a base model and a fine-tuned model on a set of general
    (non-security) code-generation tasks. The forgetting delta is the
    difference in execution accuracy:

        delta = tuned_exec_accuracy - base_exec_accuracy

    A negative delta means the fine-tuned model forgot general coding ability.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.evaluation.backends import MockBackend
    from app.evaluation.general_capability import (
        DEFAULT_GENERAL_TASKS,
        MockCodeTestRunner,
        RegressionConfig,
        run_regression_analysis,
    )

    config = RegressionConfig(
        base_model=base_model,
        tuned_model=tuned_model,
        timeout_seconds=timeout_seconds,
    )

    if mock:
        # MockBackend returns a trivial function for every prompt;
        # MockCodeTestRunner returns canned pass/fail without subprocess.
        base_backend = MockBackend(
            default="def solution():\n    return None\n",
            responses={},
        )
        tuned_backend = MockBackend(
            default="def solution():\n    return None\n",
            responses={},
        )
        code_runner = MockCodeTestRunner(default_passed=True)
    else:
        from app.evaluation.backends import QwenBackend
        from app.evaluation.general_capability import LocalCodeTestRunner

        base_backend = QwenBackend(model_name=base_model)
        tuned_backend = QwenBackend(model_name=tuned_model)
        code_runner = LocalCodeTestRunner(timeout_seconds=timeout_seconds)

    typer.echo("Running Stage 7: regression / forgetting analysis")
    typer.echo(f"Base model:  {base_model}")
    typer.echo(f"Tuned model: {tuned_model}")
    typer.echo(f"Tasks:       {len(DEFAULT_GENERAL_TASKS)}")
    typer.echo(f"Mock mode:   {mock}")

    report = run_regression_analysis(
        config=config,
        base_backend=base_backend,
        tuned_backend=tuned_backend,
        runner=code_runner,
    )

    # Write report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "regression_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Print summary
    typer.echo("")
    typer.echo(f"Run ID:                    {report.run_id}")
    typer.echo(f"Base exec accuracy:       {report.base_metrics.execution_accuracy:.4f}")
    typer.echo(f"Tuned exec accuracy:      {report.tuned_metrics.execution_accuracy:.4f}")
    typer.echo(f"Forgetting delta:         {report.forgetting_delta:+.4f}")
    typer.echo("")
    if report.forgetting_delta >= 0:
        typer.echo(
            "✅ No forgetting — tuned model maintains or improves general capability."
        )
    else:
        typer.echo(
            "⚠️  Forgetting detected — tuned model lost general coding ability."
        )
    typer.echo("")
    typer.echo(f"Report written to: {report_path}")


# Stage 8 subcommands


@app.command(name="stage8")
def stage8(
    source_checkpoint: str = typer.Option(
        ..., "--source-checkpoint", "-s",
        help="Path to the Stage 5 trained checkpoint to quantize.",
    ),
    base_model: str = typer.Option(
        "Qwen/Qwen2.5-Coder-7B-Instruct", "--base-model", "-b",
        help="Base model name (for reporting / metadata).",
    ),
    output_dir: str = typer.Option(
        "./output/stage8", "--output-dir", "-o",
        help="Directory to write the QuantReport JSON.",
    ),
    methods: str = typer.Option(
        "gptq,awq,gguf", "--methods", "-m",
        help="Comma-separated list of quantization methods (gptq, awq, gguf, none).",
    ),
    bit_widths: str = typer.Option(
        "4", "--bits",
        help="Comma-separated bit-widths (for GPTQ/AWQ; GGUF uses its own quant_types).",
    ),
    mock: bool = typer.Option(
        False, "--mock",
        help="Use MockQuantizer (deterministic, no ML deps, no GPU).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Use heuristic estimates only (no quantization, no ML deps).",
    ),
    target_vram_gb: float = typer.Option(
        None, "--target-vram",
        help="VRAM budget in GB — filters results in select_best_config.",
    ),
    target_size_gb: float = typer.Option(
        None, "--target-size",
        help="On-disk size budget in GB — filters results in select_best_config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the Stage 8 quantization matrix (GPTQ / AWQ / GGUF).

    Produces a ``QuantReport`` JSON with per-method×bit-width results
    and a best-config recommendation based on quality vs. VRAM / size.

    In ``--mock`` mode, no heavy ML dependencies are required.
    In ``--dry-run`` mode, heuristic estimates are used instead of
    calling any quantizer.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.schemas.quantization import QuantMethod, QuantStatus
    from app.quantization.config import QuantConfig
    from app.quantization.quantizer import run_quantization_matrix

    # Parse comma-separated method and bit-width strings.
    method_map = {m.value: m for m in QuantMethod}
    resolved_methods: list[QuantMethod] = []
    for m_str in methods.split(","):
        key = m_str.strip().lower()
        if not key:
            continue
        if key not in method_map:
            typer.echo(
                f"Error: unknown quantization method '{m_str}'. "
                f"Valid: {', '.join(method_map.keys())}",
                err=True,
            )
            raise typer.Exit(1)
        resolved_methods.append(method_map[key])

    resolved_bits = [int(b) for b in bit_widths.split(",")]

    config = QuantConfig(
        base_model=base_model,
        source_checkpoint=source_checkpoint,
        output_base=output_dir,
        methods=resolved_methods,
        bit_widths=resolved_bits,
        dry_run=dry_run,
        mock=mock,
        target_vram_gb=target_vram_gb,
        target_size_gb=target_size_gb,
    )

    # Print validation warnings.
    for warning in config.all_warnings():
        typer.echo(f"Warning: {warning}", err=True)

    typer.echo("Running Stage 8: quantization matrix")
    typer.echo(f"Base model:    {base_model}")
    typer.echo(f"Checkpoint:    {source_checkpoint}")
    typer.echo(f"Methods:       {[m.value for m in resolved_methods]}")
    typer.echo(f"Bit widths:    {resolved_bits}")
    typer.echo(f"Mock mode:     {mock}")
    typer.echo(f"Dry run:       {dry_run}")

    report = run_quantization_matrix(config)

    # Write report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "quant_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Print summary
    typer.echo("")
    typer.echo(f"Run ID:            {report.run_id}")
    typer.echo(f"Total results:     {len(report.results)}")
    if report.best_result:
        best = report.best_result
        typer.echo(f"Best config:       {best.quant_method.value} @ {best.bit_width}-bit")
        typer.echo(f"  Estimated VRAM:  {best.estimated_vram_gb} GB")
        typer.echo(f"  Est. size:       {best.quantized_model_size_gb} GB")
        if best.model_cwe_macro_f1 is not None:
            typer.echo(f"  Est. CWE-F1:     {best.model_cwe_macro_f1:.4f}")
        if best.tokens_per_sec is not None:
            typer.echo(f"  Est. tokens/sec: {best.tokens_per_sec:.1f}")
    else:
        typer.echo("Best config:       none (no completed results)")
    typer.echo("")
    typer.echo("Per-result summary:")
    for r in report.results:
        status_icon = "✓" if r.status == QuantStatus.COMPLETED else "✗"
        typer.echo(
            f"  {status_icon} {r.quant_method.value:5s} @ {str(r.bit_width or '?'):>2s}-bit  "
            f"VRAM={r.estimated_vram_gb:>4.1f}GB  "
            f"size={r.quantized_model_size_gb:>4.1f}GB  "
            f"[{r.status.value}]"
        )
    typer.echo("")
    typer.echo(f"Report written to: {report_path}")


# Legacy Stage 4 commands — keep existing behavior.


@app.command()
def baseline(
    gold_eval: str = typer.Option(
        ..., "--gold-eval", "-g",
        help="Path to gold-eval JSONL file (one VulnSample per line).",
    ),
    output_dir: str = typer.Option(
        "./output/stage4", "--output-dir", "-o",
        help="Directory to write predictions and metrics to.",
    ),
    strategy: str = typer.Option(
        "zero_shot", "--strategy", "-s",
        help="Prompting strategy: zero_shot or few_shot.",
    ),
    num_shots: int = typer.Option(
        3, "--num-shots", "-n",
        help="Number of in-context examples (few-shot only).",
    ),
    model: str = typer.Option(
        "Qwen/Qwen2.5-Coder-7B-Instruct", "--model", "-m",
        help="Base model to evaluate.",
    ),
    temperature: float = typer.Option(
        0.2, "--temperature", "-t",
        help="Sampling temperature (lower = more deterministic).",
    ),
    max_new_tokens: int = typer.Option(
        2048, "--max-new-tokens",
        help="Maximum new tokens to generate per sample.",
    ),
    few_shot_examples: str = typer.Option(
        None, "--few-shot-examples", "-f",
        help="Path to Stage 3 train JSONL for few-shot examples (few-shot strategy only).",
    ),
    mock: bool = typer.Option(
        False, "--mock",
        help="Use MockBackend (for testing — produces deterministic fake predictions).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run baseline evaluation: zero-shot or few-shot on the gold-eval set."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if strategy not in ("zero_shot", "few_shot"):
        typer.echo(
            f"Error: --strategy must be 'zero_shot' or 'few_shot', got '{strategy}'",
            err=True,
        )
        raise typer.Exit(1)

    config = BaselineConfig(
        strategy=strategy,
        num_shots=num_shots,
        base_model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if mock:
        backend = MockBackend(
            responses={
                "CWE-89": '{"cwe_id": "CWE-89", "severity": "high", '
                            '"explanation": "SQL injection via string concatenation.", '
                            '"patch_diff": "--- a/app.py\\n+++ b/app.py\\n- old\\n+ new"}',
            },
            default='{"cwe_id": "CWE-89", "severity": "high", '
                     '"explanation": "Mock explanation.", "patch_diff": ""}',
        )
    else:
        backend = None  # run_baseline will create a QwenBackend

    typer.echo(f"Running Stage 4 baseline (strategy={strategy}, model={model})")
    typer.echo(f"Gold-eval: {gold_eval}")
    typer.echo(f"Output:    {output_dir}")
    if strategy == "few_shot":
        if not few_shot_examples:
            typer.echo(
                "Warning: --few-shot-examples not provided, falling back to zero-shot",
                err=True,
            )
        else:
            typer.echo(f"Examples:  {few_shot_examples} ({num_shots} shots)")

    try:
        result = run_baseline(
            gold_eval_path=gold_eval,
            output_dir=output_dir,
            config=config,
            backend=backend,
            few_shot_examples_path=few_shot_examples,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    # Print summary
    typer.echo("")
    typer.echo(f"Run ID:       {result.run_id}")
    typer.echo(f"Predictions:  {result.num_predictions}")
    typer.echo(f"Parse failures: {result.num_parse_failures}")
    typer.echo(f"Total attempted: {result.total_attempted}")
    typer.echo("")
    typer.echo("Metrics:")
    typer.echo(f"  CWE Macro-F1:          {result.metrics.cwe_macro_f1:.4f}")
    typer.echo(f"  CWE Micro Accuracy:    {result.metrics.cwe_micro_accuracy:.4f}")
    typer.echo(f"  Severity Accuracy:     {result.metrics.severity_accuracy:.4f}")
    typer.echo(f"  Hallucination Rate:    {result.metrics.hallucination_rate:.4f}")
    typer.echo(f"  Patch Coverage:        {result.metrics.patch_coverage:.4f}")
    typer.echo("")
    typer.echo("Per-class F1:")
    for cwe, stats in sorted(result.metrics.per_class.items()):
        typer.echo(
            f"  {cwe:10s}  P={stats['precision']:.4f}  "
            f"R={stats['recall']:.4f}  F1={stats['f1']:.4f}  n={stats['support']}"
        )
    typer.echo("")
    typer.echo(f"Predictions written to: {os.path.join(output_dir, 'predictions.jsonl')}")
    typer.echo(f"Metrics written to:     {os.path.join(output_dir, 'metrics.json')}")
    typer.echo(f"Manifest written to:    {os.path.join(output_dir, 'manifest.json')}")


@app.command()
def evaluate(
    predictions: str = typer.Option(
        ..., "--predictions", "-p",
        help="Path to predictions.jsonl from a previous baseline run.",
    ),
    gold_eval: str = typer.Option(
        ..., "--gold-eval", "-g",
        help="Path to gold-eval JSONL file for ground truth.",
    ),
    output_dir: str = typer.Option(
        None, "--output-dir", "-o",
        help="Optional directory to write re-computed metrics.json.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-compute metrics from saved predictions without re-running inference."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load predictions
    preds: list[ModelPrediction] = []
    with open(predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            preds.append(ModelPrediction(**data))

    # Load gold-eval
    gold_samples: list[VulnSample] = []
    with open(gold_eval, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            gold_samples.append(VulnSample(**data))

    run_id = preds[0].run_id if preds else "re-evaluated"
    metrics = compute_metrics(preds, gold_samples, run_id=run_id)

    typer.echo(f"Loaded {len(preds)} predictions and {len(gold_samples)} gold-eval samples")
    typer.echo(f"Run ID: {run_id}")
    typer.echo("")
    typer.echo("Metrics:")
    typer.echo(f"  CWE Macro-F1:          {metrics.cwe_macro_f1:.4f}")
    typer.echo(f"  CWE Micro Accuracy:    {metrics.cwe_micro_accuracy:.4f}")
    typer.echo(f"  Severity Accuracy:     {metrics.severity_accuracy:.4f}")
    typer.echo(f"  Hallucination Rate:    {metrics.hallucination_rate:.4f}")
    typer.echo(f"  Patch Coverage:        {metrics.patch_coverage:.4f}")
    typer.echo("")
    typer.echo("Per-class F1:")
    for cwe, stats in sorted(metrics.per_class.items()):
        typer.echo(
            f"  {cwe:10s}  P={stats['precision']:.4f}  "
            f"R={stats['recall']:.4f}  F1={stats['f1']:.4f}  n={stats['support']}"
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.__dict__, f, indent=2)
        typer.echo(f"\nMetrics written to: {metrics_path}")


if __name__ == "__main__":
    app()
