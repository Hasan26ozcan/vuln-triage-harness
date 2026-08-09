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

app = typer.Typer(help="Stage 4: pre-fine-tuning baseline evaluation.")


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
