"""Stage 5 CLI: SFT training, LoRA rank sweep, DPO, and run inspection.

Usage:
    # Full-parameter SFT (no quantization — needs >=15 GB VRAM)
    python -m app.training.cli sft --train-jsonl output/stage3/train.jsonl \\
        --val-jsonl output/stage3/val.jsonl --no-4bit

    # QLoRA SFT (4-bit NF4 — fits 8 GB VRAM)
    python -m app.training.cli sft --train-jsonl output/stage3/train.jsonl \\
        --val-jsonl output/stage3/val.jsonl

    # LoRA rank sweep across [8, 16, 32, 64, 128]
    python -m app.training.cli lora-sweep --train-jsonl output/stage3/train.jsonl \\
        --val-jsonl output/stage3/val.jsonl --ranks 8,16,32,64,128

    # DPO preference alignment (from an SFT checkpoint)
    python -m app.training.cli dpo --train-jsonl output/stage3/train.jsonl \\
        --sft-checkpoint output/stage5/sft_qlora/final_checkpoint --beta 0.1

    # Dry-run: estimate steps/VRAM without a GPU
    python -m app.training.cli sft --train-jsonl output/stage3/train.jsonl --dry-run

    # List all training runs in PostgreSQL
    python -m app.training.cli list-runs

    # Inspect a specific run
    python -m app.training.cli inspect <run_id>
"""

from __future__ import annotations

import json
import logging
import os

import typer

from app.training.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DPO_BETA,
    DEFAULT_FAST_MODEL,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DPOConfig,
    SFTConfig,
    SweepConfig,
)
from app.training.trainer_sft import TrainingUnavailableError

app = typer.Typer(help="Stage 5: SFT / LoRA sweep / DPO training.")


# ---------------------------------------------------------------------------
# Shared option defaults
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT: str = "./output/stage5"


def _resolve_train_path(train_jsonl: str, config: SFTConfig | DPOConfig) -> str:
    """Pick the train JSONL path from CLI or config."""
    return train_jsonl or config.train_jsonl


def _resolve_val_path(val_jsonl: str, config: SFTConfig | DPOConfig) -> str:
    return val_jsonl or config.val_jsonl


# ---------------------------------------------------------------------------
# SFT command
# ---------------------------------------------------------------------------


@app.command()
def sft(
    train_jsonl: str = typer.Option(
        "",
        "--train-jsonl",
        "-t",
        help="Path to Stage 3 train.jsonl.",
    ),
    val_jsonl: str = typer.Option(
        "",
        "--val-jsonl",
        "-v",
        help="Path to Stage 3 val.jsonl.",
    ),
    model: str = typer.Option(
        DEFAULT_BASE_MODEL,
        "--model",
        "-m",
        help="Base model to fine-tune.",
    ),
    output_dir: str = typer.Option(
        DEFAULT_OUTPUT,
        "--output-dir",
        "-o",
        help="Directory for checkpoints and logs.",
    ),
    no_4bit: bool = typer.Option(
        False,
        "--no-4bit",
        help="Disable 4-bit quantization (full-parameter SFT — needs >=15 GB VRAM).",
    ),
    lora_r: int = typer.Option(8, "--lora-r", help="LoRA rank."),
    lora_alpha: int = typer.Option(16, "--lora-alpha", help="LoRA alpha."),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout", help="LoRA dropout."),
    learning_rate: float = typer.Option(
        DEFAULT_LEARNING_RATE,
        "--learning-rate",
        help="Learning rate.",
    ),
    epochs: int = typer.Option(
        DEFAULT_NUM_TRAIN_EPOCHS,
        "--epochs",
        "-e",
        help="Number of training epochs.",
    ),
    batch_size: int = typer.Option(1, "--batch-size", help="Per-device batch size."),
    grad_accum: int = typer.Option(8, "--grad-accum", help="Gradient accumulation steps."),
    run_name: str = typer.Option(None, "--run-name", help="Human-readable run name."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate steps/VRAM without training."),
    verbose: bool = typer.Option(False, "--verbose", "-V"),
) -> None:
    """Run SFT (Supervised Fine-Tuning) — full or QLoRA."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    config = SFTConfig(
        base_model=model,
        output_dir=output_dir,
        use_4bit=not no_4bit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        run_name=run_name,
    )

    # Validate config
    warnings = _safe_validate(config)
    for w in warnings:
        typer.echo(f"Warning: {w}", err=True)

    try:
        from app.training.trainer_sft import run_sft

        result = run_sft(config, dry_run=dry_run)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except TrainingUnavailableError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo("Hint: use --dry-run to estimate steps without a GPU.", err=True)
        raise typer.Exit(1) from exc

    _print_training_result(result, typer)

    # Persist training_result.json to the output directory (mirrors DPO pattern).
    if result.status == "completed":
        from dataclasses import asdict

        result_path = os.path.join(config.output_dir, "training_result.json")
        os.makedirs(config.output_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        typer.echo(f"Saved result to {result_path}")


# ---------------------------------------------------------------------------
# LoRA sweep command
# ---------------------------------------------------------------------------


@app.command(name="lora-sweep")
def lora_sweep(
    train_jsonl: str = typer.Option(
        "",
        "--train-jsonl",
        "-t",
        help="Path to Stage 3 train.jsonl.",
    ),
    val_jsonl: str = typer.Option(
        "",
        "--val-jsonl",
        "-v",
        help="Path to Stage 3 val.jsonl.",
    ),
    model: str = typer.Option(
        DEFAULT_BASE_MODEL,
        "--model",
        "-m",
        help="Base model to fine-tune.",
    ),
    output_dir: str = typer.Option(
        DEFAULT_OUTPUT,
        "--output-dir",
        "-o",
        help="Base directory for sweep outputs.",
    ),
    ranks: str = typer.Option(
        "8,16,32,64,128",
        "--ranks",
        "-r",
        help="Comma-separated LoRA ranks to try (e.g. 8,16,32,64,128).",
    ),
    learning_rate: float = typer.Option(DEFAULT_LEARNING_RATE, "--learning-rate"),
    epochs: int = typer.Option(DEFAULT_NUM_TRAIN_EPOCHS, "--epochs", "-e"),
    grad_accum: int = typer.Option(8, "--grad-accum"),
    run_name: str = typer.Option(None, "--run-name"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    no_persist: bool = typer.Option(False, "--no-persist", help="Skip PostgreSQL persistence."),
    verbose: bool = typer.Option(False, "--verbose", "-V"),
) -> None:
    """Run a LoRA rank sweep across multiple ranks."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    rank_list = [int(r.strip()) for r in ranks.split(",") if r.strip()]
    if not rank_list:
        typer.echo("Error: --ranks must contain at least one integer.", err=True)
        raise typer.Exit(1)

    config = SweepConfig(
        base_model=model,
        output_dir=output_dir,
        ranks=rank_list,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        lora_alpha=16,
        lora_dropout=0.05,
        run_name=run_name,
    )

    warnings = _safe_validate(config)
    for w in warnings:
        typer.echo(f"Warning: {w}", err=True)

    typer.echo(f"Starting LoRA sweep: ranks={rank_list}, model={model}")

    try:
        from app.training.sweep import run_lora_sweep

        result = run_lora_sweep(
            config,
            dry_run=dry_run,
            persist=not no_persist,
        )
    except TrainingUnavailableError as exc:
        typer.echo(f"Error: {exc}", err=True)
        if not dry_run:
            typer.echo("Hint: use --dry-run to estimate steps without a GPU.", err=True)
        raise typer.Exit(1) from exc

    typer.echo("")
    typer.echo(f"Sweep: {result.sweep_name}  ({result.num_runs} runs)")
    typer.echo(f"Best rank: {result.best_rank}  (val_loss={result.best_val_loss})")
    typer.echo("")
    typer.echo("Per-rank summary:")
    typer.echo(
        f"  {'rank':>5s}  {'status':>10s}  {'train_loss':>12s}  "
        f"{'val_loss':>10s}  {'vram_gb':>8s}  {'minutes':>8s}"
    )
    for r in result.results:
        rank = r.hyperparams.get("lora_r", "?")
        tl = f"{r.final_train_loss:.4f}" if r.final_train_loss is not None else "—"
        vl = f"{r.final_val_loss:.4f}" if r.final_val_loss is not None else "—"
        typer.echo(
            f"  {str(rank):>5s}  {r.status:>10s}  {tl:>12s}  {vl:>10s}  "
            f"{r.peak_vram_gb:>8.2f}  {r.train_time_minutes:>8.2f}"
        )


# ---------------------------------------------------------------------------
# DPO command
# ---------------------------------------------------------------------------


@app.command()
def dpo(
    train_jsonl: str = typer.Option(
        "",
        "--train-jsonl",
        "-t",
        help="Path to Stage 3 train.jsonl (the 'chosen' responses).",
    ),
    val_jsonl: str = typer.Option(
        "",
        "--val-jsonl",
        "-v",
        help="Optional validation JSONL.",
    ),
    model: str = typer.Option(
        DEFAULT_FAST_MODEL,
        "--model",
        "-m",
        help="Base model to DPO-tune.",
    ),
    sft_checkpoint: str = typer.Option(
        "",
        "--sft-checkpoint",
        "-s",
        help="Path to an SFT checkpoint to initialize from (recommended).",
    ),
    output_dir: str = typer.Option(
        DEFAULT_OUTPUT,
        "--output-dir",
        "-o",
    ),
    beta: float = typer.Option(
        DEFAULT_DPO_BETA,
        "--beta",
        "-b",
        help="DPO KL penalty coefficient.",
    ),
    learning_rate: float = typer.Option(DEFAULT_LEARNING_RATE, "--learning-rate"),
    epochs: int = typer.Option(DEFAULT_NUM_TRAIN_EPOCHS, "--epochs", "-e"),
    batch_size: int = typer.Option(1, "--batch-size"),
    grad_accum: int = typer.Option(8, "--grad-accum"),
    run_name: str = typer.Option(None, "--run-name"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "--verbose", "-V"),
) -> None:
    """Run DPO (Direct Preference Optimization) training."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    config = DPOConfig(
        base_model=model,
        output_dir=output_dir,
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        train_jsonl=train_jsonl,
        sft_checkpoint=sft_checkpoint,
        run_name=run_name,
    )

    warnings = _safe_validate(config)
    for w in warnings:
        typer.echo(f"Warning: {w}", err=True)

    try:
        from app.training.trainer_dpo import run_dpo

        result = run_dpo(config, dry_run=dry_run)
    except FileNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    except TrainingUnavailableError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo("Hint: use --dry-run to estimate steps without a GPU.", err=True)
        raise typer.Exit(1) from exc

    _print_training_result(result, typer)

    # Persist training_result.json to the output directory (mirrors SFT pattern).
    if result.status == "completed":
        from dataclasses import asdict

        result_path = os.path.join(config.output_dir, "training_result.json")
        os.makedirs(config.output_dir, exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        typer.echo(f"Saved result to {result_path}")


# ---------------------------------------------------------------------------
# Inspection commands
# ---------------------------------------------------------------------------


@app.command(name="list-runs")
def list_runs(
    limit: int = typer.Option(50, "--limit", "-n", help="Max runs to list."),
    method: str = typer.Option(None, "--method", "-m", help="Filter by method."),
    status: str = typer.Option(None, "--status", help="Filter by status."),
) -> None:
    """List training runs from PostgreSQL."""
    from app.training.experiment import list_training_runs

    try:
        runs = list_training_runs(limit=limit, method=method, status=status)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Error querying runs: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not runs:
        typer.echo("No training runs found.")
        return

    typer.echo(
        f"{'run_id':<40s}  {'method':<12s}  {'status':<12s}  {'train_loss':>10s}  {'val_loss':>10s}"
    )
    typer.echo("-" * 90)
    for r in runs:
        tl = f"{r.final_train_loss:.4f}" if r.final_train_loss is not None else "—"
        vl = f"{r.final_val_loss:.4f}" if r.final_val_loss is not None else "—"
        typer.echo(f"{r.run_id:<40s}  {r.method:<12s}  {r.status:<12s}  {tl:>10s}  {vl:>10s}")


@app.command()
def inspect(
    run_id: str = typer.Option(..., "--run-id", "-r", help="Run ID to inspect."),
) -> None:
    """Inspect a single training run's metadata."""
    from app.training.experiment import load_training_run

    run = load_training_run(run_id)
    if run is None:
        typer.echo(f"Run not found: {run_id}")
        raise typer.Exit(1)

    typer.echo(f"Run ID:    {run.run_id}")
    typer.echo(f"Method:    {run.method}")
    typer.echo(f"Base model: {run.base_model}")
    typer.echo(f"Status:    {run.status}")
    typer.echo(f"Train loss: {run.final_train_loss}")
    if run.final_val_loss is not None:
        typer.echo(f"Val loss:  {run.final_val_loss}")
    typer.echo(f"Peak VRAM: {run.peak_vram_gb:.2f} GB")
    typer.echo(f"Train time: {run.train_time_minutes:.2f} min")
    typer.echo(f"Train set size: {run.train_set_size}")
    typer.echo(f"Checkpoint URI: {run.checkpoint_uri}")
    typer.echo(f"Hyperparams: {json.dumps(run.hyperparams, indent=2)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_validate(config) -> list[str]:
    """Run config validation without importing ML deps."""
    try:
        from app.training.config import validate_config

        return validate_config(config)
    except Exception as exc:  # noqa: BLE001
        return [f"Validation error: {exc}"]


def _print_training_result(result, typer_module) -> None:
    """Pretty-print a ``TrainingResult`` to the CLI."""
    typer_module.echo("")
    typer_module.echo(f"Run ID:    {result.run_id}")
    typer_module.echo(f"Method:    {result.method}")
    typer_module.echo(f"Base model: {result.base_model}")
    typer_module.echo(f"Status:    {result.status}")
    typer_module.echo(f"Train set: {result.train_set_size} examples")
    if result.final_train_loss is not None:
        typer_module.echo(f"Train loss: {result.final_train_loss:.4f}")
    else:
        typer_module.echo("Train loss: —")
    if result.final_val_loss is not None:
        typer_module.echo(f"Val loss:  {result.final_val_loss:.4f}")
    typer_module.echo(f"Peak VRAM: {result.peak_vram_gb:.2f} GB")
    typer_module.echo(f"Train time: {result.train_time_minutes:.2f} min")
    typer_module.echo(f"Checkpoint: {result.checkpoint_uri or '(not saved)'}")

    if result.train_loss_history:
        typer_module.echo(f"\nLoss history ({len(result.train_loss_history)} entries):")
        for i, loss in enumerate(result.train_loss_history[-10:]):  # last 10
            typer_module.echo(f"  step {i}: {loss:.4f}")

    # Persist if completed (not dry-run)
    if result.status == "completed":
        try:
            from app.training.experiment import persist_training_run

            persist_training_run(result)
            typer_module.echo(f"\nPersisted run {result.run_id} to PostgreSQL.")
        except Exception as exc:  # noqa: BLE001
            typer_module.echo(f"Warning: failed to persist run: {exc}", err=True)


if __name__ == "__main__":
    app()
