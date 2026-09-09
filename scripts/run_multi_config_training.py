#!/usr/bin/env python
"""Run multiple LoRA configuration experiments for Stage 5.

Tests different LoRA ranks, alphas, learning rates, and early stopping
patiences to find the optimal configuration. Based on the previous run
(r=8, alpha=16, lr=2e-4) which stopped early at epoch 3 with patience=3.

Configurations tested (from app/training/config.py presets):
  1. lora_r16_alpha32_lr5e5     — r=16, alpha=32, lr=5e-5, patience=7
  2. lora_r32_alpha64_lr5e5     — r=32, alpha=64, lr=5e-5, patience=10
  3. lora_r32_alpha64_lr3e5_pat10_e75 — r=32, alpha=64, lr=3e-5, 75 epochs, patience=10
  4. lora_r16_alpha32_lr5e5_pat5_e50  — r=16, alpha=32, lr=5e-5, patience=5

Usage::

    python scripts/run_multi_config_training.py --all
    python scripts/run_multi_config_training.py --dry-run

Results are saved to output/stage5/multi_config_results.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.security.paths import validate_output_path, validate_path
from app.training.config import (
    LORA_PRESETS_EXTENDED,
    LoRAPreset,
    get_preset,
    preset_to_sft_config,
)
from app.training.trainer_sft import run_sft

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_STAGE3_TRAIN = "output/stage3/train.jsonl"
DEFAULT_STAGE3_VAL = "output/stage3/val.jsonl"
DEFAULT_OUTPUT_DIR = "output/stage5/multi_config"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 5 — Multi-config LoRA training experiments")
    ap.add_argument(
        "--presets",
        type=str,
        default=None,
        help=(
            "Comma-separated preset names to run "
            "(e.g. 'lora_r16_alpha32_lr5e5,lora_r32_alpha64_lr5e5'). "
            "Default: all presets from LORA_PRESETS_EXTENDED"
        ),
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Run all presets (default behavior).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate steps/VRAM without actual training.",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_train_epochs for all presets.",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning_rate for all presets.",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Override early_stopping_patience for all presets.",
    )
    ap.add_argument(
        "--train-jsonl",
        type=str,
        default=DEFAULT_STAGE3_TRAIN,
        help="Path to Stage 3 train.jsonl.",
    )
    ap.add_argument(
        "--val-jsonl",
        type=str,
        default=DEFAULT_STAGE3_VAL,
        help="Path to Stage 3 val.jsonl.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Base output directory.",
    )
    ap.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Base model to fine-tune.",
    )
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization (use CPU-compatible LoRA).",
    )
    ap.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Enable verbose logging.",
    )
    return ap.parse_args()


def _get_presets_to_run(args: argparse.Namespace) -> list[LoRAPreset]:
    """Determine which presets to run based on CLI arguments."""
    if args.presets:
        names = [n.strip() for n in args.presets.split(",") if n.strip()]
        presets = []
        for name in names:
            preset = get_preset(name)
            if preset is None:
                raise ValueError(
                    f"Unknown preset: {name!r}. Run with --list-presets to see options."
                )
            presets.append(preset)
        return presets

    if not args.all and not args.presets:
        # Default: run the first two (most impactful) presets
        preset1 = get_preset("lora_r16_alpha32_lr5e5")
        preset2 = get_preset("lora_r32_alpha64_lr5e5")
        if preset1 is None or preset2 is None:
            raise RuntimeError("Required presets not found")
        return [preset1, preset2]

    return list(LORA_PRESETS_EXTENDED)


def _apply_overrides(preset: LoRAPreset, args: argparse.Namespace) -> LoRAPreset:
    """Apply CLI overrides to a preset using dataclasses.replace."""
    kwargs = {}
    if args.epochs is not None:
        kwargs["num_train_epochs"] = args.epochs
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr
    if args.patience is not None:
        kwargs["early_stopping_patience"] = args.patience
    return replace(preset, **kwargs) if kwargs else preset


def run_experiment(
    preset: LoRAPreset,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run a single LoRA preset experiment and return results."""
    from app.training.experiment import generate_run_id

    preset = _apply_overrides(preset, args)
    use_4bit = not args.no_4bit

    # Build SFTConfig from preset
    config = preset_to_sft_config(
        preset=preset,
        base_model=args.base_model,
        output_dir=args.output_dir,
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
    )
    # Override use_4bit based on CLI
    config = config.__class__(
        **{**dict(config.__dict__), "use_4bit": use_4bit}
    )

    run_name = f"{preset.name}_{args.base_model.split('/')[-1]}"
    logger.info(
        "Starting experiment: %s | r=%d, alpha=%d, lr=%.1e, epochs=%d, patience=%d",
        preset.name,
        preset.lora_r,
        preset.lora_alpha,
        preset.learning_rate,
        preset.num_train_epochs,
        preset.early_stopping_patience,
    )
    logger.info("  Base model: %s", args.base_model)
    logger.info("  Use 4-bit: %s", use_4bit)
    logger.info("  Train: %s, Val: %s", args.train_jsonl, args.val_jsonl)

    start = time.time()
    try:
        result = run_sft(config, dry_run=args.dry_run)
        elapsed = time.time() - start

        result_dict: dict[str, Any] = {
            "preset_name": preset.name,
            "preset_description": preset.description,
            "run_id": result.run_id,
            "method": result.method,
            "base_model": result.base_model,
            "hyperparams": result.hyperparams,
            "train_set_size": result.train_set_size,
            "train_time_minutes": result.train_time_minutes,
            "peak_vram_gb": result.peak_vram_gb,
            "final_train_loss": result.final_train_loss,
            "final_val_loss": result.final_val_loss,
            "checkpoint_uri": result.checkpoint_uri,
            "status": result.status,
            "run_name": run_name,
            "stopped_early": result.hyperparams.get("stopped_early", False),
            "elapsed_seconds": round(elapsed, 2),
            "loss_history": result.train_loss_history,
        }

        if not args.dry_run and result.train_loss_history:
            # Identify epoch where best val_loss occurred
            val_losses = result.train_loss_history
            best_epoch = len(val_losses)  # approximate
            logger.info(
                "  Final: train_loss=%.4f, val_loss=%.4f",
                result.final_train_loss,
                result.final_val_loss,
            )
            if result.final_val_loss is not None:
                logger.info(
                    "  Best val_loss: %.4f at epoch ~%d",
                    result.final_val_loss,
                    best_epoch,
                )

        logger.info("  Completed in %.1fs", elapsed)

    except Exception as exc:
        elapsed = time.time() - start
        logger.exception("Experiment %s failed: %s", preset.name, exc)
        result_dict = {
            "preset_name": preset.name,
            "run_id": generate_run_id(preset.name),
            "status": "failed",
            "error": str(exc),
            "elapsed_seconds": round(elapsed, 2),
        }

    return result_dict


def main() -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    # Print available presets
    print("\n" + "=" * 70)
    print("Stage 5 — Multi-Config LoRA Training Experiments")
    print("=" * 70)
    print(f"\nBase model: {args.base_model}")
    print(f"Train set: {args.train_jsonl}")
    print(f"Val set: {args.val_jsonl}")
    print(f"Dry run: {args.dry_run}")
    print(f"Use 4-bit: {not args.no_4bit}")
    if args.epochs:
        print(f"Epoch override: {args.epochs}")
    if args.lr:
        print(f"LR override: {args.lr}")
    if args.patience:
        print(f"Patience override: {args.patience}")

    # Validate user-provided paths before use (prevent path traversal)
    safe_output_dir = validate_output_path(args.output_dir, allow_temp=True)
    safe_train_jsonl = validate_path(args.train_jsonl, allow_temp=True)
    safe_val_jsonl = validate_path(args.val_jsonl, allow_temp=True)
    args.output_dir = str(safe_output_dir)
    args.train_jsonl = str(safe_train_jsonl)
    args.val_jsonl = str(safe_val_jsonl)

    presets_to_run = _get_presets_to_run(args)

    print(f"\nPresets to run ({len(presets_to_run)}):")
    for p in presets_to_run:
        print(
            f"  - {p.name}: r={p.lora_r}, alpha={p.lora_alpha}, "
            f"lr={p.learning_rate:.1e}, epochs={p.num_train_epochs}, "
            f"patience={p.early_stopping_patience}"
        )
        print(f"    {p.description}")

    # Run all experiments
    results: list[dict[str, Any]] = []
    for i, preset in enumerate(presets_to_run, 1):
        _print_experiment_header(presets_to_run, i, preset)
        result_dict = run_experiment(preset, args)
        results.append(result_dict)
        _print_experiment_result(result_dict)

    # Save summary
    summary = _build_summary_dict(args, results)
    _save_summary(summary)

    # Print summary
    _print_summary_table(results)
    _print_best_result(summary)

    return summary


def _build_summary_dict(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the summary dict for all experiment results."""
    completed = [r for r in results if r["status"] == "completed"]
    best = None
    if completed:
        best = min(completed, key=lambda r: r.get("final_val_loss", float("inf")))

    avg_train_loss = (
        sum(r["final_train_loss"] for r in completed) / len(completed) if completed else None
    )
    avg_val_loss = (
        sum(r["final_val_loss"] for r in completed if r.get("final_val_loss") is not None)
        / len([r for r in completed if r.get("final_val_loss") is not None])
        if completed and any(r.get("final_val_loss") is not None for r in completed)
        else None
    )

    return {
        "experiment_name": "multi_config_lora_stage5",
        "base_model": args.base_model,
        "total_runs": len(results),
        "completed_runs": sum(1 for r in results if r["status"] == "completed"),
        "dry_run": args.dry_run,
        "results": results,
        "summary": {
            "best_by_val_loss": {
                "preset_name": best["preset_name"],
                "final_train_loss": best["final_train_loss"],
                "final_val_loss": best["final_val_loss"],
                "checkpoint_uri": best["checkpoint_uri"],
                "stopped_early": best.get("stopped_early", False),
            }
            if best
            else None,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "total_completed": len(completed),
            "failed_runs": [r["preset_name"] for r in results if r["status"] == "failed"],
        },
    }


def _save_summary(summary: dict[str, Any]) -> None:
    """Write the summary JSON to disk after validating the output path."""
    output_path = validate_output_path(
        str(Path(summary["output_dir"]) / "multi_config_results.json"),
        allow_temp=True,
    )
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_path}")


def _print_experiment_header(presets_to_run: list[LoRAPreset], i: int, preset: LoRAPreset) -> None:
    """Print the header for a single experiment."""
    print(f"\n{'=' * 50}")
    print(f"Experiment {i}/{len(presets_to_run)}: {preset.name}")
    print(f"{'=' * 50}")


def _print_experiment_result(result_dict: dict[str, Any]) -> None:
    """Print the result of a single experiment."""
    status = result_dict["status"]
    if status == "completed":
        print(
            f"  Status: {status} | "
            f"Train loss: {result_dict.get('final_train_loss', 'N/A'):.4f} | "
            f"Val loss: {result_dict.get('final_val_loss', 'N/A'):.4f}"
        )
    else:
        print(f"  Status: {status} | Error: {result_dict.get('error', 'Unknown')}")


def _print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print the formatted summary results table."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"Total runs: {len(results)}")
    completed = sum(1 for r in results if r["status"] == "completed")
    print(f"Completed: {completed}")
    print(
        f"\n{'Preset':<35s} {'Status':<12s} "
        f"{'Train Loss':>12s} {'Val Loss':>12s} "
        f"{'VRAM':>8s} {'Time':>8s}"
    )
    print("-" * 85)
    for r in results:
        preset_name = r.get("preset_name", "?")[:33]
        status = r["status"]
        tl = f"{r.get('final_train_loss', 'N/A'):.4f}" if r.get("final_train_loss") else "N/A"
        vl = f"{r.get('final_val_loss', 'N/A'):.4f}" if r.get("final_val_loss") else "N/A"
        vg = f"{r.get('peak_vram_gb', 0):.2f}" if r.get("peak_vram_gb") else "—"
        tm = f"{r.get('train_time_minutes', 0):.1f}" if r.get("train_time_minutes") else "—"
        print(f"  {preset_name:<33s} {status:<12s} {tl:>12s} {vl:>12s} {vg:>8s} {tm:>8s}")


def _print_best_result(summary: dict[str, Any]) -> None:
    """Print the best result by val_loss if available."""
    best = summary["summary"].get("best_by_val_loss")
    if best:
        print(
            f"\nBest by val_loss: {best['preset_name']} "
            f"(val_loss={best['final_val_loss']:.4f}, train_loss={best['final_train_loss']:.4f})"
        )


if __name__ == "__main__":
    main()
