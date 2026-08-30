#!/usr/bin/env python
"""Run real GPU-based QLoRA training on Qwen 1.5B and save results.

Uses 4-bit NF4 quantization (QLoRA) via bitsandbytes to fit the model
into 8 GB VRAM on an RTX 4060 Laptop GPU. Config mirrors the design
constraints from the project README tech-stack table.

Usage::

    python scripts/run_gpu_training.py [--epochs 3] [--lr 2e-4]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from app.security.paths import validate_output_path, validate_path
from app.training.config import SFTConfig
from app.training.trainer_sft import run_sft

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STAGE3_DIR = Path("output/stage3")
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def _make_subset(src: Path, dst: Path, n: int) -> None:
    """Write at most ``n`` lines from *src* to *dst`` (UTF-8)."""
    safe_src = validate_path(src, allow_temp=True)
    safe_dst = validate_output_path(dst, allow_temp=True)
    safe_dst.parent.mkdir(parents=True, exist_ok=True)
    with open(safe_src, encoding="utf-8") as f, open(safe_dst, "w", encoding="utf-8") as out:
        for i, line in enumerate(f):
            if i >= n:
                break
            out.write(line)


def main():
    ap = argparse.ArgumentParser(description="Stage 5 — GPU QLoRA training")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap training samples (writes a subset JSONL). Default: all of Stage 3",
    )
    ap.add_argument("--train-jsonl", default="output/stage3/train.jsonl")
    ap.add_argument("--val-jsonl", default="output/stage3/val.jsonl")
    ap.add_argument("--output-dir", default="output/stage5/qwen_lora_gpu")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument(
        "--no-4bit",
        action="store_true",
        default=False,
        help="Disable 4-bit QLoRA (use full-precision LoRA). Required on CPU-only machines — "
        "4-bit quantization (bitsandbytes) needs CUDA.",
    )
    args = ap.parse_args()

    # --- Detect compute device ---
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
        logger.info("GPU: %s (%d MB VRAM)", gpu_name, vram_mb)
        use_4bit = not args.no_4bit
        if args.no_4bit:
            logger.warning("--no-4bit set: running LoRA without 4-bit quantization.")
    else:
        logger.warning("No CUDA GPU detected. Falling back to CPU training.")
        logger.warning(
            "QLoRA (4-bit) requires CUDA — switching to CPU-compatible LoRA (use_4bit=False)."
        )
        use_4bit = False

    gpu_name = gpu_name if has_cuda else "CPU"

    # --- Determine train/val paths (possibly truncated) ---
    train_jsonl = str(validate_path(args.train_jsonl, allow_temp=True))
    val_jsonl = str(validate_path(args.val_jsonl, allow_temp=True))
    if args.max_samples is not None:
        subset_train = Path(f"output/stage5/tmp_train_{args.max_samples}.jsonl")
        subset_val = Path(f"output/stage5/tmp_val_{args.max_samples}.jsonl")
        _make_subset(
            validate_path(args.train_jsonl, allow_temp=True), subset_train, args.max_samples
        )
        val_n = max(20, args.max_samples // 5)
        _make_subset(validate_path(args.val_jsonl, allow_temp=True), subset_val, val_n)
        train_jsonl = str(subset_train)
        val_jsonl = str(subset_val)
        print(f"Using subset: {args.max_samples} train samples (val: {val_n})")

    safe_output_dir = str(validate_output_path(args.output_dir, allow_temp=True))

    config = SFTConfig(
        base_model=args.base_model,
        output_dir=safe_output_dir,
        use_4bit=use_4bit,  # QLoRA (4-bit) on CUDA, LoRA (bfloat16) on CPU
        lora_r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,  # 1 sample per forward, fits 8 GB
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,  # effective batch = 8
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        run_name=f"qwen-1.5b-qlora-{gpu_name}",
    )

    mode = "QLoRA (4-bit GPU)" if config.use_4bit else "LoRA (bfloat16 CPU)"
    print(f"Starting SFT training ({mode}) on {gpu_name}...")
    print(f"  4-bit NF4: {config.use_4bit}")
    print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"  LR={config.learning_rate}, epochs={config.num_train_epochs}")
    print(
        f"  batch_size={config.per_device_train_batch_size}, "
        f"grad_accum={config.gradient_accumulation_steps} "
        f"(effective={config.per_device_train_batch_size * config.gradient_accumulation_steps})"
    )

    start = time.time()
    result = run_sft(config, dry_run=False)
    elapsed = time.time() - start

    print(f"\nTraining complete in {elapsed:.1f}s!")
    print(f"  Status: {result.status}")
    print(f"  Train loss: {result.final_train_loss:.4f}")
    print(f"  Val loss: {result.final_val_loss}")
    print(f"  Train time: {result.train_time_minutes:.2f} min")
    print(f"  Peak VRAM: {result.peak_vram_gb:.2f} GB")
    print(f"  Loss history entries: {len(result.train_loss_history)}")
    print(f"  Checkpoint: {result.checkpoint_uri}")

    # Save training result as JSON
    result_dict = {
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
        "run_name": result.run_name,
        "train_loss_history": result.train_loss_history,
    }
    out_path = Path("output/stage5/training_result.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result_dict, indent=2))
    print(f"Saved result to {out_path}")

    # Clean up subset files if we created them
    if args.max_samples is not None:
        for p in (  # NOSONAR - max_samples is argparse type=int, no path separators possible
            Path(f"output/stage5/tmp_train_{args.max_samples}.jsonl"),
            Path(f"output/stage5/tmp_val_{args.max_samples}.jsonl"),
        ):
            if p.exists():
                p.unlink()
                print(f"Cleaned up {p}")

    return result_dict


if __name__ == "__main__":
    main()
