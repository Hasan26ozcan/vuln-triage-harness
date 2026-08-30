"""Run real CPU-compatible LoRA training on Qwen 1.5B and save results.

Usage::

    python scripts/run_cpu_training.py [--max-samples N] [--epochs E]
        [--base-model MODEL] [--run-name NAME]

When ``--max-samples`` is given the script writes a subset of Stage 3
train/val JSONL (first N lines) to ``output/stage5/tmp_train_N.jsonl``
so the trainer sees exactly N examples without touching the original
Stage 3 artifacts.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from app.security.paths import validate_output_path, validate_path
from app.training.config import SFTConfig
from app.training.trainer_sft import run_sft

logging.basicConfig(level=logging.INFO)
torch.set_num_threads(8)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
STAGE3_DIR = Path("output/stage3")


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
    ap = argparse.ArgumentParser(description="CPU-compatible LoRA training (Qwen 1.5B)")
    ap.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap training samples (writes a subset JSONL). Default: all of Stage 3",
    )
    ap.add_argument("--epochs", type=int, default=2, help="Number of training epochs (default: 2)")
    ap.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model to fine-tune (default: {DEFAULT_BASE_MODEL})",
    )
    ap.add_argument(
        "--run-name",
        default="qwen-1.5b-lora-cpu",
        help="Human-readable run name for the checkpoint",
    )
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank (default: 8)")
    args = ap.parse_args()

    # --- Determine train/val paths (possibly truncated) ---
    train_jsonl = str(STAGE3_DIR / "train.jsonl")
    val_jsonl = str(STAGE3_DIR / "val.jsonl")

    if args.max_samples is not None:
        subset_train = Path(f"output/stage5/tmp_train_{args.max_samples}.jsonl")
        subset_val = Path(f"output/stage5/tmp_val_{args.max_samples}.jsonl")
        _make_subset(STAGE3_DIR / "train.jsonl", subset_train, args.max_samples)
        # Cap val subset to 20 % of train, min 20
        val_n = max(20, args.max_samples // 5)
        _make_subset(STAGE3_DIR / "val.jsonl", subset_val, val_n)
        train_jsonl = str(subset_train)
        val_jsonl = str(subset_val)
        print(f"Using subset: {args.max_samples} train samples (val: {val_n})")

    config = SFTConfig(
        base_model=args.base_model,
        output_dir="./output/stage5/qwen_lora_cpu",
        use_4bit=False,  # CPU-compatible (no 4-bit quantization)
        lora_r=args.lora_r,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        run_name=args.run_name,
    )

    print("Starting real CPU-compatible training...")
    start = time.time()
    result = run_sft(config, dry_run=False)
    elapsed = time.time() - start

    print(f"Training complete in {elapsed:.1f}s!")
    print(f"  Status: {result.status}")
    print(f"  Train loss: {result.final_train_loss:.4f}")
    print(f"  Val loss: {result.final_val_loss}")
    print(f"  Train time: {result.train_time_minutes:.2f} min")
    print(f"  Train set size: {result.train_set_size}")
    print(f"  Loss history length: {len(result.train_loss_history)}")
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
    out_path = validate_output_path("output/stage5/training_result.json", allow_temp=True)
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
