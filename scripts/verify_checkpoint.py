"""Verify a Stage 5 LoRA checkpoint is complete before it's used downstream.

This exists because Stage 7 previously produced a silently-wrong regression
report: the checkpoint directory had ``adapter_config.json`` but no
``adapter_model.safetensors`` / ``.bin``, and ``QwenBackend`` fell back to the
base model without raising — so "tuned model" results were actually the base
model compared against itself (forgetting_delta == 0.0 for the wrong reason).

Run this before Stage 6/7/8 against any LoRA checkpoint::

    python scripts/verify_checkpoint.py ./output/stage5/qwen_lora_gpu/final_checkpoint

Exit code is non-zero if the checkpoint is incomplete, so it's safe to use
as a CI/pipeline gate (``&&`` it before the next stage's script).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.security.paths import validate_path  # noqa: E402

ADAPTER_WEIGHT_NAMES = ("adapter_model.safetensors", "adapter_model.bin")


def _sha256_of(path: Path, chunk_size: int = 1024 * 1024) -> str:
    safe_path = validate_path(path, allow_temp=True)
    h = hashlib.sha256()
    with open(safe_path, "rb") as f:  # NOSONAR — path validated above
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(checkpoint_dir: str) -> dict:
    """Check that a checkpoint directory contains real, loadable weights.

    Returns a fingerprint dict on success. Raises ``FileNotFoundError`` (no
    such directory) or ``RuntimeError`` (directory exists but weights are
    missing/empty) on failure — callers get a clear reason rather than a
    silent False.
    """
    ckpt = validate_path(checkpoint_dir, allow_temp=True)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt}")

    has_lora_config = (ckpt / "adapter_config.json").exists()

    if not has_lora_config:
        # Not a LoRA checkpoint — check it looks like a full model instead
        # (must have at least one weights file of some kind).
        full_model_weight_globs = ["*.safetensors", "*.bin", "pytorch_model*.bin"]
        found = [p for pat in full_model_weight_globs for p in ckpt.glob(pat)]
        if not found:
            raise RuntimeError(
                f"{ckpt} has neither adapter_config.json (LoRA) nor any "
                f"model weight files (full model). This directory looks "
                f"like a tokenizer-only export, not a usable checkpoint."
            )
        return {
            "checkpoint_dir": str(ckpt.resolve()),
            "checkpoint_type": "full_model",
            "weight_files": [str(p.name) for p in found],
            "total_size_bytes": sum(p.stat().st_size for p in found),
        }

    weight_file = next(
        (ckpt / name for name in ADAPTER_WEIGHT_NAMES if (ckpt / name).exists()),
        None,
    )
    if weight_file is None:
        raise RuntimeError(
            f"{ckpt} has adapter_config.json but no adapter_model.safetensors "
            f"or adapter_model.bin. This is exactly the incomplete-checkpoint "
            f"state that previously caused Stage 7 to silently benchmark the "
            f"base model against itself. Re-run Stage 5 training (or check "
            f"whether the weight file was stripped by a cleanup step / "
            f".gitignore-aware export) before proceeding."
        )

    size_bytes = weight_file.stat().st_size
    if size_bytes == 0:
        raise RuntimeError(f"{weight_file} exists but is empty (0 bytes).")

    return {
        "checkpoint_dir": str(ckpt.resolve()),
        "checkpoint_type": "lora",
        "adapter_weight_file": weight_file.name,
        "adapter_size_bytes": size_bytes,
        "adapter_sha256": _sha256_of(weight_file),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir", help="Path to the checkpoint directory to verify")
    ap.add_argument(
        "--json", action="store_true", help="Print the fingerprint as JSON (for piping)"
    )
    args = ap.parse_args()

    try:
        fingerprint = verify_checkpoint(args.checkpoint_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(fingerprint, indent=2))
    else:
        print("[OK] Checkpoint verified:")
        for k, v in fingerprint.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
