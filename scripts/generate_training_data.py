"""Generate Stage 3 training data from the gold set.

The gold set (``eval/gold_set/gold.jsonl``) contains 59 hand-curated
``VulnSample`` records, but they all carry ``split="gold_eval"``.
``run_stage3`` only picks up ``train``/``val``/``test`` splits, so this
script performs that split explicitly (80 / 10 / 10) and builds
``InstructionExample`` records — the format Stage 5 expects.

Usage::

    python scripts/generate_training_data.py
        --input  eval/gold_set/gold.jsonl
        --output-dir  ./output/stage3
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

from app.security.paths import validate_output_path, validate_path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = "eval/gold_set/gold.jsonl"
DEFAULT_OUTPUT_DIR = "./output/stage3"
DEFAULT_MAX_TOKENS = 4096
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
# TEST_RATIO = 0.1 (remainder)
SEED = 42


def load_gold_samples(path: str) -> list:
    """Load gold-set JSONL lines as ``VulnSample`` objects."""
    from app.schemas.vuln import VulnSample

    samples: list[VulnSample] = []
    safe_path = validate_path(path, allow_temp=True)
    with open(safe_path, encoding="utf-8") as f:  # NOSONAR
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            # Gold samples have the same keys as VulnSample; just construct.
            # Strip any extra keys that aren't in the schema.
            allowed = set(VulnSample.model_fields.keys())
            filtered = {k: v for k, v in raw.items() if k in allowed}
            samples.append(VulnSample(**filtered))
    return samples


def split_samples(
    samples: list,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SEED,
) -> tuple[list, list, list]:
    """Deterministically split samples into train / val / test."""
    rng = random.Random(seed)  # nosec B311
    indexed = list(enumerate(samples))
    rng.shuffle(indexed)
    shuffled = [s for _, s in indexed]

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


def write_jsonl(examples: list, path: str) -> int:
    """Write InstructionExample records to JSONL."""
    safe_path = validate_output_path(path, allow_temp=True)
    with open(safe_path, "w", encoding="utf-8") as f:  # NOSONAR
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
    return len(examples)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate Stage 3 data from gold set")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Path to gold.jsonl")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    from app.data.formatting.builder import BuildResult, build_examples
    from app.data.formatting.tokenizer import TokenCounter

    print(f"Loading gold samples from {args.input} ...")
    samples = load_gold_samples(args.input)
    print(f"  Loaded {len(samples)} samples")

    train, val, test = split_samples(samples, seed=args.seed)
    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")

    counter = TokenCounter()
    max_tokens = args.max_tokens

    results: dict[str, BuildResult] = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        if not split:
            print(f"  [{name}] 0 samples — skipping")
            results[name] = BuildResult(examples=[], dropped=[])
            continue
        r = build_examples(split, token_counter=counter, max_tokens=max_tokens)
        results[name] = r
        print(f"  [{name}] {len(r.examples)} kept, {len(r.dropped)} dropped")

    out = validate_output_path(args.output_dir, allow_temp=True)
    out.mkdir(parents=True, exist_ok=True)  # NOSONAR

    total_examples = 0
    total_dropped = 0
    manifest_splits: dict = {}
    for name in ("train", "val", "test"):
        r = results[name]
        path = out / f"{name}.jsonl"
        n = write_jsonl(r.examples, str(path))
        total_examples += n
        total_dropped += len(r.dropped)
        manifest_splits[name] = {
            "path": str(path),
            "n_examples": n,
            "n_dropped": len(r.dropped),
        }

    manifest = {
        "max_tokens": max_tokens,
        "token_counter_model": counter.model_name if hasattr(counter, "model_name") else "default",
        "splits": manifest_splits,
        "source": "eval/gold_set/gold.jsonl",
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # NOSONAR

    print()
    print(f"Total examples: {total_examples}  Dropped: {total_dropped}")
    print(f"Output written to: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
