"""Stage 3 orchestration: load split samples → build instruction examples → write JSONL.

This module ties together the Stage 2 output (deduped, leakage-safe-split
``VulnSample`` records in Postgres + MinIO) with the Stage 3 builder
(prompt template, token budget). It produces JSONL files — one per split —
that are ready to be consumed by the Stage 5 training scripts.

The pipeline mirrors Stage 2's storage access pattern (load from Postgres +
MinIO via the same `VulnSampleRow` / `get_json` helpers), so it is a
drop-in consumer of whatever Stage 2 persisted. It also supports loading
from a local HF datasets directory (produced by Stage 2's `export` command)
for environments that don't have a live Postgres/MinIO stack.

Output layout (default):
    output_dir/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl

Each line is a JSON object matching the ``InstructionExample`` schema.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from app.data.cleaning.hf_dataset import load_dataset_locally
from app.data.cleaning.pipeline import load_samples_from_storage
from app.data.formatting.builder import BuildResult, build_examples
from app.data.formatting.tokenizer import DEFAULT_MAX_TOKENS, TokenCounter
from app.schemas.dataset import InstructionExample
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)

# Splits that get JSONL output files. ``gold_eval`` is intentionally excluded —
# it is a separate, manually-curated set used only for eval (Stage 6+), not
# for training.
OUTPUT_SPLITS: tuple[str, ...] = ("train", "val", "test")


@dataclass
class Stage3Result:
    """Full output of the Stage 3 pipeline for one run."""

    total_samples_loaded: int
    examples_by_split: dict[str, list[InstructionExample]]
    dropped_by_split: dict[str, list[tuple[str, int]]]
    token_counter: TokenCounter | None
    output_dir: str
    max_tokens: int

    @property
    def total_examples(self) -> int:
        return sum(len(ex) for ex in self.examples_by_split.values())

    @property
    def total_dropped(self) -> int:
        return sum(len(d) for d in self.dropped_by_split.values())

    def counts(self) -> dict[str, dict[str, int]]:
        """Per-split counts of kept examples and dropped samples."""
        return {
            split: {
                "kept": len(self.examples_by_split.get(split, [])),
                "dropped": len(self.dropped_by_split.get(split, [])),
            }
            for split in OUTPUT_SPLITS
        }


def write_jsonl(examples: list[InstructionExample], path: str) -> int:
    """Write a list of ``InstructionExample`` records to a JSONL file.

    Returns the number of lines written.
    """
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
    return len(examples)


def _filter_by_split(samples: list[VulnSample], split_name: str) -> list[VulnSample]:
    """Return only samples whose ``.split`` matches ``split_name``."""
    return [s for s in samples if s.split == split_name]


def run_stage3(
    *,
    token_counter: TokenCounter | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    output_dir: str = "./output/stage3",
    samples: list[VulnSample] | None = None,
) -> Stage3Result:
    """Run the complete Stage 3 pipeline.

    Parameters
    ----------
    token_counter:
        Optional injectable ``TokenCounter``. Used to enforce the token budget.
        If None, a default ``TokenCounter`` (Qwen tokenizer with heuristic
        fallback) is created.
    max_tokens:
        Maximum prompt + target tokens per example. Samples exceeding this
        are dropped from the output.
    output_dir:
        Directory where JSONL files are written. Created if it doesn't exist.
    samples:
        Optional pre-loaded list of ``VulnSample`` records (e.g. loaded from
        an HF datasets directory). If None, samples are loaded from
        Postgres + MinIO via ``load_samples_from_storage()`` (the Stage 2
        persistence path).

    Returns
    -------
    ``Stage3Result`` with per-split example lists, drop records, and the
    output directory path.
    """
    counter = token_counter or TokenCounter()

    # Step 1: Load samples
    if samples is not None:
        all_samples = samples
        logger.info("Stage 3: using %d pre-loaded samples", len(all_samples))
    else:
        all_samples = load_samples_from_storage()
        logger.info("Stage 3: loaded %d samples from Postgres/MinIO", len(all_samples))

    if not all_samples:
        raise RuntimeError(
            "No samples found. Run Stage 1 (collect) and Stage 2 (clean) first:\n"
            "  python -m app.data.collectors.cli collect --db-path ./CVEfixes.db\n"
            "  python -m app.data.cleaning.cli clean"
        )

    # Step 2 + 3: Build instruction examples per split, enforcing token budget
    examples_by_split: dict[str, list[InstructionExample]] = {}
    dropped_by_split: dict[str, list[tuple[str, int]]] = {}

    for split_name in OUTPUT_SPLITS:
        split_samples = _filter_by_split(all_samples, split_name)
        logger.info("Stage 3: building %d examples for split=%s", len(split_samples), split_name)

        result: BuildResult = build_examples(
            split_samples,
            token_counter=counter,
            max_tokens=max_tokens,
        )

        examples_by_split[split_name] = result.examples
        dropped_by_split[split_name] = result.dropped

    # Step 4: Write JSONL
    os.makedirs(output_dir, exist_ok=True)
    for split_name in OUTPUT_SPLITS:
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        n_written = write_jsonl(examples_by_split[split_name], path)
        logger.info("Stage 3: wrote %d examples to %s", n_written, path)

    # Write a manifest with metadata
    manifest = {
        "max_tokens": max_tokens,
        "token_counter_model": counter.model_name if hasattr(counter, "model_name") else "unknown",
        "splits": {
            split_name: {
                "path": os.path.join(output_dir, f"{split_name}.jsonl"),
                "n_examples": len(examples_by_split[split_name]),
                "n_dropped": len(dropped_by_split[split_name]),
            }
            for split_name in OUTPUT_SPLITS
        },
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Stage 3: wrote manifest to %s", manifest_path)

    return Stage3Result(
        total_samples_loaded=len(all_samples),
        examples_by_split=examples_by_split,
        dropped_by_split=dropped_by_split,
        token_counter=counter,
        output_dir=output_dir,
        max_tokens=max_tokens,
    )


def load_from_hf_dataset(
    path: str,
    splits: tuple[str, ...] = OUTPUT_SPLITS,
) -> list[VulnSample]:
    """Load VulnSample records from a local HF datasets directory.

    This is the inverse of Stage 2's ``export`` command. Each row in the
    HF dataset is converted back to a ``VulnSample``.

    Parameters
    ----------
    path:
        Path to a local HF datasets directory (created by Stage 2's export
        ``save_to_disk``).
    splits:
        Which splits to load. Defaults to train/val/test.
    """
    from app.schemas.vuln import VulnSample

    dataset_dict = load_dataset_locally(path)
    samples: list[VulnSample] = []
    for split_name in splits:
        if split_name in dataset_dict:
            for row in dataset_dict[split_name]:
                field_map = {k: v for k, v in row.items() if k in VulnSample.model_fields}
                samples.append(VulnSample(**field_map))
    return samples
