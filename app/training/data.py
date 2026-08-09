"""Stage 5 — data loading from Stage 3 JSONL output.

Loads ``InstructionExample`` records from JSONL files (produced by
``python -m app.data.formatting.cli build``) and converts them into a format
suitable for training (either a HuggingFace ``datasets.Dataset`` or a plain
list of dicts).

Heavy ML imports (``datasets``, ``torch``) are **never** performed at module-
import time — they are imported lazily inside ``make_hf_dataset`` so that the
data module (and the CLI) work without ML dependencies installed. This is the
same lazy-import pattern used by ``QwenBackend`` (Stage 4) and
``TokenCounter`` (Stage 3).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Protocol

from app.schemas.dataset import InstructionExample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Injectable loaders — allow tests to pass pre-built examples without touching
# the filesystem, mirroring the injectable-backend pattern from Stages 2-4.
# ---------------------------------------------------------------------------


class DataLoadable(Protocol):
    """Protocol for anything that can load InstructionExample lists from a path."""

    def load(self, path: str) -> list[InstructionExample]: ...


class JsonlDataLoader:
    """Default loader: reads JSONL files produced by Stage 3."""

    def load(self, path: str) -> list[InstructionExample]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        examples: list[InstructionExample] = []
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    examples.append(InstructionExample(**payload))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping invalid line %d in %s: %s", line_num, path, exc)
        logger.info("Loaded %d examples from %s", len(examples), path)
        return examples


def load_examples(
    path: str,
    loader: DataLoadable | None = None,
) -> list[InstructionExample]:
    """Load ``InstructionExample`` records from a JSONL file.

    Parameters
    ----------
    path:
        Path to a JSONL file (typically Stage 3's ``train.jsonl``).
    loader:
        Injectable loader for testing. Defaults to ``JsonlDataLoader``.
    """
    loader = loader or JsonlDataLoader()
    return loader.load(path)


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Summary statistics for a loaded dataset."""

    n_examples: int
    cwe_distribution: dict[str, int]
    severity_distribution: dict[str, int]
    avg_token_estimate: float

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "cwe_distribution": self.cwe_distribution,
            "severity_distribution": self.severity_distribution,
            "avg_token_estimate": round(self.avg_token_estimate, 2),
        }


def compute_stats(examples: list[InstructionExample]) -> DatasetStats:
    """Compute summary statistics for a list of ``InstructionExample`` records."""
    cwe_dist: dict[str, int] = {}
    sev_dist: dict[str, int] = {}
    total_tokens = 0

    for ex in examples:
        cwe_dist[ex.target_cwe] = cwe_dist.get(ex.target_cwe, 0) + 1
        sev_dist[ex.target_severity] = sev_dist.get(ex.target_severity, 0) + 1
        total_tokens += ex.token_count_estimate

    n = len(examples)
    avg_tokens = total_tokens / n if n > 0 else 0.0

    return DatasetStats(
        n_examples=n,
        cwe_distribution=cwe_dist,
        severity_distribution=sev_dist,
        avg_token_estimate=avg_tokens,
    )


def examples_to_dict_list(
    examples: list[InstructionExample],
) -> list[dict]:
    """Convert ``InstructionExample`` records to a list of dicts.

    The dict format (``{"prompt": ..., "completion": ...}``) is the standard
    input format for HuggingFace ``datasets`` and most SFT trainers. The
    ``completion`` field contains the JSON target string that the model should
    generate.
    """
    rows: list[dict] = []
    for ex in examples:
        # The "completion" is the model's expected output — a JSON object
        # with cwe_id, severity, explanation, and patch_diff.
        completion_obj = {
            "cwe_id": ex.target_cwe,
            "severity": ex.target_severity,
            "explanation": ex.target_explanation,
            "patch_diff": ex.target_patch_diff,
        }
        rows.append(
            {
                "prompt": ex.prompt,
                "completion": json.dumps(completion_obj, ensure_ascii=False),
                "cwe_id": ex.target_cwe,
            }
        )
    return rows


def make_hf_dataset(
    examples: list[InstructionExample],
) -> object:
    """Convert ``InstructionExample`` records to a HuggingFace ``datasets.Dataset``.

    Lazy-imports ``datasets`` so the module works without it installed.

    The resulting dataset has columns: ``prompt`` (str), ``completion`` (str),
    ``cwe_id`` (str) — ready to be tokenised and fed to a Trainer.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is not installed. Run `pip install -e '.[ml]'` to use make_hf_dataset."
        ) from exc

    rows = examples_to_dict_list(examples)
    return Dataset.from_dict(
        {
            "prompt": [r["prompt"] for r in rows],
            "completion": [r["completion"] for r in rows],
            "cwe_id": [r["cwe_id"] for r in rows],
        }
    )


def make_hf_dataset_pair(
    train_examples: list[InstructionExample],
    val_examples: list[InstructionExample],
) -> dict[str, object]:
    """Create a ``DatasetDict``-like dict with ``train`` and ``validation`` splits."""
    return {
        "train": make_hf_dataset(train_examples),
        "validation": make_hf_dataset(val_examples),
    }


def load_stage3_dataset(
    train_path: str,
    val_path: str = "",
    loader: DataLoadable | None = None,
) -> dict[str, list[InstructionExample]]:
    """Load train (and optionally val) datasets from Stage 3 JSONL files.

    Parameters
    ----------
    train_path:
        Path to ``train.jsonl`` from Stage 3.
    val_path:
        Optional path to ``val.jsonl``. If empty, ``{"train": [...], "val": []}``
        is returned with an empty val list.
    loader:
        Injectable loader for testing.
    """
    train = load_examples(train_path, loader=loader)
    val = load_examples(val_path, loader=loader) if val_path else []

    return {"train": train, "val": val}
