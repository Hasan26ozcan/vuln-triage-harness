"""HuggingFace ``datasets`` library integration for Stage 2.

This module bridges between the project's Pydantic ``VulnSample`` schema and
the HuggingFace ``datasets.Dataset`` / ``DatasetDict`` format. It enables two
workflows:

1. **Export to HF Hub**: after cleaning/splitting, push the train/val/test
   splits as a single ``DatasetDict`` to the HuggingFace Hub under a
   configurable repo name (e.g. ``vuln-triage/vuln-triagedataset``).

2. **Load from HF Hub**: re-load a previously pushed dataset dict back into
   ``VulnSample`` objects for training or evaluation — useful for Stage 3
   (instruction formatting) and Stage 5 (training matrix).

The actual dataset schema follows the project's Pydantic contracts — we
export `repo_name`, `cwe_id`, `severity`, `language`, `vulnerable_code`,
`fixed_code`, `static_findings`, `description`, and `split`.

Note: this requires ``pip install -e ".[ml]"`` (which includes ``datasets``)
plus a HuggingFace token (``HF_TOKEN`` env var or ``huggingface-cli login``)
for Hub push/pull. For offline use, datasets can be saved/loaded from local
disk via ``save_to_disk`` / ``load_from_disk``.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from datasets import DatasetDict  # noqa: F401 — only for type hints

    from app.schemas.vuln import VulnSample  # noqa: F401


# Columns exported to HuggingFace datasets format. Must match the Pydantic
# VulnSample schema fields (minus `id` which is the row key, and `source`
# which is Stage-1 metadata not needed for training).
HF_COLUMNS: list[str] = [
    "id",
    "repo_name",
    "commit_sha",
    "cve_id",
    "cwe_id",
    "severity",
    "language",
    "vulnerable_code",
    "fixed_code",
    "static_findings",
    "description",
    "split",
]


def _check_datasets_available() -> None:
    """Raise a clear error if `datasets` is not installed."""
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` library is not installed. Run "
            "`pip install datasets` or `pip install -e '.[ml]'`."
        ) from exc


def samples_to_hf_dataset(samples: list) -> DatasetDict:
    """Convert a flat list of VulnSample objects into an HF DatasetDict with
    train/val/test splits, inferring the split from each sample's `.split`
    field.

    Samples without a `.split` field are dropped (they should have been
    assigned one by Stage 2's split module).
    """
    from datasets import Dataset, DatasetDict

    _check_datasets_available()

    # Group by split
    by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    for s in samples:
        if s.split in by_split:
            by_split[s.split].append(s)
        else:
            logger.warning(
                "Sample %s has split=%r — skipping (expected train/val/test)",
                s.id,
                s.split,
            )

    # Build dict-of-lists for each split
    split_datasets: dict[str, Dataset] = {}
    for split_name, split_samples in by_split.items():
        if not split_samples:
            logger.warning("Split %s is empty — creating an empty placeholder", split_name)
            split_datasets[split_name] = Dataset.from_dict({col: [] for col in HF_COLUMNS})
            continue

        data: dict[str, list] = {col: [] for col in HF_COLUMNS}
        for s in split_samples:
            dump = s.model_dump()
            for col in HF_COLUMNS:
                data[col].append(dump.get(col))

        split_datasets[split_name] = Dataset.from_dict(data)

    return DatasetDict(split_datasets)


def save_dataset_locally(dataset_dict, path: str) -> None:
    """Save a HuggingFace DatasetDict to disk for offline use."""
    dataset_dict.save_to_disk(path)
    logger.info("Saved dataset to %s", path)


def load_dataset_locally(path: str) -> DatasetDict:
    """Load a HuggingFace DatasetDict from disk."""
    from datasets import DatasetDict

    _check_datasets_available()

    dataset_dict = DatasetDict.load_from_disk(path)
    logger.info("Loaded dataset from %s (splits: %s)", path, list(dataset_dict.keys()))
    return dataset_dict


def push_to_hub(
    dataset_dict,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
) -> str:
    """Push the dataset to the HuggingFace Hub.

    Parameters
    ----------
    dataset_dict:
        The DatasetDict to push.
    repo_id:
        Full repo ID, e.g. ``vuln-triage/vuln-triage-dataset``.
    token:
        HF API token. If None, reads ``HF_TOKEN`` from env.
    private:
        Whether the dataset repo should be private.

    Returns the repo URL on the hub.
    """
    _check_datasets_available()

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No HuggingFace token provided. Set the HF_TOKEN environment "
            "variable or call huggingface-cli login, then retry."
        )

    dataset_dict.push_to_hub(
        repo_id,
        token=token,
        private=private,
    )
    logger.info("Pushed dataset to https://huggingface.co/datasets/%s", repo_id)
    return f"https://huggingface.co/datasets/{repo_id}"


def pull_from_hub(
    repo_id: str = "vuln-triage/vuln-triage-dataset",
    token: str | None = None,
    split: str | None = None,
    revision: str | None = None,
) -> DatasetDict:
    """Download a dataset from the HuggingFace Hub.

    Parameters
    ----------
    repo_id:
        Full repo ID on the HF Hub.
    token:
        HF API token. If None, reads ``HF_TOKEN`` from env (or uses the
        cached login from ``huggingface-cli login``).
    split:
        If provided, return only this split as a Dataset (not DatasetDict).
    revision:
        Pin to a specific commit SHA for reproducibility (e.g. ``"main"``
        or a full commit hash). Pinning is recommended for production runs
        — unpinned downloads may change between runs as the hub repo is
        updated.
    """
    from datasets import load_dataset

    _check_datasets_available()

    token = token or os.environ.get("HF_TOKEN")

    dataset = load_dataset(repo_id, token=token, revision=revision)

    if split is not None:
        return dataset[split]
    return dataset
