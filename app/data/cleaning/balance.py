"""Stage 2, step 4: per-split CWE class-balance check."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from app.data.collectors.cwe_scope import CWE_SCOPE
from app.schemas.vuln import VulnSample

SPLIT_NAMES = ("train", "val", "test", "gold_eval")


@dataclass
class BalanceReport:
    counts: dict[str, dict[str, int]]  # split -> cwe_id -> count
    missing: list[tuple[str, str]]  # (split, cwe_id) pairs with zero samples


def check_class_balance(samples: list[VulnSample]) -> BalanceReport:
    raw_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sample in samples:
        if sample.split is None:
            continue
        raw_counts[sample.split][sample.cwe_id] += 1

    missing: list[tuple[str, str]] = []
    for split in SPLIT_NAMES:
        for spec in CWE_SCOPE:
            if raw_counts.get(split, {}).get(spec.cwe_id, 0) == 0:
                missing.append((split, spec.cwe_id))

    counts = {split: dict(cwe_counts) for split, cwe_counts in raw_counts.items()}
    return BalanceReport(counts=counts, missing=missing)
