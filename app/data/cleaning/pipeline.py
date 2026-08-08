"""Stage 2 orchestration, in the order the roadmap lists the steps:
dedup -> split -> contamination check -> balance report.

Dedup runs before split so near-duplicates don't get artificially spread
across different splits by the grouping step. Contamination check runs
after split because it specifically checks gold_eval against train, which
only exist once the split has happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.data.cleaning.balance import BalanceReport, check_class_balance
from app.data.cleaning.contamination import ContaminationResult, remove_contaminated
from app.data.cleaning.dedup import DuplicatePair, dedup_samples
from app.data.cleaning.embeddings import EmbeddingBackend
from app.data.cleaning.split import SplitManifest, leakage_safe_split
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    samples: list[VulnSample]
    manifest: SplitManifest
    duplicate_pairs: list[DuplicatePair]
    contamination_results: list[ContaminationResult]
    balance_report: BalanceReport


def run_cleaning_pipeline(
    samples: list[VulnSample],
    embedding_backend: EmbeddingBackend | None = None,
    dedup_threshold: float = 0.95,
    contamination_n: int = 20,
    contamination_threshold: float = 0.5,
    seed: int = 42,
) -> CleaningResult:
    deduped, dup_pairs = dedup_samples(
        samples, backend=embedding_backend, threshold=dedup_threshold
    )
    logger.info(
        "Dedup: %d -> %d samples (%d near-duplicates removed)",
        len(samples), len(deduped), len(dup_pairs),
    )

    split_samples, manifest = leakage_safe_split(deduped, seed=seed)

    train = [s for s in split_samples if s.split == "train"]
    gold = [s for s in split_samples if s.split == "gold_eval"]
    clean_gold, contamination_results = remove_contaminated(
        gold, train, n=contamination_n, threshold=contamination_threshold
    )
    removed_gold_ids = {s.id for s in gold} - {s.id for s in clean_gold}
    final_samples = [s for s in split_samples if s.id not in removed_gold_ids]
    logger.info("Contamination check: removed %d gold_eval samples", len(removed_gold_ids))

    balance_report = check_class_balance(final_samples)
    if balance_report.missing:
        logger.warning("Class balance gaps (split, cwe_id): %s", balance_report.missing)

    return CleaningResult(
        samples=final_samples,
        manifest=manifest,
        duplicate_pairs=dup_pairs,
        contamination_results=contamination_results,
        balance_report=balance_report,
    )
