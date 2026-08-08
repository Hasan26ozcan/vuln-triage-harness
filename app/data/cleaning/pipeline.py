"""Stage 2 orchestration: load -> dedup -> split -> contamination check.

This module ties together the Stage 2 sub-components into a single pipeline:

  1. **Load** VulnSample records from Postgres + MinIO (where Stage 1
     persisted them).
  2. **Dedup** near-duplicate samples using embedding similarity.
  3. **Split** into train/val/test using the repo-based, leakage-safe splitter.
  4. **Contamination check** between train and eval/test sets.
  5. **Persist** the updated `split` field back to Postgres (so Stage 3+
     and Stage 6+ know which split each sample belongs to).

The pipeline is designed to be partially re-runnable: if you've already
collected samples in Postgres, you can run Stage 2 without re-running Stage 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.data.cleaning.contamination import (
    ContaminationReport,
    check_contamination,
    contamination_is_acceptable,
)
from app.data.cleaning.dedup import DuplicatePair, dedup_samples
from app.data.cleaning.embeddings import EmbeddingBackend
from app.data.cleaning.split import (
    DEFAULT_SEED,
    SplitConfig,
    SplitResult,
    split_leakage_safe,
    verify_no_leakage,
)
from app.schemas.vuln import VulnSample
from app.storage.db import VulnSampleRow, get_session
from app.storage.object_store import get_json

logger = logging.getLogger(__name__)


@dataclass
class Stage2Result:
    """Full output of the Stage 2 cleaning pipeline."""

    samples_loaded: int
    samples_after_dedup: int
    duplicate_pairs: list[DuplicatePair]
    split_result: SplitResult
    contamination_report: ContaminationReport
    contamination_ok: bool


def load_samples_from_storage() -> list[VulnSample]:
    """Load all VulnSample records from Postgres + MinIO.

    Postgres holds the lightweight metadata (CWE, repo, severity, split,
    the MinIO object key); MinIO holds the full code payloads.
    """
    session = get_session()
    try:
        rows = session.query(VulnSampleRow).all()
        samples: list[VulnSample] = []
        for row in rows:
            key = row.object_store_key
            try:
                payload = get_json(key)
                samples.append(VulnSample(**payload))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load sample %s from MinIO (key=%s): %s", row.id, key, exc)
        return samples
    finally:
        session.close()


def persist_splits(samples: list[VulnSample]) -> None:
    """Write the updated `split` field for each sample back to Postgres.

    Only the `split` column is updated — the full payload stays in MinIO
    untouched. This is idempotent: running Stage 2 again just re-assigns
    the same splits (given the same seed).
    """
    session = get_session()
    try:
        for s in samples:
            session.query(VulnSampleRow).filter(VulnSampleRow.id == s.id).update(
                {"split": s.split}
            )
        session.commit()
        logger.info("Persisted split assignments for %d samples to Postgres", len(samples))
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_stage2(
    *,
    dedup_threshold: float = 0.95,
    split_config: SplitConfig | None = None,
    contamination_n: int = 5,
    max_contamination: float = 0.05,
    persist: bool = True,
    embedding_backend: EmbeddingBackend | None = None,
) -> Stage2Result:
    """Run the complete Stage 2 pipeline.

    Parameters
    ----------
    dedup_threshold:
        Cosine similarity threshold for near-duplicate detection.
    split_config:
        SplitConfig for the leakage-safe split. Defaults to 70/15/15, seed 42.
    contamination_n:
        N-gram length for contamination check (default 5).
    max_contamination:
        Max acceptable contamination rate (default 5%).
    persist:
        If True, write the updated split assignments to Postgres.
    embedding_backend:
        Optional custom EmbeddingBackend (e.g. with a mock model for tests).
    """
    split_config = split_config or SplitConfig(seed=DEFAULT_SEED)

    # Step 1: Load
    samples = load_samples_from_storage()
    logger.info("Stage 2: loaded %d samples from storage", len(samples))
    if not samples:
        raise RuntimeError(
            "No samples found in Postgres/MinIO. Run Stage 1 first: "
            "python -m app.data.collectors.cli collect --db-path ./CVEfixes.db"
        )

    # Step 2: Dedup
    backend = embedding_backend or EmbeddingBackend()
    deduped, dup_pairs = dedup_samples(samples, backend=backend, threshold=dedup_threshold)
    logger.info(
        "Stage 2: dedup removed %d near-duplicates (threshold=%.2f)",
        len(dup_pairs), dedup_threshold,
    )

    # Step 3: Leakage-safe split
    split_result = split_leakage_safe(deduped, config=split_config)
    verify_no_leakage(split_result)
    logger.info("Stage 2: split — %s", split_result.counts())

    # Step 4: Contamination check (train vs test)
    contamination_report = check_contamination(
        train_samples=split_result.train,
        eval_samples=split_result.test,
        n=contamination_n,
    )
    contamination_ok = contamination_is_acceptable(
        contamination_report.contamination_rate,
        max_threshold=max_contamination,
    )
    logger.info(
        "Stage 2: contamination rate=%.4f (threshold=%.4f, ok=%s)",
        contamination_report.contamination_rate,
        max_contamination,
        contamination_ok,
    )
    if not contamination_ok:
        logger.warning(
            "Contamination check FAILED: %.1f%% of eval n-grams appear in train. "
            "Consider increasing dedup_threshold or reviewing the split.",
            contamination_report.contamination_rate * 100,
        )

    # Step 5: Persist
    if persist:
        persist_splits(deduped)

    return Stage2Result(
        samples_loaded=len(samples),
        samples_after_dedup=len(deduped),
        duplicate_pairs=dup_pairs,
        split_result=split_result,
        contamination_report=contamination_report,
        contamination_ok=contamination_ok,
    )
