"""Stage 2: Cleaning, dedup, leakage-safe split, contamination check."""

from app.data.cleaning.contamination import (
    ContaminationReport,
    check_contamination,
    ngram_contamination_rate,
)
from app.data.cleaning.dedup import DuplicatePair, dedup_samples, find_near_duplicates
from app.data.cleaning.embeddings import EmbeddingBackend, cosine_similarity
from app.data.cleaning.split import LeakAwareSplit, SplitResult, split_leakage_safe

__all__ = [
    "EmbeddingBackend",
    "cosine_similarity",
    "find_near_duplicates",
    "dedup_samples",
    "DuplicatePair",
    "LeakAwareSplit",
    "split_leakage_safe",
    "SplitResult",
    "check_contamination",
    "ngram_contamination_rate",
    "ContaminationReport",
]
