"""Stage 2, step 2: embedding-based near-duplicate removal.

Roadmap threshold: cosine similarity > 0.95 on `vulnerable_code` counts as
a near-duplicate; keep only one of each pair. O(n^2) pairwise comparison —
fine at this project's scale (low hundreds to ~1k samples across 6 CWE
classes); would need blocking/ANN search well before that stops being true.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.data.cleaning.embeddings import EmbeddingBackend, cosine_similarity
from app.schemas.vuln import VulnSample


@dataclass
class DuplicatePair:
    keep_id: str
    remove_id: str
    similarity: float


def find_near_duplicates(
    samples: list[VulnSample],
    backend: EmbeddingBackend | None = None,
    threshold: float = 0.95,
) -> list[DuplicatePair]:
    backend = backend or EmbeddingBackend()
    vectors = backend.embed([s.vulnerable_code for s in samples])

    pairs: list[DuplicatePair] = []
    removed: set[str] = set()
    n = len(samples)

    for i in range(n):
        if samples[i].id in removed:
            continue
        for j in range(i + 1, n):
            if samples[j].id in removed:
                continue
            similarity = cosine_similarity(vectors[i], vectors[j])
            if similarity > threshold:
                pairs.append(
                    DuplicatePair(
                        keep_id=samples[i].id, remove_id=samples[j].id, similarity=similarity
                    )
                )
                removed.add(samples[j].id)

    return pairs


def dedup_samples(
    samples: list[VulnSample],
    backend: EmbeddingBackend | None = None,
    threshold: float = 0.95,
) -> tuple[list[VulnSample], list[DuplicatePair]]:
    """Returns (kept_samples, duplicate_pairs). The first sample in
    insertion order within a near-duplicate cluster is kept.
    """
    pairs = find_near_duplicates(samples, backend, threshold)
    removed_ids = {p.remove_id for p in pairs}
    kept = [s for s in samples if s.id not in removed_ids]
    return kept, pairs
