"""Stage 2, step 3: contamination check.

For every gold_eval sample, compute what fraction of its 20-grams (token
windows, roadmap-specified n=20) already appear somewhere in the train
set. A high containment ratio means the "held-out" eval sample is really
just a near-copy of something the model will have trained on — remove it
from gold_eval rather than let it inflate eval numbers.

`threshold=0.5` is a documented assumption, not a roadmap-specified value
(the roadmap says "yüksek overlap varsa çıkar" without pinning a number).
Half of a sample's 20-grams matching train is already a strong signal at
n=20 — real matches at that window size are rare by chance, but it's a
value worth revisiting once real data makes it possible to look at the
score distribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.vuln import VulnSample


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|\S", text)


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class ContaminationResult:
    sample_id: str
    containment_ratio: float
    contaminated: bool


def build_ngram_index(samples: list[VulnSample], n: int = 20) -> set[tuple[str, ...]]:
    """Union of all n-grams across train's vulnerable_code AND fixed_code —
    a gold sample matching either side of a train pair is still leakage.
    """
    index: set[tuple[str, ...]] = set()
    for sample in samples:
        index |= _ngrams(_tokenize(sample.vulnerable_code), n)
        if sample.fixed_code:
            index |= _ngrams(_tokenize(sample.fixed_code), n)
    return index


def check_contamination(
    gold_samples: list[VulnSample],
    train_ngram_index: set[tuple[str, ...]],
    n: int = 20,
    threshold: float = 0.5,
) -> list[ContaminationResult]:
    results = []
    for sample in gold_samples:
        grams = _ngrams(_tokenize(sample.vulnerable_code), n)
        if not grams:
            results.append(ContaminationResult(sample.id, 0.0, False))
            continue
        overlap = len(grams & train_ngram_index)
        ratio = overlap / len(grams)
        results.append(ContaminationResult(sample.id, ratio, ratio > threshold))
    return results


def remove_contaminated(
    gold_samples: list[VulnSample],
    train_samples: list[VulnSample],
    n: int = 20,
    threshold: float = 0.5,
) -> tuple[list[VulnSample], list[ContaminationResult]]:
    index = build_ngram_index(train_samples, n)
    results = check_contamination(gold_samples, index, n, threshold)
    contaminated_ids = {r.sample_id for r in results if r.contaminated}
    clean = [s for s in gold_samples if s.id not in contaminated_ids]
    return clean, results
