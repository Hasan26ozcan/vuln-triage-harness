"""Stage 2, step 4: n-gram contamination checker for the gold-eval set.

After splitting, we need to verify that the eval/test data hasn't leaked into
the training set — and more importantly, that the gold-eval set doesn't
contain examples that were likely in the base model's pre-training corpus
(which would inflate metrics on a fine-tuned model).

This module provides two checks:

1. **Train-to-eval contamination** (`check_contamination`): generates n-grams
   from the eval set's `vulnerable_code` and checks whether any of those
   n-grams appear in the training set. If they do, the eval set is
   contaminated — the model could be scoring well simply because it has
   memorised the training examples.

2. **Gold-eval contamination rate** (`ngram_contamination_rate`): a scalar
   metric — what fraction of gold-eval n-grams appear in the training set.
   This is reported in the eval gate (Stage 10 CI).

N-grams are computed over word tokens (whitespace + punctuation split) rather
than characters, which gives a good precision/recall trade-off for code:
character n-grams are too noisy (common 3-grams like `def`, `for`), word
n-grams capture the structural signature of a vulnerability without being
so specific that nothing ever matches.

The default n-gram length is 5, following the GLIDE/code-contamination
literature (e.g., "Extracting Training Data from Code Models" — 5-grams
at 85%+ overlap signal contamination).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.vuln import VulnSample

# Default n-gram length — 5 tokens is the sweet spot from the contamination
# literature for code models. See docstring for references.
DEFAULT_N: int = 5

# Minimum overlap fraction for an n-gram to count as "contaminated".
DEFAULT_OVERLAP_THRESHOLD: float = 1.0  # exact match on all n tokens


@dataclass
class ContaminationReport:
    """Report on contamination between a reference (train) set and a query
    (eval) set.
    """

    n_train_samples: int
    n_eval_samples: int
    n_eval_ngrams: int
    n_contaminated_ngrams: int
    contaminated_samples: set[str] = field(default_factory=set)
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD

    @property
    def contamination_rate(self) -> float:
        """Fraction of eval n-grams that appear in the training set."""
        if self.n_eval_ngrams == 0:
            return 0.0
        return self.n_contaminated_ngrams / self.n_eval_ngrams

    @property
    def contaminated_sample_rate(self) -> float:
        """Fraction of eval samples that have >=1 contaminated n-gram."""
        if self.n_eval_samples == 0:
            return 0.0
        return len(self.contaminated_samples) / self.n_eval_samples

    def summary(self) -> str:
        pct = self.contamination_rate * 100
        sample_pct = self.contaminated_sample_rate * 100
        return (
            f"Contamination report:\n"
            f"  Train samples: {self.n_train_samples}, "
            f"Eval samples: {self.n_eval_samples}\n"
            f"  Eval n-grams: {self.n_eval_ngrams}\n"
            f"  Contaminated n-grams: {self.n_contaminated_ngrams} ({pct:.1f}%)\n"
            f"  Contaminated samples: {len(self.contaminated_samples)} "
            f"({sample_pct:.1f}% of eval set)"
        )


def tokenize_code(code: str) -> list[str]:
    """Tokenise a code snippet into word-level tokens.

    Uses a simple regex that splits on whitespace and common punctuation
    boundaries, preserving string literals, identifiers, and operators as
    separate tokens. This is intentionally lightweight — it doesn't need a
    full parser, just enough signal to detect near-identical code.
    """
    # Match: identifiers/words, numbers, operators, punctuation, strings
    token_pattern = re.compile(
        r"""
        [a-zA-Z_]\w*            # identifiers
        | \d+(?:\.\d+)?           # numbers
        | [+\-*/=<>!&|^~@.,;:{}()\[\]]  # operators/punctuation
        | ["'][^"']*["']          # string literals
        """,
        re.VERBOSE,
    )
    return token_pattern.findall(code)


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Generate word-level n-grams from a token list.

    Consecutive duplicate n-grams within the same sample are collapsed to
    avoid double-counting (e.g. a loop body repeated twice).
    """
    if len(tokens) < n:
        return [tuple(tokens)] if tokens else []

    seen: set[tuple[str, ...]] = set()
    ngrams: list[tuple[str, ...]] = []
    for i in range(len(tokens) - n + 1):
        gram = tuple(tokens[i : i + n])
        if gram not in seen:
            seen.add(gram)
            ngrams.append(gram)
    return ngrams


def ngram_contamination_rate(
    train_samples: list[VulnSample],
    eval_samples: list[VulnSample],
    n: int = DEFAULT_N,
) -> float:
    """Compute the fraction of eval n-grams that appear in the training set.

    A n-gram is "contaminated" if it appears verbatim in any training sample's
    `vulnerable_code`. The rate is:

        contaminated_eval_ngrams / total_unique_eval_ngrams

    Returns 0.0 if the eval set produces no n-grams.
    """
    train_ngrams = _build_ngram_set(train_samples, n)
    eval_ngrams = _build_ngram_set(eval_samples, n)

    if not eval_ngrams:
        return 0.0

    contaminated = eval_ngrams & train_ngrams
    return len(contaminated) / len(eval_ngrams)


def _build_ngram_set(samples: list[VulnSample], n: int) -> set[tuple[str, ...]]:
    """Build a set of all unique n-grams across a list of samples."""
    ngrams: set[tuple[str, ...]] = set()
    for s in samples:
        tokens = tokenize_code(s.vulnerable_code)
        ngrams.update(extract_ngrams(tokens, n))
    return ngrams


def check_contamination(
    train_samples: list[VulnSample],
    eval_samples: list[VulnSample],
    n: int = DEFAULT_N,
) -> ContaminationReport:
    """Full contamination check between train and eval (or test/gold_eval) sets.

    Produces a ContaminationReport with per-sample contamination tracking,
    not just the aggregate rate.
    """
    train_ngrams = _build_ngram_set(train_samples, n)

    contaminated_samples: set[str] = set()
    total_eval_ngrams = 0
    contaminated_ngram_count = 0

    for s in eval_samples:
        tokens = tokenize_code(s.vulnerable_code)
        sample_ngrams = extract_ngrams(tokens, n)
        total_eval_ngrams += len(sample_ngrams)
        sample_contaminated = False
        for gram in sample_ngrams:
            if gram in train_ngrams:
                contaminated_ngram_count += 1
                sample_contaminated = True
        if sample_contaminated:
            contaminated_samples.add(s.id)

    return ContaminationReport(
        n_train_samples=len(train_samples),
        n_eval_samples=len(eval_samples),
        n_eval_ngrams=total_eval_ngrams,
        n_contaminated_ngrams=contaminated_ngram_count,
        contaminated_samples=contaminated_samples,
        overlap_threshold=DEFAULT_OVERLAP_THRESHOLD,
    )


def contamination_is_acceptable(
    rate: float,
    max_threshold: float = 0.05,
) -> bool:
    """The project's contamination gate: the gold-eval set must have <5%
    n-gram overlap with the training set. This is checked in CI Stage 10
    before a checkpoint's eval metrics are accepted.
    """
    return rate < max_threshold
