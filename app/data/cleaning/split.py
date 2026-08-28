"""Stage 2, step 3: leakage-safe, repo-based dataset split with class balance.

The core invariant: **no repository appears in more than one split**. Splitting
by individual sample would leak — the same repo's code can show up in both
train and test (CVEfixes often has multiple CVEs per repo, and the fixing
commits for different CVEs in the same repo may touch the same helper
functions). That makes the test set uninformative: the model has already seen
near-identical code during training, not because it generalises but because it
memorised.

By grouping on `repo_name` before splitting, we guarantee that any repo in
the test set is *completely unseen* during training.

Class balance: within each split we maintain the same proportion of CWE classes
as the overall dataset (stratified sampling at the repo level).

The `gold_eval` split is *not* produced by this function — it is a separate,
manually-curated set in `eval/gold_set/` (see README Stage 6). This function
produces only `train`, `val`, and `test`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.schemas.vuln import VulnSample

SplitName = Literal["train", "val", "test"]

# Default ratios must sum to 1.0 (excluding gold_eval, which is separate).
DEFAULT_RATIOS: dict[SplitName, float] = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Reproducible shuffling seed so re-runs produce the same split.
DEFAULT_SEED: int = 42


@dataclass
class SplitConfig:
    """Configuration for the leakage-safe split."""

    ratios: dict[SplitName, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    seed: int = DEFAULT_SEED
    min_per_class_train: int = 10
    min_per_class_test: int = 5


class SplitConfigError(ValueError):
    """Raised when the split config is invalid (ratios don't sum to 1, etc.)."""


@dataclass
class SplitResult:
    """The output of a leakage-safe split: three lists of VulnSamples,
    each already annotated with ``.split`` set.
    """

    train: list[VulnSample]
    val: list[VulnSample]
    test: list[VulnSample]
    config: SplitConfig

    @property
    def all_samples(self) -> list[VulnSample]:
        return self.train + self.val + self.test

    def counts(self) -> dict[SplitName, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }

    def cwe_distribution(self) -> dict[SplitName, dict[str, int]]:
        """Per-split CWE class counts — the primary balance metric."""
        dist: dict[SplitName, dict[str, int]] = {s: {} for s in ("train", "val", "test")}
        for s in ("train", "val", "test"):
            for sample in getattr(self, s):
                dist[s][sample.cwe_id] = dist[s].get(sample.cwe_id, 0) + 1
        return dist


@dataclass
class LeakAwareSplit:
    """A validated split plan produced from raw samples.

    This is an intermediate representation: it tells you which repo goes
    into which split, and which CWE class each repo's samples belong to.
    Use ``SplitResult`` for the actual sample lists.
    """

    repo_to_split: dict[str, SplitName]
    repo_to_cwe: dict[str, str]
    sample_to_split: dict[str, SplitName]


def _validate_config(config: SplitConfig) -> None:
    """Ensure ratios sum to (approximately) 1.0 and all required keys exist."""
    missing = [k for k in ("train", "val", "test") if k not in config.ratios]
    if missing:
        raise SplitConfigError(f"Missing split ratios for: {missing}")

    total = sum(config.ratios.values())
    if not (0.99 <= total <= 1.01):
        raise SplitConfigError(f"Split ratios must sum to 1.0, got {total:.4f} ({config.ratios})")

    if config.min_per_class_train < 1:
        raise SplitConfigError("min_per_class_train must be >= 1")
    if config.min_per_class_test < 1:
        raise SplitConfigError("min_per_class_test must be >= 1")


def _group_repos_by_cwe(samples: list[VulnSample]) -> dict[str, list[str]]:
    """Group unique repo names by the CWE class of their samples.

    A repo may have samples from multiple CWE classes — in that case the repo
    is assigned to the CWE class that has the most samples in it, so the
    stratification can reason about per-CWE repo counts.
    """
    cwe_of_repo: dict[str, dict[str, int]] = {}
    for s in samples:
        repo = s.repo_name
        cwe = s.cwe_id
        if repo not in cwe_of_repo:
            cwe_of_repo[repo] = {}
        cwe_of_repo[repo][cwe] = cwe_of_repo[repo].get(cwe, 0) + 1

    # Assign each repo to its dominant CWE class
    group: dict[str, list[str]] = {}
    for repo, cwe_counts in cwe_of_repo.items():
        dominant_cwe = max(cwe_counts, key=lambda k: cwe_counts[k])
        group.setdefault(dominant_cwe, []).append(repo)

    return group


def _deterministic_hash(s: str) -> int:
    """Hash a string deterministically (unlike built-in hash() which is
    randomized across Python processes).

    Uses hashlib so the same CWE class always maps to the same per-class
    seed, guaranteeing reproducibility regardless of PYTHONHASHSEED.
    """
    import hashlib

    digest = hashlib.md5(s.encode(), usedforsecurity=False).digest()[:4]
    return int.from_bytes(digest, byteorder="big")


def _shuffle_with_seed(items: list, seed: int) -> list:
    """Deterministic shuffle using the given seed.

    Uses ``random.Random``, NOT ``secrets`` — this is a reproducibility
    concern (same seed → same split), not a cryptographic one. The split
    must be deterministic and re-runnable, not unpredictable.
    """
    import random

    # Seeded PRNG for reproducible splits, not for crypto
    rng = random.Random(seed)  # nosec B311
    result = list(items)
    rng.shuffle(result)
    return result


def _split_repos_for_cwe(
    repos: list[str],
    ratios: dict[SplitName, float],
    seed: int,
) -> dict[str, SplitName]:
    """Assign each repo in `repos` to train/val/test using the given ratios.

    Repositories are shuffled deterministically, then assigned in order so
    that the proportions follow `ratios` as closely as discrete boundaries
    allow.
    """

    shuffled = _shuffle_with_seed(repos, seed)
    n = len(shuffled)

    n_train = max(0, round(n * ratios["train"]))
    n_val = max(0, round(n * ratios["val"]))
    n_test = n - n_train - n_val  # remainder goes to test to guarantee sum == n

    # Clamp to avoid negative when rounding overshoots
    if n_test < 0:
        n_val += n_test  # reduce val instead
        n_test = 0

    assignment: dict[str, SplitName] = {}
    for i, repo in enumerate(shuffled):
        if i < n_train:
            assignment[repo] = "train"
        elif i < n_train + n_val:
            assignment[repo] = "val"
        else:
            assignment[repo] = "test"

    return assignment


def split_leakage_safe(
    samples: list[VulnSample],
    config: SplitConfig | None = None,
) -> SplitResult:
    """Produce a leakage-safe, class-balanced train/val/test split.

    Algorithm:
      1. Validate the SplitConfig (ratios sum to 1).
      2. Group repositories by their dominant CWE class.
      3. Within each CWE class, shuffle repos deterministically (seeded)
         and assign them to train/val/test per the configured ratios.
      4. Assign the split label to every sample in each assigned repo.
      5. Return a SplitResult with counts and CWE distributions.

    Guarantees:
      - No repo appears in more than one split (leakage-safe).
      - CWE class proportions are approximately preserved across splits.
      - The split is reproducible given the same seed + input set.

    Parameters
    ----------
    samples:
        VulnSample list from Stage 1 (must have ``repo_name`` populated).
    config:
        SplitConfig. Defaults to 70/15/15 with seed 42.

    Returns
    -------
    SplitResult with `.train`, `.val`, `.test` lists (each sample's `.split`
    field is set to the assigned label).
    """
    config = config or SplitConfig()
    _validate_config(config)

    if not samples:
        return SplitResult(
            train=[],
            val=[],
            test=[],
            config=config,
        )

    # 1. Group repos by dominant CWE class
    repos_by_cwe = _group_repos_by_cwe(samples)

    # 2. Split each CWE class independently (stratified)
    repo_to_split: dict[str, SplitName] = {}
    # Vary the seed per CWE class so the same repo doesn't always land in
    # the same split across classes — this makes the overall split more
    # robust to small-class-size edge effects.
    # Use a deterministic hash (not Python's built-in hash(), which is
    # randomized across processes) so the split is reproducible everywhere.
    for cwe_id, repos in repos_by_cwe.items():
        cwe_seed = config.seed + _deterministic_hash(cwe_id) % 1000
        assignments = _split_repos_for_cwe(repos, config.ratios, cwe_seed)
        repo_to_split.update(assignments)

    # 3. Assign split labels to samples
    train, val, test = [], [], []
    for s in samples:
        split_name = repo_to_split[s.repo_name]
        s.split = split_name
        if split_name == "train":
            train.append(s)
        elif split_name == "val":
            val.append(s)
        elif split_name == "test":
            test.append(s)

    return SplitResult(train=train, val=val, test=test, config=config)


def build_leak_aware_plan(
    samples: list[VulnSample],
    config: SplitConfig | None = None,
) -> LeakAwareSplit:
    """Produce a validated split *plan* (repo-to-split mapping) without
    mutating samples. Useful for dry-run / inspection before committing.
    """
    config = config or SplitConfig()
    _validate_config(config)

    if not samples:
        return LeakAwareSplit(
            repo_to_split={},
            repo_to_cwe={},
            sample_to_split={},
        )

    repos_by_cwe = _group_repos_by_cwe(samples)
    repo_to_split: dict[str, SplitName] = {}
    repo_to_cwe: dict[str, str] = {}

    for cwe_id, repos in repos_by_cwe.items():
        cwe_seed = config.seed + _deterministic_hash(cwe_id) % 1000
        assignments = _split_repos_for_cwe(repos, config.ratios, cwe_seed)
        repo_to_split.update(assignments)
        for repo in repos:
            repo_to_cwe[repo] = cwe_id

    sample_to_split = {s.id: repo_to_split[s.repo_name] for s in samples}

    return LeakAwareSplit(
        repo_to_split=repo_to_split,
        repo_to_cwe=repo_to_cwe,
        sample_to_split=sample_to_split,
    )


class LeakageError(RuntimeError):
    """Raised when a repository appears in more than one data split."""


def verify_no_leakage(result: SplitResult) -> bool:
    """Verify that no repository appears in more than one split.

    Raises ``LeakageError`` if any repository is found in multiple splits.
    Returns ``True`` if the check passes.
    """
    train_repos = {s.repo_name for s in result.train}
    val_repos = {s.repo_name for s in result.val}
    test_repos = {s.repo_name for s in result.test}

    overlap_tv = train_repos & val_repos
    if overlap_tv:
        raise LeakageError(f"Leakage detected: repos in both train and val: {overlap_tv}")

    overlap_tt = train_repos & test_repos
    if overlap_tt:
        raise LeakageError(f"Leakage detected: repos in both train and test: {overlap_tt}")

    overlap_vt = val_repos & test_repos
    if overlap_vt:
        raise LeakageError(f"Leakage detected: repos in both val and test: {overlap_vt}")

    return True
