"""Tests for Stage 2's leakage-safe repo-based split.

These tests verify:
  - No repository appears in more than one split (the leakage invariant).
  - CWE class proportions are approximately preserved across splits.
  - The split is reproducible given the same seed.
  - Config validation works (ratios must sum to 1).
  - The split assignment is deterministic and balanced.
"""

import pytest

from app.data.cleaning.split import (
    DEFAULT_RATIOS,
    LeakageError,
    SplitConfig,
    SplitConfigError,
    SplitResult,
    build_leak_aware_plan,
    split_leakage_safe,
    verify_no_leakage,
)
from app.schemas.vuln import VulnSample


def _sample(
    id_: str,
    cwe: str = "CWE-89",
    repo: str = "org/repo",
    severity: str = "high",
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        commit_sha="abc123",
        cve_id=f"CVE-2024-{id_}",
        cwe_id=cwe,
        severity=severity,
        language="python",
        vulnerable_code=f"# {id_}\ndef foo(): return '{id_}'",
        description=f"Test vuln {id_}",
    )


def _make_balanced_samples(
    n_per_class: int = 20,
    cwe_ids: list[str] | None = None,
) -> list[VulnSample]:
    """Generate a balanced dataset: equal repos per CWE class."""
    if cwe_ids is None:
        cwe_ids = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]

    samples: list[VulnSample] = []
    counter = 0
    for cwe in cwe_ids:
        for _ in range(n_per_class):
            # Each sample gets a unique repo — so splitting by repo means
            # splitting by sample, which is the worst case for repo-level
            # leakage (each repo has exactly one sample).
            counter += 1
            samples.append(
                _sample(
                    f"{cwe[:3]}_{counter}",
                    cwe=cwe,
                    repo=f"org/repo_{counter}",
                )
            )
    return samples


# --- Config validation ---


def test_split_config_rejects_ratios_not_summing_to_1():
    config = SplitConfig(ratios={"train": 0.5, "val": 0.3, "test": 0.3})
    # 0.5 + 0.3 + 0.3 = 1.1 — not 1.0
    with pytest.raises(SplitConfigError, match="sum to 1"):
        split_leakage_safe([], config=config)


def test_split_config_rejects_missing_ratio():
    config = SplitConfig(ratios={"train": 0.7, "val": 0.3})
    with pytest.raises(SplitConfigError, match="Missing"):
        split_leakage_safe([], config=config)


def test_split_config_accepts_valid_ratios():
    config = SplitConfig(ratios={"train": 0.6, "val": 0.2, "test": 0.2})
    result = split_leakage_safe([], config=config)
    assert result.train == []
    assert result.val == []
    assert result.test == []


# --- Leakage invariant ---


def test_no_repo_appears_in_more_than_one_split():
    samples = _make_balanced_samples(n_per_class=20)
    result = split_leakage_safe(samples)

    assert verify_no_leakage(result) is True


def test_split_assigns_every_sample():
    samples = _make_balanced_samples(n_per_class=15)
    result = split_leakage_safe(samples)

    all_split = result.train + result.val + result.test
    assert len(all_split) == len(samples)
    # Every sample has a non-None split
    for s in all_split:
        assert s.split is not None
        assert s.split in ("train", "val", "test")


def test_split_proportions_approximately_correct():
    samples = _make_balanced_samples(n_per_class=50)
    result = split_leakage_safe(samples)

    total = len(samples)
    for split_name, expected_ratio in DEFAULT_RATIOS.items():
        actual_ratio = len(getattr(result, split_name)) / total
        # Allow ±10% tolerance due to discrete repo-level assignment
        assert abs(actual_ratio - expected_ratio) < 0.10, (
            f"{split_name}: expected ~{expected_ratio:.0%}, got {actual_ratio:.0%}"
        )


# --- Reproducibility ---


def test_split_is_reproducible_same_seed():
    samples = _make_balanced_samples(n_per_class=20)
    r1 = split_leakage_safe(samples, config=SplitConfig(seed=42))
    r2 = split_leakage_safe(samples, config=SplitConfig(seed=42))

    assert {s.id for s in r1.train} == {s.id for s in r2.train}
    assert {s.id for s in r1.val} == {s.id for s in r2.val}
    assert {s.id for s in r1.test} == {s.id for s in r2.test}


def test_split_differs_with_different_seed():
    samples = _make_balanced_samples(n_per_class=20)
    r1 = split_leakage_safe(samples, config=SplitConfig(seed=42))
    r2 = split_leakage_safe(samples, config=SplitConfig(seed=99))

    # Different seeds should produce different test sets (very high probability
    # with 120 samples)
    assert {s.id for s in r1.test} != {s.id for s in r2.test}


# --- Class balance ---


def test_cwe_distribution_across_splits_is_balanced():
    """Each CWE class should be represented proportionally in each split."""
    samples = _make_balanced_samples(n_per_class=50)
    result = split_leakage_safe(samples)

    dist = result.cwe_distribution()
    total_per_cwe = {
        cwe: dist["train"].get(cwe, 0) + dist["val"].get(cwe, 0) + dist["test"].get(cwe, 0)
        for cwe in dist["train"]
    }

    for cwe, total in total_per_cwe.items():
        for split_name in ("train", "val", "test"):
            split_count = dist[split_name].get(cwe, 0)
            if total > 0:
                ratio = split_count / total
                expected_ratio = DEFAULT_RATIOS[split_name]
                # Allow ±15% tolerance for class balance at the repo level
                assert abs(ratio - expected_ratio) < 0.15, (
                    f"CWE {cwe} in {split_name}: expected ~{expected_ratio:.0%}, "
                    f"got {ratio:.0%}"
                )


# --- Multiple samples per repo (realistic scenario) ---


def test_multiple_samples_same_repo_go_to_same_split():
    """When a repo has multiple CVEs, all its samples must land in the same
    split — that's the whole point of the leakage-safe split."""
    samples = [
        _sample(f"s{i}", repo="org/project_a") for i in range(10)
    ] + [
        _sample(f"r{i}", repo="org/project_b", cwe="CWE-79") for i in range(10)
    ]

    result = split_leakage_safe(samples)

    # All samples from project_a must be in the same split
    project_a_splits = {s.split for s in result.all_samples if s.repo_name == "org/project_a"}
    assert len(project_a_splits) == 1, (
        f"project_a samples ended up in multiple splits: {project_a_splits}"
    )

    project_b_splits = {s.split for s in result.all_samples if s.repo_name == "org/project_b"}
    assert len(project_b_splits) == 1


def test_build_leak_aware_plan_does_not_mutate_samples():
    samples = _make_balanced_samples(n_per_class=10)
    original_splits = [s.split for s in samples]

    build_leak_aware_plan(samples, config=SplitConfig(seed=42))

    # The plan function should NOT mutate samples (unlike split_leakage_safe)
    assert all(s.split is None for s in samples)
    assert [s.split for s in samples] == original_splits


# --- Edge cases ---


def test_empty_samples_returns_empty_result():
    result = split_leakage_safe([])
    assert result.train == []
    assert result.val == []
    assert result.test == []
    assert verify_no_leakage(result) is True


def test_single_class_split():
    """When all samples are one CWE class, split should still work."""
    samples = [_sample(f"s{i}", repo=f"org/repo_{i}") for i in range(30)]
    result = split_leakage_safe(samples)

    assert len(result.train) + len(result.val) + len(result.test) == 30
    assert verify_no_leakage(result) is True


def test_verify_no_leakage_raises_on_conflicting_split_result():
    """If a SplitResult has repos in multiple splits, verify_no_leakage
    must raise LeakageError (not silently return False)."""
    s1 = _sample("a", repo="org/shared")
    s2 = _sample("b", repo="org/shared")

    result = SplitResult(
        train=[s1],
        val=[s2],  # same repo as s1 -> leakage
        test=[],
        config=SplitConfig(),
    )

    with pytest.raises(LeakageError, match="both train and val"):
        verify_no_leakage(result)


def test_split_with_default_ratios():
    """DefaultSplitConfig should use the standard 70/15/15 ratios."""
    assert DEFAULT_RATIOS == {"train": 0.70, "val": 0.15, "test": 0.15}
    config = SplitConfig()
    assert config.ratios == DEFAULT_RATIOS
    assert config.seed == 42
