import random

import pytest

from app.data.cleaning.split import (
    SplitManifest,
    apply_manifest,
    leakage_safe_split,
    load_manifest,
    save_manifest,
)
from app.schemas.vuln import VulnSample

_CWES = ("CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502")


def _sample(id_: str, repo: str, cwe: str) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        cwe_id=cwe,
        severity="high",
        language="python",
        vulnerable_code=f"code {id_}",
        description="d",
    )


def _many_small_repos(n: int = 200, seed: int = 3) -> list[VulnSample]:
    """One sample per repo — the case where the greedy bin-packer has the
    most freedom, so ratios should land close to target.
    """
    rng = random.Random(seed)
    return [_sample(f"s{i}", f"org/repo{i}", _CWES[rng.randrange(len(_CWES))]) for i in range(n)]


def test_split_ratios_sum_validation():
    with pytest.raises(ValueError):
        leakage_safe_split([_sample("a", "r", "CWE-89")], ratios={"train": 0.5, "val": 0.3})


def test_empty_input_returns_empty():
    updated, manifest = leakage_safe_split([], seed=1)
    assert updated == []
    assert manifest.assignment == {}


def test_ratios_approximate_target_with_many_small_repos():
    samples = _many_small_repos()
    updated, _manifest = leakage_safe_split(samples, seed=1)
    total = len(updated)
    counts = {name: 0 for name in ("train", "val", "test", "gold_eval")}
    for s in updated:
        counts[s.split] += 1

    # Each repo is a single sample here, so the bin-packer has fine-grained
    # control; allow a modest tolerance rather than demanding exact ratios.
    assert counts["train"] / total == pytest.approx(0.7, abs=0.08)
    assert counts["val"] / total == pytest.approx(0.1, abs=0.06)
    assert counts["test"] / total == pytest.approx(0.1, abs=0.06)
    assert counts["gold_eval"] / total == pytest.approx(0.1, abs=0.06)


def test_reproducible_given_same_seed():
    samples = _many_small_repos()
    _u1, m1 = leakage_safe_split(samples, seed=7)
    _u2, m2 = leakage_safe_split(samples, seed=7)
    assert m1.assignment == m2.assignment


def test_manifest_round_trips_through_disk(tmp_path):
    samples = _many_small_repos(n=20)
    _updated, manifest = leakage_safe_split(samples, seed=5)

    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded.seed == manifest.seed
    assert loaded.assignment == manifest.assignment


def test_apply_manifest_reproduces_original_split():
    samples = _many_small_repos(n=20)
    updated, manifest = leakage_safe_split(samples, seed=5)

    reapplied = apply_manifest(samples, manifest)
    original_by_id = {s.id: s.split for s in updated}
    reapplied_by_id = {s.id: s.split for s in reapplied}
    assert original_by_id == reapplied_by_id


def test_apply_manifest_raises_on_unknown_repo():
    manifest = SplitManifest(seed=1, ratios={"train": 1.0}, assignment={"org/known": "train"})
    unknown = [_sample("x", "org/unknown", "CWE-89")]
    with pytest.raises(ValueError):
        apply_manifest(unknown, manifest)
