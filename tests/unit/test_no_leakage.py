"""Stage 2 Definition of Done (per roadmap, verbatim):
'hiçbir repo'nun birden fazla split'te görünmediğini otomatik test eden
bir test yazılmış ve CI'da çalışıyor' — this file is that test, and it
runs in CI via .github/workflows/ci.yml.
"""

import random

from app.data.cleaning.split import leakage_safe_split
from app.schemas.vuln import VulnSample

_CWES = ("CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502")


def _sample(id_: str, repo: str, cwe: str, idx: int) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        cwe_id=cwe,
        severity="high",
        language="python",
        vulnerable_code=f"vulnerable code body number {idx} in {repo}",
        description="synthetic sample for leakage testing",
    )


def _synthetic_dataset(n_repos: int = 60, max_per_repo: int = 4, seed: int = 1) -> list[VulnSample]:
    rng = random.Random(seed)
    samples = []
    for r in range(n_repos):
        repo = f"org/repo{r}"
        n = rng.randint(1, max_per_repo)
        for i in range(n):
            cwe = _CWES[rng.randrange(len(_CWES))]
            samples.append(_sample(f"s{r}_{i}", repo, cwe, i))
    return samples


def test_no_repo_appears_in_multiple_splits():
    samples = _synthetic_dataset()
    updated, _manifest = leakage_safe_split(samples, seed=7)

    repo_to_splits: dict[str, set[str]] = {}
    for sample in updated:
        repo_to_splits.setdefault(sample.repo_name, set()).add(sample.split)

    offending = {repo: splits for repo, splits in repo_to_splits.items() if len(splits) > 1}
    assert offending == {}, f"Leakage detected — repo(s) spanning multiple splits: {offending}"


def test_no_leakage_holds_across_multiple_seeds():
    """The guarantee is structural (whole repo-groups are atomic units), so
    it must hold regardless of seed — this isn't a property we got lucky
    on with one seed.
    """
    samples = _synthetic_dataset()
    for seed in (0, 1, 7, 42, 12345):
        updated, _manifest = leakage_safe_split(samples, seed=seed)
        repo_to_splits: dict[str, set[str]] = {}
        for sample in updated:
            repo_to_splits.setdefault(sample.repo_name, set()).add(sample.split)
        assert all(len(splits) == 1 for splits in repo_to_splits.values()), (
            f"Leakage at seed={seed}"
        )


def test_every_sample_gets_assigned_a_split():
    samples = _synthetic_dataset()
    updated, _manifest = leakage_safe_split(samples, seed=7)
    assert all(s.split in ("train", "val", "test", "gold_eval") for s in updated)
