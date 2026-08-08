import random

from app.data.cleaning.pipeline import run_cleaning_pipeline
from app.schemas.vuln import VulnSample

_CWES = ("CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502")


def _sample(id_: str, repo: str, cwe: str, code: str) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        cwe_id=cwe,
        severity="high",
        language="python",
        vulnerable_code=code,
        description="d",
    )


def _dataset_with_a_duplicate() -> list[VulnSample]:
    rng = random.Random(9)
    samples = []
    dup_code = "cursor.execute('SELECT * FROM t WHERE id = ' + x)"
    samples.append(_sample("dup_a", "org/repoA", "CWE-89", dup_code))
    samples.append(_sample("dup_b", "org/repoB", "CWE-89", dup_code))  # near-dupe, different repo

    for i in range(60):
        cwe = _CWES[rng.randrange(len(_CWES))]
        samples.append(_sample(f"s{i}", f"org/other{i}", cwe, f"distinct code body {i} {'x' * i}"))
    return samples


def test_pipeline_removes_duplicate_and_produces_valid_split():
    samples = _dataset_with_a_duplicate()
    result = run_cleaning_pipeline(samples, seed=3)

    # One of the two near-duplicates should be gone.
    assert len(result.duplicate_pairs) == 1
    remaining_ids = {s.id for s in result.samples}
    assert len({"dup_a", "dup_b"} & remaining_ids) == 1

    # Every remaining sample has a split, and no repo spans multiple splits.
    repo_to_splits: dict[str, set[str]] = {}
    for s in result.samples:
        assert s.split is not None
        repo_to_splits.setdefault(s.repo_name, set()).add(s.split)
    assert all(len(v) == 1 for v in repo_to_splits.values())


def test_pipeline_result_exposes_balance_report():
    samples = _dataset_with_a_duplicate()
    result = run_cleaning_pipeline(samples, seed=3)
    assert result.balance_report is not None
    assert isinstance(result.balance_report.missing, list)


def test_pipeline_is_reproducible_given_same_seed():
    samples = _dataset_with_a_duplicate()
    result1 = run_cleaning_pipeline(samples, seed=11)
    result2 = run_cleaning_pipeline(samples, seed=11)
    assert result1.manifest.assignment == result2.manifest.assignment
    assert {s.id for s in result1.samples} == {s.id for s in result2.samples}
