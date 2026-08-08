from app.data.cleaning.balance import check_class_balance
from app.data.collectors.cwe_scope import CWE_SCOPE
from app.schemas.vuln import VulnSample


def _sample(id_: str, split: str, cwe: str) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=f"org/repo-{id_}",
        cwe_id=cwe,
        severity="high",
        language="python",
        vulnerable_code="code",
        description="d",
        split=split,
    )


def test_missing_class_detected():
    samples = [_sample("1", "train", "CWE-89")]
    report = check_class_balance(samples)
    assert ("train", "CWE-79") in report.missing


def test_unsplit_samples_ignored():
    samples = [_sample("1", None, "CWE-89")]  # type: ignore[arg-type]
    report = check_class_balance(samples)
    # Nothing to report on for classes with no split assigned yet.
    assert report.counts == {}


def test_no_gaps_when_every_split_has_every_class():
    samples = []
    i = 0
    for split in ("train", "val", "test", "gold_eval"):
        for spec in CWE_SCOPE:
            samples.append(_sample(f"s{i}", split, spec.cwe_id))
            i += 1

    report = check_class_balance(samples)

    assert report.missing == []
    assert report.counts["train"]["CWE-89"] == 1


def test_counts_are_accurate():
    samples = [
        _sample("1", "train", "CWE-89"),
        _sample("2", "train", "CWE-89"),
        _sample("3", "train", "CWE-79"),
    ]
    report = check_class_balance(samples)
    assert report.counts["train"]["CWE-89"] == 2
    assert report.counts["train"]["CWE-79"] == 1
