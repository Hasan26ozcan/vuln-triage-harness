from app.data.cleaning.contamination import (
    build_ngram_index,
    check_contamination,
    remove_contaminated,
)
from app.schemas.vuln import VulnSample

_LONG_SNIPPET = " ".join(f"token{i}" for i in range(30))
_UNRELATED_SNIPPET = " ".join(f"other{i}" for i in range(30))


def _sample(id_: str, code: str, fixed: str | None = None) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name="org/repo",
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code=code,
        fixed_code=fixed,
        description="d",
    )


def test_identical_snippet_flagged_and_removed():
    train = [_sample("t1", _LONG_SNIPPET)]
    gold = [_sample("g1", _LONG_SNIPPET)]

    clean, results = remove_contaminated(gold, train, n=20, threshold=0.5)

    assert clean == []
    assert results[0].contaminated is True
    assert results[0].containment_ratio == 1.0


def test_unrelated_snippet_not_flagged():
    train = [_sample("t1", _LONG_SNIPPET)]
    gold = [_sample("g1", _UNRELATED_SNIPPET)]

    clean, results = remove_contaminated(gold, train, n=20, threshold=0.5)

    assert len(clean) == 1
    assert results[0].contaminated is False
    assert results[0].containment_ratio == 0.0


def test_contamination_checks_fixed_code_too():
    """A gold sample copying train's *fixed* code (not just the vulnerable
    side) is still leakage — the model would have seen the exact patch.
    """
    train = [_sample("t1", _UNRELATED_SNIPPET, fixed=_LONG_SNIPPET)]
    gold = [_sample("g1", _LONG_SNIPPET)]

    index = build_ngram_index(train, n=20)
    results = check_contamination(gold, index, n=20, threshold=0.5)

    assert results[0].contaminated is True


def test_short_snippet_below_ngram_size_handled_gracefully():
    train = [_sample("t1", "short code")]
    gold = [_sample("g1", "also short")]

    # Should not raise even though neither snippet reaches 20 tokens.
    clean, results = remove_contaminated(gold, train, n=20, threshold=0.5)
    assert len(results) == 1
