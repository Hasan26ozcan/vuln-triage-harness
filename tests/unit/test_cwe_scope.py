from app.data.collectors.cwe_scope import CWE_SCOPE, cwe_spec, in_scope


def test_scope_stays_narrow():
    """Roadmap Stage 1 DoD: 5-8 classes, not more."""
    assert 5 <= len(CWE_SCOPE) <= 8


def test_no_duplicate_cwe_ids():
    ids = [spec.cwe_id for spec in CWE_SCOPE]
    assert len(ids) == len(set(ids))


def test_in_scope_lookup():
    assert in_scope("CWE-89") is True
    assert in_scope("CWE-999") is False


def test_cwe_spec_lookup_returns_none_for_out_of_scope():
    assert cwe_spec("CWE-999") is None
    assert cwe_spec("CWE-89") is not None


def test_every_spec_has_positive_min_samples():
    for spec in CWE_SCOPE:
        assert spec.min_samples > 0
