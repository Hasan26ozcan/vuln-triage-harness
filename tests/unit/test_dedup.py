from app.data.cleaning.dedup import dedup_samples, find_near_duplicates
from app.schemas.vuln import VulnSample


def _sample(id_: str, code: str, repo: str = "org/repo") -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code=code,
        description="d",
    )


def test_exact_duplicate_detected_and_removed():
    code = "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)"
    a = _sample("a", code)
    b = _sample("b", code)

    kept, pairs = dedup_samples([a, b])

    assert len(kept) == 1
    assert len(pairs) == 1
    assert pairs[0].similarity > 0.95


def test_dissimilar_samples_both_kept():
    a = _sample("a", "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)")
    b = _sample("b", "os.system('rm -rf ' + path)")

    kept, pairs = dedup_samples([a, b])

    assert len(kept) == 2
    assert pairs == []


def test_near_duplicate_with_renamed_variable_detected():
    a = _sample("a", "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)")
    b = _sample("b", "cursor.execute('SELECT * FROM users WHERE id = ' + uid)")

    pairs = find_near_duplicates([a, b], threshold=0.8)

    assert len(pairs) == 1


def test_default_threshold_does_not_flag_moderately_different_code():
    a = _sample("a", "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)")
    b = _sample("b", "requests.get(BASE_URL + '/users/' + user_id, timeout=5)")

    pairs = find_near_duplicates([a, b], threshold=0.95)

    assert pairs == []


def test_keeps_first_sample_in_insertion_order():
    code = "x = 1"
    a = _sample("first", code)
    b = _sample("second", code)

    kept, _pairs = dedup_samples([a, b])

    assert kept[0].id == "first"
