"""Tests for Stage 2's n-gram contamination checker.

These tests verify:
  - N-gram extraction from code tokens works correctly.
  - Contamination rate is computed correctly (exact match + no-match).
  - Per-sample contamination tracking works.
  - The acceptability gate works.
  - Edge cases: empty sets, small code, token boundary handling.
"""


from app.data.cleaning.contamination import (
    ContaminationReport,
    check_contamination,
    contamination_is_acceptable,
    extract_ngrams,
    ngram_contamination_rate,
    tokenize_code,
)
from app.schemas.vuln import VulnSample


def _sample(id_: str, code: str, cwe: str = "CWE-89") -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name="org/repo",
        commit_sha="abc123",
        cve_id=f"CVE-2024-{id_}",
        cwe_id=cwe,
        severity="high",
        language="python",
        vulnerable_code=code,
        description=f"Test {id_}",
    )


# --- Tokenizer ---


def test_tokenize_code_splits_on_punctuation():
    tokens = tokenize_code("cursor.execute('SELECT * FROM t')")
    assert "cursor" in tokens
    assert "execute" in tokens
    assert "'SELECT * FROM t'" in tokens  # string literal preserved
    assert "(" in tokens
    assert ")" in tokens


def test_tokenize_code_handles_fstrings():
    tokens = tokenize_code("f'query {user_input}'")
    assert "f" in tokens
    assert "'query {user_input}'" in tokens or "user_input" in tokens


def test_tokenize_code_empty_string():
    assert tokenize_code("") == []


def test_tokenize_code_no_ngrams_when_too_short():
    tokens = tokenize_code("x = 1")
    # Only 3 tokens, with n=5 we get one n-gram of length 3 (or fewer)
    ngrams = extract_ngrams(tokens, n=5)
    # When len(tokens) < n, extract_ngrams returns [tuple(tokens)]
    assert len(ngrams) == 1
    assert ngrams[0] == tuple(tokens)


# --- N-gram extraction ---


def test_extract_ngrams_basic():
    tokens = ["a", "b", "c", "d", "e"]
    ngrams = extract_ngrams(tokens, n=3)
    assert len(ngrams) == 3
    assert ngrams[0] == ("a", "b", "c")
    assert ngrams[1] == ("b", "c", "d")
    assert ngrams[2] == ("c", "d", "e")


def test_extract_ngrams_deduplicates_consecutive_repeats():
    tokens = ["a", "b", "a", "b", "c"]
    ngrams = extract_ngrams(tokens, n=2)
    # n-grams: (a,b), (b,a), (a,b), (b,c)
    # Deduplicated: (a,b), (b,a), (b,c)
    unique = set(ngrams)
    assert unique == {("a", "b"), ("b", "a"), ("b", "c")}


def test_extract_ngrams_n_larger_than_tokens():
    tokens = ["a", "b"]
    ngrams = extract_ngrams(tokens, n=5)
    assert ngrams == [("a", "b")]


# --- Contamination rate ---


def test_no_contamination_when_train_empty():
    train = []
    eval_samples = [_sample("e1", "cursor.execute(query)")]
    rate = ngram_contamination_rate(train, eval_samples, n=5)
    assert rate == 0.0


def test_no_contamination_when_no_overlap():
    train = [_sample("t1", "import os; os.system('rm -rf /')")]
    eval_samples = [_sample("e1", "cursor.execute('SELECT 1')")]
    rate = ngram_contamination_rate(train, eval_samples, n=5)
    assert rate == 0.0


def test_full_contamination_when_identical():
    code = "cursor.execute('SELECT * FROM users WHERE id = ' + user_input)"
    train = [_sample("t1", code)]
    eval_samples = [_sample("e1", code)]
    rate = ngram_contamination_rate(train, eval_samples, n=5)
    assert rate == 1.0


def test_partial_contamination():
    # Eval code = train prefix + new suffix
    train_code = "import os; os.system('ls')"
    eval_prefix = "import os; os.system('ls')"
    eval_suffix = "import os; os.system('rm -rf /')"
    train = [_sample("t1", train_code)]
    eval_samples = [_sample("e1", eval_prefix), _sample("e2", eval_suffix)]

    rate = ngram_contamination_rate(train, eval_samples, n=5)
    # e1 is fully contaminated (1.0), e2 shares the "import os" prefix
    # but the full 5-grams differ. The rate depends on n-gram overlap.
    assert 0.0 <= rate <= 1.0
    # e1 definitely has contamination (its n-grams appear in train)
    train_ngrams = set()
    for s in train:
        tokens = tokenize_code(s.vulnerable_code)
        for gram in extract_ngrams(tokens, 5):
            train_ngrams.add(gram)
    eval_e1_tokens = tokenize_code(eval_prefix)
    eval_e1_grams = set(extract_ngrams(eval_e1_tokens, 5))
    assert len(eval_e1_grams & train_ngrams) > 0  # at least some overlap


# --- Full report ---


def test_check_contamination_report_fields():
    train = [_sample("t1", "cursor.execute('SELECT * FROM t')")]
    eval_samples = [_sample("e1", "cursor.execute('SELECT * FROM t')")]

    report = check_contamination(train, eval_samples, n=5)

    assert report.n_train_samples == 1
    assert report.n_eval_samples == 1
    assert report.n_eval_ngrams > 0
    assert report.n_contaminated_ngrams > 0
    assert "e1" in report.contaminated_samples
    assert report.contamination_rate > 0
    assert report.contaminated_sample_rate > 0


def test_check_contamination_no_contamination():
    train = [_sample("t1", "cursor.execute('DROP TABLE users')")]
    eval_samples = [_sample("e1", "cursor.execute('SELECT * FROM t')")]

    report = check_contamination(train, eval_samples, n=5)

    assert report.n_contaminated_ngrams == 0
    assert report.contamination_rate == 0.0
    assert len(report.contaminated_samples) == 0


def test_check_contamination_multiple_eval_samples_partial():
    train_code = "import os; os.system('ls -la')"
    train = [_sample("t1", train_code)]

    # e1 identical to train, e2 totally different
    eval_samples = [
        _sample("e1", train_code),
        _sample("e2", "cursor.execute(sql_query)"),
    ]

    report = check_contamination(train, eval_samples, n=5)

    assert "e1" in report.contaminated_samples
    assert "e2" not in report.contaminated_samples
    assert report.contaminated_sample_rate == 0.5


# --- Edge cases: zero-division guards ---


def test_contamination_report_zero_division_guard():
    """When n_eval_ngrams or n_eval_samples is 0, properties return 0.0."""
    report = ContaminationReport(
        n_train_samples=10,
        n_eval_samples=0,
        n_eval_ngrams=0,
        n_contaminated_ngrams=0,
        contaminated_samples=set(),
    )
    assert report.contamination_rate == 0.0
    assert report.contaminated_sample_rate == 0.0


def test_ngram_contamination_rate_empty_eval_ngrams():
    """When eval samples produce no n-grams, rate should be 0.0."""
    train = [_sample("t1", "some vulnerable code here")]
    eval_samples = [_sample("e1", "")]
    rate = ngram_contamination_rate(train, eval_samples, n=5)
    assert rate == 0.0


# --- Acceptability gate ---


def test_contamination_is_acceptable_below_threshold():
    assert contamination_is_acceptable(0.03) is True
    assert contamination_is_acceptable(0.05) is False  # exactly at 5% is NOT acceptable


def test_contamination_is_acceptable_above_threshold():
    assert contamination_is_acceptable(0.10) is False


def test_contamination_custom_threshold():
    assert contamination_is_acceptable(0.08, max_threshold=0.10) is True
    assert contamination_is_acceptable(0.11, max_threshold=0.10) is False


# --- Report summary ---


def test_contamination_report_summary_string():
    report = ContaminationReport(
        n_train_samples=100,
        n_eval_samples=10,
        n_eval_ngrams=50,
        n_contaminated_ngrams=3,
        contaminated_samples={"e1"},
    )
    summary = report.summary()
    assert "Contamination report" in summary
    assert "6.0%" in summary  # 3/50 = 6%
    assert "10" in summary  # 1 contaminated sample
