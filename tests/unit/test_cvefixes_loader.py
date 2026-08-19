"""Tests the CVEfixes loader against a synthetic SQLite DB built with the
*real* v1.0.8 schema (see cvefixes_loader.py module docstring for the
source of truth). This does not touch the actual multi-GB CVEfixes.db —
it validates that our SQL joins are correct against a known-shape fixture.
"""

import sqlite3

import pytest

from app.data.collectors.cvefixes_loader import CveFixesLoader

SCHEMA = """
CREATE TABLE cve (cve_id TEXT PRIMARY KEY, description TEXT, severity TEXT);
CREATE TABLE cwe_classification (cve_id TEXT, cwe_id TEXT);
CREATE TABLE fixes (cve_id TEXT, hash TEXT, repo_url TEXT);
CREATE TABLE repository (repo_url TEXT PRIMARY KEY, repo_name TEXT);
CREATE TABLE file_change (
    file_change_id TEXT PRIMARY KEY,
    hash TEXT,
    programming_language TEXT,
    code_before TEXT,
    code_after TEXT
);
CREATE TABLE method_change (
    method_change_id TEXT PRIMARY KEY,
    file_change_id TEXT,
    name TEXT,
    code TEXT,
    before_change INTEGER
);
"""


@pytest.fixture
def cvefixes_db(tmp_path):
    db_path = tmp_path / "CVEfixes.db"
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA)

    # A CWE-89 (in-scope) sample with method-level before/after pair.
    con.execute("INSERT INTO cve VALUES ('CVE-2024-0001', 'SQLi bug', 'HIGH')")
    con.execute("INSERT INTO cwe_classification VALUES ('CVE-2024-0001', 'CWE-89')")
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-0001', 'sha_method', 'https://github.com/acme/app')"
    )
    con.execute("INSERT INTO repository VALUES ('https://github.com/acme/app', 'acme/app')")
    con.execute(
        "INSERT INTO file_change VALUES "
        "('fc_1', 'sha_method', 'python', 'FILE_BEFORE', 'FILE_AFTER')"
    )
    con.execute("INSERT INTO method_change VALUES ('mc_1', 'fc_1', 'get_user', 'VULN CODE', 1)")
    con.execute("INSERT INTO method_change VALUES ('mc_2', 'fc_1', 'get_user', 'FIXED CODE', 0)")

    # A CWE-79 (in-scope) sample with NO method_change rows -> must hit the
    # file-level fallback path.
    con.execute("INSERT INTO cve VALUES ('CVE-2024-0002', 'XSS bug', 'MEDIUM')")
    con.execute("INSERT INTO cwe_classification VALUES ('CVE-2024-0002', 'CWE-79')")
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-0002', 'sha_file', 'https://github.com/acme/web')"
    )
    con.execute("INSERT INTO repository VALUES ('https://github.com/acme/web', 'acme/web')")
    con.execute(
        "INSERT INTO file_change VALUES ('fc_2', 'sha_file', 'javascript', 'JS_BEFORE', 'JS_AFTER')"
    )

    # An out-of-scope CWE — must never appear in results.
    con.execute("INSERT INTO cve VALUES ('CVE-2024-0003', 'Unrelated bug', 'LOW')")
    con.execute("INSERT INTO cwe_classification VALUES ('CVE-2024-0003', 'CWE-999')")
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-0003', 'sha_oos', 'https://github.com/acme/other')"
    )
    con.execute("INSERT INTO repository VALUES ('https://github.com/acme/other', 'acme/other')")
    con.execute("INSERT INTO file_change VALUES ('fc_3', 'sha_oos', 'python', 'X', 'Y')")

    con.commit()
    con.close()
    return db_path


def test_missing_db_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        CveFixesLoader(tmp_path / "does_not_exist.db")


def test_method_level_pair_extracted(cvefixes_db):
    loader = CveFixesLoader(cvefixes_db)
    pairs = loader.load_pairs()

    method_pairs = [p for p in pairs if p.granularity == "method"]
    assert len(method_pairs) == 1
    pair = method_pairs[0]
    assert pair.cve_id == "CVE-2024-0001"
    assert pair.cwe_id == "CWE-89"
    assert pair.repo_name == "acme/app"
    assert pair.vulnerable_code == "VULN CODE"
    assert pair.fixed_code == "FIXED CODE"


def test_file_level_fallback_used_when_no_method_change(cvefixes_db):
    loader = CveFixesLoader(cvefixes_db)
    pairs = loader.load_pairs()

    file_pairs = [p for p in pairs if p.granularity == "file"]
    assert len(file_pairs) == 1
    pair = file_pairs[0]
    assert pair.cve_id == "CVE-2024-0002"
    assert pair.vulnerable_code == "JS_BEFORE"
    assert pair.fixed_code == "JS_AFTER"


def test_out_of_scope_cwe_excluded(cvefixes_db):
    loader = CveFixesLoader(cvefixes_db)
    pairs = loader.load_pairs()
    assert all(p.cve_id != "CVE-2024-0003" for p in pairs)


def test_language_filter(cvefixes_db):
    loader = CveFixesLoader(cvefixes_db)
    pairs = loader.load_pairs(languages={"python"})
    assert all(p.language == "python" for p in pairs)
    assert not any(p.cve_id == "CVE-2024-0002" for p in pairs)  # that one is javascript
