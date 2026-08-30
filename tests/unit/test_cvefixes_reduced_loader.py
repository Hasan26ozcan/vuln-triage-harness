"""Tests for the reduced-schema CVEfixes loader.

Covers:
  - RawVulnPair dataclass shape.
  - _derive_repo_name with various URL formats and edge cases.
  - ReducedCveFixesLoader.__init__ file-existence checks.
  - _load_cwe_mapping JSON parsing.
  - _prepare_temp_tables temp-table creation and population.
  - load_pairs full path against a synthetic SQLite DB.
  - Language filtering and default language set.
  - Out-of-scope CVE / CWE exclusion.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from app.data.collectors.cvefixes_reduced_loader import (
    RawVulnPair,
    ReducedCveFixesLoader,
    _derive_repo_name,
)

# ---------------------------------------------------------------------------
# _derive_repo_name
# ---------------------------------------------------------------------------


class TestDeriveRepoName:
    def test_github_url(self):
        assert _derive_repo_name("https://github.com/owner/repo") == "owner/repo"

    def test_github_url_with_trailing_slash(self):
        assert _derive_repo_name("https://github.com/owner/repo/") == "owner/repo"

    def test_github_url_with_dot_git(self):
        assert _derive_repo_name("https://github.com/owner/repo.git") == "owner/repo"

    def test_gitlab_url(self):
        assert _derive_repo_name("https://gitlab.com/org/project") == "org/project"

    def test_bitbucket_url(self):
        assert _derive_repo_name("https://bitbucket.org/user/repo") == "user/repo"

    def test_url_with_path_segments(self):
        url = "https://github.com/deep/nested/path/repo"
        assert _derive_repo_name(url) == "path/repo"

    def test_empty_url(self):
        assert _derive_repo_name("") == "unknown"

    def test_url_with_no_path_parts(self):
        """A URL that has no path segments falls back to the URL itself."""
        assert _derive_repo_name("not-a-url") == "not-a-url"

    def test_url_with_query_params(self):
        url = "https://github.com/owner/repo?query=1"
        assert _derive_repo_name(url) == "owner/repo"


# ---------------------------------------------------------------------------
# RawVulnPair
# ---------------------------------------------------------------------------


class TestRawVulnPair:
    def test_can_create_with_all_fields(self):
        pair = RawVulnPair(
            cve_id="CVE-2024-0001",
            cwe_id="CWE-89",
            repo_name="owner/repo",
            commit_sha="abc123",
            language="python",
            vulnerable_code="vuln",
            fixed_code="fixed",
            granularity="file",
        )
        assert pair.cve_id == "CVE-2024-0001"
        assert pair.granularity == "file"

    def test_fixed_code_can_be_none(self):
        pair = RawVulnPair(
            cve_id="CVE-2024-0001",
            cwe_id="CWE-89",
            repo_name="owner/repo",
            commit_sha="abc123",
            language="python",
            vulnerable_code="vuln",
            fixed_code=None,
            granularity="file",
        )
        assert pair.fixed_code is None


# ---------------------------------------------------------------------------
# ReducedCveFixesLoader — constructor & file checks
# ---------------------------------------------------------------------------

# Build a synthetic reduced-schema DB: fixes + commits + file_change + cwe mapping.
CODE_BEFORE_89 = (
    "def foo():\n"
    "    query = 'SELECT * FROM users WHERE id=' + user_input\n"
    "    cursor.execute(query)\n"
    "    return cursor.fetchall()"
)
CODE_AFTER_89 = (
    "def foo():\n"
    "    query = 'SELECT * FROM users WHERE id=?'\n"
    "    cursor.execute(query, (user_input,))\n"
    "    return cursor.fetchall()"
)
CODE_BEFORE_79 = "document.write('Hello ' + user_input + more); // xss"
CODE_AFTER_79 = "document.createTextNode('Hello ' + user_input)"
CODE_BEFORE_999 = "some vulnerable code here"

REDUCED_SCHEMA = """
CREATE TABLE fixes (
    cve_id TEXT,
    hash TEXT,
    repo_url TEXT
);
CREATE TABLE commits (
    hash TEXT PRIMARY KEY,
    message TEXT
);
CREATE TABLE file_change (
    file_change_id TEXT PRIMARY KEY,
    hash TEXT,
    programming_language TEXT,
    code_before TEXT,
    code_after TEXT
);
"""


def _write_cwe_mapping(path):
    """Write a minimal CVE→CWE mapping file."""
    mapping = {
        "CVE-2024-0001": "CWE-89",
        "CVE-2024-0002": "CWE-79",
        "CVE-2024-0003": "CWE-999",  # out of scope
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)


@pytest.fixture
def reduced_setup(tmp_path):
    """Create a reduced-schema DB and a CWE mapping file."""
    db_path = tmp_path / "CVEfixes_reduced.db"
    cwe_path = tmp_path / "cve_cwe_mapping.json"
    con = sqlite3.connect(str(db_path))
    con.executescript(REDUCED_SCHEMA)

    # In-scope CWE-89 (Python)
    con.execute("INSERT INTO fixes VALUES ('CVE-2024-0001', 'sha1', 'https://github.com/acme/app')")
    con.execute("INSERT INTO commits VALUES ('sha1', 'commit message')")
    con.execute(
        "INSERT INTO file_change VALUES ('fc1', 'sha1', 'Python', ?, ?)",
        (CODE_BEFORE_89, CODE_AFTER_89),
    )

    # In-scope CWE-79 (JavaScript)
    con.execute("INSERT INTO fixes VALUES ('CVE-2024-0002', 'sha2', 'https://github.com/acme/web')")
    con.execute("INSERT INTO commits VALUES ('sha2', 'commit message 2')")
    con.execute(
        "INSERT INTO file_change VALUES ('fc2', 'sha2', 'JavaScript', ?, ?)",
        (CODE_BEFORE_79, CODE_AFTER_79),
    )

    # Out-of-scope CWE-999
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-0003', 'sha3', 'https://github.com/acme/other')"
    )
    con.execute("INSERT INTO commits VALUES ('sha3', 'commit message 3')")
    con.execute(
        "INSERT INTO file_change VALUES ('fc3', 'sha3', 'Python', ?, ?)",
        (CODE_BEFORE_999, "CODE_AFTER_999"),
    )

    con.commit()
    con.close()
    _write_cwe_mapping(str(cwe_path))

    return db_path, cwe_path


class TestReducedCveFixesLoaderInit:
    def test_missing_db_raises(self, tmp_path):
        cwe_path = tmp_path / "cve_cwe_mapping.json"
        _write_cwe_mapping(str(cwe_path))
        with pytest.raises(FileNotFoundError, match="CVEfixes.db not found"):
            ReducedCveFixesLoader(tmp_path / "nonexistent.db", str(cwe_path))

    def test_missing_cwe_mapping_raises(self, tmp_path):
        db_path = tmp_path / "CVEfixes.db"
        # Create a minimal DB so the db check passes
        sqlite3.connect(str(db_path)).close()
        with pytest.raises(FileNotFoundError, match="CWE mapping file not found"):
            ReducedCveFixesLoader(db_path, str(tmp_path / "nonexistent.json"))

    def test_successful_init(self, reduced_setup):
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        assert loader.db_path == db_path
        assert loader._cwe_mapping == {
            "CVE-2024-0001": "CWE-89",
            "CVE-2024-0002": "CWE-79",
            "CVE-2024-0003": "CWE-999",
        }

    def test_default_cwe_mapping_path(self, reduced_setup):
        """The default cwe_mapping_path is 'data/cve_cwe_mapping.json'."""
        db_path, _ = reduced_setup
        # The default path 'data/cve_cwe_mapping.json' exists in the repo root,
        # so init should succeed with the default path.
        loader = ReducedCveFixesLoader(db_path)
        assert loader._cwe_mapping is not None
        assert isinstance(loader._cwe_mapping, dict)


# ---------------------------------------------------------------------------
# _load_cwe_mapping
# ---------------------------------------------------------------------------


class TestLoadCweMapping:
    def test_loads_valid_json(self, tmp_path):
        cwe_path = tmp_path / "mapping.json"
        _write_cwe_mapping(str(cwe_path))
        db_path = tmp_path / "CVEfixes.db"
        sqlite3.connect(str(db_path)).close()
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        assert len(loader._cwe_mapping) == 3
        assert loader._cwe_mapping["CVE-2024-0001"] == "CWE-89"


# ---------------------------------------------------------------------------
# load_pairs
# ---------------------------------------------------------------------------


class TestLoadPairs:
    def test_load_all_pairs(self, reduced_setup):
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs()

        # CVE-2024-0003 (CWE-999) is out of scope → 2 pairs expected.
        assert len(pairs) == 2
        pair_ids = {p.cve_id for p in pairs}
        assert pair_ids == {"CVE-2024-0001", "CVE-2024-0002"}

    def test_pair_fields(self, reduced_setup):
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs()

        pair = pairs[0]
        assert pair.cve_id == "CVE-2024-0001"
        assert pair.cwe_id == "CWE-89"
        assert pair.repo_name == "acme/app"
        assert pair.commit_sha == "sha1"
        assert pair.language == "python"
        assert pair.vulnerable_code == CODE_BEFORE_89
        assert pair.fixed_code == CODE_AFTER_89
        assert pair.granularity == "file"

    def test_repo_name_derived_from_url(self, reduced_setup):
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs()

        web_pair = [p for p in pairs if p.cve_id == "CVE-2024-0002"][0]
        assert web_pair.repo_name == "acme/web"

    def test_language_filter(self, reduced_setup):
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs(languages={"Python"})
        assert all(p.language == "python" for p in pairs)
        assert len(pairs) == 1
        assert pairs[0].cve_id == "CVE-2024-0001"

    def test_default_languages(self, reduced_setup):
        """When no languages passed, defaults to Python + JS + TS."""
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs()
        langs = {p.language for p in pairs}
        assert langs == {"python", "javascript"}

    def test_missing_code_before_filtered_out(self, reduced_setup):
        """Rows with code_before IS NULL should not appear."""
        db_path, cwe_path = reduced_setup
        con = sqlite3.connect(str(db_path))
        con.execute(
            "INSERT INTO fixes VALUES ('CVE-2024-0004', 'sha4', 'https://github.com/acme/null')"
        )
        con.execute("INSERT INTO commits VALUES ('sha4', 'msg')")
        con.execute(
            "INSERT INTO file_change VALUES ('fc4', 'sha4', 'Python', NULL, ?)",
            (CODE_BEFORE_89,),
        )
        con.commit()
        con.close()

        # Add to mapping
        mapping = {"CVE-2024-0001": "CWE-89", "CVE-2024-0004": "CWE-89"}
        cwe_path.write_text(json.dumps(mapping), encoding="utf-8")

        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs()
        # CVE-2024-0004 should be filtered out (code_before IS NULL)
        assert "CVE-2024-0004" not in {p.cve_id for p in pairs}

    def test_short_code_before_filtered_out(self, reduced_setup):
        """code_before with LENGTH <= 50 should be filtered out."""
        db_path, cwe_path = reduced_setup
        con = sqlite3.connect(str(db_path))
        con.execute(
            "INSERT INTO fixes VALUES ('CVE-2024-0005', 'sha5', 'https://github.com/acme/short')"
        )
        con.execute("INSERT INTO commits VALUES ('sha5', 'msg')")
        con.execute(
            "INSERT INTO file_change VALUES ('fc5', 'sha5', 'Python', 'x' * 30, ?)",
            (CODE_BEFORE_89,),
        )
        con.commit()
        con.close()

        mapping = {"CVE-2024-0001": "CWE-89", "CVE-2024-0005": "CWE-89"}
        cwe_path.write_text(json.dumps(mapping), encoding="utf-8")

        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs()
        # CVE-2024-0005 should be filtered out (code_before too short)
        assert "CVE-2024-0005" not in {p.cve_id for p in pairs}

    def test_cwe_mapping_only_in_scope_cwes(self, reduced_setup):
        """Only CWEs in CWE_SCOPE appear in temp tables."""
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        pairs = loader.load_pairs()

        # CWE-999 is out of scope — its pair should not appear.
        for p in pairs:
            assert p.cwe_id != "CWE-999"

    def test_query_returns_empty_when_no_matches(self, reduced_setup):
        db_path = reduced_setup[0]
        # Use a mapping with only out-of-scope CWEs
        bad_mapping_path = tmp_path_str(reduced_setup, "bad_mapping.json")
        with open(bad_mapping_path, "w") as f:
            json.dump({"CVE-2024-0001": "CWE-999"}, f)

        loader = ReducedCveFixesLoader(db_path, bad_mapping_path)
        pairs = loader.load_pairs()
        assert pairs == []


def tmp_path_str(reduced_setup, filename):
    db_path, _ = reduced_setup
    return str(db_path.parent / filename)


# ---------------------------------------------------------------------------
# _filter_sql (returns a static SQL fragment — verify it's usable)
# ---------------------------------------------------------------------------


class TestFilterSql:
    def test_returns_filter_clause(self, reduced_setup):
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))
        sql = loader._filter_sql()
        assert "ic.cwe_id IN (SELECT cwe_id FROM temp.in_scope_cwe_list)" in sql


# ---------------------------------------------------------------------------
# _prepare_temp_tables
# ---------------------------------------------------------------------------


class TestPrepareTempTables:
    def test_temp_tables_created_and_populated(self, reduced_setup):
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        loader._prepare_temp_tables(con)

        # Check in_scope_cve_list has 2 entries (CVE-999 is excluded)
        rows = con.execute("SELECT cve_id, cwe_id FROM in_scope_cve_list").fetchall()
        cve_ids = {row["cve_id"] for row in rows}
        assert cve_ids == {"CVE-2024-0001", "CVE-2024-0002"}

        # Check in_scope_cwe_list has all 6 scope CWEs
        rows = con.execute("SELECT cwe_id FROM in_scope_cwe_list").fetchall()
        cwe_ids = {row["cwe_id"] for row in rows}
        assert "CWE-89" in cwe_ids
        assert "CWE-79" in cwe_ids
        assert "CWE-999" not in cwe_ids

        con.close()

    def test_temp_tables_are_idempotent(self, reduced_setup):
        """Calling _prepare_temp_tables twice doesn't duplicate rows."""
        db_path, cwe_path = reduced_setup
        loader = ReducedCveFixesLoader(db_path, str(cwe_path))

        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        loader._prepare_temp_tables(con)
        loader._prepare_temp_tables(con)

        count = con.execute("SELECT COUNT(*) as c FROM in_scope_cve_list").fetchone()["c"]
        # INSERT OR IGNORE prevents duplicates
        assert count == 2
        con.close()
