"""Integration tests for Stage 1 end-to-end pipeline.

These tests exercise the full ``run_pipeline()`` orchestration — CVEfixes DB
loading → CWE scope filtering → NVD enrichment → sample construction — against
a synthetic SQLite DB built with the *real* CVEfixes v1.0.8 schema.

Unlike unit tests (which mock ``build_vuln_sample`` at the function boundary),
these tests let ``run_pipeline`` drive the loader, the mock NVD client, and
the sample-construction logic together, then validate the aggregate result:

  - In-scope CWEs produce VulnSample records with enriched severity/description.
  - Out-of-scope CWEs are silently filtered (no skipped entry, just absent).
  - Samples that fail enrichment (e.g. NVD lookup raises) land in the skipped
    list without aborting the batch.
  - ``dry_run=True`` skips persistence entirely.
  - ``dry_run=False`` calls ``persist`` once per successful sample.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.data.collectors.nvd_client import NvdEnrichment
from app.data.collectors.pipeline import PipelineResult, run_pipeline
from app.schemas.vuln import VulnSample

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


def _build_cvefixes_db(db_path: Path) -> Path:
    """Create a synthetic CVEfixes.db using the real v1.0.8 schema.

    Inserts four CVEs:
      - CVE-2024-1001 / CWE-89  (in-scope, method-level)        → should succeed
      - CVE-2024-1002 / CWE-79  (in-scope, file-level fallback)  → should succeed
      - CVE-2024-1003 / CWE-999 (out-of-scope)                   → filtered by loader
      - CVE-2024-1004 / CWE-22  (in-scope, will fail NVD fetch) → should be skipped
    """
    con = sqlite3.connect(str(db_path))
    con.executescript(SCHEMA)

    # --- CVE-2024-1001 / CWE-89 (in-scope, method-level) ---
    con.execute("INSERT INTO cve VALUES ('CVE-2024-1001', 'SQLi bug', 'HIGH')")
    con.execute("INSERT INTO cwe_classification VALUES ('CVE-2024-1001', 'CWE-89')")
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-1001', 'sha_1', 'https://github.com/acme/app')"
    )
    con.execute("INSERT INTO repository VALUES ('https://github.com/acme/app', 'acme/app')")
    con.execute(
        "INSERT INTO file_change VALUES "
        "('fc_1', 'sha_1', 'python', 'FILE_BEFORE', 'FILE_AFTER')"
    )
    con.execute(
        "INSERT INTO method_change VALUES ('mc_1', 'fc_1', 'get_user', 'VULN CODE', 1)"
    )
    con.execute(
        "INSERT INTO method_change VALUES ('mc_2', 'fc_1', 'get_user', 'FIXED CODE', 0)"
    )

    # --- CVE-2024-1002 / CWE-79 (in-scope, file-level fallback) ---
    con.execute("INSERT INTO cve VALUES ('CVE-2024-1002', 'XSS bug', 'MEDIUM')")
    con.execute("INSERT INTO cwe_classification VALUES ('CVE-2024-1002', 'CWE-79')")
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-1002', 'sha_2', 'https://github.com/acme/web')"
    )
    con.execute("INSERT INTO repository VALUES ('https://github.com/acme/web', 'acme/web')")
    con.execute(
        "INSERT INTO file_change VALUES ('fc_2', 'sha_2', 'javascript', 'JS_BEFORE', 'JS_AFTER')"
    )

    # --- CVE-2024-1003 / CWE-999 (out-of-scope — loader should exclude it) ---
    con.execute("INSERT INTO cve VALUES ('CVE-2024-1003', 'Unrelated bug', 'LOW')")
    con.execute("INSERT INTO cwe_classification VALUES ('CVE-2024-1003', 'CWE-999')")
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-1003', 'sha_3', 'https://github.com/acme/other')"
    )
    con.execute("INSERT INTO repository VALUES ('https://github.com/acme/other', 'acme/other')")
    con.execute(
        "INSERT INTO file_change VALUES ('fc_3', 'sha_3', 'python', 'X', 'Y')"
    )

    # --- CVE-2024-1004 / CWE-22 (in-scope, but NVD fetch will fail) ---
    con.execute("INSERT INTO cve VALUES ('CVE-2024-1004', 'Path traversal bug', 'HIGH')")
    con.execute("INSERT INTO cwe_classification VALUES ('CVE-2024-1004', 'CWE-22')")
    con.execute(
        "INSERT INTO fixes VALUES ('CVE-2024-1004', 'sha_4', 'https://github.com/acme/fs')"
    )
    con.execute("INSERT INTO repository VALUES ('https://github.com/acme/fs', 'acme/fs')")
    con.execute(
        "INSERT INTO file_change VALUES "
        "('fc_4', 'sha_4', 'python', 'TRAVERSE_BEFORE', 'TRAVERSE_AFTER')"
    )
    con.execute(
        "INSERT INTO method_change VALUES ('mc_4a', 'fc_4', 'read_file', 'OPEN FILE', 1)"
    )
    con.execute(
        "INSERT INTO method_change VALUES ('mc_4b', 'fc_4', 'read_file', 'SAFE READ', 0)"
    )

    con.commit()
    con.close()
    return db_path


@pytest.fixture
def cvefixes_db(tmp_path):
    return _build_cvefixes_db(tmp_path / "CVEfixes.db")


class _MockNvdClient:
    """Mock NVD client that returns canned enrichments or raises.

    The ``failures`` set contains CVE IDs whose ``fetch()`` should raise,
    simulating a sample that fails NVD enrichment inside ``build_vuln_sample``.
    """

    def __init__(self, enrichments, failures=None):
        self._enrichments = enrichments
        self._failures = failures or set()
        self.fetch_calls = []

    def fetch(self, cve_id: str, max_retries: int = 3) -> NvdEnrichment:
        self.fetch_calls.append(cve_id)
        if cve_id in self._failures:
            raise RuntimeError(f"NVD lookup failed for {cve_id}")
        return self._enrichments[cve_id]


# --- Helpers ---


def _nvd_enrichment(cve_id: str, severity: str, description: str, score: float | None = 7.5):
    return NvdEnrichment(
        cve_id=cve_id,
        severity=severity,
        description=description,
        cvss_score=score,
    )


def _mock_persist():
    """Patch ``persist`` in pipeline.py so no real Postgres/MinIO is needed."""
    return patch("app.data.collectors.pipeline.persist", MagicMock())


# --- Tests ---


def test_pipeline_dry_run_produces_correct_samples(cvefixes_db):
    """Full pipeline with dry_run=True: produces enriched VulnSamples, no persist."""
    enrichments = {
        "CVE-2024-1001": _nvd_enrichment(
            "CVE-2024-1001", "high", "SQL injection via string concat."
        ),
        "CVE-2024-1002": _nvd_enrichment(
            "CVE-2024-1002", "medium", "Reflected XSS in template."
        ),
        "CVE-2024-1004": _nvd_enrichment(
            "CVE-2024-1004", "high", "Path traversal via user input."
        ),
    }
    client = _MockNvdClient(enrichments)

    with _mock_persist() as mock_persist:
        result = run_pipeline(
            str(cvefixes_db),
            nvd_client=client,
            run_static_analysis=False,
            dry_run=True,
        )

    # 3 in-scope samples → 3 VulnSample records (CVE-2024-1003 is out-of-scope, filtered by loader)
    assert len(result.samples) == 3
    assert len(result.skipped) == 0

    # Persist should NOT be called in dry-run mode
    mock_persist.assert_not_called()

    # Validate each sample
    by_cve = {s.cve_id: s for s in result.samples}
    assert set(by_cve.keys()) == {"CVE-2024-1001", "CVE-2024-1002", "CVE-2024-1004"}

    # CWE-89 sample (method-level)
    s1 = by_cve["CVE-2024-1001"]
    assert isinstance(s1, VulnSample)
    assert s1.cwe_id == "CWE-89"
    assert s1.severity == "high"  # from NVD enrichment
    assert s1.description == "SQL injection via string concat."
    assert s1.vulnerable_code == "VULN CODE"
    assert s1.fixed_code == "FIXED CODE"
    assert s1.repo_name == "acme/app"
    assert s1.commit_sha == "sha_1"
    assert s1.source == "cve_real"
    assert s1.language == "python"
    assert s1.split is None  # assigned in Stage 2

    # CWE-79 sample (file-level fallback)
    s2 = by_cve["CVE-2024-1002"]
    assert s2.cwe_id == "CWE-79"
    assert s2.severity == "medium"
    assert s2.description == "Reflected XSS in template."
    assert s2.vulnerable_code == "JS_BEFORE"
    assert s2.fixed_code == "JS_AFTER"
    assert s2.repo_name == "acme/web"
    assert s2.language == "javascript"

    # CWE-22 sample
    s4 = by_cve["CVE-2024-1004"]
    assert s4.cwe_id == "CWE-22"
    assert s4.severity == "high"
    assert s4.vulnerable_code == "OPEN FILE"
    assert s4.fixed_code == "SAFE READ"

    # Verify NVD client was called exactly for the in-scope samples
    assert set(client.fetch_calls) == {"CVE-2024-1001", "CVE-2024-1002", "CVE-2024-1004"}


def test_pipeline_excludes_out_of_scope_cwe(cvefixes_db):
    """CVE-2024-1003 (CWE-999) must never appear in samples or skipped."""
    enrichments = {
        "CVE-2024-1001": _nvd_enrichment("CVE-2024-1001", "high", "desc"),
        "CVE-2024-1002": _nvd_enrichment("CVE-2024-1002", "medium", "desc"),
        "CVE-2024-1004": _nvd_enrichment("CVE-2024-1004", "high", "desc"),
    }
    client = _MockNvdClient(enrichments)

    result = run_pipeline(
        str(cvefixes_db),
        nvd_client=client,
        run_static_analysis=False,
        dry_run=True,
    )

    # CWE-999 was filtered by the loader → not in samples, not in skipped
    assert all(s.cwe_id != "CWE-999" for s in result.samples)
    assert all(s.cve_id != "CVE-2024-1003" for s in result.samples)
    assert all("CVE-2024-1003" not in str(reason) for _, reason in result.skipped)
    # The loader itself should have loaded only 3 in-scope pairs
    assert len(result.samples) + len(result.skipped) == 3


def test_pipeline_skips_sample_on_nvd_failure(cvefixes_db):
    """When NVD fetch raises, the pipeline catches it, skips the sample, and
    continues processing the remaining pairs without aborting."""
    enrichments = {
        "CVE-2024-1001": _nvd_enrichment("CVE-2024-1001", "high", "desc"),
        "CVE-2024-1002": _nvd_enrichment("CVE-2024-1002", "medium", "desc"),
        "CVE-2024-1004": _nvd_enrichment("CVE-2024-1004", "high", "desc"),
    }
    # CVE-2024-1004 fails NVD fetch → should be skipped
    client = _MockNvdClient(enrichments, failures={"CVE-2024-1004"})

    with _mock_persist():
        result = run_pipeline(
            str(cvefixes_db),
            nvd_client=client,
            run_static_analysis=False,
            dry_run=True,
        )

    # 2 succeeded, 1 skipped
    assert len(result.samples) == 2
    assert len(result.skipped) == 1

    # The skipped entry should reference CVE-2024-1004
    skipped_pair, skip_reason = result.skipped[0]
    assert skipped_pair.cve_id == "CVE-2024-1004"
    assert "NVD lookup failed" in skip_reason

    # The other two samples have enriched data
    sample_cves = {s.cve_id for s in result.samples}
    assert sample_cves == {"CVE-2024-1001", "CVE-2024-1002"}


def test_pipeline_non_dry_run_calls_persist(cvefixes_db):
    """With dry_run=False, persist() must be called once per successful sample."""
    enrichments = {
        "CVE-2024-1001": _nvd_enrichment("CVE-2024-1001", "high", "desc"),
        "CVE-2024-1002": _nvd_enrichment("CVE-2024-1002", "medium", "desc"),
        "CVE-2024-1004": _nvd_enrichment("CVE-2024-1004", "high", "desc"),
    }
    client = _MockNvdClient(enrichments)

    with _mock_persist() as mock_persist:
        result = run_pipeline(
            str(cvefixes_db),
            nvd_client=client,
            run_static_analysis=False,
            dry_run=False,
        )

    # persist should have been called once per sample (3 successful, 0 skipped)
    assert mock_persist.call_count == 3
    # Each call receives a VulnSample
    for call in mock_persist.call_args_list:
        args = call.args
        assert len(args) >= 1
        assert isinstance(args[0], VulnSample)

    # The returned result should still have the same samples
    assert len(result.samples) == 3


def test_pipeline_language_filter(cvefixes_db):
    """Passing languages={'python'} should exclude the JavaScript sample."""
    enrichments = {
        "CVE-2024-1001": _nvd_enrichment("CVE-2024-1001", "high", "desc"),
        "CVE-2024-1002": _nvd_enrichment("CVE-2024-1002", "medium", "desc"),
        "CVE-2024-1004": _nvd_enrichment("CVE-2024-1004", "high", "desc"),
    }
    client = _MockNvdClient(enrichments)

    with _mock_persist():
        result = run_pipeline(
            str(cvefixes_db),
            languages={"python"},
            nvd_client=client,
            run_static_analysis=False,
            dry_run=True,
        )

    # CVE-2024-1002 was javascript → filtered out
    sample_cves = {s.cve_id for s in result.samples}
    assert "CVE-2024-1002" not in sample_cves
    assert len(result.samples) == 2  # CWE-89 and CWE-22 remain
    assert all(s.language == "python" for s in result.samples)


def test_pipeline_returns_pipeline_result_type(cvefixes_db):
    """The return value should be a PipelineResult with samples and skipped lists."""
    enrichments = {
        "CVE-2024-1001": _nvd_enrichment("CVE-2024-1001", "high", "desc"),
        "CVE-2024-1002": _nvd_enrichment("CVE-2024-1002", "medium", "desc"),
        "CVE-2024-1004": _nvd_enrichment("CVE-2024-1004", "high", "desc"),
    }
    client = _MockNvdClient(enrichments)

    result = run_pipeline(
        str(cvefixes_db),
        nvd_client=client,
        run_static_analysis=False,
        dry_run=True,
    )

    assert isinstance(result, PipelineResult)
    assert isinstance(result.samples, list)
    assert isinstance(result.skipped, list)


def test_pipeline_description_fallback_when_nvd_description_empty(cvefixes_db):
    """If NVD returns an empty description, build_vuln_sample generates a
    fallback description from the CWE ID and repo name."""
    enrichments = {
        "CVE-2024-1001": _nvd_enrichment("CVE-2024-1001", "high", description=""),
        "CVE-2024-1002": _nvd_enrichment("CVE-2024-1002", "medium", "desc"),
        "CVE-2024-1004": _nvd_enrichment("CVE-2024-1004", "high", "desc"),
    }
    client = _MockNvdClient(enrichments)

    with _mock_persist():
        result = run_pipeline(
            str(cvefixes_db),
            nvd_client=client,
            run_static_analysis=False,
            dry_run=True,
        )

    s1 = next(s for s in result.samples if s.cve_id == "CVE-2024-1001")
    assert s1.description == "CWE-89 in acme/app"
