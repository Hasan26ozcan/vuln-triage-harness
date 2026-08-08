"""Tests build_vuln_sample() in isolation — no Postgres, no MinIO, no real
NVD or Semgrep calls. persist() and run_pipeline()'s I/O paths are
integration-test territory (tests/integration), not unit-test territory.
"""

from unittest.mock import MagicMock

import pytest

from app.data.collectors.cvefixes_loader import RawVulnPair
from app.data.collectors.nvd_client import NvdEnrichment
from app.data.collectors.pipeline import build_vuln_sample


def _fake_nvd_client(severity="high", description="A SQL injection bug."):
    client = MagicMock()
    client.fetch.return_value = NvdEnrichment(
        cve_id="CVE-2024-0001", severity=severity, description=description, cvss_score=7.5
    )
    return client


def _sample_pair(**overrides) -> RawVulnPair:
    defaults = dict(
        cve_id="CVE-2024-0001",
        cwe_id="CWE-89",
        repo_name="acme/app",
        commit_sha="abc123",
        language="python",
        vulnerable_code="cursor.execute('SELECT * FROM t WHERE id=' + x)",
        fixed_code="cursor.execute('SELECT * FROM t WHERE id=?', (x,))",
        granularity="method",
    )
    defaults.update(overrides)
    return RawVulnPair(**defaults)


def test_build_vuln_sample_happy_path():
    pair = _sample_pair()
    sample = build_vuln_sample(pair, _fake_nvd_client(), run_static_analysis=False)

    assert sample.source == "cve_real"
    assert sample.repo_name == "acme/app"
    assert sample.commit_sha == "abc123"
    assert sample.cve_id == "CVE-2024-0001"
    assert sample.cwe_id == "CWE-89"
    assert sample.severity == "high"
    assert sample.split is None  # Stage 2 hasn't run yet
    assert sample.static_findings == []


def test_build_vuln_sample_rejects_out_of_scope_cwe():
    pair = _sample_pair(cwe_id="CWE-999")
    with pytest.raises(ValueError):
        build_vuln_sample(pair, _fake_nvd_client(), run_static_analysis=False)


def test_build_vuln_sample_falls_back_to_generated_description_if_nvd_empty():
    pair = _sample_pair()
    client = _fake_nvd_client(description="")
    sample = build_vuln_sample(pair, client, run_static_analysis=False)
    assert sample.description == "CWE-89 in acme/app"
