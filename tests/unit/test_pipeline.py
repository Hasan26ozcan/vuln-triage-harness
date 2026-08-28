"""Tests for Stage 1 data-collection pipeline (app/data/collectors/pipeline.py).

Covers build_vuln_sample() (with and without static analysis), persist(),
and run_pipeline() — all with mocked NVD client, Semgrep, Postgres, and MinIO.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.data.collectors.cvefixes_loader import RawVulnPair
from app.data.collectors.nvd_client import NvdEnrichment
from app.data.collectors.pipeline import (
    build_vuln_sample,
    persist,
    run_pipeline,
)
from app.data.collectors.semgrep_runner import SemgrepUnavailableError
from app.schemas.vuln import StaticFinding, VulnSample


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


# --- build_vuln_sample with static analysis ---


def test_build_vuln_sample_with_static_analysis_includes_findings():
    """When run_static_analysis=True, Semgrep findings are attached."""
    pair = _sample_pair()
    client = _fake_nvd_client()

    findings = [
        StaticFinding(
            tool="semgrep", rule_id="python.sqli", message="Possible SQLi", line_range=(3, 5)
        )
    ]

    with patch("app.data.collectors.pipeline.run_semgrep", return_value=findings) as mock_semgrep:
        sample = build_vuln_sample(pair, client, run_static_analysis=True)

    mock_semgrep.assert_called_once_with(pair.vulnerable_code, pair.language)
    assert sample.static_findings == findings


def test_build_vuln_sample_handles_semgrep_unavailable():
    """SemgrepUnavailableError is caught; static_findings remains empty."""
    pair = _sample_pair()
    client = _fake_nvd_client()

    with patch(
        "app.data.collectors.pipeline.run_semgrep",
        side_effect=SemgrepUnavailableError("no semgrep"),
    ):
        sample = build_vuln_sample(pair, client, run_static_analysis=True)

    assert sample.static_findings == []


def test_build_vuln_sample_handles_semgrep_crash():
    """A generic Semgrep exception is caught; static_findings remains empty."""
    pair = _sample_pair()
    client = _fake_nvd_client()

    with patch(
        "app.data.collectors.pipeline.run_semgrep",
        side_effect=RuntimeError("semgrep crashed"),
    ):
        sample = build_vuln_sample(pair, client, run_static_analysis=True)

    assert sample.static_findings == []


# --- persist ---


@patch("app.data.collectors.pipeline.get_session")
@patch("app.data.collectors.pipeline.object_store")
def test_persist_writes_to_minio_and_postgres(mock_object_store, mock_get_session):
    """persist() writes payload to MinIO and metadata to Postgres."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session
    mock_object_store.put_json.return_value = "s3://vuln-triage/vuln_samples/CWE-89/abc.json"

    sample = VulnSample(
        id="abc",
        source="cve_real",
        repo_name="acme/app",
        commit_sha="sha1",
        cve_id="CVE-2024-0001",
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code="bad",
        description="d",
    )

    uri = persist(sample)  # type: ignore[func-returns-value]

    mock_object_store.put_json.assert_called_once()
    mock_session.merge.assert_called_once()
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()
    assert uri == "s3://vuln-triage/vuln_samples/CWE-89/abc.json"


# --- run_pipeline ---


@patch("app.data.collectors.pipeline.persist")
@patch("app.data.collectors.pipeline.build_vuln_sample")
@patch("app.data.collectors.pipeline.CveFixesLoader")
def test_run_pipeline_dry_run_skips_persist(mock_loader_cls, mock_build, mock_persist):
    """With dry_run=True, persist() is not called."""
    pair = _sample_pair()
    mock_loader = MagicMock()
    mock_loader.load_pairs.return_value = [pair]
    mock_loader_cls.return_value = mock_loader

    sample = VulnSample(
        id="abc",
        source="cve_real",
        repo_name="acme/app",
        commit_sha="sha1",
        cve_id="CVE-2024-0001",
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code="bad",
        description="d",
    )
    mock_build.return_value = sample

    client = _fake_nvd_client()
    result = run_pipeline("dummy.db", nvd_client=client, run_static_analysis=False, dry_run=True)

    assert len(result.samples) == 1
    assert len(result.skipped) == 0
    mock_persist.assert_not_called()


@patch("app.data.collectors.pipeline.persist")
@patch("app.data.collectors.pipeline.build_vuln_sample")
@patch("app.data.collectors.pipeline.CveFixesLoader")
def test_run_pipeline_normal_run_persists(mock_loader_cls, mock_build, mock_persist):
    """Without dry_run, persist() is called for each sample."""
    pair = _sample_pair()
    mock_loader = MagicMock()
    mock_loader.load_pairs.return_value = [pair]
    mock_loader_cls.return_value = mock_loader

    sample = VulnSample(
        id="abc",
        source="cve_real",
        repo_name="acme/app",
        commit_sha="sha1",
        cve_id="CVE-2024-0001",
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code="bad",
        description="d",
    )
    mock_build.return_value = sample

    client = _fake_nvd_client()
    result = run_pipeline("dummy.db", nvd_client=client, run_static_analysis=False, dry_run=False)

    assert len(result.samples) == 1
    mock_persist.assert_called_once_with(sample)


@patch("app.data.collectors.pipeline.persist")
@patch("app.data.collectors.pipeline.build_vuln_sample")
@patch("app.data.collectors.pipeline.CveFixesLoader")
def test_run_pipeline_skips_failing_pairs(mock_loader_cls, mock_build, mock_persist):
    """Pairs that raise during build_vuln_sample are added to skipped."""
    pair_ok = _sample_pair()
    pair_bad = _sample_pair(cve_id="CVE-2024-BAD")
    mock_loader = MagicMock()
    mock_loader.load_pairs.return_value = [pair_ok, pair_bad]
    mock_loader_cls.return_value = mock_loader

    good_sample = VulnSample(
        id="abc",
        source="cve_real",
        repo_name="acme/app",
        commit_sha="sha1",
        cve_id="CVE-2024-0001",
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code="bad",
        description="d",
    )

    mock_build.side_effect = [good_sample, ValueError("NVD lookup failed")]

    client = _fake_nvd_client()
    result = run_pipeline("dummy.db", nvd_client=client, run_static_analysis=False, dry_run=True)

    assert len(result.samples) == 1
    assert len(result.skipped) == 1
    assert result.skipped[0][0].cve_id == "CVE-2024-BAD"


@patch("app.data.collectors.pipeline.CveFixesLoader")
def test_run_pipeline_empty_pairs(mock_loader_cls):
    """When no pairs are loaded, the result is empty."""
    mock_loader = MagicMock()
    mock_loader.load_pairs.return_value = []
    mock_loader_cls.return_value = mock_loader

    client = _fake_nvd_client()
    result = run_pipeline("dummy.db", nvd_client=client, dry_run=True)

    assert len(result.samples) == 0
    assert len(result.skipped) == 0


@patch("app.data.collectors.pipeline.CveFixesLoader")
def test_run_pipeline_passes_language_filter(mock_loader_cls):
    """The languages filter is forwarded to the loader."""
    mock_loader = MagicMock()
    mock_loader.load_pairs.return_value = []
    mock_loader_cls.return_value = mock_loader

    client = _fake_nvd_client()
    run_pipeline("dummy.db", languages={"python", "javascript"}, nvd_client=client, dry_run=True)

    mock_loader.load_pairs.assert_called_once_with(languages={"python", "javascript"})
