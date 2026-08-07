"""Stage 1 orchestration: raw CVEfixes pairs -> enriched VulnSample rows,
persisted to Postgres (metadata) + MinIO (full payload).

This intentionally does NOT run Semgrep or hit the NVD API inside a tight
loop without any error isolation: a single bad sample (Semgrep crash on
weird syntax, one CVE missing from NVD) must not abort the whole batch.
Failures are collected and reported, not silently swallowed and not fatal.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from app.data.collectors.cvefixes_loader import CveFixesLoader, RawVulnPair
from app.data.collectors.cwe_scope import cwe_spec
from app.data.collectors.nvd_client import NvdClient
from app.data.collectors.semgrep_runner import SemgrepUnavailableError, run_semgrep
from app.schemas.vuln import VulnSample
from app.storage import object_store
from app.storage.db import VulnSampleRow, get_session

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    samples: list[VulnSample]
    skipped: list[tuple[RawVulnPair, str]]  # (pair, reason)


def build_vuln_sample(
    pair: RawVulnPair,
    nvd_client: NvdClient,
    run_static_analysis: bool = True,
) -> VulnSample:
    """Turn one raw CVEfixes pair into a fully enriched VulnSample.

    Raises on hard failures (bad NVD response, unsupported language) so the
    caller (`run_pipeline`) can decide whether to skip-and-log or abort.
    """
    spec = cwe_spec(pair.cwe_id)
    if spec is None:
        raise ValueError(f"{pair.cwe_id} is out of scope — should have been filtered earlier")

    enrichment = nvd_client.fetch(pair.cve_id)

    static_findings = []
    if run_static_analysis:
        try:
            static_findings = run_semgrep(pair.vulnerable_code, pair.language)
        except SemgrepUnavailableError:
            logger.warning("Semgrep not available — proceeding with static_findings=[]")
        except Exception as exc:  # noqa: BLE001 — one bad sample shouldn't kill the batch
            logger.warning("Semgrep failed for %s@%s: %s", pair.cve_id, pair.commit_sha, exc)

    return VulnSample(
        id=str(uuid.uuid4()),
        source="cve_real",
        repo_name=pair.repo_name,
        commit_sha=pair.commit_sha,
        cve_id=pair.cve_id,
        cwe_id=pair.cwe_id,
        severity=enrichment.severity,
        language=pair.language,
        vulnerable_code=pair.vulnerable_code,
        fixed_code=pair.fixed_code,
        static_findings=static_findings,
        description=enrichment.description or f"{pair.cwe_id} in {pair.repo_name}",
        split=None,  # assigned in Stage 2
    )


def persist(sample: VulnSample) -> None:
    """Write the full sample to MinIO, and its metadata to Postgres."""
    key = f"vuln_samples/{sample.cwe_id}/{sample.id}.json"
    uri = object_store.put_json(key, sample.model_dump())

    session = get_session()
    try:
        row = VulnSampleRow(
            id=sample.id,
            source=sample.source,
            repo_name=sample.repo_name,
            commit_sha=sample.commit_sha,
            cve_id=sample.cve_id,
            cwe_id=sample.cwe_id,
            severity=sample.severity,
            language=sample.language,
            description=sample.description,
            static_findings=[f.model_dump() for f in sample.static_findings],
            split=sample.split,
            object_store_key=key,
        )
        session.merge(row)
        session.commit()
    finally:
        session.close()

    return uri


def run_pipeline(
    db_path: str,
    languages: set[str] | None = None,
    nvd_client: NvdClient | None = None,
    run_static_analysis: bool = True,
    dry_run: bool = False,
) -> PipelineResult:
    loader = CveFixesLoader(db_path)
    nvd_client = nvd_client or NvdClient()

    pairs = loader.load_pairs(languages=languages)
    logger.info("Loaded %d in-scope raw pairs from CVEfixes", len(pairs))

    samples: list[VulnSample] = []
    skipped: list[tuple[RawVulnPair, str]] = []

    for pair in pairs:
        try:
            sample = build_vuln_sample(pair, nvd_client, run_static_analysis=run_static_analysis)
        except Exception as exc:  # noqa: BLE001
            skipped.append((pair, str(exc)))
            continue

        samples.append(sample)
        if not dry_run:
            persist(sample)

    logger.info("Built %d VulnSample records, skipped %d", len(samples), len(skipped))
    return PipelineResult(samples=samples, skipped=skipped)
