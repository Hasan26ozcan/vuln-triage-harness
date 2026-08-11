"""Stage 10 — security-scan output parsers (Gitleaks, Trivy).

Provides helper functions to parse JSON output from ``gitleaks detect``
and ``trivy fs`` so the CI workflow can produce structured
``SecurityScanSummary`` artifacts alongside the regression gate result.

Both parsers are intentionally defensive: they tolerate truncated or
empty output and always return a valid ``SecurityScanSummary``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.schemas.ci import GateStatus, SecurityScanSummary

logger = logging.getLogger(__name__)


def _resolve_raw(raw: str | Path | None) -> str:
    """Resolve *raw* into a string of JSON content.

    Accepts ``None`` (→ empty string), a ``Path``, or a plain string. If
    the string looks like a path to an existing file, the file contents
    are read; otherwise the string is returned as-is (treated as raw JSON).
    """
    if raw is None:
        return ""
    if isinstance(raw, Path):
        return raw.read_text(encoding="utf-8") if raw.exists() else ""
    # plain string — could be JSON content or a file path
    if "\n" not in raw and raw.endswith(".json") and Path(raw).exists():
        return Path(raw).read_text(encoding="utf-8")
    return raw


# ---------------------------------------------------------------------------
# Public parsers
# ---------------------------------------------------------------------------


def parse_gitleaks_output(raw: str | Path | None = None) -> SecurityScanSummary:
    """Parse Gitleaks JSON output (``gitleaks detect --report-format json``).

    Gitleaks writes a JSON array of finding objects, each with keys like
    ``rule``, ``line``, ``offender``, ``commit``, ``file``, ``severity``,
    etc. An empty array means no secrets were found.

    Parameters
    ----------
    raw:
        A JSON string, a path to a report file, a ``Path`` object, or
        ``None`` (treated as zero findings).
    """
    text = _resolve_raw(raw)
    findings = _parse_gitleaks_json(text)

    severity_counts = _count_severities(findings, "severity")
    status = GateStatus.FAIL if findings else GateStatus.PASS

    return SecurityScanSummary(
        tool="gitleaks",
        status=status,
        findings_count=len(findings),
        severity_counts=severity_counts,
        details=findings[:50],  # cap stored details to keep CI artifacts small
    )


def parse_trivy_output(raw: str | Path | None = None) -> SecurityScanSummary:
    """Parse Trivy JSON output (``trivy fs . --format json``).

    Trivy's JSON has a top-level ``Results`` array, each element containing
    ``Target``, ``Class``, ``Type``, and a ``Vulnerabilities`` or
    ``Misconfigurations`` list. Each finding may carry a ``Severity`` field.

    Parameters
    ----------
    raw:
        A JSON string, a path to a report file, a ``Path`` object, or
        ``None`` (treated as zero findings).
    """
    text = _resolve_raw(raw)
    findings = _parse_trivy_json(text)

    severity_counts = _count_severities(findings, "Severity")
    status = GateStatus.FAIL if findings else GateStatus.PASS

    return SecurityScanSummary(
        tool="trivy",
        status=status,
        findings_count=len(findings),
        severity_counts=severity_counts,
        details=findings[:50],
    )


# ---------------------------------------------------------------------------
# Internal JSON parsers
# ---------------------------------------------------------------------------


def _parse_gitleaks_json(text: str) -> list[dict]:
    """Parse a Gitleaks JSON report string into a list of finding dicts."""
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Gitleaks output was not valid JSON — treating as 0 findings")
        return []
    if isinstance(data, list):
        return [f for f in data if isinstance(f, dict)]
    if isinstance(data, dict) and "findings" in data:
        return [f for f in data["findings"] if isinstance(f, dict)]
    return []


def _parse_trivy_json(text: str) -> list[dict]:
    """Parse a Trivy JSON report string into a flat list of finding dicts."""
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Trivy output was not valid JSON — treating as 0 findings")
        return []
    findings: list[dict] = []
    if isinstance(data, dict):
        results = data.get("Results", [])
        for entry in results:
            if not isinstance(entry, dict):
                continue
            # Trivy nests findings under vulnerability-type keys.
            for key in ("Vulnerabilities", "Misconfigurations", "Secrets", "Licenses"):
                for f in entry.get(key, []):
                    if isinstance(f, dict):
                        # Annotate with the target so it's traceable.
                        f = {**f, "target": entry.get("Target", "unknown")}
                        findings.append(f)
    return findings


def _count_severities(findings: list[dict], severity_key: str) -> dict[str, int]:
    """Count findings by their severity field.

    Different tools use different field names: Gitleaks uses ``severity``
    (lowercase), Trivy uses ``Severity``. Some findings may not have a
    severity at all — those are bucketed under ``"UNKNOWN"``.
    """
    counts: dict[str, int] = {}
    for f in findings:
        sev = f.get(severity_key, f.get("severity", f.get("Severity", "UNKNOWN")))
        if sev is None:
            sev = "UNKNOWN"
        sev = str(sev).upper()
        counts[sev] = counts.get(sev, 0) + 1
    return counts
