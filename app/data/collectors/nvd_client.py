"""NVD REST API client (Stage 1, step 3: severity/description enrichment).

NVD rate-limits unauthenticated clients to ~5 requests / 30s, and API-key
holders to ~50 requests / 30s. We default to the conservative limit and
let the caller pass a key via `NVD_API_KEY` to go faster.

Docs: https://nvd.nist.gov/developers/vulnerabilities
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal

import httpx

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

Severity = Literal["low", "medium", "high", "critical"]


class NvdRateLimiter:
    """Simple sleep-based limiter, good enough for a batch job."""

    def __init__(self, requests_per_window: int, window_seconds: float = 30.0):
        self.min_interval = window_seconds / requests_per_window
        self._last_call: float = 0.0

    def wait(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()


@dataclass
class NvdEnrichment:
    cve_id: str
    severity: Severity
    description: str
    cvss_score: float | None


def _cvss_to_severity(score: float | None) -> Severity:
    """Map a CVSS v3.x base score to our four-bucket severity enum.

    Thresholds follow the standard NVD qualitative rating:
    0.1-3.9 low, 4.0-6.9 medium, 7.0-8.9 high, 9.0-10.0 critical.
    A missing score (rare, older CVEs) defaults to "medium" rather than
    silently dropping the sample — Stage 2 can filter it out later if
    that turns out to be too noisy.
    """
    if score is None:
        return "medium"
    if score >= 9.0:
        return "critical"
    if score >= 7.0:
        return "high"
    if score >= 4.0:
        return "medium"
    return "low"


class NvdClient:
    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        rate_limiter: NvdRateLimiter | None = None,
    ):
        self.api_key = api_key or os.environ.get("NVD_API_KEY")
        self._client = client or httpx.Client(timeout=30.0)
        requests_per_window = 50 if self.api_key else 5
        self._limiter = rate_limiter or NvdRateLimiter(requests_per_window)

    def fetch(self, cve_id: str, max_retries: int = 3) -> NvdEnrichment:
        headers = {"apiKey": self.api_key} if self.api_key else {}
        last_exc: Exception | None = None

        for attempt in range(max_retries):
            self._limiter.wait()
            try:
                resp = self._client.get(NVD_BASE_URL, params={"cveId": cve_id}, headers=headers)
                resp.raise_for_status()
                return self._parse(cve_id, resp.json())
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 429:
                    time.sleep(2**attempt)  # backoff on rate-limit
                    continue
                raise
            except httpx.RequestError as exc:
                last_exc = exc
                time.sleep(2**attempt)

        raise RuntimeError(
            f"NVD fetch failed for {cve_id} after {max_retries} retries"
        ) from last_exc

    @staticmethod
    def _parse(cve_id: str, payload: dict) -> NvdEnrichment:
        vulns = payload.get("vulnerabilities", [])
        if not vulns:
            raise ValueError(f"NVD returned no data for {cve_id}")

        cve = vulns[0]["cve"]

        description = ""
        for desc in cve.get("descriptions", []):
            if desc.get("lang") == "en":
                description = desc.get("value", "")
                break

        score = None
        metrics = cve.get("metrics", {})
        # Prefer CVSS v3.1, fall back to v3.0, then v2 (scaled 0-10 already).
        for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            if key in metrics and metrics[key]:
                score = metrics[key][0]["cvssData"]["baseScore"]
                break

        return NvdEnrichment(
            cve_id=cve_id,
            severity=_cvss_to_severity(score),
            description=description,
            cvss_score=score,
        )
