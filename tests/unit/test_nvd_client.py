"""Tests the NVD client's parsing and severity-mapping logic against
canned responses via httpx.MockTransport — no real network call, and no
dependency on NVD's availability/rate limits during CI.
"""

import httpx
import pytest

from app.data.collectors.nvd_client import NvdClient, NvdRateLimiter, _cvss_to_severity


def _canned_nvd_response(cve_id: str, score: float | None, description: str) -> dict:
    metrics = {}
    if score is not None:
        metrics["cvssMetricV31"] = [{"cvssData": {"baseScore": score}}]
    return {
        "vulnerabilities": [
            {
                "cve": {
                    "id": cve_id,
                    "descriptions": [{"lang": "en", "value": description}],
                    "metrics": metrics,
                }
            }
        ]
    }


def _client_with_canned_response(payload: dict, status_code: int = 200) -> NvdClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    # requests_per_window=1000 -> effectively no rate-limit sleep in tests
    return NvdClient(client=http_client, rate_limiter=NvdRateLimiter(1000))


@pytest.mark.parametrize(
    "score,expected",
    [
        (None, "medium"),
        (2.0, "low"),
        (3.9, "low"),
        (4.0, "medium"),
        (6.9, "medium"),
        (7.0, "high"),
        (8.9, "high"),
        (9.0, "critical"),
        (10.0, "critical"),
    ],
)
def test_cvss_to_severity_thresholds(score, expected):
    assert _cvss_to_severity(score) == expected


def test_fetch_parses_description_and_severity():
    payload = _canned_nvd_response("CVE-2024-0001", 9.8, "Critical SQLi bug.")
    client = _client_with_canned_response(payload)

    result = client.fetch("CVE-2024-0001")

    assert result.cve_id == "CVE-2024-0001"
    assert result.severity == "critical"
    assert result.description == "Critical SQLi bug."
    assert result.cvss_score == 9.8


def test_fetch_defaults_to_medium_when_no_score():
    payload = _canned_nvd_response("CVE-2024-0002", None, "No CVSS available.")
    client = _client_with_canned_response(payload)

    result = client.fetch("CVE-2024-0002")

    assert result.severity == "medium"
    assert result.cvss_score is None


def test_fetch_raises_on_empty_vulnerabilities():
    client = _client_with_canned_response({"vulnerabilities": []})
    with pytest.raises(ValueError):
        client.fetch("CVE-2024-9999")
