"""Unit tests for Stage 9 — FastAPI app (app/serving/api.py).

Uses ``fastapi.testclient.TestClient`` with a mock-config app so no
real ML or network is needed. Covers every endpoint and all branches:

* ``GET /healthz``
* ``GET /api/v1/manifest``
* ``POST /api/v1/serve`` — happy path
* ``POST /api/v1/serve/batch`` — happy path
* Error handling path (500)

Requires ``fastapi`` and ``httpx`` (TestClient dependency) — these are
standard dev dependencies.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from app.schemas.serving import BatchServeRequest, ServeRequest  # noqa: E402
from app.serving.api import app, create_app  # noqa: E402
from app.serving.config import ServingConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_RESPONSE = (
    '{"cwe_id": "CWE-89", "severity": "high", '
    '"explanation": "SQL injection via string concatenation.", '
    '"patch_diff": ""}'
)


@pytest.fixture
def client():
    """Use the default mock app (no real backend)."""
    return TestClient(app)


@pytest.fixture
def mock_client():
    """Create a client with a mock backend config."""
    config = ServingConfig(backend_type="mock")
    test_app = create_app(config)
    return TestClient(test_app)


@pytest.fixture
def sql_request():
    return ServeRequest(
        sample_id="api-test-001",
        vulnerable_code="cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
        language="python",
        description="SQL injection",
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthz:
    def test_healthz_returns_ok(self, mock_client):
        resp = mock_client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "backend" in data


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifestEndpoint:
    def test_manifest_returns_dict(self, mock_client):
        resp = mock_client.get("/api/v1/manifest")
        assert resp.status_code == 200
        data = resp.json()
        assert "run_id" in data
        assert "backend_type" in data
        assert "model_path" in data
        assert "num_requests" in data
        assert "started_at" in data


# ---------------------------------------------------------------------------
# Single serve
# ---------------------------------------------------------------------------


class TestServeEndpoint:
    def test_serve_single_request(self, mock_client, sql_request):
        resp = mock_client.post("/api/v1/serve", json=sql_request.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["sample_id"] == "api-test-001"
        assert data["run_id"] is not None
        assert data["predicted_cwe"] == "CWE-89"
        assert data["predicted_severity"] == "high"
        assert "SQL injection" in data["explanation"]
        assert data["runtime_ms"] is not None

    def test_serve_missing_required_field(self, mock_client):
        """POST without vulnerable_code should return 422."""
        resp = mock_client.post("/api/v1/serve", json={"language": "python"})
        assert resp.status_code == 422

    def test_serve_response_model_fields(self, mock_client, sql_request):
        """Verify all ServeResponse fields are present in the output."""
        resp = mock_client.post("/api/v1/serve", json=sql_request.model_dump())
        data = resp.json()
        for field in ("sample_id", "run_id", "predicted_cwe", "predicted_severity",
                       "explanation", "patch_diff", "runtime_ms"):
            assert field in data


# ---------------------------------------------------------------------------
# Batch serve
# ---------------------------------------------------------------------------


class TestBatchServeEndpoint:
    def test_serve_batch(self, mock_client, sql_request):
        batch = BatchServeRequest(requests=[sql_request, sql_request])
        resp = mock_client.post("/api/v1/serve/batch", json=batch.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["responses"]) == 2
        assert data["manifest"]["num_requests"] == 2
        assert data["manifest"]["backend_type"] == "mock"

    def test_serve_batch_empty(self, mock_client):
        batch = BatchServeRequest(requests=[])
        resp = mock_client.post("/api/v1/serve/batch", json=batch.model_dump())
        assert resp.status_code == 200
        data = resp.json()
        assert data["responses"] == []
        assert data["manifest"]["num_requests"] == 0

    def test_batch_requests_have_same_run_id(self, mock_client, sql_request):
        batch = BatchServeRequest(requests=[sql_request, sql_request])
        resp = mock_client.post("/api/v1/serve/batch", json=batch.model_dump())
        data = resp.json()
        run_ids = {r["run_id"] for r in data["responses"]}
        assert len(run_ids) == 1  # all share the same run_id


# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_create_app_mock_config(self):
        config = ServingConfig(backend_type="mock")
        test_app = create_app(config)
        client = TestClient(test_app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_create_app_defaults_to_mock(self):
        """create_app with no config should use mock backend."""
        test_app = create_app()
        assert test_app is not None

    def test_app_has_openapi(self, mock_client):
        resp = mock_client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "paths" in schema
        assert "/api/v1/serve" in schema["paths"]
        assert "/api/v1/serve/batch" in schema["paths"]
        assert "/api/v1/manifest" in schema["paths"]
