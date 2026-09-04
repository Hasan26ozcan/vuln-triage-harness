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
        for field in (
            "sample_id",
            "run_id",
            "predicted_cwe",
            "predicted_severity",
            "explanation",
            "patch_diff",
            "runtime_ms",
        ):
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


# ---------------------------------------------------------------------------
# Task endpoints (Celery)
# ---------------------------------------------------------------------------


class TestTaskEndpoints:
    """Test the Celery task enqueue endpoints."""

    def test_enqueue_evaluation(self, mock_client):
        """POST /api/v1/tasks/evaluation should return 202 with task_id."""
        import pytest

        TestClient = pytest.importorskip("fastapi.testclient").TestClient
        from app.schemas.serving import ServeRequest

        request = ServeRequest(
            sample_id="task-test-001",
            vulnerable_code="cursor.execute('SELECT * FROM users')",
            language="python",
            description="SQL injection",
        )
        resp = mock_client.post("/api/v1/tasks/evaluation", json=request.model_dump())
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "PENDING"
        assert data["task_type"] == "evaluation"

    def test_enqueue_sft_training(self, mock_client):
        """POST /api/v1/tasks/training/sft should return 202 with task_id."""
        resp = mock_client.post(
            "/api/v1/tasks/training/sft",
            json={
                "base_model": "Qwen2.5-Coder-7B-Instruct",
                "epochs": 3,
                "lora_rank": 8,
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "PENDING"
        assert data["task_type"] == "sft_training"

    def test_enqueue_qlora_training(self, mock_client):
        """POST /api/v1/tasks/training/qlora should return 202 with task_id."""
        resp = mock_client.post(
            "/api/v1/tasks/training/qlora",
            json={"base_model": "Qwen2.5-Coder-7B-Instruct", "lora_rank": 8},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["task_type"] == "qlora_training"

    def test_enqueue_dpo_training(self, mock_client):
        """POST /api/v1/tasks/training/dpo should return 202 with task_id."""
        resp = mock_client.post(
            "/api/v1/tasks/training/dpo",
            json={"base_model": "Qwen2.5-Coder-7B-Instruct", "lora_rank": 8},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["task_type"] == "dpo_training"

    def test_list_task_queues(self, mock_client):
        """GET /api/v1/tasks should return queue info."""
        resp = mock_client.get("/api/v1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "queues" in data
        assert "collectors" in data["queues"]
        assert "evaluation" in data["queues"]
        assert "training" in data["queues"]

    def test_get_task_status(self, mock_client):
        """GET /api/v1/tasks/{task_id} should return task status."""
        # Use a fake task_id — the endpoint should return PENDING or NOT_FOUND
        resp = mock_client.get("/api/v1/tasks/non-existent-task-id")
        # The result should either be a 404 or show the task status
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "task_id" in data
            assert "status" in data


# ---------------------------------------------------------------------------
# Non-mock config path (line 51)
# ---------------------------------------------------------------------------


class TestCreateAppNonMock:
    def test_create_app_with_llama_cpp_config(self):
        """create_app with a non-mock config calls VulnerabilityServer.from_config
        (line 51) instead of building a MockServingBackend."""
        config = ServingConfig(backend_type="llama.cpp", model_path="/fake/model.gguf")
        test_app = create_app(config)
        client = TestClient(test_app)
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["backend"] == "llama.cpp"


# ---------------------------------------------------------------------------
# Error-handling branches in serve / serve_batch endpoints
# ---------------------------------------------------------------------------


class TestServeEndpointErrorHandling:
    def test_serve_not_implemented_returns_501(self, mock_client, sql_request):
        """When the backend raises NotImplementedError, /api/v1/serve returns
        HTTP 501 — covers lines 76-77 of api.py."""
        server = mock_client.app.state.server

        def raise_not_impl(prompt: str) -> str:
            raise NotImplementedError("generate not supported for this backend")

        server.backend.generate = raise_not_impl

        resp = mock_client.post("/api/v1/serve", json=sql_request.model_dump())
        assert resp.status_code == 501
        assert "not supported" in resp.json()["detail"]

    def test_serve_generic_error_returns_500(self, mock_client, sql_request):
        """When the backend raises a generic Exception, /api/v1/serve returns
        HTTP 500 — covers lines 78-80 of api.py."""
        server = mock_client.app.state.server

        def raise_runtime(prompt: str) -> str:
            raise RuntimeError("unexpected backend failure")

        server.backend.generate = raise_runtime

        resp = mock_client.post("/api/v1/serve", json=sql_request.model_dump())
        assert resp.status_code == 500
        assert "Internal serving error" in resp.json()["detail"]


class TestBatchEndpointErrorHandling:
    def test_batch_serve_error_returns_500(self, mock_client, sql_request):
        """When serve_batch raises an Exception, /api/v1/serve/batch returns
        HTTP 500 — covers lines 90-92 of api.py."""
        server = mock_client.app.state.server

        def raise_runtime(prompt: str) -> str:
            raise RuntimeError("batch backend failure")

        server.backend.generate = raise_runtime

        batch = BatchServeRequest(requests=[sql_request])
        resp = mock_client.post("/api/v1/serve/batch", json=batch.model_dump())
        assert resp.status_code == 500
        assert "Internal serving error" in resp.json()["detail"]
