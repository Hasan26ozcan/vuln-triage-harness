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
    """Create a client with a mock backend config.

    Tasks are patched so self.update_state() calls are no-ops,
    preventing Redis connections during eager task execution.
    """
    from unittest.mock import patch

    config = ServingConfig(backend_type="mock")
    test_app = create_app(config)
    with patch("celery.app.task.Task.update_state", lambda *a, **kw: None):
        yield TestClient(test_app)


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

        pytest.importorskip("fastapi.testclient")
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


# ---------------------------------------------------------------------------
# Enqueue evaluation error branches (lines 150-154, 203-205)
# ---------------------------------------------------------------------------


class TestEnqueueEvaluationErrors:
    def test_enqueue_evaluation_not_implemented(self, mock_client, sql_request):
        """When serve_sample raises NotImplementedError, enqueue_evaluation
        returns HTTP 501 — covers lines 150-154."""
        server = mock_client.app.state.server
        server.backend.generate = lambda p: (_ for _ in ()).throw(
            NotImplementedError("not implemented")
        )
        resp = mock_client.post("/api/v1/tasks/evaluation", json=sql_request.model_dump())
        assert resp.status_code == 501
        assert "not implemented" in resp.json()["detail"]

    def test_enqueue_evaluation_inner_generic_error(self, mock_client, sql_request):
        """When serve_sample raises a generic exception inside enqueue_evaluation,
        HTTP 500 is returned from the inner except handler at lines 152-157."""
        server = mock_client.app.state.server
        server.backend.generate = lambda p: (_ for _ in ()).throw(
            RuntimeError("backend crash")
        )
        resp = mock_client.post("/api/v1/tasks/evaluation", json=sql_request.model_dump())
        assert resp.status_code == 500
        assert "Internal serving error" in resp.json()["detail"]

    def test_enqueue_evaluation_outer_error(self, mock_client, sql_request, monkeypatch):
        """When run_evaluation_task.delay raises, HTTP 503 is returned
        — covers lines 203-205."""
        from app.tasks.evaluation import run_evaluation_task

        def raise_error(*args, **kwargs):
            raise RuntimeError("broker down")

        monkeypatch.setattr(run_evaluation_task, "delay", raise_error)
        resp = mock_client.post("/api/v1/tasks/evaluation", json=sql_request.model_dump())
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Enqueue SFT/QLoRA/DPO error branches (lines 246-248, 279-281, 312-314)
# ---------------------------------------------------------------------------


class TestEnqueueTrainingErrors:
    def test_enqueue_sft_outer_error(self, mock_client, monkeypatch):
        """When run_sft_task.delay raises, HTTP 503 is returned
        — covers lines 246-248."""
        from unittest.mock import MagicMock


        def raise_error(*args, **kwargs):
            raise RuntimeError("SFT broker down")

        mock_task = MagicMock()
        mock_task.delay = raise_error
        monkeypatch.setattr("app.tasks.training.run_sft_task", mock_task)
        resp = mock_client.post(  # noqa: E501
            "/api/v1/tasks/training/sft", json={"base_model": "test", "epochs": 1}
        )
        assert resp.status_code == 503
        assert "SFT broker down" in resp.json()["detail"]

    def test_enqueue_qlora_outer_error(self, mock_client, monkeypatch):
        """When run_qlora_task.delay raises, HTTP 503 is returned
        — covers lines 279-281."""
        from unittest.mock import MagicMock


        def raise_error(*args, **kwargs):
            raise RuntimeError("QLoRA broker down")

        mock_task = MagicMock()
        mock_task.delay = raise_error
        monkeypatch.setattr("app.tasks.training.run_qlora_task", mock_task)
        resp = mock_client.post("/api/v1/tasks/training/qlora", json={"base_model": "test"})
        assert resp.status_code == 503
        assert "QLoRA broker down" in resp.json()["detail"]

    def test_enqueue_dpo_outer_error(self, mock_client, monkeypatch):
        """When run_dpo_task.delay raises, HTTP 503 is returned
        — covers lines 312-314."""
        from unittest.mock import MagicMock


        def raise_error(*args, **kwargs):
            raise RuntimeError("DPO broker down")

        mock_task = MagicMock()
        mock_task.delay = raise_error
        monkeypatch.setattr("app.tasks.training.run_dpo_task", mock_task)
        resp = mock_client.post("/api/v1/tasks/training/dpo", json={"base_model": "test"})
        assert resp.status_code == 503
        assert "DPO broker down" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# get_task_status branches (lines 331-342, 348-350)
# ---------------------------------------------------------------------------


class TestGetTaskStatus:
    def test_get_task_status_broker_unreachable(self, mock_client, monkeypatch):
        """When AsyncResult raises (broker not reachable),
        the outer except returns HTTP 500 — covers lines 348-350."""
        from app.celery_app import celery_app

        def raise_error(*args, **kwargs):
            raise RuntimeError("broker unreachable")

        monkeypatch.setattr(celery_app, "AsyncResult", raise_error)
        resp = mock_client.get("/api/v1/tasks/some-task-id")
        assert resp.status_code == 500

    def test_get_task_status_result_ready(self, mock_client, monkeypatch):
        """When result is ready and successful, response includes result
        — covers lines 331-335."""
        from unittest.mock import MagicMock

        from app.celery_app import celery_app

        mock_result = MagicMock()
        mock_result.status = "SUCCESS"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = True
        mock_result.result = {"status": "done"}

        def mock_async_result(task_id):
            return mock_result

        monkeypatch.setattr(celery_app, "AsyncResult", mock_async_result)
        resp = mock_client.get("/api/v1/tasks/some-task-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "SUCCESS"
        assert data["result"] == {"status": "done"}

    def test_get_task_status_result_not_ready(self, mock_client, monkeypatch):
        """When result is not ready, info is included in response
        — covers lines 336-342."""
        from unittest.mock import MagicMock

        from app.celery_app import celery_app

        mock_result = MagicMock()
        mock_result.ready.return_value = False
        mock_result.successful.return_value = False
        mock_result.info = {"stage": "tier2"}

        def mock_async_result(task_id):
            return mock_result

        monkeypatch.setattr(celery_app, "AsyncResult", mock_async_result)
        resp = mock_client.get("/api/v1/tasks/some-task-id")
        assert resp.status_code == 200
        data = resp.json()
        assert "info" in data

    def test_get_task_status_result_failed(self, mock_client, monkeypatch):
        """When result is ready but unsuccessful, error field is set
        — covers line 335."""
        from unittest.mock import MagicMock

        from app.celery_app import celery_app

        mock_result = MagicMock()
        mock_result.status = "FAILURE"
        mock_result.ready.return_value = True
        mock_result.successful.return_value = False
        mock_result.result = Exception("task failed")

        def mock_async_result(task_id):
            return mock_result

        monkeypatch.setattr(celery_app, "AsyncResult", mock_async_result)
        resp = mock_client.get("/api/v1/tasks/some-task-id")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    def test_get_task_status_info_not_json_safe(self, mock_client, monkeypatch):
        """When result.info is not JSON-serializable, it is converted to str
        — covers lines 340-342."""
        from unittest.mock import MagicMock

        from app.celery_app import celery_app

        mock_result = MagicMock()
        mock_result.ready.return_value = False
        mock_result.successful.return_value = False
        mock_result.info = object()  # not JSON-safe

        def mock_async_result(task_id):
            return mock_result

        monkeypatch.setattr(celery_app, "AsyncResult", mock_async_result)
        resp = mock_client.get("/api/v1/tasks/some-task-id")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"] == str(mock_result.info)


# ---------------------------------------------------------------------------
# list_task_queues branches (lines 363-364, 378-380)
# ---------------------------------------------------------------------------


class TestListTaskQueues:
    def test_list_task_queues_inspect_error(self, mock_client, monkeypatch):
        """When inspect.active() raises (broker not reachable),
        empty dicts are returned — covers lines 365-367."""
        from unittest.mock import MagicMock

        from app.celery_app import celery_app

        mock_inspect = MagicMock()
        mock_inspect.active.side_effect = Exception("broker down")
        mock_inspect.scheduled.return_value = {}
        mock_inspect.reserved.return_value = {}

        def mock_inspect_fn(*args, **kwargs):
            return mock_inspect

        monkeypatch.setattr(celery_app.control, "inspect", mock_inspect_fn)
        resp = mock_client.get("/api/v1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_tasks"] == {}
        assert data["scheduled_tasks"] == {}
        assert data["reserved_tasks"] == {}

    def test_list_task_queues_inspect_active_ok(self, mock_client, monkeypatch):
        """When inspect returns normally, scheduled/reserved lines are
        exercised — covers lines 363-364."""
        from unittest.mock import MagicMock

        from app.celery_app import celery_app

        mock_inspect = MagicMock()
        mock_inspect.active.return_value = {}
        mock_inspect.scheduled.return_value = {}
        mock_inspect.reserved.return_value = {}

        def mock_inspect_fn(*args, **kwargs):
            return mock_inspect

        monkeypatch.setattr(celery_app.control, "inspect", mock_inspect_fn)
        resp = mock_client.get("/api/v1/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_tasks"] == {}
        assert data["scheduled_tasks"] == {}
        assert data["reserved_tasks"] == {}

    def test_list_task_queues_outer_error(self, mock_client, monkeypatch):
        """When inspect() itself raises, HTTP 503 is returned
        — covers lines 378-380."""
        from app.celery_app import celery_app

        def mock_inspect_fn(*args, **kwargs):
            raise RuntimeError("no broker")

        monkeypatch.setattr(celery_app.control, "inspect", mock_inspect_fn)
        resp = mock_client.get("/api/v1/tasks")
        assert resp.status_code == 503
        assert "Celery worker" in resp.json()["detail"]
