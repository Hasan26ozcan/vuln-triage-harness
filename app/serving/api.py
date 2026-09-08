"""Stage 9 — FastAPI application for the air-gapped serving layer.

Exposes endpoints under ``/api/v1/``:

* ``POST /api/v1/serve``              — single vulnerability analysis.
* ``POST /api/v1/serve/batch``        — batch vulnerability analysis.
* ``GET  /api/v1/manifest``           — provenance / run info.
* ``POST /api/v1/tasks/evaluation``   — enqueue async evaluation.
* ``POST /api/v1/tasks/training/sft`` — enqueue async SFT training.
* ``GET  /api/v1/tasks/{task_id}``    — check task status/result.
* ``GET  /api/v1/tasks/health``       — Celery worker health check.

The app is created via ``create_app(config)`` so that it can be
configured at import time (for ``uvicorn app.serving.api:app``) or at
test time (``TestClient(create_app(MockServingConfig))``).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException

from app.schemas.serving import (
    BatchServeRequest,
    BatchServeResponse,
    ServeRequest,
    ServeResponse,
)
from app.serving.backends import MockServingBackend
from app.serving.config import ServingConfig
from app.serving.serve import VulnerabilityServer

logger = logging.getLogger(__name__)

__all__ = ["app", "create_app"]

#: Default training data path used across all training endpoints.
DEFAULT_TRAIN_DATA = "data/train.jsonl"

#: Common error responses for the serving endpoints.
_SERVE_RESPONSES: dict[int | str, dict[str, Any]] = {
    501: {"description": "Backend does not support this operation."},
    500: {"description": "Internal serving error."},
}

#: Common error responses for Celery task endpoints.
_TASK_RESPONSES: dict[int | str, dict[str, Any]] = {
    503: {"description": "Celery worker not available."},
    500: {"description": "Internal server error."},
}


async def _serve_endpoint(request: ServeRequest, server: VulnerabilityServer) -> ServeResponse:
    """Single-endpoint serve handler with 501/500 HTTPException mapping."""
    try:
        return server.serve_sample(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error serving sample")
        raise HTTPException(
            status_code=500,
            detail=f"Internal serving error: {exc}",
        ) from exc


async def _serve_batch_endpoint(
    batch: BatchServeRequest, server: VulnerabilityServer
) -> BatchServeResponse:
    """Batch serve handler with automatic 500 HTTPException mapping."""
    try:
        return server.serve_batch(batch)
    except Exception as exc:
        logger.exception("Error serving batch")
        raise HTTPException(
            status_code=500,
            detail=f"Internal serving error: {exc}",
        ) from exc


async def _evaluation_endpoint(request: ServeRequest, server: VulnerabilityServer) -> dict:
    """Evaluation enqueue handler — runs inference then enqueues the 4-tier eval."""
    try:
        serve_response = server.serve_sample(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error serving sample for evaluation")
        raise HTTPException(
            status_code=500,
            detail=f"Internal serving error: {exc}",
        ) from exc

    try:
        from app.schemas.prediction_eval import ModelPrediction
        from app.schemas.vuln import VulnSample
        from app.serving.serve import _normalize_severity
        from app.tasks.evaluation import run_evaluation_task

        sample = VulnSample(
            id=serve_response.sample_id,
            source="synthetic_injected",
            repo_name="serving-request",
            cwe_id=request.cwe_id or "CWE-999",
            severity=_normalize_severity(request.severity),
            language=request.language,
            vulnerable_code=request.vulnerable_code,
            description=request.description or "",
            static_findings=request.static_findings,
        )
        prediction = ModelPrediction(
            sample_id=serve_response.sample_id,
            run_id=serve_response.run_id,
            predicted_cwe=serve_response.predicted_cwe,
            predicted_severity=serve_response.predicted_severity,
            suggested_patch_diff=serve_response.patch_diff,
            rationale=serve_response.explanation,
        )

        result = run_evaluation_task.delay(
            samples_json=json.dumps([sample.model_dump()]),
            predictions_json=json.dumps([prediction.model_dump()]),
            sandbox_mode="docker",
            skip_tier3=False,
            skip_tier4=False,
        )
        return {
            "task_id": result.id,
            "status": "PENDING",
            "task_type": "evaluation",
            "message": "Evaluation task enqueued successfully.",
        }
    except Exception as exc:
        logger.exception("Failed to enqueue evaluation task")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _task_status_endpoint(task_id: str) -> dict:
    """Check the status and result of a Celery task."""
    try:
        from app.celery_app import celery_app
        result = celery_app.AsyncResult(task_id)
        response: dict[str, Any] = {"task_id": task_id}
        try:
            response["status"] = result.status
            if result.ready():
                if result.successful():
                    response["result"] = result.result
                else:
                    response["error"] = str(result.result) if result.result else "Unknown error"
            else:
                info = result.info
                json_safe_types = (dict, list, str, int, float, bool, type(None))
                response["info"] = info if isinstance(info, json_safe_types) else str(info)
        except Exception:
            response["status"] = "PENDING"
            response["info"] = "Broker unreachable; task status unknown"
        return response
    except Exception as exc:
        logger.exception("Failed to get task status")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


async def _list_queues_endpoint() -> dict:
    """List active Celery queues and their status."""
    try:
        from app.celery_app import celery_app
        inspect = celery_app.control.inspect()
        try:
            active = inspect.active() or {}
            scheduled = inspect.scheduled() or {}
            reserved = inspect.reserved() or {}
        except Exception:
            active, scheduled, reserved = {}, {}, {}
        return {
            "active_tasks": active,
            "scheduled_tasks": scheduled,
            "reserved_tasks": reserved,
            "queues": {
                "collectors": "CVE data collection",
                "evaluation": "Four-tier evaluation pipeline",
                "training": "SFT/QLoRA/DPO training",
            },
        }
    except Exception as exc:
        logger.exception("Failed to list task queues")
        raise HTTPException(status_code=503, detail="Celery worker not available") from exc


def _make_training_handler(task_fn, train_data_key: str, checkpoint_prefix: str):
    """Return an async handler that enqueues a training task via *task_fn*."""
    async def handler(config: dict[str, Any]) -> dict:
        try:
            config_json = __import__("json").dumps(config)
            result = task_fn.delay(
                train_data_key=config.get("train_data_key", train_data_key),
                config_json=config_json,
                checkpoint_key=config.get(
                    "checkpoint_key",
                    f"checkpoints/{checkpoint_prefix}-{uuid.uuid4().hex[:8]}",
                ),
            )
            return {
                "task_id": result.id,
                "status": "PENDING",
                "task_type": f"{checkpoint_prefix}_training",
                "message": f"{checkpoint_prefix.upper()} training task enqueued successfully.",
            }
        except Exception as exc:
            logger.exception("Failed to enqueue %s training task", checkpoint_prefix)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return handler


def create_app(config: ServingConfig | None = None) -> FastAPI:
    """Create and configure a FastAPI application for serving.

    Parameters
    ----------
    config:
        ``ServingConfig`` controlling which backend to use. If ``None``,
        a mock config is used so the API is testable without ML deps.
    """
    cfg = config or ServingConfig(backend_type="mock", model_path="")
    # Inject a mock backend by default; real backends are heavier.
    if cfg.is_mock():
        backend = MockServingBackend()
        server = VulnerabilityServer(backend=backend, config=cfg)
    else:
        server = VulnerabilityServer.from_config(cfg)

    _started_at = datetime.now(UTC).isoformat()
    app = FastAPI(
        title="Vuln-Triage Harness — Serving API",
        version="9.0.0",
        description="Air-gapped vulnerability triage serving (Stage 9).",
    )

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "backend": server.backend.model_info.get("backend", "unknown")}

    @app.get("/api/v1/manifest")
    async def manifest() -> dict:
        """Return the current run manifest."""
        m = server.get_manifest()
        m["started_at"] = _started_at
        return m

    @app.post("/api/v1/serve", responses=_SERVE_RESPONSES)
    async def serve(request: ServeRequest) -> ServeResponse:
        """Analyze a single vulnerability sample."""
        return await _serve_endpoint(request, server)

    @app.post("/api/v1/serve/batch", responses={500: {"description": "Internal serving error."}})
    async def serve_batch(batch: BatchServeRequest) -> BatchServeResponse:
        """Analyze a batch of vulnerability samples."""
        return await _serve_batch_endpoint(batch, server)

    @app.post("/api/v1/tasks/evaluation", status_code=202, responses=_TASK_RESPONSES)
    async def enqueue_evaluation(request: ServeRequest) -> dict:
        """Enqueue a four-tier evaluation task asynchronously."""
        return await _evaluation_endpoint(request, server)

    @app.post("/api/v1/tasks/training/sft", status_code=202, responses=_TASK_RESPONSES)
    async def enqueue_sft_training(config: dict[str, Any]) -> dict:
        """Enqueue an SFT training task asynchronously."""
        from app.tasks.training import run_sft_task
        return await _make_training_handler(run_sft_task, DEFAULT_TRAIN_DATA, "sft")(config)

    @app.post("/api/v1/tasks/training/qlora", status_code=202, responses=_TASK_RESPONSES)
    async def enqueue_qlora_training(config: dict[str, Any]) -> dict:
        """Enqueue a QLoRA fine-tuning task asynchronously."""
        from app.tasks.training import run_qlora_task
        return await _make_training_handler(run_qlora_task, DEFAULT_TRAIN_DATA, "qlora")(config)

    @app.post("/api/v1/tasks/training/dpo", status_code=202, responses=_TASK_RESPONSES)
    async def enqueue_dpo_training(config: dict[str, Any]) -> dict:
        """Enqueue a DPO training task asynchronously."""
        from app.tasks.training import run_dpo_task
        return await _make_training_handler(run_dpo_task, DEFAULT_TRAIN_DATA, "dpo")(config)

    @app.get("/api/v1/tasks/{task_id}", responses={404: {"description": "Task not found."}})
    async def get_task_status(task_id: str) -> dict:
        """Check the status and result of a Celery task."""
        return await _task_status_endpoint(task_id)

    @app.get("/api/v1/tasks")
    async def list_task_queues() -> dict:
        """List active Celery queues and their status."""
        return await _list_queues_endpoint()

    app.state.server = server  # store for external access (e.g. lifespan)
    return app


# Default app instance for `uvicorn app.serving.api:app`
app = create_app()
