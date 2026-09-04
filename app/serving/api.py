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

    _serve_responses: dict[int | str, dict[str, Any]] = {
        501: {"description": "Backend does not support this operation."},
        500: {"description": "Internal serving error."},
    }

    @app.post(
        "/api/v1/serve",
        responses=_serve_responses,
    )
    async def serve(request: ServeRequest) -> ServeResponse:
        """Analyze a single vulnerability sample."""
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

    @app.post(
        "/api/v1/serve/batch",
        responses={500: {"description": "Internal serving error."}},
    )
    async def serve_batch(batch: BatchServeRequest) -> BatchServeResponse:
        """Analyze a batch of vulnerability samples."""
        try:
            return server.serve_batch(batch)
        except Exception as exc:
            logger.exception("Error serving batch")
            raise HTTPException(
                status_code=500,
                detail=f"Internal serving error: {exc}",
            ) from exc

    # ------------------------------------------------------------------ #
    # Async task endpoints (Celery)
    # ------------------------------------------------------------------ #

    @app.post(
        "/api/v1/tasks/evaluation",
        status_code=202,
        responses={
            503: {"description": "Celery worker not available."},
            500: {"description": "Internal server error."},
        },
    )
    async def enqueue_evaluation(
        request: ServeRequest,
    ) -> dict:
        """Enqueue a four-tier evaluation task asynchronously.

        Returns immediately with a task_id. Use
        ``GET /api/v1/tasks/{task_id}`` to check status/results.
        """
        try:
            from app.celery_app import celery_app
            from app.tasks.evaluation import run_evaluation_task

            samples = [
                request.model_dump()
                for _ in range(1)  # Single-sample eval for now
            ]
            samples_json = __import__("json").dumps(samples)
            predictions_json = __import__("json").dumps([])

            result = run_evaluation_task.delay(
                samples_json=samples_json,
                predictions_json=predictions_json,
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

    @app.post(
        "/api/v1/tasks/training/sft",
        status_code=202,
        responses={
            503: {"description": "Celery worker not available."},
            500: {"description": "Internal server error."},
        },
    )
    async def enqueue_sft_training(
        config: dict[str, Any],
    ) -> dict:
        """Enqueue an SFT training task asynchronously.

        Parameters
        ----------
        config:
            Training configuration including base_model, epochs,
            lora_rank, and hyperparameters.

        Returns immediately with a task_id. Use
        ``GET /api/v1/tasks/{task_id}`` to check status/results.
        """
        try:
            from app.celery_app import celery_app
            from app.tasks.training import run_sft_task

            config_json = __import__("json").dumps(config)
            result = run_sft_task.delay(
                train_data_key=config.get("train_data_key", "data/train.jsonl"),
                config_json=config_json,
                checkpoint_key=config.get(
                    "checkpoint_key", f"checkpoints/sft-{uuid.uuid4().hex[:8]}"
                ),
            )
            return {
                "task_id": result.id,
                "status": "PENDING",
                "task_type": "sft_training",
                "message": "SFT training task enqueued successfully.",
            }
        except Exception as exc:
            logger.exception("Failed to enqueue SFT training task")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/v1/tasks/training/qlora",
        status_code=202,
        responses={
            503: {"description": "Celery worker not available."},
            500: {"description": "Internal server error."},
        },
    )
    async def enqueue_qlora_training(
        config: dict[str, Any],
    ) -> dict:
        """Enqueue a QLoRA fine-tuning task asynchronously."""
        try:
            from app.tasks.training import run_qlora_task

            config_json = __import__("json").dumps(config)
            result = run_qlora_task.delay(
                train_data_key=config.get("train_data_key", "data/train.jsonl"),
                config_json=config_json,
                checkpoint_key=config.get(
                    "checkpoint_key", f"checkpoints/qlora-{uuid.uuid4().hex[:8]}"
                ),
            )
            return {
                "task_id": result.id,
                "status": "PENDING",
                "task_type": "qlora_training",
                "message": "QLoRA training task enqueued successfully.",
            }
        except Exception as exc:
            logger.exception("Failed to enqueue QLoRA training task")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/v1/tasks/training/dpo",
        status_code=202,
        responses={
            503: {"description": "Celery worker not available."},
            500: {"description": "Internal server error."},
        },
    )
    async def enqueue_dpo_training(
        config: dict[str, Any],
    ) -> dict:
        """Enqueue a DPO training task asynchronously."""
        try:
            from app.tasks.training import run_dpo_task

            config_json = __import__("json").dumps(config)
            result = run_dpo_task.delay(
                train_data_key=config.get("train_data_key", "data/train.jsonl"),
                config_json=config_json,
                checkpoint_key=config.get(
                    "checkpoint_key", f"checkpoints/dpo-{uuid.uuid4().hex[:8]}"
                ),
            )
            return {
                "task_id": result.id,
                "status": "PENDING",
                "task_type": "dpo_training",
                "message": "DPO training task enqueued successfully.",
            }
        except Exception as exc:
            logger.exception("Failed to enqueue DPO training task")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(
        "/api/v1/tasks/{task_id}",
        responses={
            404: {"description": "Task not found."},
        },
    )
    async def get_task_status(task_id: str) -> dict:
        """Check the status and result of a Celery task."""
        try:
            from app.celery_app import celery_app

            result = celery_app.AsyncResult(task_id)
            response: dict[str, Any] = {
                "task_id": task_id,
                "status": result.status,
            }
            if result.ready():
                if result.successful():
                    response["result"] = result.result
                else:
                    response["error"] = str(result.result) if result.result else "Unknown error"
            else:
                response["info"] = result.info  # Current progress/state
            return response
        except Exception as exc:
            logger.exception("Failed to get task status")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get(
        "/api/v1/tasks",
    )
    async def list_task_queues() -> dict:
        """List active Celery queues and their status."""
        try:
            from app.celery_app import celery_app

            inspect = celery_app.control.inspect()
            active = inspect.active() or {}
            scheduled = inspect.scheduled() or {}
            reserved = inspect.reserved() or {}
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

    app.state.server = server  # store for external access (e.g. lifespan)
    return app


# Default app instance for `uvicorn app.serving.api:app`
app = create_app()
