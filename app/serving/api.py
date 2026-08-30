"""Stage 9 — FastAPI application for the air-gapped serving layer.

Exposes three endpoints under ``/api/v1/``:

* ``POST /api/v1/serve``       — single vulnerability analysis.
* ``POST /api/v1/serve/batch`` — batch vulnerability analysis.
* ``GET  /api/v1/manifest``    — provenance / run info.

The app is created via ``create_app(config)`` so that it can be
configured at import time (for ``uvicorn app.serving.api:app``) or at
test time (``TestClient(create_app(MockServingConfig))``).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

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

    @app.post(
        "/api/v1/serve",
        response_model=ServeResponse,
        responses={501: {"description": "Backend does not support this operation."}},
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

    @app.post("/api/v1/serve/batch", response_model=BatchServeResponse)
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

    app.state.server = server  # store for external access (e.g. lifespan)
    return app


# Default app instance for `uvicorn app.serving.api:app`
app = create_app()
