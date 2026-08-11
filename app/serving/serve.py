"""Stage 9 — the ``VulnerabilityServer``: a thin orchestrator that ties a
ServingBackend to the Stage 4 prompt/response pipeline.

``VulnerabilityServer`` is the **single object** that the FastAPI app
(``api.py``) and the Typer CLI (``cli.py``) both use.  It takes a
``ServingBackend`` (anything implementing the ``ServingBackend`` Protocol—
``LlamaCppBackend`` / ``OllamaBackend`` / ``MockServingBackend``) and
provides:

* ``serve_sample(sample) -> ServeResponse``
    Build a zero-shot prompt (Stage 4 ``build_zero_shot_prompt``), run it
    through the backend, and parse the JSON response (Stage 4
    ``parse_prediction``) into a ``ServeResponse``.

* ``serve_batch(requests) -> BatchServeResponse``
    Run the above over a list of ``ServeRequest`` objects, collecting
    per-request timing into a ``ServeManifest``.

Heavy/optional imports (``llama_cpp``, ``httpx``) are **not** imported
here — they live inside the individual backend modules and are only
loaded when that backend is instantiated.
"""

from __future__ import annotations

import logging
import time
import uuid

from app.evaluation.parser import ParseError, parse_prediction
from app.evaluation.prompt import build_zero_shot_prompt
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.serving import (
    BatchServeRequest,
    BatchServeResponse,
    ServeRequest,
    ServeResponse,
)
from app.schemas.vuln import VulnSample
from app.serving.backends import (
    LlamaCppBackend,
    MockServingBackend,
    OllamaBackend,
    ServingBackend,
)
from app.serving.config import ServingConfig

logger = logging.getLogger(__name__)

__all__ = ["VulnerabilityServer"]


class VulnerabilityServer:
    """Orchestrates prompt building → backend generation → response parsing.

    Parameters
    ----------
    backend:
        Any object implementing the ``ServingBackend`` Protocol. This is
        the injection point: tests pass ``MockServingBackend``; production
        passes ``LlamaCppBackend`` or ``OllamaBackend``.
    config:
        Optional ``ServingConfig``. When provided, the server's
        ``model_path`` / ``backend_type`` / ``run_name`` are sourced from
        it for manifest provenance. When ``None``, defaults are used.
    """

    def __init__(
        self,
        backend: ServingBackend,
        config: ServingConfig | None = None,
    ):
        # Structural check (Protocol is runtime_checkable)
        if not hasattr(backend, "generate") or not hasattr(backend, "model_info"):
            raise TypeError(
                f"backend={type(backend).__name__} does not implement the "
                f"ServingBackend Protocol (missing 'generate' or 'model_info')."
            )
        self.backend = backend
        self.config = config or ServingConfig()
        self.run_id = str(uuid.uuid4())
        self._request_count = 0

    # ------------------------------------------------------------------ #
    # Single sample
    # ------------------------------------------------------------------ #
    def serve_sample(self, sample: ServeRequest) -> ServeResponse:
        """Analyze one ``ServeRequest`` → ``ServeResponse``.

        Pipeline:
        1. Build a ``VulnSample`` from the ``ServeRequest`` fields
           (the Stage 4 prompt builder requires a ``VulnSample``).
        2. ``build_zero_shot_prompt`` (Stage 4) produces the zero-shot prompt.
        3. ``self.backend.generate(prompt)`` — lazy-loads the real ML model
           on the first call.
        4. ``parse_prediction`` (Stage 4) parses the JSON string into a
           ``ModelPrediction`` (or ``ParseError``).
        5. Build ``ServeResponse`` with timing + run metadata.
        """
        self._request_count += 1

        # Build a VulnSample from the ServeRequest (Stage 4 pattern requires it)
        sample_id = sample.sample_id or f"sample-{self._request_count}"
        vuln_sample = VulnSample(
            id=sample_id,
            source="synthetic_injected",
            repo_name="serving-request",
            cwe_id=sample.cwe_id or "CWE-999",
            severity=sample.severity or "medium",
            language=sample.language,
            vulnerable_code=sample.vulnerable_code,
            description=sample.description or "",
            static_findings=sample.static_findings,
        )

        # Build prompt (Stage 4 pattern)
        prompt = build_zero_shot_prompt(vuln_sample)

        # Time the backend call
        start = time.perf_counter()
        raw_output = self.backend.generate(prompt)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Parse the model's JSON response (Stage 4 pattern)
        # parse_prediction returns ModelPrediction | ParseError
        result = parse_prediction(raw_output, sample_id, self.run_id)

        if isinstance(result, ParseError):
            # Non-fatal parse failure — return PARSE_ERROR in the response
            response = ServeResponse(
                sample_id=sample_id,
                run_id=self.run_id,
                predicted_cwe="PARSE_ERROR",
                predicted_severity="unknown",
                explanation=result.reason,
                patch_diff="",
                runtime_ms=round(elapsed_ms, 2),
            )
        else:
            # ModelPrediction
            prediction: ModelPrediction = result
            response = ServeResponse(
                sample_id=sample_id,
                run_id=self.run_id,
                predicted_cwe=prediction.predicted_cwe,
                predicted_severity=prediction.predicted_severity,
                explanation=prediction.rationale,
                patch_diff=prediction.suggested_patch_diff,
                runtime_ms=round(elapsed_ms, 2),
            )
        return response

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #
    def serve_batch(self, batch: BatchServeRequest) -> BatchServeResponse:
        """Serve a batch of ``ServeRequest`` objects.

        Returns a ``BatchServeResponse`` with all individual
        ``ServeResponse`` entries and a manifest containing
        aggregate provenance (run_id, backend, model_path, counts, avg latency).
        """
        responses: list[ServeResponse] = []
        total_ms = 0.0

        for req in batch.requests:
            resp = self.serve_sample(req)
            responses.append(resp)
            if resp.runtime_ms is not None:
                total_ms += resp.runtime_ms

        # Build manifest dict
        model_info = self.backend.model_info
        manifest: dict = {
            "run_id": self.run_id,
            "backend_type": model_info.get("backend", "unknown"),
            "model_path": model_info.get("model_path", ""),
            "num_requests": len(responses),
            "started_at": "",
            "avg_runtime_ms": round(total_ms / len(responses), 2) if responses else None,
        }

        return BatchServeResponse(
            responses=responses,
            manifest=manifest,
        )

    # ------------------------------------------------------------------ #
    # Convenience: create server from config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: ServingConfig) -> VulnerabilityServer:
        """Create a ``VulnerabilityServer`` by instantiating the right backend.

        This is the main entry point used by CLI and API code:

        * ``backend_type="llama.cpp"`` → ``LlamaCppBackend``
        * ``backend_type="ollama"``     → ``OllamaBackend``
        * ``backend_type="mock"``       → ``MockServingBackend``
        """
        if config.backend_type == "llama.cpp":
            backend: ServingBackend = LlamaCppBackend(
                model_path=config.model_path,
                num_ctx=config.num_ctx,
                num_threads=config.num_threads,
                n_gpu_layers=config.n_gpu_layers,
                f16_kv=config.f16_kv,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
            )
        elif config.backend_type == "ollama":
            backend = OllamaBackend(
                model=config.model_path,
                host=config.host,
                temperature=config.temperature,
                max_new_tokens=config.max_new_tokens,
                request_timeout=config.request_timeout,
            )
        elif config.backend_type == "mock":
            backend = MockServingBackend()
        else:
            raise ValueError(f"Unknown backend_type: {config.backend_type!r}")

        return cls(backend=backend, config=config)

    # ------------------------------------------------------------------ #
    # Manifest / info
    # ------------------------------------------------------------------ #
    def get_manifest(self) -> dict:
        """Return the current run manifest as a plain dict."""
        model_info = self.backend.model_info
        return {
            "run_id": self.run_id,
            "backend_type": model_info.get("backend", "unknown"),
            "model_path": model_info.get("model_path", ""),
            "num_requests": self._request_count,
        }
