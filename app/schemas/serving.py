"""Stage 9 — air-gapped serving contracts.

Defines the Pydantic request/response schemas for the serving layer:
single-request inference (``ServeRequest``/``ServeResponse``) and batch
inference (``BatchServeRequest``/``BatchServeResponse``), plus a
``ServeManifest`` for run provenance.

These live in ``app.schemas`` (following the convention of every other
stage) so that the API, CLI, and tests can all import them from a single
location. The schemas deliberately stay minimal — they are thin wrappers
around the same fields the Stage 4/``VulnSample``/``ModelPrediction``
schemas already use, plus a ``runtime_ms`` latency field unique to serving.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.vuln import StaticFinding


class ServeRequest(BaseModel):
    """A single vulnerability-analysis request sent to the serving API/CLI.

    Mirrors ``VulnSample`` fields that are needed for prompt construction
    — the caller supplies the vulnerable code, language, and optional
    static-analysis findings. ``cwe_id`` and ``severity`` are optional
    hints (used when the caller already has them from a prior stage);
    they are **not** used as ground truth for the model — the model
    predicts fresh on every call.

    Attributes
    ----------
    sample_id:
        Optional identifier echoed back in the response for correlation.
    vulnerable_code:
        The source code snippet containing the vulnerability.
    language:
        Programming language of the code (used for prompt formatting).
    cwe_id:
        Optional CWE hint (informational only, not used for prediction).
    severity:
        Optional severity hint (informational only).
    description:
        Short human-readable description of the vulnerability.
    static_findings:
        Static-analysis findings (e.g. Semgrep results) to include in
        the prompt — same type as ``VulnSample.static_findings``.
    temperature:
        Optional per-request override for sampling temperature.
    max_new_tokens:
        Optional per-request override for max generation tokens.
    """

    sample_id: str | None = None
    vulnerable_code: str
    language: str = "python"
    cwe_id: str | None = None
    severity: str | None = None
    description: str = ""
    static_findings: list[StaticFinding] = Field(default_factory=list)
    temperature: float | None = None
    max_new_tokens: int | None = None


class ServeResponse(BaseModel):
    """A single vulnerability-analysis response from the serving layer.

    Fields mirror ``ModelPrediction`` but are named for the API surface
    (``predicted_cwe``, ``predicted_severity``, ``explanation``,
    ``patch_diff``) so they are self-documenting to API consumers who
    may not know about the internal ``ModelPrediction`` name.

    ``runtime_ms`` is the wall-clock inference latency for this single
    request — a serving-specific metric not present in Stage 4/6 schemas.

    If the model response could not be parsed, ``predicted_cwe`` is
    set to ``"PARSE_ERROR"`` and the parse error reason is in
    ``explanation``.
    """

    sample_id: str
    run_id: str
    predicted_cwe: str
    predicted_severity: str
    explanation: str
    patch_diff: str
    runtime_ms: float | None = None


class BatchServeRequest(BaseModel):
    """A batch of ``ServeRequest`` records submitted in a single HTTP call."""

    requests: list[ServeRequest] = Field(default_factory=list)


class BatchServeResponse(BaseModel):
    """Aggregated batch response — a list of ``ServeResponse`` plus provenance."""

    responses: list[ServeResponse] = Field(default_factory=list)
    manifest: dict = Field(default_factory=dict)


class ServeManifest(BaseModel):
    """Run provenance for a serving session.

    Attributes
    ----------
    run_id:
        Unique identifier shared across all requests in this session.
    backend_type:
        Which backend was used (``"llama.cpp"``, ``"ollama"``, ``"mock"``).
    model_path:
        Path/URI of the loaded model checkpoint.
    num_requests:
        Total number of requests served.
    started_at:
        ISO-8601 timestamp of when the server was created.
    avg_runtime_ms:
        Average per-request wall-clock latency across all requests.
    """

    run_id: str
    backend_type: str
    model_path: str
    num_requests: int = 0
    started_at: str = ""
    avg_runtime_ms: float | None = None
