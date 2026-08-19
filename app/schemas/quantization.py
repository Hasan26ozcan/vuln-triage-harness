"""Quantization experiment contracts (Stage 8).

Defines the structured records produced by the quantization matrix:
``QuantResult`` (one row per quantized checkpoint) and ``QuantReport``
(the full matrix + best-config recommendation). These live in
``app.schemas`` so they can be imported by Stage 6/7 evaluation
results for cross-stage comparison (quality vs. size/speed trade-offs).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class QuantMethod(StrEnum):
    """Quantization backend methods supported by Stage 8."""

    GPTQ = "gptq"  # AutoGPTQ — 2–4-bit, GPU-based quantization
    AWQ = "awq"  # AutoAWQ — 4-bit, activation-aware weight quantization
    GGUF = "gguf"  # llama.cpp GGUF — CPU/GPU, various bit widths
    NONE = "none"  # No quantization (baseline for comparison)


class QuantStatus(StrEnum):
    """Lifecycle status for a quantization run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # e.g. dry-run or unavailable backend


class QuantResult(BaseModel):
    """Result of quantizing a single checkpoint with one method + bit-width.

    Attributes
    ----------
    quant_method:
        Which quantization backend was used (gptq, awq, gguf, none).
    bit_width:
        Target precision in bits (e.g. 2, 3, 4, 8). ``8`` means
        FP8/int8-equivalent; ``4`` = Q4; ``2`` = Q2. ``None`` for
        methods like AWQ that use mixed-precision.
    quantized_model_size_gb:
        On-disk size of the quantized checkpoint (GB).
    estimated_vram_gb:
        VRAM needed to load the quantized model for inference.
    measured_vram_gb:
        Actually measured VRAM (when a GPU is available); ``None``
        if only an estimate was computed.
    tokens_per_sec:
        Inference throughput (tokens generated per second).
    model_cwe_macro_f1:
        CWE Macro-F1 of the quantized model on the gold-eval set
        (re-evaluated via Stage 6 or carried over from saved metrics).
        ``None`` when quality re-evaluation was not performed.
    exec_pass_rate:
        Exec pass rate of the quantized model on the gold-eval set.
        ``None`` when not measured.
    status:
        Lifecycle status of this quantization attempt.
    error:
        Error message if the run failed. ``None`` on success.
    checkpoint_path:
        Path / URI to the quantized checkpoint artifact.
    notes:
        Free-form notes (e.g. "GPTQ with groupsize 128").
    """

    quant_method: QuantMethod
    bit_width: int | None = None
    quantized_model_size_gb: float = Field(..., ge=0.0)
    estimated_vram_gb: float = Field(..., ge=0.0)
    measured_vram_gb: float | None = None
    tokens_per_sec: float | None = None
    model_cwe_macro_f1: float | None = None
    exec_pass_rate: float | None = None
    status: QuantStatus = QuantStatus.PENDING
    error: str | None = None
    checkpoint_path: str = ""
    notes: str | None = None


class QuantReport(BaseModel):
    """Full quantization-matrix report from Stage 8.

    Combines per-result metrics into a single, JSON-serializable document
    that the Stage 9 serving layer and Stage 10 CI gate can consume.

    Attributes
    ----------
    run_id:
        Unique identifier for this quantization matrix run.
    base_model:
        The model that was quantized (e.g. ``"Qwen/Qwen2.5-Coder-7B-Instruct"``).
    source_checkpoint:
        Path / URI of the trained checkpoint that was quantized.
    results:
        One ``QuantResult`` per method × bit-width combination attempted.
    best_result:
        The recommended configuration — the result with the best
        quality/speed/size score (see ``score_quality_size_speed``).
    manifest:
        Run provenance (config, timestamps, elapsed time, environment).
    """

    run_id: str
    base_model: str
    source_checkpoint: str
    results: list[QuantResult]
    best_result: QuantResult | None = None
    manifest: dict = Field(default_factory=dict)

    class Config:
        """Allow QuantMethod / QuantStatus enums to serialize as strings."""

        use_enum_values = False  # We rely on StrEnum's value behavior.


class QuantRecommendation(BaseModel):
    """A single "best configuration" recommendation.

    Used as the top-level output of ``select_best_config``: which method +
    bit-width to deploy for a given target (e.g. "fits in 8 GB VRAM on CPU").
    """

    quant_method: QuantMethod
    bit_width: int | None
    expected_vram_gb: float
    expected_size_gb: float
    expected_cwe_macro_f1: float | None
    expected_exec_pass_rate: float | None
    rationale: str
