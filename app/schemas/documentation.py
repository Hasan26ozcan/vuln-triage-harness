"""Stage 11 — documentation & interview deliverables.

Pydantic contracts that carry the data needed to render the three Stage 11
deliverables: the model card, the training report, and the demo script output.

These schemas live in ``app.schemas`` (following the convention of every
other stage) so that the generator, CLI, and tests can all import them from a
single location.  The generator (``app.stage11.generator``) turns instances of
these dataclasses into Markdown documents and JSON artefacts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Constants — from the project README
# ---------------------------------------------------------------------------

CWE_SCOPE: list[str] = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]

LANGUAGE_SCOPE: list[str] = ["python", "javascript"]

BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

TRAINING_METHODS: list[str] = ["sft_qlora", "sft_full", "lora", "dpo"]


class EvalMetricsSnapshot(BaseModel):
    """Snapshot of evaluation metrics from a single run.

    Captures the key numbers that the model card and training report cite —
    drawn from Stage 4 (baseline), Stage 6 (four-tier eval), and Stage 7
    (forgetting analysis).
    """

    stage: int  # 4, 6, or 7
    run_id: str
    base_model: str
    cwe_macro_f1: float = 0.0
    severity_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    patch_coverage: float = 0.0
    exec_pass_rate: float = 0.0
    forgetting_delta: float | None = None
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    manifest: dict[str, Any] = Field(default_factory=dict)


class TrainingRunData(BaseModel):
    """Metadata about a single training run (SFT, LoRA sweep, or DPO).

    Mirrors the fields persisted by Stage 5's ``TrainingResult`` dataclass,
    so the report can be rendered from saved checkpoint metadata or from
    PostgreSQL experiment records.
    """

    run_id: str
    method: str  # sft_qlora, sft_full, lora, dpo
    base_model: str = BASE_MODEL
    hyperparams: dict[str, Any] = Field(default_factory=dict)
    train_set_size: int = 0
    train_time_minutes: float = 0.0
    peak_vram_gb: float = 0.0
    final_train_loss: float = 0.0
    final_val_loss: float | None = None
    checkpoint_uri: str = ""
    train_loss_history: list[float] = Field(default_factory=list)


class QuantResultData(BaseModel):
    """One quantization result row for the training report.

    Mirrors ``app.schemas.quantization.QuantResult`` but in a lightweight,
    non-optional form so reports always have concrete numbers to print.
    """

    quant_method: str  # gptq, awq, gguf, none
    bit_width: int | None = None
    quantized_model_size_gb: float = 0.0
    estimated_vram_gb: float = 0.0
    tokens_per_sec: float | None = None
    model_cwe_macro_f1: float | None = None
    exec_pass_rate: float | None = None
    status: str = "skipped"  # completed, skipped, failed


class ModelCardData(BaseModel):
    """Data contract for generating ``docs/model_card.md``.

    A model card is a short, human-readable document that accompanies a
    released model checkpoint.  It describes the model's intended use,
    training data, evaluation results, and known limitations.

    See https://huggingface.co/docs/hub/model-cards for the canonical
    format; our version is a Markdown subset tailored to the project's
    documentation style.
    """

    model_name: str = "vuln-triage-qwen2.5-coder-1.5b"
    base_model: str = BASE_MODEL
    fine_tuned: bool = True
    training_method: str = "sft_qlora"
    lora_rank: int | None = 64
    quant_method: str | None = None
    quant_bit_width: int | None = None
    cwe_scope: list[str] = Field(default_factory=lambda: list(CWE_SCOPE))
    language: str = "python"
    training_data_size: int = 0
    metrics: EvalMetricsSnapshot = Field(
        default_factory=lambda: EvalMetricsSnapshot(
            stage=6, run_id="unknown", base_model=BASE_MODEL
        )
    )
    quantization_options: list[QuantResultData] = Field(default_factory=list)
    serving_backends: list[str] = Field(default_factory=lambda: ["llama.cpp", "ollama", "mock"])
    limitations: list[str] = Field(default_factory=list)
    ethical_considerations: list[str] = Field(default_factory=list)
    intended_use: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class TrainingReportData(BaseModel):
    """Data contract for generating ``docs/training_report.md``.

    A training report is a detailed technical document that records the
    training methodology, hyperparameters, loss curves, evaluation results,
    and conclusions from a fine-tuning experiment.
    """

    report_id: str = ""
    model_name: str = "vuln-triage-qwen2.5-coder-1.5b"
    base_model: str = BASE_MODEL
    training_runs: list[TrainingRunData] = Field(default_factory=list)
    baseline_metrics: EvalMetricsSnapshot | None = None
    tuned_metrics: EvalMetricsSnapshot | None = None
    regression_report: EvalMetricsSnapshot | None = None  # Stage 7
    quant_results: list[QuantResultData] = Field(default_factory=list)
    gate_result: dict[str, Any] = Field(default_factory=dict)  # Stage 10 gate result
    conclusions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class DemoResult(BaseModel):
    """Output of the Stage 11 demo script.

    Contains a summary of the demo run — the model's predictions on the
    gold-eval set in mock mode, plus the four-tier evaluation results.
    """

    run_id: str
    model_name: str
    num_gold_samples: int
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    stage6_report: dict[str, Any] = Field(default_factory=dict)
    succeeded: bool = True
    error: str | None = None
