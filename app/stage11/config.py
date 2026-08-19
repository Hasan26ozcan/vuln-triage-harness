"""Stage 11 — documentation & interview package configuration.

Flat, immutable dataclass with sensible defaults drawn from the project's
README.  Heavy ML imports (``transformers``, ``torch``, etc.) are **never**
performed at module-import time — the config only carries metadata needed to
render documentation, so it works in CI without any optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas.documentation import (
    BASE_MODEL,
    CWE_SCOPE,
    EvalMetricsSnapshot,
    QuantResultData,
    TrainingRunData,
)

# ---------------------------------------------------------------------------
# Defaults — from the project README
# ---------------------------------------------------------------------------


def _derive_model_name(base_model: str) -> str:
    """Derive a short model name from a HuggingFace base model ID.

    e.g. ``Qwen/Qwen2.5-Coder-1.5B-Instruct`` → ``vuln-triage-qwen2.5-coder-1.5b``
         ``Qwen/Qwen2.5-Coder-7B-Instruct``  → ``vuln-triage-qwen2.5-coder-7b``

    The project is *designed* for the 7B-Instruct variant but CPU-only runs use
    the 1.5B variant due to VRAM constraints. This helper keeps the derived name
    consistent with whichever ``base_model`` was actually used.
    """
    # Extract the size token (e.g. "1.5B", "7B") from "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    import re

    match = re.search(r"Coder-(\d(?:\.\d)?)B", base_model)
    if match:
        size = match.group(1)  # "1.5" or "7"
        return f"vuln-triage-qwen2.5-coder-{size}b"
    # Fallback for unexpected model names
    return f"vuln-triage-{base_model.split('/')[-1].lower().replace('-', '_')}"


# Default base model — designed-for target (Qwen2.5-Coder-7B-Instruct).
# CPU-only validation runs use the 1.5B variant instead (see _derive_model_name).
DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_FAST_MODEL: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Derive model name from the actually-used base model. The default matches the
# CPU training run (1.5B) so docs reflect reality, not the designed-for spec.
DEFAULT_MODEL_NAME: str = _derive_model_name(DEFAULT_FAST_MODEL)
DEFAULT_TRAINING_METHOD: str = "sft_qlora"
DEFAULT_LORA_RANK: int = 64
DEFAULT_TRAINING_DATA_SIZE: int = 5000
DEFAULT_OUTPUT_DIR: str = "./output/stage11"
DEFAULT_DOCS_DIR: str = "docs"


@dataclass(frozen=True)
class Stage11Config:
    """Configuration for Stage 11 documentation generation.

    Attributes
    ----------
    base_model:
        The base model that was fine-tuned (default: Qwen2.5-Coder-7B-Instruct).
    model_name:
        Name for the fine-tuned model (used in the model card).
    training_method:
        Which Stage 5 method was used (``sft_qlora``, ``sft_full``, ``dpo``).
    lora_rank:
        LoRA rank used during training (None for full-parameter SFT).
    quant_method:
        Quantization method chosen from Stage 8 (e.g. ``"gguf"``).
    quant_bit_width:
        Bit-width of the quantized model (e.g. 4 for Q4_0).
    cwe_scope:
        The 6 CWE classes the model was trained on.
    language:
        Primary language of the training data (``"python"``).
    training_data_size:
        Number of samples in the training set (after Stage 2 cleaning).
    training_runs:
        Optional list of past training runs for the report.
    quant_results:
        Optional quantization matrix results for the report.
    output_dir:
        Directory to write Stage 11 artifacts (demo output, etc.).
    docs_dir:
        Directory where ``model_card.md`` and ``training_report.md`` live.
    """

    base_model: str = BASE_MODEL
    model_name: str = DEFAULT_MODEL_NAME
    training_method: str = DEFAULT_TRAINING_METHOD
    lora_rank: int | None = DEFAULT_LORA_RANK
    quant_method: str | None = None
    quant_bit_width: int | None = None
    cwe_scope: list[str] = field(default_factory=lambda: list(CWE_SCOPE))
    language: str = "python"
    training_data_size: int = DEFAULT_TRAINING_DATA_SIZE
    training_runs: list[TrainingRunData] = field(default_factory=list)
    quant_results: list[QuantResultData] = field(default_factory=list)
    baseline_metrics: EvalMetricsSnapshot | None = None
    tuned_metrics: EvalMetricsSnapshot | None = None
    execution_environment: str = "mock"  # "mock", "cpu", or "cuda"
    output_dir: str = DEFAULT_OUTPUT_DIR
    docs_dir: str = DEFAULT_DOCS_DIR
