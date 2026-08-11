"""Stage 8 — quantization matrix (GPTQ / AWQ / GGUF).

This package implements the quantization matrix described in the README:
it compares quantization methods across bit-widths and measures the
quality (CWE Macro-F1, exec pass rate) vs. speed (tokens/sec) vs. VRAM
trade-off.

Public API
----------
* ``QuantConfig``        — top-level config dataclass (``config.py``)
* ``QuantResult``        — single-result schema (``app.schemas.quantization``)
* ``QuantReport``        — full-matrix report (``app.schemas.quantization``)
* ``QuantMethod``        — enum: GPTQ / AWQ / GGUF / NONE
* ``QuantStatus``        — enum: PENDING / RUNNING / COMPLETED / FAILED / SKIPPED
* ``Quantizer``          — Protocol for injectable quantizer backends
* ``MockQuantizer``      — deterministic mock for testing
* ``GPTQQuantizer``      — AutoGPTQ backend (``export_gptq.py``)
* ``AWQQuantizer``       — AutoAWQ backend (``export_awq.py``)
* ``GGUFQuantizer``      — llama.cpp GGUF backend (``export_gguf.py``)
* ``quantize_single``    — quantize once with one method + bit-width
* ``select_best_config`` — pick the best result given VRAM / size constraints
* ``run_quantization_matrix`` — full matrix orchestrator

Usage::

    from app.quantization import QuantConfig, run_quantization_matrix
    from app.schemas.quantization import QuantMethod

    config = QuantConfig(
        source_checkpoint="output/stage5/checkpoint",
        mock=True,
    )
    report = run_quantization_matrix(config)
    print(report.model_dump_json(indent=2))
"""

from app.quantization.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_BASE,
    AWQConfig,
    GGUFConfig,
    GPTQConfig,
    QuantConfig,
    estimate_model_size_gb,
    estimate_quality,
    estimate_tokens_per_sec,
    estimate_vram_gb,
)
from app.quantization.quantizer import (
    AWQQuantizer,
    GGUFQuantizer,
    GPTQQuantizer,
    MockQuantizer,
    Quantizer,
    quantize_single,
    run_quantization_matrix,
    score_quality_size_speed,
    select_best_config,
)
from app.schemas.quantization import (
    QuantMethod,
    QuantRecommendation,
    QuantReport,
    QuantResult,
    QuantStatus,
)

__all__ = [
    # Schemas
    "QuantMethod",
    "QuantReport",
    "QuantRecommendation",
    "QuantResult",
    "QuantStatus",
    # Config
    "DEFAULT_BASE_MODEL",
    "DEFAULT_OUTPUT_BASE",
    "GPTQConfig",
    "AWQConfig",
    "GGUFConfig",
    "QuantConfig",
    # Heuristics
    "estimate_model_size_gb",
    "estimate_quality",
    "estimate_tokens_per_sec",
    "estimate_vram_gb",
    # Quantizers
    "Quantizer",
    "MockQuantizer",
    "GPTQQuantizer",
    "AWQQuantizer",
    "GGUFQuantizer",
    # Runner
    "quantize_single",
    "select_best_config",
    "run_quantization_matrix",
    "score_quality_size_speed",
]
