"""Stage 8 — quantization configuration.

Mirrors the dataclass-config pattern from ``app.training.config`` (Stage 5):
flat, immutable dataclasses with sensible defaults drawn from the project's
README tech-stack table (``AutoGPTQ``, ``AutoAWQ``, ``llama.cpp GGUF``).

Heavy ML imports (``auto_gptq``, ``autoawq``, ``llama_cpp``) are **never**
performed at module-import time — they are imported lazily inside the
quantizer implementations so that this module (and the CLI) work without a
GPU or even ``torch`` installed (same pattern as ``QwenBackend._load`` in
Stage 4 and ``TokenCounter._load`` in Stage 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.schemas.quantization import QuantMethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — defaults from the project README tech-stack table
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Output / checkpoint directories
DEFAULT_OUTPUT_BASE: str = "./output/stage8"

# GPTQ defaults
DEFAULT_GPTQ_BITS: int = 4          # 2–4 bit is the usual range
DEFAULT_GPTQ_GROUP_SIZE: int = 128  # groupsize for GPTQ
DEFAULT_GPTQ_DESC_ACT: int = 2       # desc_act (whether to quantize activations)
DEFAULT_GPTQ_DAMPING: float = 0.06   # dampening factor for Hessian

# AWQ defaults
DEFAULT_AWQ_BITS: int = 4
DEFAULT_AWQ_GROUP_SIZE: int = 128
DEFAULT_AWQ_ZERO_POINT: bool = True

# GGUF defaults
DEFAULT_GGUF_QUANT_TYPES: list[str] = [
    "Q2_K",   # 2-bit, ~2.5 bytes/param
    "Q3_K",   # 3-bit, ~3.3 bytes/param
    "Q4_0",   # 4-bit, ~4.0 bytes/param (original GGUF format)
    "Q4_K",   # 4-bit, ~4.3 bytes/param (improved)
    "Q5_K",   # 5-bit, ~5.3 bytes/param
    "Q8_0",   # 8-bit, ~8.5 bytes/param
]
DEFAULT_GGUF_F16: bool = False  # f16 is the "no quant" path for llama.cpp

# Quality estimation fallback (used in dry-run / mock mode).
# These are rough heuristics — real quality is measured via Stage 6 re-evaluation.
_QUALITY_BY_BITS: dict[int, float] = {
    2: 0.60,
    3: 0.78,
    4: 0.92,
    8: 0.98,
}

# Rough VRAM estimates for a 7B parameter model (float32 = ~14 GB).
_VRAM_BY_BITS: dict[int, float] = {
    2: 4.0,
    3: 5.5,
    4: 6.5,
    8: 10.0,
}

# Rough on-disk size estimates for a 7B param model (float32 = ~14 GB).
_SIZE_BY_BITS: dict[int, float] = {
    2: 4.0,
    3: 5.5,
    4: 6.5,
    8: 10.0,
}


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPTQConfig:
    """Configuration for GPTQ quantization (AutoGPTQ)."""

    bits: int = DEFAULT_GPTQ_BITS
    group_size: int = DEFAULT_GPTQ_GROUP_SIZE
    desc_act: int = DEFAULT_GPTQ_DESC_ACT
    damping: float = DEFAULT_GPTQ_DAMPING

    def validate(self) -> list[str]:
        warnings: list[str] = []
        if self.bits not in (2, 3, 4):
            warnings.append(f"GPTQ bits={self.bits} — only 2/3/4 are well-supported.")
        if self.group_size not in (32, 64, 128, 256):
            warnings.append(
                f"GPTQ group_size={self.group_size} — common values: 32/64/128/256."
            )
        if not (0.0 < self.damping < 1.0):
            warnings.append(
                f"GPTQ damping={self.damping} — expected value in (0.0, 1.0)."
            )
        return warnings


@dataclass(frozen=True)
class AWQConfig:
    """Configuration for AWQ quantization (AutoAWQ)."""

    bits: int = DEFAULT_AWQ_BITS
    group_size: int = DEFAULT_AWQ_GROUP_SIZE
    zero_point: bool = DEFAULT_AWQ_ZERO_POINT

    def validate(self) -> list[str]:
        warnings: list[str] = []
        if self.bits not in (2, 3, 4, 5):
            warnings.append(f"AWQ bits={self.bits} — only 2–5 are well-supported.")
        return warnings


@dataclass(frozen=True)
class GGUFConfig:
    """Configuration for GGUF quantization (llama.cpp)."""

    quant_types: list[str] = field(
        default_factory=lambda: list(DEFAULT_GGUF_QUANT_TYPES)
    )
    f16_fallback: bool = DEFAULT_GGUF_F16

    def validate(self) -> list[str]:
        warnings: list[str] = []
        valid_types = {
            "Q2_K", "Q3_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
            "Q4_0", "Q4_K", "Q4_K_S", "Q4_K_M", "Q4_K_L",
            "Q5_K", "Q5_K_S", "Q5_K_M", "Q5_K_L",
            "Q6_K", "Q8_0", "F16", "F32",
        }
        for qt in self.quant_types:
            if qt not in valid_types:
                warnings.append(f"GGUF quant_type={qt!r} — not a recognized type.")
        return warnings


@dataclass
class QuantConfig:
    """Top-level configuration for the Stage 8 quantization matrix.

    Attributes
    ----------
    base_model:
        The HuggingFace model ID of the base checkpoint (e.g.
        ``"Qwen/Qwen2.5-Coder-7B-Instruct"``).
    source_checkpoint:
        Path or S3 URI to the trained Stage 5 checkpoint to quantize.
    output_base:
        Base directory for quantized outputs (one subdirectory per method).
    methods:
        Which quantization methods to try. Empty → all three (gptq, awq, gguf).
    bit_widths:
        Bit-widths to try. When non-empty, each method iterates over these.
    gptq_config:
        Per-method GPTQ options.
    awq_config:
        Per-method AWQ options.
    gguf_config:
        Per-method GGUF options.
    dry_run:
        If True, skip actual quantization and return heuristic estimates.
    mock:
        If True, use ``MockQuantizer`` (fully deterministic, no external deps).
    target_vram_gb:
        If set, ``select_best_config`` will prefer results that fit in this
        VRAM budget.
    target_size_gb:
        If set, prefer results that fit in this on-disk size budget.
    """

    base_model: str = DEFAULT_BASE_MODEL
    source_checkpoint: str = ""
    output_base: str = DEFAULT_OUTPUT_BASE
    methods: list[QuantMethod] = field(default_factory=list)
    bit_widths: list[int] = field(default_factory=lambda: [2, 3, 4, 8])
    gptq_config: GPTQConfig = field(default_factory=GPTQConfig)
    awq_config: AWQConfig = field(default_factory=AWQConfig)
    gguf_config: GGUFConfig = field(default_factory=GGUFConfig)
    dry_run: bool = False
    mock: bool = False
    target_vram_gb: float | None = None
    target_size_gb: float | None = None

    def __post_init__(self) -> None:
        # If no methods specified, try all three.
        if not self.methods:
            self.methods = [QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.GGUF]

    @property
    def run_name(self) -> str:
        """A human-readable label for this quantization run."""
        return f"quant_{self.base_model.split('/')[-1]}"

    def all_warnings(self) -> list[str]:
        """Collect validation warnings from all sub-configs."""
        warnings: list[str] = []
        warnings.extend(self.gptq_config.validate())
        warnings.extend(self.awq_config.validate())
        # gguf_config is a GGUFConfig field on this dataclass.
        warnings.extend(self.gguf_config.validate())
        return warnings


# ---------------------------------------------------------------------------
# Heuristics — used in dry-run / mock mode (no GPU, no torch)
# ---------------------------------------------------------------------------


def estimate_vram_gb(method: QuantMethod, bits: int) -> float:
    """Estimate VRAM (GB) needed to load a quantized 7B model.

    Falls back to config-based lookups; returns a conservative estimate.
    """
    if method == QuantMethod.GPTQ:
        # GPTQ: weights quantized, activations stay in the compute dtype.
        base = _VRAM_BY_BITS.get(bits, 6.5)
        return round(base + 1.0, 2)  # +1 GB for KV cache + activations
    if method == QuantMethod.AWQ:
        base = _VRAM_BY_BITS.get(bits, 6.5)
        return round(base + 1.0, 2)
    if method == QuantMethod.GGUF:
        # GGUF loads weights quantized; VRAM usage is lower on GPU.
        base = _VRAM_BY_BITS.get(bits, 6.5)
        return round(base + 0.5, 2)  # slightly less overhead than GPTQ/AWQ
    if method == QuantMethod.NONE:
        return 15.0  # full FP16
    return 8.0


def estimate_model_size_gb(method: QuantMethod, bits: int) -> float:
    """Estimate on-disk checkpoint size (GB) for a quantized 7B model."""
    if method == QuantMethod.NONE:
        return 14.0  # FP16 ≈ 14 GB
    if method == QuantMethod.GGUF:
        # GGUF file sizes scale slightly differently due to metadata overhead.
        base = _SIZE_BY_BITS.get(bits, 6.5)
        return round(base + 0.3, 2)
    return round(_SIZE_BY_BITS.get(bits, 6.5), 2)


def estimate_quality(method: QuantMethod, bits: int) -> float:
    """Estimate CWE Macro-F1 retention after quantization (rough heuristic).

    ``None`` for bit-widths not in the lookup means "unmeasured".
    """
    return _QUALITY_BY_BITS.get(bits, 0.50)


def estimate_tokens_per_sec(method: QuantMethod, bits: int, device: str = "gpu") -> float:
    """Estimate inference throughput (tokens/sec) for a quantized 7B model.

    Quantized models are typically faster than full-precision due to lower
    memory bandwidth pressure. GGUF on CPU is slower than on GPU.
    """
    if device == "cpu":
        if method == QuantMethod.GGUF:
            # GGUF is designed for CPU inference.
            return round(8.0 * (4.0 / bits) if bits else 8.0, 2)
        return round(4.0 * (4.0 / bits) if bits else 4.0, 2)

    # GPU estimates — quantized models saturate compute earlier.
    base = {2: 45.0, 3: 40.0, 4: 35.0, 8: 25.0, 16: 20.0}
    return round(base.get(bits, 30.0), 2)
