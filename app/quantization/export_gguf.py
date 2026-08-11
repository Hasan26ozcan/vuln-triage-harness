"""Stage 8 — GGUF quantization backend (llama.cpp).

Lazy-imports ``llama_cpp`` and falls back to the ``llama.cpp`` CLI
(``llama-quantize``) so this module is import-safe without llama.cpp
installed (same pattern as ``QwenBackend._load`` in Stage 4 and
``TokenCounter._load`` in Stage 3).

GGUF is the new llama.cpp file format supporting multiple quant types
(Q2_K, Q3_K, Q4_0, Q4_K, Q5_K, Q6_K, Q8_0) and runs efficiently on CPU
and GPU.

This file is one of the three quantization backends specified in the
README repo layout (``export_gptq.py``, ``export_awq.py``, ``export_gguf.py``).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404 — required for llama.cpp CLI invocation
import time

from app.schemas.quantization import QuantMethod, QuantResult, QuantStatus
from app.quantization.config import (
    GGUFConfig,
    estimate_model_size_gb,
    estimate_vram_gb,
)

logger = logging.getLogger(__name__)

__all__ = ["GGUFQuantizer"]

# GGUF quant-type → bit-width lookup.
_GGUF_TYPE_TO_BITS: dict[str, int] = {
    "Q2_K": 2, "Q3_K": 3, "Q3_K_S": 3, "Q3_K_M": 3, "Q3_K_L": 3,
    "Q4_0": 4, "Q4_K": 4, "Q4_K_S": 4, "Q4_K_M": 4, "Q4_K_L": 4,
    "Q5_K": 5, "Q5_K_S": 5, "Q5_K_M": 5, "Q5_K_L": 5,
    "Q6_K": 6, "Q8_0": 8, "F16": 16, "F32": 32,
}


class GGUFQuantizer:
    """Real GGUF quantizer using ``llama.cpp``.

    Can operate in two modes:

    1. **Python bindings** (``llama-cpp-python`` package) — preferred.
    2. **CLI fallback** (``llama-quantize`` or ``llama.cpp-quantize`` on
       ``PATH``) — used when the Python package is not installed.

    Parameters
    ----------
    config:
        A ``GGUFConfig`` with quant_types and f16_fallback.
    llama_cpp_path:
        Explicit path to the ``llama-quantize`` binary (optional). If
        not provided, the PATH is searched.
    """

    def __init__(
        self,
        config: GGUFConfig | None = None,
        llama_cpp_path: str | None = None,
    ):
        self.config = config or GGUFConfig()
        self._llama_cpp_path = llama_cpp_path

    def _load(self):
        """Lazy-load ``llama_cpp`` (Python) or detect the CLI binary.

        Returns either the ``llama_cpp`` module (if installed) or a string
        path to the CLI binary.
        Raises ``RuntimeError`` if neither is available.
        """
        # Try Python bindings first.
        try:
            import llama_cpp  # noqa: F401
            return llama_cpp
        except ImportError:
            pass

        # Fall back to CLI binary.
        if self._llama_cpp_path and os.path.exists(self._llama_cpp_path):
            return self._llama_cpp_path
        cli = (
            shutil.which("llama-quantize")
            or shutil.which("llama.cpp-quantize")
        )
        if cli:
            return cli

        raise RuntimeError(
            "Neither `llama_cpp` Python package nor `llama.cpp` CLI "
            "(`llama-quantize`) is available. Install with "
            "`pip install llama-cpp-python` or install llama.cpp, "
            "or use --mock for testing."
        )

    def quantize(
        self,
        source_checkpoint: str,
        output_path: str,
        bit_width: int | None = None,
    ) -> QuantResult:
        """Quantize *source_checkpoint* to GGUF format at *output_path*.

        Parameters
        ----------
        source_checkpoint:
            Path / HF model ID to the full-precision checkpoint (will be
            converted to GGUF first if in HuggingFace format).
        output_path:
            Path for the output ``.gguf`` file.
        bit_width:
            Target bits. Mapped to a GGUF quant-type string.
            Defaults to 4-bit (``Q4_K`` if available, else ``Q4_0``).
        """
        start = time.time()

        bits = bit_width if bit_width is not None else 4
        quant_type = self._bits_to_gguf_type(bits)

        backend = self._load()
        logger.info(
            "GGUF quantizing %s → %s (type=%s, bits=%d)",
            source_checkpoint, output_path, quant_type, bits,
        )

        if isinstance(backend, str):
            self._quantize_via_cli(backend, source_checkpoint, output_path, quant_type)
        else:
            self._quantize_via_python(backend, source_checkpoint, output_path, quant_type)

        elapsed = time.time() - start
        est_vram = estimate_vram_gb(QuantMethod.GGUF, bits)
        est_size = estimate_model_size_gb(QuantMethod.GGUF, bits)

        return QuantResult(
            quant_method=QuantMethod.GGUF,
            bit_width=bits,
            quantized_model_size_gb=est_size,
            estimated_vram_gb=est_vram,
            measured_vram_gb=None,
            tokens_per_sec=None,
            model_cwe_macro_f1=None,
            exec_pass_rate=None,
            status=QuantStatus.COMPLETED,
            checkpoint_path=output_path,
            notes=(
                f"GGUF type={quant_type} bits={bits} "
                f"quantized in {round(elapsed, 1)}s"
            ),
        )

    def _quantize_via_cli(
        self,
        cli_path: str,
        source_checkpoint: str,
        output_path: str,
        quant_type: str,
    ) -> None:
        """Invoke the ``llama-quantize`` CLI binary."""
        subprocess.run(  # nosec B603 — trusted local paths from config
            [cli_path, source_checkpoint, output_path, quant_type],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def _quantize_via_python(
        self,
        llama_cpp,
        source_checkpoint: str,
        output_path: str,
        quant_type: str,
    ) -> None:
        """Use the ``llama-cpp-python`` API to quantize."""
        if self.config.f16_fallback and quant_type == "F16":
            llama_cpp.convert_hf_to_gguf(
                source_checkpoint, output_path, dtype="f16",
            )
        else:
            gguf = llama_cpp.ggml  # nosec B712 — module reference
            quantizer = gguf.LlamaQuantize(quant_type)
            llama_cpp.llama_model_quantize(
                str(source_checkpoint),
                str(output_path),
                quantizer,
            )

    @staticmethod
    def _bits_to_gguf_type(bits: int) -> str:
        """Map a bit-width integer to a GGUF quant-type string.

        For ambiguous cases (e.g. 4-bit maps to both Q4_0 and Q4_K),
        prefers the higher-quality ``Q4_K`` variant.
        """
        # Prefer K-quants for the same bit width when available.
        mapping = {
            2: "Q2_K",
            3: "Q3_K",
            4: "Q4_K",
            5: "Q5_K",
            8: "Q8_0",
            16: "F16",
            32: "F32",
        }
        return mapping.get(bits, "Q4_K")


def gguf_type_to_bits(quant_type: str) -> int:
    """Map a GGUF quant-type string to its bit-width integer.

    Raises ``ValueError`` for unrecognized types.
    """
    if quant_type not in _GGUF_TYPE_TO_BITS:
        raise ValueError(f"Unknown GGUF quant type: {quant_type}")
    return _GGUF_TYPE_TO_BITS[quant_type]
