"""Stage 8 — AWQ quantization backend (AutoAWQ).

Lazy-imports ``autoawq`` and ``torch`` so this module is import-safe
without a GPU or PyTorch installed (same pattern as
``QwenBackend._load`` in Stage 4 and ``TokenCounter._load`` in Stage 3).

AWQ (Activation-aware Weight Quantization) is a 4-bit weight-only
quantization that preserves activation channels with large magnitudes,
yielding better accuracy than naive rounding.

This file is one of the three quantization backends specified in the
README repo layout (``export_gptq.py``, ``export_awq.py``, ``export_gguf.py``).
"""

from __future__ import annotations

import logging
import time

from app.quantization.config import (
    AWQConfig,
    estimate_model_size_gb,
    estimate_vram_gb,
)
from app.schemas.quantization import QuantMethod, QuantResult, QuantStatus

logger = logging.getLogger(__name__)

__all__ = ["AWQQuantizer"]


class AWQQuantizer:
    """Real AWQ quantizer using ``autoawq``.

    Parameters
    ----------
    config:
        An ``AWQConfig`` with bits, group_size, zero_point.
    device:
        Target device (default ``cuda:0``).
    """

    def __init__(self, config: AWQConfig | None = None, device: str = "cuda:0"):
        self.config = config or AWQConfig()
        self.device = device

    def _load(self):
        """Lazy-load ``autoawq``. Raises ``RuntimeError`` if unavailable."""
        try:
            from awq import AutoAWQForCausalLM
            return AutoAWQForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "autoawq is not installed. Run "
                "`pip install -e '.[quantization]'` to use AWQQuantizer, "
                "or use --mock for testing."
            ) from exc

    def quantize(
        self,
        source_checkpoint: str,
        output_path: str,
        bit_width: int | None = None,
    ) -> QuantResult:
        """Quantize *source_checkpoint* with AWQ and save to *output_path*.

        Parameters
        ----------
        source_checkpoint:
            Path / HF model ID to the full-precision checkpoint.
        output_path:
            Directory to write the quantized model (safetensors).
        bit_width:
            Target bits (typically 4). Defaults to ``config.bits``.
        """
        start = time.time()
        bits = bit_width if bit_width is not None else self.config.bits
        AutoAWQForCausalLM = self._load()

        logger.info(
            "AWQ quantizing %s → %s (bits=%d, group_size=%d, zero_point=%s)",
            source_checkpoint, output_path, bits,
            self.config.group_size, self.config.zero_point,
        )

        model = AutoAWQForCausalLM.from_pretrained(
            source_checkpoint,
            device_map="auto",
            torch_dtype="auto",
        )

        quant_config = {
            "zero_point": self.config.zero_point,
            "q_order": "tloss",
            "auto": self.config.zero_point,
        }
        model.quantize(
            tokenizer=None,
            quant_config=quant_config,
        )
        model.save_quantized(output_path, safetensors=True)

        elapsed = time.time() - start
        est_vram = estimate_vram_gb(QuantMethod.AWQ, bits)
        est_size = estimate_model_size_gb(QuantMethod.AWQ, bits)

        return QuantResult(
            quant_method=QuantMethod.AWQ,
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
                f"AWQ bits={bits} group_size={self.config.group_size} "
                f"zero_point={self.config.zero_point} "
                f"quantized in {round(elapsed, 1)}s"
            ),
        )
