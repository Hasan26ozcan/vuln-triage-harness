"""Stage 8 — GPTQ quantization backend (AutoGPTQ).

Lazy-imports ``auto_gptq`` and ``torch`` so this module is import-safe
without a GPU or even PyTorch installed (same pattern as
``QwenBackend._load`` in Stage 4 and ``TokenCounter._load`` in Stage 3).

This file is one of the three quantization backends specified in the
README repo layout (``export_gptq.py``, ``export_awq.py``, ``export_gguf.py``).
"""

from __future__ import annotations

import logging
import time

from app.schemas.quantization import QuantMethod, QuantResult, QuantStatus
from app.quantization.config import (
    GPTQConfig,
    estimate_model_size_gb,
    estimate_vram_gb,
)

logger = logging.getLogger(__name__)

__all__ = ["GPTQQuantizer"]


class GPTQQuantizer:
    """Real GPTQ quantizer using ``auto_gptq``.

    GPTQ is a GPU-based post-training quantization method that supports
    2–4 bit weights. It does not require a separate calibration dataset
    (hence no ``dataset`` parameter unlike some other methods).

    Parameters
    ----------
    config:
        A ``GPTQConfig`` with bits, group_size, desc_act, damping.
    device:
        CUDA device string (default ``cuda:0``).
    """

    def __init__(self, config: GPTQConfig | None = None, device: str = "cuda:0"):
        self.config = config or GPTQConfig()
        self.device = device

    def _load(self):
        """Lazy-load ``auto_gptq``. Raises ``RuntimeError`` if unavailable."""
        try:
            from auto_gptq import AutoGPTQForCausalLM
            return AutoGPTQForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "auto_gptq is not installed. Run "
                "`pip install -e '.[quantization]'` to use GPTQQuantizer, "
                "or use --mock for testing."
            ) from exc

    def quantize(
        self,
        source_checkpoint: str,
        output_path: str,
        bit_width: int | None = None,
    ) -> QuantResult:
        """Quantize *source_checkpoint* with GPTQ and save to *output_path*.

        Parameters
        ----------
        source_checkpoint:
            Path / HF model ID to the full-precision checkpoint.
        output_path:
            Directory to write the quantized model.
        bit_width:
            Target bits (2, 3, or 4). Defaults to ``config.bits``.
        """
        start = time.time()
        bits = bit_width if bit_width is not None else self.config.bits
        AutoGPTQForCausalLM = self._load()

        logger.info(
            "GPTQ quantizing %s → %s (bits=%d, group_size=%d, desc_act=%d, damping=%s)",
            source_checkpoint, output_path, bits,
            self.config.group_size, self.config.desc_act,
            self.config.damping,
        )

        AutoGPTQForCausalLM.quantize(
            model_name_path=source_checkpoint,
            save_path=output_path,
            use_triton=False,
            quant_path=output_path,
            bit_width=bits,
            group_size=self.config.group_size,
            desc_act=self.config.desc_act,
            damp_percent=self.config.damping,
        )

        elapsed = time.time() - start
        est_vram = estimate_vram_gb(QuantMethod.GPTQ, bits)
        est_size = estimate_model_size_gb(QuantMethod.GPTQ, bits)

        return QuantResult(
            quant_method=QuantMethod.GPTQ,
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
                f"GPTQ bits={bits} group_size={self.config.group_size} "
                f"desc_act={self.config.desc_act} damping={self.config.damping} "
                f"quantized in {round(elapsed, 1)}s"
            ),
        )
