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
import subprocess  # nosec B404
import time

from app.quantization.config import (
    DEFAULT_BASE_MODEL,
    GGUFConfig,
    estimate_model_size_gb,
    estimate_vram_gb,
)
from app.schemas.quantization import QuantMethod, QuantResult, QuantStatus
from app.security.paths import validate_output_path, validate_path

logger = logging.getLogger(__name__)

__all__ = ["GGUFQuantizer", "convert_hf_to_gguf_f16"]

# GGUF quant-type → bit-width lookup.
_GGUF_TYPE_TO_BITS: dict[str, int] = {
    "Q2_K": 2,
    "Q3_K": 3,
    "Q3_K_S": 3,
    "Q3_K_M": 3,
    "Q3_K_L": 3,
    "Q4_0": 4,
    "Q4_K": 4,
    "Q4_K_S": 4,
    "Q4_K_M": 4,
    "Q4_K_L": 4,
    "Q5_K": 5,
    "Q5_K_S": 5,
    "Q5_K_M": 5,
    "Q5_K_L": 5,
    "Q6_K": 6,
    "Q8_0": 8,
    "F16": 16,
    "F32": 32,
}

# GGUF file type integer values (general.file_type).
# Ref: gguf.GGMLQuantizationType
#   0 = F32, 1 = F16, 2 = BF16 (not always supported), ...
_GGUF_F32 = 0
_GGUF_F16 = 1


# ---------------------------------------------------------------------------
# HF → GGUF conversion
# ---------------------------------------------------------------------------

# Maps HuggingFace parameter suffixes to GGUF tensor-name fragments for Qwen2.
_HF_TENSOR_MAP: dict[str, str] = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.bias": "attn_output.bias",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate_proj.bias": "ffn_gate.bias",
    "mlp.up_proj.bias": "ffn_up.bias",
    "mlp.down_proj.bias": "ffn_down.bias",
}


def _hf_name_to_gguf(hf_name: str) -> str | None:
    """Translate a HuggingFace parameter name to its GGUF tensor name.

    Returns ``None`` for parameters that are not stored as weights
    (e.g. buffers, ``_orig_mod`` prefixes).
    """
    # Top-level embeddings / norms.
    if hf_name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if hf_name == "model.norm.weight":
        return "output_norm.weight"
    if hf_name == "lm_head.weight":
        return "output.weight"

    # Per-layer: model.layers.{i}.{suffix}
    if hf_name.startswith("model.layers."):
        rest = hf_name[len("model.layers.") :]
        # rest is like "3.self_attn.q_proj.weight"
        parts = rest.split(".")
        idx = parts[0]
        suffix = ".".join(parts[1:])

        mapped = _HF_TENSOR_MAP.get(suffix)
        if mapped is not None:
            return f"blk.{idx}.{mapped}"

    logger.debug("Unmapped tensor name: %s", hf_name)
    return None


def _load_hf_state_dict(
    source_checkpoint: str,
    base_model: str | None = None,
) -> tuple[dict, dict]:
    """Load a HuggingFace checkpoint as a state dict + config dict.

    Handles two layouts:

    1. **PEFT / LoRA adapter** – directory contains ``adapter_config.json``.
       The base model is loaded from HuggingFace (``base_model`` or the path
       embedded in the adapter config), the LoRA weights are merged in, and
       ``merge_and_unload()`` produces a vanilla state dict.

    2. **Full HF checkpoint** – directory contains ``config.json`` and
       ``model.safetensors`` / ``pytorch_model.bin``. Loaded directly.

    Returns ``(state_dict, config_dict)``.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # Validate the source checkpoint path to prevent path traversal (CWE-22).
    # Allow model IDs (e.g. "Qwen/Qwen2.5-Coder-7B-Instruct") and temp dirs
    # (for tests / CI intermediate checkpoints).
    safe_ckpt = validate_path(
        source_checkpoint, allow_model_id=True, allow_temp=True
    )
    adapter_path = os.path.join(str(safe_ckpt), "adapter_config.json")
    if os.path.exists(adapter_path):
        # PEFT / LoRA adapter path.
        import json

        with open(adapter_path) as f:  # NOSONAR
            adapter_config = json.load(f)
        # Prefer the base model from the adapter config — it knows exactly
        # which base it was trained on.  Only fall back to the explicitly
        # passed ``base_model`` parameter (or DEFAULT_BASE_MODEL) when the
        # adapter config doesn't carry one.
        adapter_base = adapter_config.get("base_model_name_or_path") or adapter_config.get(
            "base_model"
        )
        if adapter_base:
            base_model = adapter_base
        elif base_model is None:
            base_model = DEFAULT_BASE_MODEL

        logger.info("Loading base model %s + LoRA adapter from %s", base_model, source_checkpoint)
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)  # nosec B615
        config_dict = config.to_dict()

        # Load base model in float16 (avoids 4-bit quantization artifacts).
        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA adapter and merge.
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(safe_ckpt), is_trainable=False)
        model = model.merge_and_unload()
        logger.info("Merged LoRA adapter into base model")
    else:
        # Full HF checkpoint.
        logger.info("Loading full HF checkpoint from %s", str(safe_ckpt))
        config = AutoConfig.from_pretrained(str(safe_ckpt), trust_remote_code=True)  # nosec B615
        config_dict = config.to_dict()

        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            str(safe_ckpt),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if base_model is None:
            base_model = config_dict.get("model_name_or_path", DEFAULT_BASE_MODEL)

    state_dict = {k: v for k, v in model.state_dict().items()}
    return state_dict, config_dict


def convert_hf_to_gguf_f16(
    source_checkpoint: str,
    output_path: str,
    base_model: str | None = None,
) -> str:
    """Convert a HuggingFace checkpoint (full or PEFT adapter) to F16 GGUF.

    This is the first half of the GGUF pipeline: write the model in
    ``float16`` GGUF format.  The F16 GGUF can then be quantised to
    ``Q4_K`` / ``Q2_K`` / etc. by ``llama-quantize``.
    """
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter

    # Validate the output path to prevent path traversal (CWE-22).
    safe_output_path = validate_output_path(output_path, allow_temp=True)

    state_dict, config = _load_hf_state_dict(source_checkpoint, base_model)

    # --- Determine architecture & metadata ---
    model_type = config.get("model_type", "qwen2")
    arch = model_type  # gguf uses the model_type as the arch prefix
    num_hidden = config.get("num_hidden_layers", 28)
    hidden = config.get("hidden_size", 1536)
    intermediate = config.get("intermediate_size", hidden * 2)
    ctx = config.get("max_position_embeddings", 32768)
    vocab = config.get("vocab_size", 151936)
    n_heads = config.get("num_attention_heads", 12)
    n_kv_heads = config.get("num_key_value_heads", 2)
    rope_freq = config.get("rope_theta", 1000000.0)
    rms_eps = config.get("rms_norm_eps", 1e-6)
    head_dim = hidden // n_heads

    model_name = config.get("model_name_or_path", "qwen2-gguf")

    # --- Open GGUF writer ---
    writer = GGUFWriter(str(safe_output_path), arch, use_temp_file=False)
    writer.add_architecture()
    writer.add_string("general.name", model_name)
    writer.add_file_type(_GGUF_F16)

    writer.add_block_count(num_hidden)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_context_length(ctx)
    writer.add_vocab_size(vocab)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_rope_freq_base(rope_freq)
    writer.add_rope_dimension_count(head_dim)
    writer.add_layer_norm_rms_eps(rms_eps)

    bos_id = config.get("bos_token_id", 151643)
    eos_id = config.get("eos_token_id", 151645)
    pad_id = config.get("pad_token_id", config.get("eos_token_id", 151645))
    writer.add_bos_token_id(bos_id)
    writer.add_eos_token_id(eos_id)
    writer.add_pad_token_id(pad_id)

    # In gguf >= 0.19.0, the writer follows a state machine:
    #   write_header_to_file()  -> opens file, writes header (state: HEADER)
    #   write_kv_data_to_file() -> writes all metadata KV pairs (state: KV_DATA)
    #   add_tensor() + write_tensor_data() -> emit each tensor (state: WEIGHTS)
    #   close() -> finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()

    # --- Write tensors ---
    seen_names: set[str] = set()
    n_written = 0
    # write_tensors_to_file() then flushes everything to disk.
    for name, param in state_dict.items():
        gguf_name = _hf_name_to_gguf(name)
        if gguf_name is None or gguf_name in seen_names:
            continue
        seen_names.add(gguf_name)

        # Convert to numpy array.
        arr = param.detach().cpu().contiguous().numpy()
        if arr.dtype == np.float16:
            pass  # already correct
        elif arr.dtype == np.float32 or arr.dtype == np.bfloat16:
            arr = arr.astype(np.float16)
        elif arr.dtype in (np.float64,):
            arr = arr.astype(np.float32)
        else:
            logger.warning(
                "Unexpected dtype %s for tensor %s — converting to float16", arr.dtype, name
            )
            arr = arr.astype(np.float16)

        writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F16)
        writer.write_tensor_data(arr)
        n_written += 1

    writer.close()
    logger.info("Wrote %d tensors to F16 GGUF: %s", n_written, str(safe_output_path))
    return str(safe_output_path)


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
        base_model: str | None = None,
    ):
        self.config = config or GGUFConfig()
        self._llama_cpp_path = llama_cpp_path
        self._base_model = base_model

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
        cli = shutil.which("llama-quantize") or shutil.which("llama.cpp-quantize")
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
            Path to the full-precision checkpoint (GGUF, HF, or PEFT adapter).
            For HF/PEFT sources, call ``convert_hf_to_gguf_f16()`` first to
            produce an intermediate F16 GGUF, then pass that path here.
        output_path:
            Path for the output ``.gguf`` file.
        bit_width:
            Target bits. Mapped to a GGUF quant-type string.
            Defaults to 4-bit (``Q4_K`` if available, else ``Q4_0``).
        """
        start = time.time()

        bits = bit_width if bit_width is not None else 4
        quant_type = self._bits_to_gguf_type(bits)

        logger.info(
            "GGUF quantizing %s -> %s (type=%s, bits=%d)",
            source_checkpoint,
            output_path,
            quant_type,
            bits,
        )

        backend = self._load()

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
            notes=(f"GGUF type={quant_type} bits={bits} quantized in {round(elapsed, 1)}s"),
        )

    def _quantize_via_cli(
        self,
        cli_path: str,
        source_checkpoint: str,
        output_path: str,
        quant_type: str,
    ) -> None:
        """Invoke the ``llama-quantize`` CLI binary."""
        # Trusted local paths from config
        subprocess.run(  # nosec B603
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
                source_checkpoint,
                output_path,
                dtype="f16",
            )
        else:
            gguf = llama_cpp.ggml
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
