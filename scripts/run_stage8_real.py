#!/usr/bin/env python
"""Stage 8 — real quantization matrix on a trained checkpoint.

After Stage 5 training (SFT / QLoRA / DPO) and Stage 6/7 evaluation, Stage 8
quantizes the trained checkpoint with GPTQ, AWQ, and GGUF — producing a real
``QuantReport`` with measured file sizes, VRAM usage, and inference throughput.

This script mirrors the pattern of ``scripts/run_stage6_only.py`` and
``scripts/run_stage7_only.py``: it loads the Stage 5 trained LoRA checkpoint,
merges the adapter into the base model (so quantizers receive a full-precision
HF model), runs each available quantizer method, measures real metrics, and
optionally re-evaluates the best quantized checkpoint through Stage 6 to get
quality numbers (CWE Macro-F1, exec pass rate).

Usage::

    python scripts/run_stage8_real.py \\
        --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --checkpoint ./output/stage5/qwen_lora_gpu/final_checkpoint \\
        --output-dir ./output/stage8

Options:
    --skip-eval       Skip Stage 6 re-evaluation of quantized checkpoints (faster)
    --methods gptq,gguf   Restrict which quantization methods to run
    --bits 2,3,4,8      Bit-widths for GPTQ/AWQ
    --target-vram 6.0   VRAM budget filter for best-config selection
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

import torch

# Ensure project root is on sys.path when run as a standalone script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Unbuffered output for background runs.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_CHECKPOINT = "./output/stage5/qwen_lora_gpu/final_checkpoint"
DEFAULT_OUTPUT_DIR = "./output/stage8"
DEFAULT_GOLD_EVAL = "eval/gold_set/gold.jsonl"


def _merge_lora_to_hf(
    base_model: str,
    checkpoint: str,
    merged_dir: str,
) -> str:
    """Merge a LoRA/PEFT adapter into its base model and save as a standalone HF checkpoint.

    Quantizers (GPTQ, AWQ) call ``AutoGPTQForCausalLM.from_pretrained()`` /
    ``AutoAWQForCausalLM.from_pretrained()`` which expect a full-precision model
    directory — they do not understand PEFT adapter layouts. So we merge first
    and save the merged state to ``merged_dir``.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Merging LoRA adapter %s into base %s ...", checkpoint, base_model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    # Use the non-deprecated parameter name when available (transformers >= 4.50).
    import inspect as _inspect
    if "dtype" in _inspect.signature(AutoModelForCausalLM.from_pretrained).parameters:
        kwargs = {"dtype": torch.float16, "device_map": "auto", "trust_remote_code": True}
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)  # nosec B615

    # Apply LoRA adapter and merge.
    model = PeftModel.from_pretrained(model, checkpoint)
    model = model.merge_and_unload()
    model = model.eval()

    Path(merged_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    # Free GPU memory.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("Merged model saved to %s", merged_dir)
    return merged_dir


def _measure_vram_gb() -> float | None:
    """Return currently allocated GPU memory in GB (or None if no CUDA)."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / (1024**3)


def _measure_throughput(
    model,
    tokenizer,
    prompt: str = "def hello(): pass",
    max_new_tokens: int = 32,
    runs: int = 3,
) -> float | None:
    """Measure inference throughput (tokens/sec) on a simple prompt."""
    try:
        import time as _time

        inputs = tokenizer(prompt, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        # Move inputs to the same device as the model to avoid device-mismatch
        # errors with quantized backends (ExLlama, etc.) that live on CUDA.
        model_device = next(model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        # Warmup.
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=8)

        times = []
        with torch.no_grad():
            for _ in range(runs):
                t0 = _time.perf_counter()
                output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                _ = output
                times.append(_time.perf_counter() - t0)

        avg_time = sum(times) / len(times)
        total_tokens = input_len + max_new_tokens
        return round(total_tokens / avg_time, 2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Throughput measurement failed: %s", exc)
        return None


def _measure_file_size_gb(path: str) -> float:
    """Measure on-disk size of a checkpoint directory or file in GB."""
    total = 0
    p = Path(path)
    if p.is_dir():
        for f in p.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    elif p.is_file():
        total = p.stat().st_size
    return round(total / (1024**3), 3)


def _patch_attention_type():
    """Context manager that patches ``nn.Module.__getattr__`` to delegate
    unknown attribute lookups to ``self.module`` (the wrapped layer).

    This works around a known incompatibility: auto_gptq 0.7.x replaces each
    decoder layer with a local ``LayerHijacker`` class whose ``__getattr__``
    only checks its own ``_parameters`` / ``_buffers`` / ``_modules`` dicts.
    In transformers >= 4.52, ``Qwen2Model.forward`` accesses
    ``decoder_layer.attention_type`` — an attribute of the original layer
    but not the hijacker — causing ``AttributeError``.

    The context manager temporarily patches ``nn.Module.__getattr__`` so that
    if a ``LayerHijacker`` fails to find an attribute, it falls through to
    the wrapped ``self.module``. The original behaviour is restored on exit.

    Usage::

        with _patch_attention_type():
            model.quantize(examples, batch_size=1)

    Returns
    -------
    A context manager.
    """
    import contextlib
    import torch.nn as nn

    @contextlib.contextmanager
    def _ctx():
        orig_getattr = nn.Module.__getattr__

        def patched_getattr(self, name: str):
            try:
                return orig_getattr(self, name)
            except AttributeError:
                # LayerHijacker stores the original module as self.module
                # (registered as a submodule, so it's in self._modules).
                if "module" in self._modules:
                    wrapped = self._modules["module"]
                    try:
                        return getattr(wrapped, name)
                    except AttributeError:
                        pass
                raise

        nn.Module.__getattr__ = patched_getattr
        try:
            yield
        finally:
            nn.Module.__getattr__ = orig_getattr

    return _ctx()


def _patch_qwen2_decoder_tuple_return():
    """Context manager that patches ``Qwen2DecoderLayer.forward`` to return
    a tuple ``(hidden_states,)`` instead of a bare tensor.

    auto_gptq 0.7.x's quantize loop (in ``_base.py``) does:

        layer_output = layer(layer_input, **additional_layer_inputs)[0]

    This assumes the decoder layer returns a *tuple* (as Llama, Mistral, etc.
    do) so that ``[0]`` extracts the hidden-states tensor.  However, in
    transformers >= 4.52, ``Qwen2DecoderLayer.forward`` returns a bare
    ``torch.Tensor`` — not a tuple.  Consequently ``tensor[0]`` slices the
    batch dimension (dim-0), stripping it and reducing the shape from
    ``[1, seq_len, hidden]`` to ``[seq_len, hidden]``.

    The 2-D tensor then flows into the next layer's attention, where
    ``Qwen2Attention.forward`` reshapes it to the wrong head dimensions,
    causing::

        RuntimeError: The size of tensor a (12) must match the size of
        tensor b (128) at non-singleton dimension 3

    at the ``apply_rotary_pos_emb`` call.

    The fix wraps the forward return value in a 1-tuple during quantization
    only.  This is safe because:

    * During the first forward pass (``self.model(**example)``), layer 0 is
      replaced by a ``LayerHijacker`` whose ``forward`` raises ``ValueError``
      before any other layer is reached, so the patched ``Qwen2DecoderLayer``
      forward is never called through ``Qwen2Model.forward``.
    * During the calibration / re-forward passes, each layer is called
      directly (``layer(layer_input, ...)``), not through the model, so
      returning a tuple has no downstream effect.
    * The patch is scoped to the ``quantize()`` call via a context manager
      and restored afterwards, so normal model usage (``generate``,
      ``Qwen2Model.forward``) is unaffected.
    """
    import contextlib
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

    original_forward = Qwen2DecoderLayer.forward

    def patched_forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        if not isinstance(output, tuple):
            return (output,)
        return output

    @contextlib.contextmanager
    def _ctx():
        Qwen2DecoderLayer.forward = patched_forward
        try:
            yield
        finally:
            Qwen2DecoderLayer.forward = original_forward

    return _ctx()


def _patch_gptq_cholesky_resilience():
    """Monkey-patch ``GPTQ.fasterquant`` and ``GPTQ.add_batch`` to handle
    Cholesky decomposition failures caused by NaN/Inf contamination.

    auto_gptq's GPTQ computes the Hessian ``H`` from calibration activations.
    For some layers (notably output projections in transformer models), the
    float16 activations can overflow to Inf/NaN (attention scores can explode),
    which when accumulated into the Hessian via ``H += inp.matmul(inp.t())``
    produces NaN values. A NaN-contaminated Hessian is never positive-definite,
    so ``torch.linalg.cholesky`` fails regardless of damping.

    This wrapper:
    1. Patches ``GPTQ.add_batch`` to sanitize inputs with ``nan_to_num``
       before accumulation — prevents NaN from entering ``H`` in the first
       place.
    2. Saves ``self.H`` and ``self.nsamples`` *before* calling the original
       ``fasterquant`` (which deletes ``self.H`` internally before the Cholesky
       call).
    3. On Cholesky failure, sanitizes ``H`` with ``nan_to_num``, restores
       ``H``/``nsamples``, and retries with escalating ``percdamp``
       (10x, 100x, 1000x).
    4. If all scaled attempts fail, adds a minimum absolute damping floor
       to the diagonal (1e-4 to 1e-1) as a last resort.
    """
    import torch
    from auto_gptq.quantization.gptq import GPTQ

    # --- Patch add_batch to sanitize float16 overflow artifacts ---
    original_add_batch = GPTQ.add_batch

    def sanitized_add_batch(self, inp, out):
        # Replace NaN/Inf in calibration activations (float16 overflow
        # in attention/o_proj) before they contaminate the Hessian.
        if inp.dtype in (torch.float16, torch.bfloat16):
            inp = torch.nan_to_num(inp.float(), nan=0.0, posinf=65504.0, neginf=-65504.0)
        if out.dtype in (torch.float16, torch.bfloat16):
            out = torch.nan_to_num(out.float(), nan=0.0, posinf=65504.0, neginf=-65504.0)
        return original_add_batch(self, inp, out)

    GPTQ.add_batch = sanitized_add_batch
    logger.info("[GPTQ] Patched add_batch with nan_to_num (float16 overflow guard)")

    # --- Patch fasterquant with retry logic ---
    original_fasterquant = GPTQ.fasterquant

    def resilient_fasterquant(self, blocksize=128, percdamp=0.01, group_size=-1,
                              actorder=False, static_groups=False):
        # Save H and nsamples before fasterquant deletes self.H.
        saved_H = getattr(self, "H", None)
        saved_nsamples = getattr(self, "nsamples", None)

        # Sanitize H: replace any NaN/Inf that slipped past add_batch.
        if saved_H is not None:
            if torch.isnan(saved_H).any() or torch.isinf(saved_H).any():
                logger.warning(
                    "[GPTQ] H contained NaN/Inf — applying nan_to_num "
                    "(nan=0, inf=1e4) before fasterquant")
                saved_H = torch.nan_to_num(saved_H, nan=0.0, posinf=1e4, neginf=-1e4)

        try:
            return original_fasterquant(
                self, blocksize=blocksize, percdamp=percdamp,
                group_size=group_size, actorder=actorder,
                static_groups=static_groups,
            )
        except (RuntimeError, torch.linalg.LinAlgError) as exc:
            err = str(exc).lower()
            if "cholesky" not in err and "not positive" not in err:
                raise

            logger.warning(
                "[GPTQ] Cholesky failed (percdamp=%s) — retrying with "
                "escalating damping ...",
                percdamp,
            )

            # Retry with escalating percdamp.
            for factor in [10, 100, 1000]:
                if saved_H is None:
                    break
                self.H = saved_H.clone()
                if saved_nsamples is not None:
                    self.nsamples = saved_nsamples
                try:
                    return original_fasterquant(
                        self, blocksize=blocksize,
                        percdamp=percdamp * factor,
                        group_size=group_size, actorder=actorder,
                        static_groups=static_groups,
                    )
                except (RuntimeError, torch.linalg.LinAlgError) as exc2:
                    err2 = str(exc2).lower()
                    if "cholesky" not in err2 and "not positive" not in err2:
                        raise
                    logger.warning(
                        "[GPTQ] Cholesky still failing "
                        "(percdamp=%s) — trying next factor",
                        percdamp * factor,
                    )

            # Last resort: add absolute damping floor to diagonal.
            # The problem is that ``damp = percdamp * mean(diag(H))`` is
            # relative, and when mean(diag(H)) ≈ 0 the relative damping is
            # useless. We add an absolute floor to the diagonal instead.
            if saved_H is not None:
                for floor_val in [1e-4, 1e-3, 1e-2, 1e-1]:
                    logger.warning(
                        "[GPTQ] Retrying with absolute floor=%s on diagonal",
                        floor_val,
                    )
                    self.H = saved_H.clone()
                    if saved_nsamples is not None:
                        self.nsamples = saved_nsamples
                    diag_idx = torch.arange(self.H.shape[0], device=self.H.device)
                    self.H[diag_idx, diag_idx] += floor_val
                    try:
                        return original_fasterquant(
                            self, blocksize=blocksize, percdamp=percdamp * 1000,
                            group_size=group_size, actorder=actorder,
                            static_groups=static_groups,
                        )
                    except (RuntimeError, torch.linalg.LinAlgError) as exc3:
                        err3 = str(exc3).lower()
                        if "cholesky" not in err3 and "not positive" not in err3:
                            raise
                    except Exception as exc4:
                        logger.warning("[GPTQ] Non-Cholesky error: %s", exc4)

            raise exc  # All retries failed.

    GPTQ.fasterquant = resilient_fasterquant
    logger.info("[GPTQ] Patched fasterquant with Cholesky resilience")


def _check_dep_availability(method: str) -> bool:
    """Check if the ML library for a quantization method is installed."""
    if method == "gptq":
        try:
            import auto_gptq  # noqa: F401

            return True
        except ImportError:
            return False
    if method == "awq":
        try:
            import awq  # noqa: F401

            return True
        except ImportError:
            return False
    if method == "gguf":
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            # Also check for CLI fallback.
            import shutil as _shutil

            return _shutil.which("llama-quantize") is not None or _shutil.which("llama.cpp-quantize") is not None
    return False


def _run_gptq(
    source_checkpoint: str,
    output_path: str,
    bit_width: int,
    config_dict: dict,
) -> dict:
    """Run real GPTQ quantization using AutoGPTQ.

    Uses the auto_gptq 0.7.x API: load model with ``BaseQuantizeConfig``,
    call ``model.quantize(calibration_examples)``, then ``save_pretrained()``.
    Returns a dict with measured metrics (not QuantResult — we assemble that
    at the caller level to keep the measurement logic shared).
    """
    from auto_gptq import AutoGPTQForCausalLM
    from auto_gptq.modeling._base import BaseQuantizeConfig
    from transformers import AutoTokenizer

    logger.info("[GPTQ] Starting %d-bit quantization ...", bit_width)
    start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Build quantize config.
    group_size = config_dict.get("group_size", 128)
    damping = config_dict.get("damping", 0.1)
    desc_act = bool(config_dict.get("desc_act", 2))

    quantize_config = BaseQuantizeConfig(
        bits=bit_width,
        group_size=group_size,
        damp_percent=damping,
        desc_act=desc_act,
    )

    quant_config = {
        "desc_act": desc_act,
        "group_size": group_size,
        "damp_percent": damping,
    }

    # Load tokenizer for calibration.
    tokenizer = AutoTokenizer.from_pretrained(source_checkpoint, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build calibration examples — a small set of tokenized code snippets.
    calib_path = config_dict.get("calib_dataset")
    calib_texts: list[str] = []
    if calib_path and os.path.exists(calib_path):
        logger.info("[GPTQ] Loading calibration dataset from %s", calib_path)
        # Load JSONL directly (avoids datasets/pyarrow compatibility issues).
        with open(calib_path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        # Use up to 128 samples for calibration.
        records = records[:128]
        if "text" in records[0]:
            calib_texts = [r["text"] for r in records]
        elif "prompt" in records[0]:
            calib_texts = [r["prompt"] for r in records]
    else:
        # Fallback: longer, more diverse code snippets for better Hessian conditioning.
        logger.info("[GPTQ] No calibration dataset — using fallback code prompts")
        calib_texts = [
            # SQL injection patterns
            "def get_user(username, password):\n"
            "    query = f'SELECT * FROM users WHERE username=\"{username}\" AND password=\"{password}\"\'\n"
            "    cursor.execute(query)\n    return cursor.fetchone()\n",
            # XSS / HTML sanitization
            "def render_comment(comment):\n"
            "    html = f'<div>{comment}</div>'\n"
            "    return html\n",
            # Password hashing
            "def verify_password(stored_hash, provided_password):\n"
            "    return stored_hash == provided_password\n",
            # Command injection
            "def run_ping(host):\n"
            "    import subprocess\n"
            "    result = subprocess.check_output(f'ping {host}', shell=True)\n"
            "    return result\n",
            # Path traversal
            "def read_file(filename):\n"
            "    with open(f'/var/data/{filename}', 'r') as f:\n        return f.read()\n",
            # CSRF / token handling
            "def create_session(user_id):\n"
            "    token = str(user_id) + 'abc'\n"
            "    return token\n",
            # SSRF
            "def fetch_url(url):\n"
            "    import requests\n"
            "    resp = requests.get(url)\n    return resp.text\n",
            # Deserialization
            "def load_config(path):\n"
            "    import pickle\n"
            "    with open(path, 'rb') as f:\n        return pickle.load(f)\n",
            # JWT / crypto
            "def decode_jwt(token):\n"
            "    import base64\n"
            "    parts = token.split('.')\n"
            "    payload = base64.b64decode(parts[1])\n"
            "    return payload\n",
            # File upload
            "def save_upload(file_obj, filename):\n"
            "    dest = '/uploads/' + filename\n"
            "    file_obj.save(dest)\n"
            "    return dest\n",
        ]

    logger.info("[GPTQ] Tokenizing %d calibration samples ...", len(calib_texts))
    examples: list[dict] = []
    for text in calib_texts[:128]:
        tokenized = tokenizer(
            text,
            max_length=2048,
            truncation=True,
            return_tensors="pt",
        )
        examples.append({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        })

    logger.info("[GPTQ] Loading model and quantizing ...")
    # Use attn_implementation="eager" to avoid SDPA attention issues.
    model = AutoGPTQForCausalLM.from_pretrained(
        source_checkpoint,
        quantize_config=quantize_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    # --- Compatibility shims for auto_gptq 0.7.x + transformers >= 4.52 ---
    # 1. Qwen2Model.forward accesses ``decoder_layer.attention_type``, but
    #    auto_gptq's LayerHijacker (defined inside quantize()) doesn't
    #    delegate that attribute to the wrapped module. We patch
    #    nn.Module.__getattr__ to delegate to self.module during quantization.
    # 2. Qwen2DecoderLayer.forward returns a bare Tensor (not a tuple), but
    #    auto_gptq's quantize loop does layer(...)[0] — on a bare tensor this
    #    slices the batch dimension, corrupting rotary-embedding shapes in
    #    subsequent layers. We patch forward to return (hidden_states,) as a
    #    1-tuple so [0] correctly extracts the tensor.
    # 3. GPTQ's Cholesky decomposition can fail on layers with near-singular
    #    Hessians (e.g. o_proj); we monkey-patch fasterquant to retry with
    #    escalating damping and a nan_to_num guard on add_batch.
    _patch_gptq_cholesky_resilience()

    with _patch_attention_type(), _patch_qwen2_decoder_tuple_return():
        model.quantize(
            examples=examples,
            batch_size=1,
            use_triton=False,
        )

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, use_safetensors=True)
    logger.info("[GPTQ] Quantized model saved to %s", output_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = time.time() - start
    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.empty_cache()

    actual_size = _measure_file_size_gb(output_path)

    # Try to load the quantized model for throughput measurement.
    tps = None
    measured_vram = None
    try:
        logger.info("[GPTQ] Loading quantized model for throughput measurement ...")
        q_model = AutoGPTQForCausalLM.from_quantized(
            output_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        q_model.eval()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            tps = _measure_throughput(q_model, tokenizer)
            measured_vram = torch.cuda.memory_allocated() / (1024**3)
        else:
            tps = _measure_throughput(q_model, tokenizer)

        del q_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GPTQ] Throughput measurement skipped: %s", exc)

    from app.quantization.config import estimate_vram_gb, estimate_model_size_gb, estimate_tokens_per_sec

    est_vram = estimate_vram_gb("gptq", bit_width)

    return {
        "method": "gptq",
        "bit_width": bit_width,
        "quantized_model_size_gb": actual_size,
        "estimated_vram_gb": est_vram,
        "measured_vram_gb": round(measured_vram, 2) if measured_vram else peak_vram,
        "tokens_per_sec": tps,
        "elapsed_seconds": round(elapsed, 2),
        "config": quant_config,
        "checkpoint_path": output_path,
        "notes": f"GPTQ bits={bit_width} group_size={group_size} "
        f"desc_act={desc_act} damping={damping} "
        f"quantized in {round(elapsed, 1)}s",
    }


def _run_awq(
    source_checkpoint: str,
    output_path: str,
    bit_width: int,
    config_dict: dict,
) -> dict:
    """Run real AWQ quantization using AutoAWQ."""
    from awq import AutoAWQForCausalLM

    logger.info("[AWQ] Starting %d-bit quantization ...", bit_width)
    start = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    from app.quantization.config import AWQConfig, estimate_vram_gb, estimate_model_size_gb, estimate_tokens_per_sec

    awq_cfg = AWQConfig(
        bits=bit_width,
        group_size=config_dict.get("group_size", 128),
        zero_point=config_dict.get("zero_point", True),
    )

    model = AutoAWQForCausalLM.from_pretrained(
        source_checkpoint,
        device_map="auto",
        torch_dtype="auto",
    )

    quant_config = {
        "zero_point": awq_cfg.zero_point,
        "q_order": "tloss",
        "auto": awq_cfg.zero_point,
    }

    model.quantize(
        tokenizer=None,
        quant_config=quant_config,
    )
    model.save_quantized(output_path, safetensors=True)

    elapsed = time.time() - start
    peak_vram = None
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        torch.cuda.empty_cache()

    actual_size = _measure_file_size_gb(output_path)

    # Throughput measurement.
    tps = None
    measured_vram = None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("[AWQ] Loading quantized model for throughput measurement ...")
        q_model = AutoModelForCausalLM.from_quantized(
            output_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        q_model.eval()
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct", trust_remote_code=True)  # nosec B615
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            tps = _measure_throughput(q_model, tokenizer)
            measured_vram = torch.cuda.memory_allocated() / (1024**3)
        else:
            tps = _measure_throughput(q_model, tokenizer)

        del q_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[AWQ] Throughput measurement skipped: %s", exc)

    est_vram = estimate_vram_gb("awq", bit_width)
    est_size = estimate_model_size_gb("awq", bit_width)

    return {
        "method": "awq",
        "bit_width": bit_width,
        "quantized_model_size_gb": actual_size,
        "estimated_vram_gb": est_vram,
        "measured_vram_gb": round(measured_vram, 2) if measured_vram else peak_vram,
        "tokens_per_sec": tps,
        "elapsed_seconds": round(elapsed, 2),
        "config": quant_config,
        "checkpoint_path": output_path,
        "notes": f"AWQ bits={bit_width} group_size={awq_cfg.group_size} "
        f"zero_point={awq_cfg.zero_point} "
        f"quantized in {round(elapsed, 1)}s",
    }


def _run_gguf(
    source_checkpoint: str,
    output_path: str,
    bit_width: int,
    base_model: str,
    config_dict: dict,
) -> dict:
    """Run real GGUF quantization (HF → F16 GGUF conversion → quantize).

    The source checkpoint can be a LoRA adapter or a full HF model. We first
    convert it to an F16 GGUF file (merging LoRA if needed), then quantize
    using the requested GGUF quant type.
    """
    from app.quantization.config import GGUFConfig, estimate_vram_gb, estimate_model_size_gb, estimate_tokens_per_sec
    from app.quantization.export_gguf import GGUFQuantizer, convert_hf_to_gguf_f16

    logger.info("[GGUF] Starting quantization (target bits=%d) ...", bit_width)
    start = time.time()

    # Ensure the output path has a .gguf extension.
    if not output_path.endswith(".gguf"):
        output_path = output_path + ".gguf"

    quant_type = config_dict.get("quant_type", "Q4_K")

    # Step 1: Convert HF/LoRA checkpoint → F16 GGUF (intermediate).
    f16_gguf_path = output_path.replace(".gguf", "_f16.gguf")
    logger.info("[GGUF] Converting HF checkpoint → F16 GGUF: %s", f16_gguf_path)
    convert_hf_to_gguf_f16(
        source_checkpoint=source_checkpoint,
        output_path=f16_gguf_path,
        base_model=base_model,
    )

    # Step 2: Quantize F16 GGUF → target quant type.
    gguf_cfg = GGUFConfig(quant_types=[quant_type], f16_fallback=False)
    quantizer = GGUFQuantizer(config=gguf_cfg, base_model=base_model)

    # Use the quantizer's _load and _quantize methods directly.
    backend = quantizer._load()
    if isinstance(backend, str):
        # CLI mode.
        import subprocess  # nosec B404
        subprocess.run(  # nosec B603
            [backend, f16_gguf_path, output_path, quant_type],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
    else:
        # Python API mode.
        import llama_cpp
        gguf = llama_cpp.ggml
        quantizer_obj = gguf.LlamaQuantize(quant_type)
        llama_cpp.llama_model_quantize(
            str(f16_gguf_path),
            str(output_path),
            quantizer_obj,
        )

    # Clean up intermediate F16 GGUF.
    if os.path.exists(f16_gguf_path):
        os.remove(f16_gguf_path)

    elapsed = time.time() - start
    actual_size = _measure_file_size_gb(output_path)

    # Throughput on CPU.
    tps = None
    try:
        import llama_cpp
        from transformers import AutoTokenizer

        logger.info("[GGUF] Loading quantized model for throughput measurement ...")
        llm = llama_cpp.Llama(
            model_path=output_path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,  # CPU
            verbose=False,
        )
        tps = _gguf_throughput(llm, AutoTokenizer.from_pretrained(base_model, trust_remote_code=True))  # nosec B615
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GGUF] Throughput measurement skipped: %s", exc)

    est_vram = estimate_vram_gb("gguf", bit_width)
    est_size = estimate_model_size_gb("gguf", bit_width)

    return {
        "method": "gguf",
        "bit_width": bit_width,
        "quantized_model_size_gb": actual_size,
        "estimated_vram_gb": est_vram,
        "measured_vram_gb": None,  # GGUF runs on CPU, no GPU VRAM to measure
        "tokens_per_sec": tps,
        "elapsed_seconds": round(elapsed, 2),
        "config": {"quant_type": config_dict.get("quant_type", "Q4_K")},
        "checkpoint_path": output_path,
        "notes": f"GGUF type={config_dict.get('quant_type', 'Q4_K')} bits={bit_width} "
        f"quantized in {round(elapsed, 1)}s",
    }


def _gguf_throughput(llm, tokenizer, prompt: str = "def hello(): pass", max_new_tokens: int = 32) -> float | None:
    """Measure GGUF model throughput on CPU."""
    try:
        import time as _time

        # Warmup.
        _ = llm(prompt, max_tokens=8)

        times = []
        for _ in range(3):
            t0 = _time.perf_counter()
            _ = llm(prompt, max_tokens=max_new_tokens)
            times.append(_time.perf_counter() - t0)

        avg = sum(times) / len(times)
        total_tokens = len(tokenizer.encode(prompt)) + max_new_tokens
        return round(total_tokens / avg, 2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GGUF throughput measurement failed: %s", exc)
        return None


def _quantize_single_real(
    method: str,
    bit_width: int,
    source_checkpoint: str,
    merged_dir: str,
    output_base: str,
    base_model: str,
    config_overrides: dict,
) -> dict | None:
    """Dispatch to the right real quantizer for *method*.

    Returns a measured-metrics dict, or ``None`` if the method is unavailable.
    """
    output_path = os.path.join(output_base, f"{method}_bits{bit_width}")

    if method == "gptq":
        if not _check_dep_availability("gptq"):
            logger.warning("[GPTQ] auto_gptq not installed — skipping")
            return None
        try:
            return _run_gptq(merged_dir, output_path, bit_width, config_overrides.get("gptq", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("[GPTQ] Failed: %s", exc)
            return None

    if method == "awq":
        if not _check_dep_availability("awq"):
            logger.warning("[AWQ] autoawq not installed — skipping")
            return None
        from app.quantization.config import AWQConfig
        try:
            return _run_awq(merged_dir, output_path, bit_width, config_overrides.get("awq", {}))
        except Exception as exc:  # noqa: BLE001
            logger.error("[AWQ] Failed: %s", exc)
            return None

    if method == "gguf":
        # GGUF can convert directly from the LoRA checkpoint (no merge needed).
        output_path = os.path.join(output_base, f"gguf_bits{bit_width}.gguf")
        # Map bit_width to GGUF quant type.
        gguf_map = {2: "Q2_K", 3: "Q3_K", 4: "Q4_K", 5: "Q5_K", 8: "Q8_0"}
        qt = gguf_map.get(bit_width, "Q4_K")
        try:
            return _run_gguf(source_checkpoint, output_path, bit_width, base_model, {"quant_type": qt})
        except Exception as exc:  # noqa: BLE001
            logger.error("[GGUF] Failed: %s", exc)
            return None

    return None


class _QuantizedModelBackend:
    """Lightweight backend wrapper for quantized models.

    Implements the ``ModelBackend`` Protocol (``generate(prompt) -> str``).
    GPTQ/AWQ quantized checkpoints cannot be loaded via
    ``transformers.pipeline`` (which uses ``AutoModelForCausalLM``); they
    require their respective loading APIs (``AutoGPTQForCausalLM.from_quantized``
    or ``AutoAWQForCausalLM.from_quantized``). This wrapper bridges that gap
    so the quantized checkpoint can be fed through ``run_baseline``.
    """

    def __init__(self, model, tokenizer, max_new_tokens=512, temperature=0.2):
        self._model = model
        self._tokenizer = tokenizer
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline

        self._pipeline = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            device_map="auto",
            framework="pt",
        )
        return self._pipeline

    def generate(self, prompt: str) -> str:
        pipe = self._load()
        result = pipe(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            top_p=0.95,
            do_sample=True,
            return_full_text=False,
        )
        if isinstance(result, list) and len(result) > 0:
            entry = result[0]
            text = entry.get("generated_text", entry.get("text", ""))
        else:
            text = str(result)
        return text.strip()


def _load_quantized_backend(
    quant_method: str,
    quantized_checkpoint: str,
    base_model: str,
) -> _QuantizedModelBackend:
    """Load a quantized checkpoint and return a ``_QuantizedModelBackend``.

    * GPTQ → ``AutoGPTQForCausalLM.from_quantized()``
    * AWQ  → ``AutoAWQForCausalLM.from_quantized()``
    """
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_method == "gptq":
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            quantized_checkpoint,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()
    elif quant_method == "awq":
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_quantized(
            quantized_checkpoint,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        model.eval()
    else:
        raise ValueError(f"Unsupported quant method for re-eval: {quant_method}")

    logger.info("[Re-Eval] Loaded %s quantized model from %s", quant_method.upper(), quantized_checkpoint)
    return _QuantizedModelBackend(model, tokenizer)


def _reevaluate_with_stage6(
    quantized_checkpoint: str,
    quant_method: str,
    bit_width: int,
    base_model: str,
    gold_eval_path: str,
    output_dir: str,
) -> dict | None:
    """Re-evaluate a quantized checkpoint through Stage 6 on the gold-eval set.

    Loads the quantized model using its specific API (``from_quantized``) and
    wraps it in a ``_QuantizedModelBackend`` so it can be passed to
    ``run_baseline``.

    Returns a dict with ``model_cwe_macro_f1`` and ``exec_pass_rate``, or
    ``None`` if evaluation fails.
    """
    logger.info("[Re-Eval] Evaluating %s@%d-bit on gold-eval set ...", quant_method, bit_width)

    try:
        from app.evaluation.baseline import BaselineConfig, run_baseline

        # GGUF requires a llama.cpp backend, not HF transformers — skip.
        if quant_method == "gguf":
            logger.info("[Re-Eval] GGUF re-eval not available via QwenBackend — skipping")
            return None

        backend = _load_quantized_backend(quant_method, quantized_checkpoint, base_model)

        config = BaselineConfig(
            strategy="zero_shot",
            base_model=f"{quant_method}_{bit_width}bit",
        )

        result = run_baseline(
            gold_eval_path=gold_eval_path,
            output_dir=os.path.join(output_dir, f"reeval_{quant_method}_bits{bit_width}"),
            config=config,
            backend=backend,
        )

        return {
            "model_cwe_macro_f1": result.metrics.cwe_macro_f1,
            "exec_pass_rate": result.metrics.patch_coverage,
            "cwe_micro_accuracy": result.metrics.cwe_micro_accuracy,
            "severity_accuracy": result.metrics.severity_accuracy,
            "hallucination_rate": result.metrics.hallucination_rate,
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Re-Eval] Evaluation failed for %s@%d-bit: %s", quant_method, bit_width, exc)
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Stage 8 — real quantization matrix (GPTQ / AWQ / GGUF)",
    )
    ap.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})",
    )
    ap.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to Stage 5 trained checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    ap.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for QuantReport JSON",
    )
    ap.add_argument(
        "--methods",
        default="gptq,awq,gguf",
        help="Comma-separated methods (gptq, awq, gguf, none)",
    )
    ap.add_argument(
        "--bits",
        default="2,3,4,8",
        help="Comma-separated bit-widths for GPTQ/AWQ",
    )
    ap.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip Stage 6 re-evaluation of quantized checkpoints (faster)",
    )
    ap.add_argument(
        "--target-vram",
        type=float,
        default=None,
        help="VRAM budget filter for best-config selection (GB)",
    )
    ap.add_argument(
        "--target-size",
        type=float,
        default=None,
        help="On-disk size budget filter for best-config selection (GB)",
    )
    ap.add_argument(
        "--calib-dataset",
        default=None,
        help="Path to calibration dataset JSONL (default: Stage 3 train.jsonl)",
    )
    ap.add_argument(
        "--gold-eval",
        default=DEFAULT_GOLD_EVAL,
        help=f"Path to gold-eval JSONL for re-evaluation (default: {DEFAULT_GOLD_EVAL})",
    )
    ap.add_argument(
        "--no-gptq",
        action="store_true",
        help="Skip GPTQ quantization",
    )
    ap.add_argument(
        "--no-awq",
        action="store_true",
        help="Skip AWQ quantization",
    )
    ap.add_argument(
        "--no-gguf",
        action="store_true",
        help="Skip GGUF quantization",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not os.path.exists(args.checkpoint):
        logger.error("Checkpoint not found: %s", args.checkpoint)
        logger.error("Run Stage 5 training first:")
        logger.error("  python scripts/run_gpu_training.py")
        sys.exit(1)

    assert torch.cuda.is_available(), (  # nosec B101 — runtime guard
        "CUDA GPU required for real quantization. Use --mock for testing."
    )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory // 1024 // 1024
    logger.info("GPU: %s (%d MB VRAM)", gpu_name, gpu_vram)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Parse methods and bits
    # ------------------------------------------------------------------
    methods = []
    for m in args.methods.split(","):
        m = m.strip().lower()
        if m:
            methods.append(m)

    if args.no_gptq and "gptq" in methods:
        methods.remove("gptq")
    if args.no_awq and "awq" in methods:
        methods.remove("awq")
    if args.no_gguf and "gguf" in methods:
        methods.remove("gguf")

    bit_widths = [int(b) for b in args.bits.split(",") if b.strip()]

    # Calib dataset for GPTQ.
    calib_path = args.calib_dataset
    if not calib_path:
        default_calib = "output/stage3/train.jsonl"
        if os.path.exists(default_calib):
            calib_path = default_calib

    config_overrides = {
        "gptq": {"calib_dataset": calib_path} if calib_path else {},
        "awq": {},
        "gguf": {},
    }

    # ------------------------------------------------------------------
    # Step 1: Merge LoRA → full-precision HF checkpoint
    # ------------------------------------------------------------------
    logger.info("=== Stage 8: Real Quantization Matrix ===")
    logger.info("Base model:   %s", args.base_model)
    logger.info("Checkpoint:   %s", args.checkpoint)
    logger.info("Methods:      %s", methods)
    logger.info("Bit widths:   %s", bit_widths)
    logger.info("Skip eval:    %s", args.skip_eval)
    logger.info("")

    is_lora = os.path.exists(os.path.join(args.checkpoint, "adapter_config.json"))
    merged_dir = os.path.join(args.output_dir, "_merged_model")

    if is_lora:
        logger.info("LoRA checkpoint detected — merging adapter into base model ...")
        _merge_lora_to_hf(args.base_model, args.checkpoint, merged_dir)
        quant_source = merged_dir
    else:
        logger.info("Full HF checkpoint — using directly")
        quant_source = args.checkpoint

    # ------------------------------------------------------------------
    # Step 2: Check dependency availability
    # ------------------------------------------------------------------
    logger.info("")
    for m in methods:
        available = _check_dep_availability(m)
        if available:
            logger.info("  [%s] ✓ available", m.upper())
        else:
            logger.info("  [%s] ✗ not installed (will skip)", m.upper())

    available_methods = [m for m in methods if _check_dep_availability(m)]
    skipped_methods = [m for m in methods if not _check_dep_availability(m)]

    if not available_methods:
        logger.error("No quantization methods available — install at least one:")
        logger.error("  pip install auto-gptq    (for GPTQ)")
        logger.error("  pip install autoawq      (for AWQ)")
        logger.error("  pip install llama-cpp-python  (for GGUF)")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Run quantization matrix
    # ------------------------------------------------------------------
    from app.schemas.quantization import QuantMethod, QuantResult, QuantStatus
    from app.quantization.config import (
        estimate_vram_gb,
        estimate_model_size_gb,
        estimate_quality,
        estimate_tokens_per_sec,
    )
    from app.quantization.quantizer import score_quality_size_speed, select_best_config

    run_id = f"stage8-real-{int(time.time())}"
    start_time = time.time()
    results: list[QuantResult] = []

    for method in available_methods:
        if method == "gguf":
            # GGUF uses quant types, not bit_widths directly.
            for bits in bit_widths:
                logger.info("")
                logger.info("--- %s @ %d-bit ---", method.upper(), bits)
                measured = _quantize_single_real(
                    method=method,
                    bit_width=bits,
                    source_checkpoint=args.checkpoint,
                    merged_dir=quant_source,
                    output_base=args.output_dir,
                    base_model=args.base_model,
                    config_overrides=config_overrides,
                )
                if measured:
                    result = QuantResult(
                        quant_method=QuantMethod.GGUF,
                        bit_width=bits,
                        quantized_model_size_gb=measured["quantized_model_size_gb"],
                        estimated_vram_gb=measured["estimated_vram_gb"],
                        measured_vram_gb=measured["measured_vram_gb"],
                        tokens_per_sec=measured["tokens_per_sec"],
                        model_cwe_macro_f1=None,  # set by re-eval
                        exec_pass_rate=None,
                        status=QuantStatus.COMPLETED,
                        checkpoint_path=measured["checkpoint_path"],
                        notes=measured["notes"],
                    )
                    results.append(result)
        else:
            for bits in bit_widths:
                logger.info("")
                logger.info("--- %s @ %d-bit ---", method.upper(), bits)
                measured = _quantize_single_real(
                    method=method,
                    bit_width=bits,
                    source_checkpoint=args.checkpoint,
                    merged_dir=quant_source,
                    output_base=args.output_dir,
                    base_model=args.base_model,
                    config_overrides=config_overrides,
                )
                if measured:
                    # Optional re-evaluation.
                    quality_metrics = None
                    if not args.skip_eval and measured:
                        quality_metrics = _reevaluate_with_stage6(
                            quantized_checkpoint=measured["checkpoint_path"],
                            quant_method=method,
                            bit_width=bits,
                            base_model=args.base_model,
                            gold_eval_path=args.gold_eval if hasattr(args, "gold_eval") else DEFAULT_GOLD_EVAL,
                            output_dir=args.output_dir,
                        )

                    result = QuantResult(
                        quant_method=QuantMethod(method),
                        bit_width=bits,
                        quantized_model_size_gb=measured["quantized_model_size_gb"],
                        estimated_vram_gb=measured["estimated_vram_gb"],
                        measured_vram_gb=measured["measured_vram_gb"],
                        tokens_per_sec=measured["tokens_per_sec"],
                        model_cwe_macro_f1=quality_metrics["model_cwe_macro_f1"] if quality_metrics else None,
                        exec_pass_rate=quality_metrics["exec_pass_rate"] if quality_metrics else None,
                        status=QuantStatus.COMPLETED,
                        checkpoint_path=measured["checkpoint_path"],
                        notes=measured["notes"],
                    )
                    results.append(result)

    # ------------------------------------------------------------------
    # Step 4: Best-config selection
    # ------------------------------------------------------------------
    best = select_best_config(
        results,
        target_vram_gb=args.target_vram,
        target_size_gb=args.target_size,
    )

    elapsed = time.time() - start_time

    manifest = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "base_model": args.base_model,
        "source_checkpoint": args.checkpoint,
        "checkpoint_type": "lora" if is_lora else "full_model",
        "methods_requested": [m for m in methods],
        "methods_attempted": available_methods,
        "methods_skipped": skipped_methods,
        "bit_widths": bit_widths,
        "dry_run": False,
        "mock": False,
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram,
        "calib_dataset": calib_path,
        "skip_eval": args.skip_eval,
    }

    from app.schemas.quantization import QuantReport

    report = QuantReport(
        run_id=run_id,
        base_model=args.base_model,
        source_checkpoint=args.checkpoint,
        results=results,
        best_result=best,
        manifest=manifest,
    )

    # ------------------------------------------------------------------
    # Step 5: Write QuantReport
    # ------------------------------------------------------------------
    report_path = os.path.join(args.output_dir, "quant_report.json")
    report_data = json.loads(report.model_dump_json(indent=2))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    logger.info("QuantReport written to %s", report_path)

    # ------------------------------------------------------------------
    # Step 6: Write a summary JSON for Stage 10 / Stage 11
    # ------------------------------------------------------------------
    summary_path = os.path.join(args.output_dir, "quant_summary.json")
    summary = {
        "run_id": run_id,
        "base_model": args.base_model,
        "source_checkpoint": args.checkpoint,
        "checkpoint_type": "lora" if is_lora else "full_model",
        "methods_attempted": available_methods,
        "methods_skipped": skipped_methods,
        "elapsed_seconds": round(elapsed, 2),
        "gpu_name": gpu_name,
        "results": [
            {
                "method": r.quant_method.value,
                "bit_width": r.bit_width,
                "size_gb": r.quantized_model_size_gb,
                "vram_gb": r.measured_vram_gb or r.estimated_vram_gb,
                "tokens_per_sec": r.tokens_per_sec,
                "cwe_macro_f1": r.model_cwe_macro_f1,
                "exec_pass_rate": r.exec_pass_rate,
            }
            for r in results
        ],
        "best": {
            "method": best.quant_method.value if best else None,
            "bit_width": best.bit_width if best else None,
            "size_gb": best.quantized_model_size_gb if best else None,
            "vram_gb": best.measured_vram_gb or best.estimated_vram_gb if best else None,
        } if best else None,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Summary written to %s", summary_path)

    # ------------------------------------------------------------------
    # Step 7: Clean up merged model (save disk space)
    # ------------------------------------------------------------------
    if is_lora and os.path.exists(merged_dir):
        logger.info("Cleaning up merged model directory: %s", merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print()
    print("=== Stage 8 Complete ===")
    print(f"  Run ID:      {run_id}")
    print(f"  Base model:  {args.base_model}")
    print(f"  GPU:         {gpu_name} ({gpu_vram} MB)")
    print(f"  Methods:     attempted={available_methods}, skipped={skipped_methods}")
    print(f"  Total configs tried: {len(results)}")
    print(f"  Elapsed:     {elapsed:.1f}s")
    print()

    if best:
        print("  Best config:")
        print(f"    Method:      {best.quant_method.value}")
        print(f"    Bit width:   {best.bit_width}")
        print(f"    Size:        {best.quantized_model_size_gb} GB")
        if best.measured_vram_gb:
            print(f"    Meas. VRAM:  {best.measured_vram_gb:.2f} GB")
        print(f"    Est. VRAM:   {best.estimated_vram_gb} GB")
        if best.tokens_per_sec:
            print(f"    Throughput:  {best.tokens_per_sec} t/s")
        print(f"    Path:        {best.checkpoint_path}")

    print()
    print("  Per-config results:")
    for r in results:
        status_icon = "[OK]" if r.status == QuantStatus.COMPLETED else "[XX]"
        vram = f"{r.measured_vram_gb:.2f}" if r.measured_vram_gb else f"~{r.estimated_vram_gb}"
        tps = f"{r.tokens_per_sec} t/s" if r.tokens_per_sec else "N/A"
        f1 = f"{r.model_cwe_macro_f1:.4f}" if r.model_cwe_macro_f1 is not None else "N/A"
        print(f"    {status_icon} {r.quant_method.value:5s} @ {str(r.bit_width or '?'):>2s}-bit  "
              f"size={r.quantized_model_size_gb:>5.2f}GB  VRAM={vram:>5s}GB  tps={tps:>8s}  F1={f1}")
    print()
    print(f"  Report:   {report_path}")
    print(f"  Summary:  {summary_path}")


if __name__ == "__main__":
    main()
