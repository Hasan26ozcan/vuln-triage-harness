"""Model backend abstraction for Stage 4 (pre-fine-tuning baseline) and
Stage 6 (four-tier evaluation).

The baseline evaluation needs to call a code LLM and get back a text response.
Rather than hard-code a single inference path (``transformers``, ``vLLM``,
``Ollama``, etc.), we define a ``ModelBackend`` Protocol with a single
``generate`` method. This makes every downstream component (prompt builder,
parser, metrics) trivially testable with a mock — the same pattern used by
``TokenCounter`` in Stage 3 and ``EmbeddingBackend`` in Stage 2.

The default production backend (``QwenBackend``) uses ``transformers`` to load
the project's base model (Qwen2.5-Coder-7B-Instruct). If ``transformers`` is
not installed or the model can't be downloaded, the backend raises a clear
``RuntimeError`` — callers should fall back to ``MockBackend`` for tests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

# The project's base model, as stated in the README tech stack.
DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Generation defaults — conservative so we don't blow past context windows.
DEFAULT_MAX_NEW_TOKENS: int = 2048
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TOP_P: float = 0.95


class MissingAdapterWeightsError(RuntimeError):
    """Raised when a LoRA checkpoint directory has adapter_config.json but
    no adapter_model.safetensors/.bin — i.e. the checkpoint is incomplete
    and can't actually be applied. Callers that need real fine-tuned
    weights (Stage 7 regression, Stage 6 eval, etc.) should let this
    propagate rather than catching it, so an incomplete checkpoint fails
    the run instead of silently benchmarking the base model.
    """


class ModelBackend(Protocol):
    """Anything that can take a prompt string and return a generated string.

    The ``generate`` method is intentionally minimal: a single prompt in, one
    text response out. All decoding parameters are captured at construction time
    so they're consistent across a run.
    """

    def generate(self, prompt: str) -> str: ...


class QwenBackend:
    """Production backend using ``transformers`` to run Qwen2.5-Coder.

    Loads the model lazily on first ``generate`` call, so simply constructing
    the backend is cheap (no model download until actually used).

    Parameters
    ----------
    model_name:
        HuggingFace model ID or local checkpoint path. Defaults to the
        project's base model.
    base_model:
        When ``model_name`` is a PEFT/LoRA adapter directory, this must be
        the base model ID (e.g. ``"Qwen/Qwen2.5-Coder-1.5B-Instruct"``) to
        load first before applying the adapter. When ``None``, ``model_name``
        is loaded directly as a full model.
    max_new_tokens:
        Maximum tokens to generate in the response.
    temperature:
        Sampling temperature — low for deterministic classification.
    top_p:
        Nucleus sampling threshold.
    device:
        Device to load the model on (``"auto"`` detects GPU).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_BASE_MODEL,
        base_model: str | None = None,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        device: str = "auto",
        allow_base_fallback: bool = False,
    ):
        self.model_name = model_name
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self._pipeline = None
        # When True, a LoRA checkpoint with a missing adapter weights file
        # silently falls back to evaluating the base model. Defaults to
        # False: an incomplete checkpoint is a hard error, because a silent
        # fallback here previously produced a Stage 7 "tuned" report that
        # was actually just the base model evaluated against itself
        # (forgetting_delta == 0.0 for the wrong reason).
        self.allow_base_fallback = allow_base_fallback
        # Set to True only once a LoRA adapter has actually been resolved
        # and merged into the loaded pipeline. Callers (e.g. Stage 7) should
        # check this rather than trusting a pre-load file-existence check,
        # since it reflects what was actually loaded, not just what was on
        # disk before loading started.
        self.adapter_applied: bool = False

    def _load(self):
        """Lazy-load the HuggingFace text-generation pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "transformers is not installed. Run "
                "`pip install -e '.[ml]'` to use the QwenBackend."
            ) from exc

        import os as _os

        # Detect PEFT/LoRA adapter checkpoint (adapter_config.json present,
        # no config.json for full model).  We also require the adapter
        # weights file to exist — a bare adapter_config.json with no
        # ``adapter_model.safetensors`` / ``.bin`` means the checkpoint is
        # incomplete (e.g. weights excluded from the repo via .gitignore)
        # and PEFT's ``from_pretrained`` would raise a confusing
        # ``HFValidationError``.
        has_lora_config = _os.path.exists(_os.path.join(self.model_name, "adapter_config.json"))
        has_adapter_weights = has_lora_config and (
            _os.path.exists(_os.path.join(self.model_name, "adapter_model.safetensors"))
            or _os.path.exists(_os.path.join(self.model_name, "adapter_model.bin"))
        )

        if has_adapter_weights and self.base_model:
            # Resolve to an absolute path so PEFT/huggingface_hub treats it
            # as a local directory rather than a repo ID.
            lora_path = str(Path(self.model_name).resolve())
            logger.info(
                "Loading LoRA checkpoint %s on top of %s",
                lora_path,
                self.base_model,
            )
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model = AutoModelForCausalLM.from_pretrained(  # nosec B615
                self.base_model,
                device_map=self.device,
                trust_remote_code=True,
            )
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
                self.base_model, trust_remote_code=True
            )

            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map=self.device,
            )
            self.adapter_applied = True
        elif has_lora_config and self.base_model:
            # LoRA adapter config exists but the weights file is missing —
            # the checkpoint is incomplete (e.g. training didn't finish, or
            # .safetensors/.bin were stripped before this run). This used to
            # silently fall back to the base model, which made "tuned model"
            # evaluations secretly evaluate the base model instead — a bug
            # that produced a false forgetting_delta of 0.0 in Stage 7.
            # Fail loudly by default; only proceed if the caller explicitly
            # opted into the fallback.
            msg = (
                f"LoRA adapter config found at {self.model_name!r} but no "
                f"adapter_model.safetensors/.bin file is present. Refusing "
                f"to silently evaluate the base model ({self.base_model!r}) "
                f"as if it were the fine-tuned checkpoint. Verify the "
                f"checkpoint was saved correctly (see "
                f"scripts/verify_checkpoint.py), or pass "
                f"allow_base_fallback=True if you deliberately want to "
                f"benchmark the base model under this backend."
            )
            if not self.allow_base_fallback:
                raise MissingAdapterWeightsError(msg)
            logger.warning("%s (proceeding: allow_base_fallback=True)", msg)
            logger.info("Loading model %s on device=%s", self.base_model, self.device)
            self._pipeline = pipeline(
                "text-generation",
                model=self.base_model,
                device_map=self.device,
            )
            self.adapter_applied = False
        else:
            logger.info("Loading model %s on device=%s", self.model_name, self.device)
            self._pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                device_map=self.device,
            )
            self.adapter_applied = False
        return self._pipeline

    def generate(self, prompt: str) -> str:
        """Generate a response from the model for the given prompt."""
        pipe = self._load()
        result = pipe(
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
            return_full_text=False,
        )
        # transformers pipeline returns a list of dicts; each dict has
        # either "generated_text" (single str) or "text" (older versions).
        if isinstance(result, list) and len(result) > 0:
            entry = result[0]
            text = entry.get("generated_text", entry.get("text", ""))
        else:
            text = str(result)
        return text.strip()


class MockBackend:
    """Deterministic backend for testing — returns a canned response.

    The ``responses`` dict maps a key (or callable) to a canned output string.
    In its simplest form, pass a single string and every ``generate`` call
    returns it. For per-sample control, pass a dict keyed by a substring of
    the prompt.

    Parameters
    ----------
    responses:
        A dict mapping prompt-substring -> response string. If a prompt
        contains any key as a substring, that response is returned.
        If none match, ``default`` is returned.
    default:
        Fallback response when no key matches.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "{}",
    ):
        self._responses = responses or {}
        self._default = default
        self.call_count = 0
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.calls.append(prompt[:80] + "..." if len(prompt) > 80 else prompt)
        for key, response in self._responses.items():
            if key in prompt:
                return response
        return self._default
