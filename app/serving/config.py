"""Stage 9 — serving configuration.

Mirrors the dataclass-config pattern from ``app.quantization.config`` (Stage 8)
and ``app.training.config`` (Stage 5): a flat, mutable dataclass with sensible
defaults drawn from the project README tech-stack table (``llama.cpp`` for
air-gapped CPU inference, ``Ollama`` as an alternative runtime).

All backend imports (``llama_cpp``, ``httpx``) are performed **inside**
the backend implementations, never at module-import time, so that this
module — and the CLI — work without the serving backends installed. This
follows the same lazy-import pattern as ``QwenBackend._load`` (Stage 4)
and ``TokenCounter._load`` (Stage 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — defaults from the project README tech-stack table
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH: str = ""  # must be provided for real backends
DEFAULT_BACKEND_TYPE: str = "llama.cpp"  # llama.cpp | llama-server | ollama | mock

# llama.cpp defaults (Qwen2.5-Coder-7B GGUF)
DEFAULT_NUM_CTX: int = 4096      # context length
DEFAULT_NUM_THREADS: int = 4     # CPU threads for inference
DEFAULT_N_GPU_LAYERS: int = 0    # CPU-only by default (air-gapped)
DEFAULT_F16_KV: bool = True      # keep KV cache in fp16 to save VRAM

# Generation defaults — same as Stage 4 QwenBackend
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_MAX_NEW_TOKENS: int = 2048

# Network defaults
# Air-gapped/local serving default; overridable via config/CLI
DEFAULT_HOST: str = "0.0.0.0"  # nosec B104
DEFAULT_PORT: int = 8000

# Backend choices
_VALID_BACKEND_TYPES: frozenset[str] = frozenset({
    "llama.cpp", "llama-server", "ollama", "mock",
})


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class ServingConfig:
    """Configuration for the Stage 9 air-gapped serving layer.

    Attributes
    ----------
    model_path:
        Path/URI to the quantized checkpoint (GGUF file from Stage 8,
        or an Ollama model name). Empty string is allowed only when
        ``backend_type="mock"``.
    backend_type:
        ``"llama.cpp"`` (load GGUF via ``llama-cpp-python``),
        ``"llama-server"`` (spawn ``llama-server`` binary + HTTP),
        ``"ollama"`` (call the local Ollama HTTP API), or
        ``"mock"`` (deterministic test backend).
    num_ctx:
        Context window size passed to the llama.cpp backend.
    num_threads:
        Number of CPU threads for llama.cpp inference.
    n_gpu_layers:
        Number of model layers to offload to GPU. 0 = CPU-only.
    f16_kv:
        Whether to keep the KV cache in fp16 (saves memory).
    temperature:
        Sampling temperature for generation.
    max_new_tokens:
        Maximum tokens to generate per request.
    host:
        Bind address for the uvicorn server (CLI/API mode).
    port:
        Bind port for the uvicorn server (CLI/API mode).
    request_timeout:
        Per-request timeout in seconds when using the Ollama backend.
    """

    model_path: str = DEFAULT_MODEL_PATH
    backend_type: str = DEFAULT_BACKEND_TYPE
    num_ctx: int = DEFAULT_NUM_CTX
    num_threads: int = DEFAULT_NUM_THREADS
    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS
    f16_kv: bool = DEFAULT_F16_KV
    temperature: float = DEFAULT_TEMPERATURE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    request_timeout: float = 30.0

    def __post_init__(self) -> None:
        """Validate after dataclass init."""
        if self.backend_type not in _VALID_BACKEND_TYPES:
            raise ValueError(
                f"backend_type={self.backend_type!r} — "
                f"valid: {sorted(_VALID_BACKEND_TYPES)}"
            )

    @property
    def run_name(self) -> str:
        """A human-readable label for this serving configuration."""
        if self.model_path:
            name = self.model_path.split("/")[-1].split("\\")[-1]
            return f"serve_{name}_{self.backend_type}"
        return f"serve_mock_{self.backend_type}"

    def all_warnings(self) -> list[str]:
        """Collect validation warnings for this config."""
        warnings: list[str] = []

        if self.backend_type != "mock" and not self.model_path:
            warnings.append(
                f"model_path is empty for backend_type={self.backend_type!r} — "
                "the backend will fail to load a model."
            )

        if self.backend_type in ("llama.cpp", "ollama") and self.num_threads < 1:
            warnings.append(f"num_threads={self.num_threads} — should be >= 1.")

        if self.max_new_tokens < 1:
            warnings.append(f"max_new_tokens={self.max_new_tokens} — should be >= 1.")

        if self.num_ctx < 512:
            warnings.append(
                f"num_ctx={self.num_ctx} — very small context window; "
                "consider >= 512 for vulnerability analysis."
            )

        return warnings

    def is_mock(self) -> bool:
        """Return True if this config uses the mock backend."""
        return self.backend_type == "mock"
