"""Stage 9 — serving backends.

Defines the ``ServingBackend`` Protocol (same injectable-backend pattern as
``ModelBackend`` in Stage 4/6/7, ``EmbeddingBackend`` in Stage 2, ``Quantizer``
in Stage 8) and three implementations:

* ``LlamaCppBackend`` — loads a GGUF checkpoint via ``llama-cpp-python``
  (CPU/GPU, the air-gapped default per the README tech-stack table).
* ``OllamaBackend`` — calls the local Ollama HTTP API (``http://localhost:11434``).
* ``MockServingBackend`` — deterministic, no-deps backend for testing.

Heavy imports (``llama_cpp``, ``httpx``) are performed inside
``_load()`` / ``generate()`` — never at module-import time — so this
module is import-safe without those packages installed.
"""

from __future__ import annotations

import logging
import urllib.parse
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

__all__ = [
    "ServingBackend",
    "LlamaCppBackend",
    "OllamaBackend",
    "MockServingBackend",
]


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ServingBackend(Protocol):
    """Anything that can take a prompt string and return a generated string.

    This is structurally identical to ``ModelBackend`` but lives in the
    serving package to avoid a circular import (evaluation imports serving,
    serving does not import evaluation). Tests inject ``MockServingBackend``;
    production code uses ``LlamaCppBackend`` or ``OllamaBackend``.
    """

    def generate(self, prompt: str) -> str: ...

    @property
    def model_info(self) -> dict: ...


# ---------------------------------------------------------------------------
# LlamaCppBackend — real GGUF serving via llama-cpp-python
# ---------------------------------------------------------------------------


class LlamaCppBackend:
    """Serving backend that loads a GGUF checkpoint via ``llama-cpp-python``.

    This is the **default** air-gapped / CPU-only backend for the project:
    a quantized GGUF model (produced by Stage 8) is loaded with
    ``llama-cpp-python``, which has no external network dependency at
    runtime.

    Parameters
    ----------
    model_path:
        Filesystem path to the ``.gguf`` checkpoint (from Stage 8).
    num_ctx:
        Context window size.
    num_threads:
        Number of CPU threads for inference.
    n_gpu_layers:
        Number of transformer layers to offload to the GPU (0 = CPU only).
    f16_kv:
        Keep the KV cache in fp16 to save memory.
    temperature:
        Sampling temperature.
    max_new_tokens:
        Maximum tokens to generate per request.
    """

    def __init__(
        self,
        model_path: str,
        num_ctx: int = 4096,
        num_threads: int = 4,
        n_gpu_layers: int = 0,
        f16_kv: bool = True,
        temperature: float = 0.2,
        max_new_tokens: int = 2048,
    ):
        self.model_path = model_path
        self.num_ctx = num_ctx
        self.num_threads = num_threads
        self.n_gpu_layers = n_gpu_layers
        self.f16_kv = f16_kv
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._llm: object | None = None  # Llama instance, created on first use

    def _load(self) -> object:
        """Lazy-load the ``Llama`` class from ``llama-cpp-python``.

        Returns the ``Llama`` class so callers instantiate it. Raises
        ``RuntimeError`` if ``llama-cpp-python`` is not installed.
        """
        try:
            from llama_cpp import Llama  # noqa: F401
            return Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Run "
                "`pip install llama-cpp-python` to use LlamaCppBackend, "
                "or use backend_type='mock' for testing."
            ) from exc

    @property
    def model_info(self) -> dict:
        """Return metadata about the loaded model configuration."""
        return {
            "backend": "llama.cpp",
            "model_path": self.model_path,
            "num_ctx": self.num_ctx,
            "num_threads": self.num_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "f16_kv": self.f16_kv,
        }

    def generate(self, prompt: str) -> str:
        """Generate a response from the GGUF model for *prompt*."""
        if self._llm is None:
            LlamaCls = self._load()
            self._llm = LlamaCls(
                model_path=self.model_path,
                n_ctx=self.num_ctx,
                n_threads=self.num_threads,
                n_gpu_layers=self.n_gpu_layers,
                f16_kv=self.f16_kv,
            )

        # ``self._llm`` is now a Llama instance (created by the class above).
        output = self._llm(  # type: ignore[operator]
            prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            stop=["```\n\n"],
            echo=False,
        )

        # llama-cpp-python returns a dict with "choices" or "generation" key
        # depending on the version. Handle both formats.
        if isinstance(output, dict):
            choices = output.get("choices", [])
            if choices and isinstance(choices[0], dict):
                text = choices[0].get("text", "")
            else:
                text = output.get("generation", "")
        elif isinstance(output, str):
            text = output
        else:
            text = str(output)

        return text.strip()


# ---------------------------------------------------------------------------
# OllamaBackend — calling a local Ollama HTTP API
# ---------------------------------------------------------------------------


class OllamaBackend:
    """Serving backend that calls the local Ollama HTTP API.

    Ollama runs as a local daemon (``http://localhost:11434`` by default).
    This backend sends a chat-completion-style request and returns the
    first response message. No external network is needed if Ollama is
    running locally — this is the air-gapped-friendly alternative to
    ``llama.cpp`` when the user prefers the Ollama runtime.

    Parameters
    ----------
    model:
        The Ollama model name (e.g. ``"qwen2.5-coder:7b-base-gguf"``).
    host:
        The Ollama API base URL (default ``http://localhost:11434``).
    temperature:
        Sampling temperature.
    max_new_tokens:
        Maximum tokens to generate per request.
    request_timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_new_tokens: int = 2048,
        request_timeout: float = 30.0,
    ):
        self.model = model
        self.host = host
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.request_timeout = request_timeout
        self._client: object | None = None  # httpx.Client, created on first use

    def _load(self) -> object:
        """Lazy-import ``httpx`` and return a client factory.

        Raises ``RuntimeError`` if ``httpx`` is not installed.
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is not installed. Run "
                "`pip install httpx` to use OllamaBackend, "
                "or use backend_type='mock' for testing."
            ) from exc
        return httpx

    @property
    def model_info(self) -> dict:
        """Return metadata about the Ollama configuration."""
        return {
            "backend": "ollama",
            "model": self.model,
            "host": self.host,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }

    def generate(self, prompt: str) -> str:
        """Generate a response from the Ollama model for *prompt*."""
        httpx = self._load()

        if self._client is None:
            self._client = httpx.Client(timeout=self.request_timeout)

        url = urllib.parse.urljoin(self.host, "/api/chat")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            },
            "stream": False,
        }

        resp = self._client.post(url, json=body)  # type: ignore[union-attr]
        resp.raise_for_status()
        data = resp.json()

        # Ollama chat format: data["message"]["content"]
        if isinstance(data, dict):
            message = data.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                return content.strip()
        return ""


# ---------------------------------------------------------------------------
# MockServingBackend — deterministic, no external deps
# ---------------------------------------------------------------------------


class MockServingBackend:
    """Deterministic serving backend for testing.

    Mirrors ``MockBackend`` from ``app.evaluation.backends``: returns a
    canned response string for every ``generate`` call. The ``responses``
    dict maps a prompt-substring key to a response, so different samples
    can get different canned outputs in a single test.

    Parameters
    ----------
    responses:
        A dict mapping a substring → response string. If a prompt
        contains any key as a substring, that response is returned.
    default:
        Fallback response when no key matches.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = '{"cwe_id": "CWE-89", "severity": "high", '
        '"explanation": "SQL injection via string concatenation.", '
        '"patch_diff": ""}',
    ):
        self._responses = responses or {}
        self._default = default
        self.call_count = 0
        self.calls: list[str] = []

    @property
    def model_info(self) -> dict:
        """Return mock backend metadata."""
        return {
            "backend": "mock",
            "model_path": "mock",
        }

    def generate(self, prompt: str) -> str:
        """Return the canned response for *prompt*."""
        self.call_count += 1
        self.calls.append(prompt[:80] + "..." if len(prompt) > 80 else prompt)
        for key, response in self._responses.items():
            if key in prompt:
                return response
        return self._default
