"""Stage 9 — serving backends.

Defines the ``ServingBackend`` Protocol (same injectable-backend pattern as
``ModelBackend`` in Stage 4/6/7, ``EmbeddingBackend`` in Stage 2, ``Quantizer``
in Stage 8) and three implementations:

* ``LlamaCppBackend`` — loads a GGUF checkpoint via ``llama-cpp-python``
  (CPU/GPU, the air-gapped default per the README tech-stack table).
* ``LlamaServerBackend`` — spawns a ``llama-server`` subprocess (useful
  when ``llama-cpp-python`` can't be installed, e.g. no C compiler on Windows).
* ``TransformersBackend`` — loads a HuggingFace-format model
  (``model.safetensors`` / ``*.bin``) via ``transformers`` + ``torch``;
  used as a GPU-capable fallback when the llama.cpp stack is unavailable.
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

from app.security.paths import validate_path

logger = logging.getLogger(__name__)

__all__ = [
    "ServingBackend",
    "LlamaCppBackend",
    "LlamaServerBackend",
    "TransformersBackend",
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
# LlamaServerBackend — llama.cpp server binary via HTTP (/completion)
# ---------------------------------------------------------------------------


class LlamaServerBackend:
    """Serving backend that spawns a ``llama.cpp`` server subprocess and
    communicates with it over the local HTTP API.

    This backend is the air-gapped-friendly alternative to ``LlamaCppBackend``
    when ``llama-cpp-python`` cannot be installed (e.g. no C compiler on
    Windows).  It manages the lifecycle of the ``llama-server.exe`` binary:
    starting it as a subprocess, waiting for the HTTP readiness probe, and
    shutting it down on ``close()``.

    The llama.cpp server exposes a simple REST API:
    ``GET /health`` for readiness,
    ``POST /completion`` with ``{"prompt": ..., "n_predict": ..., "temperature": ...}``
    returning ``{"content": "...", ...}``.

    Parameters
    ----------
    model_path:
        Filesystem path to the ``.gguf`` checkpoint (from Stage 8).
    server_binary:
        Path to the ``llama-server`` / ``llama-server.exe`` binary.
    host:
        Bind address for the server subprocess.
    port:
        Port for the server subprocess.
    num_threads:
        Number of CPU threads (passed to the binary via ``--threads``).
    num_ctx:
        Context window size (passed via ``--ctx-size``).
    temperature:
        Sampling temperature for generation.
    max_new_tokens:
        Maximum tokens to generate per request.
    request_timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model_path: str,
        server_binary: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        num_threads: int = 4,
        num_ctx: int = 4096,
        temperature: float = 0.2,
        max_new_tokens: int = 2048,
        request_timeout: float = 30.0,
    ):
        self.model_path = model_path
        self.server_binary = server_binary or _find_llama_server()
        self.host = host
        self.port = port
        self.num_threads = num_threads
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.request_timeout = request_timeout
        self._process: object | None = None  # subprocess.Popen
        self._client: object | None = None  # httpx.Client

    def _ensure_running(self) -> None:
        """Start the llama-server subprocess if not already running."""
        import os
        import subprocess  # nosec B404
        import time as _time

        if self._process is not None:
            return

        if not self.server_binary or not os.path.exists(self.server_binary):
            raise RuntimeError(
                f"llama-server binary not found at {self.server_binary!r}. "
                "Set server_binary to a valid path to llama-server.exe."
            )
        if not os.path.exists(self.model_path):
            raise RuntimeError(
                f"GGUF model not found at {self.model_path!r}. Run Stage 8 quantization first."
            )

        cmd = [
            self.server_binary,
            "--model",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--threads",
            str(self.num_threads),
            "--ctx-size",
            str(self.num_ctx),
        ]

        logger.info("Starting llama-server: %s", " ".join(cmd))
        # Trusted local paths
        self._process = subprocess.Popen(  # nosec B603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for the server to be ready (poll /health).
        httpx = _import_httpx()
        self._client = httpx.Client(timeout=self.request_timeout)
        health_url = f"http://{self.host}:{self.port}/health"
        for _ in range(60):  # up to ~60 s
            if self._process and self._process.poll() is not None:  # type: ignore[union-attr]
                raise RuntimeError(
                    f"llama-server exited early with code {self._process.returncode}  "  # type: ignore[union-attr]
                    f"(stderr: {self._process.stderr.read().decode()[:500]})"  # type: ignore[union-attr]
                )
            try:
                resp = self._client.get(health_url)  # type: ignore[union-attr]
                if resp.status_code == 200:
                    logger.info("llama-server is ready on port %d", self.port)
                    return
            # Polling server readiness
            except Exception:  # nosec B110
                pass
            _time.sleep(1)

        raise RuntimeError(f"llama-server did not become healthy within 60 s (port {self.port}).")

    @property
    def model_info(self) -> dict:
        """Return metadata about the serving configuration."""
        return {
            "backend": "llama-server",
            "model_path": self.model_path,
            "server_binary": self.server_binary,
            "host": self.host,
            "port": self.port,
            "num_threads": self.num_threads,
            "num_ctx": self.num_ctx,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }

    def generate(self, prompt: str) -> str:
        """Generate a response from the GGUF model for *prompt*.

        Sends a ``POST /completion`` request to the running llama-server
        subprocess and returns the generated text.
        """
        self._ensure_running()

        url = f"http://{self.host}:{self.port}/completion"
        body = {
            "prompt": prompt,
            "n_predict": self.max_new_tokens,
            "temperature": self.temperature,
            "stream": False,
        }

        resp = self._client.post(url, json=body)  # type: ignore[union-attr]
        resp.raise_for_status()
        data = resp.json()

        # llama.cpp /completion returns {"content": "..."}
        content = data.get("content", "")
        if not content and "choices" in data:
            # OpenAI-compatible fallback
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                content = choices[0].get("text", "")
        return content.strip()

    def close(self) -> None:
        """Shut down the llama-server subprocess."""
        if self._process is not None:
            self._process.terminate()  # type: ignore[union-attr]
            try:
                self._process.wait(timeout=10)  # type: ignore[union-attr]
            except Exception:
                self._process.kill()  # type: ignore[union-attr]
            self._process = None
        if self._client is not None:
            self._client.close()  # type: ignore[union-attr]
            self._client = None


# ---------------------------------------------------------------------------
# TransformersBackend — HF-format model via transformers + torch (GPU capable)
# ---------------------------------------------------------------------------


class TransformersBackend:
    """Serving backend that loads a HuggingFace-format model with
    ``transformers`` + ``torch``.

    This backend is the GPU-capable fallback when the llama.cpp stack
    (either ``llama-cpp-python`` or the ``llama-server`` binary) is
    unavailable — for example, when no C compiler is present to build
    ``llama-cpp-python`` from source, or the bundled ``llama-server.exe``
    crashes at startup on Windows.

    Unlike the GGUF-based backends, this requires a **directory** in
    HuggingFace format (``config.json``, ``model.safetensors`` or
    ``*.bin``, ``tokenizer.json``) rather than a ``.gguf`` file.

    Parameters
    ----------
    model_dir:
        Filesystem path to the HuggingFace-format model directory.
    num_ctx:
        Context window size (used for truncation).
    num_threads:
        Number of CPU threads (informational on GPU; passed to torch).
    temperature:
        Sampling temperature.
    max_new_tokens:
        Maximum tokens to generate per request.
    """

    def __init__(
        self,
        model_dir: str,
        num_ctx: int = 4096,
        num_threads: int = 4,
        temperature: float = 0.2,
        max_new_tokens: int = 2048,
    ):
        self.model_dir = model_dir
        self.num_ctx = num_ctx
        self.num_threads = num_threads
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._device: str | None = None

    def _load(self):
        """Lazy-import and instantiate the HF model + tokenizer."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
    self.model_dir, trust_remote_code=True
)

        if self._model is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            self.model_dir,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
            torch.set_num_threads(self.num_threads)

        return self._model, self._tokenizer

    @property
    def model_info(self) -> dict:
        """Return metadata about the serving configuration."""
        return {
            "backend": "transformers",
            "model_dir": self.model_dir,
            "num_ctx": self.num_ctx,
            "num_threads": self.num_threads,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "device": self._device or "not-loaded",
        }

    def generate(self, prompt: str) -> str:
        """Generate a response from the model for *prompt*."""
        model, tokenizer = self._load()
        import torch

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.num_ctx,
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Strip the prompt tokens, decode only the new tokens.
        generated = gen_ids[0][input_ids.shape[-1] :]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    def close(self) -> None:
        """Release model memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


# Helper to find a HuggingFace model dir that corresponds to a .gguf path.
def _find_hf_model_dir(gguf_path: str) -> str | None:
    """Derive the HuggingFace model directory from a ``.gguf`` file path.

    Looks for a sibling directory in the same parent that contains
    ``config.json`` (the HF model directory marker).
    """
    import glob
    import os

    # Validate the GGUF path to prevent path traversal (CWE-22).
    safe_path = validate_path(gguf_path, allow_temp=True)
    gguf_dir = os.path.dirname(os.path.abspath(str(safe_path)))
    candidates = sorted(glob.glob(os.path.join(gguf_dir, "*", "config.json")))  # NOSONAR
    for cfg in candidates:
        d = os.path.dirname(cfg)
        # Check for model weights.
        if (
            os.path.exists(os.path.join(d, "model.safetensors"))
            or os.path.exists(os.path.join(d, "pytorch_model.bin"))
            or os.path.exists(os.path.join(d, "model.safetensors.index.json"))
        ):
            return d
        # Also check for sharded safetensors.
        if glob.glob(os.path.join(d, "model-*.safetensors")):
            return d
    return None


def _find_llama_server() -> str | None:
    """Locate the ``llama-server`` binary on disk or PATH.

    Checks the project's ``tools/llama-cpp/`` directory first, then falls
    back to ``shutil.which``.
    """
    import os
    import shutil

    # Look in the project's tools/llama-cpp directory.
    # __file__ = .../app/serving/backends.py → 3 dirnames → repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(repo_root, "tools", "llama-cpp", "llama-server.exe"),
        os.path.join(repo_root, "tools", "llama-cpp", "llama-server"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return shutil.which("llama-server") or shutil.which("llama-server.exe")


def _import_httpx():
    """Lazy-import ``httpx`` (used by both OllamaBackend and LlamaServerBackend)."""
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError(
            "httpx is not installed. Run `pip install httpx` to use "
            "LlamaServerBackend or OllamaBackend."
        ) from exc
    return httpx


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
