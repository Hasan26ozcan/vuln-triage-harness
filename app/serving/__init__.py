"""Stage 9 — air-gapped vulnerability serving layer.

Provides llama.cpp / Ollama / Mock backends behind an injectable
``ServingBackend`` Protocol, plus a ``VulnerabilityServer`` orchestrator,
a FastAPI app, and a Typer CLI.

Typical usage::

    from app.serving.config import ServingConfig
    from app.serving.serve import VulnerabilityServer

    config = ServingConfig(model_path="model/ggml-q4_0.gguf", backend_type="llama.cpp")
    server = VulnerabilityServer.from_config(config)
    response = server.serve_sample(serve_request)
"""

from app.serving.config import ServingConfig
from app.serving.backends import (
    ServingBackend,
    LlamaCppBackend,
    OllamaBackend,
    MockServingBackend,
)
from app.serving.serve import VulnerabilityServer

__all__ = [
    "ServingConfig",
    "ServingBackend",
    "LlamaCppBackend",
    "OllamaBackend",
    "MockServingBackend",
    "VulnerabilityServer",
]
