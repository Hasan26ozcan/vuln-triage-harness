"""Stage 9 — CLI entry point for the air-gapped serving layer.

Registered as a ``typer`` subcommand on the shared ``app`` from
``app.evaluation.cli`` (which is itself a Typer app). This makes
``vulntriage serve ...`` available alongside ``stage1``...``stage8``.

The CLI supports two modes:

* **Serve mode** (default): start a uvicorn server with the configured
  backend (llama.cpp / Ollama / mock).
* **Analyze mode** (``--analyze``): read a single sample from a JSON
  file, run it through the backend, and print the parsed prediction to
  stdout — useful for CLI-only usage without starting a server.
* **Batch mode** (``--batch``): read a JSON array of samples, serve
  them in a batch, and print results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from app.schemas.serving import BatchServeRequest, ServeRequest
from app.serving.config import DEFAULT_BACKEND_TYPE, ServingConfig
from app.serving.serve import VulnerabilityServer

logger = logging.getLogger(__name__)

__all__ = ["app"]

app = typer.Typer(help="Stage 9 — air-gapped vulnerability serving (llama.cpp / Ollama).")


@app.command()
def serve(
    # --- Model config ---
    model_path: str = typer.Option(
        "",
        "--model-path",
        "-m",
        help="Path to the GGUF checkpoint (llama.cpp) or model name (Ollama).",
    ),
    backend_type: str = typer.Option(
        DEFAULT_BACKEND_TYPE,
        "--backend",
        "-b",
        help="Backend type: 'llama.cpp' | 'llama-server' | 'ollama' | 'mock'.",
    ),
    num_ctx: int = typer.Option(4096, "--num-ctx", help="Context window size."),
    num_threads: int = typer.Option(4, "--num-threads", help="CPU threads (llama.cpp only)."),
    n_gpu_layers: int = typer.Option(0, "--n-gpu-layers", help="GPU layers (llama.cpp only)."),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature."),
    max_new_tokens: int = typer.Option(2048, "--max-new-tokens", help="Max tokens to generate."),
    request_timeout: float = typer.Option(30.0, "--request-timeout", help="HTTP timeout (Ollama)."),
    # Air-gapped/local serving CLI; overridable via --host
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        "-h",
        help="Bind address.",  # nosec
    ),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    # --- Modes ---
    analyze: bool = typer.Option(
        False,
        "--analyze",
        "-a",
        help="Analyze a single sample from --input-file and print result; do not start a server.",
    ),
    batch: bool = typer.Option(
        False,
        "--batch",
        help="Analyze samples from --input-file as a batch; do not start a server.",
    ),
    input_file: str = typer.Option(
        "",
        "--input-file",
        "-i",
        help="Path to a JSON file with a single ServeRequest or a JSON array of ServeRequests.",
    ),
    # --- Output ---
    output_file: str = typer.Option(
        "",
        "--output-file",
        "-o",
        help="Optional path to write results as JSON.",
    ),
    # --- Dry-run (no real model) ---
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print config and exit without starting a server or backend (shows warnings).",
    ),
) -> None:
    """Run the Stage 9 serving CLI."""

    config = ServingConfig(
        model_path=model_path,
        backend_type=backend_type,
        num_ctx=num_ctx,
        num_threads=num_threads,
        n_gpu_layers=n_gpu_layers,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        host=host,
        port=port,
        request_timeout=request_timeout,
    )

    if dry_run:
        _run_dry_run(config)
        raise typer.Exit(0)

    if analyze or batch:
        _run_analyze_batch(config, input_file, output_file, batch)
        raise typer.Exit(0)

    _start_server(config)


def _run_dry_run(config: ServingConfig) -> None:
    """Print the serving configuration and any warnings, then exit."""
    typer.echo("=== Stage 9 Serving Config (dry-run) ===")
    typer.echo(f"  Backend type: {config.backend_type}")
    typer.echo(f"  Model path:   {config.model_path or '(none)'}")
    typer.echo(f"  Run name:     {config.run_name}")
    typer.echo(f"  Host/Port:    {config.host}:{config.port}")
    warnings = config.all_warnings()
    if warnings:
        typer.echo("  Warnings:")
        for w in warnings:
            typer.echo(f"    [WARN] {w}")
    else:
        typer.echo("  Warnings:     (none)")
    typer.echo("=== (dry-run, exiting) ===")


def _run_analyze_batch(
    config: ServingConfig,
    input_file: str,
    output_file: str,
    batch: bool,
) -> None:
    """Run analyze or batch mode: load input, serve, print, optionally write output."""
    if not input_file:
        typer.echo("Error: --input-file is required when using --analyze or --batch", err=True)
        raise typer.Exit(1)

    server = VulnerabilityServer.from_config(config)
    input_path = Path(input_file)

    if not input_path.exists():
        typer.echo(f"Error: input file not found: {input_path}", err=True)
        raise typer.Exit(1)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    if batch or isinstance(data, list):
        output = _serve_batch(server, data, output_file)
    else:
        output = _serve_single(server, data, output_file)

    typer.echo(output)


def _serve_batch(
    server: VulnerabilityServer,
    data: list | dict,
    output_file: str,
) -> str:
    """Serve a batch of requests and return JSON output."""
    if isinstance(data, dict):
        # Single dict -> wrap in list
        data = [data]
    requests = [ServeRequest(**d) for d in data]
    batch_req = BatchServeRequest(requests=requests)
    result = server.serve_batch(batch_req)
    output = result.model_dump_json(indent=2)
    if output_file:
        Path(output_file).write_text(output, encoding="utf-8")
        typer.echo(f"\nResults written to {output_file}", err=True)
    return output


def _serve_single(
    server: VulnerabilityServer,
    data: dict,
    output_file: str,
) -> str:
    """Serve a single request and return JSON output."""
    request = ServeRequest(**data)
    response = server.serve_sample(request)
    output = response.model_dump_json(indent=2)
    if output_file:
        Path(output_file).write_text(output, encoding="utf-8")
        typer.echo(f"\nResult written to {output_file}", err=True)
    return output


def _start_server(config: ServingConfig) -> None:
    """Start the uvicorn server with the given configuration."""
    warnings = config.all_warnings()
    if warnings:
        typer.echo("Configuration warnings:", err=True)
        for w in warnings:
            typer.echo(f"  [WARN] {w}", err=True)

    typer.echo(
        f"Starting Stage 9 server - {config.run_name} "
        f"({config.backend_type}) on {config.host}:{config.port}"
    )

    # We import here to avoid pulling in FastAPI at module-import time.
    from app.serving.api import create_app

    fastapi_app = create_app(config)

    import uvicorn

    uvicorn.run(fastapi_app, host=config.host, port=config.port)


# Re-export the typer app so it can be registered in evaluation.cli
cli_app = app

if __name__ == "__main__":
    app()
