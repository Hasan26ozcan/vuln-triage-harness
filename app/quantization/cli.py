"""Stage 8 CLI: GGUF / GPTQ / AWQ quantization.

Usage:

    # Real GGUF quantization (4-bit) from a Stage 5 checkpoint
    python -m app.quantization.cli run \\
        --checkpoint ./output/stage5/dpo/final_checkpoint \\
        --method gguf --bits 4 \\
        --output-dir ./output/stage8

    # Dry-run (heuristic estimates only, no GPU needed)
    python -m app.quantization.cli run \\
        --checkpoint ./output/stage5/dpo/final_checkpoint \\
        --method gguf --bits 4 --dry-run

    # Mock mode (deterministic, for tests)
    python -m app.quantization.cli run \\
        --checkpoint ./output/stage5/dpo/final_checkpoint \\
        --method gguf --bits 4 --mock
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime

import typer

from app.quantization.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_OUTPUT_BASE,
)
from app.schemas.quantization import QuantMethod, QuantResult, QuantStatus

logger = logging.getLogger(__name__)

app = typer.Typer(help="Stage 8: Quantization matrix (GPTQ / AWQ / GGUF).")

# Path to the pre-built llama.cpp CLI binary (b1047 win-cpu-x64).
_LLAMA_QUANTIZE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tools",
    "llama-cpp",
    "llama-quantize.exe",
)


@app.command()
def run(
    checkpoint: str = typer.Option(
        "",
        "--checkpoint",
        "-c",
        help="Path to the Stage 5 checkpoint (HF dir or .gguf file).",
    ),
    method: str = typer.Option(
        "gguf",
        "--method",
        "-m",
        help="Quantization method: gguf, gptq, awq.",
    ),
    bits: int = typer.Option(
        4,
        "--bits",
        "-b",
        help="Target bit-width (2, 3, 4, 8).",
    ),
    output_dir: str = typer.Option(
        DEFAULT_OUTPUT_BASE,
        "--output-dir",
        "-o",
        help="Output directory for the quantized checkpoint + report.",
    ),
    base_model: str = typer.Option(
        DEFAULT_BASE_MODEL,
        "--base-model",
        help="Base HF model ID (for PEFT adapter resolution).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Heuristic estimates only — no GPU/external tools needed.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock quantizer (deterministic, for tests).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Verbose logging.",
    ),
) -> None:
    """Quantize a checkpoint and write a GGUF/GPTQ/AWQ report."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not checkpoint:
        typer.echo("Error: --checkpoint is required", err=True)
        raise typer.Exit(1)

    if not os.path.exists(checkpoint) and not _is_hf_id(checkpoint):
        typer.echo(f"Error: checkpoint not found: {checkpoint}", err=True)
        raise typer.Exit(1)

    method_enum = _parse_method(method)
    output_path = os.path.join(
        output_dir,
        f"{method_enum.value}_bits{bits}.gguf",
    )
    os.makedirs(output_dir, exist_ok=True)

    # Run quantization.
    if dry_run:
        result = _dry_run_quantize(method_enum, bits, checkpoint, output_path)
    elif mock:
        from app.quantization import MockQuantizer

        q = MockQuantizer(default_method=method_enum, default_bit_width=bits)
        result = q.quantize(checkpoint, output_path, bits)
    else:
        from app.quantization.config import GGUFConfig
        from app.quantization.export_awq import AWQQuantizer
        from app.quantization.export_gguf import GGUFQuantizer
        from app.quantization.export_gptq import GPTQQuantizer

        if method_enum == QuantMethod.GGUF:
            quantizer = GGUFQuantizer(
                config=GGUFConfig(quant_types=[_bits_to_gguf(bits)]),
                llama_cpp_path=_LLAMA_QUANTIZE if os.path.exists(_LLAMA_QUANTIZE) else None,
                base_model=base_model,
            )
        elif method_enum == QuantMethod.GPTQ:
            quantizer = GPTQQuantizer()
        elif method_enum == QuantMethod.AWQ:
            quantizer = AWQQuantizer()
        else:
            typer.echo(f"Error: unsupported method {method}", err=True)
            raise typer.Exit(1)

        result = quantizer.quantize(checkpoint, output_path, bits)

    # Write report.
    report = {
        "run_id": f"stage8-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "method": result.quant_method.value,
        "bit_width": result.bit_width,
        "status": result.status.value,
        "checkpoint_path": result.checkpoint_path,
        "quantized_model_size_gb": result.quantized_model_size_gb,
        "estimated_vram_gb": result.estimated_vram_gb,
        "measured_vram_gb": result.measured_vram_gb,
        "tokens_per_sec": result.tokens_per_sec,
        "model_cwe_macro_f1": result.model_cwe_macro_f1,
        "exec_pass_rate": result.exec_pass_rate,
        "error": result.error,
        "notes": result.notes,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    report_path = os.path.join(output_dir, "quant_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    typer.echo(f"Quantization complete: {result.status.value}")
    typer.echo(f"  Output: {result.checkpoint_path}")
    typer.echo(f"  Report: {report_path}")
    typer.echo(f"  Size: {result.quantized_model_size_gb:.2f} GB")
    if result.error:
        typer.echo(f"  Error: {result.error}", err=True)


@app.command()
def inspect(
    gguf_path: str = typer.Argument(..., help="Path to a .gguf file."),
) -> None:
    """Print metadata from a GGUF file."""
    from gguf import GGUFReader

    reader = GGUFReader(gguf_path)
    typer.echo(f"Architecture: {reader.architecture}")
    typer.echo(f"Tensors: {len(reader.tensors)}")
    for kv in reader.fields.items():
        typer.echo(f"  {kv[0]}: {kv[1]}")


def _parse_method(method: str) -> QuantMethod:
    """Parse a method string into a QuantMethod enum."""
    try:
        return QuantMethod(method.lower())
    except ValueError:
        valid = ", ".join(m.value for m in QuantMethod)
        typer.echo(f"Error: method must be one of: {valid}", err=True)
        raise typer.Exit(1) from None


def _bits_to_gguf(bits: int) -> str:
    """Map bit-width integer to a GGUF quant-type string."""
    mapping = {
        2: "Q2_K",
        3: "Q3_K",
        4: "Q4_K",
        8: "Q8_0",
        16: "F16",
        32: "F32",
    }
    if bits not in mapping:
        typer.echo(f"Error: unsupported bits={bits} for GGUF. Use 2, 3, 4, 8, 16, 32.", err=True)
        raise typer.Exit(1)
    return mapping[bits]


def _is_hf_id(path: str) -> bool:
    """Check if *path* looks like a HuggingFace model ID (e.g. org/model)."""
    return "/" in path and not os.path.exists(path) and not os.path.isabs(path)


def _dry_run_quantize(
    method: QuantMethod,
    bits: int,
    source: str,
    output_path: str,
) -> QuantResult:
    """Produce heuristic estimates without calling any external tools."""
    from app.quantization.config import (
        estimate_model_size_gb,
        estimate_quality,
        estimate_tokens_per_sec,
        estimate_vram_gb,
    )

    est_size = estimate_model_size_gb(method, bits)
    est_vram = estimate_vram_gb(method, bits)
    est_quality = estimate_quality(method, bits)
    tps = estimate_tokens_per_sec(method, bits)

    return QuantResult(
        quant_method=method,
        bit_width=bits,
        quantized_model_size_gb=est_size,
        estimated_vram_gb=est_vram,
        measured_vram_gb=None,
        tokens_per_sec=tps,
        model_cwe_macro_f1=None,
        exec_pass_rate=None,
        status=QuantStatus.COMPLETED,
        checkpoint_path=output_path,
        notes=f"dry-run {method.value} @ {bits}-bit (est. quality {est_quality:.2f})",
    )


if __name__ == "__main__":
    app()
