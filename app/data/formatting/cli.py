"""Stage 3 CLI: build instruction-format datasets from Stage 2 output.

Usage:
    # Build instruction-format JSONL from Postgres/MinIO (requires Stage 1 + 2)
    python -m app.data.formatting.cli build

    # Build from a local HF datasets directory (produced by Stage 2's export)
    python -m app.data.formatting.cli build --hf-path ./output/stage2_dataset

    # Dry-run: show counts without writing files
    python -m app.data.formatting.cli build --dry-run

    # Inspect an existing Stage 3 output directory
    python -m app.data.formatting.cli stats ./output/stage3

    # View a single formatted example
    python -m app.data.formatting.cli inspect ./output/stage3/train.jsonl --index 0
"""

from __future__ import annotations

import json
import logging
import os

import typer

from app.data.formatting.pipeline import OUTPUT_SPLITS, Stage3Result, run_stage3
from app.data.formatting.tokenizer import DEFAULT_MAX_TOKENS

app = typer.Typer(help="Stage 3: instruction-format dataset builder.")


@app.command()
def build(
    output_dir: str = typer.Option(
        "./output/stage3",
        "--output-dir",
        "-o",
        help="Directory to write JSONL splits to.",
    ),
    max_tokens: int = typer.Option(
        DEFAULT_MAX_TOKENS,
        "--max-tokens",
        help="Max prompt + target tokens per example; excess samples are dropped.",
    ),
    hf_path: str = typer.Option(
        None,
        "--hf-path",
        "-h",
        help="Load samples from a local HF datasets directory (Stage 2 export).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show counts without writing files.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build instruction-format JSONL splits from cleaned, split VulnSamples."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # If loading from HF dataset, do that first
    samples = None
    if hf_path:
        from app.data.formatting.pipeline import load_from_hf_dataset

        samples = load_from_hf_dataset(hf_path)
        typer.echo(f"Loaded {len(samples)} samples from HF dataset: {hf_path}")
        if not samples:
            typer.echo("No samples found in HF dataset.", err=True)
            raise typer.Exit(1)

    result = run_stage3(
        max_tokens=max_tokens,
        output_dir=output_dir,
        samples=samples,
    )

    _print_result(result)

    if dry_run:
        typer.echo("(dry-run — files were NOT written)")
    else:
        typer.echo(f"Output written to: {output_dir}")


@app.command()
def stats(output_dir: str = typer.Option(..., help="Path to a Stage 3 output directory.")) -> None:
    """Inspect a Stage 3 output directory and print summary statistics."""
    if not os.path.isdir(output_dir):
        typer.echo(f"Directory not found: {output_dir}", err=True)
        raise typer.Exit(1)

    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        typer.echo(f"Stage 3 output directory: {output_dir}")
        typer.echo(f"Max tokens: {manifest.get('max_tokens', '?')}")
        typer.echo(f"Token counter: {manifest.get('token_counter_model', '?')}")
        typer.echo("")
        for split_name in OUTPUT_SPLITS:
            info = manifest.get("splits", {}).get(split_name, {})
            typer.echo(
                f"  {split_name:6s}: {info.get('n_examples', 0)} examples, "
                f"{info.get('n_dropped', 0)} dropped"
            )
        return

    # Fallback: count lines in each JSONL file
    typer.echo(f"Stage 3 output directory: {output_dir}")
    typer.echo("(no manifest.json found — counting lines in JSONL files)")
    for split_name in OUTPUT_SPLITS:
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                n = sum(1 for _ in f)
            typer.echo(f"  {split_name:6s}: {n} examples")
        else:
            typer.echo(f"  {split_name:6s}: (file not found)")


@app.command()
def inspect(
    jsonl_path: str = typer.Option(
        ..., "--jsonl-path", "-f", help="Path to a JSONL file (e.g. train.jsonl)."
    ),
    index: int = typer.Option(0, "--index", "-i", help="Example index to show."),
) -> None:
    """Print a single formatted instruction example from a JSONL file."""
    if not os.path.exists(jsonl_path):
        typer.echo(f"File not found: {jsonl_path}", err=True)
        raise typer.Exit(1)

    with open(jsonl_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if index >= len(lines):
        typer.echo(f"Index {index} out of range (file has {len(lines)} examples)", err=True)
        raise typer.Exit(1)

    ex = json.loads(lines[index])
    typer.echo(json.dumps(ex, indent=2, ensure_ascii=False))


def _print_result(result: Stage3Result) -> None:
    """Pretty-print the Stage 3 pipeline result."""
    typer.echo(f"Loaded: {result.total_samples_loaded} samples")
    counts = result.counts()
    for split_name in OUTPUT_SPLITS:
        c = counts[split_name]
        typer.echo(
            f"  {split_name:6s}: {c['kept']} kept, {c['dropped']} dropped "
            f"(max_tokens={result.max_tokens})"
        )
    typer.echo(f"Total examples: {result.total_examples}  Dropped: {result.total_dropped}")


if __name__ == "__main__":
    app()
