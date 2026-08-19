"""CLI entry point for Stage 1 data collection.

Usage:
    python -m app.data.collectors.cli collect --db-path ./CVEfixes.db
    python -m app.data.collectors.cli collect --db-path ./CVEfixes.db --dry-run
    python -m app.data.collectors.cli scope
"""

from __future__ import annotations

import logging

import typer

from app.data.collectors.cwe_scope import CWE_SCOPE
from app.data.collectors.pipeline import run_pipeline
from app.storage.db import init_db
from app.storage.object_store import ensure_bucket

app = typer.Typer(help="Stage 1: vulnerability data collection.")


@app.command()
def scope() -> None:
    """Print the CWE classes this project targets and why."""
    for spec in CWE_SCOPE:
        typer.echo(
            f"{spec.cwe_id:10s} {spec.name:35s} lang={spec.language:10s} min={spec.min_samples}"
        )


@app.command()
def collect(
    db_path: str = typer.Option(..., help="Path to a local CVEfixes.db (SQLite)."),
    languages: str = typer.Option(
        "", help="Comma-separated language filter, e.g. python,javascript"
    ),
    no_static_analysis: bool = typer.Option(
        False, help="Skip Semgrep (faster, no static_findings)."
    ),
    dry_run: bool = typer.Option(
        False, help="Build VulnSample records without writing to Postgres/MinIO."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full Stage 1 pipeline against a local CVEfixes.db."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    lang_set = {lang.strip().lower() for lang in languages.split(",") if lang.strip()} or None

    if not dry_run:
        init_db()
        ensure_bucket()

    result = run_pipeline(
        db_path=db_path,
        languages=lang_set,
        run_static_analysis=not no_static_analysis,
        dry_run=dry_run,
    )

    typer.echo(f"Built: {len(result.samples)}  Skipped: {len(result.skipped)}")
    if result.skipped:
        typer.echo("First few skip reasons:")
        for pair, reason in result.skipped[:5]:
            typer.echo(f"  {pair.cve_id}: {reason}")


if __name__ == "__main__":
    app()
