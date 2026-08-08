"""CLI entry point for Stage 2: dedup, leakage-safe split, contamination check.

Usage:
    python -m app.data.cleaning.cli run
    python -m app.data.cleaning.cli run --dry-run -v
"""

from __future__ import annotations

import logging

import typer

from app.data.cleaning.pipeline import run_cleaning_pipeline
from app.data.cleaning.split import save_manifest
from app.schemas.vuln import VulnSample
from app.storage import object_store
from app.storage.db import VulnSampleRow, get_session

app = typer.Typer(help="Stage 2: cleaning, dedup, leakage-safe split, contamination check.")


def _load_all_samples() -> list[VulnSample]:
    session = get_session()
    try:
        rows = session.query(VulnSampleRow).all()
        samples = []
        for row in rows:
            payload = object_store.get_json(row.object_store_key)
            samples.append(VulnSample(**payload))
        return samples
    finally:
        session.close()


def _persist_updated(samples: list[VulnSample]) -> None:
    session = get_session()
    try:
        for sample in samples:
            row = session.get(VulnSampleRow, sample.id)
            if row is None:
                continue  # sample was removed by dedup/contamination — row stays, split untouched
            row.split = sample.split
            object_store.put_json(row.object_store_key, sample.model_dump())
        session.commit()
    finally:
        session.close()


@app.command()
def run(
    manifest_out: str = typer.Option(
        "data/split_manifest.json", help="Where to save the split manifest (step 5)."
    ),
    seed: int = typer.Option(42),
    dedup_threshold: float = typer.Option(0.95),
    contamination_threshold: float = typer.Option(0.5),
    dry_run: bool = typer.Option(False, help="Compute but don't write back to Postgres/MinIO."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    samples = _load_all_samples()
    typer.echo(f"Loaded {len(samples)} samples from Postgres/MinIO.")
    if not samples:
        typer.echo("No samples found — run Stage 1 collection first.")
        raise typer.Exit(1)

    result = run_cleaning_pipeline(
        samples,
        seed=seed,
        dedup_threshold=dedup_threshold,
        contamination_threshold=contamination_threshold,
    )

    typer.echo(f"After dedup + contamination removal: {len(result.samples)} samples")
    typer.echo(f"Near-duplicate pairs removed: {len(result.duplicate_pairs)}")
    contaminated = sum(1 for r in result.contamination_results if r.contaminated)
    typer.echo(f"Contaminated gold_eval samples removed: {contaminated}")
    if result.balance_report.missing:
        typer.echo(f"WARNING — class balance gaps (split, cwe_id): {result.balance_report.missing}")
    else:
        typer.echo("Class balance OK: every split has every in-scope CWE class.")

    save_manifest(result.manifest, manifest_out)
    typer.echo(
        f"Split manifest saved to {manifest_out} (seed={seed}) — this is now the frozen split."
    )

    if not dry_run:
        _persist_updated(result.samples)
        typer.echo("Split assignment written back to Postgres/MinIO.")


if __name__ == "__main__":
    app()
