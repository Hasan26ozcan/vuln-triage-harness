"""Stage 2 CLI: cleaning, dedup, leakage-safe split, contamination check.

Usage:
    # Run the full Stage 2 pipeline (load from Postgres, dedup, split,
    # contamination check, persist results back to Postgres)
    python -m app.data.cleaning.cli clean

    # Dry-run: show the plan without writing to Postgres
    python -m app.data.cleaning.cli clean --dry-run

    # Inspect current split plan without running dedup
    python -m app.data.cleaning.cli plan

    # Export the cleaned dataset to HuggingFace Hub
    python -m app.data.cleaning.cli export --repo-id vuln-triage/vuln-triage-dataset

    # Check contamination between train and a local gold-eval JSONL
    python -m app.data.cleaning.cli check-contamination --gold-eval eval/gold_set/gold.jsonl
"""

from __future__ import annotations

import json
import logging
import sys

import typer

from app.data.cleaning.contamination import check_contamination
from app.data.cleaning.hf_dataset import samples_to_hf_dataset
from app.data.cleaning.pipeline import run_stage2
from app.data.cleaning.split import DEFAULT_RATIOS, DEFAULT_SEED, SplitConfig, build_leak_aware_plan
from app.schemas.vuln import VulnSample
from app.storage.db import VulnSampleRow, get_session
from app.storage.object_store import get_json

app = typer.Typer(help="Stage 2: cleaning, dedup, leakage-safe split, contamination check.")


def _load_all_vuln_samples() -> list[VulnSample]:
    """Load all VulnSample records from Postgres + MinIO (used by CLI subcommands)."""
    session = get_session()
    try:
        rows = session.query(VulnSampleRow).all()
        samples: list[VulnSample] = []
        for row in rows:
            payload = get_json(row.object_store_key)
            samples.append(VulnSample(**payload))
        return samples
    finally:
        session.close()


def _load_gold_eval(path: str) -> list[VulnSample]:
    """Load gold-eval samples from a local JSONL file (Stage 6 gold set)."""
    samples: list[VulnSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            samples.append(VulnSample(**payload))
    return samples


@app.command()
def clean(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show counts and plan without writing to Postgres."
    ),
    dedup_threshold: float = typer.Option(
        0.95, help="Cosine similarity threshold for near-duplicate detection."
    ),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed for reproducible splits."),
    train_ratio: float = typer.Option(DEFAULT_RATIOS["train"], help="Fraction for train split."),
    val_ratio: float = typer.Option(DEFAULT_RATIOS["val"], help="Fraction for val split."),
    test_ratio: float = typer.Option(DEFAULT_RATIOS["test"], help="Fraction for test split."),
    contamination_n: int = typer.Option(5, help="N-gram length for contamination check."),
    max_contamination: float = typer.Option(0.05, help="Max acceptable contamination rate."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full Stage 2 pipeline: dedup, split, contamination check."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    config = SplitConfig(
        ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio},
        seed=seed,
    )

    try:
        result = run_stage2(
            dedup_threshold=dedup_threshold,
            split_config=config,
            contamination_n=contamination_n,
            max_contamination=max_contamination,
            persist=not dry_run,
        )
    except RuntimeError as exc:
        if "No samples found" in str(exc):
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        raise

    typer.echo(f"Loaded:      {result.samples_loaded}")
    n_dups = len(result.duplicate_pairs)
    typer.echo(f"After dedup: {result.samples_after_dedup} (removed {n_dups} duplicates)")
    typer.echo(f"Split:       {result.split_result.counts()}")

    dist = result.split_result.cwe_distribution()
    for split_name in ("train", "val", "test"):
        typer.echo(f"  {split_name} CWE distribution: {dist[split_name]}")

    typer.echo(
        f"Contamination: {result.contamination_report.contamination_rate:.4f} "
        f"(ok={result.contamination_ok})"
    )

    if dry_run:
        typer.echo("(dry-run — splits were NOT persisted to Postgres)")


@app.command()
def plan(
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed for reproducible plan."),
) -> None:
    """Show the leakage-safe split plan without running dedup or mutation."""
    samples = _load_all_vuln_samples()
    if not samples:
        typer.echo("No samples found. Run Stage 1 first.")
        raise typer.Exit(1)

    plan = build_leak_aware_plan(samples, config=SplitConfig(seed=seed))

    # Count repos per split
    from collections import Counter

    split_counts = Counter(plan.repo_to_split.values())
    cwe_split_counts: dict[str, Counter] = {}
    for repo, cwe in plan.repo_to_cwe.items():
        split_name = plan.repo_to_split[repo]
        cwe_split_counts.setdefault(cwe, Counter())[split_name] += 1

    typer.echo(f"Total samples: {len(samples)}")
    typer.echo(f"Total repos:   {len(plan.repo_to_split)}")
    typer.echo(f"Repos per split: {dict(split_counts)}")
    typer.echo("CWE x split repo counts:")
    for cwe, counts in sorted(cwe_split_counts.items()):
        typer.echo(f"  {cwe}: {dict(counts)}")


@app.command(name="export")
def export_dataset(
    repo_id: str = typer.Option(
        "vuln-triage/vuln-triage-dataset",
        "--repo-id", "-r",
        help="HuggingFace Hub repo ID for the dataset.",
    ),
    local_path: str = typer.Option(
        None, "--local-path", "-p",
        help="If provided, save to disk instead of (or in addition to) hub.",
    ),
    private: bool = typer.Option(False, "--private", help="Make the Hub repo private."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Export the cleaned, split dataset to HuggingFace format.

    Converts all VulnSample records from Postgres/MinIO into an HF DatasetDict
    with train/val/test splits, and either saves to disk or pushes to the Hub.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    samples = _load_all_vuln_samples()
    if not samples:
        typer.echo("No samples found. Run Stage 1 then Stage 2 clean first.")
        raise typer.Exit(1)

    ds = samples_to_hf_dataset(samples)
    typer.echo(f"Exported {len(ds)} splits with {sum(len(ds[s]) for s in ds)} total rows")

    if local_path:
        ds.save_to_disk(local_path)
        typer.echo(f"Saved to disk: {local_path}")

    try:
        from app.data.cleaning.hf_dataset import push_to_hub

        url = push_to_hub(ds, repo_id=repo_id, private=private)
        typer.echo(f"Pushed to Hub: {url}")
    except RuntimeError as exc:
        if "HF_TOKEN" in str(exc) or "token" in str(exc).lower():
            typer.echo(str(exc), err=True)
            raise typer.Exit(1) from exc
        raise


@app.command(name="check-contamination")
def check_contamination_cmd(
    gold_eval: str = typer.Option(
        ..., "--gold-eval", "-g",
        help="Path to gold-eval JSONL file (one VulnSample per line).",
    ),
    contamination_n: int = typer.Option(5, help="N-gram length."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Check n-gram contamination between train set and a gold-eval file."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    train_samples = [
        s for s in _load_all_vuln_samples()
        if s.split == "train"
    ]
    gold_eval_samples = _load_gold_eval(gold_eval)

    typer.echo(f"Train samples: {len(train_samples)}")
    typer.echo(f"Gold-eval samples: {len(gold_eval_samples)}")

    report = check_contamination(train_samples, gold_eval_samples, n=contamination_n)
    typer.echo(report.summary())

    if report.contamination_rate > 0.05:
        typer.echo(
            f"WARNING: contamination rate {report.contamination_rate:.4f} "
            f"exceeds the 5% threshold.",
            err=True,
        )
        sys.exit(2)
    else:
        typer.echo("OK: contamination within acceptable bounds.")


if __name__ == "__main__":
    app()
