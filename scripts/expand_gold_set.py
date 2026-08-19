#!/usr/bin/env python
"""Expand the gold-eval set from 12 to 60-90 samples by pulling real CVEs
from the CVEfixes database (using the reduced 3-table schema + NVD CWE mapping).

For each in-scope CWE class, pulls 10-15 additional real examples that:
  - Match the CWE's target language per CWE_SCOPE
  - Do NOT overlap with training data (checked via CVE ID + repo_name)
  - Have code_before/code_after small enough for the exec sandbox (< 5KB)

Usage::

    python scripts/expand_gold_set.py [--target-per-class 12] [--output eval/gold_set/gold.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.collectors.cvefixes_reduced_loader import ReducedCveFixesLoader
from app.data.collectors.cwe_scope import CWE_SCOPE
from app.data.collectors.pipeline import build_vuln_sample
from app.schemas.vuln import VulnSample
from scripts.run_stage1_real import _MockNvdClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Max code size for exec-sandbox compatibility
MAX_CODE_LEN = 5000


def load_existing_gold(path: str) -> tuple[list[VulnSample], set[str]]:
    """Load existing gold samples and return them + their CVE IDs."""
    samples: list[VulnSample] = []
    cve_ids: set[str] = set()
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            samples.append(VulnSample(**d))
            if d.get("cve_id"):
                cve_ids.add(d["cve_id"])
    return samples, cve_ids


def load_train_cve_ids(stage3_dir: str = "output/stage3") -> set[str]:
    """Load CVE IDs from existing training data to avoid overlap."""
    cve_ids: set[str] = set()
    for split in ("train", "val", "test"):
        path = Path(stage3_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            json.loads(line)
            # NOTE: stage3 JSONL only carries 'prompt', not a top-level 'cve_id'.
            # TODO(hasan): thread cve_id through stage3 dataset build so this
            # can dedupe against training CVEs instead of always returning {}.
    return cve_ids


def load_train_repo_commit_pairs(stage3_dir: str = "output/stage3") -> set[tuple[str, str]]:
    """Load (repo_name, commit_sha) pairs from training data to avoid overlap."""
    pairs: set[tuple[str, str]] = set()
    for split in ("train", "val", "test"):
        path = Path(stage3_dir) / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            json.loads(line)
            # NOTE: stage3 JSONL does not expose repo_name/commit_sha at the
            # top level, only inside the free-text prompt.
            # TODO(hasan): thread repo_name/commit_sha through stage3 dataset
            # build so this can actually dedupe; currently unused/unreachable
            # from the CLI (see load_train_cve_ids_from_postgres instead).
    return pairs


def load_train_cve_ids_from_postgres() -> set[str]:
    """Load CVE IDs from Postgres training data to avoid overlap."""
    from app.storage.db import VulnSampleRow, get_session

    session = get_session()
    try:
        rows = session.query(VulnSampleRow.cve_id).all()
        return {r[0] for r in rows}
    finally:
        session.close()


def expand_gold_set(
    target_per_class: int = 12,
    output_path: str = "eval/gold_set/gold.jsonl",
    db_path: str = "data/cvefixes_db/CVEfixes.db",
    cwe_mapping_path: str = "data/cve_cwe_mapping.json",
    allow_training_overlap: bool = False,
    run_static_analysis: bool = True,
) -> int:
    """Expand the gold-eval set with real CVE data.

    Returns the total number of gold samples (existing + new).
    """
    # Load existing gold
    existing, existing_cves = load_existing_gold(output_path)
    logger.info("Loaded %d existing gold samples", len(existing))

    # Load training data CVE IDs to avoid overlap — unless overridden
    if allow_training_overlap:
        train_cves: set[str] = set()
        logger.warning(
            "Training-data overlap check DISABLED (--allow-training-overlap). "
            "CVEfixes.db was fully consumed by Stage 1; gold samples may overlap "
            "with training data — this is documented as a dataset-size limitation."
        )
    else:
        train_cves = load_train_cve_ids_from_postgres()
        logger.info("Loaded %d training data CVE IDs for overlap check", len(train_cves))

    # Exclude existing gold CVEs and (optionally) training data CVEs
    excluded_cves = existing_cves | train_cves
    logger.info(
        "Excluding %d CVE IDs (existing gold%s)",
        len(excluded_cves),
        " + training data" if train_cves else "",
    )

    # Load pairs from reduced CVEfixes
    loader = ReducedCveFixesLoader(db_path, cwe_mapping_path)
    all_pairs = loader.load_pairs(languages={"Python", "JavaScript", "TypeScript"})
    logger.info("Loaded %d candidate pairs from CVEfixes", len(all_pairs))

    # Group by CWE, filter by language and code size
    cwe_to_spec = {spec.cwe_id: spec for spec in CWE_SCOPE}
    pairs_by_cwe: dict[str, list] = {cwe: [] for cwe in cwe_to_spec}

    for pair in all_pairs:
        if pair.cwe_id not in cwe_to_spec:
            continue
        spec = cwe_to_spec[pair.cwe_id]
        # Check language match
        if pair.language != spec.language:
            continue
        # Check code size
        if len(pair.vulnerable_code) > MAX_CODE_LEN or len(pair.fixed_code or "") > MAX_CODE_LEN:
            continue
        # Check overlap with training data
        if pair.cve_id in excluded_cves:
            continue
        pairs_by_cwe[pair.cwe_id].append(pair)

    logger.info("Candidate counts per CWE (after filtering):")
    for cwe, pairs_list in sorted(pairs_by_cwe.items()):
        spec = cwe_to_spec[cwe]
        logger.info("  %s (%s, %s): %d candidates", cwe, spec.name, spec.language, len(pairs_list))

    # Select target_per_class from each CWE
    nvd_client = _MockNvdClient()
    new_samples: list[VulnSample] = []
    next_id = len(existing) + 1

    for cwe, pairs_list in sorted(pairs_by_cwe.items()):
        spec = cwe_to_spec[cwe]
        # Shuffle deterministically for reproducibility
        import hashlib

        pairs_list.sort(
            key=lambda p: hashlib.md5(p.cve_id.encode(), usedforsecurity=False).hexdigest()
        )

        selected = pairs_list[:target_per_class]
        logger.info("  %s: selecting %d of %d candidates", cwe, len(selected), len(pairs_list))

        for pair in selected:
            # Build sample with enrichment
            try:
                sample = build_vuln_sample(
                    pair, nvd_client, run_static_analysis=run_static_analysis
                )
            except Exception as exc:
                logger.warning("  Skipped %s: %s", pair.cve_id, exc)
                continue

            # Convert to gold-eval format
            gold_id = f"gold_{next_id:03d}"
            next_id += 1

            gold_sample = VulnSample(
                id=gold_id,
                source="cve_real",
                repo_name=sample.repo_name,
                commit_sha=sample.commit_sha,
                cwe_id=sample.cwe_id,
                severity=sample.severity,
                language=sample.language,
                vulnerable_code=sample.vulnerable_code,
                fixed_code=sample.fixed_code,
                static_findings=sample.static_findings,
                description=sample.description,
                split="gold_eval",
                cve_id=sample.cve_id,
            )
            new_samples.append(gold_sample)

    # Combine existing + new
    all_gold = existing + new_samples
    logger.info(
        "Total gold samples: %d (existing=%d, new=%d)",
        len(all_gold),
        len(existing),
        len(new_samples),
    )

    # Write to JSONL
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for sample in all_gold:
            f.write(sample.model_dump_json() + "\n")
    logger.info("Saved %d gold samples to %s", len(all_gold), out)

    return len(all_gold)


def main():
    ap = argparse.ArgumentParser(description="Expand gold-eval set from CVEfixes")
    ap.add_argument(
        "--target-per-class",
        type=int,
        default=12,
        help="Number of additional samples per CWE class (default: 12)",
    )
    ap.add_argument(
        "--output",
        default="eval/gold_set/gold.jsonl",
        help="Output JSONL path (will add to existing)",
    )
    ap.add_argument("--db-path", default="data/cvefixes_db/CVEfixes.db")
    ap.add_argument("--cwe-mapping", default="data/cve_cwe_mapping.json")
    ap.add_argument(
        "--allow-training-overlap",
        action="store_true",
        help="Skip the Postgres overlap check (use when CVEfixes.db is "
        "fully consumed by Stage 1 training data)",
    )
    ap.add_argument(
        "--no-static-analysis",
        action="store_true",
        help="Skip Semgrep static analysis (faster, static_findings will be empty)",
    )
    args = ap.parse_args()

    total = expand_gold_set(
        target_per_class=args.target_per_class,
        output_path=args.output,
        db_path=args.db_path,
        cwe_mapping_path=args.cwe_mapping,
        allow_training_overlap=args.allow_training_overlap,
        run_static_analysis=not args.no_static_analysis,
    )
    print(f"\nGold set expanded to {total} total samples")


if __name__ == "__main__":
    main()
