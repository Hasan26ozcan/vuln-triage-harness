#!/usr/bin/env python
"""Run Stage 1 data collection against the real CVEfixes.db.

This script processes real CVE-patch pairs from the CVEfixes v1.0.8 SQLite
database. To avoid the NVD API rate-limit bottleneck (5 unauthenticated
requests / 30s, which would take ~34 hours for 20k+ CVEs), we inject a
lightweight mock NvdClient that derives severity from the CVE year and
description from the pair itself. Semgrep static analysis runs for real
using the bundled rule packs.

Usage::

    python scripts/run_stage1_real.py [--max-pairs 300] [--no-static-analysis]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow direct execution: python scripts/run_stage1_real.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.data.collectors.cvefixes_loader import CveFixesLoader
from app.data.collectors.cvefixes_reduced_loader import ReducedCveFixesLoader
from app.data.collectors.cwe_scope import CWE_SCOPE
from app.data.collectors.nvd_client import NvdEnrichment
from app.data.collectors.pipeline import build_vuln_sample, persist
from app.storage.db import init_db
from app.storage.object_store import ensure_bucket, get_client

logger = logging.getLogger(__name__)


class _MockNvdClient:
    """Stand-in for NvdClient that avoids real HTTP calls.

    Derives a conservative severity from the CVE year and builds a description
    from the vulnerable code length. This keeps the pipeline functional
    without hammering the NVD API (rate-limited to 5 req / 30s unauthenticated).
    """

    def __init__(self) -> None:
        # Intentionally empty: this offline client is stateless — no NVD API
        # credentials or session objects to initialise.  All enrichment is
        # heuristic-based (see ``fetch``).
        pass

    def fetch(self, cve_id: str, max_retries: int = 3) -> NvdEnrichment:  # NOSONAR
        # Heuristic severity: CVEs before 2010 are "high" historically,
        # recent ones default to "medium" for safety.
        year = _cve_year(cve_id)
        if year < 2010:
            severity = "high"
        else:
            severity = "medium"
        description = (
            f"Vulnerability patched in CVE {cve_id} (enrichment simulated for offline run)."
        )
        cvss_score = 5.0
        return NvdEnrichment(
            cve_id=cve_id,
            severity=severity,
            description=description,
            cvss_score=cvss_score,
        )


def _cve_year(cve_id: str) -> int:
    try:
        return int(cve_id.split("-")[1])
    except (IndexError, ValueError):
        return 2020


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 — real CVEfixes.db collection")
    ap.add_argument("--db-path", default="data/cvefixes_db/CVEfixes.db")
    ap.add_argument(
        "--max-pairs", type=int, default=2000, help="Maximum number of raw pairs to process"
    )
    ap.add_argument("--no-static-analysis", action="store_true", help="Skip Semgrep (faster)")
    ap.add_argument(
        "--languages",
        default="Python,JavaScript,TypeScript",
        help="Comma-separated languages to include (use exact DB casing)",
    )
    ap.add_argument(
        "--reduced-schema",
        action="store_true",
        help="Use the reduced 3-table loader with NVD CWE mapping "
        "(used when CVEfixes.db lacks the cwe_classification table)",
    )
    ap.add_argument(
        "--cwe-mapping",
        default="data/cve_cwe_mapping.json",
        help="Path to CVE-to-CWE JSON mapping (for --reduced-schema mode)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.info("CWE scope: %s", [spec.cwe_id for spec in CWE_SCOPE])
    logging.info("Languages filter: %s", args.languages.split(","))

    # Initialize infra
    init_db()
    client = get_client()
    ensure_bucket(client)
    logging.info("Postgres + MinIO initialized.")

    # Load pairs — try the full-schema loader first; fall back to reduced
    langs = {lang.strip() for lang in args.languages.split(",") if lang.strip()} or None
    if args.reduced_schema:
        loader = ReducedCveFixesLoader(args.db_path, cwe_mapping_path=args.cwe_mapping)
        logging.info("Using ReducedCveFixesLoader (3-table schema + NVD CWE mapping)")
    else:
        loader = CveFixesLoader(args.db_path)
        logging.info("Using CveFixesLoader (full schema)")
    all_pairs = loader.load_pairs(languages=langs)
    logging.info(
        "Loaded %d in-scope pairs (lang=%s). Limiting to %d.", len(all_pairs), langs, args.max_pairs
    )
    pairs = all_pairs[: args.max_pairs]

    # Inject mock NVD client to avoid API bottleneck
    nvd_client = _MockNvdClient()

    samples_built = 0
    skipped = 0
    start = time.monotonic()

    for i, pair in enumerate(pairs):
        try:
            sample = build_vuln_sample(
                pair,
                nvd_client,
                run_static_analysis=not args.no_static_analysis,
            )
        except Exception as exc:
            skipped += 1
            logging.warning("Skipped %s: %s", pair.cve_id, exc)
            continue

        persist(sample)
        samples_built += 1

        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logging.info(
                "Progress: %d/%d pairs processed (%d built, %d skipped), %.1f pairs/sec",
                i + 1,
                len(pairs),
                samples_built,
                skipped,
                rate,
            )

    elapsed = time.monotonic() - start
    logging.info(
        "Stage 1 complete: %d built, %d skipped in %.1fs (%.1f pairs/sec)",
        samples_built,
        skipped,
        elapsed,
        len(pairs) / elapsed if elapsed > 0 else 0,
    )


if __name__ == "__main__":
    main()
