"""Stage 11 - Demo script for the vulnerability triage harness.

This script demonstrates the project's capabilities in **mock mode** (no model
download, no GPU, no Docker required).  It runs the four-tier evaluation
harness on the gold-eval set and prints a summary of results.

Run::

    python docs/demo.py
    python docs/demo.py --gold-eval eval/gold_set/gold.jsonl
    python docs/demo.py --verbose
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if "--gold-eval" in sys.argv:
        gold_eval = sys.argv[sys.argv.index("--gold-eval") + 1]
    else:
        gold_eval = os.path.join("eval", "gold_set", "gold.jsonl")
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Ensure the project root is on the path when run from docs/.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import logging
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # --- Step 1: Stage 4 baseline (mock) ---
    print("=" * 60)
    print("Stage 11 Demo - Mock Mode")
    print("=" * 60)
    print("\n1. Stage 4 - Baseline evaluation (mock backend)")
    print(f"   Gold-eval: {gold_eval}")

    from typer.testing import CliRunner as _CliRunner

    import app.evaluation.cli as cli_mod  # noqa: F811

    runner = _CliRunner()
    result = runner.invoke(
        cli_mod.app,
        [
            "baseline",
            "--gold-eval", str(project_root / gold_eval),
            "--strategy", "zero_shot",
            "--mock",
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage4"),
        ],
    )
    if result.exit_code != 0:
        print(f"   ERROR: {result.output}")
        sys.exit(1)
    print("   [OK] Baseline complete")
    if verbose:
        print(result.output)

    # --- Step 2: Stage 6 four-tier evaluation (mock) ---
    print("\n2. Stage 6 - Four-tier evaluation (mock sandbox)")
    stage4_dir = project_root / "output" / "stage11_demo" / "stage4"
    result = runner.invoke(
        cli_mod.app,
        [
            "stage6",
            "--gold-eval", str(project_root / gold_eval),
            "--predictions", str(stage4_dir / "predictions.jsonl"),
            "--sandbox-mode", "mock",
            "--skip-tier4",
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage6"),
        ],
    )
    if result.exit_code != 0:
        print(f"   ERROR: {result.output}")
        sys.exit(1)
    print("   [OK] Evaluation complete")
    if verbose:
        print(result.output)

    # --- Step 3: Stage 7 regression (mock) ---
    print("\n3. Stage 7 - Regression / forgetting analysis (mock)")
    result = runner.invoke(
        cli_mod.app,
        [
            "stage7",
            "--mock",
            "--base-model", "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--tuned-model", "stage11-demo-checkpoint",
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage7"),
        ],
    )
    if result.exit_code != 0:
        print(f"   ERROR: {result.output}")
        sys.exit(1)
    print("   [OK] Regression analysis complete")
    if verbose:
        print(result.output)

    # --- Step 4: Stage 10 gate ---
    print("\n4. Stage 10 - Regression gate")
    stage6_dir = project_root / "output" / "stage11_demo" / "stage6"
    stage7_dir = project_root / "output" / "stage11_demo" / "stage7"
    result = runner.invoke(
        cli_mod.app,
        [
            "stage10",
            "--baseline-metrics", str(stage4_dir / "metrics.json"),
            "--predictions", str(stage4_dir / "predictions.jsonl"),
            "--stage6-report", str(stage6_dir / "eval_report.json"),
            "--stage7-report", str(stage7_dir / "regression_report.json"),
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage10"),
        ],
    )
    print(f"   [OK] Gate result (exit code {result.exit_code})")
    if verbose:
        print(result.output)

    # --- Step 5: Print final summary ---
    print(f"\n{'=' * 60}")
    print("Demo complete! All stages ran successfully in mock mode.")
    print(f"{'=' * 60}")
    print(f"\nArtifacts written to: {project_root / 'output' / 'stage11_demo'}")
    print("\nDocumentation deliverables:")
    print("  - docs/model_card.md")
    print("  - docs/training_report.md")
    print("  - docs/demo.py (this file)")


if __name__ == "__main__":
    main()
