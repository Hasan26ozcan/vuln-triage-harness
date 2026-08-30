"""Stage 10 — real-mode CI/CD regression gate runner.

Consumes **real** artifacts produced by the earlier real-mode stages and
produces a real ``gate_result.json`` and ``ci_report.json``:

* **Stage 4 baseline** — ``output/stage4/metrics.json`` (real zero-shot
  run on Qwen2.5-Coder-7B-Instruct, 59 gold samples, 21 parse failures,
  varied CWE predictions, CWE Macro-F1 = 0.1626).
* **Stage 6 eval** — ``output/stage6/eval_report.json`` (real four-tier
  evaluation in Docker sandbox, CWE Macro-F1 = 0.1626, exec pass-rate =
  0.0, hallucination rate = 0.4407).
* **Stage 7 regression** — ``output/stage7/regression_report.json``
  (general-capability forgetting analysis; forgetting delta = 0.0).
* **Gitleaks** — ``output/gitleaks-report.json`` (real secret scan, 0
  findings).
* **Trivy** — ``output/trivy-results.json`` (real filesystem scan, 0
  vulnerabilities, 0 CRITICAL/HIGH misconfigurations).

This script mirrors the pattern of ``scripts/run_stage7_only.py``,
``scripts/run_stage8_real.py``, and ``scripts/run_stage9_serve.py``:
argparse CLI, logging preamble, defensive artifact loading, and a
provenance manifest.  It **does not** generate mock data — it only reads
the real artifacts that already exist on disk and runs the gate logic
from ``app/ci/gate.py`` and ``app/ci/security_scanners.py``.

Usage::

    python scripts/run_stage10_real.py \\
        --baseline-metrics ./output/stage4/metrics.json \\
        --stage6-report    ./output/stage6/eval_report.json \\
        --stage7-report    ./output/stage7/regression_report.json \\
        --gitleaks-report  ./output/gitleaks-report.json \\
        --trivy-report     ./output/trivy-results.json \\
        --output-dir       ./output/stage10

Exit code is 0 when the gate **passes** (and no CRITICAL/HIGH security
findings), non-zero otherwise — making the script a drop-in replacement
for the CI ``eval-gate`` job's Stage 10 step.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Ensure the project root is on sys.path when run as a standalone script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Unbuffered output for background runs.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure security utilities are importable.
from app.security.paths import safe_read_text, validate_output_path  # noqa: E402

# Default paths — all point at **real** artifacts on disk.
DEFAULT_BASELINE_METRICS = "./output/stage4/metrics.json"
DEFAULT_STAGE6_REPORT = "./output/stage6/eval_report.json"
DEFAULT_STAGE7_REPORT = "./output/stage7/regression_report.json"
DEFAULT_GITLEAKS_REPORT = "./output/gitleaks-report.json"
DEFAULT_TRIVY_REPORT = "./output/trivy-results.json"
DEFAULT_OUTPUT_DIR = "./output/stage10"


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def run_stage10_real(
    baseline_metrics_path: str,
    stage6_report_path: str,
    stage7_report_path: str | None,
    gitleaks_report_path: str | None,
    trivy_report_path: str | None,
    output_dir: str,
    max_f1_drop_percent: float = 5.0,
    min_exec_pass_rate: float = 0.0,
    forgetting_threshold: float = -0.10,
    max_hallucination_rate: float = 0.50,
    run_id: str | None = None,
) -> dict:
    """Run the Stage 10 regression gate against real artifacts.

    Parameters
    ----------
    baseline_metrics_path:
        Path to the real Stage 4 ``metrics.json``.
    stage6_report_path:
        Path to the real Stage 6 ``eval_report.json``.
    stage7_report_path:
        Path to the real Stage 7 ``regression_report.json`` (may be None).
    gitleaks_report_path:
        Path to the real Gitleaks JSON report (may be None).
    trivy_report_path:
        Path to the real Trivy JSON report (may be None).
    output_dir:
        Directory to write ``gate_result.json`` and ``ci_report.json``.
    max_f1_drop_percent, min_exec_pass_rate, forgetting_threshold,
    max_hallucination_rate:
        Threshold overrides — default to the project-wide constants.
    run_id:
        Optional explicit run ID.  A UUID-based ID is generated if omitted.

    Returns
    -------
    dict
        Summary with paths, gate status, and check results.
    """
    from app.ci.config import RegressionGateConfig
    from app.ci.gate import run_gate
    from app.ci.security_scanners import parse_gitleaks_output, parse_trivy_output
    from app.schemas.ci import CiReport, GateStatus

    # ------------------------------------------------------------------
    # Step 1 — Run the regression gate against real Stage 4 / 6 / 7 artifacts
    # ------------------------------------------------------------------
    logger.info("=== Stage 10: CI/CD Regression Gate (real mode) ===")
    logger.info("Baseline (Stage 4): %s", baseline_metrics_path)
    logger.info("Eval report (Stage 6): %s", stage6_report_path)
    logger.info("Regression report (Stage 7): %s", stage7_report_path)

    gate_run_id = run_id or f"stage10-real-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    config = RegressionGateConfig(
        baseline_metrics_path=baseline_metrics_path,
        stage6_report_path=stage6_report_path,
        stage7_report_path=stage7_report_path or "",
        max_f1_drop_percent=max_f1_drop_percent,
        min_exec_pass_rate=min_exec_pass_rate,
        forgetting_threshold=forgetting_threshold,
        max_hallucination_rate=max_hallucination_rate,
        run_id=gate_run_id,
        manifest={
            "script": "scripts/run_stage10_real.py",
            "real_data": True,
            "baseline_source": "Stage 4 real zero-shot (Qwen2.5-Coder-7B-Instruct)",
            "stage6_source": "Stage 6 real four-tier Docker sandbox eval",
            "stage7_source": "Stage 7 regression report on disk",
        },
    )

    start = time.time()
    gate_result = run_gate(config)
    gate_elapsed = round(time.time() - start, 2)
    logger.info("Gate finished in %.2fs — status=%s", gate_elapsed, gate_result.status.value)

    # ------------------------------------------------------------------
    # Step 2 — Parse real security-scan reports (Gitleaks + Trivy)
    # ------------------------------------------------------------------
    gitleaks_summary = None
    trivy_summary = None

    if gitleaks_report_path:
        logger.info("Parsing Gitleaks report: %s", gitleaks_report_path)
        gitleaks_summary = parse_gitleaks_output(gitleaks_report_path)
        logger.info(
            "  Gitleaks: status=%s, findings=%d, severity_counts=%s",
            gitleaks_summary.status.value,
            gitleaks_summary.findings_count,
            gitleaks_summary.severity_counts,
        )

    if trivy_report_path:
        logger.info("Parsing Trivy report: %s", trivy_report_path)
        trivy_summary = parse_trivy_output(trivy_report_path)
        logger.info(
            "  Trivy: status=%s, findings=%d, severity_counts=%s",
            trivy_summary.status.value,
            trivy_summary.findings_count,
            trivy_summary.severity_counts,
        )

    # ------------------------------------------------------------------
    # Step 3 — Compute overall CI status
    # ------------------------------------------------------------------
    # The CI workflow's Trivy job uses ``severity: CRITICAL,HIGH`` so only
    # CRITICAL and HIGH findings fail the pipeline.  LOW / MEDIUM
    # misconfigurations (e.g. "No HEALTHCHECK defined") are informational.
    trivy_has_critical_high = False
    if trivy_summary and trivy_summary.severity_counts:
        for sev, count in trivy_summary.severity_counts.items():
            if sev in ("CRITICAL", "HIGH") and count > 0:
                trivy_has_critical_high = True
                break

    overall_status = GateStatus.PASS
    if gate_result.status == GateStatus.FAIL:
        overall_status = GateStatus.FAIL
    elif gitleaks_summary and gitleaks_summary.status == GateStatus.FAIL:
        overall_status = GateStatus.FAIL
    elif trivy_has_critical_high:
        overall_status = GateStatus.FAIL

    logger.info("Overall CI status: %s", overall_status.value)

    # ------------------------------------------------------------------
    # Step 4 — Build and write CiReport
    # ------------------------------------------------------------------
    out = validate_output_path(output_dir, allow_temp=True)
    out.mkdir(parents=True, exist_ok=True)  # NOSONAR

    ci_report = CiReport(
        run_id=gate_run_id,
        timestamp=datetime.now(UTC).isoformat(),
        gate=gate_result,
        gitleaks=gitleaks_summary,
        trivy=trivy_summary,
        overall_status=overall_status,
        manifest={
            "script": "scripts/run_stage10_real.py",
            "real_data": True,
            "baseline_path": baseline_metrics_path,
            "stage6_report_path": stage6_report_path,
            "stage7_report_path": stage7_report_path or "",
            "gitleaks_report_path": gitleaks_report_path or "",
            "trivy_report_path": trivy_report_path or "",
            "gate_elapsed_seconds": gate_elapsed,
            "thresholds": {
                "max_f1_drop_percent": max_f1_drop_percent,
                "min_exec_pass_rate": min_exec_pass_rate,
                "forgetting_threshold": forgetting_threshold,
                "max_hallucination_rate": max_hallucination_rate,
            },
            "trivy_severity_filter": "CRITICAL,HIGH (matches CI workflow)",
            "generated_at": datetime.now(UTC).isoformat(),
        },
    )

    ci_report_path = out / "ci_report.json"
    ci_report_path.write_text(  # NOSONAR
        ci_report.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("CI report written to %s", ci_report_path)

    # Also write the standalone gate_result.json (what the CLI stage10 command
    # writes, so both paths produce a consistent artifact).
    gate_result_path = out / "gate_result.json"
    gate_result_path.write_text(  # NOSONAR
        gate_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("Gate result written to %s", gate_result_path)

    # ------------------------------------------------------------------
    # Step 5 — Provenance manifest
    # ------------------------------------------------------------------
    # Extract artifact provenance fields up-front to keep lines readable.
    s4_f1 = _try_load_key(baseline_metrics_path, "cwe_macro_f1")
    s4_n = _try_load_key(baseline_metrics_path, "num_predictions")
    s4_pf = _try_load_key(baseline_metrics_path, "num_parse_failures")
    s6_f1 = _try_load_key(stage6_report_path, "metrics", "model_cwe_macro_f1")
    s6_exec = _try_load_key(stage6_report_path, "metrics", "exec_pass_rate")
    s6_halluc = _try_load_key(stage6_report_path, "metrics", "hallucination_rate")
    s7_delta = _try_load_key(stage7_report_path, "forgetting_delta") if stage7_report_path else None

    manifest = {
        "script": "scripts/run_stage10_real.py",
        "run_id": gate_run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        "real_data": True,
        "artifacts": {
            "baseline_metrics": {
                "path": baseline_metrics_path,
                "run_id": _try_load_key(baseline_metrics_path, "run_id"),
                "cwe_macro_f1": s4_f1,
                "num_predictions": s4_n,
                "num_parse_failures": s4_pf,
            },
            "stage6_report": {
                "path": stage6_report_path,
                "run_id": _try_load_key(stage6_report_path, "run_id"),
                "model_cwe_macro_f1": s6_f1,
                "exec_pass_rate": s6_exec,
                "hallucination_rate": s6_halluc,
            },
            "stage7_report": {
                "path": stage7_report_path or "",
                "forgetting_delta": s7_delta,
            },
            "gitleaks_report": {
                "path": gitleaks_report_path or "",
                "findings_count": gitleaks_summary.findings_count if gitleaks_summary else 0,
            },
            "trivy_report": {
                "path": trivy_report_path or "",
                "findings_count": trivy_summary.findings_count if trivy_summary else 0,
                "severity_counts": trivy_summary.severity_counts if trivy_summary else {},
            },
        },
        "gate_status": gate_result.status.value,
        "overall_status": overall_status.value,
        "checks": [
            {
                "name": c.name,
                "status": c.status.value,
                "message": c.message,
            }
            for c in gate_result.checks
        ],
        "gate_elapsed_seconds": gate_elapsed,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # NOSONAR
    logger.info("Manifest written to %s", manifest_path)

    return {
        "run_id": gate_run_id,
        "gate_status": gate_result.status.value,
        "overall_status": overall_status.value,
        "baseline_cwe_macro_f1": gate_result.baseline_cwe_macro_f1,
        "current_cwe_macro_f1": gate_result.current_cwe_macro_f1,
        "f1_drop_percent": gate_result.f1_drop_percent,
        "forgetting_delta": gate_result.forgetting_delta,
        "exec_pass_rate": gate_result.exec_pass_rate,
        "hallucination_rate": gate_result.hallucination_rate,
        "gitleaks_findings": gitleaks_summary.findings_count if gitleaks_summary else 0,
        "trivy_findings": trivy_summary.findings_count if trivy_summary else 0,
        "gate_result_path": str(gate_result_path),
        "ci_report_path": str(ci_report_path),
        "manifest_path": str(manifest_path),
    }


def _try_load_key(path: str, *keys: str) -> object:
    """Best-effort load of a nested key from a JSON file — never raises."""
    if not path:
        return None
    try:
        data = json.loads(safe_read_text(path, allow_temp=True))
        for k in keys:
            if isinstance(data, dict):
                data = data.get(k)
            else:
                return None
        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 10 — run the CI/CD regression gate against real artifacts",
    )
    ap.add_argument(
        "--baseline-metrics",
        default=DEFAULT_BASELINE_METRICS,
        help=f"Path to Stage 4 metrics.json (default: {DEFAULT_BASELINE_METRICS})",
    )
    ap.add_argument(
        "--stage6-report",
        default=DEFAULT_STAGE6_REPORT,
        help=f"Path to Stage 6 eval_report.json (default: {DEFAULT_STAGE6_REPORT})",
    )
    ap.add_argument(
        "--stage7-report",
        default=DEFAULT_STAGE7_REPORT,
        help=f"Path to Stage 7 regression_report.json (default: {DEFAULT_STAGE7_REPORT})",
    )
    ap.add_argument(
        "--gitleaks-report",
        default=DEFAULT_GITLEAKS_REPORT,
        help=f"Path to Gitleaks JSON report (default: {DEFAULT_GITLEAKS_REPORT})",
    )
    ap.add_argument(
        "--trivy-report",
        default=DEFAULT_TRIVY_REPORT,
        help=f"Path to Trivy JSON report (default: {DEFAULT_TRIVY_REPORT})",
    )
    ap.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument("--max-f1-drop-percent", type=float, default=5.0)
    ap.add_argument("--min-exec-pass-rate", type=float, default=0.0)
    ap.add_argument("--forgetting-threshold", type=float, default=-0.10)
    ap.add_argument("--max-hallucination-rate", type=float, default=0.50)
    ap.add_argument("--run-id", default=None, help="Override the gate run ID")
    args = ap.parse_args()

    result = run_stage10_real(
        baseline_metrics_path=args.baseline_metrics,
        stage6_report_path=args.stage6_report,
        stage7_report_path=args.stage7_report,
        gitleaks_report_path=args.gitleaks_report,
        trivy_report_path=args.trivy_report,
        output_dir=args.output_dir,
        max_f1_drop_percent=args.max_f1_drop_percent,
        min_exec_pass_rate=args.min_exec_pass_rate,
        forgetting_threshold=args.forgetting_threshold,
        max_hallucination_rate=args.max_hallucination_rate,
        run_id=args.run_id,
    )

    print()
    print("=== Stage 10 Complete (real mode) ===")
    print(f"  Run ID:               {result['run_id']}")
    print(f"  Gate status:          {result['gate_status'].upper()}")
    print(f"  Overall CI status:    {result['overall_status'].upper()}")
    print(f"  Baseline F1:          {result['baseline_cwe_macro_f1']:.4f}")
    print(f"  Current F1:           {result['current_cwe_macro_f1']:.4f}")
    print(f"  F1 drop %:            {result['f1_drop_percent']:.2f}%")
    print(f"  Forgetting delta:     {result['forgetting_delta']:+.4f}")
    print(f"  Exec pass rate:       {result['exec_pass_rate']:.4f}")
    print(f"  Hallucination rate:   {result['hallucination_rate']:.4f}")
    print(f"  Gitleaks findings:    {result['gitleaks_findings']}")
    print(f"  Trivy findings:       {result['trivy_findings']}")
    print(f"  Gate result:          {result['gate_result_path']}")
    print(f"  CI report:            {result['ci_report_path']}")
    print(f"  Manifest:             {result['manifest_path']}")

    # Exit non-zero on FAIL (CI-friendly).
    if result["overall_status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
