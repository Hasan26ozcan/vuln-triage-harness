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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when run as a standalone script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Unbuffered output for background runs.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
sys.stderr.reconfigure(line_buffering=True)  # type: ignore[union-attr]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure security utilities are importable.
from app.ci.config import (  # noqa: E402
    DEFAULT_FORGETTING_THRESHOLD,
    DEFAULT_MAX_F1_DROP_PERCENT,
    DEFAULT_MAX_HALLUCINATION_RATE,
    DEFAULT_MIN_EXEC_PASS_RATE,
)
from app.security.paths import safe_read_text, validate_output_path  # noqa: E402

# Default paths — all point at **real** artifacts on disk.
DEFAULT_BASELINE_METRICS = "./output/stage4/metrics.json"
DEFAULT_STAGE6_REPORT = "./output/stage6/eval_report.json"
DEFAULT_STAGE7_REPORT = "./output/stage7/regression_report.json"
DEFAULT_GITLEAKS_REPORT = "./output/gitleaks-report.json"
DEFAULT_TRIVY_REPORT = "./output/trivy-results.json"
DEFAULT_OUTPUT_DIR = "./output/stage10"

# Script name used in provenance manifests — extracted to avoid the
# duplicate-literal smell (SonarQube S1132).
SCRIPT_NAME = "scripts/run_stage10_real.py"

# Metrics dict keys — shared across the provenance manifest, the gate metrics
# builder, and downstream result parsing (SonarQube S1132).
_KEY_RUN_ID = "run_id"
_KEY_EXEC_PASS_RATE = "exec_pass_rate"  # nosec B105 — "pass" here is "pass rate", not a password
_KEY_HALLUCINATION_RATE = "hallucination_rate"
_KEY_OVERALL_STATUS = "overall_status"
_KEY_FORGETTING_DELTA = "forgetting_delta"
_KEY_METRICS = "metrics"
_KEY_SCRIPT = "script"
_KEY_REAL_DATA = "real_data"
_KEY_GATE_STATUS = "gate_status"

# Additional keys used in the artifacts sub-dict (SonarQube S1132).
_KEY_BASELINE_METRICS = "baseline_metrics"
_KEY_STAGE6_REPORT = "stage6_report"
_KEY_STAGE7_REPORT = "stage7_report"
_KEY_GITLEAKS_REPORT = "gitleaks_report"
_KEY_TRIVY_REPORT = "trivy_report"
_KEY_FINDINGS_COUNT = "findings_count"
_KEY_SEVERITY_COUNTS = "severity_counts"
_KEY_PATH = "path"


@dataclass(frozen=True)
class GateThresholds:
    """Threshold overrides for the Stage 10 regression gate."""

    max_f1_drop_percent: float = DEFAULT_MAX_F1_DROP_PERCENT
    min_exec_pass_rate: float = DEFAULT_MIN_EXEC_PASS_RATE
    forgetting_threshold: float = DEFAULT_FORGETTING_THRESHOLD
    max_hallucination_rate: float = DEFAULT_MAX_HALLUCINATION_RATE

# Severity values that fail the CI pipeline (Trivy job uses severity: CRITICAL,HIGH).
_CRITICAL_SEVERITIES = ("CRITICAL", "HIGH")


def _parse_security_reports(
    gitleaks_report_path: str | None,
    trivy_report_path: str | None,
) -> tuple:
    """Parse real Gitleaks and Trivy JSON reports.

    Returns ``(gitleaks_summary, trivy_summary)`` — either may be ``None``
    when the corresponding path is ``None``.
    """
    from app.ci.security_scanners import parse_gitleaks_output, parse_trivy_output

    gitleaks_summary = None
    if gitleaks_report_path:
        logger.info("Parsing Gitleaks report: %s", gitleaks_report_path)
        gitleaks_summary = parse_gitleaks_output(gitleaks_report_path)
        logger.info(
            "  Gitleaks: status=%s, findings=%d, severity_counts=%s",
            gitleaks_summary.status.value,
            gitleaks_summary.findings_count,
            gitleaks_summary.severity_counts,
        )

    trivy_summary = None
    if trivy_report_path:
        logger.info("Parsing Trivy report: %s", trivy_report_path)
        trivy_summary = parse_trivy_output(trivy_report_path)
        logger.info(
            "  Trivy: status=%s, findings=%d, severity_counts=%s",
            trivy_summary.status.value,
            trivy_summary.findings_count,
            trivy_summary.severity_counts,
        )

    return gitleaks_summary, trivy_summary


def _trivy_has_critical_high(trivy_summary: Any | None) -> bool:
    """Return True if Trivy reported any CRITICAL or HIGH findings."""
    if not trivy_summary or not trivy_summary.severity_counts:
        return False
    return any(
        sev in _CRITICAL_SEVERITIES and count > 0
        for sev, count in trivy_summary.severity_counts.items()
    )


def _compute_ci_status(
    gate_result: Any,
    gitleaks_summary: Any | None,
    trivy_has_critical: bool,
) -> Any:
    """Determine overall CI pass/fail from gate, gitleaks, and trivy results."""
    from app.schemas.ci import GateStatus

    if gate_result.status == GateStatus.FAIL:
        return GateStatus.FAIL
    if gitleaks_summary and gitleaks_summary.status == GateStatus.FAIL:
        return GateStatus.FAIL
    if trivy_has_critical:
        return GateStatus.FAIL
    return GateStatus.PASS


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


def _gather_artifact_provenance(
    baseline_metrics_path: str,
    stage6_report_path: str,
    stage7_report_path: str | None,
    gitleaks_report_path: str | None,
    trivy_report_path: str | None,
    gitleaks_summary: Any | None,
    trivy_summary: Any | None,
) -> dict:
    """Build the ``artifacts`` sub-dict for the provenance manifest."""
    s4_f1 = _try_load_key(baseline_metrics_path, "cwe_macro_f1")
    s4_n = _try_load_key(baseline_metrics_path, "num_predictions")
    s4_pf = _try_load_key(baseline_metrics_path, "num_parse_failures")
    s6_f1 = _try_load_key(stage6_report_path, _KEY_METRICS, "model_cwe_macro_f1")
    s6_exec = _try_load_key(stage6_report_path, _KEY_METRICS, _KEY_EXEC_PASS_RATE)
    s6_halluc = _try_load_key(stage6_report_path, _KEY_METRICS, _KEY_HALLUCINATION_RATE)
    s7_delta = (
        _try_load_key(stage7_report_path, _KEY_FORGETTING_DELTA) if stage7_report_path else None
    )

    return {
        _KEY_BASELINE_METRICS: {
            _KEY_PATH: baseline_metrics_path,
            _KEY_RUN_ID: _try_load_key(baseline_metrics_path, _KEY_RUN_ID),
            "cwe_macro_f1": s4_f1,
            "num_predictions": s4_n,
            "num_parse_failures": s4_pf,
        },
        _KEY_STAGE6_REPORT: {
            _KEY_PATH: stage6_report_path,
            _KEY_RUN_ID: _try_load_key(stage6_report_path, _KEY_RUN_ID),
            "model_cwe_macro_f1": s6_f1,
            _KEY_EXEC_PASS_RATE: s6_exec,
            _KEY_HALLUCINATION_RATE: s6_halluc,
        },
        _KEY_STAGE7_REPORT: {
            _KEY_PATH: stage7_report_path or "",
            _KEY_FORGETTING_DELTA: s7_delta,
        },
        _KEY_GITLEAKS_REPORT: {
            _KEY_PATH: gitleaks_report_path or "",
            _KEY_FINDINGS_COUNT: gitleaks_summary.findings_count if gitleaks_summary else 0,
        },
        _KEY_TRIVY_REPORT: {
            _KEY_PATH: trivy_report_path or "",
            _KEY_FINDINGS_COUNT: trivy_summary.findings_count if trivy_summary else 0,
            _KEY_SEVERITY_COUNTS: trivy_summary.severity_counts if trivy_summary else {},
        },
    }


def _build_manifest(
    gate_run_id: str,
    artifacts: dict,
    gate_result: Any,
    overall_status: Any,
    gate_elapsed: float,
) -> dict:
    """Build the provenance manifest dict."""
    return {
        _KEY_SCRIPT: SCRIPT_NAME,
        _KEY_RUN_ID: gate_run_id,
        "timestamp": datetime.now(UTC).isoformat(),
        _KEY_REAL_DATA: True,
        "artifacts": artifacts,
        _KEY_GATE_STATUS: gate_result.status.value,
        _KEY_OVERALL_STATUS: overall_status.value,
        "checks": [
            {"name": c.name, "status": c.status.value, "message": c.message}
            for c in gate_result.checks
        ],
        "gate_elapsed_seconds": gate_elapsed,
    }


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
    thresholds: GateThresholds | None = None,
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
    thresholds:
        Threshold overrides — defaults to the project-wide constants.
    run_id:
        Optional explicit run ID.  A UUID-based ID is generated if omitted.

    Returns
    -------
    dict
        Summary with paths, gate status, and check results.
    """
    from app.ci.config import RegressionGateConfig
    from app.ci.gate import run_gate
    from app.schemas.ci import CiReport

    t = thresholds or GateThresholds()

    logger.info("=== Stage 10: CI/CD Regression Gate (real mode) ===")
    logger.info("Baseline (Stage 4): %s", baseline_metrics_path)
    logger.info("Eval report (Stage 6): %s", stage6_report_path)
    logger.info("Regression report (Stage 7): %s", stage7_report_path)

    gate_run_id = run_id or f"stage10-real-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

    config = RegressionGateConfig(
        baseline_metrics_path=baseline_metrics_path,
        stage6_report_path=stage6_report_path,
        stage7_report_path=stage7_report_path or "",
        max_f1_drop_percent=t.max_f1_drop_percent,
        min_exec_pass_rate=t.min_exec_pass_rate,
        forgetting_threshold=t.forgetting_threshold,
        max_hallucination_rate=t.max_hallucination_rate,
        run_id=gate_run_id,
        manifest={
            _KEY_SCRIPT: SCRIPT_NAME,
            _KEY_REAL_DATA: True,
            "baseline_source": "Stage 4 real zero-shot (Qwen2.5-Coder-7B-Instruct)",
            "stage6_source": "Stage 6 real four-tier Docker sandbox eval",
            "stage7_source": "Stage 7 regression report on disk",
        },
    )

    start = time.time()
    gate_result = run_gate(config)
    gate_elapsed = round(time.time() - start, 2)
    logger.info("Gate finished in %.2fs — status=%s", gate_elapsed, gate_result.status.value)

    gitleaks_summary, trivy_summary = _parse_security_reports(
        gitleaks_report_path, trivy_report_path
    )

    trivy_critical = _trivy_has_critical_high(trivy_summary)
    overall_status = _compute_ci_status(gate_result, gitleaks_summary, trivy_critical)
    logger.info("Overall CI status: %s", overall_status.value)

    # Step 4 — Build and write CiReport
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
            _KEY_SCRIPT: SCRIPT_NAME,
            _KEY_REAL_DATA: True,
            "baseline_path": baseline_metrics_path,
            "stage6_report_path": stage6_report_path,
            "stage7_report_path": stage7_report_path or "",
            "gitleaks_report_path": gitleaks_report_path or "",
            "trivy_report_path": trivy_report_path or "",
            "gate_elapsed_seconds": gate_elapsed,
            "thresholds": {
                "max_f1_drop_percent": t.max_f1_drop_percent,
                "min_exec_pass_rate": t.min_exec_pass_rate,
                "forgetting_threshold": t.forgetting_threshold,
                "max_hallucination_rate": t.max_hallucination_rate,
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

    gate_result_path = out / "gate_result.json"
    gate_result_path.write_text(  # NOSONAR
        gate_result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info("Gate result written to %s", gate_result_path)

    # Step 5 — Provenance manifest
    artifacts = _gather_artifact_provenance(
        baseline_metrics_path,
        stage6_report_path,
        stage7_report_path,
        gitleaks_report_path,
        trivy_report_path,
        gitleaks_summary,
        trivy_summary,
    )
    manifest = _build_manifest(
        gate_run_id,
        artifacts,
        gate_result,
        overall_status,
        gate_elapsed,
    )
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # NOSONAR
    logger.info("Manifest written to %s", manifest_path)

    return {
        _KEY_RUN_ID: gate_run_id,
        _KEY_GATE_STATUS: gate_result.status.value,
        _KEY_OVERALL_STATUS: overall_status.value,
        "baseline_cwe_macro_f1": gate_result.baseline_cwe_macro_f1,
        "current_cwe_macro_f1": gate_result.current_cwe_macro_f1,
        "f1_drop_percent": gate_result.f1_drop_percent,
        _KEY_FORGETTING_DELTA: gate_result.forgetting_delta,
        _KEY_EXEC_PASS_RATE: gate_result.exec_pass_rate,
        _KEY_HALLUCINATION_RATE: gate_result.hallucination_rate,
        "gitleaks_findings": gitleaks_summary.findings_count if gitleaks_summary else 0,
        "trivy_findings": trivy_summary.findings_count if trivy_summary else 0,
        "gate_result_path": str(gate_result_path),
        "ci_report_path": str(ci_report_path),
        "manifest_path": str(manifest_path),
    }


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
    ap.add_argument(
        "--max-f1-drop-percent",
        type=float,
        default=DEFAULT_MAX_F1_DROP_PERCENT,
        help=f"Max allowed F1 drop (%%) vs baseline (default: {DEFAULT_MAX_F1_DROP_PERCENT})",
    )
    ap.add_argument(
        "--min-exec-pass-rate",
        type=float,
        default=DEFAULT_MIN_EXEC_PASS_RATE,
        help=f"Min execution pass rate (default: {DEFAULT_MIN_EXEC_PASS_RATE})",
    )
    ap.add_argument(
        "--forgetting-threshold",
        type=float,
        default=DEFAULT_FORGETTING_THRESHOLD,
        help=f"Max forgetting delta (default: {DEFAULT_FORGETTING_THRESHOLD})",
    )
    ap.add_argument(
        "--max-hallucination-rate",
        type=float,
        default=DEFAULT_MAX_HALLUCINATION_RATE,
        help=f"Max hallucination rate (default: {DEFAULT_MAX_HALLUCINATION_RATE})",
    )
    ap.add_argument("--run-id", default=None, help="Override the gate run ID")
    args = ap.parse_args()

    thresholds = GateThresholds(
        max_f1_drop_percent=args.max_f1_drop_percent,
        min_exec_pass_rate=args.min_exec_pass_rate,
        forgetting_threshold=args.forgetting_threshold,
        max_hallucination_rate=args.max_hallucination_rate,
    )

    result = run_stage10_real(
        baseline_metrics_path=args.baseline_metrics,
        stage6_report_path=args.stage6_report,
        stage7_report_path=args.stage7_report,
        gitleaks_report_path=args.gitleaks_report,
        trivy_report_path=args.trivy_report,
        output_dir=args.output_dir,
        thresholds=thresholds,
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
    if result[_KEY_OVERALL_STATUS] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
