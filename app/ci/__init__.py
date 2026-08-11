"""Stage 10 — CI/CD regression gate.

Implements the regression gate that compares a fine-tuned checkpoint's
CWE Macro-F1 against the Stage 4 baseline (from ``metrics.json``), checks
for catastrophic forgetting (Stage 7 ``regression_report.json``), verifies
exec pass-rate and hallucination thresholds, and produces a structured
``RegressionGateResult`` for the CI workflow.

Also provides Gitleaks and Trivy output parsers for the security-scan
jobs in the CI workflow.

Usage (CLI)::

    python -m app.evaluation.cli stage10 \\
        --baseline-metrics ./output/stage4/metrics.json \\
        --predictions ./output/stage4/predictions.jsonl \\
        --stage6-report ./output/stage6/eval_report.json \\
        --stage7-report ./output/stage7/regression_report.json

Usage (programmatic)::

    from app.ci.gate import RegressionGate, RegressionGateConfig

    config = RegressionGateConfig(
        baseline_metrics_path="./output/stage4/metrics.json",
        stage6_report_path="./output/stage6/eval_report.json",
        stage7_report_path="./output/stage7/regression_report.json",
    )
    result = RegressionGate(config).run_gate()
    if not result.passed:
        raise SystemExit(1)
"""

from app.ci.config import RegressionGateConfig
from app.ci.gate import (
    RegressionGate,
    load_baseline_metrics,
    load_stage6_report,
    load_stage7_report,
    run_gate,
)
from app.ci.security_scanners import (
    parse_gitleaks_output,
    parse_trivy_output,
)

__all__ = [
    "RegressionGateConfig",
    "RegressionGate",
    "run_gate",
    "load_baseline_metrics",
    "load_stage6_report",
    "load_stage7_report",
    "parse_gitleaks_output",
    "parse_trivy_output",
]
