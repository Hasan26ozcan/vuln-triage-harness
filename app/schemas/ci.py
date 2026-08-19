"""Stage 10 — CI/CD regression gate and security-scan contracts.

Pydantic models for the regression gate that compares a fine-tuned model's
CWE Macro-F1 against the Stage 4 baseline, checks for catastrophic
forgetting (Stage 7), and aggregates the results into a single
``RegressionGateResult`` that CI turns into a pass/fail decision.

Also includes ``SecurityScanSummary`` for Gitleaks and Trivy output
parsing — both tools are wired into the CI workflow as separate jobs and
their summaries are reported alongside the regression gate result.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class GateStatus(StrEnum):
    """Outcome of a gate check or the overall gate."""

    # Enum value, not a password — safe from B105 false positive
    PASS = "pass"  # nosec B105
    FAIL = "fail"
    SKIP = "skip"


class GateCheck(BaseModel):
    """Result of a single individual check within the regression gate.

    Each ``RegressionGate`` run produces a list of ``GateCheck`` objects —
    one per metric being verified. A gate that contains any ``FAIL`` check
    overall status is ``FAIL``.

    Attributes
    ----------
    name:
        Human-readable identifier for the check (e.g. ``"cwe_f1_regression"``).
    status:
        ``GateStatus.PASS`` / ``FAIL`` / ``SKIP``.
    message:
        Short human-readable explanation of the result.
    details:
        Arbitrary key-value context (the raw numbers that were compared).
    """

    name: str
    status: GateStatus
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class RegressionGateResult(BaseModel):
    """Full result of running the Stage 10 regression gate.

    Written to ``gate_result.json`` in CI. If any check has
    ``status == GateStatus.FAIL``, the CI step exits non-zero.

    Attributes
    ----------
    status:
        Overall gate status — ``FAIL`` if any check failed, ``PASS`` otherwise.
    run_id:
        Unique identifier for this gate run.
    timestamp:
        ISO-8601 timestamp of when the gate was evaluated.
    baseline_cwe_macro_f1:
        The Stage 4 baseline CWE Macro-F1 (the "before" number).
    current_cwe_macro_f1:
        The Stage 6 model CWE Macro-F1 (the "after" number).
    f1_drop_percent:
        Percentage drop from baseline to current:
        ``(baseline - current) / baseline * 100``. Positive means a drop.
    max_allowed_f1_drop_percent:
        The threshold beyond which the gate fails (default 5.0).
    exec_pass_rate:
        Fraction of predictions where the proposed patch passes exec-eval tests.
    min_exec_pass_rate:
        Minimum acceptable exec pass rate (default 0.0 — no hard floor unless
        configured, since mock runs with trivial code may produce low rates).
    hallucination_rate:
        Fraction of predictions with an out-of-scope CWE ID.
    max_hallucination_rate:
        Maximum acceptable hallucination rate (default 0.50).
    forgetting_delta:
        Stage 7 general-capability delta (``tuned_acc - base_acc``), or ``None``
        if the Stage 7 report was not provided.
    forgetting_threshold:
        The forgetting-delta floor (default -0.10); a delta below this is a FAIL.
    checks:
        Individual ``GateCheck`` results for each verification.
    manifest:
        Run provenance — config path, report paths, CI run ID, etc.
    """

    status: GateStatus = GateStatus.PASS
    run_id: str
    timestamp: str
    baseline_cwe_macro_f1: float
    current_cwe_macro_f1: float
    f1_drop_percent: float
    max_allowed_f1_drop_percent: float
    exec_pass_rate: float
    min_exec_pass_rate: float
    hallucination_rate: float
    max_hallucination_rate: float
    forgetting_delta: float | None = None
    forgetting_threshold: float | None = None
    checks: list[GateCheck] = Field(default_factory=list)
    manifest: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when the overall gate status is PASS."""
        return self.status == GateStatus.PASS


class SecurityScanSummary(BaseModel):
    """Summary of a single security-scanning tool run (Gitleaks or Trivy).

    Attributes
    ----------
    tool:
        ``"gitleaks"`` or ``"trivy"``.
    status:
        ``GateStatus.PASS`` if no findings, ``FAIL`` otherwise.
    findings_count:
        Total number of findings reported by the tool.
    severity_counts:
        Per-severity breakdown (e.g. ``{"HIGH": 2, "CRITICAL": 1}``).
    details:
        Raw finding records (or a truncated sample for very large outputs).
    """

    tool: str
    status: GateStatus
    findings_count: int
    severity_counts: dict[str, int] = Field(default_factory=dict)
    details: list[dict[str, Any]] = Field(default_factory=list)


class CiReport(BaseModel):
    """Top-level CI report aggregating all Stage 10 checks.

    Combines the regression gate result with security-scan summaries into
    a single artifact (``ci_report.json``) that CI writes for archival and
    local inspection.
    """

    run_id: str
    timestamp: str
    gate: RegressionGateResult | None = None
    gitleaks: SecurityScanSummary | None = None
    trivy: SecurityScanSummary | None = None
    overall_status: GateStatus = GateStatus.PASS
    manifest: dict[str, Any] = Field(default_factory=dict)
