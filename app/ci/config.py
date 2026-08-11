"""Stage 10 — regression gate configuration.

Flat, immutable dataclass with sensible defaults drawn from the README's
Stage 10 todo notes. Heavy imports (``sqlalchemy``, ``pydantic``, etc.)
are **never** performed at module import time — the gate loads artifacts
on demand so this config is usable in CI without any optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Defaults — from the README Stage 10 notes
# ---------------------------------------------------------------------------

DEFAULT_MAX_F1_DROP_PERCENT: float = 5.0
"""Fail the gate if model_cwe_macro_f1 drops more than 5% below baseline."""

DEFAULT_MIN_EXEC_PASS_RATE: float = 0.0
"""Minimum exec pass rate — 0.0 means no hard floor unless configured.

In CI mock mode the exec pass rate depends on the generated patch diffs;
a non-zero floor is recommended for real-run CI only.
"""

DEFAULT_FORGETTING_THRESHOLD: float = -0.10
"""Fail the gate if Stage 7 forgetting_delta falls below -0.10 (10% drop)."""

DEFAULT_MAX_HALLUCINATION_RATE: float = 0.50
"""Fail the gate if more than 50% of predictions hallucinate a CWE ID."""


@dataclass(frozen=True)
class RegressionGateConfig:
    """Configuration for the Stage 10 regression gate.

    Attributes
    ----------
    baseline_metrics_path:
        Path to the Stage 4 ``metrics.json`` (contains ``cwe_macro_f1``).
    stage6_report_path:
        Path to the Stage 6 ``eval_report.json`` (contains ``metrics.model_cwe_macro_f1``).
    stage7_report_path:
        Path to the Stage 7 ``regression_report.json`` (contains ``forgetting_delta``).
        May be ``None`` — the forgetting check is then skipped.
    max_f1_drop_percent:
        Maximum allowed percentage drop in CWE Macro-F1 from baseline to
        current. Default 5.0 (5%).
    min_exec_pass_rate:
        Minimum acceptable exec pass rate. Default 0.0 (no floor).
    forgetting_threshold:
        Forgetting-delta floor. If the Stage 7 delta is below this, the
        gate fails. Default -0.10.
    max_hallucination_rate:
        Maximum acceptable hallucination rate (fraction of predictions
        with an out-of-scope CWE ID). Default 0.50.
    run_id:
        Optional override for the gate run ID. If empty, a UUID-based ID
        is generated at evaluation time.
    manifest:
        Extra provenance key-value pairs to fold into the result.
    """

    baseline_metrics_path: str
    stage6_report_path: str
    stage7_report_path: str | None = None
    max_f1_drop_percent: float = DEFAULT_MAX_F1_DROP_PERCENT
    min_exec_pass_rate: float = DEFAULT_MIN_EXEC_PASS_RATE
    forgetting_threshold: float = DEFAULT_FORGETTING_THRESHOLD
    max_hallucination_rate: float = DEFAULT_MAX_HALLUCINATION_RATE
    run_id: str = ""
    manifest: dict | None = None
