"""Stage 10 — regression gate: the core decision engine.

Loads Stage 4 baseline metrics, Stage 6 eval report, and Stage 7
regression report from disk, then evaluates a set of threshold checks
to decide whether a checkpoint is worthy of promotion.

The gate is designed to be **CI-friendly**: it works with mock-mode
artifacts (no GPU, no model download) and degrades gracefully when
optional artifacts (e.g. Stage 7 report) are absent.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.ci.config import RegressionGateConfig
from app.schemas.ci import GateCheck, GateStatus, RegressionGateResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact loaders — each returns a plain dict from the JSON file on disk.
# ---------------------------------------------------------------------------


def load_baseline_metrics(path: str | Path) -> dict:
    """Load Stage 4 ``metrics.json``.

    Expected keys include ``cwe_macro_f1``, ``cwe_micro_accuracy``,
    ``severity_accuracy``, ``hallucination_rate``, ``patch_coverage``,
    and ``per_class``.

    Raises ``FileNotFoundError`` if the file does not exist.
    Raises ``RuntimeError`` if the required ``cwe_macro_f1`` key is absent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline metrics file not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "cwe_macro_f1" not in data:
        raise RuntimeError(
            f"Baseline metrics at {p} is missing 'cwe_macro_f1' — "
            "did you run Stage 4 baseline?"
        )
    logger.info("Loaded Stage 4 baseline metrics from %s (F1=%.4f)", p, data["cwe_macro_f1"])
    return data


def load_stage6_report(path: str | Path) -> dict:
    """Load Stage 6 ``eval_report.json``.

    Expected structure (mirrors ``EvalReport``):
    ``{"metrics": {"model_cwe_macro_f1": ..., "exec_pass_rate": ..., ...}, ...}``

    Raises ``FileNotFoundError`` if the file does not exist.
    Raises ``RuntimeError`` if the ``metrics`` block or required keys are absent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stage 6 report not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    metrics = data.get("metrics")
    if metrics is None:
        # Tolerate flattened structure: metrics at top level.
        metrics = data
    if "model_cwe_macro_f1" not in metrics:
        raise RuntimeError(
            f"Stage 6 report at {p} is missing 'metrics.model_cwe_macro_f1'"
        )
    logger.info("Loaded Stage 6 eval report from %s (F1=%.4f)", p, metrics["model_cwe_macro_f1"])
    return data


def load_stage7_report(path: str | Path) -> dict:
    """Load Stage 7 ``regression_report.json``.

    Expected structure (mirrors ``RegressionReport``):
    ``{"forgetting_delta": ..., "base_model": ..., "tuned_model": ..., ...}``

    Raises ``FileNotFoundError`` if the file does not exist.
    Raises ``RuntimeError`` if ``forgetting_delta`` is absent.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stage 7 report not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if "forgetting_delta" not in data:
        raise RuntimeError(
            f"Stage 7 report at {p} is missing 'forgetting_delta'"
        )
    logger.info(
        "Loaded Stage 7 regression report from %s (delta=%.4f)",
        p,
        data["forgetting_delta"],
    )
    return data


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


class RegressionGate:
    """Stage 10 regression gate — compares baseline vs. current model quality.

    Loads artifacts from disk (Stage 4 baseline, Stage 6 eval report, optional
    Stage 7 regression report) and evaluates a set of threshold checks:

    1. **CWE F1 regression** — ``model_cwe_macro_f1`` must not drop more than
       ``max_f1_drop_percent`` below the Stage 4 baseline.
    2. **Forgetting** — if a Stage 7 report is provided, ``forgetting_delta``
       must not fall below ``forgetting_threshold``.
    3. **Exec pass rate** — ``exec_pass_rate`` must meet ``min_exec_pass_rate``.
    4. **Hallucination rate** — must not exceed ``max_hallucination_rate``.

    Parameters
    ----------
    config:
        ``RegressionGateConfig`` with artifact paths and thresholds.
    baseline_metrics:
        Optional pre-loaded baseline metrics dict (bypasses file load).
    stage6_report:
        Optional pre-loaded Stage 6 report dict (bypasses file load).
    stage7_report:
        Optional pre-loaded Stage 7 report dict (bypasses file load).
    """

    def __init__(
        self,
        config: RegressionGateConfig,
        baseline_metrics: dict | None = None,
        stage6_report: dict | None = None,
        stage7_report: dict | None = None,
    ):
        self.config = config
        self._baseline = baseline_metrics
        self._stage6 = stage6_report
        self._stage7 = stage7_report

    def _get_baseline(self) -> dict:
        if self._baseline is None:
            self._baseline = load_baseline_metrics(self.config.baseline_metrics_path)
        return self._baseline

    def _get_stage6(self) -> dict:
        if self._stage6 is None:
            self._stage6 = load_stage6_report(self.config.stage6_report_path)
        return self._stage6

    def _get_stage7(self) -> dict | None:
        if self._stage7 is not None:
            return self._stage7
        if self.config.stage7_report_path:
            self._stage7 = load_stage7_report(self.config.stage7_report_path)
        return self._stage7

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_f1_regression(self) -> GateCheck:
        """Verify that CWE Macro-F1 hasn't regressed beyond the allowed threshold.

        Compares ``baseline_cwe_macro_f1`` (Stage 4) against
        ``current_cwe_macro_f1`` (Stage 6). The percentage drop is
        ``(baseline - current) / baseline * 100``. If it exceeds
        ``max_f1_drop_percent``, the check fails.

        A positive ``f1_drop_percent`` means a regression (the model got worse).
        A negative value means the model improved.
        """
        baseline = self._get_baseline()
        report6 = self._get_stage6()

        baseline_f1 = float(baseline["cwe_macro_f1"])
        current_f1 = float(report6["metrics"]["model_cwe_macro_f1"])

        if baseline_f1 > 0:
            drop_percent = (baseline_f1 - current_f1) / baseline_f1 * 100.0
        else:
            # If baseline is 0, any positive current is an improvement.
            drop_percent = 100.0 if current_f1 <= 0 else -100.0

        drop_percent = round(drop_percent, 2)
        allowed = self.config.max_f1_drop_percent

        if drop_percent <= allowed:
            status = GateStatus.PASS
            msg = (
                f"CWE Macro-F1 regression within threshold: "
                f"{baseline_f1:.4f} -> {current_f1:.4f} "
                f"(drop={drop_percent:.2f}%, allowed<={allowed:.1f}%)"
            )
        else:
            status = GateStatus.FAIL
            msg = (
                f"CWE Macro-F1 regression exceeds threshold: "
                f"{baseline_f1:.4f} -> {current_f1:.4f} "
                f"(drop={drop_percent:.2f}%, allowed<={allowed:.1f}%)"
            )

        return GateCheck(
            name="cwe_f1_regression",
            status=status,
            message=msg,
            details={
                "baseline_cwe_macro_f1": baseline_f1,
                "current_cwe_macro_f1": current_f1,
                "f1_drop_percent": drop_percent,
                "max_allowed_f1_drop_percent": allowed,
            },
        )

    def check_forgetting(self) -> GateCheck:
        """Verify that the Stage 7 forgetting delta meets the threshold.

        Only runs if a Stage 7 report was loaded. The ``forgetting_delta``
        (tuned_acc − base_acc) must not fall below ``forgetting_threshold``
        (default -0.10, i.e. no more than 10% drop in general coding ability).
        """
        report7 = self._get_stage7()

        if report7 is None:
            return GateCheck(
                name="forgetting_check",
                status=GateStatus.SKIP,
                message="Stage 7 report not provided — forgetting check skipped.",
                details={},
            )

        delta = float(report7["forgetting_delta"])
        threshold = self.config.forgetting_threshold

        if delta >= threshold:
            status = GateStatus.PASS
            msg = (
                f"Forgetting delta within threshold: {delta:+.4f} "
                f"(threshold>={threshold:+.4f})"
            )
        else:
            status = GateStatus.FAIL
            msg = (
                f"Forgetting delta below threshold: {delta:+.4f} "
                f"(threshold>={threshold:+.4f})"
            )

        return GateCheck(
            name="forgetting_check",
            status=status,
            message=msg,
            details={
                "forgetting_delta": delta,
                "forgetting_threshold": threshold,
                "base_model": report7.get("base_model", "unknown"),
                "tuned_model": report7.get("tuned_model", "unknown"),
            },
        )

    def check_exec_pass_rate(self) -> GateCheck:
        """Verify that the exec pass rate meets the minimum threshold."""
        report6 = self._get_stage6()
        metrics = report6.get("metrics", report6)
        exec_rate = float(metrics.get("exec_pass_rate", 0.0))
        min_rate = self.config.min_exec_pass_rate

        if exec_rate >= min_rate:
            status = GateStatus.PASS
            msg = (
                f"Exec pass rate meets minimum: {exec_rate:.4f} "
                f"(minimum={min_rate:.4f})"
            )
        else:
            status = GateStatus.FAIL
            msg = (
                f"Exec pass rate below minimum: {exec_rate:.4f} "
                f"(minimum={min_rate:.4f})"
            )

        return GateCheck(
            name="exec_pass_rate",
            status=status,
            message=msg,
            details={
                "exec_pass_rate": exec_rate,
                "min_exec_pass_rate": min_rate,
            },
        )

    def check_hallucination_rate(self) -> GateCheck:
        """Verify that the hallucination rate is below the maximum threshold."""
        report6 = self._get_stage6()
        metrics = report6.get("metrics", report6)
        hall_rate = float(metrics.get("hallucination_rate", 0.0))
        max_rate = self.config.max_hallucination_rate

        if hall_rate <= max_rate:
            status = GateStatus.PASS
            msg = (
                f"Hallucination rate within threshold: {hall_rate:.4f} "
                f"(maximum={max_rate:.4f})"
            )
        else:
            status = GateStatus.FAIL
            msg = (
                f"Hallucination rate exceeds threshold: {hall_rate:.4f} "
                f"(maximum={max_rate:.4f})"
            )

        return GateCheck(
            name="hallucination_rate",
            status=status,
            message=msg,
            details={
                "hallucination_rate": hall_rate,
                "max_hallucination_rate": max_rate,
            },
        )

    # ------------------------------------------------------------------
    # Top-level gate
    # ------------------------------------------------------------------

    def run_gate(self) -> RegressionGateResult:
        """Run all gate checks and return the aggregated result.

        The overall status is ``FAIL`` if any check fails, ``PASS``
        otherwise (skipped checks do not affect the outcome).
        """
        run_id = self.config.run_id or f"stage10-{uuid.uuid4().hex[:8]}"
        start = time.time()

        checks: list[GateCheck] = []
        checks.append(self.check_f1_regression())
        checks.append(self.check_forgetting())
        checks.append(self.check_exec_pass_rate())
        checks.append(self.check_hallucination_rate())

        # Determine overall status.
        any_fail = any(c.status == GateStatus.FAIL for c in checks)
        overall = GateStatus.FAIL if any_fail else GateStatus.PASS

        baseline = self._get_baseline()
        report6 = self._get_stage6()
        stage6_metrics = report6.get("metrics", report6)
        report7 = self._get_stage7()

        result = RegressionGateResult(
            status=overall,
            run_id=run_id,
            timestamp=datetime.now(UTC).isoformat(),
            baseline_cwe_macro_f1=float(baseline["cwe_macro_f1"]),
            current_cwe_macro_f1=float(stage6_metrics["model_cwe_macro_f1"]),
            f1_drop_percent=next(
                c.details["f1_drop_percent"] for c in checks
                if c.name == "cwe_f1_regression"
            ),
            max_allowed_f1_drop_percent=self.config.max_f1_drop_percent,
            exec_pass_rate=float(stage6_metrics.get("exec_pass_rate", 0.0)),
            min_exec_pass_rate=self.config.min_exec_pass_rate,
            hallucination_rate=float(stage6_metrics.get("hallucination_rate", 0.0)),
            max_hallucination_rate=self.config.max_hallucination_rate,
            forgetting_delta=float(report7["forgetting_delta"]) if report7 else None,
            forgetting_threshold=self.config.forgetting_threshold,
            checks=checks,
            manifest={
                "config": {
                    "baseline_metrics_path": self.config.baseline_metrics_path,
                    "stage6_report_path": self.config.stage6_report_path,
                    "stage7_report_path": self.config.stage7_report_path,
                },
                "max_f1_drop_percent": self.config.max_f1_drop_percent,
                "min_exec_pass_rate": self.config.min_exec_pass_rate,
                "forgetting_threshold": self.config.forgetting_threshold,
                "max_hallucination_rate": self.config.max_hallucination_rate,
                "elapsed_seconds": round(time.time() - start, 2),
                **(self.config.manifest or {}),
            },
        )

        logger.info(
            "Stage 10 gate finished: status=%s, F1 drop=%.2f%%, forgetting=%s",
            overall.value,
            result.f1_drop_percent,
            f"{result.forgetting_delta:+.4f}" if result.forgetting_delta is not None else "N/A",
        )

        return result


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def run_gate(config: RegressionGateConfig) -> RegressionGateResult:
    """Create a ``RegressionGate`` from *config* and run it immediately.

    This is the primary entry point for CLI usage.
    """
    gate = RegressionGate(config)
    return gate.run_gate()
