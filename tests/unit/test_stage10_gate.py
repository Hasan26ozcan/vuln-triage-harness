"""Unit tests for Stage 10 — regression gate.

Covers:
  - Schema validation (GateStatus, GateCheck, RegressionGateResult, SecurityScanSummary)
  - Artifact loaders (load_baseline_metrics, load_stage6_report, load_stage7_report)
  - RegressionGate individual checks (F1 regression, forgetting, exec pass rate, hallucination rate)
  - Security-scan parsers (parse_gitleaks_output, parse_trivy_output)
  - JSON round-trip serialization
  - Edge cases: zero baseline F1, missing Stage 7 report, file-not-found
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

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
from app.schemas.ci import (
    CiReport,
    GateCheck,
    GateStatus,
    RegressionGateResult,
    SecurityScanSummary,
)

# ---------------------------------------------------------------------------
# Helpers — build synthetic artifacts that mirror what Stages 4-7 write
# ---------------------------------------------------------------------------


def _baseline_metrics(f1: float = 0.80, halluc: float = 0.05) -> dict:
    return {
        "cwe_macro_f1": f1,
        "cwe_micro_accuracy": 0.85,
        "severity_accuracy": 0.75,
        "hallucination_rate": halluc,
        "patch_coverage": 0.90,
        "per_class": {"CWE-89": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 2}},
    }


def _stage6_report(f1: float = 0.78, exec_rate: float = 0.60, hall: float = 0.10) -> dict:
    return {
        "run_id": "stage6-test",
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "stage": 6,
        "num_samples": 12,
        "num_predictions": 12,
        "tier1_results": [],
        "tier2_results": [],
        "exec_results": [],
        "llm_judge_scores": [],
        "metrics": {
            "num_samples": 12,
            "num_predictions": 12,
            "tier1_cwe_macro_f1": 0.95,
            "tier1_coverage": 1.0,
            "tier2_cwe_macro_f1": 0.90,
            "tier2_coverage": 1.0,
            "model_cwe_macro_f1": f1,
            "exec_pass_rate": exec_rate,
            "patch_applies_rate": 0.90,
            "build_succeeds_rate": 0.95,
            "hallucination_rate": hall,
            "avg_patch_coverage": 0.95,
            "per_class": {},
        },
        "manifest": {"test": "data"},
    }


def _stage7_report(delta: float = -0.05) -> dict:
    return {
        "run_id": "stage7-test",
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "tuned_model": "sft_qlora_r8",
        "base_metrics": {
            "num_tasks": 12,
            "num_passed": 10,
            "execution_accuracy": 0.8333,
            "task_results": [],
        },
        "tuned_metrics": {
            "num_tasks": 12,
            "num_passed": 9,
            "execution_accuracy": 0.75,
            "task_results": [],
        },
        "forgetting_delta": delta,
        "manifest": {"test": "data"},
    }


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestGateStatus:
    def test_values(self):
        assert GateStatus.PASS.value == "pass"
        assert GateStatus.FAIL.value == "fail"
        assert GateStatus.SKIP.value == "skip"

    def test_is_str_enum(self):
        # GateStatus is a str Enum so it serializes as its string value.
        assert GateStatus.PASS.value == "pass"
        assert isinstance(GateStatus.PASS, str)


class TestGateCheck:
    def test_default_details(self):
        check = GateCheck(name="test", status=GateStatus.PASS, message="ok")
        assert check.details == {}

    def test_with_details(self):
        check = GateCheck(
            name="test",
            status=GateStatus.FAIL,
            message="bad",
            details={"val": 1},
        )
        assert check.details == {"val": 1}


class TestRegressionGateResult:
    def test_defaults(self):
        result = RegressionGateResult(
            run_id="test",
            timestamp="2024-01-01T00:00:00Z",
            baseline_cwe_macro_f1=0.80,
            current_cwe_macro_f1=0.78,
            f1_drop_percent=2.5,
            max_allowed_f1_drop_percent=5.0,
            exec_pass_rate=0.60,
            min_exec_pass_rate=0.0,
            hallucination_rate=0.10,
            max_hallucination_rate=0.50,
        )
        assert result.status == GateStatus.PASS
        assert result.passed is True
        assert result.forgetting_delta is None
        assert result.checks == []

    def test_overall_fail_when_check_fails(self):
        result = RegressionGateResult(
            run_id="test",
            timestamp="2024-01-01T00:00:00Z",
            baseline_cwe_macro_f1=0.80,
            current_cwe_macro_f1=0.60,
            f1_drop_percent=25.0,
            max_allowed_f1_drop_percent=5.0,
            exec_pass_rate=0.60,
            min_exec_pass_rate=0.0,
            hallucination_rate=0.10,
            max_hallucination_rate=0.50,
            status=GateStatus.FAIL,
            checks=[
                GateCheck(name="f1", status=GateStatus.FAIL, message="bad"),
            ],
        )
        assert result.passed is False


class TestSecurityScanSummary:
    def test_default_severity_counts(self):
        summary = SecurityScanSummary(tool="gitleaks", status=GateStatus.PASS, findings_count=0)
        assert summary.severity_counts == {}
        assert summary.details == []


class TestCiReport:
    def test_defaults(self):
        report = CiReport(run_id="ci-1", timestamp="2024-01-01T00:00:00Z")
        assert report.overall_status == GateStatus.PASS
        assert report.gate is None
        assert report.gitleaks is None
        assert report.trivy is None


# ---------------------------------------------------------------------------
# Artifact loader tests
# ---------------------------------------------------------------------------


class TestLoadBaselineMetrics:
    def test_load_valid(self, tmp_path):
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(_baseline_metrics(0.85)), encoding="utf-8")
        data = load_baseline_metrics(path)
        assert data["cwe_macro_f1"] == 0.85

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_baseline_metrics(tmp_path / "nonexistent.json")

    def test_missing_f1_key(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"other_key": 1}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="cwe_macro_f1"):
            load_baseline_metrics(path)


class TestLoadStage6Report:
    def test_load_valid(self, tmp_path):
        path = tmp_path / "eval_report.json"
        path.write_text(json.dumps(_stage6_report(0.78)), encoding="utf-8")
        data = load_stage6_report(path)
        assert data["metrics"]["model_cwe_macro_f1"] == 0.78

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_stage6_report(tmp_path / "nonexistent.json")

    def test_missing_model_f1_key(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"metrics": {"other": 1}}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="model_cwe_macro_f1"):
            load_stage6_report(path)

    def test_flattened_structure(self, tmp_path):
        """Stage 6 report with metrics at top level (no nested 'metrics' key)."""
        path = tmp_path / "eval_report.json"
        flat_report = {
            "run_id": "stage6-flat",
            "model_cwe_macro_f1": 0.78,
            "exec_pass_rate": 0.60,
            "hallucination_rate": 0.10,
        }
        path.write_text(json.dumps(flat_report), encoding="utf-8")
        data = load_stage6_report(path)
        assert data["model_cwe_macro_f1"] == 0.78


class TestLoadStage7Report:
    def test_load_valid(self, tmp_path):
        path = tmp_path / "regression_report.json"
        path.write_text(json.dumps(_stage7_report(-0.05)), encoding="utf-8")
        data = load_stage7_report(path)
        assert data["forgetting_delta"] == -0.05

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_stage7_report(tmp_path / "nonexistent.json")

    def test_missing_delta_key(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"other": 1}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="forgetting_delta"):
            load_stage7_report(path)


# ---------------------------------------------------------------------------
# RegressionGate — individual checks
# ---------------------------------------------------------------------------


class TestCheckF1Regression:
    def _make_gate(self, baseline_f1=0.80, current_f1=0.78, max_drop=5.0):
        return RegressionGate(
            config=RegressionGateConfig(
                baseline_metrics_path="dummy",
                stage6_report_path="dummy",
                max_f1_drop_percent=max_drop,
            ),
            baseline_metrics=_baseline_metrics(baseline_f1),
            stage6_report=_stage6_report(current_f1),
        )

    def test_pass_within_threshold(self):
        gate = self._make_gate(baseline_f1=0.80, current_f1=0.78, max_drop=5.0)
        check = gate.check_f1_regression()
        assert check.status == GateStatus.PASS
        assert check.name == "cwe_f1_regression"
        assert check.details["f1_drop_percent"] == 2.5  # (0.80-0.78)/0.80*100

    def test_fail_exceeds_threshold(self):
        gate = self._make_gate(baseline_f1=0.80, current_f1=0.70, max_drop=5.0)
        check = gate.check_f1_regression()
        assert check.status == GateStatus.FAIL
        assert check.details["f1_drop_percent"] == round((0.80 - 0.70) / 0.80 * 100, 2)

    def test_improvement_is_pass(self):
        """A higher current F1 (negative drop) should pass."""
        gate = self._make_gate(baseline_f1=0.80, current_f1=0.90, max_drop=5.0)
        check = gate.check_f1_regression()
        assert check.status == GateStatus.PASS
        assert check.details["f1_drop_percent"] < 0

    def test_zero_baseline(self):
        """When baseline F1 is 0 and current is also 0, drop is 100%."""
        gate = self._make_gate(baseline_f1=0.0, current_f1=0.0, max_drop=5.0)
        check = gate.check_f1_regression()
        assert check.status == GateStatus.FAIL
        assert check.details["f1_drop_percent"] == 100.0

    def test_negative_drop_when_improving(self):
        gate = self._make_gate(baseline_f1=0.80, current_f1=0.85, max_drop=5.0)
        check = gate.check_f1_regression()
        assert check.details["f1_drop_percent"] < 0


class TestCheckForgetting:
    def _make_gate(self, delta=-0.05, threshold=-0.10, with_stage7=True):
        stage7 = _stage7_report(delta) if with_stage7 else None
        return RegressionGate(
            config=RegressionGateConfig(
                baseline_metrics_path="dummy",
                stage6_report_path="dummy",
                stage7_report_path="dummy" if with_stage7 else None,
                forgetting_threshold=threshold,
            ),
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78),
            stage7_report=stage7,
        )

    def test_within_threshold(self):
        gate = self._make_gate(delta=-0.05, threshold=-0.10)
        check = gate.check_forgetting()
        assert check.status == GateStatus.PASS
        assert check.details["forgetting_delta"] == -0.05

    def test_exceeds_threshold(self):
        gate = self._make_gate(delta=-0.15, threshold=-0.10)
        check = gate.check_forgetting()
        assert check.status == GateStatus.FAIL

    def test_exact_threshold(self):
        """Delta equal to threshold should pass (boundary)."""
        gate = self._make_gate(delta=-0.10, threshold=-0.10)
        check = gate.check_forgetting()
        assert check.status == GateStatus.PASS

    def test_no_stage7_report(self):
        gate = self._make_gate(with_stage7=False)
        check = gate.check_forgetting()
        assert check.status == GateStatus.SKIP
        assert "skipped" in check.message.lower()

    def test_positive_delta_passes(self):
        """Improvement (positive delta) should always pass."""
        gate = self._make_gate(delta=0.10, threshold=-0.10)
        check = gate.check_forgetting()
        assert check.status == GateStatus.PASS

    def test_stage7_loaded_from_file(self, tmp_path):
        """When stage7_report is not pre-loaded, _get_stage7 loads from the file path."""
        stage7_path = tmp_path / "regression_report.json"
        stage7_path.write_text(json.dumps(_stage7_report(-0.05)), encoding="utf-8")

        config = RegressionGateConfig(
            baseline_metrics_path="dummy",
            stage6_report_path="dummy",
            stage7_report_path=str(stage7_path),
            forgetting_threshold=-0.10,
        )
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78),
        )
        check = gate.check_forgetting()
        assert check.status == GateStatus.PASS
        assert check.details["forgetting_delta"] == -0.05


class TestCheckExecPassRate:
    def _make_gate(self, exec_rate=0.60, min_rate=0.25):
        return RegressionGate(
            config=RegressionGateConfig(
                baseline_metrics_path="dummy",
                stage6_report_path="dummy",
                min_exec_pass_rate=min_rate,
            ),
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78, exec_rate=exec_rate),
        )

    def test_above_minimum(self):
        gate = self._make_gate(exec_rate=0.60, min_rate=0.25)
        check = gate.check_exec_pass_rate()
        assert check.status == GateStatus.PASS

    def test_below_minimum(self):
        gate = self._make_gate(exec_rate=0.10, min_rate=0.25)
        check = gate.check_exec_pass_rate()
        assert check.status == GateStatus.FAIL

    def test_zero_floor_always_passes(self):
        gate = self._make_gate(exec_rate=0.0, min_rate=0.0)
        check = gate.check_exec_pass_rate()
        assert check.status == GateStatus.PASS


class TestCheckHallucinationRate:
    def _make_gate(self, hall_rate=0.10, max_rate=0.50):
        return RegressionGate(
            config=RegressionGateConfig(
                baseline_metrics_path="dummy",
                stage6_report_path="dummy",
                max_hallucination_rate=max_rate,
            ),
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78, hall=hall_rate),
        )

    def test_within_threshold(self):
        gate = self._make_gate(hall_rate=0.10, max_rate=0.50)
        check = gate.check_hallucination_rate()
        assert check.status == GateStatus.PASS

    def test_exceeds_threshold(self):
        gate = self._make_gate(hall_rate=0.60, max_rate=0.50)
        check = gate.check_hallucination_rate()
        assert check.status == GateStatus.FAIL

    def test_zero_rate_passes(self):
        gate = self._make_gate(hall_rate=0.0, max_rate=0.50)
        check = gate.check_hallucination_rate()
        assert check.status == GateStatus.PASS


# ---------------------------------------------------------------------------
# RegressionGate — run_gate end-to-end
# ---------------------------------------------------------------------------


class TestRunGate:
    def _make_config(self, **overrides):
        defaults = dict(
            baseline_metrics_path="dummy",
            stage6_report_path="dummy",
            stage7_report_path=None,
            max_f1_drop_percent=5.0,
            min_exec_pass_rate=0.0,
            forgetting_threshold=-0.10,
            max_hallucination_rate=0.50,
        )
        defaults.update(overrides)
        return RegressionGateConfig(**defaults)

    def test_all_pass(self):
        config = self._make_config(stage7_report_path="dummy")
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78, exec_rate=0.60, hall=0.10),
            stage7_report=_stage7_report(-0.05),
        )
        result = gate.run_gate()
        assert result.status == GateStatus.PASS
        assert result.passed is True
        assert len(result.checks) == 4
        assert all(c.status != GateStatus.FAIL for c in result.checks)

    def test_f1_drop_fails(self):
        config = self._make_config(stage7_report_path="dummy")
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.60, exec_rate=0.60, hall=0.10),
            stage7_report=_stage7_report(-0.05),
        )
        result = gate.run_gate()
        assert result.status == GateStatus.FAIL
        f1_check = next(c for c in result.checks if c.name == "cwe_f1_regression")
        assert f1_check.status == GateStatus.FAIL

    def test_forgetting_fails(self):
        config = self._make_config(stage7_report_path="dummy")
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78),
            stage7_report=_stage7_report(-0.20),  # exceeds -0.10 threshold
        )
        result = gate.run_gate()
        assert result.status == GateStatus.FAIL
        forget_check = next(c for c in result.checks if c.name == "forgetting_check")
        assert forget_check.status == GateStatus.FAIL

    def test_forgetting_skipped_without_stage7(self):
        config = self._make_config(stage7_report_path=None)
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78),
        )
        result = gate.run_gate()
        # No Stage 7 → forgetting check skipped, gate should still pass.
        forget_check = next(c for c in result.checks if c.name == "forgetting_check")
        assert forget_check.status == GateStatus.SKIP
        assert result.status == GateStatus.PASS

    def test_multiple_failures(self):
        config = self._make_config(
            stage7_report_path="dummy",
            max_f1_drop_percent=5.0,
            forgetting_threshold=-0.05,
            max_hallucination_rate=0.10,
        )
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.70, exec_rate=0.10, hall=0.50),
            stage7_report=_stage7_report(-0.20),
        )
        result = gate.run_gate()
        assert result.status == GateStatus.FAIL
        fail_count = sum(1 for c in result.checks if c.status == GateStatus.FAIL)
        assert fail_count >= 2

    def test_result_serializable(self):
        config = self._make_config(stage7_report_path="dummy")
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78),
            stage7_report=_stage7_report(-0.05),
        )
        result = gate.run_gate()
        json_str = result.model_dump_json(indent=2)
        data = json.loads(json_str)
        assert data["run_id"]
        assert data["status"] in ("pass", "fail", "skip")
        assert len(data["checks"]) == 4

    def test_run_id_from_config(self):
        config = self._make_config(
            stage7_report_path="dummy",
            run_id="my-custom-id",
        )
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78),
            stage7_report=_stage7_report(-0.05),
        )
        result = gate.run_gate()
        assert result.run_id == "my-custom-id"

    def test_generated_run_id(self):
        config = self._make_config(stage7_report_path="dummy")
        gate = RegressionGate(
            config=config,
            baseline_metrics=_baseline_metrics(0.80),
            stage6_report=_stage6_report(0.78),
            stage7_report=_stage7_report(-0.05),
        )
        result = gate.run_gate()
        assert result.run_id.startswith("stage10-")


def test_run_gate_convenience_function(tmp_path):
    """The module-level run_gate() helper should work with real files on disk."""
    baseline_path = tmp_path / "metrics.json"
    baseline_path.write_text(json.dumps(_baseline_metrics(0.80)), encoding="utf-8")

    stage6_path = tmp_path / "eval_report.json"
    stage6_path.write_text(json.dumps(_stage6_report(0.79)), encoding="utf-8")

    config = RegressionGateConfig(
        baseline_metrics_path=str(baseline_path),
        stage6_report_path=str(stage6_path),
        stage7_report_path=None,
    )
    result = run_gate(config)
    # No Stage 7 → forgetting check skipped, F1 drop within 5%, exec rate 0 ≥ 0.0.
    assert result.status == GateStatus.PASS
    assert result.checks[0].name == "cwe_f1_regression"


# ---------------------------------------------------------------------------
# Security scan parser tests
# ---------------------------------------------------------------------------


class TestParseGitleaksOutput:
    def test_empty_output(self):
        summary = parse_gitleaks_output(None)
        assert summary.tool == "gitleaks"
        assert summary.status == GateStatus.PASS
        assert summary.findings_count == 0

    def test_no_findings(self):
        summary = parse_gitleaks_output("[]")
        assert summary.status == GateStatus.PASS
        assert summary.findings_count == 0

    def test_with_findings(self):
        raw = json.dumps(
            [
                {"rule": "aws-access-key", "line": 10, "severity": "HIGH", "file": "a.py"},
                {"rule": "generic-secret", "line": 20, "severity": "CRITICAL", "file": "b.py"},
            ]
        )
        summary = parse_gitleaks_output(raw)
        assert summary.status == GateStatus.FAIL
        assert summary.findings_count == 2
        assert summary.severity_counts.get("HIGH") == 1
        assert summary.severity_counts.get("CRITICAL") == 1

    def test_invalid_json(self):
        summary = parse_gitleaks_output("not json{{{")
        assert summary.status == GateStatus.PASS
        assert summary.findings_count == 0

    def test_from_file(self, tmp_path):
        report = tmp_path / "gitleaks.json"
        report.write_text(
            json.dumps(
                [
                    {"rule": "test", "severity": "MEDIUM"},
                ]
            ),
            encoding="utf-8",
        )
        summary = parse_gitleaks_output(str(report))
        assert summary.findings_count == 1
        assert summary.status == GateStatus.FAIL

    def test_from_file_path_object(self, tmp_path):
        """Passing a Path object (not a string) to resolve_raw."""
        report = tmp_path / "gitleaks.json"
        report.write_text(
            json.dumps(
                [
                    {"rule": "test", "severity": "HIGH"},
                ]
            ),
            encoding="utf-8",
        )
        summary = parse_gitleaks_output(report)
        assert summary.findings_count == 1
        assert summary.status == GateStatus.FAIL

    def test_nonexistent_path_object(self):
        """_resolve_raw with a non-existent Path returns empty string."""
        from pathlib import Path

        from app.ci.security_scanners import _resolve_raw

        result = _resolve_raw(Path("/nonexistent/file.json"))
        assert result == ""

    def test_dict_with_findings_key(self):
        """Gitleaks output as a dict with a 'findings' key (not a bare array)."""
        raw = json.dumps({"findings": [{"severity": "HIGH"}, {"severity": "LOW"}]})
        summary = parse_gitleaks_output(raw)
        assert summary.findings_count == 2
        assert summary.severity_counts.get("HIGH") == 1
        assert summary.severity_counts.get("LOW") == 1

    def test_non_list_non_findings_dict(self):
        """Gitleaks output as a dict without 'findings' key → 0 findings (fallback)."""
        raw = json.dumps({"some_other_key": "value"})
        summary = parse_gitleaks_output(raw)
        assert summary.findings_count == 0
        assert summary.status == GateStatus.PASS

    def test_json_primitive(self):
        """Gitleaks output as a JSON primitive (e.g. string) → 0 findings."""
        summary = parse_gitleaks_output(json.dumps("just a string"))
        assert summary.findings_count == 0
        assert summary.status == GateStatus.PASS


class TestParseTrivyOutput:
    def test_empty_output(self):
        summary = parse_trivy_output(None)
        assert summary.tool == "trivy"
        assert summary.status == GateStatus.PASS
        assert summary.findings_count == 0

    def test_no_vulnerabilities(self):
        raw = json.dumps({"Results": []})
        summary = parse_trivy_output(raw)
        assert summary.status == GateStatus.PASS
        assert summary.findings_count == 0

    def test_with_vulnerabilities(self):
        raw = json.dumps(
            {
                "Results": [
                    {
                        "Target": "python-pkg:requests@2.28.0",
                        "Class": "os-pkgs",
                        "Type": "python",
                        "Vulnerabilities": [
                            {
                                "VulnerabilityID": "CVE-2023-1234",
                                "Severity": "HIGH",
                                "Title": "test vuln",
                            },
                        ],
                    },
                ],
            }
        )
        summary = parse_trivy_output(raw)
        assert summary.status == GateStatus.FAIL
        assert summary.findings_count == 1
        assert summary.details[0]["target"] == "python-pkg:requests@2.28.0"
        assert summary.severity_counts.get("HIGH") == 1

    def test_with_misconfigurations(self):
        raw = json.dumps(
            {
                "Results": [
                    {
                        "Target": "Dockerfile",
                        "Class": "config",
                        "Misconfigurations": [
                            {"MisconfigID": "DS002", "Severity": "HIGH", "Title": "test config"},
                        ],
                    },
                ],
            }
        )
        summary = parse_trivy_output(raw)
        assert summary.findings_count == 1

    def test_invalid_json(self):
        summary = parse_trivy_output("garbage{")
        assert summary.status == GateStatus.PASS
        assert summary.findings_count == 0

    def test_from_file(self, tmp_path):
        report = tmp_path / "trivy.json"
        report.write_text(
            json.dumps(
                {
                    "Results": [
                        {
                            "Target": "pkg",
                            "Vulnerabilities": [{"VulnerabilityID": "CVE-1", "Severity": "LOW"}],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        summary = parse_trivy_output(str(report))
        assert summary.findings_count == 1
        assert summary.status == GateStatus.FAIL

    def test_from_file_path_object(self, tmp_path):
        """Passing a Path object (not a string) to resolve_raw for trivy."""
        report = tmp_path / "trivy.json"
        report.write_text(
            json.dumps(
                {
                    "Results": [
                        {
                            "Target": "pkg",
                            "Vulnerabilities": [
                                {"VulnerabilityID": "CVE-1", "Severity": "CRITICAL"}
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        summary = parse_trivy_output(report)
        assert summary.findings_count == 1
        assert summary.status == GateStatus.FAIL

    def test_non_dict_entry_in_results(self):
        """A non-dict entry in Trivy Results array is skipped."""
        text = json.dumps(
            {
                "Results": [
                    "not-a-dict",
                    {"Target": "pkg", "Vulnerabilities": [{"Severity": "HIGH"}]},
                ],
            }
        )
        summary = parse_trivy_output(text)
        assert summary.findings_count == 1

    def test_secrets_and_licenses_keys(self):
        """Trivy findings under Secrets and Licenses keys are also collected."""
        text = json.dumps(
            {
                "Results": [
                    {
                        "Target": "src",
                        "Secrets": [{"Severity": "CRITICAL", "Title": "secret"}],
                        "Licenses": [{"Severity": "LOW", "Title": "license"}],
                    },
                ],
            }
        )
        summary = parse_trivy_output(text)
        assert summary.findings_count == 2


# ---------------------------------------------------------------------------
# Count severity edge cases
# ---------------------------------------------------------------------------


class TestCountSeverities:
    def test_empty(self):
        from app.ci.security_scanners import _count_severities

        assert _count_severities([], "severity") == {}

    def test_missing_severity(self):
        from app.ci.security_scanners import _count_severities

        counts = _count_severities([{"rule": "x"}], "severity")
        assert counts.get("UNKNOWN") == 1

    def test_mixed_case(self):
        from app.ci.security_scanners import _count_severities

        counts = _count_severities(
            [{"severity": "high"}, {"severity": "HIGH"}],
            "severity",
        )
        assert counts.get("HIGH") == 2

    def test_none_severity_value(self):
        """A finding whose severity value is explicitly None is bucketed as UNKNOWN."""
        from app.ci.security_scanners import _count_severities

        counts = _count_severities([{"severity": None}], "severity")
        assert counts.get("UNKNOWN") == 1


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestRegressionGateConfig:
    def test_defaults(self):
        config = RegressionGateConfig(
            baseline_metrics_path="a.json",
            stage6_report_path="b.json",
        )
        assert config.max_f1_drop_percent == 5.0
        assert config.min_exec_pass_rate == 0.0
        assert config.forgetting_threshold == -0.10
        assert config.max_hallucination_rate == 0.50
        assert config.stage7_report_path is None
        assert config.run_id == ""

    def test_frozen(self):
        """RegressionGateConfig is a frozen dataclass."""
        config = RegressionGateConfig(
            baseline_metrics_path="a.json",
            stage6_report_path="b.json",
        )
        with pytest.raises(FrozenInstanceError):  # frozen=True
            config.max_f1_drop_percent = 10.0
