"""Integration tests for Stage 8 — quantization matrix.

These tests exercise the full Stage 8 pipeline end-to-end:

  1. **Mock mode** — ``run_quantization_matrix`` with ``mock=True``,
     producing a full QuantReport with results and a best-config.
  2. **Dry-run mode** — heuristic estimates only (no ML deps).
  3. **Real-mode failure handling** — the matrix catches import errors
     and produces ``FAILED`` results instead of crashing.
  4. **CLI integration** — ``stage8 --mock`` via Typer's CliRunner,
     verifying report output file and console summary.
  5. **Score / select cross-checks** — the best result from the matrix
     is consistent with ``select_best_config``.

No GPU or heavy ML dependencies are required — all paths use mock or
dry-run mode, or expect (and verify) failures when deps are absent.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from app.schemas.quantization import (
    QuantMethod,
    QuantResult,
    QuantStatus,
)
from app.quantization.config import QuantConfig
from app.quantization.quantizer import (
    run_quantization_matrix,
    select_best_config,
    score_quality_size_speed,
)


# ---------------------------------------------------------------------------
# 1. Mock mode — full matrix
# ---------------------------------------------------------------------------


class TestMockMatrixEndToEnd:
    """Full quantization matrix in mock mode — fully deterministic."""

    def test_full_matrix_all_methods(self):
        config = QuantConfig(
            source_checkpoint="dummy_source",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            output_base="/tmp/stage8_integration",
            methods=[QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.GGUF],
            bit_widths=[2, 3, 4, 8],
            mock=True,
        )
        report = run_quantization_matrix(config)

        assert report.run_id.startswith("stage8-")
        assert report.base_model == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert report.source_checkpoint == "dummy_source"

        # GPTQ: 4 bit_widths, AWQ: 4 bit_widths, GGUF: 6 quant_types
        assert len(report.results) == 14

        # All should be COMPLETED in mock mode.
        for r in report.results:
            assert r.status == QuantStatus.COMPLETED
            assert r.quantized_model_size_gb > 0
            assert r.estimated_vram_gb > 0

    def test_best_result_is_highest_scoring(self):
        """The best_result should match manual select_best_config on results."""
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.GGUF],
            bit_widths=[2, 3, 4, 8],
        )
        report = run_quantization_matrix(config)

        manual_best = select_best_config(report.results)
        assert report.best_result is not None
        assert manual_best is not None
        assert report.best_result.quant_method == manual_best.quant_method
        assert report.best_result.bit_width == manual_best.bit_width

    def test_lower_bits_smaller_size(self):
        """2-bit should produce smaller checkpoints than 8-bit."""
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ],
            bit_widths=[2, 8],
        )
        report = run_quantization_matrix(config)

        sizes = {r.bit_width: r.quantized_model_size_gb for r in report.results}
        assert sizes[2] < sizes[8]

    def test_json_round_trip(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ, QuantMethod.GGUF],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        data = json.loads(report.model_dump_json())

        assert data["run_id"].startswith("stage8-")
        assert len(data["results"]) == 1 + 6  # GPTQ(1) + GGUF(6)
        assert data["best_result"] is not None
        assert data["manifest"]["mock"] is True


# ---------------------------------------------------------------------------
# 2. Dry-run mode
# ---------------------------------------------------------------------------


class TestDryRunMatrix:
    def test_dry_run_produces_results(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=False,
            dry_run=True,
            methods=[QuantMethod.GPTQ, QuantMethod.AWQ],
            bit_widths=[4, 8],
        )
        report = run_quantization_matrix(config)

        assert len(report.results) == 4  # 2 methods × 2 bit_widths
        for r in report.results:
            assert r.status == QuantStatus.COMPLETED
            assert "dry-run" in r.notes
            assert r.tokens_per_sec is not None
            assert r.model_cwe_macro_f1 is not None


# ---------------------------------------------------------------------------
# 3. Real-mode failure handling
# ---------------------------------------------------------------------------


class TestRealModeFailureHandling:
    def test_matrix_does_not_crash_without_deps(self):
        """When no ML backends are installed, all results should be FAILED
        but the matrix still returns a valid report."""
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_real_integration",
            methods=[QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.GGUF],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)

        # Should have results for all methods.
        assert len(report.results) > 0

        # All should be FAILED (no ML deps in test env).
        for r in report.results:
            assert r.status == QuantStatus.FAILED
            assert r.error is not None

        # best_result should be None (no completed results).
        assert report.best_result is None


# ---------------------------------------------------------------------------
# 4. CLI integration
# ---------------------------------------------------------------------------


class TestStage8CLI:
    def test_cli_mock_mode(self, tmp_path):
        """CLI stage8 --mock should run and write a quant_report.json."""
        app_module = pytest.importorskip("app.evaluation.cli")
        runner = CliRunner()

        result = runner.invoke(
            app_module.app,
            [
                "stage8",
                "--source-checkpoint", "dummy_source",
                "--mock",
                "--output-dir", str(tmp_path / "stage8_out"),
                "--methods", "gptq,gguf",
                "--bits", "4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Stage 8" in result.output
        assert "Best config:" in result.output

        report_path = tmp_path / "stage8_out" / "quant_report.json"
        assert report_path.exists()

        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["run_id"].startswith("stage8-")
        assert "gptq" in data["manifest"]["methods"]
        assert "gguf" in data["manifest"]["methods"]
        assert data["manifest"]["mock"] is True

    def test_cli_dry_run_mode(self, tmp_path):
        """CLI stage8 --dry-run should run without mock and write a report."""
        app_module = pytest.importorskip("app.evaluation.cli")
        runner = CliRunner()

        result = runner.invoke(
            app_module.app,
            [
                "stage8",
                "--source-checkpoint", "dummy_source",
                "--dry-run",
                "--output-dir", str(tmp_path / "stage8_dry"),
                "--methods", "gptq",
                "--bits", "4",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run:       True" in result.output

        report_path = tmp_path / "stage8_dry" / "quant_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["manifest"]["dry_run"] is True

    def test_cli_mock_with_vram_constraint(self, tmp_path):
        """CLI with --target-vram should filter results."""
        app_module = pytest.importorskip("app.evaluation.cli")
        runner = CliRunner()

        result = runner.invoke(
            app_module.app,
            [
                "stage8",
                "--source-checkpoint", "dummy",
                "--mock",
                "--output-dir", str(tmp_path / "stage8_vram"),
                "--methods", "gptq",
                "--bits", "2,4,8",
                "--target-vram", "6.0",
            ],
        )
        assert result.exit_code == 0, result.output

        report_path = tmp_path / "stage8_vram" / "quant_report.json"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert data["best_result"] is not None
        assert data["best_result"]["estimated_vram_gb"] <= 6.0

    def test_cli_invalid_method(self, tmp_path):
        """CLI with an unknown method should fail with exit code 1."""
        app_module = pytest.importorskip("app.evaluation.cli")
        runner = CliRunner()

        result = runner.invoke(
            app_module.app,
            [
                "stage8",
                "--source-checkpoint", "dummy",
                "--mock",
                "--output-dir", str(tmp_path / "stage8_bad"),
                "--methods", "bogus",
            ],
        )
        assert result.exit_code == 1
        assert "unknown quantization method" in result.output.lower()

    def test_cli_no_source_checkpoint(self):
        """CLI without --source-checkpoint should fail (it's required)."""
        app_module = pytest.importorskip("app.evaluation.cli")
        runner = CliRunner()

        result = runner.invoke(
            app_module.app,
            ["stage8", "--mock"],
        )
        assert result.exit_code != 0

    def test_cli_all_methods_default(self, tmp_path):
        """CLI with no --methods should use all three by default."""
        app_module = pytest.importorskip("app.evaluation.cli")
        runner = CliRunner()

        result = runner.invoke(
            app_module.app,
            [
                "stage8",
                "--source-checkpoint", "dummy",
                "--mock",
                "--output-dir", str(tmp_path / "stage8_all"),
                "--bits", "4",
            ],
        )
        assert result.exit_code == 0, result.output

        report_path = tmp_path / "stage8_all" / "quant_report.json"
        data = json.loads(report_path.read_text(encoding="utf-8"))
        assert sorted(data["manifest"]["methods"]) == ["awq", "gguf", "gptq"]


# ---------------------------------------------------------------------------
# 5. Cross-check: score consistency
# ---------------------------------------------------------------------------


class TestScoreConsistency:
    def test_best_has_highest_score(self):
        """The best_result from the matrix should have the highest
        score_quality_size_speed among all completed results."""
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.GGUF],
            bit_widths=[2, 3, 4, 8],
        )
        report = run_quantization_matrix(config)

        completed = [r for r in report.results if r.status == QuantStatus.COMPLETED]
        assert len(completed) > 0

        scores = [score_quality_size_speed(r) for r in completed]
        best_idx = scores.index(max(scores))

        assert report.best_result is not None
        assert report.best_result.quant_method == completed[best_idx].quant_method
        assert report.best_result.bit_width == completed[best_idx].bit_width


# ---------------------------------------------------------------------------
# 6. Report structure & provenance
# ---------------------------------------------------------------------------


class TestReportStructure:
    def test_manifest_has_required_fields(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)

        manifest = report.manifest
        assert "run_id" in manifest
        assert "started_at" in manifest
        assert "elapsed_seconds" in manifest
        assert "base_model" in manifest
        assert "source_checkpoint" in manifest
        assert "methods" in manifest
        assert "bit_widths" in manifest
        assert "mock" in manifest
        assert "dry_run" in manifest

    def test_result_fields_populated(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        r = report.results[0]

        assert r.quant_method == QuantMethod.GPTQ
        assert r.bit_width == 4
        assert r.status == QuantStatus.COMPLETED
        assert r.quantized_model_size_gb > 0
        assert r.estimated_vram_gb > 0
        assert r.checkpoint_path != ""

    def test_none_method_in_matrix(self):
        """QuantMethod.NONE should produce a pass-through baseline result."""
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.NONE],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        r = report.results[0]
        assert r.quant_method == QuantMethod.NONE
        assert r.status == QuantStatus.COMPLETED
        # In mock mode, NONE goes through MockQuantizer (estimates size via heuristics).
        assert "mock none" in r.notes
