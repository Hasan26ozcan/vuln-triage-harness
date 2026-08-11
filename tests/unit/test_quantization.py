"""Unit tests for Stage 8 — quantization config, backends, and selection.

Tests cover:

* Config validation warnings (GPTQ, AWQ, GGUF).
* Heuristic estimation functions (VRAM, size, quality, tokens/sec).
* ``MockQuantizer`` deterministic behavior.
* ``Quantizer`` Protocol duck-typing (``runtime_checkable``).
* ``select_best_config`` with / without VRAM and size constraints.
* ``score_quality_size_speed`` for completed and failed results.
* ``quantize_single`` in mock / dry-run / real (lazy-import failure) modes.
* ``gguf_type_to_bits`` / ``_bits_to_gguf_type`` mappings.
* ``_estimate_unquantized_size`` on a real temp directory.
* ``_NoOpQuantizer`` pass-through behavior.
"""

from __future__ import annotations

import os

import pytest

from app.schemas.quantization import (
    QuantMethod,
    QuantResult,
    QuantStatus,
)
from app.quantization.config import (
    AWQConfig,
    GGUFConfig,
    GPTQConfig,
    QuantConfig,
    estimate_model_size_gb,
    estimate_quality,
    estimate_tokens_per_sec,
    estimate_vram_gb,
)
from app.quantization.export_gguf import gguf_type_to_bits
from app.quantization.quantizer import (
    MockQuantizer,
    Quantizer,
    _NoOpQuantizer,
    _estimate_unquantized_size,
    quantize_single,
    run_quantization_matrix,
    score_quality_size_speed,
    select_best_config,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestGPTQConfigValidation:
    def test_default_bits_ok(self):
        w = GPTQConfig().validate()
        assert w == []

    def test_invalid_bits_warning(self):
        w = GPTQConfig(bits=6).validate()
        assert any("only 2/3/4" in msg for msg in w)

    def test_invalid_group_size_warning(self):
        w = GPTQConfig(group_size=50).validate()
        assert any("group_size" in msg for msg in w)

    def test_invalid_damping_warning(self):
        w = GPTQConfig(damping=1.5).validate()
        assert any("damping" in msg for msg in w)


class TestAWQConfigValidation:
    def test_default_ok(self):
        assert AWQConfig().validate() == []

    def test_invalid_bits_warning(self):
        w = AWQConfig(bits=6).validate()
        assert any("only 2–5" in msg for msg in w)


class TestGGUFConfigValidation:
    def test_default_ok(self):
        assert GGUFConfig().validate() == []

    def test_unknown_quant_type_warning(self):
        w = GGUFConfig(quant_types=["Q9_K"]).validate()
        assert any("not a recognized type" in msg for msg in w)

    def test_valid_quant_types_no_warning(self):
        w = GGUFConfig(quant_types=["Q4_K", "Q8_0", "F16"]).validate()
        assert w == []


class TestQuantConfigDefaults:
    def test_methods_defaults_to_all_three(self):
        c = QuantConfig(source_checkpoint="dummy")
        assert QuantMethod.GPTQ in c.methods
        assert QuantMethod.AWQ in c.methods
        assert QuantMethod.GGUF in c.methods

    def test_methods_override(self):
        c = QuantConfig(
            source_checkpoint="dummy",
            methods=[QuantMethod.GPTQ],
        )
        assert c.methods == [QuantMethod.GPTQ]

    def test_run_name(self):
        c = QuantConfig(
            source_checkpoint="dummy",
            base_model="Qwen/Qwen2.5-Coder-7B",
        )
        assert c.run_name == "quant_Qwen2.5-Coder-7B"

    def test_all_warnings_combines_subconfigs(self):
        c = QuantConfig(
            source_checkpoint="dummy",
            gptq_config=GPTQConfig(bits=6),
            awq_config=AWQConfig(bits=6),
            gguf_config=GGUFConfig(quant_types=["Q9_K"]),
        )
        warnings = c.all_warnings()
        assert len(warnings) >= 3


# ---------------------------------------------------------------------------
# Heuristic estimation functions
# ---------------------------------------------------------------------------


class TestEstimateVRAM:
    def test_gptq_4_bit(self):
        assert estimate_vram_gb(QuantMethod.GPTQ, 4) == 7.5

    def test_awq_2_bit(self):
        assert estimate_vram_gb(QuantMethod.AWQ, 2) == 5.0

    def test_gguf_4_bit(self):
        assert estimate_vram_gb(QuantMethod.GGUF, 4) == 7.0

    def test_none_returns_15(self):
        assert estimate_vram_gb(QuantMethod.NONE, 4) == 15.0

    def test_unknown_method_returns_default(self):
        assert estimate_vram_gb(QuantMethod.NONE, 4) == 15.0


class TestEstimateModelSize:
    def test_none_is_14gb(self):
        assert estimate_model_size_gb(QuantMethod.NONE, 4) == 14.0

    def test_gguf_overhead(self):
        size = estimate_model_size_gb(QuantMethod.GGUF, 4)
        assert size == 6.8  # 6.5 + 0.3

    def test_gptq_4_bit(self):
        assert estimate_model_size_gb(QuantMethod.GPTQ, 4) == 6.5


class TestEstimateQuality:
    def test_4_bit_quality(self):
        assert estimate_quality(QuantMethod.GPTQ, 4) == 0.92

    def test_2_bit_quality(self):
        assert estimate_quality(QuantMethod.GPTQ, 2) == 0.60

    def test_unknown_bits_default(self):
        assert estimate_quality(QuantMethod.GPTQ, 6) == 0.50


class TestEstimateTokensPerSec:
    def test_gpu_4_bit(self):
        assert estimate_tokens_per_sec(QuantMethod.GPTQ, 4) == 35.0

    def test_cpu_gguf_4_bit(self):
        assert estimate_tokens_per_sec(QuantMethod.GGUF, 4, device="cpu") == 8.0

    def test_cpu_non_gguf(self):
        assert estimate_tokens_per_sec(QuantMethod.AWQ, 4, device="cpu") == 4.0


# ---------------------------------------------------------------------------
# MockQuantizer
# ---------------------------------------------------------------------------


class TestMockQuantizer:
    def test_basic_quantize(self):
        m = MockQuantizer(
            default_method=QuantMethod.GPTQ,
            default_bit_width=4,
        )
        result = m.quantize("/fake/source", "/fake/output", 4)

        assert result.quant_method == QuantMethod.GPTQ
        assert result.bit_width == 4
        assert result.status == QuantStatus.COMPLETED
        assert result.checkpoint_path == "/fake/output"
        assert "mock gptq" in result.notes
        assert m.call_count == 1
        assert m.last_call["source"] == "/fake/source"

    def test_call_count_increment(self):
        m = MockQuantizer(default_method=QuantMethod.AWQ, default_bit_width=4)
        m.quantize("a", "b", 4)
        m.quantize("a", "b", 4)
        assert m.call_count == 2

    def test_canned_result_override(self):
        canned = QuantResult(
            quant_method=QuantMethod.GPTQ,
            bit_width=4,
            quantized_model_size_gb=99.0,
            estimated_vram_gb=99.0,
            status=QuantStatus.COMPLETED,
        )
        m = MockQuantizer(
            default_method=QuantMethod.GPTQ,
            default_bit_width=4,
            results={"gptq:4": canned},
        )
        result = m.quantize("src", "out", 4)
        assert result.quantized_model_size_gb == 99.0

    def test_last_call_tracks_args(self):
        m = MockQuantizer(default_method=QuantMethod.NONE, default_bit_width=8)
        m.quantize("src.pt", "out/gptq", 2)
        assert m.last_call["bits"] == 2
        assert m.last_call["output"] == "out/gptq"


# ---------------------------------------------------------------------------
# Quantizer Protocol
# ---------------------------------------------------------------------------


class TestQuantizerProtocol:
    def test_mock_quantizer_satisfies_protocol(self):
        m = MockQuantizer(
            default_method=QuantMethod.NONE,
            default_bit_width=4,
        )
        assert isinstance(m, Quantizer)

    def test_noop_quantizer_satisfies_protocol(self):
        n = _NoOpQuantizer()
        assert isinstance(n, Quantizer)

    def test_non_quantizer_does_not_satisfy(self):
        assert not isinstance("not a quantizer", Quantizer)


# ---------------------------------------------------------------------------
# NoOpQuantizer
# ---------------------------------------------------------------------------


class TestNoOpQuantizer:
    def test_noop_quantize(self, tmp_path):
        n = _NoOpQuantizer()
        result = n.quantize(str(tmp_path), str(tmp_path / "out"))

        assert result.quant_method == QuantMethod.NONE
        assert result.bit_width == 16
        assert result.status == QuantStatus.COMPLETED
        assert result.notes == "no quantization (FP16 baseline)"


# ---------------------------------------------------------------------------
# _estimate_unquantized_size
# ---------------------------------------------------------------------------


class TestEstimateUnquantizedSize:
    def test_dir_size(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "a.bin").write_bytes(b"x" * (1024 ** 3))  # 1 GB
        assert _estimate_unquantized_size(str(d)) == 1.0

    def test_file_size(self, tmp_path):
        f = tmp_path / "model.safetensors"
        f.write_bytes(b"x" * (512 * 1024 * 1024))  # 0.5 GB
        assert _estimate_unquantized_size(str(f)) == 0.5

    def test_nonexistent_returns_zero(self):
        assert _estimate_unquantized_size("/nonexistent/path") == 0.0


# ---------------------------------------------------------------------------
# GGUF type / bit-width mappings
# ---------------------------------------------------------------------------


class TestGguTypeMappings:
    def test_gptq_to_type(self):
        assert gguf_type_to_bits("Q4_K") == 4
        assert gguf_type_to_bits("Q8_0") == 8
        assert gguf_type_to_bits("F16") == 16

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown GGUF quant type"):
            gguf_type_to_bits("Q99_K")

    def test_bits_to_gguf_type_in_quantizer(self):
        from app.quantization.export_gguf import GGUFQuantizer
        assert GGUFQuantizer._bits_to_gguf_type(4) == "Q4_K"
        assert GGUFQuantizer._bits_to_gguf_type(2) == "Q2_K"
        assert GGUFQuantizer._bits_to_gguf_type(8) == "Q8_0"
        assert GGUFQuantizer._bits_to_gguf_type(99) == "Q4_K"  # default


# ---------------------------------------------------------------------------
# score_quality_size_speed
# ---------------------------------------------------------------------------


class TestScoreQualitySizeSpeed:
    def test_perfect_score(self):
        r = QuantResult(
            quant_method=QuantMethod.GPTQ,
            bit_width=4,
            quantized_model_size_gb=0.0,
            estimated_vram_gb=0.0,
            tokens_per_sec=100.0,
            model_cwe_macro_f1=1.0,
            status=QuantStatus.COMPLETED,
        )
        score = score_quality_size_speed(r)
        assert score == pytest.approx(1.0, abs=0.02)

    def test_zero_quality(self):
        r = QuantResult(
            quant_method=QuantMethod.GPTQ,
            bit_width=4,
            quantized_model_size_gb=6.5,
            estimated_vram_gb=7.5,
            tokens_per_sec=35.0,
            model_cwe_macro_f1=0.0,
            status=QuantStatus.COMPLETED,
        )
        score = score_quality_size_speed(r)
        # quality=0, size_score=1-6.5/14≈0.536, speed_score=35/30=1.0
        expected = 0.6 * 0.0 + 0.2 * 0.5357 + 0.2 * 1.0
        assert score == pytest.approx(expected, abs=0.02)

    def test_failed_result_is_zero(self):
        r = QuantResult(
            quant_method=QuantMethod.GPTQ,
            bit_width=4,
            quantized_model_size_gb=6.5,
            estimated_vram_gb=7.5,
            status=QuantStatus.FAILED,
        )
        assert score_quality_size_speed(r) == 0.0

    def test_none_quality_uses_estimate(self):
        r = QuantResult(
            quant_method=QuantMethod.GPTQ,
            bit_width=4,
            quantized_model_size_gb=6.5,
            estimated_vram_gb=7.5,
            tokens_per_sec=35.0,
            model_cwe_macro_f1=None,  # None → estimate_quality(GPTQ, 4) = 0.92
            status=QuantStatus.COMPLETED,
        )
        score = score_quality_size_speed(r)
        assert score > 0.5  # quality is high at 4-bit


# ---------------------------------------------------------------------------
# select_best_config
# ---------------------------------------------------------------------------


def _make_result(method, bits, vram, size, quality=0.92, tps=35.0, status=QuantStatus.COMPLETED):
    return QuantResult(
        quant_method=method,
        bit_width=bits,
        quantized_model_size_gb=size,
        estimated_vram_gb=vram,
        tokens_per_sec=tps,
        model_cwe_macro_f1=quality,
        status=status,
    )


class TestSelectBestConfig:
    def test_empty_results_returns_none(self):
        assert select_best_config([]) is None

    def test_no_constraints_picks_highest_score(self):
        results = [
            _make_result(QuantMethod.GPTQ, 4, 6.5, 6.5, quality=0.92, tps=35.0),
            _make_result(QuantMethod.GPTQ, 2, 4.0, 4.0, quality=0.60, tps=45.0),
        ]
        best = select_best_config(results)
        assert best is not None
        assert best.bit_width == 4  # 0.92 quality beats smaller size

    def test_vram_filter(self):
        results = [
            _make_result(QuantMethod.GPTQ, 2, 4.0, 4.0),
            _make_result(QuantMethod.GPTQ, 8, 10.0, 10.0),
        ]
        best = select_best_config(results, target_vram_gb=6.0)
        assert best is not None
        assert best.bit_width == 2  # only 4.0 GB fits under 6

    def test_size_filter(self):
        results = [
            _make_result(QuantMethod.GPTQ, 4, 6.5, 6.5),
            _make_result(QuantMethod.GPTQ, 8, 10.0, 10.0),
        ]
        best = select_best_config(results, target_size_gb=7.0)
        assert best is not None
        assert best.bit_width == 4  # 6.5 fits, 10.0 doesn't

    def test_no_results_pass_filter(self):
        results = [
            _make_result(QuantMethod.GPTQ, 8, 10.0, 10.0),
        ]
        best = select_best_config(results, target_vram_gb=4.0)
        assert best is None

    def test_failed_results_excluded(self):
        results = [
            _make_result(QuantMethod.GPTQ, 2, 4.0, 4.0, status=QuantStatus.FAILED),
            _make_result(QuantMethod.GPTQ, 4, 6.5, 6.5, status=QuantStatus.COMPLETED),
        ]
        best = select_best_config(results)
        assert best is not None
        assert best.status == QuantStatus.COMPLETED


# ---------------------------------------------------------------------------
# quantize_single in mock / dry-run / real modes
# ---------------------------------------------------------------------------


class TestQuantizeSingle:
    def test_mock_mode(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_mock",
            mock=True,
        )
        result = quantize_single(QuantMethod.GPTQ, 4, config)
        assert result.status == QuantStatus.COMPLETED
        assert result.quant_method == QuantMethod.GPTQ
        assert result.bit_width == 4
        assert "mock" in result.notes
        assert os.path.join("/tmp/stage8_mock", "gptq_bits4") in result.checkpoint_path

    def test_dry_run_mode(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_dry",
            dry_run=True,
        )
        result = quantize_single(QuantMethod.AWQ, 4, config)
        assert result.status == QuantStatus.COMPLETED
        assert "dry-run" in result.notes
        assert result.tokens_per_sec is not None
        assert result.model_cwe_macro_f1 is not None

    def test_real_mode_import_error_caught(self):
        """When auto_gptq isn't installed, quantize_single should propagate
        the RuntimeError — but _try_quantize (called by the matrix runner)
        catches it."""
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_real",
        )
        # GPTQQuantizer._load() raises RuntimeError if auto_gptq is missing.
        with pytest.raises(RuntimeError, match="auto_gptq is not installed"):
            quantize_single(QuantMethod.GPTQ, 4, config)

    def test_real_mode_awq_import_error(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_awq",
        )
        with pytest.raises(RuntimeError, match="autoawq is not installed"):
            quantize_single(QuantMethod.AWQ, 4, config)

    def test_real_mode_gguf_import_error(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_gguf",
        )
        with pytest.raises(RuntimeError, match="llama_cpp"):
            quantize_single(QuantMethod.GGUF, 4, config)


# ---------------------------------------------------------------------------
# run_quantization_matrix in mock mode
# ---------------------------------------------------------------------------


class TestRunQuantizationMatrixMock:
    def test_mock_matrix_produces_results(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_matrix",
            methods=[QuantMethod.GPTQ, QuantMethod.AWQ, QuantMethod.GGUF],
            bit_widths=[2, 3, 4, 8],
            mock=True,
        )
        report = run_quantization_matrix(config)

        assert report.run_id.startswith("stage8-")
        assert report.base_model == QuantConfig.__dataclass_fields__["base_model"].default
        assert len(report.results) == 4 + 4 + 6  # GPTQ(4) + AWQ(4) + GGUF(6 quant_types)

    def test_mock_matrix_best_result_not_none(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        assert report.best_result is not None
        assert report.best_result.quant_method == QuantMethod.GPTQ

    def test_mock_matrix_json_serializable(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ, QuantMethod.AWQ],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        json_str = report.model_dump_json(indent=2)
        import json
        data = json.loads(json_str)
        assert data["run_id"].startswith("stage8-")
        assert len(data["results"]) == 2
        assert data["best_result"] is not None

    def test_matrix_with_vram_constraint(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ, QuantMethod.GGUF],
            bit_widths=[2, 4, 8],
            target_vram_gb=6.0,
        )
        report = run_quantization_matrix(config)
        assert report.best_result is not None
        assert report.best_result.estimated_vram_gb <= 6.0

    def test_matrix_manifest_contents(self):
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GPTQ],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        manifest = report.manifest
        assert "run_id" in manifest
        assert manifest["mock"] is True
        assert manifest["dry_run"] is False
        assert "gptq" in manifest["methods"]
        assert 4 in manifest["bit_widths"]
        assert "started_at" in manifest
        assert "elapsed_seconds" in manifest


# ---------------------------------------------------------------------------
# run_quantization_matrix in real mode — failure is caught
# ---------------------------------------------------------------------------


class TestRunQuantizationMatrixRealMode:
    def test_real_mode_gptq_fails_to_failed_status(self):
        """Without auto_gptq installed, the matrix should produce a FAILED
        result rather than crashing."""
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_real_fail",
            methods=[QuantMethod.GPTQ],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        assert len(report.results) == 1
        r = report.results[0]
        assert r.status == QuantStatus.FAILED
        assert r.error is not None
        assert "auto_gptq" in r.error

    def test_real_mode_mixed_methods(self):
        """Some methods may fail, but the matrix continues."""
        config = QuantConfig(
            source_checkpoint="dummy",
            output_base="/tmp/stage8_mixed",
            methods=[QuantMethod.GPTQ, QuantMethod.GGUF],
            bit_widths=[4],
        )
        report = run_quantization_matrix(config)
        assert len(report.results) == 1 + 6  # GPTQ(1 bit) + GGUF(6 quant_types)
        # All should be FAILED (no ML deps installed).
        for r in report.results:
            assert r.status == QuantStatus.FAILED
