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
from unittest.mock import MagicMock, patch

import pytest

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
    _estimate_unquantized_size,
    _NoOpQuantizer,
    quantize_single,
    run_quantization_matrix,
    score_quality_size_speed,
    select_best_config,
)
from app.schemas.quantization import (
    QuantMethod,
    QuantResult,
    QuantStatus,
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

    def test_unrecognized_method_returns_8_gb(self):
        """A method that isn't GPTQ/AWQ/GGUF/NONE falls through to the
        final ``return 8.0`` fallback (line 237)."""
        assert estimate_vram_gb("unknown_method", 4) == 8.0


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


class TestMakeQuantizer:
    def test_none_method_returns_noop(self):
        """_make_quantizer(QuantMethod.NONE) returns a _NoOpQuantizer (lines 223-224)."""
        from app.quantization.quantizer import _make_quantizer

        config = QuantConfig(source_checkpoint="dummy")
        q = _make_quantizer(QuantMethod.NONE, config)
        assert isinstance(q, _NoOpQuantizer)

    def test_unknown_method_raises_value_error(self):
        """_make_quantizer with an unrecognized method raises ValueError (line 225)."""
        from app.quantization.quantizer import _make_quantizer

        config = QuantConfig(source_checkpoint="dummy")
        with pytest.raises(ValueError, match="Unknown quantization method"):
            _make_quantizer("bogus_method", config)


class TestEstimateUnquantizedSize:
    def test_dir_size(self, tmp_path):
        d = tmp_path / "ckpt"
        d.mkdir()
        (d / "a.bin").write_bytes(b"x" * (1024**3))  # 1 GB
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

    def test_none_tokens_per_sec_uses_neutral_score(self):
        """When tokens_per_sec is None and status is COMPLETED,
        speed_score defaults to 0.5 (line 322)."""
        r = QuantResult(
            quant_method=QuantMethod.GPTQ,
            bit_width=4,
            quantized_model_size_gb=6.5,
            estimated_vram_gb=7.5,
            tokens_per_sec=None,  # None → speed_score = 0.5
            model_cwe_macro_f1=0.92,
            status=QuantStatus.COMPLETED,
        )
        score = score_quality_size_speed(r)
        # quality=0.92, size_score=1-6.5/14≈0.5357, speed_score=0.5
        expected = 0.6 * 0.92 + 0.2 * 0.5357 + 0.2 * 0.5
        assert score == pytest.approx(expected, abs=0.02)


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

    def test_matrix_gguf_skips_unrecognized_quant_type(self):
        """When a GGUF quant_type is not in the bit-width lookup, the
        ValueError from gguf_type_to_bits is caught and that type is skipped
        (lines 397-398).
        """
        config = QuantConfig(
            source_checkpoint="dummy",
            mock=True,
            methods=[QuantMethod.GGUF],
            bit_widths=[4],
            gguf_config=GGUFConfig(quant_types=["Q4_K", "Q9_K", "Q8_0"]),
        )
        report = run_quantization_matrix(config)
        # "Q9_K" is unrecognized → skipped. Only Q4_K and Q8_0 produce results.
        assert len(report.results) == 2


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


# ---------------------------------------------------------------------------
# AWQQuantizer — mocked success path (lines 51 + 80-107)
# ---------------------------------------------------------------------------


class TestAWQQuantizerMocked:
    def test_load_returns_autoawq_class_when_installed(self):
        """When autoawq IS installed, _load() should return the class."""
        from app.quantization.export_awq import AWQQuantizer

        fake_module = MagicMock()
        fake_module.AutoAWQForCausalLM = MagicMock(name="AutoAWQForCausalLM")

        quantizer = AWQQuantizer()
        quantizer._tokenizer = None  # ensure we go through the import path

        with patch.dict("sys.modules", {"awq": fake_module}):
            result = quantizer._load()
        assert result is fake_module.AutoAWQForCausalLM

    def test_quantize_success(self):
        """Mock autoawq and verify quantize() returns a completed result.

        Covers lines 51 (return from _load) and 80-107 (quantize body).
        """
        from app.quantization.export_awq import AWQQuantizer

        fake_model = MagicMock()
        fake_autoawq = MagicMock()
        fake_autoawq.from_pretrained.return_value = fake_model

        fake_module = MagicMock()
        fake_module.AutoAWQForCausalLM = fake_autoawq

        quantizer = AWQQuantizer(
            config=AWQConfig(bits=4, group_size=128, zero_point=True),
            device="cuda:0",
        )

        with patch.dict("sys.modules", {"awq": fake_module}):
            result = quantizer.quantize("source/path", "output/path", bit_width=4)

        assert result.quant_method == QuantMethod.AWQ
        assert result.bit_width == 4
        assert result.status == QuantStatus.COMPLETED
        assert result.checkpoint_path == "output/path"
        assert "AWQ bits=4" in result.notes
        # The model should have been loaded, quantized, and saved.
        fake_autoawq.from_pretrained.assert_called_once_with(
            "source/path", device_map="auto", torch_dtype="auto"
        )
        fake_model.quantize.assert_called_once()
        fake_model.save_quantized.assert_called_once_with("output/path", safetensors=True)

    def test_quantize_uses_config_bits_when_bit_width_is_none(self):
        """When bit_width is None, quantize() falls back to config.bits."""
        from app.quantization.export_awq import AWQQuantizer

        fake_model = MagicMock()
        fake_autoawq = MagicMock()
        fake_autoawq.from_pretrained.return_value = fake_model

        fake_module = MagicMock()
        fake_module.AutoAWQForCausalLM = fake_autoawq

        quantizer = AWQQuantizer(
            config=AWQConfig(bits=3, group_size=64, zero_point=False),
        )

        with patch.dict("sys.modules", {"awq": fake_module}):
            result = quantizer.quantize("src", "out")

        assert result.bit_width == 3  # from config, not bit_width param
        assert "AWQ bits=3" in result.notes


# ---------------------------------------------------------------------------
# GGUFQuantizer — mocked success paths (lines 80, 86, 92, 126-140, 165, 181-188)
# ---------------------------------------------------------------------------


class TestGGUFQuantizerMocked:
    def test_load_returns_llama_cpp_module_when_installed(self):
        """When llama_cpp IS importable, _load() returns the module (line 80)."""
        from app.quantization.export_gguf import GGUFQuantizer

        fake_module = MagicMock()
        quantizer = GGUFQuantizer()

        with patch.dict("sys.modules", {"llama_cpp": fake_module}):
            result = quantizer._load()
        assert result is fake_module

    def test_load_returns_explicit_path(self):
        """When llama_cpp_path is provided and exists, _load() returns it (line 86)."""
        from app.quantization.export_gguf import GGUFQuantizer

        # Make llama_cpp import fail so we fall through to the CLI path check.
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with patch("os.path.exists", return_value=True):
                quantizer = GGUFQuantizer(llama_cpp_path="/fake/llama-quantize")
                result = quantizer._load()
        assert result == "/fake/llama-quantize"

    def test_load_returns_cli_from_path(self):
        """When llama_cpp_path is not provided but CLI binary is on PATH (line 92)."""
        from app.quantization.export_gguf import GGUFQuantizer

        quantizer = GGUFQuantizer(llama_cpp_path=None)

        # Make llama_cpp import fail (so we fall through to CLI check),
        # and make shutil.which find "llama-quantize".
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with patch("shutil.which", return_value="/usr/local/bin/llama-quantize"):
                result = quantizer._load()
        assert result == "/usr/local/bin/llama-quantize"

    def test_load_raises_when_nothing_available(self):
        """When neither llama_cpp nor CLI is available, _load() raises RuntimeError."""
        from app.quantization.export_gguf import GGUFQuantizer

        quantizer = GGUFQuantizer(llama_cpp_path="/nonexistent/path")

        with patch.dict("sys.modules", {"llama_cpp": None}):
            with patch("os.path.exists", return_value=False):
                with patch("shutil.which", return_value=None):
                    with pytest.raises(RuntimeError, match="Neither .llama_cpp."):
                        quantizer._load()

    def test_quantize_via_cli(self):
        """quantize() with a string backend calls _quantize_via_cli (lines 126-140, 165)."""
        from app.quantization.export_gguf import GGUFQuantizer

        quantizer = GGUFQuantizer()

        # Make _load return a string path (simulating CLI mode).
        fake_subprocess = patch("app.quantization.export_gguf.subprocess.run")
        with patch.object(quantizer, "_load", return_value="/fake/llama-quantize"):
            with fake_subprocess as mock_run:
                result = quantizer.quantize("source", "output/model.gguf", bit_width=4)

        assert result.quant_method == QuantMethod.GGUF
        assert result.bit_width == 4
        assert result.status == QuantStatus.COMPLETED
        assert result.checkpoint_path == "output/model.gguf"
        assert "GGUF type=Q4_K" in result.notes
        mock_run.assert_called_once_with(
            ["/fake/llama-quantize", "source", "output/model.gguf", "Q4_K"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def test_quantize_via_python(self):
        """quantize() with a module backend calls _quantize_via_python (lines 181-188)."""
        from app.quantization.export_gguf import GGUFQuantizer

        fake_llama_cpp = MagicMock()
        fake_ggml = MagicMock()
        fake_llama_cpp.ggml = fake_ggml
        fake_quantizer = MagicMock()
        fake_ggml.LlamaQuantize.return_value = fake_quantizer

        quantizer = GGUFQuantizer()
        with patch.object(quantizer, "_load", return_value=fake_llama_cpp):
            result = quantizer.quantize("source", "output/model.gguf", bit_width=4)

        assert result.quant_method == QuantMethod.GGUF
        assert result.status == QuantStatus.COMPLETED
        # f16_fallback is False by default, so the else branch runs.
        fake_ggml.LlamaQuantize.assert_called_once_with("Q4_K")
        fake_llama_cpp.llama_model_quantize.assert_called_once_with(
            "source", "output/model.gguf", fake_quantizer
        )

    def test_quantize_via_python_f16_fallback(self):
        """When f16_fallback is True and quant_type is F16, uses convert_hf_to_gguf."""
        from app.quantization.export_gguf import GGUFQuantizer

        fake_llama_cpp = MagicMock()

        quantizer = GGUFQuantizer(config=GGUFConfig(f16_fallback=True))
        with patch.object(quantizer, "_load", return_value=fake_llama_cpp):
            result = quantizer.quantize("source", "output/model.gguf", bit_width=16)

        # bit_width=16 maps to "F16", and f16_fallback=True, so convert_hf_to_gguf is called.
        fake_llama_cpp.convert_hf_to_gguf.assert_called_once_with(
            "source", "output/model.gguf", dtype="f16"
        )
        assert result.bit_width == 16


# ---------------------------------------------------------------------------
# GPTQQuantizer — mocked success path (lines 51, 80-102)
# ---------------------------------------------------------------------------


class TestGPTQQuantizerMocked:
    def test_load_returns_autogptq_class_when_installed(self):
        """When auto_gptq IS installed, _load() returns the class (line 51)."""
        from app.quantization.export_gptq import GPTQQuantizer

        fake_module = MagicMock()
        fake_module.AutoGPTQForCausalLM = MagicMock(name="AutoGPTQForCausalLM")
        quantizer = GPTQQuantizer()

        with patch.dict("sys.modules", {"auto_gptq": fake_module}):
            result = quantizer._load()
        assert result is fake_module.AutoGPTQForCausalLM

    def test_quantize_success(self):
        """Mock auto_gptq and verify quantize() returns a completed result.

        Covers lines 51 (return from _load) and 80-102 (quantize body).
        """
        from app.quantization.export_gptq import GPTQQuantizer

        fake_module = MagicMock()
        quantizer = GPTQQuantizer(
            config=GPTQConfig(bits=4, group_size=128, desc_act=2, damping=0.06),
            device="cuda:0",
        )

        with patch.dict("sys.modules", {"auto_gptq": fake_module}):
            result = quantizer.quantize("source/path", "output/path", bit_width=4)

        assert result.quant_method == QuantMethod.GPTQ
        assert result.bit_width == 4
        assert result.status == QuantStatus.COMPLETED
        assert result.checkpoint_path == "output/path"
        assert "GPTQ bits=4" in result.notes
        # AutoGPTQForCausalLM.quantize should have been called with params.
        fake_module.AutoGPTQForCausalLM.quantize.assert_called_once()

    def test_quantize_uses_config_bits_when_bit_width_is_none(self):
        """When bit_width is None, quantize() falls back to config.bits."""
        from app.quantization.export_gptq import GPTQQuantizer

        fake_module = MagicMock()
        quantizer = GPTQQuantizer(
            config=GPTQConfig(bits=3, group_size=64, desc_act=0, damping=0.03),
        )

        with patch.dict("sys.modules", {"auto_gptq": fake_module}):
            result = quantizer.quantize("src", "out")

        assert result.bit_width == 3  # from config
        assert "GPTQ bits=3" in result.notes
