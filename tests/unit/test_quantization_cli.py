"""Unit tests for Stage 8 — quantization CLI (``app/quantization/cli.py``).

Covers every branch of the ``run`` and ``inspect`` commands plus the
helper functions ``_parse_method``, ``_bits_to_gguf``, ``_is_hf_id``,
and ``_dry_run_quantize``.

All tests use ``--mock`` or ``--dry-run`` or patch the quantizer backends,
so no GPU or external tools are required.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

typer = pytest.importorskip("typer")
from app.quantization.cli import (  # noqa: E402
    _bits_to_gguf,
    _dry_run_quantize,
    _is_hf_id,
    _parse_method,
    inspect,
    run,
)
from app.schemas.quantization import QuantMethod, QuantStatus  # noqa: E402

# ---------------------------------------------------------------------------
# _parse_method
# ---------------------------------------------------------------------------


class TestParseMethod:
    def test_gguf(self):
        assert _parse_method("gguf") == QuantMethod.GGUF

    def test_gptq(self):
        assert _parse_method("gptq") == QuantMethod.GPTQ

    def test_awq(self):
        assert _parse_method("awq") == QuantMethod.AWQ

    def test_uppercase(self):
        assert _parse_method("GGUF") == QuantMethod.GGUF

    def test_none_method(self):
        assert _parse_method("none") == QuantMethod.NONE

    def test_invalid(self):
        with pytest.raises(typer.Exit) as exc_info:
            _parse_method("invalid")
        assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# _bits_to_gguf
# ---------------------------------------------------------------------------


class TestBitsToGguf:
    def test_2_bit(self):
        assert _bits_to_gguf(2) == "Q2_K"

    def test_3_bit(self):
        assert _bits_to_gguf(3) == "Q3_K"

    def test_4_bit(self):
        assert _bits_to_gguf(4) == "Q4_K"

    def test_8_bit(self):
        assert _bits_to_gguf(8) == "Q8_0"

    def test_16_bit(self):
        assert _bits_to_gguf(16) == "F16"

    def test_32_bit(self):
        assert _bits_to_gguf(32) == "F32"

    def test_unknown_bits_raises_exit(self):
        """Bits not in the mapping raises typer.Exit(1)."""
        with pytest.raises(typer.Exit) as exc_info:
            _bits_to_gguf(99)
        assert exc_info.value.exit_code == 1

    def test_invalid_bits_raises_exit(self):
        with pytest.raises(typer.Exit) as exc_info:
            _bits_to_gguf(1)
        assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# _is_hf_id
# ---------------------------------------------------------------------------


class TestIsHfId:
    def test_hf_id_with_slash(self):
        assert _is_hf_id("Qwen/Qwen2.5-Coder-7B") is True

    def test_absolute_path(self):
        assert _is_hf_id("/abs/path/model") is False

    def test_local_path_without_slash(self, tmp_path):
        local = str(tmp_path / "model")
        assert _is_hf_id(local) is False

    def test_empty_string(self):
        assert _is_hf_id("") is False

    def test_local_dir_not_absolute(self, tmp_path):
        local = str(tmp_path / "model")
        assert _is_hf_id(local) is False


# ---------------------------------------------------------------------------
# _dry_run_quantize
# ---------------------------------------------------------------------------


class TestDryRunQuantize:
    def test_returns_valid_result(self):
        result = _dry_run_quantize(QuantMethod.GGUF, 4, "/source", "/output/model.gguf")
        assert result.quant_method == QuantMethod.GGUF
        assert result.bit_width == 4
        assert result.status == QuantStatus.COMPLETED
        assert result.checkpoint_path == "/output/model.gguf"
        assert result.estimated_vram_gb > 0
        assert result.quantized_model_size_gb > 0
        assert "dry-run" in result.notes

    def test_gptq_method(self):
        result = _dry_run_quantize(QuantMethod.GPTQ, 4, "/source", "/output/model.gguf")
        assert result.quant_method == QuantMethod.GPTQ
        assert result.bit_width == 4
        assert "dry-run" in result.notes

    def test_awq_method(self):
        result = _dry_run_quantize(QuantMethod.AWQ, 2, "/source", "/output/model.gguf")
        assert result.quant_method == QuantMethod.AWQ
        assert result.bit_width == 2

    def test_none_method(self):
        result = _dry_run_quantize(QuantMethod.NONE, 4, "/source", "/output/model.gguf")
        assert result.quant_method == QuantMethod.NONE
        assert result.bit_width == 4


# ---------------------------------------------------------------------------
# run command — error paths
# ---------------------------------------------------------------------------


DEFAULTS = dict(
    method="gguf",
    bits=4,
    output_dir="",  # overridden per-test
    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    dry_run=False,
    mock=False,
    verbose=False,
)


def _run_kwargs(**overrides):
    """Return valid defaults for every parameter of ``run``."""
    kwargs = dict(DEFAULTS)
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------------------
# run command — error paths
# ---------------------------------------------------------------------------


class TestRunErrors:
    def test_no_checkpoint(self, capsys, tmp_path):
        """Empty checkpoint → error + Exit(1)."""
        with pytest.raises(typer.Exit) as exc_info:
            run(
                **_run_kwargs(
                    checkpoint="",
                    output_dir=str(tmp_path / "out"),
                )
            )
        assert exc_info.value.exit_code == 1
        err = capsys.readouterr().err
        assert "checkpoint" in err.lower()

    def test_checkpoint_not_found(self, tmp_path):
        """Non-existent checkpoint (not an HF ID) → error + Exit(1)."""
        fake_path = str(tmp_path / "nonexistent_checkpoint")
        with pytest.raises(typer.Exit) as exc_info:
            run(
                **_run_kwargs(
                    checkpoint=fake_path,
                    output_dir=str(tmp_path / "out"),
                )
            )
        assert exc_info.value.exit_code == 1

    def test_invalid_method(self, tmp_path):
        """Invalid method string → error + Exit(1)."""
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        with pytest.raises(typer.Exit) as exc_info:
            run(
                **_run_kwargs(
                    checkpoint=ckpt,
                    output_dir=str(tmp_path / "out"),
                    method="invalid",
                )
            )
        assert exc_info.value.exit_code == 1

    def test_invalid_bits_for_gguf_real_mode(self, tmp_path):
        """Invalid bits for GGUF in real mode → _bits_to_gguf raises Exit(1)."""
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        with pytest.raises(typer.Exit) as exc_info:
            run(
                **_run_kwargs(
                    checkpoint=ckpt,
                    output_dir=str(tmp_path / "out"),
                    method="gguf",
                    bits=1,
                )
            )
        assert exc_info.value.exit_code == 1

    def test_unsupported_method_real_mode(self, tmp_path):
        """QuantMethod.NONE in real mode → unsupported method error + Exit(1)."""
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        with pytest.raises(typer.Exit) as exc_info:
            run(
                **_run_kwargs(
                    checkpoint=ckpt,
                    output_dir=str(tmp_path / "out"),
                    method="none",
                )
            )
        assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# run command — dry-run mode
# ---------------------------------------------------------------------------


class TestRunDryRun:
    def test_dry_run_writes_report(self, tmp_path):
        """Dry-run mode writes a quant_report.json and prints summary."""
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, dry_run=True))

        report_path = os.path.join(output_dir, "quant_report.json")
        assert os.path.exists(report_path)
        with open(report_path, encoding="utf-8") as f:
            report = json.loads(f.read())
        assert report["status"] == "completed"
        assert report["method"] == "gguf"
        assert report["bit_width"] == 4
        assert "dry-run" in report["notes"]

    def test_dry_run_prints_summary(self, tmp_path, capsys):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, dry_run=True))

        out = capsys.readouterr().out
        assert "Quantization complete" in out
        assert "Output:" in out
        assert "Report:" in out
        assert "Size:" in out

    def test_dry_run_gptq(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")
        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, dry_run=True, method="gptq"))
        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))

    def test_dry_run_awq(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")
        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, dry_run=True, method="awq"))
        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))

    def test_dry_run_with_error_in_result(self, tmp_path, capsys):
        """When result has an error field set, the error is printed to stderr."""
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        from app.schemas.quantization import QuantResult

        fake_result = QuantResult(
            quant_method=QuantMethod.GGUF,
            bit_width=4,
            quantized_model_size_gb=1.0,
            estimated_vram_gb=2.0,
            measured_vram_gb=None,
            tokens_per_sec=None,
            model_cwe_macro_f1=None,
            exec_pass_rate=None,
            status=QuantStatus.COMPLETED,
            checkpoint_path=str(tmp_path / "out" / "gguf_bits4.gguf"),
            notes="dry-run gguf @ 4-bit",
            error="something went wrong",
        )
        with patch("app.quantization.cli._dry_run_quantize", return_value=fake_result):
            run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, dry_run=True))

        err = capsys.readouterr().err
        assert "something went wrong" in err


# ---------------------------------------------------------------------------
# run command — mock mode
# ---------------------------------------------------------------------------


class TestRunMock:
    def test_mock_mode_writes_report(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, mock=True))

        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))
        with open(os.path.join(output_dir, "quant_report.json"), encoding="utf-8") as f:
            report = json.loads(f.read())
        assert report["status"] == "completed"
        assert report["method"] == "gguf"

    def test_mock_mode_gptq(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")
        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, mock=True, method="gptq"))
        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))

    def test_mock_mode_awq(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")
        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, mock=True, method="awq"))
        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))


# ---------------------------------------------------------------------------
# run command — real mode (with patched quantizer backends)
# ---------------------------------------------------------------------------


class TestRunRealMode:
    def test_real_mode_gguf_success(self, tmp_path):
        """Real mode with GGUF — quantizer is patched to avoid real LLM."""
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        from app.schemas.quantization import QuantResult

        fake_result = QuantResult(
            quant_method=QuantMethod.GGUF,
            bit_width=4,
            quantized_model_size_gb=5.0,
            estimated_vram_gb=6.5,
            measured_vram_gb=None,
            tokens_per_sec=None,
            model_cwe_macro_f1=None,
            exec_pass_rate=None,
            status=QuantStatus.COMPLETED,
            checkpoint_path=str(tmp_path / "out" / "gguf_bits4.gguf"),
            notes="GGUF type=Q4_K bits=4 quantized in 1.2s",
        )

        with patch("app.quantization.export_gguf.GGUFQuantizer") as MockQuant:
            MockQuant.return_value.quantize.return_value = fake_result
            run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, method="gguf"))

        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))

    def test_real_mode_gptq_success(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        from app.schemas.quantization import QuantResult

        fake_result = QuantResult(
            quant_method=QuantMethod.GPTQ,
            bit_width=4,
            quantized_model_size_gb=5.0,
            estimated_vram_gb=7.5,
            status=QuantStatus.COMPLETED,
            checkpoint_path=str(tmp_path / "out" / "gptq_bits4.gguf"),
            notes="GPTQ bits=4 group_size=128",
        )

        with patch("app.quantization.export_gptq.GPTQQuantizer") as MockQuant:
            MockQuant.return_value.quantize.return_value = fake_result
            run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, method="gptq"))

        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))

    def test_real_mode_awq_success(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        from app.schemas.quantization import QuantResult

        fake_result = QuantResult(
            quant_method=QuantMethod.AWQ,
            bit_width=4,
            quantized_model_size_gb=5.0,
            estimated_vram_gb=7.5,
            status=QuantStatus.COMPLETED,
            checkpoint_path=str(tmp_path / "out" / "awq_bits4.gguf"),
            notes="AWQ bits=4",
        )

        with patch("app.quantization.export_awq.AWQQuantizer") as MockQuant:
            MockQuant.return_value.quantize.return_value = fake_result
            run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, method="awq"))

        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))

    def test_real_mode_hf_id_checkpoint(self, tmp_path):
        """HF ID checkpoint bypasses os.path.exists check."""
        output_dir = str(tmp_path / "output")
        from app.schemas.quantization import QuantResult

        fake_result = QuantResult(
            quant_method=QuantMethod.GGUF,
            bit_width=4,
            quantized_model_size_gb=5.0,
            estimated_vram_gb=7.0,
            status=QuantStatus.COMPLETED,
            checkpoint_path=str(tmp_path / "out" / "gguf_bits4.gguf"),
            notes="GGUF type=Q4_K bits=4 quantized in 1.2s",
        )

        with patch("app.quantization.export_gguf.GGUFQuantizer") as MockQuant:
            MockQuant.return_value.quantize.return_value = fake_result
            run(
                **_run_kwargs(
                    checkpoint="Qwen/Qwen2.5-Coder-7B-Instruct",
                    output_dir=output_dir,
                    method="gguf",
                )
            )

        assert os.path.exists(os.path.join(output_dir, "quant_report.json"))


# ---------------------------------------------------------------------------
# run command — verbose mode
# ---------------------------------------------------------------------------


class TestRunVerbose:
    def test_verbose_sets_debug_level(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        from app.schemas.quantization import QuantResult

        fake_result = QuantResult(
            quant_method=QuantMethod.GGUF,
            bit_width=4,
            quantized_model_size_gb=5.0,
            estimated_vram_gb=7.0,
            status=QuantStatus.COMPLETED,
            checkpoint_path=str(tmp_path / "out"),
            notes="GGUF type=Q4_K bits=4",
        )

        with patch("app.quantization.MockQuantizer") as MockQuant:
            MockQuant.return_value.quantize.return_value = fake_result
            run(
                **_run_kwargs(
                    checkpoint=ckpt,
                    output_dir=output_dir,
                    mock=True,
                    verbose=True,
                )
            )


# ---------------------------------------------------------------------------
# run command — report output
# ---------------------------------------------------------------------------


class TestRunReport:
    def test_report_contains_all_fields(self, tmp_path):
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt)
        output_dir = str(tmp_path / "output")

        run(**_run_kwargs(checkpoint=ckpt, output_dir=output_dir, dry_run=True))

        report_path = os.path.join(output_dir, "quant_report.json")
        with open(report_path, encoding="utf-8") as f:
            report = json.loads(f.read())

        # Verify all expected fields are present
        assert report["run_id"].startswith("stage8-")
        assert "method" in report
        assert "bit_width" in report
        assert "status" in report
        assert "checkpoint_path" in report
        assert "quantized_model_size_gb" in report
        assert "estimated_vram_gb" in report
        assert "measured_vram_gb" in report
        assert "tokens_per_sec" in report
        assert "model_cwe_macro_f1" in report
        assert "exec_pass_rate" in report
        assert "error" in report
        assert "notes" in report
        assert "timestamp" in report


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


class TestInspect:
    def test_inspect_prints_metadata(self, capsys):
        """inspect() reads GGUF metadata and prints architecture, tensor count, fields."""
        fake_reader = MagicMock()
        fake_reader.architecture = "qwen2"
        fake_reader.tensors = [MagicMock(), MagicMock(), MagicMock()]
        fake_reader.fields = {"general.name": "test", "general.architecture": "qwen2"}

        with patch("gguf.GGUFReader", return_value=fake_reader):
            inspect("/fake/model.gguf")

        out = capsys.readouterr().out
        assert "Architecture: qwen2" in out
        assert "Tensors: 3" in out
        assert "general.name" in out
        assert "test" in out

    def test_inspect_with_empty_fields(self, capsys):
        """inspect() handles a reader with no fields."""
        fake_reader = MagicMock()
        fake_reader.architecture = "qwen2"
        fake_reader.tensors = []
        fake_reader.fields = {}

        with patch("gguf.GGUFReader", return_value=fake_reader):
            inspect("/fake/model.gguf")

        out = capsys.readouterr().out
        assert "Architecture: qwen2" in out
        assert "Tensors: 0" in out
