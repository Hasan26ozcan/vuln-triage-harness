"""Unit tests for Stage 9 — CLI entry point (app/serving/cli.py).

Covers every branch of the ``serve`` function:

* Dry-run mode — config + warnings output (with and without warnings).
* Analyze/batch modes — input file parsing, single-dict wrapping,
  output-file writing, missing-file / missing-arg errors.
* Server mode — uvicorn invocation (mocked).

All tests use ``backend_type="mock"`` so no real ML or network is needed.
``uvicorn.run`` is patched to avoid binding a port.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

typer = pytest.importorskip("typer")

from app.serving.cli import serve  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def serve_request_dict():
    return {
        "vulnerable_code": "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
        "language": "python",
        "description": "SQL injection via string concatenation",
    }


def _write_input(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _all_kwargs(**overrides):
    """Return valid defaults for every parameter of ``serve``.

    ``typer.Option`` defaults are ``OptionInfo`` objects, so we must supply
    real Python values when calling the function directly.
    """
    defaults = dict(
        model_path="",
        backend_type="llama.cpp",
        num_ctx=4096,
        num_threads=4,
        n_gpu_layers=0,
        temperature=0.2,
        max_new_tokens=2048,
        request_timeout=30.0,
        host="0.0.0.0",
        port=8000,
        analyze=False,
        batch=False,
        input_file="",
        output_file="",
        dry_run=False,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Dry-run mode (lines 91-119)
# ---------------------------------------------------------------------------


class TestDryRun:
    def test_dry_run_with_warnings(self, capsys):
        """Dry-run with llama.cpp + empty model_path produces warnings."""
        kwargs = _all_kwargs(dry_run=True, backend_type="llama.cpp", model_path="")
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 0

        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "llama.cpp" in out
        assert "[WARN]" in out

    def test_dry_run_no_warnings(self, capsys):
        """Dry-run with mock backend produces no warnings."""
        kwargs = _all_kwargs(dry_run=True, backend_type="mock", model_path="")
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 0

        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "Warnings:     (none)" in out


# ---------------------------------------------------------------------------
# Analyze / batch modes — error paths (lines 123-125, 130-132)
# ---------------------------------------------------------------------------


class TestAnalyzeBatchErrors:
    def test_analyze_without_input_file(self):
        """--analyze without --input-file → error (lines 123-125)."""
        kwargs = _all_kwargs(analyze=True, backend_type="mock")
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 1

    def test_batch_without_input_file(self):
        """--batch without --input-file → error (lines 123-125)."""
        kwargs = _all_kwargs(batch=True, backend_type="mock")
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 1

    def test_analyze_file_not_found(self, tmp_path):
        """Input file doesn't exist → error (lines 130-132)."""
        kwargs = _all_kwargs(
            analyze=True,
            input_file=str(tmp_path / "nonexistent.json"),
            backend_type="mock",
        )
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# Analyze mode — happy path (lines 127-159)
# ---------------------------------------------------------------------------


class TestAnalyzeMode:
    def test_analyze_single_request(self, tmp_path, capsys, serve_request_dict):
        """Analyze mode reads a single JSON dict, serves it, prints response."""
        input_file = tmp_path / "request.json"
        _write_input(input_file, serve_request_dict)

        kwargs = _all_kwargs(
            analyze=True,
            input_file=str(input_file),
            backend_type="mock",
        )
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 0

        out = capsys.readouterr().out
        assert "sample_id" in out
        assert "predicted_cwe" in out

    def test_analyze_with_output_file(self, tmp_path, serve_request_dict):
        """Analyze mode writes results to --output-file (lines 155-157)."""
        input_file = tmp_path / "request.json"
        _write_input(input_file, serve_request_dict)
        output_file = tmp_path / "result.json"

        kwargs = _all_kwargs(
            analyze=True,
            input_file=str(input_file),
            output_file=str(output_file),
            backend_type="mock",
        )
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 0

        assert output_file.exists()
        saved = json.loads(output_file.read_text(encoding="utf-8"))
        assert saved["predicted_cwe"] == "CWE-89"


# ---------------------------------------------------------------------------
# Batch mode — happy path (lines 136-159)
# ---------------------------------------------------------------------------


class TestBatchMode:
    def test_batch_array(self, tmp_path, capsys, serve_request_dict):
        """Batch mode reads a JSON array of requests (lines 136-148)."""
        input_file = tmp_path / "batch.json"
        _write_input(input_file, [serve_request_dict, serve_request_dict])

        kwargs = _all_kwargs(
            batch=True,
            input_file=str(input_file),
            backend_type="mock",
        )
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 0

        out = capsys.readouterr().out
        assert "responses" in out
        assert "manifest" in out

    def test_batch_wraps_single_dict(self, tmp_path, serve_request_dict):
        """When batch input is a single JSON dict (not a list), it is wrapped
        in a list (lines 138-140)."""
        input_file = tmp_path / "single.json"
        _write_input(input_file, serve_request_dict)

        kwargs = _all_kwargs(
            batch=True,
            input_file=str(input_file),
            backend_type="mock",
        )
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 0

    def test_batch_with_output_file(self, tmp_path, serve_request_dict):
        """Batch mode writes results to --output-file (lines 146-148)."""
        input_file = tmp_path / "batch.json"
        _write_input(input_file, [serve_request_dict])
        output_file = tmp_path / "batch_result.json"

        kwargs = _all_kwargs(
            batch=True,
            input_file=str(input_file),
            output_file=str(output_file),
            backend_type="mock",
        )
        with pytest.raises(typer.Exit) as exc_info:
            serve(**kwargs)
        assert exc_info.value.exit_code == 0

        assert output_file.exists()
        saved = json.loads(output_file.read_text(encoding="utf-8"))
        assert len(saved["responses"]) == 1


# ---------------------------------------------------------------------------
# Server mode (lines 161-178)
# ---------------------------------------------------------------------------


class TestServerMode:
    def test_server_mode_starts_uvicorn(self, capsys):
        """Server mode imports create_app and runs uvicorn (lines 161-178).

        ``uvicorn.run`` is patched so no real server is started.
        """
        with patch("uvicorn.run"):
            serve(**_all_kwargs(backend_type="mock", model_path=""))

    def test_server_mode_prints_warnings(self, capsys):
        """Server mode with llama.cpp + empty model_path prints warnings
        (lines 162-166)."""
        with patch("uvicorn.run"):
            serve(**_all_kwargs(backend_type="llama.cpp", model_path=""))

        err = capsys.readouterr().err
        assert "[WARN]" in err

    def test_server_mode_prints_starting_message(self, capsys):
        """Server mode prints the starting message (lines 168-169)."""
        with patch("uvicorn.run"):
            serve(**_all_kwargs(backend_type="mock", model_path=""))

        out = capsys.readouterr().out
        assert "Starting Stage 9 server" in out
        assert "mock" in out
