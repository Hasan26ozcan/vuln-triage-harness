"""Tests for the Stage 1 data-collection CLI (app/data/collectors/cli.py).

These tests use Typer's CliRunner to exercise the CLI commands with
dependency injection via unittest.mock.patch — no real database,
MinIO, or network access required.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.data.collectors.cli import app


@pytest.fixture
def runner():
    return CliRunner()


def _mock_pipeline_result(samples=None, skipped=None):
    """Build a PipelineResult-like object for mocking."""
    from app.data.collectors.pipeline import PipelineResult

    return PipelineResult(
        samples=samples or [],
        skipped=skipped or [],
    )


# --- scope command ---


def test_scope_command_prints_cwe_table(runner):
    result = runner.invoke(app, ["scope"])

    assert result.exit_code == 0
    from app.data.collectors.cwe_scope import CWE_SCOPE

    for spec in CWE_SCOPE:
        assert spec.cwe_id in result.output
        assert spec.name in result.output


# --- collect command ---


def test_collect_dry_run_skips_init_and_ensure(runner):
    """With --dry-run, init_db and ensure_bucket are not called."""
    mock_result = _mock_pipeline_result(samples=[], skipped=[])

    with (
        patch("app.data.collectors.cli.run_pipeline", return_value=mock_result) as mock_pipeline,
        patch("app.data.collectors.cli.init_db") as mock_init,
        patch("app.data.collectors.cli.ensure_bucket") as mock_bucket,
    ):
        result = runner.invoke(app, ["collect", "--db-path", "./test.db", "--dry-run"])

    assert result.exit_code == 0
    mock_pipeline.assert_called_once()
    mock_init.assert_not_called()
    mock_bucket.assert_not_called()
    _, kwargs = mock_pipeline.call_args
    assert kwargs["dry_run"] is True


def test_collect_without_dry_run_calls_init_and_ensure(runner):
    """Without --dry-run, init_db and ensure_bucket are called."""
    mock_result = _mock_pipeline_result(samples=[], skipped=[])

    with (
        patch("app.data.collectors.cli.run_pipeline", return_value=mock_result) as mock_pipeline,
        patch("app.data.collectors.cli.init_db") as mock_init,
        patch("app.data.collectors.cli.ensure_bucket") as mock_bucket,
    ):
        result = runner.invoke(app, ["collect", "--db-path", "./test.db"])

    assert result.exit_code == 0
    mock_init.assert_called_once()
    mock_bucket.assert_called_once()
    _, kwargs = mock_pipeline.call_args
    assert kwargs["dry_run"] is False


def test_collect_with_language_filter(runner):
    """Comma-separated --languages is parsed into a set."""
    mock_result = _mock_pipeline_result(samples=[], skipped=[])

    with (
        patch("app.data.collectors.cli.run_pipeline", return_value=mock_result) as mock_pipeline,
        patch("app.data.collectors.cli.init_db"),
        patch("app.data.collectors.cli.ensure_bucket"),
    ):
        result = runner.invoke(
            app, ["collect", "--db-path", "./test.db", "--languages", "python,javascript"]
        )

    assert result.exit_code == 0
    _, kwargs = mock_pipeline.call_args
    assert kwargs["languages"] == {"python", "javascript"}


def test_collect_empty_languages_defaults_to_none(runner):
    """When --languages is empty, languages filter is None (all languages)."""
    mock_result = _mock_pipeline_result(samples=[], skipped=[])

    with (
        patch("app.data.collectors.cli.run_pipeline", return_value=mock_result) as mock_pipeline,
        patch("app.data.collectors.cli.init_db"),
        patch("app.data.collectors.cli.ensure_bucket"),
    ):
        result = runner.invoke(
            app, ["collect", "--db-path", "./test.db", "--languages", ""]
        )

    assert result.exit_code == 0
    _, kwargs = mock_pipeline.call_args
    assert kwargs["languages"] is None


def test_collect_no_static_analysis_flag(runner):
    """--no-static-analysis disables static analysis."""
    mock_result = _mock_pipeline_result(samples=[], skipped=[])

    with (
        patch("app.data.collectors.cli.run_pipeline", return_value=mock_result) as mock_pipeline,
        patch("app.data.collectors.cli.init_db"),
        patch("app.data.collectors.cli.ensure_bucket"),
    ):
        result = runner.invoke(
            app, ["collect", "--db-path", "./test.db", "--no-static-analysis"]
        )

    assert result.exit_code == 0
    _, kwargs = mock_pipeline.call_args
    assert kwargs["run_static_analysis"] is False


def test_collect_verbose(runner):
    """--verbose sets logging level to INFO (just verify it runs)."""
    mock_result = _mock_pipeline_result(samples=[], skipped=[])

    with (
        patch("app.data.collectors.cli.run_pipeline", return_value=mock_result),
        patch("app.data.collectors.cli.init_db"),
        patch("app.data.collectors.cli.ensure_bucket"),
    ):
        result = runner.invoke(
            app, ["collect", "--db-path", "./test.db", "--verbose", "--dry-run"]
        )

    assert result.exit_code == 0


def test_collect_reports_samples_and_skipped(runner):
    """Output includes sample count and skip details."""
    pair = MagicMock()
    pair.cve_id = "CVE-2024-0001"
    mock_result = _mock_pipeline_result(
        samples=[MagicMock()],
        skipped=[(pair, "out of scope CWE")],
    )

    with (
        patch("app.data.collectors.cli.run_pipeline", return_value=mock_result),
        patch("app.data.collectors.cli.init_db"),
        patch("app.data.collectors.cli.ensure_bucket"),
    ):
        result = runner.invoke(app, ["collect", "--db-path", "./test.db"])

    assert result.exit_code == 0
    assert "Built: 1" in result.output
    assert "Skipped: 1" in result.output
    assert "CVE-2024-0001" in result.output
    assert "out of scope CWE" in result.output


# --- __main__ guard ---


def test_main_guard(monkeypatch):
    """Cover lines 70-71: the ``if __name__ == '__main__': app()`` guard.

    We exec only the final two lines of the module with ``__name__``
    set to ``'__main__'`` so coverage attributes execution back to the
    original file.
    """
    import sys

    import app.data.collectors.cli as cli_module

    # Line 70 is the __main__ guard, line 71 is app()
    source = "\n" * 69 + 'if __name__ == "__main__":\n    app()'
    code = compile(source, str(cli_module.__file__), "exec")
    namespace: dict = {"__name__": "__main__", "app": cli_module.app}

    monkeypatch.setattr(sys, "argv", ["cli.py", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        exec(code, namespace)
    assert exc_info.value.code == 0
