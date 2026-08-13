"""Tests for the Semgrep runner (app/data/collectors/semgrep_runner.py).

Since ``semgrep`` may not be installed in the test environment, all subprocess
calls are mocked. We verify the full code path: unsupported language, missing
config, unavailable semgrep, clean run, findings, and error exit codes.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.data.collectors.semgrep_runner import (
    SemgrepUnavailableError,
    run_semgrep,
)

# --- Unsupported language ---


def test_run_semgrep_raises_on_unsupported_language():
    """An unknown language gets ValueError, not a Semgrep call."""
    with pytest.raises(ValueError, match="Unsupported language"):
        run_semgrep("code", "rust")


def test_run_semgrep_raises_on_no_bundled_config():
    """A language with no extension and no bundled config raises ValueError."""
    # 'c' and 'cpp' have extensions but no bundled rules
    with pytest.raises(ValueError, match="No bundled Semgrep rule pack"):
        run_semgrep("int main() {}", "c")


# --- Semgrep unavailable ---


def test_run_semgrep_raises_when_binary_missing():
    """When shutil.which('semgrep') returns None, raise SemgrepUnavailableError."""
    with patch("app.data.collectors.semgrep_runner.shutil.which", return_value=None):
        with pytest.raises(SemgrepUnavailableError, match="not installed"):
            run_semgrep("code", "python")


# --- Successful runs ---


def _mock_semgrep_ok(stdout_json: dict | None = None):
    """Build a mock subprocess result for a clean semgrep run (returncode 0/1)."""
    mock_result = MagicMock()
    mock_result.returncode = 1  # 1 = findings present
    mock_result.stdout = json.dumps(stdout_json or {"results": []})
    mock_result.stderr = ""
    return mock_result


def test_run_semgrep_returns_findings():
    """When semgrep returns findings, they are parsed into StaticFinding objects."""
    raw_results = [
        {
            "check_id": "python.sql-injection",
            "extra": {"message": "Possible SQL injection"},
            "start": {"line": 5},
            "end": {"line": 7},
        },
    ]

    mock_result = _mock_semgrep_ok({"results": raw_results})

    with (
        patch("app.data.collectors.semgrep_runner.shutil.which", return_value="/usr/bin/semgrep"),
        patch("app.data.collectors.semgrep_runner.subprocess.run", return_value=mock_result),
    ):
        findings = run_semgrep("cursor.execute('SELECT * FROM t')", "python")

    assert len(findings) == 1
    assert findings[0].rule_id == "python.sql-injection"
    assert findings[0].tool == "semgrep"
    assert findings[0].message == "Possible SQL injection"
    assert findings[0].line_range == (5, 7)


def test_run_semgrep_clean_run_returns_empty():
    """A clean run (returncode 0, no results) returns an empty list."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"results": []})
    mock_result.stderr = ""

    with (
        patch("app.data.collectors.semgrep_runner.shutil.which", return_value="/usr/bin/semgrep"),
        patch("app.data.collectors.semgrep_runner.subprocess.run", return_value=mock_result),
    ):
        findings = run_semgrep("safe code here", "python")

    assert findings == []


def test_run_semgrep_passes_config_and_timeout():
    """Custom config and timeout are forwarded to subprocess.run."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"results": []})
    mock_result.stderr = ""

    with (
        patch(
            "app.data.collectors.semgrep_runner.shutil.which",
            return_value="/usr/bin/semgrep",
        ),
        patch(
            "app.data.collectors.semgrep_runner.subprocess.run",
            return_value=mock_result,
        ) as mock_run,
    ):
        run_semgrep("code", "python", config="/custom/config.yaml", timeout=120)

    call_args = mock_run.call_args
    argv = call_args.kwargs.get("args") or call_args[0][0]
    assert "/custom/config.yaml" in argv
    assert call_args.kwargs["timeout"] == 120


# --- Error exit code ---


def test_run_semgrep_raises_on_error_exit_code():
    """A returncode outside {0, 1} raises RuntimeError with stderr."""
    mock_result = MagicMock()
    mock_result.returncode = 2
    mock_result.stdout = ""
    mock_result.stderr = "semgrep crashed"

    with (
        patch("app.data.collectors.semgrep_runner.shutil.which", return_value="/usr/bin/semgrep"),
        patch("app.data.collectors.semgrep_runner.subprocess.run", return_value=mock_result),
    ):
        with pytest.raises(RuntimeError, match="semgrep failed"):
            run_semgrep("code", "python")


# --- _to_static_finding defaults ---


def test_to_static_finding_handles_missing_fields():
    """Missing start/end/extra fields default gracefully."""
    from app.data.collectors.semgrep_runner import _to_static_finding

    finding = _to_static_finding({"check_id": "test-rule"})

    assert finding.rule_id == "test-rule"
    assert finding.message == ""
    assert finding.line_range == (0, 0)


def test_to_static_finding_defaults_end_to_start():
    """If end line is missing, it defaults to start line."""
    from app.data.collectors.semgrep_runner import _to_static_finding

    finding = _to_static_finding({
        "check_id": "test-rule",
        "start": {"line": 3},
    })

    assert finding.line_range == (3, 3)
