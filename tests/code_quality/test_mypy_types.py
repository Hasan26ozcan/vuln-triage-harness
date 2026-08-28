"""SonarQube-style static type checking via mypy.

Runs ``mypy`` against the ``app/`` package as a pytest test, so that
type errors are caught in CI the same way test failures are — no separate
``mypy`` invocation needed in the Makefile or CI workflow.

The mypy configuration lives in ``pyproject.toml`` under ``[tool.mypy]``.
Strict mode is enabled globally; ML-heavy modules (training, quantization,
serving backends, etc.) have per-module overrides that relax specific checks
because they use lazy imports for torch/transformers/peft.

Run directly::

    pytest tests/code_quality/test_mypy_types.py -v
    python -m mypy app --config-file pyproject.toml
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# --- Locate the project root (two levels up from this test file) --------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = PROJECT_ROOT / "app"
PYPROJECT = PROJECT_ROOT / "pyproject.toml"

# --- mypy availability guard --------------------------------------------------
_MYPY_AVAILABLE = True
try:
    import mypy  # noqa: F401
except ImportError:
    _MYPY_AVAILABLE = False


def _run_mypy() -> tuple[int, str, str]:
    """Run mypy on ``app/`` and return (exit_code, stdout, stderr)."""
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "app",
        "--config-file",
        str(PYPROJECT),
        "--no-error-summary",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.skipif(not _MYPY_AVAILABLE, reason="mypy is not installed")
class TestMypyTypeChecks:
    """Ensure mypy passes cleanly on the ``app/`` package.

    This mirrors SonarQube's "Reliability" / "Maintainability" quality gates:
    type errors are treated as test failures and must not accumulate.
    """

    def test_mypy_passes_on_app(self) -> None:
        """mypy reports zero type errors across all of ``app/``."""
        exit_code, stdout, stderr = _run_mypy()
        if exit_code != 0:
            combined = stdout + stderr
            # Strip ANSI colour codes for a cleaner failure message.
            import re

            clean = re.sub(r"\x1b\[[0-9;]*m", "", combined)
            pytest.fail(
                f"mypy found {len(clean.strip().splitlines())} type issue(s):\n"
                f"--- mypy output ---\n{clean}\n"
                f"--- end output ---\n"
                f"To fix: run ``python -m mypy app`` and address the errors.",
                pytrace=False,
            )

    def test_mypy_exit_code_is_zero(self) -> None:
        """mypy process exits with code 0 (explicit exit-code check)."""
        exit_code, stdout, stderr = _run_mypy()
        assert exit_code == 0, (
            f"mypy exited with code {exit_code}.\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )

    def test_mypy_config_exists(self) -> None:
        """The project must contain a ``[tool.mypy]`` section in pyproject.toml."""
        content = PYPROJECT.read_text(encoding="utf-8")
        assert "[tool.mypy]" in content, "pyproject.toml is missing [tool.mypy] section"

    def test_mypy_plugin_enabled(self) -> None:
        """The pydantic mypy plugin must be enabled (for Pydantic v2 models)."""
        content = PYPROJECT.read_text(encoding="utf-8")
        assert "pydantic.mypy" in content, (
            "pydantic.mypy plugin is not enabled in mypy config — "
            "Pydantic models will not be type-checked properly."
        )
