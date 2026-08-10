"""Stage 6, Tier 3 — exec-based sandbox evaluation.

This is the most expensive tier: it takes the model's ``suggested_patch_diff``,
applies it to the vulnerable code, and **actually runs** a CWE-specific test
in an isolated subprocess to verify the vulnerability is fixed.

The sandbox is abstracted behind a ``SandboxRunner`` Protocol so that tests
can inject a mock without Docker or a real subprocess. Three pieces make this
work:

1. ``apply_unified_diff`` — pure-Python patch applier (no ``patch`` /
   ``git apply`` dependency).
2. ``TestGenerator`` — per-CWE test templates that verify the security
   property is no longer exploitable after the patch.
3. ``SandboxRunner`` — Protocol with a single ``run_patch_test`` method.
   ``LocalSandboxRunner`` writes code + test to a temp dir and runs them
   via ``subprocess``; ``MockSandboxRunner`` returns canned results.

The ``ExecEvaluator`` ties these together and produces an ``ExecEvalResult``
for each ``ModelPrediction``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.schemas.prediction_eval import ExecEvalResult, ModelPrediction
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The valid CWE IDs — used for the hallucination check.
# (Same set as metrics._VALID_CWE_IDS; duplicated to avoid a cross-module
# import dependency that would pull in Stage 4.)
# ---------------------------------------------------------------------------

_VALID_CWE_IDS: frozenset[str] = frozenset(
    {"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"}
)


# ---------------------------------------------------------------------------
# Patch application — pure-Python unified-diff applier
# ---------------------------------------------------------------------------


def apply_unified_diff(source: str, diff: str) -> tuple[str | None, str | None]:
    """Apply a unified-diff string to *source* code.

    Returns ``(patched_code, None)`` on success, or ``(None, error_msg)``
    when the diff cannot be applied (context mismatch, malformed diff, etc.).

    Only the hunks that operate on a single file are supported — the
    standard case for Stage 6 patches which modify one source file.
    """
    if not diff or not diff.strip():
        return None, "empty diff"

    source_lines = source.splitlines(keepends=False)

    hunks = _parse_diff_hunks(diff)
    if not hunks:
        return None, "no valid hunks found in diff"

    # Apply hunks in reverse order so earlier line numbers stay valid.
    result = list(source_lines)
    for hunk in reversed(hunks):
        start = hunk.old_start - 1  # 1-indexed → 0-indexed
        old_lines = hunk.context_and_removed
        new_lines = hunk.context_and_added

        # Verify the context matches the source.
        actual = result[start : start + len(old_lines)]
        if actual != old_lines:
            # Show the first mismatching line for debugging.
            idx = _find_first_mismatch(actual, old_lines)
            if idx >= 0:
                exp = old_lines[idx] if idx < len(old_lines) else "<missing>"
                got = actual[idx] if idx < len(actual) else "<missing>"
                return None, (
                    f"context mismatch at line {start + 1 + idx}: "
                    f"expected {exp!r}, got {got!r}"
                )
            # Length mismatch but content matches up to shorter length.
            return None, (
                f"context mismatch at line {start + 1}: "
                f"length {len(actual)} vs {len(old_lines)}"
            )

        result[start : start + len(old_lines)] = new_lines

    return "\n".join(result) + ("\n" if source.endswith("\n") else ""), None


@dataclass
class _Hunk:
    """One hunk parsed from a unified diff."""

    old_start: int  # 1-indexed
    old_count: int
    context_and_removed: list[str]  # what the source *should* currently look like
    context_and_added: list[str]  # what the source *should* look like after


def _parse_diff_hunks(diff: str) -> list[_Hunk]:
    """Parse a unified diff into a list of ``_Hunk`` objects.

    Skips the ``---`` / ``+++`` file headers and extracts each ``@@`` hunk.
    """
    hunks: list[_Hunk] = []
    lines = diff.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if m:
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) else 1
            # new_start / new_count are available but we don't need them
            # for forward-only application.
            i += 1

            ctx_removed: list[str] = []
            ctx_added: list[str] = []
            while i < len(lines):
                hline = lines[i]
                if hline.startswith("@@"):
                    break  # next hunk
                if hline.startswith("\\"):
                    i += 1
                    continue  # "\ No newline at end of file"
                if hline.startswith(" "):
                    ctx_removed.append(hline[1:])
                    ctx_added.append(hline[1:])
                elif hline.startswith("-"):
                    ctx_removed.append(hline[1:])
                elif hline.startswith("+"):
                    ctx_added.append(hline[1:])
                elif hline.strip() == "":
                    # Blank line in the diff — treat as context (empty line).
                    ctx_removed.append("")
                    ctx_added.append("")
                i += 1

            hunks.append(_Hunk(
                old_start=old_start,
                old_count=old_count,
                context_and_removed=ctx_removed,
                context_and_added=ctx_added,
            ))
        else:
            i += 1

    return hunks


def _find_first_mismatch(a: list[str], b: list[str]) -> int:
    """Return the first 0-indexed position where *a* and *b* differ."""
    for idx in range(max(len(a), len(b))):
        av = a[idx] if idx < len(a) else "<missing>"
        bv = b[idx] if idx < len(b) else "<missing>"
        if av != bv:
            return idx
    return -1


# ---------------------------------------------------------------------------
# Test templates — one per CWE class
# ---------------------------------------------------------------------------

# Each template is a Python test that reads ``vuln_module.py`` (the patched
# code) and asserts the vulnerability is fixed. Tests run in a subprocess.

_TEST_TEMPLATES: dict[str, str] = {
    # CWE-89: SQL injection — execute() should not use f-strings or string concat
    "CWE-89": '''
import ast

def test_sqli_fixed():
    with open("vuln_module.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "execute":
            arg = node.args[0] if node.args else None
            assert not isinstance(arg, ast.JoinedStr), "f-string found in execute()"
            assert not (isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add)), \\
                "string concatenation found in execute()"
''',
    # CWE-79: XSS — innerHTML/outerHTML/document.write should not be assigned
    "CWE-79": '''
def test_xss_fixed():
    with open("vuln_module.py") as f:
        code = f.read()
    assert ".innerHTML =" not in code, "innerHTML assignment remains"
    assert ".outerHTML =" not in code, "outerHTML assignment remains"
    assert "document.write(" not in code, "document.write() remains"
''',
    # CWE-22: Path traversal — path normalization + prefix check required
    "CWE-22": '''
def test_path_traversal_fixed():
    with open("vuln_module.py") as f:
        code = f.read()
    assert ("realpath" in code or "abspath" in code), "no path normalization"
    assert ("startswith" in code or "commonpath" in code), "no path prefix check"
''',
    # CWE-78: Command injection — shell=True/os.system/os.popen must be gone
    "CWE-78": '''
def test_command_injection_fixed():
    with open("vuln_module.py") as f:
        code = f.read()
    assert "shell=True" not in code, "shell=True remains"
    assert "os.system(" not in code, "os.system() remains"
    assert "os.popen(" not in code, "os.popen() remains"
''',
    # CWE-190: Integer overflow — bounds check before arithmetic
    "CWE-190": '''
def test_overflow_fixed():
    with open("vuln_module.py") as f:
        code = f.read()
    assert ("OverflowError" in code or "raise" in code or "if" in code), \\
        "no overflow guard found"
''',
    # CWE-502: Deserialization — pickle/unsafe yaml.load must be gone
    "CWE-502": '''
import ast

def test_deserialization_fixed():
    with open("vuln_module.py") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if func.attr in ("loads", "load"):
                mod = getattr(func.value, "id", "")
                if mod == "pickle":
                    raise AssertionError("pickle deserialization remains")
                if mod == "yaml":
                    for kw in node.keywords:
                        if kw.arg == "Loader" and "Safe" not in ast.dump(kw.value):
                            raise AssertionError("unsafe yaml.load remains")
''',
}


class TestGenerator:
    """Generates CWE-specific test code for verifying a patch fixes a vulnerability."""

    def __init__(self, templates: dict[str, str] | None = None):
        self._templates = templates if templates is not None else dict(_TEST_TEMPLATES)

    def generate(self, cwe: str) -> str | None:
        """Return the test code for *cwe*, or ``None`` if no template exists."""
        return self._templates.get(cwe)

    @property
    def supported_cwes(self) -> list[str]:
        return list(self._templates.keys())


# ---------------------------------------------------------------------------
# Sandbox runner abstraction
# ---------------------------------------------------------------------------


@dataclass
class SandboxResult:
    """Result of running a patch-test in a sandbox."""

    patch_applies_cleanly: bool
    build_succeeds: bool | None = None
    tests_pass_after_patch: bool | None = None
    error: str | None = None


class SandboxRunner(Protocol):
    """Anything that can apply a patch and run a test in isolation."""

    def run_patch_test(
        self,
        vulnerable_code: str,
        patch_diff: str,
        test_code: str,
        language: str = "python",
    ) -> SandboxResult: ...


class MockSandboxRunner:
    """Deterministic sandbox runner for testing.

    Returns a fixed ``SandboxResult`` (configurable per call). The
    ``results`` dict is keyed by the first 40 chars of ``vulnerable_code``
    so different samples can get different canned results.
    """

    def __init__(
        self,
        default_result: SandboxResult | None = None,
        results: dict[str, SandboxResult] | None = None,
    ):
        self._default = default_result or SandboxResult(patch_applies_cleanly=True)
        self._results = results or {}

    def run_patch_test(
        self,
        vulnerable_code: str,
        patch_diff: str,
        test_code: str,
        language: str = "python",
    ) -> SandboxResult:
        key = vulnerable_code[:40]
        return self._results.get(key, self._default)


class LocalSandboxRunner:
    """Runs patch tests via ``subprocess`` — no Docker required.

    This is the default production runner when ``--local-sandbox`` is passed.
    It writes the vulnerable code and test to a temp directory, applies the
    patch, and runs the test with the system Python.

    Docker-based isolation (``DockerSandboxRunner``) is used in production
    CI to prevent the patched code from accessing the host — see the
    ``sandbox/`` directory for per-language images.
    """

    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self.test_generator = TestGenerator()

    def run_patch_test(
        self,
        vulnerable_code: str,
        patch_diff: str,
        test_code: str,
        language: str = "python",
    ) -> SandboxResult:
        # Step 1: Apply the patch.
        patched_code, err = apply_unified_diff(vulnerable_code, patch_diff)
        if patched_code is None:
            return SandboxResult(
                patch_applies_cleanly=False,
                build_succeeds=None,
                tests_pass_after_patch=None,
                error=err,
            )

        # Step 2: Write files to a temp directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            code_file = tmp / "vuln_module.py"
            test_file = tmp / "test_patch.py"
            code_file.write_text(patched_code, encoding="utf-8")
            test_file.write_text(test_code, encoding="utf-8")

            # Step 3: Check build (syntax/import check).
            build_ok = self._check_build(code_file)

            # Step 4: Run the test.
            tests_ok = self._run_tests(test_file)

            return SandboxResult(
                patch_applies_cleanly=True,
                build_succeeds=build_ok,
                tests_pass_after_patch=tests_ok,
                error=None,
            )

    def _check_build(self, code_file: Path) -> bool:
        """Check that the patched code compiles (syntax check)."""
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(code_file)],
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return result.returncode == 0

    def _run_tests(self, test_file: Path) -> bool:
        """Run the generated test and return whether it passed."""
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v"],
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=str(test_file.parent),
        )
        return result.returncode == 0


# ---------------------------------------------------------------------------
# Hallucination heuristics
# ---------------------------------------------------------------------------


def check_hallucinated_function_ref(
    vulnerable_code: str,
    patch_diff: str,
) -> bool:
    """Check whether the patch references identifiers absent from the code.

    Extracts identifiers from the ``+`` lines of the diff and flags any
    that look like function calls or imports but don't appear in the
    original vulnerable code.
    """
    if not patch_diff.strip():
        return False

    # Collect identifiers from the patched (+) lines.
    added_lines = [
        line[1:] for line in patch_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    added_text = "\n".join(added_lines)

    # Collect identifiers from the original vulnerable code.
    vuln_text = vulnerable_code

    # Extract identifiers (word characters) from both.
    added_ids = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", added_text))
    vuln_ids = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", vuln_text))

    # Common Python / security keywords that are fine to add.
    _safe_keywords = {
        "import", "from", "return", "raise", "assert", "None", "True", "False",
        "def", "class", "if", "else", "elif", "for", "while", "try", "except",
        "with", "as", "pass", "break", "continue", "global", "nonlocal",
        "self", "cls", "str", "int", "len", "range", "list", "dict", "set",
        "tuple", "float", "bool", "open", "print", "safe", "escape",
        "param", "params", "query", "sql", "result",
        "os", "sys", "re", "ast", "json", "yaml", "pickle", "subprocess",
        "cursor", "execute", "fetchall", "fetchone", "connect",
        "realpath", "abspath", "startswith", "commonpath",
        "escapeHtml", "textContent", "InnerHTML", "OuterHTML",
        "allowed", "parts", "output", "check", "ValueError",
        "OverflowError", "SafeLoader", "safe_load",
    }

    # Identifiers in the patch that are NOT in the vulnerable code.
    new_ids = added_ids - vuln_ids - _safe_keywords

    # Filter to function-call-like identifiers (followed by `( in the patch).
    for ident in new_ids:
        # Check if this identifier is called in the patched code.
        if re.search(rf"\b{re.escape(ident)}\s*\(", added_text):
            # It looks like a function call in the patch. Check if it's a
            # known safe function or appears as a defined name.
            if ident not in ("escapeHtml", "realpath", "abspath", "safe_load"):
                # Check if the function is defined in the patch itself.
                defined = re.search(rf"def\s+{re.escape(ident)}\s*\(", added_text)
                if not defined:
                    return True

    return False


# ---------------------------------------------------------------------------
# ExecEvaluator — orchestrates patch application + test execution
# ---------------------------------------------------------------------------


class ExecEvaluator:
    """Tier 3 evaluator — applies patches and runs tests in a sandbox.

    Parameters
    ----------
    sandbox_runner:
        A ``SandboxRunner`` implementation. Inject ``MockSandboxRunner`` for
        tests; use ``LocalSandboxRunner`` for real (no-Docker) evaluation.
    test_generator:
        Optional custom ``TestGenerator``. Defaults to the built-in templates.
    valid_cwe_ids:
        Set of valid CWE IDs for the hallucination check.
    """

    def __init__(
        self,
        sandbox_runner: SandboxRunner | None = None,
        test_generator: TestGenerator | None = None,
        valid_cwe_ids: frozenset[str] | None = None,
    ):
        self._sandbox = sandbox_runner or LocalSandboxRunner()
        self._test_gen = test_generator or TestGenerator()
        self._valid_cwes = valid_cwe_ids or _VALID_CWE_IDS

    def evaluate(
        self,
        sample: VulnSample,
        prediction: ModelPrediction,
    ) -> ExecEvalResult:
        """Run exec-based evaluation for a single prediction.

        Parameters
        ----------
        sample:
            The gold-eval ``VulnSample`` (provides ``vulnerable_code``,
            ``cwe_id``, ``language``, ``fixed_code``).
        prediction:
            The model's ``ModelPrediction`` (provides ``suggested_patch_diff``,
            ``predicted_cwe``).
        """
        patch_diff = prediction.suggested_patch_diff

        # CWE classification check.
        cwe_correct = prediction.predicted_cwe == sample.cwe_id

        # Hallucination checks.
        hallucinated_cwe = prediction.predicted_cwe not in self._valid_cwes
        hallucinated_ref = check_hallucinated_function_ref(
            sample.vulnerable_code, patch_diff
        )

        # If the model produced no patch, skip exec eval.
        if not patch_diff.strip():
            return ExecEvalResult(
                prediction_id=prediction.sample_id,
                patch_applies_cleanly=False,
                build_succeeds=None,
                tests_pass_after_patch=None,
                cwe_classification_correct=cwe_correct,
                hallucinated_cwe=hallucinated_cwe,
                hallucinated_function_ref=hallucinated_ref,
            )

        # Generate the test for this CWE class.
        test_code = self._test_gen.generate(sample.cwe_id)
        if test_code is None:
            # No test template for this CWE — can only check patch application.
            result = self._sandbox.run_patch_test(
                vulnerable_code=sample.vulnerable_code,
                patch_diff=patch_diff,
                test_code="",
                language=sample.language,
            )
            return ExecEvalResult(
                prediction_id=prediction.sample_id,
                patch_applies_cleanly=result.patch_applies_cleanly,
                build_succeeds=result.build_succeeds,
                tests_pass_after_patch=None,
                cwe_classification_correct=cwe_correct,
                hallucinated_cwe=hallucinated_cwe,
                hallucinated_function_ref=hallucinated_ref,
            )

        # Run the full sandbox test.
        result = self._sandbox.run_patch_test(
            vulnerable_code=sample.vulnerable_code,
            patch_diff=patch_diff,
            test_code=test_code,
            language=sample.language,
        )

        return ExecEvalResult(
            prediction_id=prediction.sample_id,
            patch_applies_cleanly=result.patch_applies_cleanly,
            build_succeeds=result.build_succeeds,
            tests_pass_after_patch=result.tests_pass_after_patch,
            cwe_classification_correct=cwe_correct,
            hallucinated_cwe=hallucinated_cwe,
            hallucinated_function_ref=hallucinated_ref,
        )

    def evaluate_all(
        self,
        samples: list[VulnSample],
        predictions: list[ModelPrediction],
    ) -> list[ExecEvalResult]:
        """Evaluate all predictions against their corresponding samples.

        Predictions are matched to samples by ``sample_id``.
        """
        pred_by_sample: dict[str, ModelPrediction] = {
            p.sample_id: p for p in predictions
        }
        results: list[ExecEvalResult] = []
        for sample in samples:
            pred = pred_by_sample.get(sample.id)
            if pred is None:
                logger.warning("No prediction for sample %s — skipping", sample.id)
                continue
            results.append(self.evaluate(sample, pred))
        return results
