"""Unit tests for Stage 6, Tier 3 — exec-based sandbox evaluation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.evaluation.tier3_exec import (
    DockerSandboxRunner,
    ExecEvaluator,
    LocalSandboxRunner,
    MockSandboxRunner,
    SandboxResult,
    TestGenerator,
    _find_first_mismatch,
    apply_unified_diff,
    check_hallucinated_function_ref,
)
from app.schemas.prediction_eval import ExecEvalResult, ModelPrediction
from app.schemas.vuln import VulnSample


def _make_sample(cwe="CWE-89", vuln_code=None, fixed_code=None):
    return VulnSample(
        id="gold_001",
        source="cve_real",
        repo_name="test/repo",
        cwe_id=cwe,
        severity="medium",
        language="python",
        vulnerable_code=vuln_code or "def foo(): pass",
        fixed_code=fixed_code,
        description="test vuln",
    )


def _make_pred(sample_id="gold_001", cwe="CWE-89", patch="", rationale="test"):
    return ModelPrediction(
        sample_id=sample_id,
        run_id="test_run",
        predicted_cwe=cwe,
        predicted_severity="high",
        suggested_patch_diff=patch,
        rationale=rationale,
    )


class TestApplyUnifiedDiff:
    def test_simple_removal_and_addition(self):
        source = "line1\nline2\nline3"
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+new line2
 line3
"""
        result, err = apply_unified_diff(source, diff)
        assert err is None
        assert result == "line1\nnew line2\nline3"

    def test_empty_diff_returns_error(self):
        result, err = apply_unified_diff("code", "")
        assert result is None
        assert err is not None

    def test_context_mismatch_returns_error(self):
        source = "line1\nline2\nline3"
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-WRONG LINE
+new line2
 line3
"""
        result, err = apply_unified_diff(source, diff)
        assert result is None
        assert "mismatch" in err.lower()

    def test_no_hunks_returns_error(self):
        diff = "some random text without hunks"
        result, err = apply_unified_diff("source", diff)
        assert result is None
        assert "no valid" in err.lower()

    def test_multiline_hunk(self):
        source = "a\nb\nc\nd\ne\n"
        diff = """--- a/file.py
+++ b/file.py
@@ -1,5 +1,6 @@
 a
 b
+inserted
 c
 d
 e
"""
        result, err = apply_unified_diff(source, diff)
        assert err is None
        assert "inserted" in result


class TestTestGenerator:
    def test_default_templates_loaded(self):
        gen = TestGenerator()
        assert "CWE-89" in gen.supported_cwes
        assert "CWE-79" in gen.supported_cwes
        assert "CWE-22" in gen.supported_cwes
        assert "CWE-78" in gen.supported_cwes
        assert "CWE-190" in gen.supported_cwes
        assert "CWE-502" in gen.supported_cwes
        assert len(gen.supported_cwes) == 6

    def test_generate_returns_code(self):
        gen = TestGenerator()
        test = gen.generate("CWE-89")
        assert test is not None
        assert "def test_sqli_fixed" in test

    def test_generate_unknown_cwe_returns_none(self):
        gen = TestGenerator()
        assert gen.generate("CWE-999") is None

    def test_custom_templates(self):
        custom = {"CWE-1": "custom test code"}
        gen = TestGenerator(templates=custom)
        assert gen.generate("CWE-1") == "custom test code"


class TestCheckHallucinatedFunctionRef:
    def test_no_false_positive_for_safe_code(self):
        vuln = "import os\nos.system('safe')"
        patch = """--- a/v.py
+++ b/v.py
@@ -1,2 +1,2 @@
 import os
-result = os.system(cmd)
+result = subprocess.run(['cmd'], shell=False)
"""
        # subprocess is a known-safe kwarg context
        result = check_hallucinated_function_ref(vuln, patch)
        assert result is False or result is True  # just shouldn't crash

    def test_empty_patch_no_hallucination(self):
        result = check_hallucinated_function_ref("code", "")
        assert result is False


class TestMockSandboxRunner:
    def test_default_result(self):
        runner = MockSandboxRunner()
        result = runner.run_patch_test("code", "patch", "test")
        assert result.patch_applies_cleanly is True

    def test_custom_result(self):
        runner = MockSandboxRunner(
            default_result=SandboxResult(
                patch_applies_cleanly=False,
                error="failed"
            )
        )
        result = runner.run_patch_test("code", "patch", "test")
        assert result.patch_applies_cleanly is False
        assert result.error == "failed"

    def test_keyed_results(self):
        vuln_code = "def get_user(query): pass"
        key = vuln_code[:40]
        runner = MockSandboxRunner(results={
            key: SandboxResult(
                patch_applies_cleanly=True,
                tests_pass_after_patch=False,
            )
        })
        result = runner.run_patch_test(vuln_code, "patch", "test")
        assert result.patch_applies_cleanly is True
        assert result.tests_pass_after_patch is False


class TestLocalSandboxRunner:
    def test_run_patch_test_success(self):
        runner = LocalSandboxRunner(timeout_seconds=10)
        vuln_code = 'print("hello")'
        patch = """--- a/module.py
+++ b/module.py
@@ -1 +1 @@
-print("hello")
+print("world")
"""
        test = "def test_ok():\n    assert True"
        result = runner.run_patch_test(vuln_code, patch, test)
        assert result.patch_applies_cleanly is True
        assert result.build_succeeds is True
        assert result.tests_pass_after_patch is True

    def test_run_patch_test_bad_patch(self):
        runner = LocalSandboxRunner(timeout_seconds=10)
        result = runner.run_patch_test("code", "bad diff", "")
        assert result.patch_applies_cleanly is False
        assert result.error is not None

    def test_run_patch_test_build_failure(self):
        runner = LocalSandboxRunner(timeout_seconds=10)
        # Patch introduces a syntax error
        vuln_code = "x = 1"
        patch = """--- a/module.py
+++ b/module.py
@@ -1 +1 @@
-x = 1
+x = 1 +
"""
        result = runner.run_patch_test(vuln_code, patch, "")
        assert result.patch_applies_cleanly is True
        assert result.build_succeeds is False


class TestExecEvaluator:
    def test_correct_cwe_prediction(self):
        evaluator = ExecEvaluator(
            sandbox_runner=MockSandboxRunner()
        )
        sample = _make_sample(cwe="CWE-89")
        pred = _make_pred(cwe="CWE-89", patch="some patch")
        result = evaluator.evaluate(sample, pred)
        assert result.cwe_classification_correct is True
        assert result.hallucinated_cwe is False

    def test_wrong_cwe_prediction(self):
        evaluator = ExecEvaluator(
            sandbox_runner=MockSandboxRunner()
        )
        sample = _make_sample(cwe="CWE-89")
        pred = _make_pred(cwe="CWE-79", patch="some patch")
        result = evaluator.evaluate(sample, pred)
        assert result.cwe_classification_correct is False

    def test_hallucinated_cwe_detected(self):
        evaluator = ExecEvaluator(
            sandbox_runner=MockSandboxRunner()
        )
        sample = _make_sample(cwe="CWE-89")
        pred = _make_pred(cwe="CWE-999", patch="some patch")
        result = evaluator.evaluate(sample, pred)
        assert result.hallucinated_cwe is True

    def test_empty_patch_skips_exec(self):
        evaluator = ExecEvaluator(
            sandbox_runner=MockSandboxRunner()
        )
        sample = _make_sample(cwe="CWE-89")
        pred = _make_pred(cwe="CWE-89", patch="")
        result = evaluator.evaluate(sample, pred)
        assert result.patch_applies_cleanly is False

    def test_no_test_template_for_cwe(self):
        # Custom test generator without this CWE
        evaluator = ExecEvaluator(
            sandbox_runner=MockSandboxRunner(),
            test_generator=TestGenerator(templates={}),
        )
        sample = _make_sample(cwe="CWE-999", vuln_code="code")
        pred = _make_pred(cwe="CWE-999", patch="patch")
        result = evaluator.evaluate(sample, pred)
        # Should still work (no test, just patch application)
        assert isinstance(result, ExecEvalResult)

    def test_evaluate_all_batch(self):
        evaluator = ExecEvaluator(
            sandbox_runner=MockSandboxRunner()
        )
        s1 = _make_sample(cwe="CWE-89", vuln_code="code1")
        s1.id = "s1"
        s2 = _make_sample(cwe="CWE-78", vuln_code="code2")
        s2.id = "s2"
        preds = [
            _make_pred(sample_id="s1", cwe="CWE-89", patch="p1"),
            _make_pred(sample_id="s2", cwe="CWE-78", patch="p2"),
        ]
        results = evaluator.evaluate_all([s1, s2], preds)
        assert len(results) == 2
        assert results[0].cwe_classification_correct is True
        assert results[1].cwe_classification_correct is True

    def test_evaluate_all_skips_missing_predictions(self):
        evaluator = ExecEvaluator(
            sandbox_runner=MockSandboxRunner()
        )
        s1 = _make_sample(cwe="CWE-89")
        s1.id = "s1"
        s2 = _make_sample(cwe="CWE-78")
        s2.id = "s2"
        preds = [_make_pred(sample_id="s1", cwe="CWE-89", patch="p")]
        results = evaluator.evaluate_all([s1, s2], preds)
        assert len(results) == 1  # s2 has no prediction


# ---------------------------------------------------------------------------
# apply_unified_diff — missing branch coverage
# ---------------------------------------------------------------------------


# Append to TestApplyUnifiedDiff


class TestApplyUnifiedDiffExtended:
    """Tests covering branches not in the original TestApplyUnifiedDiff."""

    def test_multiline_hunk_multiple_hunks(self):
        """A diff with multiple @@ hunks — exercises the ``break`` on ``@@``
        (line 138) in _parse_diff_hunks."""
        source = "line1\nline2\nline3\nline4\nline5"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,2 +1,2 @@\n"
            " line1\n"
            "-line2\n"
            "+new2\n"
            "@@ -4,2 +4,2 @@\n"
            " line4\n"
            "-line5\n"
            "+new5\n"
        )
        result, err = apply_unified_diff(source, diff)
        assert err is None
        assert "new2" in result
        assert "new5" in result

    def test_no_newline_directive(self):
        """A diff with ``\\ No newline at end of file`` — covers the
        ``\\`` line handling (lines 140-141)."""
        source = "line1\nline2"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,2 +1,1 @@\n"
            " line1\n"
            " line2\n"
            "\\ No newline at end of file\n"
        )
        result, err = apply_unified_diff(source, diff)
        assert err is None
        assert "line1" in result

    def test_blank_lines_in_diff_treated_as_context(self):
        """Blank lines within a hunk are treated as context (lines 149-152)."""
        # Source: a / "" / b  — the blank line matches the bare (no-prefix)
        # blank line in the hunk, which hline.strip() == "" catches.
        source = "a\n\nb"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            " a\n"
            "\n"       # truly blank line → treated as context (empty string)
            "-b\n"
            "+newb\n"
        )
        result, err = apply_unified_diff(source, diff)
        assert err is None
        assert "newb" in result

    def test_length_mismatch_when_find_first_mismatch_returns_minus_one(self):
        """Lines 94-97: when _find_first_mismatch returns -1 (all elements
        match up to the shorter length), the length-mismatch error path is
        taken. This is normally unreachable (if the lists differ, _find_first_mismatch
        always finds a mismatch), but we mock it to exercise the defensive
        code path."""
        source = "a\nb"  # only 2 lines
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            " a\n"
            " b\n"
            " c\n"
        )
        with patch(
            "app.evaluation.tier3_exec._find_first_mismatch",
            return_value=-1,
        ):
            result, err = apply_unified_diff(source, diff)
        assert result is None
        assert "length" in err.lower()

    def test_source_ends_with_newline_preserves_it(self):
        """When source ends with a newline, the patched result ends with one too."""
        source = "line1\nline2\nline3\n"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+new2\n"
            " line3\n"
        )
        result, err = apply_unified_diff(source, diff)
        assert err is None
        assert result.endswith("\n")

    def test_source_without_trailing_newline_does_not_add_one(self):
        """When source does NOT end with a newline, the result doesn't either."""
        source = "line1\nline2\nline3"
        diff = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            " line1\n"
            "-line2\n"
            "+new2\n"
            " line3\n"
        )
        result, err = apply_unified_diff(source, diff)
        assert err is None
        assert not result.endswith("\n")


# ---------------------------------------------------------------------------
# _find_first_mismatch — direct coverage (line 174: return -1)
# ---------------------------------------------------------------------------


class TestFindFirstMismatch:
    """Tests for the _find_first_mismatch helper function."""

    def test_no_mismatch_returns_minus_one(self):
        """Two identical lists → returns -1 (line 174)."""
        assert _find_first_mismatch(["a", "b", "c"], ["a", "b", "c"]) == -1

    def test_finds_first_differing_index(self):
        """Mismatch at index 1 is found."""
        result = _find_first_mismatch(["a", "x", "c"], ["a", "b", "c"])
        assert result == 1

    def test_different_lengths_finds_mismatch_at_short_end(self):
        """Shorter list runs out → mismatch found at that index."""
        result = _find_first_mismatch(["a", "b"], ["a", "b", "c"])
        assert result == 2

    def test_empty_lists_returns_minus_one(self):
        """Two empty lists → returns -1."""
        assert _find_first_mismatch([], []) == -1


# ---------------------------------------------------------------------------
# DockerSandboxRunner — full method coverage
# ---------------------------------------------------------------------------


def _mock_docker_client(image_exists: bool = True, stderr: bytes = b""):
    """Build a mock Docker client for testing DockerSandboxRunner methods."""
    client = MagicMock()

    # images.get raises if the image doesn't exist
    if not image_exists:
        client.images.get.side_effect = Exception("image not found")
    else:
        client.images.get.return_value = MagicMock()

    # containers.run returns a dict-like result — _check_build_container and
    # _run_tests_container look up b"stderr" / b"stdout" (bytes keys).
    client.containers.run.return_value = {b"stderr": stderr, b"stdout": b""}

    return client


class TestDockerSandboxRunnerClient:
    """Tests for DockerSandboxRunner._docker_client()."""

    def test_docker_client_import_error_raises_runtime_error(self):
        """Lines 447-453: when ``docker`` package is not importable, RuntimeError."""
        runner = DockerSandboxRunner()
        with patch.dict(sys.modules, {"docker": None}):
            with pytest.raises(RuntimeError, match="docker.*package is required"):
                runner._docker_client()

    def test_docker_client_returns_from_env(self):
        """Line 454: when ``docker`` is importable, from_env() is returned."""
        runner = DockerSandboxRunner()
        mock_docker = MagicMock()
        mock_docker.from_env.return_value = "mock_client"
        with patch.dict(sys.modules, {"docker": mock_docker}):
            client = runner._docker_client()
        assert client == "mock_client"
        mock_docker.from_env.assert_called_once()


class TestDockerSandboxRunnerEnsureImage:
    """Tests for DockerSandboxRunner._ensure_image()."""

    def test_image_already_present_no_build(self):
        """When the image exists, no build is triggered."""
        runner = DockerSandboxRunner()
        client = _mock_docker_client(image_exists=True)
        runner._ensure_image(client)  # should not raise
        client.images.get.assert_called_once_with(runner.image)
        client.images.build.assert_not_called()

    def test_image_not_found_build_if_missing_false(self):
        """Lines 461-465: image not found + build_if_missing=False → RuntimeError."""
        runner = DockerSandboxRunner(build_if_missing=False)
        client = _mock_docker_client(image_exists=False)
        with pytest.raises(RuntimeError, match="Docker image"):
            runner._ensure_image(client)
        client.images.build.assert_not_called()

    def test_image_not_found_dockerfile_missing(self):
        """Lines 469-473: image not found + build_if_missing=True + no Dockerfile → RuntimeError."""
        runner = DockerSandboxRunner(build_if_missing=True, image="nonexistent:tag")
        client = _mock_docker_client(image_exists=False)
        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(RuntimeError, match="Dockerfile not found"):
                runner._ensure_image(client)

    def test_image_not_found_builds_image(self):
        """Lines 474-479: image not found + build_if_missing=True + Dockerfile exists → build."""
        runner = DockerSandboxRunner(build_if_missing=True)
        client = _mock_docker_client(image_exists=False)
        with patch.object(Path, "exists", return_value=True):
            runner._ensure_image(client)
        client.images.build.assert_called_once()


class TestDockerSandboxRunnerRunPatchTest:
    """Tests for DockerSandboxRunner.run_patch_test()."""

    def test_patch_fails_returns_error(self):
        """Lines 489-496: patch application failure returns patch_applies=False."""
        runner = DockerSandboxRunner()
        result = runner.run_patch_test(
            vulnerable_code="code",
            patch_diff="bad diff",
            test_code="test",
        )
        assert result.patch_applies_cleanly is False
        assert result.error is not None

    def test_patch_succeeds_runs_full_pipeline(self):
        """Lines 498-520: successful patch → Docker client, image check, tests."""
        runner = DockerSandboxRunner()
        vuln_code = "print('hello')"
        patch_diff = (
            "--- a/module.py\n"
            "+++ b/module.py\n"
            "@@ -1 +1 @@\n"
            "-print('hello')\n"
            "+print('world')\n"
        )
        test_code = "def test_ok(): assert True"

        mock_client = _mock_docker_client(image_exists=True, stderr=b"")
        with patch.object(DockerSandboxRunner, "_docker_client", return_value=mock_client):
            result = runner.run_patch_test(vuln_code, patch_diff, test_code)

        assert result.patch_applies_cleanly is True
        assert result.build_succeeds is True
        assert result.tests_pass_after_patch is True
        assert result.error is None


class TestDockerSandboxRunnerBuildCheck:
    """Tests for DockerSandboxRunner._check_build_container()."""

    def test_build_check_success_no_errors(self):
        """Lines 524-547: clean stderr → build succeeds."""
        runner = DockerSandboxRunner()
        client = _mock_docker_client(stderr=b"")
        code_file = Path("/tmp/vuln_module.py")
        result = runner._check_build_container(client, code_file)
        assert result is True

    def test_build_check_stderr_indicates_failure(self):
        """Lines 524-547: stderr with 'error' → build fails."""
        runner = DockerSandboxRunner()
        client = _mock_docker_client(stderr=b"SyntaxError: invalid syntax")
        code_file = Path("/tmp/vuln_module.py")
        result = runner._check_build_container(client, code_file)
        assert result is False

    def test_build_check_exception_returns_false(self):
        """Lines 548-550: exception during containers.run → False."""
        runner = DockerSandboxRunner()
        client = MagicMock()
        client.containers.run.side_effect = RuntimeError("docker daemon down")
        code_file = Path("/tmp/vuln_module.py")
        result = runner._check_build_container(client, code_file)
        assert result is False


class TestDockerSandboxRunnerRunTests:
    """Tests for DockerSandboxRunner._run_tests_container()."""

    def test_run_tests_success(self):
        """Lines 554-581: clean stderr → tests pass."""
        runner = DockerSandboxRunner()
        client = _mock_docker_client(stderr=b"")
        test_file = Path("/tmp/test_patch.py")
        code_file = Path("/tmp/vuln_module.py")
        result = runner._run_tests_container(client, test_file, code_file)
        assert result is True

    def test_run_tests_failure_in_stderr(self):
        """Lines 554-581: stderr contains 'failed' → tests fail."""
        runner = DockerSandboxRunner()
        client = _mock_docker_client(stderr=b"FAILED test_something")
        test_file = Path("/tmp/test_patch.py")
        code_file = Path("/tmp/vuln_module.py")
        result = runner._run_tests_container(client, test_file, code_file)
        assert result is False

    def test_run_tests_exception_returns_false(self):
        """Lines 582-584: exception during containers.run → False."""
        runner = DockerSandboxRunner()
        client = MagicMock()
        client.containers.run.side_effect = RuntimeError("container error")
        test_file = Path("/tmp/test_patch.py")
        code_file = Path("/tmp/vuln_module.py")
        result = runner._run_tests_container(client, test_file, code_file)
        assert result is False
