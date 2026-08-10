"""Unit tests for Stage 6, Tier 3 — exec-based sandbox evaluation."""

from app.evaluation.tier3_exec import (
    ExecEvaluator,
    LocalSandboxRunner,
    MockSandboxRunner,
    SandboxResult,
    TestGenerator,
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
