"""Stage 0 DoD: validation tests for every Pydantic schema in app/schemas.

These are intentionally minimal — one happy-path instantiation per model,
plus a couple of contract checks (required fields, enum literals) that
later stages depend on (e.g. leakage-safe split relies on repo_name being
mandatory).
"""

import pytest
from pydantic import ValidationError

from app.schemas.dataset import InstructionExample
from app.schemas.prediction_eval import (
    ExecEvalResult,
    LlmJudgeScore,
    ModelPrediction,
    RegressionSummary,
)
from app.schemas.training import TrainingRun
from app.schemas.vuln import StaticFinding, VulnSample


def test_static_finding_valid():
    finding = StaticFinding(
        tool="semgrep",
        rule_id="python.lang.security.sqli",
        message="Possible SQL injection via string formatting.",
        line_range=(10, 12),
    )
    assert finding.tool == "semgrep"


def test_vuln_sample_valid_minimal():
    sample = VulnSample(
        id="vs_0001",
        source="cve_real",
        repo_name="example/vulnerable-app",
        commit_sha="abc123",
        cve_id="CVE-2024-0001",
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code="cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
        fixed_code="cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
        static_findings=[],
        description="SQL injection via unparameterized query.",
    )
    assert sample.split is None  # unassigned until Stage 2


def test_vuln_sample_requires_repo_name_for_leakage_safe_split():
    with pytest.raises(ValidationError):
        VulnSample(
            id="vs_0002",
            source="cve_real",
            # repo_name intentionally omitted
            cwe_id="CWE-79",
            severity="medium",
            language="javascript",
            vulnerable_code="element.innerHTML = userInput;",
            description="Reflected XSS.",
        )


def test_vuln_sample_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        VulnSample(
            id="vs_0003",
            source="cve_real",
            repo_name="example/app",
            cwe_id="CWE-22",
            severity="extreme",  # not in the allowed Literal
            language="go",
            vulnerable_code="...",
            description="Path traversal.",
        )


def test_instruction_example_valid():
    example = InstructionExample(
        id="ie_0001",
        sample_id="vs_0001",
        prompt="### Task\n...",
        target_cwe="CWE-89",
        target_severity="high",
        target_explanation="Unsanitized input concatenated into SQL query.",
        target_patch_diff="--- a/app.py\n+++ b/app.py\n...",
        token_count_estimate=420,
    )
    assert example.token_count_estimate > 0


def test_training_run_valid():
    run = TrainingRun(
        id="run_0001",
        method="sft_qlora",
        base_model="Qwen2.5-Coder-7B-Instruct",
        hyperparams={"rank": 16, "alpha": 32, "lr": 2e-4, "epochs": 3},
        train_set_size=700,
        train_time_minutes=55.0,
        peak_vram_gb=7.4,
        final_train_loss=0.31,
        checkpoint_uri="s3://vuln-triage/checkpoints/run_0001",
    )
    assert run.method == "sft_qlora"


def test_model_prediction_and_exec_eval_result_valid():
    prediction = ModelPrediction(
        sample_id="vs_0001",
        run_id="run_0001",
        predicted_cwe="CWE-89",
        predicted_severity="high",
        suggested_patch_diff="--- a/app.py\n+++ b/app.py\n...",
        rationale="Query uses string concatenation with unsanitized input.",
    )
    result = ExecEvalResult(
        prediction_id=prediction.sample_id,
        patch_applies_cleanly=True,
        build_succeeds=True,
        tests_pass_after_patch=True,
        cwe_classification_correct=True,
        hallucinated_cwe=False,
        hallucinated_function_ref=False,
    )
    assert result.tests_pass_after_patch is True


def test_llm_judge_score_valid():
    score = LlmJudgeScore(
        prediction_id="vs_0001",
        explanation_quality=0.82,
        patch_minimality=0.9,
        evaluator_model="claude-sonnet-4-6",
        rationale="Explanation is accurate; patch changes only the vulnerable line.",
    )
    assert 0.0 <= score.explanation_quality <= 1.0


def test_regression_summary_valid():
    summary = RegressionSummary(
        run_id="run_0001",
        cwe_macro_f1=0.71,
        exec_pass_rate=0.58,
        hallucination_rate=0.06,
        general_capability_delta=-0.02,
        cost_per_accepted_patch_usd=0.014,
    )
    assert summary.run_id == "run_0001"
