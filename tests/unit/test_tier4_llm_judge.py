"""Unit tests for Stage 6, Tier 4 — LLM judge for explanation quality."""

from app.evaluation.tier4_llm_judge import (
    JUDGE_PROMPT,
    LlmJudge,
    MockLlmJudgeBackend,
    _parse_judge_response,
)
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample


def _make_sample(cwe="CWE-89"):
    return VulnSample(
        id="test_001",
        source="cve_real",
        repo_name="test/repo",
        cwe_id=cwe,
        severity="medium",
        language="python",
        vulnerable_code=(
            "def get_user(user_id):\n"
            "    query = 'SELECT * FROM users WHERE id=' + str(user_id)"
        ),
        fixed_code=(
            "def get_user(user_id):\n"
            "    query = 'SELECT * FROM users WHERE id = %s'\n"
            "    cursor.execute(query, (user_id,))"
        ),
        description="SQL injection vulnerability",
    )


def _make_pred(sample_id="test_001"):
    return ModelPrediction(
        sample_id=sample_id,
        run_id="test_run",
        predicted_cwe="CWE-89",
        predicted_severity="high",
        suggested_patch_diff=(
            "--- a/app.py\n+++ b/app.py\n"
            "- query = 'SELECT' + str(id)\n"
            "+ query = 'SELECT' + '%s'"
        ),
        rationale="Replace string concatenation with parameterized query.",
    )


class TestMockLlmJudgeBackend:
    def test_invoke_returns_json(self):
        backend = MockLlmJudgeBackend()
        result = backend.invoke("any prompt", "any-model", 100)
        assert "explanation_quality" in result
        assert "patch_minimality" in result
        assert "rationale" in result

    def test_default_scores(self):
        backend = MockLlmJudgeBackend(
            fallback_explanation_quality=0.8,
            fallback_patch_minimality=0.7,
        )
        result = backend.invoke("prompt", "model", 100)
        import json
        data = json.loads(result)
        assert data["explanation_quality"] == 0.8
        assert data["patch_minimality"] == 0.7


class TestParseJudgeResponse:
    def test_valid_json(self):
        text = '{"explanation_quality": 0.9, "patch_minimality": 0.8, "rationale": "good"}'
        eq, pm, r = _parse_judge_response(text)
        assert eq == 0.9
        assert pm == 0.8
        assert r == "good"

    def test_invalid_json_fallback(self):
        eq, pm, r = _parse_judge_response("not json")
        assert eq == 0.5
        assert pm == 0.5
        assert len(r) <= 200

    def test_values_clamped(self):
        text = '{"explanation_quality": 1.5, "patch_minimality": -0.5, "rationale": "x"}'
        eq, pm, r = _parse_judge_response(text)
        assert eq == 1.0
        assert pm == 0.0

    def test_missing_fields(self):
        text = '{"rationale": "only rationale"}'
        eq, pm, r = _parse_judge_response(text)
        assert eq == 0.5
        assert pm == 0.5
        assert r == "only rationale"


class TestLlmJudge:
    def test_mock_backend_default(self):
        """With no API key, should use MockLlmJudgeBackend."""
        import os
        # Ensure no API key is set
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            judge = LlmJudge()
            assert isinstance(judge._backend, MockLlmJudgeBackend)
        finally:
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_evaluate_single(self):
        backend = MockLlmJudgeBackend(
            fallback_explanation_quality=0.7,
            fallback_patch_minimality=0.6,
        )
        judge = LlmJudge(backend=backend)
        sample = _make_sample()
        pred = _make_pred()
        score = judge.evaluate(sample, pred)
        assert score.prediction_id == "test_001"
        assert score.explanation_quality == 0.7
        assert score.patch_minimality == 0.6
        assert score.evaluator_model == "gpt-4o-mini"

    def test_evaluate_all_batch(self):
        backend = MockLlmJudgeBackend()
        judge = LlmJudge(backend=backend)
        s1 = _make_sample()
        s1.id = "s1"
        s2 = _make_sample()
        s2.id = "s2"
        preds = [
            _make_pred(sample_id="s1"),
            _make_pred(sample_id="s2"),
        ]
        scores = judge.evaluate_all([s1, s2], preds)
        assert len(scores) == 2
        assert scores[0].prediction_id == "s1"
        assert scores[1].prediction_id == "s2"

    def test_evaluate_all_skips_missing(self):
        backend = MockLlmJudgeBackend()
        judge = LlmJudge(backend=backend)
        s1 = _make_sample()
        s1.id = "s1"
        s2 = _make_sample()
        s2.id = "s2"
        preds = [_make_pred(sample_id="s1")]  # no pred for s2
        scores = judge.evaluate_all([s1, s2], preds)
        assert len(scores) == 1


class TestJudgePrompt:
    def test_prompt_has_placeholders(self):
        assert "{cwe_id}" in JUDGE_PROMPT
        assert "{vulnerable_code}" in JUDGE_PROMPT
        assert "{patch_diff}" in JUDGE_PROMPT
        assert "{rationale}" in JUDGE_PROMPT

    def test_prompt_asks_for_json(self):
        assert "JSON" in JUDGE_PROMPT
        assert "explanation_quality" in JUDGE_PROMPT
        assert "patch_minimality" in JUDGE_PROMPT
