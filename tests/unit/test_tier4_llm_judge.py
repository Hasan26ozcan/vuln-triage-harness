"""Unit tests for Stage 6, Tier 4 — LLM-judge for explanation quality and patch minimality."""

import json
import logging
import sys
from unittest.mock import MagicMock, patch

from app.evaluation.tier4_llm_judge import (
    JUDGE_PROMPT,
    LlmJudge,
    LlmJudgeBackend,
    MockLlmJudgeBackend,
    _OpenAiBackend,
    _parse_judge_response,
)
from app.schemas.prediction_eval import LlmJudgeScore, ModelPrediction
from app.schemas.vuln import VulnSample


def _make_sample(cwe="CWE-89"):
    return VulnSample(
        id="gold_001",
        source="cve_real",
        repo_name="test/repo",
        cwe_id=cwe,
        severity="high",
        language="python",
        vulnerable_code="import os; os.system('rm -rf /')",
        description="Dangerous command injection",
    )


def _make_prediction(sample_id="gold_001"):
    return ModelPrediction(
        sample_id=sample_id,
        run_id="test_run",
        predicted_cwe="CWE-89",
        predicted_severity="high",
        suggested_patch_diff="--- a/app.py\n+++ b/app.py\n- old\n+ new",
        rationale="The vulnerable code uses os.system without sanitization.",
    )


# ---------------------------------------------------------------------------
# MockLlmJudgeBackend
# ---------------------------------------------------------------------------


class TestMockLlmJudgeBackend:
    """Tests for MockLlmJudgeBackend — deterministic fallback judge."""

    def test_invoke_returns_valid_json(self):
        """MockLlmJudgeBackend.invoke returns parseable JSON with all fields."""
        backend = MockLlmJudgeBackend()
        result = backend.invoke("test prompt", "mock-model")
        data = json.loads(result)
        assert "explanation_quality" in data
        assert "patch_minimality" in data
        assert "rationale" in data

    def test_invoke_uses_fallback_values(self):
        """Custom fallback values are reflected in the JSON response."""
        backend = MockLlmJudgeBackend(
            fallback_explanation_quality=0.9,
            fallback_patch_minimality=0.3,
        )
        result = backend.invoke("test prompt", "mock-model", max_tokens=100)
        data = json.loads(result)
        assert data["explanation_quality"] == 0.9
        assert data["patch_minimality"] == 0.3
        assert data["rationale"] == "mock-judge default response"


# ---------------------------------------------------------------------------
# _parse_judge_response
# ---------------------------------------------------------------------------


class TestParseJudgeResponse:
    """Tests for _parse_judge_response — parsing + fallback logic."""

    def test_valid_response_all_fields(self):
        """Valid JSON with all fields → parsed scores."""
        from app.evaluation.tier4_llm_judge import _parse_judge_response

        text = json.dumps(
            {
                "explanation_quality": 0.8,
                "patch_minimality": 0.6,
                "rationale": "good explanation",
            }
        )
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.8
        assert pm == 0.6
        assert rationale == "good explanation"

    def test_missing_fields_use_defaults(self):
        """Missing explanation_quality / patch_minimality → default 0.5."""
        from app.evaluation.tier4_llm_judge import _parse_judge_response

        text = json.dumps({"rationale": "minimal"})
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.5
        assert pm == 0.5
        assert rationale == "minimal"

    def test_values_clamped_to_0_1(self):
        """Values outside [0, 1] are clamped."""
        from app.evaluation.tier4_llm_judge import _parse_judge_response

        text = json.dumps(
            {
                "explanation_quality": 1.5,
                "patch_minimality": -0.3,
                "rationale": "clamped",
            }
        )
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 1.0
        assert pm == 0.0
        assert rationale == "clamped"

    def test_invalid_json_falls_back(self):
        """Malformed JSON → fallback (0.5, 0.5, text[:200])."""
        from app.evaluation.tier4_llm_judge import _parse_judge_response

        text = "not json at all"
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.5
        assert pm == 0.5
        assert rationale == text[:200]

    def test_json_decode_error_with_whitespace(self):
        """JSON with leading/trailing whitespace is stripped before parsing."""
        from app.evaluation.tier4_llm_judge import _parse_judge_response

        text = '  {"explanation_quality": 0.7, "patch_minimality": 0.4, "rationale": "ok"}  \n'
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.7
        assert pm == 0.4
        assert rationale == "ok"

    def test_type_error_fallback(self):
        """float() on a non-numeric type (e.g. list) → TypeError → fallback."""
        from app.evaluation.tier4_llm_judge import _parse_judge_response

        # float([1, 2]) raises TypeError, which is caught by the except clause
        text = json.dumps(
            {
                "explanation_quality": [1, 2],
                "patch_minimality": 0.5,
                "rationale": "x",
            }
        )
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.5
        assert pm == 0.5
        assert rationale == text[:200]

    def test_value_error_on_float_conversion(self):
        """JSON with a non-numeric explanation_quality string → ValueError → fallback."""
        from app.evaluation.tier4_llm_judge import _parse_judge_response

        text = json.dumps(
            {
                "explanation_quality": "not-a-number",
                "patch_minimality": 0.5,
                "rationale": "test",
            }
        )
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.5
        assert pm == 0.5
        # On fallback, rationale is the raw text truncated to 200 chars
        assert rationale == text[:200]


# ---------------------------------------------------------------------------
# LlmJudge — backend selection
# ---------------------------------------------------------------------------


class TestLlmJudgeBackendSelection:
    """Tests for LlmJudge.__init__ backend selection logic."""

    def test_injected_backend_used(self):
        """When backend is explicitly provided, it's stored directly."""
        mock_backend = MagicMock(spec=LlmJudgeBackend)
        judge = LlmJudge(backend=mock_backend)
        assert judge._backend is mock_backend

    def test_no_api_key_uses_mock_backend(self):
        """Without OPENAI_API_KEY, _build_default_backend returns MockLlmJudgeBackend."""
        with patch.dict("os.environ", {}, clear=False):
            import os

            env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            with patch.dict("os.environ", env_without_key, clear=True):
                judge = LlmJudge()
        assert isinstance(judge._backend, MockLlmJudgeBackend)

    def test_model_from_env_var(self):
        """JUDGE_MODEL env var sets the default model name."""
        with patch.dict("os.environ", {"JUDGE_MODEL": "custom-judge-model"}, clear=False):
            # Need to also clear OPENAI_API_KEY so mock backend is used
            import os

            env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            env["JUDGE_MODEL"] = "custom-judge-model"
            with patch.dict("os.environ", env, clear=True):
                judge = LlmJudge()
        assert judge._model == "custom-judge-model"

    def test_explicit_model_overrides_env(self):
        """Explicit model parameter overrides JUDGE_MODEL env var."""
        with patch.dict("os.environ", {"JUDGE_MODEL": "env-model"}, clear=False):
            mock_backend = MagicMock(spec=LlmJudgeBackend)
            judge = LlmJudge(backend=mock_backend, model="explicit-model")
        assert judge._model == "explicit-model"


# ---------------------------------------------------------------------------
# LlmJudge._build_default_backend
# ---------------------------------------------------------------------------


class TestBuildDefaultBackend:
    """Tests for LlmJudge._build_default_backend — env-var driven construction."""

    def test_no_api_key_returns_mock(self):
        """No OPENAI_API_KEY → MockLlmJudgeBackend."""
        import os

        env_without_key = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        with patch.dict("os.environ", env_without_key, clear=True):
            backend = LlmJudge._build_default_backend()
        assert isinstance(backend, MockLlmJudgeBackend)

    def test_with_api_key_returns_openai_backend(self):
        """OPENAI_API_KEY set → _OpenAiBackend constructed."""
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://custom.openai.com/v1",
            },
            clear=False,
        ):
            backend = LlmJudge._build_default_backend()
        assert isinstance(backend, _OpenAiBackend)
        assert backend.api_key == "test-key"
        assert backend.base_url == "https://custom.openai.com/v1"

    def test_with_api_key_default_base_url(self):
        """OPENAI_API_KEY set but no OPENAI_BASE_URL → default OpenAI URL."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False):
            backend = LlmJudge._build_default_backend()
        assert isinstance(backend, _OpenAiBackend)
        assert backend.base_url == "https://api.openai.com/v1"


# ---------------------------------------------------------------------------
# LlmJudge.invoke — event-loop branching
# ---------------------------------------------------------------------------


class TestLlmJudgeInvoke:
    """Tests for LlmJudge.invoke — both event-loop branches."""

    def test_invoke_loop_not_running(self):
        """Line 145: invoke when no event loop is running."""
        mock_backend = MagicMock(spec=LlmJudgeBackend)
        mock_response = json.dumps(
            {
                "explanation_quality": 0.9,
                "patch_minimality": 0.8,
                "rationale": "test",
            }
        )
        mock_backend.invoke.return_value = mock_response
        judge = LlmJudge(backend=mock_backend, model="test-model")
        with patch("app.evaluation.tier4_llm_judge.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.is_running.return_value = False
            result = judge.invoke("prompt")
        assert "explanation_quality" in result
        mock_backend.invoke.assert_called_once_with("prompt", "test-model", 512)

    def test_invoke_loop_running(self):
        """Line 144: invoke when an event loop is already running."""
        mock_backend = MagicMock(spec=LlmJudgeBackend)
        mock_response = json.dumps(
            {
                "explanation_quality": 0.9,
                "patch_minimality": 0.8,
                "rationale": "test",
            }
        )
        mock_backend.invoke.return_value = mock_response
        judge = LlmJudge(backend=mock_backend, model="test-model")
        with patch("app.evaluation.tier4_llm_judge.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.is_running.return_value = True
            result = judge.invoke("prompt")
        assert "explanation_quality" in result
        mock_backend.invoke.assert_called_once_with("prompt", "test-model", 512)


# ---------------------------------------------------------------------------
# LlmJudge.evaluate
# ---------------------------------------------------------------------------


class TestLlmJudgeEvaluate:
    """Tests for LlmJudge.evaluate — full judge flow."""

    def test_evaluate_with_mock_backend(self):
        """Evaluate formats the prompt, invokes backend, parses response."""
        backend = MockLlmJudgeBackend(
            fallback_explanation_quality=0.8,
            fallback_patch_minimality=0.6,
        )
        judge = LlmJudge(backend=backend, model="mock-judge")
        sample = _make_sample()
        pred = _make_prediction()
        score = judge.evaluate(sample, pred)
        assert isinstance(score, LlmJudgeScore)
        assert score.prediction_id == "gold_001"
        assert score.explanation_quality == 0.8
        assert score.patch_minimality == 0.6
        assert score.evaluator_model == "mock-judge"

    def test_evaluate_prompt_contains_sample_fields(self):
        """The judge prompt is formatted with sample CWE, description, etc."""
        mock_backend = MagicMock(spec=LlmJudgeBackend)
        mock_backend.invoke.return_value = json.dumps(
            {
                "explanation_quality": 0.7,
                "patch_minimality": 0.5,
                "rationale": "parsed",
            }
        )
        judge = LlmJudge(backend=mock_backend, model="judge")
        sample = _make_sample(cwe="CWE-79")
        pred = _make_prediction()
        judge.evaluate(sample, pred)
        # Verify the prompt passed to the backend includes sample fields
        called_prompt = mock_backend.invoke.call_args[0][0]
        assert "CWE-79" in called_prompt
        assert "Dangerous command injection" in called_prompt
        assert sample.vulnerable_code in called_prompt
        assert pred.suggested_patch_diff in called_prompt
        assert pred.rationale in called_prompt


# ---------------------------------------------------------------------------
# LlmJudge.evaluate_all
# ---------------------------------------------------------------------------


class TestLlmJudgeEvaluateAll:
    """Tests for LlmJudge.evaluate_all — batch scoring."""

    def test_evaluate_all_matched_predictions(self):
        """All samples have matching predictions → all scored."""
        backend = MockLlmJudgeBackend(
            fallback_explanation_quality=0.75,
            fallback_patch_minimality=0.75,
        )
        judge = LlmJudge(backend=backend, model="mock")
        samples = [
            _make_sample(cwe="CWE-89"),
            _make_sample(cwe="CWE-79"),
        ]
        samples[1].id = "gold_002"
        predictions = [
            _make_prediction(sample_id="gold_001"),
            _make_prediction(sample_id="gold_002"),
        ]
        scores = judge.evaluate_all(samples, predictions)
        assert len(scores) == 2
        assert all(s.prediction_id in ("gold_001", "gold_002") for s in scores)

    def test_evaluate_all_skips_missing_predictions(self):
        """Samples without a matching prediction are skipped with a warning."""
        backend = MockLlmJudgeBackend()
        judge = LlmJudge(backend=backend, model="mock")
        samples = [
            _make_sample(cwe="CWE-89"),
            _make_sample(cwe="CWE-79"),
        ]
        samples[1].id = "gold_002"
        # Only one prediction for gold_001; gold_002 is missing
        predictions = [_make_prediction(sample_id="gold_001")]
        with patch.object(logging.getLogger("app.evaluation.tier4_llm_judge"), "warning"):
            scores = judge.evaluate_all(samples, predictions)
        assert len(scores) == 1
        assert scores[0].prediction_id == "gold_001"

    def test_evaluate_all_empty_inputs(self):
        """Empty sample list → empty results."""
        backend = MockLlmJudgeBackend()
        judge = LlmJudge(backend=backend, model="mock")
        scores = judge.evaluate_all([], [])
        assert scores == []


# ---------------------------------------------------------------------------
# _OpenAiBackend
# ---------------------------------------------------------------------------


class TestOpenAiBackendInit:
    """Tests for _OpenAiBackend.__init__ — import success and failure paths."""

    def test_init_with_openai_available(self):
        """Lines 223-227: openai import succeeds → client is created."""
        mock_openai = MagicMock()
        mock_client_instance = MagicMock()
        mock_openai.OpenAI.return_value = mock_client_instance
        with patch.dict(sys.modules, {"openai": mock_openai}):
            backend = _OpenAiBackend(api_key="key-123", base_url="https://api.openai.com/v1")
        assert backend.api_key == "key-123"
        assert backend.base_url == "https://api.openai.com/v1"
        assert backend._client is mock_client_instance
        mock_openai.OpenAI.assert_called_once_with(
            base_url="https://api.openai.com/v1", api_key="key-123"
        )

    def test_init_with_openai_missing(self):
        """Lines 228-233: openai import fails → _client is None, error logged."""
        with patch.dict(sys.modules, {"openai": None}):
            backend = _OpenAiBackend(api_key="key-123", base_url="https://test.com/v1")
        assert backend.api_key == "key-123"
        assert backend.base_url == "https://test.com/v1"
        assert backend._client is None


class TestOpenAiBackendInvoke:
    """Tests for _OpenAiBackend.invoke — fallback vs real call paths."""

    def test_invoke_when_client_is_none_falls_back(self):
        """Lines 236-237: when _client is None, fall back to MockLlmJudgeBackend."""
        backend = _OpenAiBackend(api_key="key", base_url="https://test.com")
        # _client is None because openai isn't available
        result = backend.invoke("test prompt", "gpt-4o-mini")
        data = json.loads(result)
        assert "explanation_quality" in data
        assert "patch_minimality" in data
        assert "rationale" in data

    def test_invoke_with_mock_client_returns_content(self):
        """Lines 238-244: when _client is set, the real API call path is taken."""
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            '  {"explanation_quality": 0.9, "patch_minimality": 0.7, "rationale": "good"}  '
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict(sys.modules, {"openai": mock_openai}):
            backend = _OpenAiBackend(api_key="key", base_url="https://test.com")
        result = backend.invoke("prompt text", "gpt-4o-mini", max_tokens=256)
        assert "explanation_quality" in result
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "prompt text"}],
            max_tokens=256,
            temperature=0.2,
        )


# ---------------------------------------------------------------------------
# JUDGE_PROMPT
# ---------------------------------------------------------------------------


class TestJudgePrompt:
    """Tests for the JUDGE_PROMPT template."""

    def test_prompt_contains_cwe_id(self):
        """The prompt template includes the CWE ID placeholder."""
        sample = _make_sample(cwe="CWE-78")
        pred = _make_prediction()
        prompt = JUDGE_PROMPT.format(
            cwe_id=sample.cwe_id,
            description=sample.description,
            language=sample.language,
            vulnerable_code=sample.vulnerable_code,
            patch_diff=pred.suggested_patch_diff,
            rationale=pred.rationale,
        )
        assert "CWE-78" in prompt

    def test_prompt_contains_json_response_format(self):
        """The prompt template asks for JSON response format."""
        prompt = JUDGE_PROMPT.format(
            cwe_id="CWE-89",
            description="test",
            language="python",
            vulnerable_code="code",
            patch_diff="diff",
            rationale="rationale",
        )
        assert '"explanation_quality"' in prompt
        assert '"patch_minimality"' in prompt
        assert '"rationale"' in prompt


# ---------------------------------------------------------------------------
# _parse_judge_response — brace-matching fallback (lines 168-176)
# ---------------------------------------------------------------------------


class TestParseJudgeResponseBraceFallback:
    """When json.loads fails on the whole text, _find_json_objects is used."""

    def test_preamble_then_valid_json_object(self):
        """Text with preamble + a valid JSON object inside braces.

        json.loads on the whole text fails (due to preamble), but
        _find_json_objects finds the {…} candidate and parses it successfully.
        """
        text = (
            'Some preamble text '
            '{"explanation_quality": 0.75, "patch_minimality": 0.65, "rationale": "okay"}'
            ' trailing text'
        )
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.75
        assert pm == 0.65
        assert rationale == "okay"

    def test_multiple_json_objects_first_valid_wins(self):
        """If _find_json_objects returns multiple candidates, the first
        valid one is used."""
        text = (
            '{"bad": content} {"explanation_quality": 0.8, '
            '"patch_minimality": 0.4, "rationale": "good"}'
        )
        eq, pm, rationale = _parse_judge_response(text)
        # First candidate {"bad": content} fails to parse → second one used
        assert eq == 0.8
        assert pm == 0.4
        assert rationale == "good"


# ---------------------------------------------------------------------------
# _parse_judge_response — regex-based fallback (lines 180-204)
# ---------------------------------------------------------------------------


class TestParseJudgeResponseRegexFallback:
    """When brace-matching also fails, regex extracts scores from text."""

    def test_regex_fallback_no_braces(self):
        """Text without braces — regex extracts scores directly."""
        text = "explanation_quality: 0.9, patch_minimality: 0.6, rationale: nice work"
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.9
        assert pm == 0.6
        assert "nice work" in rationale

    def test_regex_fallback_with_quotes(self):
        """Scores with quoted keys — regex still matches."""
        text = '{"explanation_quality": 0.7, "patch_minimality": 0.3, "rationale": "ok"'
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.7
        assert pm == 0.3

    def test_regex_fallback_no_score_fields(self):
        """Text with no score fields at all → final fallback (0.5, 0.5, text[:200])."""
        text = "just some random text without any scores"
        eq, pm, rationale = _parse_judge_response(text)
        assert eq == 0.5
        assert pm == 0.5
        assert rationale == text[:200]


# ---------------------------------------------------------------------------
# LocalLlmJudgeBackend (lines 112-136)
# ---------------------------------------------------------------------------


class TestLocalLlmJudgeBackend:
    """Tests for LocalLlmJudgeBackend — __post_init__ and invoke."""

    def test_post_init_calls_model_eval_and_sets_use_cache(self):
        """__post_init__ calls model.eval() and sets config.use_cache = True."""
        from app.evaluation.tier4_llm_judge import LocalLlmJudgeBackend

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        LocalLlmJudgeBackend(model=mock_model, tokenizer=mock_tokenizer)

        mock_model.eval.assert_called_once()
        assert mock_model.config.use_cache is True

    def test_invoke_with_mocked_torch(self):
        """invoke() imports torch, tokenizes, runs generate, decodes."""
        from app.evaluation.tier4_llm_judge import LocalLlmJudgeBackend

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_torch = MagicMock()

        # Set up tokenizer to return a dict with input_ids that has a shape
        input_ids_tensor = MagicMock()
        input_ids_tensor.shape = (1, 5)  # batch_size=1, seq_len=5
        input_ids_tensor.to.return_value = input_ids_tensor

        mock_tokenizer.return_value = {
            "input_ids": input_ids_tensor,
            "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock())),
        }
        mock_tokenizer.decode.return_value = "  judged response  "
        mock_tokenizer.pad_token_id = 0

        # Set up model.generate to return output with sliceable batch
        mock_output = MagicMock()
        mock_output.__getitem__ = MagicMock(return_value="generated_ids")
        mock_model.generate.return_value = mock_output

        # Mock next(self.model.parameters()).device
        mock_param = MagicMock()
        mock_param.device = "cpu"
        mock_model.parameters = MagicMock(return_value=iter([mock_param]))

        backend = LocalLlmJudgeBackend(model=mock_model, tokenizer=mock_tokenizer)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = backend.invoke("some prompt", "test-model", max_tokens=256)

        assert result == "judged response"
        mock_torch.no_grad.assert_called_once()
        mock_model.generate.assert_called_once()
        mock_tokenizer.decode.assert_called_once()
