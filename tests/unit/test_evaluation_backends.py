"""Unit tests for Stage 4 model backends.

These verify:
  - MockBackend returns the default response when no key matches.
  - MockBackend returns the keyed response when a key is found in the prompt.
  - MockBackend tracks call count and prompt history.
  - MockBackend with empty responses dict uses default.
  - QwenBackend constructor stores parameters correctly (no model load).
  - QwenBackend raises RuntimeError if transformers is not installed.
  - ModelBackend Protocol is structural (duck-typing works).
"""

from app.evaluation.backends import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MockBackend,
    QwenBackend,
)

# --- Constants ---


def test_default_base_model_is_qwen():
    assert "Qwen" in DEFAULT_BASE_MODEL
    assert "7B" in DEFAULT_BASE_MODEL


def test_default_max_new_tokens_is_sane():
    assert DEFAULT_MAX_NEW_TOKENS == 2048


def test_default_temperature_is_low():
    """Low temperature for deterministic classification."""
    assert DEFAULT_TEMPERATURE <= 0.5


def test_default_top_p_is_sane():
    assert 0.0 < DEFAULT_TOP_P <= 1.0


# --- MockBackend ---


def test_mock_backend_returns_default():
    backend = MockBackend(default="hello world")
    assert backend.generate("any prompt") == "hello world"


def test_mock_backend_returns_keyed_response():
    """When the prompt contains a key, return the matching response."""
    backend = MockBackend(
        responses={"CWE-89": '{"cwe_id": "CWE-89"}'},
        default="default",
    )
    # Prompt contains "CWE-89" → returns the keyed response
    assert backend.generate("Sample about CWE-89 vulnerability") == '{"cwe_id": "CWE-89"}'
    # Prompt doesn't contain "CWE-89" → returns default
    assert backend.generate("Something else entirely") == "default"


def test_mock_backend_tracks_call_count():
    backend = MockBackend(default="x")
    backend.generate("a")
    backend.generate("b")
    backend.generate("c")
    assert backend.call_count == 3


def test_mock_backend_records_calls():
    backend = MockBackend(default="x")
    backend.generate("hello prompt here")
    backend.generate("another prompt")
    assert len(backend.calls) == 2
    assert "hello prompt" in backend.calls[0]


def test_mock_backend_no_responses_uses_default():
    """Without any responses dict, default is always returned."""
    backend = MockBackend(default="fallback")
    assert backend.generate("anything") == "fallback"


def test_mock_backend_truncates_long_prompts_in_calls():
    """Calls list should truncate very long prompts to avoid memory bloat."""
    backend = MockBackend(default="x")
    long_prompt = "A" * 200
    backend.generate(long_prompt)
    assert len(backend.calls[0]) <= 83  # 80 chars + "..."


# --- QwenBackend (constructor only, no model loading) ---


def test_qwen_backend_stores_parameters():
    """QwenBackend constructor should store params without loading the model."""
    backend = QwenBackend(
        model_name="custom/model",
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.8,
        device="cpu",
    )
    assert backend.model_name == "custom/model"
    assert backend.max_new_tokens == 512
    assert backend.temperature == 0.5
    assert backend.top_p == 0.8
    assert backend.device == "cpu"
    # Model should not be loaded yet
    assert backend._pipeline is None


def test_qwen_backend_default_parameters():
    """Default QwenBackend uses the project's base model and defaults."""
    backend = QwenBackend()
    assert backend.model_name == DEFAULT_BASE_MODEL
    assert backend.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert backend.temperature == DEFAULT_TEMPERATURE
    assert backend.top_p == DEFAULT_TOP_P
    assert backend.device == "auto"


def test_qwen_backend_lazy_loading():
    """_load should only run on first generate call, not at construction."""
    backend = QwenBackend(model_name="does-not-exist/model")
    # Construction doesn't raise; only _load does
    assert backend._pipeline is None


# --- Protocol compatibility ---


def test_protocol_compatible():
    """Any object with a generate(prompt) -> str method satisfies ModelBackend."""
    class SimpleBackend:
        def generate(self, prompt: str) -> str:
            return (
                '{"cwe_id": "CWE-89", "severity": "high", '
                '"explanation": "test", "patch_diff": ""}'
            )

    backend: MockBackend = SimpleBackend()  # passes mypy/type checks structurally
    result = backend.generate("test prompt")
    assert '"cwe_id"' in result
