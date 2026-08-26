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

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

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


# --- QwenBackend._load ---


def test_qwen_backend_load_returns_cached_pipeline():
    """When _pipeline is already set, _load returns it without re-importing."""
    backend = QwenBackend()
    mock_pipe = MagicMock()
    backend._pipeline = mock_pipe

    result = backend._load()

    assert result is mock_pipe


def test_qwen_backend_load_raises_without_transformers():
    """When transformers is not installed, _load raises RuntimeError."""
    backend = QwenBackend()

    # patch.dict with None makes `from transformers import pipeline` raise ImportError
    with patch.dict(sys.modules, {"transformers": None}):
        with pytest.raises(RuntimeError, match="transformers is not installed"):
            backend._load()


def test_qwen_backend_load_creates_pipeline_when_available():
    """When transformers is importable, _load creates and caches the pipeline."""
    backend = QwenBackend(model_name="test/model", device="cpu")

    mock_pipe = MagicMock()
    mock_transformers = types.ModuleType("transformers")
    mock_transformers.pipeline = MagicMock(return_value=mock_pipe)

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        result = backend._load()

    assert result is mock_pipe
    assert backend._pipeline is mock_pipe
    mock_transformers.pipeline.assert_called_once_with(
        "text-generation",
        model="test/model",
        device_map="cpu",
    )


def test_qwen_backend_load_peft_adapter_with_base_model():
    """When adapter_config.json exists and base_model is set, the LoRA path
    loads the base model, applies the PEFT adapter, merges, and unloads."""
    backend = QwenBackend(
        model_name="/fake/adapter_dir",
        base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
        device="cpu",
    )

    mock_transformers = types.ModuleType("transformers")
    mock_peft = types.ModuleType("peft")

    mock_base_model = MagicMock()
    mock_merged_model = MagicMock()
    mock_transformers.AutoModelForCausalLM = MagicMock(return_value=mock_base_model)
    mock_transformers.AutoTokenizer = MagicMock(return_value=MagicMock())
    mock_peft.PeftModel = MagicMock()

    mock_pipe = MagicMock()
    mock_transformers.pipeline = MagicMock(return_value=mock_pipe)

    # Mock the PEFT adapter flow: from_pretrained → merge_and_unload → model
    mock_peft_model_obj = MagicMock()
    mock_peft_model_obj.merge_and_unload.return_value = mock_merged_model
    # Patch PeftModel.from_pretrained as a static method
    mock_peft.PeftModel.from_pretrained = MagicMock(return_value=mock_peft_model_obj)

    mock_transformers.AutoModelForCausalLM.from_pretrained = MagicMock(
        return_value=mock_base_model
    )
    mock_transformers.AutoTokenizer.from_pretrained = MagicMock(
        return_value=MagicMock()
    )

    with patch.dict(sys.modules, {"transformers": mock_transformers, "peft": mock_peft}), \
         patch("os.path.exists", return_value=True):
        result = backend._load()

    assert result is mock_pipe
    assert backend._pipeline is mock_pipe
    mock_peft.PeftModel.from_pretrained.assert_called_once()
    mock_peft_model_obj.merge_and_unload.assert_called_once()
    mock_merged_model.eval.assert_called_once()
    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once()


def test_qwen_backend_load_peft_adapter_without_base_model():
    """When adapter_config.json exists but base_model is None, falls to the
    full-checkpoint path (no PEFT merge)."""
    backend = QwenBackend(model_name="/fake/adapter_dir", base_model=None, device="cpu")

    mock_transformers = types.ModuleType("transformers")
    mock_pipe = MagicMock()
    mock_transformers.pipeline = MagicMock(return_value=mock_pipe)

    with patch.dict(sys.modules, {"transformers": mock_transformers}), \
         patch("os.path.exists", return_value=True):
        result = backend._load()

    # is_lora=True but base_model is None → goes to else branch (full model)
    assert result is mock_pipe
    # pipeline called with the model_name directly, no PEFT merge
    mock_transformers.pipeline.assert_called_once_with(
        "text-generation",
        model="/fake/adapter_dir",
        device_map="cpu",
    )


def test_qwen_backend_load_lora_fallback_to_base_model_when_weights_missing():
    """When adapter_config.json exists but adapter weights file is missing,
    _load() falls back to the base model instead of raising."""
    backend = QwenBackend(
        model_name="/fake/adapter_dir",
        base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device="cpu",
    )

    mock_transformers = types.ModuleType("transformers")
    mock_pipe = MagicMock()
    mock_transformers.pipeline = MagicMock(return_value=mock_pipe)

    # os.path.exists returns True for adapter_config.json but False for
    # adapter_model.safetensors / adapter_model.bin
    def mock_exists(path):
        if "adapter_config.json" in path:
            return True
        if "adapter_model.safetensors" in path:
            return False
        if "adapter_model.bin" in path:
            return False
        return False

    with patch.dict(sys.modules, {"transformers": mock_transformers}), \
         patch("os.path.exists", side_effect=mock_exists):
        result = backend._load()

    assert result is mock_pipe
    # Falls back to base_model, not model_name
    mock_transformers.pipeline.assert_called_once_with(
        "text-generation",
        model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device_map="cpu",
    )


# --- QwenBackend.generate ---


def test_qwen_backend_generate_returns_stripped_generated_text():
    """generate() strips whitespace from the pipeline's generated_text output."""
    backend = QwenBackend()
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "  hello world  "}]
    backend._pipeline = mock_pipe

    result = backend.generate("test prompt")

    assert result == "hello world"
    mock_pipe.assert_called_once()


def test_qwen_backend_generate_uses_text_key_fallback():
    """When generated_text key is absent, fall back to 'text' (older transformers)."""
    backend = QwenBackend()
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"text": "old format result"}]
    backend._pipeline = mock_pipe

    result = backend.generate("test prompt")

    assert result == "old format result"


def test_qwen_backend_generate_handles_missing_text_keys():
    """Entry with neither 'generated_text' nor 'text' defaults to empty string."""
    backend = QwenBackend()
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"unexpected": "data"}]
    backend._pipeline = mock_pipe

    result = backend.generate("test prompt")

    assert result == ""


def test_qwen_backend_generate_handles_empty_list_result():
    """An empty list falls through to the else branch: str(result)."""
    backend = QwenBackend()
    mock_pipe = MagicMock()
    mock_pipe.return_value = []
    backend._pipeline = mock_pipe

    result = backend.generate("test prompt")

    assert result == "[]"


def test_qwen_backend_generate_handles_non_list_result():
    """A non-list result is converted via str()."""
    backend = QwenBackend()
    mock_pipe = MagicMock()
    mock_pipe.return_value = "direct string result"
    backend._pipeline = mock_pipe

    result = backend.generate("test prompt")

    assert result == "direct string result"


def test_qwen_backend_generate_forwards_generation_params():
    """generate() forwards max_new_tokens, temperature, top_p to the pipeline."""
    backend = QwenBackend(max_new_tokens=1024, temperature=0.1, top_p=0.9)
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{"generated_text": "response"}]
    backend._pipeline = mock_pipe

    backend.generate("prompt here")

    call_kwargs = mock_pipe.call_args.kwargs
    assert call_kwargs["max_new_tokens"] == 1024
    assert call_kwargs["temperature"] == 0.1
    assert call_kwargs["top_p"] == 0.9
    assert call_kwargs["do_sample"] is True
    assert call_kwargs["return_full_text"] is False


# --- Protocol compatibility ---


def test_protocol_compatible():
    """Any object with a generate(prompt) -> str method satisfies ModelBackend."""

    class SimpleBackend:
        def generate(self, prompt: str) -> str:
            return (
                '{"cwe_id": "CWE-89", "severity": "high", "explanation": "test", "patch_diff": ""}'
            )

    backend: MockBackend = SimpleBackend()  # passes mypy/type checks structurally
    result = backend.generate("test prompt")
    assert '"cwe_id"' in result
