"""Unit tests for Stage 9 — serving backends (app/serving/backends.py).

Covers every line of backends.py:

* ``ServingBackend`` Protocol — structural checks
* ``LlamaCppBackend`` — constructor, lazy load, model_info, generate
  (including ImportError path and output format branching)
* ``OllamaBackend`` — constructor, lazy load, model_info, generate
  (including ImportError path, URL construction, response parsing)
* ``MockServingBackend`` — all branches

No real ML dependencies are required; all tests use mock objects or
verify error paths when the real package is absent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.serving.backends import (
    LlamaCppBackend,
    MockServingBackend,
    OllamaBackend,
    ServingBackend,
)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TestServingBackendProtocol:
    """The ServingBackend Protocol should accept duck-typed objects."""

    def test_protocol_runtime_checkable(self):
        """An object with generate + model_info should satisfy the Protocol."""
        class GoodBackend:
            def generate(self, prompt: str) -> str:
                return "{}"
            @property
            def model_info(self) -> dict:
                return {"backend": "good"}

        obj = GoodBackend()
        assert isinstance(obj, ServingBackend)

    def test_protocol_rejects_missing_method(self):
        """An object missing 'generate' should not satisfy the Protocol."""
        class BadBackend:
            @property
            def model_info(self) -> dict:
                return {"backend": "bad"}

        obj = BadBackend()
        assert not isinstance(obj, ServingBackend)

    def test_protocol_rejects_missing_property(self):
        """An object missing 'model_info' should not satisfy the Protocol."""
        class BadBackend:
            def generate(self, prompt: str) -> str:
                return "{}"

        obj = BadBackend()
        assert not isinstance(obj, ServingBackend)


# ---------------------------------------------------------------------------
# LlamaCppBackend
# ---------------------------------------------------------------------------


class TestLlamaCppBackendConstructor:
    def test_stores_all_parameters(self):
        backend = LlamaCppBackend(
            model_path="/path/to/model.gguf",
            num_ctx=8192,
            num_threads=8,
            n_gpu_layers=2,
            f16_kv=False,
            temperature=0.5,
            max_new_tokens=1024,
        )
        assert backend.model_path == "/path/to/model.gguf"
        assert backend.num_ctx == 8192
        assert backend.num_threads == 8
        assert backend.n_gpu_layers == 2
        assert backend.f16_kv is False
        assert backend.temperature == 0.5
        assert backend.max_new_tokens == 1024

    def test_defaults_match_serving_config(self):
        backend = LlamaCppBackend(model_path="model.gguf")
        assert backend.num_ctx == 4096
        assert backend.num_threads == 4
        assert backend.n_gpu_layers == 0
        assert backend.f16_kv is True
        assert backend.temperature == 0.2
        assert backend.max_new_tokens == 2048

    def test_llm_not_loaded_on_init(self):
        """The Llama class is NOT loaded until generate() is called."""
        backend = LlamaCppBackend(model_path="model.gguf")
        assert backend._llm is None


class TestLlamaCppBackendModelInfo:
    def test_model_info_contents(self):
        backend = LlamaCppBackend(
            model_path="/models/test.gguf",
            num_ctx=4096,
            num_threads=4,
            n_gpu_layers=0,
            f16_kv=True,
        )
        info = backend.model_info
        assert info["backend"] == "llama.cpp"
        assert info["model_path"] == "/models/test.gguf"
        assert info["num_ctx"] == 4096
        assert info["num_threads"] == 4
        assert info["n_gpu_layers"] == 0
        assert info["f16_kv"] is True


class TestLlamaCppBackendLazyLoad:
    def test_load_success(self):
        """_load should return the Llama class when llama_cpp is installed."""
        backend = LlamaCppBackend(model_path="model.gguf")
        # Mock llama_cpp
        mock_module = MagicMock()
        mock_module.Llama = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            result = backend._load()
        assert result is mock_module.Llama

    def test_load_import_error_raises_runtime_error(self):
        """_load should raise RuntimeError with helpful message when llama_cpp is not installed."""
        backend = LlamaCppBackend(model_path="model.gguf")
        with patch.dict("sys.modules", {"llama_cpp": None}):
            with pytest.raises(RuntimeError, match="llama-cpp-python is not installed"):
                backend._load()


class TestLlamaCppBackendGenerate:
    """LlamaCppBackend.generate — the Llama class is called to create an
    instance, then the instance is called to produce output. So the mock
    needs ``return_value.return_value`` for the generate output."""

    def test_generate_instantiates_llama_once(self):
        """generate() should call Llama constructor only on first call."""
        backend = LlamaCppBackend(model_path="model.gguf", num_ctx=2048, num_threads=2)

        mock_module = MagicMock()
        mock_llama = MagicMock()
        # Llama(...) returns an instance; instance(...) returns the output
        mock_llama.return_value.return_value = {"choices": [{"text": '{"cwe_id": "CWE-89"}'}]}
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            backend.generate("test prompt")
            backend.generate("second prompt")

        # Llama constructor should be called exactly once (lazy load).
        assert mock_llama.call_count == 1

    def test_generate_dict_output_with_choices(self):
        """When output is a dict with 'choices', extract text from choices[0]."""
        backend = LlamaCppBackend(model_path="model.gguf")
        mock_module = MagicMock()
        mock_llama = MagicMock()
        mock_llama.return_value.return_value = {
            "choices": [{"text": '{"cwe_id": "CWE-89", "severity": "high"}'}]
        }
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            result = backend.generate("prompt")
        assert '"cwe_id"' in result

    def test_generate_dict_output_with_generation(self):
        """When output is a dict without valid 'choices', fall back to 'generation'."""
        backend = LlamaCppBackend(model_path="model.gguf")
        mock_module = MagicMock()
        mock_llama = MagicMock()
        # choices is empty → falls to "generation" key
        mock_llama.return_value.return_value = {"generation": '{"cwe_id": "CWE-79"}'}
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            result = backend.generate("prompt")
        assert '"cwe_id"' in result

    def test_generate_dict_output_empty_choices(self):
        """When output has empty choices list, fall back to 'generation'."""
        backend = LlamaCppBackend(model_path="model.gguf")
        mock_module = MagicMock()
        mock_llama = MagicMock()
        # choices is empty list → not choices[0] is dict → fall to generation
        mock_llama.return_value.return_value = {"choices": [], "generation": "raw text"}
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            result = backend.generate("prompt")
        assert result == "raw text"

    def test_generate_dict_output_choices_not_dict(self):
        """When choices[0] is not a dict, fall back to 'generation'."""
        backend = LlamaCppBackend(model_path="model.gguf")
        mock_module = MagicMock()
        mock_llama = MagicMock()
        # choices[0] is a string, not a dict → fall to generation
        mock_llama.return_value.return_value = {
            "choices": ["not a dict"],
            "generation": "fallback text",
        }
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            result = backend.generate("prompt")
        assert result == "fallback text"

    def test_generate_str_output(self):
        """When output is a string, return it stripped."""
        backend = LlamaCppBackend(model_path="model.gguf")
        mock_module = MagicMock()
        mock_llama = MagicMock()
        mock_llama.return_value.return_value = '{"cwe_id": "CWE-22"}'
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            result = backend.generate("prompt")
        assert '"cwe_id"' in result

    def test_generate_other_type_output(self):
        """When output is neither dict nor str, convert to str."""
        backend = LlamaCppBackend(model_path="model.gguf")
        mock_module = MagicMock()
        mock_llama = MagicMock()
        mock_llama.return_value.return_value = 42  # an int
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            result = backend.generate("prompt")
        assert result == "42"

    def test_generate_uses_correct_parameters(self):
        """Llama constructor and generate call receive the right params."""
        backend = LlamaCppBackend(
            model_path="/model/test.gguf",
            num_ctx=8192,
            num_threads=8,
            n_gpu_layers=2,
            f16_kv=False,
            temperature=0.5,
            max_new_tokens=4096,
        )
        mock_module = MagicMock()
        mock_llama = MagicMock()
        mock_llama.return_value.return_value = {"choices": [{"text": "ok"}]}
        mock_module.Llama = mock_llama
        with patch.dict("sys.modules", {"llama_cpp": mock_module}):
            backend.generate("prompt")

        # Check constructor args
        _, kwargs = mock_llama.call_args
        assert kwargs["model_path"] == "/model/test.gguf"
        assert kwargs["n_ctx"] == 8192
        assert kwargs["n_threads"] == 8
        assert kwargs["n_gpu_layers"] == 2
        assert kwargs["f16_kv"] is False

        # Check generate args (called on the instance returned by Llama(...))
        llama_instance = mock_llama.return_value
        llama_instance.assert_called_once()
        _, gen_kwargs = llama_instance.call_args
        assert gen_kwargs["max_tokens"] == 4096
        assert gen_kwargs["temperature"] == 0.5
        assert gen_kwargs["echo"] is False


# ---------------------------------------------------------------------------
# OllamaBackend
# ---------------------------------------------------------------------------


class TestOllamaBackendConstructor:
    def test_stores_parameters(self):
        backend = OllamaBackend(
            model="qwen2.5-coder:7b",
            host="http://ollama.internal:11434",
            temperature=0.3,
            max_new_tokens=512,
            request_timeout=60.0,
        )
        assert backend.model == "qwen2.5-coder:7b"
        assert backend.host == "http://ollama.internal:11434"
        assert backend.temperature == 0.3
        assert backend.max_new_tokens == 512
        assert backend.request_timeout == 60.0

    def test_defaults(self):
        backend = OllamaBackend(model="qwen2.5-coder")
        assert backend.host == "http://localhost:11434"
        assert backend.temperature == 0.2
        assert backend.max_new_tokens == 2048
        assert backend.request_timeout == 30.0

    def test_client_not_created_on_init(self):
        backend = OllamaBackend(model="qwen2.5-coder")
        assert backend._client is None


class TestOllamaBackendModelInfo:
    def test_model_info_contents(self):
        backend = OllamaBackend(
            model="qwen2.5-coder:7b",
            host="http://localhost:11434",
            temperature=0.1,
            max_new_tokens=1024,
        )
        info = backend.model_info
        assert info["backend"] == "ollama"
        assert info["model"] == "qwen2.5-coder:7b"
        assert info["host"] == "http://localhost:11434"
        assert info["temperature"] == 0.1
        assert info["max_new_tokens"] == 1024


class TestOllamaBackendLazyLoad:
    def test_load_success(self):
        backend = OllamaBackend(model="test-model")
        mock_module = MagicMock()
        with patch.dict("sys.modules", {"httpx": mock_module}):
            result = backend._load()
        assert result is mock_module

    def test_load_import_error_raises_runtime_error(self):
        backend = OllamaBackend(model="test-model")
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(RuntimeError, match="httpx is not installed"):
                backend._load()


class TestOllamaBackendGenerate:
    def test_generate_returns_content_from_message(self):
        """generate should POST to /api/chat and extract message.content."""
        backend = OllamaBackend(model="qwen2.5-coder", request_timeout=10.0)
        mock_module = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model": "qwen2.5-coder",
            "message": {"role": "assistant", "content": '{"cwe_id": "CWE-89"}'},
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_module.Client.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_module}):
            result = backend.generate("test prompt")

        assert '{"cwe_id"' in result
        # Check the URL was constructed correctly
        mock_client.post.assert_called_once()
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "http://localhost:11434/api/chat"

        # Check the request body
        body = mock_client.post.call_args[1]["json"]
        assert body["model"] == "qwen2.5-coder"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "test prompt"
        assert body["options"]["temperature"] == 0.2
        assert body["options"]["num_predict"] == 2048
        assert body["stream"] is False

    def test_generate_returns_empty_string_for_non_dict_response(self):
        """If Ollama returns something unexpected, return empty string."""
        backend = OllamaBackend(model="qwen2.5-coder")
        mock_module = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = "not a dict"
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_module.Client.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_module}):
            result = backend.generate("test prompt")

        assert result == ""

    def test_generate_returns_empty_when_no_message(self):
        """If the response dict has no 'message' key, return empty string."""
        backend = OllamaBackend(model="qwen2.5-coder")
        mock_module = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {"other": "data"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_module.Client.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_module}):
            result = backend.generate("test prompt")

        assert result == ""

    def test_generate_returns_empty_when_message_not_dict(self):
        """If the message is not a dict, return empty string."""
        backend = OllamaBackend(model="qwen2.5-coder")
        mock_module = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": "not a dict"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_module.Client.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_module}):
            result = backend.generate("test prompt")

        assert result == ""

    def test_generate_with_custom_host(self):
        """The host URL should be used correctly with urljoin."""
        backend = OllamaBackend(
            model="test",
            host="http://my-ollama:11434",
        )
        mock_module = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_module.Client.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_module}):
            result = backend.generate("prompt")

        assert result == "ok"
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "http://my-ollama:11434/api/chat"

    def test_client_created_once(self):
        """The httpx Client should be created only on first generate call."""
        backend = OllamaBackend(model="test", request_timeout=5.0)
        mock_module = MagicMock()

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "ok"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_module.Client.return_value = mock_client
        with patch.dict("sys.modules", {"httpx": mock_module}):
            backend.generate("first")
            backend.generate("second")

        # Client constructor called once (lazy)
        assert mock_module.Client.call_count == 1
        # Post called twice
        assert mock_client.post.call_count == 2


# ---------------------------------------------------------------------------
# MockServingBackend
# ---------------------------------------------------------------------------


class TestMockServingBackend:
    DEFAULT_RESPONSE = (
        '{"cwe_id": "CWE-89", "severity": "high", '
        '"explanation": "SQL injection via string concatenation.", '
        '"patch_diff": ""}'
    )

    def test_returns_default_with_no_responses(self):
        """When no responses dict is provided, default is always returned."""
        backend = MockServingBackend()
        result = backend.generate("any prompt")
        assert result == self.DEFAULT_RESPONSE

    def test_returns_default_with_empty_dict(self):
        """Explicit empty dict should also return default."""
        backend = MockServingBackend(responses={})
        assert backend.generate("anything") == self.DEFAULT_RESPONSE

    def test_returns_custom_default(self):
        """A custom default response should be returned."""
        backend = MockServingBackend(default="custom response")
        assert backend.generate("anything") == "custom response"

    def test_returns_keyed_response(self):
        """When the prompt contains a key, return the matching response."""
        backend = MockServingBackend(
            responses={"CWE-79": '{"cwe_id": "CWE-79"}'},
            default="default",
        )
        assert backend.generate("Sample about CWE-79 vulnerability") == '{"cwe_id": "CWE-79"}'

    def test_keyed_response_takes_priority_over_default(self):
        """If both a key matches and a default exists, the keyed response wins."""
        backend = MockServingBackend(
            responses={"injected": "injected-response"},
            default="default-response",
        )
        assert backend.generate("this prompt has injected code") == "injected-response"

    def test_tracks_call_count(self):
        backend = MockServingBackend(default="x")
        backend.generate("a")
        backend.generate("b")
        backend.generate("c")
        assert backend.call_count == 3

    def test_records_calls_list(self):
        backend = MockServingBackend(default="x")
        backend.generate("hello prompt")
        backend.generate("another one")
        assert len(backend.calls) == 2
        assert "hello prompt" in backend.calls[0]
        assert "another one" in backend.calls[1]

    def test_truncates_long_prompts_in_calls(self):
        """Calls list should truncate prompts longer than 80 chars."""
        backend = MockServingBackend(default="x")
        long_prompt = "A" * 200
        backend.generate(long_prompt)
        assert len(backend.calls[0]) <= 83  # 80 chars + "..."

    def test_short_prompts_not_truncated(self):
        backend = MockServingBackend(default="x")
        short = "short prompt"
        backend.generate(short)
        assert backend.calls[0] == short

    def test_model_info_mock(self):
        backend = MockServingBackend(default="x")
        info = backend.model_info
        assert info["backend"] == "mock"
        assert info["model_path"] == "mock"

    def test_multiple_keys_first_match_wins(self):
        """When multiple keys match, the first in iteration order wins
        (dict ordering is insertion order in Python 3.7+)."""
        backend = MockServingBackend(
            responses={
                "first": "response-a",
                "second": "response-b",
            },
            default="default",
        )
        result = backend.generate("contains first and second")
        assert result == "response-a"

    def test_key_not_in_prompt_returns_default(self):
        backend = MockServingBackend(
            responses={"CWE-999": "special"},
            default="default",
        )
        result = backend.generate("this prompt has no matching key")
        assert result == "default"
