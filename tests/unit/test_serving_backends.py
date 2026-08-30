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
    LlamaServerBackend,
    MockServingBackend,
    OllamaBackend,
    ServingBackend,
    TransformersBackend,
    _find_hf_model_dir,
    _find_llama_server,
    _import_httpx,
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


# ---------------------------------------------------------------------------
# LlamaServerBackend
# ---------------------------------------------------------------------------


class TestLlamaServerBackendConstructor:
    """Tests for LlamaServerBackend.__init__."""

    def test_stores_all_parameters(self):
        """All constructor parameters are stored as instance attributes."""
        backend = LlamaServerBackend(
            model_path="/models/test.gguf",
            server_binary="/bin/llama-server",
            host="0.0.0.0",
            port=9090,
            num_threads=8,
            num_ctx=8192,
            temperature=0.5,
            max_new_tokens=4096,
            request_timeout=60.0,
        )
        assert backend.model_path == "/models/test.gguf"
        assert backend.server_binary == "/bin/llama-server"
        assert backend.host == "0.0.0.0"
        assert backend.port == 9090
        assert backend.num_threads == 8
        assert backend.num_ctx == 8192
        assert backend.temperature == 0.5
        assert backend.max_new_tokens == 4096
        assert backend.request_timeout == 60.0
        assert backend._process is None
        assert backend._client is None

    def test_defaults(self):
        """Default values match the documented defaults."""
        backend = LlamaServerBackend(model_path="/models/test.gguf")
        assert backend.host == "127.0.0.1"
        assert backend.port == 8080
        assert backend.num_threads == 4
        assert backend.num_ctx == 4096
        assert backend.temperature == 0.2
        assert backend.max_new_tokens == 2048
        assert backend.request_timeout == 30.0

    def test_server_binary_defaults_to_find(self):
        """When server_binary is None, _find_llama_server() is called."""
        with patch("app.serving.backends._find_llama_server", return_value="/found/llama-server"):
            backend = LlamaServerBackend(model_path="/model.gguf")
        assert backend.server_binary == "/found/llama-server"


class TestLlamaServerBackendModelInfo:
    def test_model_info_contents(self):
        backend = LlamaServerBackend(
            model_path="/models/test.gguf",
            server_binary="/bin/llama-server",
            host="127.0.0.1",
            port=8080,
            num_threads=4,
            num_ctx=4096,
            temperature=0.2,
            max_new_tokens=2048,
        )
        info = backend.model_info
        assert info["backend"] == "llama-server"
        assert info["model_path"] == "/models/test.gguf"
        assert info["server_binary"] == "/bin/llama-server"
        assert info["host"] == "127.0.0.1"
        assert info["port"] == 8080
        assert info["num_threads"] == 4
        assert info["num_ctx"] == 4096
        assert info["temperature"] == 0.2
        assert info["max_new_tokens"] == 2048


class TestLlamaServerBackendEnsureRunning:
    """Tests for _ensure_running — subprocess management and health checks."""

    def test_process_already_running_skips(self):
        """When _process is not None, _ensure_running returns immediately."""
        backend = LlamaServerBackend(
            model_path="/model.gguf",
            server_binary="/bin/llama-server",
        )
        backend._process = MagicMock()
        # Should not call _import_httpx or try subprocess
        with patch("app.serving.backends._import_httpx") as mock_import:
            backend._ensure_running()
            mock_import.assert_not_called()

    def test_binary_not_found_raises(self):
        """When server_binary path doesn't exist, RuntimeError is raised."""
        backend = LlamaServerBackend(
            model_path="/model.gguf",
            server_binary="/nonexistent/binary",
        )
        with patch("os.path.exists", return_value=False):
            with pytest.raises(RuntimeError, match="llama-server binary not found"):
                backend._ensure_running()

    def test_model_not_found_raises(self):
        """When model_path doesn't exist, RuntimeError is raised."""
        backend = LlamaServerBackend(
            model_path="/nonexistent/model.gguf",
            server_binary="/bin/llama-server",
        )
        with patch("os.path.exists", side_effect=[True, False]):  # binary exists, model doesn't
            with pytest.raises(RuntimeError, match="GGUF model not found"):
                backend._ensure_running()

    def test_server_ready_after_health_check(self):
        """When the server becomes healthy within 60s, _ensure_running returns."""
        backend = LlamaServerBackend(
            model_path="/model.gguf",
            server_binary="/bin/llama-server",
            host="127.0.0.1",
            port=8080,
        )

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process still running
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client

        with (
            patch("os.path.exists", return_value=True),
            patch("app.serving.backends._import_httpx", return_value=mock_httpx),
            patch("subprocess.Popen", return_value=mock_process),
            patch("time.sleep"),
        ):
            backend._ensure_running()

        assert backend._process is mock_process
        assert backend._client is mock_client
        mock_client.get.assert_called_once_with("http://127.0.0.1:8080/health")

    def test_server_exits_early_raises(self):
        """When the subprocess exits before health check passes, RuntimeError."""
        backend = LlamaServerBackend(
            model_path="/model.gguf",
            server_binary="/bin/llama-server",
        )

        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # process exited
        mock_process.returncode = 1
        mock_process.stderr = MagicMock()
        mock_stderr = MagicMock()
        mock_stderr.read.return_value = b"error output"
        mock_process.stderr = mock_stderr

        mock_httpx = MagicMock()

        with (
            patch("os.path.exists", return_value=True),
            patch("app.serving.backends._import_httpx", return_value=mock_httpx),
            patch("subprocess.Popen", return_value=mock_process),
            patch("time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="llama-server exited early"):
                backend._ensure_running()

    def test_health_check_timeout_raises(self):
        """When the server never becomes healthy within 60 iterations, RuntimeError."""
        backend = LlamaServerBackend(
            model_path="/model.gguf",
            server_binary="/bin/llama-server",
        )

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # still running but never healthy

        mock_response = MagicMock()
        mock_response.status_code = 503  # health check fails

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client

        with (
            patch("os.path.exists", return_value=True),
            patch("app.serving.backends._import_httpx", return_value=mock_httpx),
            patch("subprocess.Popen", return_value=mock_process),
            patch("time.sleep"),
        ):
            with pytest.raises(RuntimeError, match="did not become healthy"):
                backend._ensure_running()

    def test_health_check_connection_error_continues_polling(self):
        """When health check raises an exception, polling continues (not crash)."""
        backend = LlamaServerBackend(
            model_path="/model.gguf",
            server_binary="/bin/llama-server",
        )

        mock_process = MagicMock()
        mock_process.poll.return_value = None

        mock_client = MagicMock()
        # First call: connection error, second: healthy
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.get.side_effect = [ConnectionError("refused"), mock_response]

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client

        with (
            patch("os.path.exists", return_value=True),
            patch("app.serving.backends._import_httpx", return_value=mock_httpx),
            patch("subprocess.Popen", return_value=mock_process),
            patch("time.sleep"),
        ):
            backend._ensure_running()

        # Should have made 2 GET calls (first failed, second succeeded)
        assert mock_client.get.call_count == 2


class TestLlamaServerBackendGenerate:
    """Tests for LlamaServerBackend.generate."""

    def _make_backend(self):
        """Create a backend with _process already set to bypass _ensure_running."""
        backend = LlamaServerBackend(
            model_path="/model.gguf",
            server_binary="/bin/llama-server",
            host="127.0.0.1",
            port=8080,
        )
        backend._process = MagicMock()  # bypass _ensure_running
        backend._client = None
        return backend

    def test_generate_posts_completion_request(self):
        """generate sends POST /completion with the right body."""
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": '{"cwe_id": "CWE-89"}'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        backend._client = mock_client

        result = backend.generate("test prompt")

        assert '{"cwe_id"' in result
        mock_client.post.assert_called_once()
        called_url = mock_client.post.call_args[0][0]
        assert called_url == "http://127.0.0.1:8080/completion"

        body = mock_client.post.call_args[1]["json"]
        assert body["prompt"] == "test prompt"
        assert body["n_predict"] == 2048
        assert body["temperature"] == 0.2
        assert body["stream"] is False

    def test_generate_with_custom_params(self):
        """generate forwards max_new_tokens and temperature from config."""
        backend = self._make_backend()
        backend.max_new_tokens = 512
        backend.temperature = 0.7

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": "ok"}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        backend._client = mock_client

        backend.generate("prompt")

        body = mock_client.post.call_args[1]["json"]
        assert body["n_predict"] == 512
        assert body["temperature"] == 0.7

    def test_generate_openai_fallback(self):
        """When 'content' is empty, fall back to choices[0].text."""
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "content": "",
            "choices": [{"text": "fallback response"}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        backend._client = mock_client

        result = backend.generate("prompt")
        assert result == "fallback response"

    def test_generate_strips_whitespace(self):
        """Generated content is stripped of surrounding whitespace."""
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": "  some response  "}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        backend._client = mock_client

        result = backend.generate("prompt")
        assert result == "some response"

    def test_generate_empty_content_no_choices(self):
        """When content is empty and no choices, return empty string."""
        backend = self._make_backend()
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": ""}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        backend._client = mock_client

        result = backend.generate("prompt")
        assert result == ""


class TestLlamaServerBackendClose:
    """Tests for LlamaServerBackend.close."""

    def test_close_terminates_process(self):
        """close() terminates the subprocess and sets _process to None."""
        backend = LlamaServerBackend(model_path="/m.gguf", server_binary="/s")
        mock_process = MagicMock()
        mock_process.wait.return_value = 0
        backend._process = mock_process
        backend._client = None

        backend.close()

        mock_process.terminate.assert_called_once()
        assert backend._process is None

    def test_close_kills_on_timeout(self):
        """If wait() raises (timeout), kill() is called instead."""
        backend = LlamaServerBackend(model_path="/m.gguf", server_binary="/s")
        mock_process = MagicMock()
        import subprocess as sp

        mock_process.wait.side_effect = sp.TimeoutExpired(cmd="llama-server", timeout=10)
        backend._process = mock_process
        backend._client = None

        backend.close()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_close_closes_client(self):
        """close() closes the httpx client if one exists."""
        backend = LlamaServerBackend(model_path="/m.gguf", server_binary="/s")
        backend._process = None
        mock_client = MagicMock()
        backend._client = mock_client

        backend.close()

        mock_client.close.assert_called_once()
        assert backend._client is None

    def test_close_with_no_process_no_client(self):
        """close() with no process and no client is a no-op."""
        backend = LlamaServerBackend(model_path="/m.gguf", server_binary="/s")
        # Should not raise
        backend.close()


class TestFindLlamaServer:
    """Tests for the module-level _find_llama_server() function."""

    def test_finds_binary_in_tools_directory(self):
        """Looks in repo_root/tools/llama-cpp/ for the binary and returns it
        without calling shutil.which."""
        with patch("os.path.exists", return_value=True), patch("shutil.which") as mock_which:
            result = _find_llama_server()
            # os.path.exists returns True for any path → first candidate found
            assert result is not None
            # Since the candidate was found, shutil.which should not be called
            mock_which.assert_not_called()

    def test_falls_back_to_shutil_which(self):
        """When not in tools dir, falls back to shutil.which."""
        with (
            patch("os.path.exists", return_value=False),
            patch("shutil.which", return_value="/usr/local/bin/llama-server"),
        ):
            result = _find_llama_server()
            assert result == "/usr/local/bin/llama-server"

    def test_falls_back_to_which_executable(self):
        """When .exe not found, tries 'llama-server' (no extension)."""
        with patch("os.path.exists", return_value=False), patch("shutil.which") as mock_which:
            # First which("llama-server") returns the path
            def _side_effect(cmd):
                return "/usr/bin/llama-server" if cmd == "llama-server" else None

            mock_which.side_effect = _side_effect
            result = _find_llama_server()
            assert result == "/usr/bin/llama-server"

    def test_returns_none_when_not_found_anywhere(self):
        """When neither tools dir nor PATH has the binary, returns None."""
        with patch("os.path.exists", return_value=False), patch("shutil.which", return_value=None):
            result = _find_llama_server()
            assert result is None


class TestImportHxtt:
    """Tests for the module-level _import_httpx() function."""

    def test_returns_httpx_when_available(self):
        """Returns the httpx module when it's importable."""
        mock_httpx = MagicMock()
        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = _import_httpx()
            assert result is mock_httpx

    def test_raises_runtime_error_when_not_installed(self):
        """Raises RuntimeError with helpful message when httpx is not installed."""
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(RuntimeError, match="httpx is not installed"):
                _import_httpx()


# ---------------------------------------------------------------------------
# TransformersBackend
# ---------------------------------------------------------------------------


class TestTransformersBackendConstructor:
    """Tests for TransformersBackend.__init__."""

    def test_stores_all_parameters(self):
        """All constructor parameters are stored as instance attributes."""
        backend = TransformersBackend(
            model_dir="/models/test",
            num_ctx=8192,
            num_threads=8,
            temperature=0.5,
            max_new_tokens=1024,
        )
        assert backend.model_dir == "/models/test"
        assert backend.num_ctx == 8192
        assert backend.num_threads == 8
        assert backend.temperature == 0.5
        assert backend.max_new_tokens == 1024

    def test_defaults(self):
        """Default values match the documented defaults."""
        backend = TransformersBackend(model_dir="/models/test")
        assert backend.num_ctx == 4096
        assert backend.num_threads == 4
        assert backend.temperature == 0.2
        assert backend.max_new_tokens == 2048

    def test_model_not_loaded_on_init(self):
        """The model/tokenizer are not loaded until _load() is called."""
        backend = TransformersBackend(model_dir="/models/test")
        assert backend._model is None
        assert backend._tokenizer is None
        assert backend._device is None


class TestTransformersBackendModelInfo:
    """Tests for TransformersBackend.model_info property."""

    def test_model_info_contents(self):
        """model_info returns a dict with all configuration fields."""
        backend = TransformersBackend(
            model_dir="/models/test",
            num_ctx=8192,
            num_threads=8,
            temperature=0.5,
            max_new_tokens=1024,
        )
        info = backend.model_info
        assert info["backend"] == "transformers"
        assert info["model_dir"] == "/models/test"
        assert info["num_ctx"] == 8192
        assert info["num_threads"] == 8
        assert info["temperature"] == 0.5
        assert info["max_new_tokens"] == 1024
        assert info["device"] == "not-loaded"

    def test_model_info_after_load_cuda(self):
        """model_info reflects 'cuda' device after _load with GPU."""
        backend = TransformersBackend(model_dir="/models/test")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            backend._load()
        info = backend.model_info
        assert info["device"] == "cuda"


class TestTransformersBackendLoad:
    """Tests for TransformersBackend._load — lazy model/tokenizer loading."""

    def test_load_success_cuda(self):
        """_load returns (model, tokenizer) and sets device to 'cuda' when CUDA is available."""
        backend = TransformersBackend(model_dir="/models/test", num_threads=4)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            model, tokenizer = backend._load()

        assert model is mock_transformers.AutoModelForCausalLM.from_pretrained.return_value
        assert tokenizer is mock_transformers.AutoTokenizer.from_pretrained.return_value
        assert backend._device == "cuda"
        # Verify dtype passed to from_pretrained
        _, kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
        assert kwargs["torch_dtype"] == "float16"
        mock_torch.set_num_threads.assert_called_once_with(4)

    def test_load_success_cpu(self):
        """_load sets device to 'cpu' and uses float32 when CUDA is not available."""
        backend = TransformersBackend(model_dir="/models/test", num_threads=2)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float16 = "float16"
        mock_torch.float32 = "float32"

        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            model, tokenizer = backend._load()

        assert model is mock_transformers.AutoModelForCausalLM.from_pretrained.return_value
        assert tokenizer is mock_transformers.AutoTokenizer.from_pretrained.return_value
        assert backend._device == "cpu"
        _, kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
        assert kwargs["torch_dtype"] == "float32"
        mock_torch.set_num_threads.assert_called_once_with(2)

    def test_load_idempotent(self):
        """When _model and _tokenizer are already set, _load returns them without re-loading."""
        backend = TransformersBackend(model_dir="/models/test")
        backend._tokenizer = "existing_tokenizer"
        backend._model = "existing_model"

        mock_torch = MagicMock()
        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            model, tokenizer = backend._load()

        assert model == "existing_model"
        assert tokenizer == "existing_tokenizer"
        # Should not re-instantiate since already loaded
        mock_transformers.AutoTokenizer.from_pretrained.assert_not_called()
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_load_tokenizer_loaded_but_model_not(self):
        """When only tokenizer is None, model loading also happens."""
        backend = TransformersBackend(model_dir="/models/test")
        backend._tokenizer = "existing_tokenizer"
        backend._model = None

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"

        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            model, tokenizer = backend._load()

        assert model is mock_transformers.AutoModelForCausalLM.from_pretrained.return_value
        assert tokenizer == "existing_tokenizer"
        # Tokenizer should NOT be re-loaded
        mock_transformers.AutoTokenizer.from_pretrained.assert_not_called()
        # Model SHOULD be loaded
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()


class TestTransformersBackendGenerate:
    """Tests for TransformersBackend.generate."""

    def test_generate_returns_decoded_stripped_text(self):
        """generate tokenizes, runs model, decodes, and returns stripped text."""
        backend = TransformersBackend(
            model_dir="/models/test",
            num_ctx=2048,
            max_new_tokens=512,
            temperature=0.7,
            num_threads=4,
        )

        # Pre-load model and tokenizer to bypass _load
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._device = "cpu"

        # Set up tokenizer call return value
        mock_input_ids = MagicMock()
        mock_input_ids.to.return_value = mock_input_ids
        mock_input_ids.shape = (1, 10)

        mock_attention_mask = MagicMock()
        mock_attention_mask.to.return_value = mock_attention_mask

        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        # Set up model.generate return value
        mock_gen_ids = MagicMock()
        mock_gen_ids.__getitem__.return_value.__getitem__.return_value = "sliced_tokens"
        mock_model.generate.return_value = mock_gen_ids

        # Set up tokenizer.decode return value
        mock_tokenizer.decode.return_value = "  generated response  "

        mock_torch = MagicMock()
        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            result = backend.generate("test prompt")

        assert result == "generated response"
        # Verify tokenizer was called with prompt
        mock_tokenizer.assert_called_once()
        _, kwargs = mock_tokenizer.call_args
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == 2048
        # Verify model.generate was called with correct params
        mock_model.generate.assert_called_once()
        _, gen_kwargs = mock_model.generate.call_args
        assert gen_kwargs["max_new_tokens"] == 512
        assert gen_kwargs["temperature"] == 0.7
        assert gen_kwargs["do_sample"] is True
        # Verify torch.no_grad was used as context manager
        mock_torch.no_grad.assert_called_once()

    def test_generate_without_attention_mask(self):
        """When tokenizer returns no attention_mask, generate still works."""
        backend = TransformersBackend(
            model_dir="/models/test", num_ctx=2048, max_new_tokens=512, temperature=0.7
        )

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._device = "cpu"

        mock_input_ids = MagicMock()
        mock_input_ids.to.return_value = mock_input_ids
        mock_input_ids.shape = (1, 5)

        # No attention_mask in the return
        mock_tokenizer.return_value = {"input_ids": mock_input_ids}

        mock_gen_ids = MagicMock()
        mock_gen_ids.__getitem__.return_value.__getitem__.return_value = "sliced"
        mock_model.generate.return_value = mock_gen_ids

        mock_tokenizer.decode.return_value = "response"

        mock_torch = MagicMock()
        mock_transformers = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": mock_transformers}):
            result = backend.generate("prompt")

        assert result == "response"
        # attention_mask should be None in the generate call
        _, gen_kwargs = mock_model.generate.call_args
        assert gen_kwargs["attention_mask"] is None

    def test_generate_with_cuda_device(self):
        """generate moves inputs to the correct device."""
        backend = TransformersBackend(model_dir="/models/test", num_ctx=1024)

        mock_model = MagicMock()
        mock_model.device = "cuda:0"
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        backend._model = mock_model
        backend._tokenizer = mock_tokenizer
        backend._device = "cuda"

        mock_input_ids = MagicMock()
        mock_input_ids.to.return_value = mock_input_ids
        mock_input_ids.shape = (1, 8)

        mock_attention_mask = MagicMock()
        mock_attention_mask.to.return_value = mock_attention_mask

        mock_tokenizer.return_value = {
            "input_ids": mock_input_ids,
            "attention_mask": mock_attention_mask,
        }

        mock_gen_ids = MagicMock()
        mock_gen_ids.__getitem__.return_value.__getitem__.return_value = "x"
        mock_model.generate.return_value = mock_gen_ids
        mock_tokenizer.decode.return_value = "cuda output"

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": MagicMock()}):
            result = backend.generate("prompt")

        assert result == "cuda output"
        # input_ids.to was called with model.device
        mock_input_ids.to.assert_called_with("cuda:0")
        mock_attention_mask.to.assert_called_with("cuda:0")


class TestTransformersBackendClose:
    """Tests for TransformersBackend.close — memory cleanup paths."""

    def test_close_with_model_and_tokenizer(self):
        """close() deletes _model, _tokenizer, calls gc.collect, and
        calls torch.cuda.empty_cache when CUDA is available."""
        backend = TransformersBackend(model_dir="/models/test")
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("gc.collect") as mock_gc, patch.dict("sys.modules", {"torch": mock_torch}):
            backend.close()

        assert backend._model is None
        assert backend._tokenizer is None
        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_called_once()

    def test_close_without_model_or_tokenizer(self):
        """close() when no model/tokenizer loaded does not crash."""
        backend = TransformersBackend(model_dir="/models/test")
        # _model and _tokenizer are None by default

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("gc.collect") as mock_gc, patch.dict("sys.modules", {"torch": mock_torch}):
            backend.close()

        assert backend._model is None
        assert backend._tokenizer is None
        mock_gc.assert_called_once()
        mock_torch.cuda.empty_cache.assert_not_called()

    def test_close_handles_torch_import_error(self):
        """close() catches ImportError when torch is not importable."""
        backend = TransformersBackend(model_dir="/models/test")
        backend._model = MagicMock()
        backend._tokenizer = MagicMock()

        with patch("gc.collect") as mock_gc, patch.dict("sys.modules", {"torch": None}):
            # ImportError: import of torch halted; None in sys.modules
            backend.close()

        assert backend._model is None
        assert backend._tokenizer is None
        mock_gc.assert_called_once()

    def test_close_with_only_model(self):
        """close() handles case where _tokenizer is None but _model is set."""
        backend = TransformersBackend(model_dir="/models/test")
        backend._model = MagicMock()
        backend._tokenizer = None

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("gc.collect"), patch.dict("sys.modules", {"torch": mock_torch}):
            backend.close()

        assert backend._model is None
        assert backend._tokenizer is None


class TestFindHfModelDir:
    """Tests for _find_hf_model_dir helper function."""

    def test_returns_none_when_no_candidates(self, tmp_path):
        """When no config.json files are found, returns None."""
        gguf_path = str(tmp_path / "model.gguf")
        with patch("glob.glob", return_value=[]):
            result = _find_hf_model_dir(gguf_path)
        assert result is None

    def test_finds_dir_with_model_safetensors(self, tmp_path):
        """Returns the directory when model.safetensors exists."""
        candidates = ["/test/model_dir/config.json"]

        def glob_side_effect(pattern):
            if "config.json" in pattern:
                return candidates
            return []

        gguf_path = str(tmp_path / "model.gguf")
        with (
            patch("glob.glob", side_effect=glob_side_effect),
            patch("os.path.exists", return_value=True),
        ):
            result = _find_hf_model_dir(gguf_path)
        assert result == "/test/model_dir"

    def test_finds_dir_with_pytorch_model_bin(self, tmp_path):
        """Returns the directory when pytorch_model.bin exists (but not safetensors)."""
        candidates = ["/test/model_dir/config.json"]

        def glob_side_effect(pattern):
            if "config.json" in pattern:
                return candidates
            return []

        def exists_side_effect(path):
            return "pytorch_model.bin" in path

        gguf_path = str(tmp_path / "model.gguf")
        with (
            patch("glob.glob", side_effect=glob_side_effect),
            patch("os.path.exists", side_effect=exists_side_effect),
        ):
            result = _find_hf_model_dir(gguf_path)
        assert result == "/test/model_dir"

    def test_finds_dir_with_safetensors_index(self, tmp_path):
        """Returns the directory when model.safetensors.index.json exists."""
        candidates = ["/test/model_dir/config.json"]

        def glob_side_effect(pattern):
            if "config.json" in pattern:
                return candidates
            return []

        def exists_side_effect(path):
            return "model.safetensors.index.json" in path

        gguf_path = str(tmp_path / "model.gguf")
        with (
            patch("glob.glob", side_effect=glob_side_effect),
            patch("os.path.exists", side_effect=exists_side_effect),
        ):
            result = _find_hf_model_dir(gguf_path)
        assert result == "/test/model_dir"

    def test_finds_dir_with_sharded_safetensors(self, tmp_path):
        """Returns the directory when sharded safetensors exist (model-*.safetensors)."""
        candidates = ["/test/model_dir/config.json"]

        def glob_side_effect(pattern):
            if "config.json" in pattern:
                return candidates
            if "model-*.safetensors" in pattern:
                return ["model-00001-of-00002.safetensors"]
            return []

        gguf_path = str(tmp_path / "model.gguf")
        with (
            patch("glob.glob", side_effect=glob_side_effect),
            patch("os.path.exists", return_value=False),
        ):
            result = _find_hf_model_dir(gguf_path)
        assert result == "/test/model_dir"

    def test_skips_candidate_without_weights(self, tmp_path):
        """Skips a candidate whose directory has no weight files and continues."""
        candidates = [
            "/test/empty_dir/config.json",
            "/test/model_dir/config.json",
        ]

        def glob_side_effect(pattern):
            if "config.json" in pattern:
                return candidates
            if "model-*.safetensors" in pattern and "empty_dir" in pattern:
                return []
            return []

        def exists_side_effect(path):
            # Only model_dir has model.safetensors
            return "model.safetensors" in path and "model_dir" in path

        gguf_path = str(tmp_path / "model.gguf")
        with (
            patch("glob.glob", side_effect=glob_side_effect),
            patch("os.path.exists", side_effect=exists_side_effect),
        ):
            result = _find_hf_model_dir(gguf_path)
        assert result == "/test/model_dir"

    def test_returns_none_when_no_weights_found(self, tmp_path):
        """When all candidates lack weight files, returns None."""
        candidates = ["/test/empty1/config.json", "/test/empty2/config.json"]

        def glob_side_effect(pattern):
            if "config.json" in pattern:
                return candidates
            return []

        gguf_path = str(tmp_path / "model.gguf")
        with (
            patch("glob.glob", side_effect=glob_side_effect),
            patch("os.path.exists", return_value=False),
        ):
            result = _find_hf_model_dir(gguf_path)
        assert result is None
