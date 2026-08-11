"""Unit tests for Stage 9 — ServingConfig (app/serving/config.py).

Covers every method, every branch, and all validation logic so that
``config.py`` has 100 % line coverage.
"""

import pytest

from app.serving.config import (
    DEFAULT_BACKEND_TYPE,
    DEFAULT_F16_KV,
    DEFAULT_HOST,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_N_GPU_LAYERS,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_THREADS,
    DEFAULT_PORT,
    DEFAULT_TEMPERATURE,
    _VALID_BACKEND_TYPES,
    ServingConfig,
)

# --- Constants ---


def test_default_model_path_is_empty():
    assert DEFAULT_MODEL_PATH == ""


def test_default_backend_type_is_llama_cpp():
    assert DEFAULT_BACKEND_TYPE == "llama.cpp"


def test_default_num_ctx_is_4096():
    assert DEFAULT_NUM_CTX == 4096


def test_default_num_threads_is_4():
    assert DEFAULT_NUM_THREADS == 4


def test_default_n_gpu_layers_is_0():
    assert DEFAULT_N_GPU_LAYERS == 0


def test_default_f16_kv_is_true():
    assert DEFAULT_F16_KV is True


def test_default_temperature_is_low():
    assert DEFAULT_TEMPERATURE == 0.2


def test_default_max_new_tokens_is_2048():
    assert DEFAULT_MAX_NEW_TOKENS == 2048


def test_default_host_is_all_interfaces():
    assert DEFAULT_HOST == "0.0.0.0"


def test_default_port_is_8000():
    assert DEFAULT_PORT == 8000


def test_valid_backend_types():
    assert _VALID_BACKEND_TYPES == frozenset({"llama.cpp", "ollama", "mock"})


# --- __post_init__ / validation ---


def test_default_config_uses_defaults():
    config = ServingConfig()
    assert config.model_path == DEFAULT_MODEL_PATH
    assert config.backend_type == DEFAULT_BACKEND_TYPE
    assert config.num_ctx == DEFAULT_NUM_CTX
    assert config.num_threads == DEFAULT_NUM_THREADS
    assert config.n_gpu_layers == DEFAULT_N_GPU_LAYERS
    assert config.f16_kv == DEFAULT_F16_KV
    assert config.temperature == DEFAULT_TEMPERATURE
    assert config.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert config.host == DEFAULT_HOST
    assert config.port == DEFAULT_PORT


def test_invalid_backend_type_raises():
    with pytest.raises(ValueError, match="backend_type='bogus'"):
        ServingConfig(backend_type="bogus")


def test_valid_backend_types_accepted():
    for bt in _VALID_BACKEND_TYPES:
        config = ServingConfig(backend_type=bt)
        assert config.backend_type == bt


# --- run_name ---


def test_run_name_with_model_path():
    config = ServingConfig(model_path="/path/to/qwen-q4_0.gguf")
    assert config.run_name == "serve_qwen-q4_0.gguf_llama.cpp"


def test_run_name_with_windows_model_path():
    config = ServingConfig(model_path="C:\\models\\qwen-q4_0.gguf")
    assert config.run_name == "serve_qwen-q4_0.gguf_llama.cpp"


def test_run_name_without_model_path():
    config = ServingConfig(model_path="", backend_type="mock")
    assert config.run_name == "serve_mock_mock"


def test_run_name_without_model_path_llama():
    config = ServingConfig(model_path="", backend_type="llama.cpp")
    assert config.run_name == "serve_mock_llama.cpp"


# --- is_mock ---


def test_is_mock_true():
    config = ServingConfig(backend_type="mock")
    assert config.is_mock() is True


def test_is_mock_false():
    config = ServingConfig(backend_type="llama.cpp", model_path="model.gguf")
    assert config.is_mock() is False


# --- all_warnings ---


def test_no_warnings_for_valid_config():
    config = ServingConfig(
        model_path="model.gguf",
        backend_type="llama.cpp",
        num_threads=4,
        max_new_tokens=2048,
        num_ctx=4096,
    )
    assert config.all_warnings() == []


def test_warning_empty_model_path_real_backend():
    config = ServingConfig(backend_type="llama.cpp", model_path="")
    warnings = config.all_warnings()
    assert any("model_path is empty" in w for w in warnings)


def test_warning_empty_model_path_ollama():
    config = ServingConfig(backend_type="ollama", model_path="")
    warnings = config.all_warnings()
    assert any("model_path is empty" in w for w in warnings)


def test_no_warning_empty_model_path_mock():
    config = ServingConfig(backend_type="mock", model_path="")
    warnings = config.all_warnings()
    assert not any("model_path is empty" in w for w in warnings)


def test_warning_num_threads_below_one():
    config = ServingConfig(backend_type="llama.cpp", model_path="m.gguf", num_threads=0)
    warnings = config.all_warnings()
    assert any("num_threads=0" in w for w in warnings)


def test_warning_num_threads_ok():
    config = ServingConfig(backend_type="llama.cpp", model_path="m.gguf", num_threads=1)
    warnings = config.all_warnings()
    assert not any("num_threads" in w for w in warnings)


def test_warning_max_new_tokens_below_one():
    config = ServingConfig(backend_type="mock", max_new_tokens=0)
    warnings = config.all_warnings()
    assert any("max_new_tokens=0" in w for w in warnings)


def test_warning_num_ctx_below_512():
    config = ServingConfig(backend_type="mock", num_ctx=256)
    warnings = config.all_warnings()
    assert any("num_ctx=256" in w for w in warnings)


def test_multiple_warnings():
    config = ServingConfig(
        backend_type="llama.cpp",
        model_path="",
        num_threads=0,
        max_new_tokens=0,
        num_ctx=100,
    )
    warnings = config.all_warnings()
    assert len(warnings) == 4


def test_no_warnings_ollama_valid():
    config = ServingConfig(
        backend_type="ollama",
        model_path="qwen2.5-coder",
        num_threads=8,
        max_new_tokens=2048,
        num_ctx=4096,
    )
    assert config.all_warnings() == []


# --- Custom configuration ---


def test_custom_config():
    config = ServingConfig(
        model_path="/models/qwen2.5-coder-ggml-q4_0.gguf",
        backend_type="llama.cpp",
        num_ctx=8192,
        num_threads=8,
        n_gpu_layers=2,
        temperature=0.1,
        max_new_tokens=4096,
        port=9000,
    )
    assert config.num_ctx == 8192
    assert config.num_threads == 8
    assert config.n_gpu_layers == 2
    assert config.temperature == 0.1
    assert config.port == 9000
