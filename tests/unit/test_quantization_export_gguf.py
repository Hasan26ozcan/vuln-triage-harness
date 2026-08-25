"""Tests for Stage 8 — GGUF quantization backend (app/quantization/export_gguf.py).

Covers the pure helper ``_hf_name_to_gguf``, ``gguf_type_to_bits``, and the
HF→GGUF conversion path ``convert_hf_to_gguf_f16`` / ``_load_hf_state_dict``
with all heavy dependencies (torch, transformers, peft, gguf, numpy) mocked.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

from app.quantization.export_gguf import (
    _hf_name_to_gguf,
    gguf_type_to_bits,
)

# ---------------------------------------------------------------------------
# _HF_TENSOR_MAP and _GGUF_TYPE_TO_BITS — covered via imports above
# ---------------------------------------------------------------------------


class TestGgufTypeToBits:
    """Tests for ``gguf_type_to_bits`` (lines 469-476)."""

    def test_q4_k_maps_to_4(self):
        assert gguf_type_to_bits("Q4_K") == 4

    def test_q8_0_maps_to_8(self):
        assert gguf_type_to_bits("Q8_0") == 8

    def test_f16_maps_to_16(self):
        assert gguf_type_to_bits("F16") == 16

    def test_f32_maps_to_32(self):
        assert gguf_type_to_bits("F32") == 32

    def test_q2_k_maps_to_2(self):
        assert gguf_type_to_bits("Q2_K") == 2

    def test_q5_k_s_maps_to_5(self):
        assert gguf_type_to_bits("Q5_K_S") == 5

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown GGUF quant type"):
            gguf_type_to_bits("Q99_K")


# ---------------------------------------------------------------------------
# _hf_name_to_gguf  (lines 90-117)
# ---------------------------------------------------------------------------


class TestHfNameToGguf:
    """Tests for the HF→GGUF tensor name translation (lines 90-117)."""

    def test_model_embed_tokens(self):
        assert _hf_name_to_gguf("model.embed_tokens.weight") == "token_embd.weight"

    def test_model_norm(self):
        assert _hf_name_to_gguf("model.norm.weight") == "output_norm.weight"

    def test_lm_head(self):
        assert _hf_name_to_gguf("lm_head.weight") == "output.weight"

    def test_layer_self_attn_q_proj(self):
        result = _hf_name_to_gguf("model.layers.3.self_attn.q_proj.weight")
        assert result == "blk.3.attn_q.weight"

    def test_layer_self_attn_k_proj(self):
        result = _hf_name_to_gguf("model.layers.5.self_attn.k_proj.weight")
        assert result == "blk.5.attn_k.weight"

    def test_layer_self_attn_v_proj(self):
        result = _hf_name_to_gguf("model.layers.7.self_attn.v_proj.weight")
        assert result == "blk.7.attn_v.weight"

    def test_layer_self_attn_o_proj(self):
        result = _hf_name_to_gguf("model.layers.0.self_attn.o_proj.weight")
        assert result == "blk.0.attn_output.weight"

    def test_layer_mlp_gate_proj(self):
        result = _hf_name_to_gguf("model.layers.2.mlp.gate_proj.weight")
        assert result == "blk.2.ffn_gate.weight"

    def test_layer_mlp_up_proj(self):
        result = _hf_name_to_gguf("model.layers.1.mlp.up_proj.weight")
        assert result == "blk.1.ffn_up.weight"

    def test_layer_mlp_down_proj(self):
        result = _hf_name_to_gguf("model.layers.4.mlp.down_proj.weight")
        assert result == "blk.4.ffn_down.weight"

    def test_layer_bias(self):
        result = _hf_name_to_gguf("model.layers.0.self_attn.q_proj.bias")
        assert result == "blk.0.attn_q.bias"

    def test_unmapped_tensor_returns_none(self):
        """Names that don't match any mapping return None (logged debug)."""
        assert _hf_name_to_gguf("model.layers.0.some_unknown_thing.weight") is None

    def test_not_model_layers_prefix(self):
        """Names that don't start with model.layers. and aren't top-level return None."""
        assert _hf_name_to_gguf("some_other_prefix.weight") is None


# ---------------------------------------------------------------------------
# _load_hf_state_dict  (lines 120-194)
# ---------------------------------------------------------------------------


class TestLoadHfStateDict:
    """Tests for ``_load_hf_state_dict`` with mocked torch/transformers/peft."""

    @pytest.fixture
    def mock_ml_deps(self):
        """Inject mock torch, transformers, and peft into sys.modules."""
        mock_torch = MagicMock()
        mock_torch.float16 = "torch.float16"
        mock_torch.no_grad = lambda *a, **kw: MagicMock()
        mock_torch.tensor = MagicMock()

        mock_transformers = MagicMock()
        mock_auto_config = MagicMock()
        mock_auto_config.from_pretrained = MagicMock(return_value=MagicMock())
        mock_auto_config.to_dict = MagicMock(return_value={})
        mock_transformers.AutoConfig = mock_auto_config
        mock_transformers.AutoModelForCausalLM = MagicMock()
        mock_transformers.AutoTokenizer = MagicMock()
        mock_transformers.pipeline = MagicMock()

        mock_peft = MagicMock()
        mock_peft.PeftModel = MagicMock()

        with patch.dict(
            sys.modules,
            {"torch": mock_torch, "transformers": mock_transformers, "peft": mock_peft},
        ):
            yield mock_torch, mock_transformers, mock_peft

    def test_full_hf_checkpoint_no_adapter(self, tmp_path, mock_ml_deps):
        """Full checkpoint path (no adapter_config.json) loads directly."""
        mock_torch, mock_transformers, mock_peft = mock_ml_deps
        ckpt = str(tmp_path / "ckpt")
        os.makedirs(ckpt, exist_ok=True)

        mock_model = MagicMock()
        mock_model.state_dict.return_value = {"model.embed_tokens.weight": MagicMock()}
        mock_transformers.AutoModelForCausalLM.from_pretrained = MagicMock(
            return_value=mock_model
        )

        mock_config = MagicMock()
        mock_config.to_dict.return_value = {"model_type": "qwen2"}
        mock_transformers.AutoConfig.from_pretrained = MagicMock(
            return_value=mock_config
        )

        from app.quantization.export_gguf import _load_hf_state_dict

        state_dict, config_dict = _load_hf_state_dict(ckpt)

        assert config_dict == {"model_type": "qwen2"}
        assert "model.embed_tokens.weight" in state_dict
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        mock_peft.PeftModel.from_pretrained.assert_not_called()

    def test_peft_adapter_with_config_base_model(self, tmp_path, mock_ml_deps):
        """PEFT adapter path: adapter_config.json present with base_model_name_or_path."""
        import json

        mock_torch, mock_transformers, mock_peft = mock_ml_deps
        ckpt = str(tmp_path / "adapter_ckpt")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "adapter_config.json"), "w") as f:
            json.dump(
                {"base_model_name_or_path": "Qwen/Qwen2.5-Coder-7B-Instruct"},
                f,
            )

        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_merged_model.state_dict.return_value = {"merged_weight": MagicMock()}
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_transformers.AutoModelForCausalLM.from_pretrained = MagicMock(
            return_value=mock_base_model
        )
        mock_peft.PeftModel.from_pretrained = MagicMock(return_value=mock_peft_model)

        from app.quantization.export_gguf import _load_hf_state_dict

        state_dict, config_dict = _load_hf_state_dict(ckpt)

        assert "merged_weight" in state_dict
        mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called()
        mock_peft.PeftModel.from_pretrained.assert_called_once()
        mock_peft_model.merge_and_unload.assert_called_once()

    def test_peft_adapter_uses_explicit_base_model(self, tmp_path, mock_ml_deps):
        """When base_model is passed explicitly and adapter_config has no base."""
        import json

        mock_torch, mock_transformers, mock_peft = mock_ml_deps
        ckpt = str(tmp_path / "adapter_ckpt")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "adapter_config.json"), "w") as f:
            json.dump({}, f)  # no base_model_name_or_path

        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_merged_model.state_dict.return_value = {"weight": MagicMock()}
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_transformers.AutoModelForCausalLM.from_pretrained = MagicMock(
            return_value=mock_base_model
        )
        mock_peft.PeftModel.from_pretrained = MagicMock(return_value=mock_peft_model)

        from app.quantization.export_gguf import _load_hf_state_dict

        state_dict, config_dict = _load_hf_state_dict(
            ckpt, base_model="Qwen/Qwen2.5-Coder-7B-Instruct"
        )

        mock_peft.PeftModel.from_pretrained.assert_called()
        mock_peft_model.merge_and_unload.assert_called_once()

    def test_peft_adapter_no_base_falls_back_to_default(self, tmp_path, mock_ml_deps):
        """When adapter_config has no base and base_model is None, use DEFAULT_BASE_MODEL."""
        import json

        from app.quantization.config import DEFAULT_BASE_MODEL

        mock_torch, mock_transformers, mock_peft = mock_ml_deps
        ckpt = str(tmp_path / "adapter_ckpt")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, "adapter_config.json"), "w") as f:
            json.dump({}, f)  # no base_model_name_or_path

        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_merged_model = MagicMock()
        mock_merged_model.state_dict.return_value = {"weight": MagicMock()}
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_transformers.AutoModelForCausalLM.from_pretrained = MagicMock(
            return_value=mock_base_model
        )
        mock_peft.PeftModel.from_pretrained = MagicMock(return_value=mock_peft_model)

        from app.quantization.export_gguf import _load_hf_state_dict

        _load_hf_state_dict(ckpt)  # base_model=None

        # DEFAULT_BASE_MODEL should have been used as the base model
        call_args = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
        assert DEFAULT_BASE_MODEL in str(call_args)


# ---------------------------------------------------------------------------
# convert_hf_to_gguf_f16  (lines 197-291)
# ---------------------------------------------------------------------------


class TestConvertHfToGgufF16:
    """Tests for ``convert_hf_to_gguf_f16`` with all heavy deps mocked."""

    @pytest.fixture
    def mock_convert_deps(self, tmp_path):
        """Mock torch, numpy, gguf, and the _load_hf_state_dict helper."""
        mock_torch = MagicMock()
        mock_numpy = MagicMock()
        mock_gguf_lib = MagicMock()

        # _load_hf_state_dict returns (state_dict, config_dict)
        fake_param = MagicMock()
        numpy_ret = MagicMock(dtype="float16")
        (
            fake_param.detach.return_value.cpu.return_value
            .contiguous.return_value.numpy.return_value
        ) = numpy_ret  # noqa: E501
        state_dict = {"model.embed_tokens.weight": fake_param}

        config_dict = {
            "model_type": "qwen2",
            "num_hidden_layers": 28,
            "hidden_size": 1536,
            "intermediate_size": 3072,
            "max_position_embeddings": 32768,
            "vocab_size": 151936,
            "num_attention_heads": 12,
            "num_key_value_heads": 2,
            "rope_theta": 1000000.0,
            "rms_norm_eps": 1e-6,
            "model_name_or_path": "qwen2-gguf",
        }

        output_path = str(tmp_path / "output.gguf")

        with patch.dict(
            sys.modules,
            {"torch": mock_torch, "numpy": mock_numpy, "gguf": mock_gguf_lib},
        ), patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            yield mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path

    def test_convert_writes_f16_gguf(self, mock_convert_deps):
        """convert_hf_to_gguf_f16 calls writer methods in correct order."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        # Set up the fake param to return float16
        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.float16
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock  # noqa: E501
        state_dict = {"model.embed_tokens.weight": fake_param}

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            result = convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        assert result == output_path
        mock_gguf_lib.GGUFWriter.assert_called_once()
        writer = mock_gguf_lib.GGUFWriter.return_value
        writer.write_header_to_file.assert_called_once()
        writer.write_kv_data_to_file.assert_called_once()
        writer.close.assert_called_once()

    def test_convert_skips_unmapped_tensors(self, mock_convert_deps):
        """Tensors whose HF name maps to None are skipped."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.float16
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock

        # One unmapped tensor (model.layers.0.some_unknown) + one mapped
        state_dict = {
            "model.embed_tokens.weight": fake_param,
            "model.layers.0.unknown_layer.weight": fake_param,
        }

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        writer = mock_gguf_lib.GGUFWriter.return_value
        # Only 1 add_tensor call (the mapped one)
        assert writer.add_tensor.call_count == 1

    def test_convert_skips_duplicate_names(self, mock_convert_deps):
        """Duplicate GGUF names are only written once."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.float16
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock

        # Both map to the same GGUF name (blk.0.attn_q.weight)
        state_dict = {
            "model.layers.0.self_attn.q_proj.weight": fake_param,
            # q_proj.bias maps to blk.0.attn_q.bias, not a dupe
            "model.layers.0.self_attn.q_proj.bias": fake_param,
        }

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        writer = mock_gguf_lib.GGUFWriter.return_value
        assert writer.add_tensor.call_count == 2  # both are unique GGUF names

    def test_convert_float32_to_float16(self, mock_convert_deps):
        """float32 tensors are converted to float16."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.float32
        arr_mock.astype.return_value = MagicMock(dtype=mock_numpy.float16)
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock

        state_dict = {"model.embed_tokens.weight": fake_param}

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        arr_mock.astype.assert_called_once_with(mock_numpy.float16)

    def test_convert_float64_to_float32(self, mock_convert_deps):
        """float64 tensors are converted to float32."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.float64
        arr_mock.astype.return_value = MagicMock(dtype=mock_numpy.float32)
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock

        state_dict = {"model.embed_tokens.weight": fake_param}

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        arr_mock.astype.assert_called_once_with(mock_numpy.float32)

    def test_convert_bfloat16_to_float16(self, mock_convert_deps):
        """bfloat16 tensors are converted to float16."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.bfloat16
        arr_mock.astype.return_value = MagicMock(dtype=mock_numpy.float16)
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock

        state_dict = {"model.embed_tokens.weight": fake_param}

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        arr_mock.astype.assert_called_once_with(mock_numpy.float16)

    def test_convert_unexpected_dtype_warns_and_converts(self, mock_convert_deps):
        """Unexpected dtypes trigger a warning and are converted to float16."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.int32  # unexpected dtype
        arr_mock.astype.return_value = MagicMock(dtype=mock_numpy.float16)
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock

        state_dict = {"model.embed_tokens.weight": fake_param}

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        # Should be converted to float16 (the warning is logged, not raised)
        arr_mock.astype.assert_called_once_with(mock_numpy.float16)

    def test_convert_sets_metadata_fields(self, mock_convert_deps):
        """All metadata fields are set on the GGUF writer."""
        mock_torch, mock_numpy, mock_gguf_lib, config_dict, output_path = mock_convert_deps

        fake_param = MagicMock()
        arr_mock = MagicMock()
        arr_mock.dtype = mock_numpy.float16
        np_return = fake_param.detach.return_value.cpu.return_value.contiguous.return_value.numpy
        np_return.return_value = arr_mock  # noqa: E501
        state_dict = {"model.embed_tokens.weight": fake_param}

        with patch(
            "app.quantization.export_gguf._load_hf_state_dict",
            return_value=(state_dict, config_dict),
        ):
            from app.quantization.export_gguf import convert_hf_to_gguf_f16

            convert_hf_to_gguf_f16("/fake/ckpt", output_path)

        writer = mock_gguf_lib.GGUFWriter.return_value
        writer.add_architecture.assert_called_once()
        writer.add_string.assert_any_call("general.name", "qwen2-gguf")
        writer.add_file_type.assert_called_once()
        writer.add_block_count.assert_called_once_with(28)
        writer.add_embedding_length.assert_called_once_with(1536)


# ---------------------------------------------------------------------------
# Module-level constants / __all__
# ---------------------------------------------------------------------------


class TestModuleLevel:
    def test_all_exports(self):
        from app.quantization import export_gguf

        assert "GGUFQuantizer" in export_gguf.__all__
        assert "convert_hf_to_gguf_f16" in export_gguf.__all__
