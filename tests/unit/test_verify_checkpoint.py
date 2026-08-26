"""Unit tests for scripts/verify_checkpoint.py.

This is the guard that prevents the Stage 7 "silent base-model fallback"
regression: a LoRA checkpoint directory with adapter_config.json but no
adapter_model.safetensors/.bin used to be treated as usable, causing
Stage 7 to secretly evaluate the base model as if it were fine-tuned
(forgetting_delta == 0.0 for the wrong reason).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from verify_checkpoint import verify_checkpoint  # noqa: E402


def test_missing_directory_raises_file_not_found(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(str(missing))


def test_lora_config_without_weights_raises(tmp_path):
    """The exact bug state: adapter_config.json present, weights absent."""
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no adapter_model"):
        verify_checkpoint(str(ckpt))


def test_lora_config_with_empty_weights_file_raises(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_bytes(b"")

    with pytest.raises(RuntimeError, match="empty"):
        verify_checkpoint(str(ckpt))


def test_lora_checkpoint_with_weights_succeeds(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    payload = b"fake-safetensors-bytes-not-real-weights"
    (ckpt / "adapter_model.safetensors").write_bytes(payload)

    fingerprint = verify_checkpoint(str(ckpt))

    assert fingerprint["checkpoint_type"] == "lora"
    assert fingerprint["adapter_weight_file"] == "adapter_model.safetensors"
    assert fingerprint["adapter_size_bytes"] == len(payload)
    assert len(fingerprint["adapter_sha256"]) == 64  # hex sha256 digest


def test_lora_checkpoint_with_bin_weights_succeeds(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.bin").write_bytes(b"fake-bin-weights")

    fingerprint = verify_checkpoint(str(ckpt))
    assert fingerprint["adapter_weight_file"] == "adapter_model.bin"


def test_full_model_checkpoint_without_lora_config_succeeds(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"fake-full-model-weights")
    (ckpt / "config.json").write_text("{}", encoding="utf-8")

    fingerprint = verify_checkpoint(str(ckpt))
    assert fingerprint["checkpoint_type"] == "full_model"
    assert "model.safetensors" in fingerprint["weight_files"]


def test_directory_with_neither_config_nor_weights_raises(tmp_path):
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tokenizer-only"):
        verify_checkpoint(str(ckpt))


def test_fingerprint_is_stable_for_same_content(tmp_path):
    """Same weight bytes → same sha256, so the fingerprint is a reliable
    way to prove which exact checkpoint a run used."""
    ckpt_a = tmp_path / "a"
    ckpt_a.mkdir()
    (ckpt_a / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt_a / "adapter_model.safetensors").write_bytes(b"identical-content")

    ckpt_b = tmp_path / "b"
    ckpt_b.mkdir()
    (ckpt_b / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt_b / "adapter_model.safetensors").write_bytes(b"identical-content")

    fp_a = verify_checkpoint(str(ckpt_a))
    fp_b = verify_checkpoint(str(ckpt_b))
    assert fp_a["adapter_sha256"] == fp_b["adapter_sha256"]
