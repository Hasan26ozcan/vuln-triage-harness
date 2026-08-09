"""Unit tests for Stage 5 training config module.

Covers:
  - TrainingMethod enum values.
  - SFTConfig defaults, method property (QLoRA vs full).
  - DPOConfig defaults and method property.
  - SweepConfig.to_sft_configs() expansion across ranks.
  - config_to_hyperparams serialization (JSON-safe, no private fields).
  - validate_config warnings for all three config types.
"""

from __future__ import annotations

from app.training.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DPO_BETA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_SWEEP_RANKS,
    DPOConfig,
    SFTConfig,
    SweepConfig,
    TrainingMethod,
    config_to_hyperparams,
    validate_config,
)

# ---------------------------------------------------------------------------
# TrainingMethod enum
# ---------------------------------------------------------------------------


class TestTrainingMethod:
    def test_enum_values(self):
        assert TrainingMethod.SFT_FULL.value == "sft_full"
        assert TrainingMethod.SFT_QLORA.value == "sft_qlora"
        assert TrainingMethod.LORA.value == "lora"
        assert TrainingMethod.DPO.value == "dpo"

    def test_is_str_enum(self):
        # TrainingMethod is str-based so it serializes cleanly to JSON
        assert isinstance(TrainingMethod.SFT_FULL, str)
        assert TrainingMethod.SFT_FULL == "sft_full"


# ---------------------------------------------------------------------------
# SFTConfig
# ---------------------------------------------------------------------------


class TestSFTConfig:
    def test_defaults(self):
        cfg = SFTConfig()
        assert cfg.base_model == DEFAULT_BASE_MODEL
        assert cfg.use_4bit is True  # QLoRA by default
        assert cfg.lora_r == 64
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.05
        assert cfg.learning_rate == DEFAULT_LEARNING_RATE
        assert cfg.num_train_epochs == DEFAULT_NUM_TRAIN_EPOCHS
        assert cfg.train_jsonl == ""
        assert cfg.val_jsonl == ""
        assert cfg.run_name is None

    def test_method_qlora_when_4bit(self):
        cfg = SFTConfig(use_4bit=True)
        assert cfg.method == TrainingMethod.SFT_QLORA
        assert cfg.method_str == "sft_qlora"

    def test_method_full_when_not_4bit(self):
        cfg = SFTConfig(use_4bit=False)
        assert cfg.method == TrainingMethod.SFT_FULL
        assert cfg.method_str == "sft_full"

    def test_method_str_matches_method_value(self):
        cfg = SFTConfig(use_4bit=True)
        assert cfg.method_str == cfg.method.value

    def test_custom_values(self):
        cfg = SFTConfig(
            base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            use_4bit=False,
            lora_r=32,
            lora_alpha=64,
            learning_rate=5e-5,
            num_train_epochs=5,
        )
        assert cfg.lora_r == 32
        assert cfg.learning_rate == 5e-5
        assert cfg.num_train_epochs == 5
        assert cfg.method == TrainingMethod.SFT_FULL


# ---------------------------------------------------------------------------
# DPOConfig
# ---------------------------------------------------------------------------


class TestDPOConfig:
    def test_defaults(self):
        cfg = DPOConfig()
        assert cfg.base_model == DEFAULT_BASE_MODEL
        assert cfg.beta == DEFAULT_DPO_BETA
        assert cfg.loss_type == "sigmoid"
        assert cfg.learning_rate == DEFAULT_LEARNING_RATE
        assert cfg.sft_checkpoint == ""
        assert cfg.run_name is None

    def test_method_is_dpo(self):
        cfg = DPOConfig()
        assert cfg.method == TrainingMethod.DPO
        assert cfg.method_str == "dpo"


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------


class TestSweepConfig:
    def test_defaults(self):
        cfg = SweepConfig()
        assert cfg.base_model == DEFAULT_BASE_MODEL
        assert cfg.ranks == list(DEFAULT_SWEEP_RANKS)
        assert cfg.ranks == [8, 16, 32, 64, 128]
        assert cfg.train_jsonl == ""
        assert cfg.run_name is None

    def test_to_sft_configs_returns_one_per_rank(self):
        cfg = SweepConfig(
            ranks=[8, 16, 32],
            base_model="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            train_jsonl="train.jsonl",
            val_jsonl="val.jsonl",
        )
        configs = cfg.to_sft_configs()
        assert len(configs) == 3
        assert all(isinstance(c, SFTConfig) for c in configs)

    def test_to_sft_configs_sets_rank_correctly(self):
        cfg = SweepConfig(ranks=[8, 16, 32])
        configs = cfg.to_sft_configs()
        assert configs[0].lora_r == 8
        assert configs[1].lora_r == 16
        assert configs[2].lora_r == 32

    def test_to_sft_configs_uses_qlora(self):
        """All sweep runs use QLoRA (4-bit) to fit 8GB VRAM."""
        cfg = SweepConfig(ranks=[8, 16])
        configs = cfg.to_sft_configs()
        assert all(c.use_4bit for c in configs)

    def test_to_sft_configs_inherits_shared_hyperparams(self):
        cfg = SweepConfig(
            ranks=[32],
            learning_rate=3e-5,
            num_train_epochs=5,
            lora_alpha=32,
            lora_dropout=0.1,
            train_jsonl="train.jsonl",
            val_jsonl="val.jsonl",
        )
        configs = cfg.to_sft_configs()
        c = configs[0]
        assert c.learning_rate == 3e-5
        assert c.num_train_epochs == 5
        assert c.lora_alpha == 32
        assert c.lora_dropout == 0.1
        assert c.train_jsonl == "train.jsonl"
        assert c.val_jsonl == "val.jsonl"

    def test_to_sft_configs_distinct_output_dirs(self):
        cfg = SweepConfig(
            ranks=[8, 16],
            output_dir="./output/stage5/sweep",
        )
        configs = cfg.to_sft_configs()
        assert configs[0].output_dir == "./output/stage5/sweep/lora-r8"
        assert configs[1].output_dir == "./output/stage5/sweep/lora-r16"

    def test_to_sft_configs_run_names(self):
        cfg = SweepConfig(
            ranks=[8, 16],
            run_name="my_sweep",
        )
        configs = cfg.to_sft_configs()
        assert configs[0].run_name == "my_sweep"
        assert configs[1].run_name == "my_sweep"

    def test_to_sft_configs_default_run_name(self):
        cfg = SweepConfig(ranks=[8, 16])
        configs = cfg.to_sft_configs()
        assert configs[0].run_name == "lora-r8"
        assert configs[1].run_name == "lora-r16"


# ---------------------------------------------------------------------------
# config_to_hyperparams
# ---------------------------------------------------------------------------


class TestConfigToHyperparams:
    def test_sft_config_serializes_to_dict(self):
        cfg = SFTConfig(lora_r=8, learning_rate=1e-5)
        hp = config_to_hyperparams(cfg)
        assert isinstance(hp, dict)
        assert hp["lora_r"] == 8
        assert hp["learning_rate"] == 1e-5
        assert hp["base_model"] == DEFAULT_BASE_MODEL

    def test_dpo_config_serializes_to_dict(self):
        cfg = DPOConfig(beta=0.5, learning_rate=3e-5)
        hp = config_to_hyperparams(cfg)
        assert hp["beta"] == 0.5
        assert hp["learning_rate"] == 3e-5

    def test_sweep_config_serializes_with_ranks_list(self):
        cfg = SweepConfig(ranks=[8, 16, 32])
        hp = config_to_hyperparams(cfg)
        assert hp["ranks"] == [8, 16, 32]

    def test_excludes_private_fields(self):
        cfg = SFTConfig()
        hp = config_to_hyperparams(cfg)
        assert all(not k.startswith("_") for k in hp)


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_sft_config_no_warnings(self):
        cfg = SFTConfig(train_jsonl="train.jsonl", val_jsonl="val.jsonl")
        warnings = validate_config(cfg)
        assert warnings == []

    def test_zero_epochs_warning(self):
        cfg = SFTConfig(num_train_epochs=0, train_jsonl="train.jsonl")
        warnings = validate_config(cfg)
        assert any("num_train_epochs is 0" in w for w in warnings)

    def test_missing_train_jsonl_warning(self):
        cfg = SFTConfig(train_jsonl="")
        warnings = validate_config(cfg)
        assert any("train_jsonl is not set" in w for w in warnings)

    def test_qlora_with_zero_rank_warning(self):
        cfg = SFTConfig(use_4bit=True, lora_r=0, train_jsonl="train.jsonl")
        warnings = validate_config(cfg)
        assert any("QLoRA enabled but lora_r=0" in w for w in warnings)

    def test_full_sft_on_7b_warning(self):
        cfg = SFTConfig(use_4bit=False, train_jsonl="train.jsonl")
        warnings = validate_config(cfg)
        assert any("Full-parameter training on 7B+ model" in w for w in warnings)

    def test_qlora_on_7b_no_vram_warning(self):
        """QLoRA on 7B should not warn about VRAM."""
        cfg = SFTConfig(use_4bit=True, train_jsonl="train.jsonl")
        warnings = validate_config(cfg)
        assert not any("Full-parameter training" in w for w in warnings)

    def test_dpo_zero_beta_warning(self):
        cfg = DPOConfig(beta=0.0, train_jsonl="train.jsonl")
        warnings = validate_config(cfg)
        assert any("DPO beta must be positive" in w for w in warnings)

    def test_dpo_from_base_model_warning(self):
        cfg = DPOConfig(train_jsonl="train.jsonl")
        warnings = validate_config(cfg)
        assert any("DPO starting from a base" in w for w in warnings)

    def test_dpo_from_sft_checkpoint_no_warning(self):
        cfg = DPOConfig(
            train_jsonl="train.jsonl",
            sft_checkpoint="/some/checkpoint",
        )
        warnings = validate_config(cfg)
        assert not any("DPO starting from a base" in w for w in warnings)

    def test_sweep_empty_ranks_warning(self):
        cfg = SweepConfig(ranks=[])
        warnings = validate_config(cfg)
        assert any("no ranks" in w for w in warnings)

    def test_sweep_negative_rank_warning(self):
        cfg = SweepConfig(ranks=[8, -1])
        warnings = validate_config(cfg)
        assert any("rank r=-1" in w for w in warnings)

    def test_dpo_config_with_zero_beta_but_sft_checkpoint(self):
        """Even with an SFT checkpoint, beta=0 should warn."""
        cfg = DPOConfig(beta=0.0, sft_checkpoint="/ckpt", train_jsonl="train.jsonl")
        warnings = validate_config(cfg)
        assert any("DPO beta must be positive" in w for w in warnings)
