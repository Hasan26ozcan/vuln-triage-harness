"""Stage 5 — training configuration dataclasses.

Mirrors the config pattern from Stage 4's ``BaselineConfig``: flat, immutable
dataclasses with sensible defaults drawn from the project's README tech-stack
table (Qwen2.5-Coder-7B-Instruct, PEFT LoRA/QLoRA, TRL DPOTrainer, bitsandbytes
4-bit NF4).

Heavy ML imports (``transformers``, ``peft``, ``bitsandbytes``, ``trl``) are
**never** performed at module-import time — they are imported lazily inside the
trainer functions so that the config module (and the CLI) work without GPU /
torch installed. This is the same lazy-import pattern used by
``QwenBackend._load`` (Stage 4) and ``TokenCounter._load`` (Stage 3).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — defaults from the project README tech-stack table
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_FAST_MODEL: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # for quick iteration

# LoRA defaults (PEFT)
DEFAULT_LORA_R: int = 64
DEFAULT_LORA_ALPHA: int = 16
DEFAULT_LORA_DROPOUT: float = 0.05

# bitsandbytes 4-bit (QLoRA)
DEFAULT_BNB_4BIT_COMPUTE_DTYPE: str = "float16"
DEFAULT_BNB_4BIT_QUANT_TYPE: str = "nf4"  # NormalFloat4 — best quality/size trade-off
DEFAULT_BNB_4BIT_USE_DOUBLE_QUANT: bool = True

# Training defaults
DEFAULT_LEARNING_RATE: float = 2e-5
DEFAULT_NUM_TRAIN_EPOCHS: int = 3
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE: int = 1  # small — fits 7B on consumer GPU
DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE: int = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS: int = 8
DEFAULT_WARMUP_RATIO: float = 0.03
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_MAX_GRAD_NORM: float = 0.3
DEFAULT_LEARNING_RATE_SCHEDULER: str = "cosine"  # transformers 5.x: cosine_with_warmup deprecated

# DPO defaults (TRL)
DEFAULT_DPO_BETA: float = 0.1
DEFAULT_DPO_LOSS_TYPE: str = "sigmoid"  # standard DPO loss

# LoRA rank sweep — the ranks tried in the sweep. These bracket the
# "useful parameter-efficient range" (rank 8–128) identified in the
# QLoRA paper (Dettmers et al., 2023 / arXiv:2305.14168).
DEFAULT_SWEEP_RANKS: list[int] = [8, 16, 32, 64, 128]

# Output / checkpoint directories
DEFAULT_OUTPUT_BASE: str = "./output/stage5"


class TrainingMethod(StrEnum):
    """Training methods supported by Stage 5."""

    SFT_FULL = "sft_full"  # full-parameter fine-tuning
    SFT_QLORA = "sft_qlora"  # SFT with 4-bit QLoRA (LoRA on top of 4-bit base)
    LORA = "lora"  # LoRA adapter fine-tuning (16-bit)
    DPO = "dpo"  # Direct Preference Optimization


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    """Configuration for SFT (Supervised Fine-Tuning) — full or QLoRA.

    When ``use_4bit`` is ``True`` the model is loaded in 4-bit NF4 via
    ``bitsandbytes`` and LoRA is applied on top (QLoRA). When ``False``, the
    model is loaded in full 16-bit precision and only LoRA adapters are trained
    (``lora_r`` / ``lora_alpha`` control the adapter; other params like
    ``learning_rate``, ``num_train_epochs`` are shared).

    Parameters mirror ``transformers.TrainingArguments`` + ``peft.LoraConfig``
    field names so they map directly when the lazy import happens inside the
    trainer.
    """

    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = DEFAULT_OUTPUT_BASE
    use_4bit: bool = True  # QLoRA by default — required for 8GB VRAM
    # LoRA adapter
    lora_r: int = DEFAULT_LORA_R
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    # Optimizer / schedule
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE
    per_device_eval_batch_size: int = DEFAULT_PER_DEVICE_EVAL_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    lr_scheduler_type: str = DEFAULT_LEARNING_RATE_SCHEDULER
    # 4-bit specifics (only used when use_4bit is True)
    bnb_4bit_compute_dtype: str = DEFAULT_BNB_4BIT_COMPUTE_DTYPE
    bnb_4bit_quant_type: str = DEFAULT_BNB_4BIT_QUANT_TYPE
    bnb_4bit_use_double_quant: bool = DEFAULT_BNB_4BIT_USE_DOUBLE_QUANT
    # Runtime
    train_jsonl: str = ""  # path to Stage 3 train.jsonl
    val_jsonl: str = ""  # path to Stage 3 val.jsonl
    run_name: str | None = None

    @property
    def method(self) -> TrainingMethod:
        if self.use_4bit:
            return TrainingMethod.SFT_QLORA
        elif self.lora_r > 0:
            return TrainingMethod.LORA  # CPU-compatible LoRA (no 4-bit)
        else:
            return TrainingMethod.SFT_FULL  # full-parameter, needs GPU

    @property
    def method_str(self) -> str:
        return self.method.value


@dataclass
class DPOConfig:
    """Configuration for DPO (Direct Preference Optimization) via TRL.

    DPO fine-tunes a model to prefer correct vulnerability classifications
    and patches over incorrect ones. The dataset must be a JSONL of
    ``InstructionExample`` records (from Stage 3) — each record's
    ``target_*`` fields serve as the "preferred" (chosen) response, and a
    naive baseline response serves as the "rejected" one.

    The ``beta`` parameter controls the KL penalty (lower = more aggressive
    deviation from the reference model). The standard value is 0.1.
    """

    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = DEFAULT_OUTPUT_BASE
    beta: float = DEFAULT_DPO_BETA
    loss_type: str = DEFAULT_DPO_LOSS_TYPE
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    lr_scheduler_type: str = DEFAULT_LEARNING_RATE_SCHEDULER
    # SFT checkpoint to initialize from (the model being DPO-tuned).
    # If empty, uses ``base_model`` directly (DPO from scratch).
    sft_checkpoint: str = ""
    # Runtime
    train_jsonl: str = ""
    run_name: str | None = None

    @property
    def method(self) -> TrainingMethod:
        return TrainingMethod.DPO

    @property
    def method_str(self) -> str:
        return self.method.value


@dataclass
class SweepConfig:
    """Configuration for the LoRA rank sweep.

    Runs multiple ``SFTConfig`` instances with the same hyperparameters except
    for ``lora_r``, which is varied across the ``ranks`` list. The best
    rank is selected by lowest validation loss.

    All runs use QLoRA (4-bit) so they fit within an 8GB VRAM GPU.
    """

    base_model: str = DEFAULT_BASE_MODEL
    output_dir: str = DEFAULT_OUTPUT_BASE
    ranks: list[int] = field(default_factory=lambda: list(DEFAULT_SWEEP_RANKS))
    train_jsonl: str = ""
    val_jsonl: str = ""
    # Shared across all sweep runs (override per-rank via the resulting
    # SFTConfig list returned by ``to_sft_configs``).
    learning_rate: float = DEFAULT_LEARNING_RATE
    num_train_epochs: int = DEFAULT_NUM_TRAIN_EPOCHS
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    run_name: str | None = None

    def to_sft_configs(self) -> list[SFTConfig]:
        """Expand the sweep into one ``SFTConfig`` per rank."""
        configs: list[SFTConfig] = []
        for r in self.ranks:
            name = self.run_name or f"lora-r{r}"
            configs.append(
                SFTConfig(
                    base_model=self.base_model,
                    output_dir=f"{self.output_dir}/lora-r{r}",
                    use_4bit=True,
                    lora_r=r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    learning_rate=self.learning_rate,
                    num_train_epochs=self.num_train_epochs,
                    train_jsonl=self.train_jsonl,
                    val_jsonl=self.val_jsonl,
                    run_name=name,
                )
            )
        return configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def config_to_hyperparams(config: SFTConfig | DPOConfig | SweepConfig) -> dict:
    """Serialise a config dataclass to a JSON-safe dict for the TrainingRun record."""
    # SweepConfig has a list field (``ranks``) — asdict handles it fine.
    return {k: v for k, v in asdict(config).items() if not k.startswith("_")}


def _validate_sft_config(config: SFTConfig) -> list[str]:
    """Return validation warnings specific to ``SFTConfig``."""
    warnings: list[str] = []
    if config.use_4bit and config.lora_r == 0:
        warnings.append("QLoRA enabled but lora_r=0 — no LoRA adapters will be applied.")
    if not config.use_4bit and "7B" in config.base_model:
        warnings.append(
            "Full-parameter training on 7B+ model requires significant VRAM. "
            "Consider QLoRA (use_4bit=True) for 8GB GPUs."
        )
    return warnings


def _validate_dpo_config(config: DPOConfig) -> list[str]:
    """Return validation warnings specific to ``DPOConfig``."""
    warnings: list[str] = []
    if config.beta <= 0:
        warnings.append("DPO beta must be positive; results may be degenerate.")
    if config.sft_checkpoint == "" and config.base_model == DEFAULT_BASE_MODEL:
        warnings.append(
            "DPO starting from a base (untrained) model — "
            "typically you want to DPO-tune an SFT checkpoint."
        )
    return warnings


def _validate_sweep_config(config: SweepConfig) -> list[str]:
    """Return validation warnings specific to ``SweepConfig``."""
    warnings: list[str] = []
    if not config.ranks:
        warnings.append("SweepConfig has no ranks to try — the sweep will be empty.")
        return warnings
    for r in config.ranks:
        if r < 1:
            warnings.append(f"Sweep rank r={r} is < 1 — skipped by the trainer.")
    return warnings


def _config_warnings_common(config: SFTConfig | DPOConfig | SweepConfig) -> list[str]:
    """Return generic validation warnings shared by all config types."""
    warnings: list[str] = []
    epochs = getattr(config, "num_train_epochs", 0)
    if epochs < 1:
        warnings.append("num_train_epochs is 0 — no training will occur.")
    if config.train_jsonl == "":
        warnings.append("train_jsonl is not set — the trainer will have no data to load.")
    return warnings


def validate_config(config: SFTConfig | DPOConfig | SweepConfig) -> list[str]:
    """Return a list of validation warnings for the given config.

    Does **not** raise — callers decide whether to fail or warn. Each
    warning is a human-readable string.
    """
    warnings = _config_warnings_common(config)

    if isinstance(config, SFTConfig):
        warnings.extend(_validate_sft_config(config))
    elif isinstance(config, DPOConfig):
        warnings.extend(_validate_dpo_config(config))
    elif isinstance(config, SweepConfig):
        warnings.extend(_validate_sweep_config(config))

    return warnings
