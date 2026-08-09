"""Stage 5 — Training matrix.

SFT (full / QLoRA), LoRA rank sweep, and DPO preference alignment.

Public API (re-exported from submodules):

- ``run_sft`` — single SFT run (full or QLoRA).
- ``run_dpo`` — single DPO run.
- ``run_lora_sweep`` — LoRA rank sweep across multiple ranks.
- ``SFTConfig``, ``DPOConfig``, ``SweepConfig`` — configuration dataclasses.
- ``TrainingResult``, ``SweepResult`` — result dataclasses.
- ``TrainingMethod`` — enum of supported methods.
- ``estimate_training_steps``, ``estimate_dpo_steps`` — step/memory estimates.
- ``TrainingUnavailableError``, ``DPOUnavailableError`` — graceful ML stack errors.
"""

from app.schemas.training import SweepResult, TrainingResult
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
from app.training.data import (
    DatasetStats,
    compute_stats,
    examples_to_dict_list,
    load_examples,
    load_stage3_dataset,
    make_hf_dataset,
)
from app.training.experiment import (
    generate_run_id,
    list_training_runs,
    load_training_run,
    persist_training_run,
)
from app.training.sweep import SweepReport, run_lora_sweep
from app.training.trainer_dpo import DPOUnavailableError, estimate_dpo_steps, run_dpo
from app.training.trainer_sft import TrainingUnavailableError, estimate_training_steps, run_sft

__all__ = [
    # Config
    "SFTConfig",
    "DPOConfig",
    "SweepConfig",
    "TrainingMethod",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_NUM_TRAIN_EPOCHS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_DPO_BETA",
    "DEFAULT_SWEEP_RANKS",
    # Data
    "DatasetStats",
    "compute_stats",
    "examples_to_dict_list",
    "load_examples",
    "load_stage3_dataset",
    "make_hf_dataset",
    # Experiment tracking
    "generate_run_id",
    "list_training_runs",
    "load_training_run",
    "persist_training_run",
    # Trainers
    "run_sft",
    "run_dpo",
    "run_lora_sweep",
    # Estimators
    "estimate_training_steps",
    "estimate_dpo_steps",
    # Errors
    "TrainingUnavailableError",
    "DPOUnavailableError",
    # Results
    "TrainingResult",
    "SweepResult",
    # Sweep helpers
    "SweepReport",
    # Validation
    "config_to_hyperparams",
    "validate_config",
]
