"""Stage 5+ — Training matrix with LoRA experiments, RAG, and patch generation.

SFT (full / QLoRA), LoRA rank sweep, DPO preference alignment,
Retrieval-Augmented Generation (RAG), and separate patch generation.

Public API (re-exported from submodules):

- ``run_sft`` — single SFT run (full or QLoRA).
- ``run_dpo`` — single DPO run.
- ``run_lora_sweep`` — LoRA rank sweep across multiple ranks.
- ``SFTConfig``, ``DPOConfig``, ``SweepConfig`` — configuration dataclasses.
- ``LoRAPreset``, ``LORA_PRESETS``, ``LORA_PRESETS_EXTENDED`` — named LoRA experiment presets.
- ``preset_to_sft_config`` — convert a preset to a concrete SFTConfig.
- ``TrainingResult``, ``SweepResult`` — result dataclasses.
- ``TrainingMethod`` — enum of supported methods.
- ``RAGRetriever``, ``build_rag_patch_prompt`` — retrieval-augmented generation.
- ``PatchGenerator``, ``PatchResult`` — separate patch generation strategy.
- ``estimate_training_steps``, ``estimate_dpo_steps`` — step/memory estimates.
- ``TrainingUnavailableError``, ``DPOUnavailableError`` — graceful ML stack errors.
"""

from app.schemas.training import SweepResult, TrainingResult
from app.training.config import (
    DEFAULT_BASE_MODEL,
    DEFAULT_DPO_BETA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_R,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DEFAULT_SWEEP_RANKS,
    LORA_PRESETS,
    LORA_PRESETS_EXTENDED,
    DPOConfig,
    LoRAPreset,
    SFTConfig,
    SweepConfig,
    TrainingMethod,
    config_to_hyperparams,
    get_preset,
    preset_to_sft_config,
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
from app.training.patch_generator import PatchGenerator, PatchResult, verify_patch_pattern

# RAG and patch generation (new)
from app.training.rag import RAGRetriever, build_rag_classification_prompt, build_rag_patch_prompt
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
    "DEFAULT_LORA_R",
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_DROPOUT",
    "DEFAULT_DPO_BETA",
    "DEFAULT_SWEEP_RANKS",
    "LORA_PRESETS",
    "LORA_PRESETS_EXTENDED",
    "LoRAPreset",
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
    # Validation & preset helpers
    "config_to_hyperparams",
    "validate_config",
    "preset_to_sft_config",
    "get_preset",
    # RAG (new)
    "RAGRetriever",
    "build_rag_patch_prompt",
    "build_rag_classification_prompt",
    # Patch generation (new)
    "PatchGenerator",
    "PatchResult",
    "verify_patch_pattern",
]
