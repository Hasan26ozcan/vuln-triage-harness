"""Training run metadata contract, populated in Stage 5 (SFT / LoRA sweep / DPO)."""

from typing import Literal

from pydantic import BaseModel


class TrainingRun(BaseModel):
    id: str
    method: Literal["sft_full", "sft_qlora", "lora", "dpo"]
    base_model: str
    hyperparams: dict  # rank, alpha, lr, epochs, quant_bits, ...
    train_set_size: int
    train_time_minutes: float
    peak_vram_gb: float
    final_train_loss: float
    checkpoint_uri: str  # MinIO/S3 path
