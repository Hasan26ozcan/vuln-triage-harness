"""Training run metadata contract, populated in Stage 5 (SFT / LoRA sweep / DPO).

Defines the structured metadata that gets persisted to PostgreSQL for every
training experiment, plus the lightweight ``TrainingResult`` dataclass that
the SFT / DPO / sweep trainers return at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel


class TrainingRun(BaseModel):
    """A single training experiment record.

    Populated by Stage 5 and read back by Stage 6 (evaluation) and
    Stage 10 (CI/CD regression gate) to identify which checkpoint to load.
    """

    id: str
    method: Literal["sft_full", "sft_qlora", "lora", "dpo"]
    base_model: str
    hyperparams: dict  # rank, alpha, lr, epochs, quant_bits, ...
    train_set_size: int
    train_time_minutes: float
    peak_vram_gb: float
    final_train_loss: float
    final_val_loss: float | None = None
    checkpoint_uri: str  # MinIO/S3 path
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    run_name: str | None = None  # human-readable label (e.g. "lora-r64-sweep")


@dataclass
class TrainingResult:
    """Runtime result returned by the SFT / DPO / sweep trainers.

    This is the in-memory counterpart of ``TrainingRun``. Trainers build it
    from loss curves and resource metrics, and the caller persists it to
    PostgreSQL via ``app.training.experiment.persist_training_run``.
    """

    run_id: str
    method: str
    base_model: str
    hyperparams: dict
    train_set_size: int
    train_time_minutes: float
    peak_vram_gb: float
    final_train_loss: float
    final_val_loss: float | None = None
    checkpoint_uri: str = ""
    status: str = "completed"
    run_name: str | None = None
    # Per-epoch loss history (for W&B / reporting).
    train_loss_history: list[float] = field(default_factory=list)


@dataclass
class SweepResult:
    """Result of a LoRA rank sweep — aggregates multiple ``TrainingResult`` records."""

    base_model: str
    sweep_name: str
    results: list[TrainingResult] = field(default_factory=list)
    best_rank: int | None = None
    best_val_loss: float | None = None

    @property
    def num_runs(self) -> int:
        return len(self.results)

    def summary(self) -> list[dict]:
        """One summary dict per sweep run, for reporting."""
        rows: list[dict] = []
        for r in self.results:
            rows.append(
                {
                    "rank": r.hyperparams.get("lora_r", "?"),
                    "train_loss": round(r.final_train_loss, 4),
                    "val_loss": round(r.final_val_loss, 4)
                    if r.final_val_loss is not None
                    else None,
                    "vram_gb": round(r.peak_vram_gb, 2),
                    "minutes": round(r.train_time_minutes, 2),
                    "checkpoint_uri": r.checkpoint_uri,
                }
            )
        return rows
