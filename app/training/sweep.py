"""Stage 5 — LoRA rank sweep.

Runs SFT (QLoRA) at multiple LoRA ranks and selects the best by validation
loss. The sweep is the central experiment of Stage 5: it answers the question
"what's the smallest LoRA rank that gives acceptable loss?" — a key input to
the quantization trade-off study in Stage 8.

The sweep delegates to ``app.training.trainer_sft.run_sft`` for each rank,
using injectable callbacks so tests can assert call sequences without a GPU.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.schemas.training import SweepResult, TrainingResult
from app.training.callbacks import WandbCallback
from app.training.config import SweepConfig
from app.training.trainer_sft import TrainingUnavailableError

logger = logging.getLogger(__name__)


@dataclass
class SweepReport:
    """Human-readable summary of a completed sweep."""

    sweep_name: str
    base_model: str
    num_runs: int
    best_rank: int | None
    best_val_loss: float | None
    best_checkpoint_uri: str
    all_runs: list[dict]  # one summary dict per run

    def to_dict(self) -> dict:
        return {
            "sweep_name": self.sweep_name,
            "base_model": self.base_model,
            "num_runs": self.num_runs,
            "best_rank": self.best_rank,
            "best_val_loss": self.best_val_loss,
            "best_checkpoint_uri": self.best_checkpoint_uri,
            "all_runs": self.all_runs,
        }


def run_lora_sweep(
    sweep_config: SweepConfig,
    *,
    callbacks_per_run: list | None = None,
    dry_run: bool = False,
    loader: Any | None = None,
    persist: bool = True,
) -> SweepResult:
    """Run a LoRA rank sweep: multiple QLoRA SFT runs at different ranks.

    Parameters
    ----------
    sweep_config:
        ``SweepConfig`` specifying base model, ranks, and shared hyperparams.
    callbacks_per_run:
        Callbacks passed to each individual ``run_sft`` call.
        If None, mock-W&B + progress callbacks are used.
    dry_run:
        If True, each rank returns an estimated ``TrainingResult`` without
        actually training (no GPU / torch required).
    loader:
        Injectable data loader for testing.
    persist:
        If True, persist each ``TrainingResult`` to PostgreSQL via
        ``app.training.experiment.persist_training_run``.
    """
    from app.training.experiment import persist_training_run
    from app.training.trainer_sft import run_sft

    configs = sweep_config.to_sft_configs()
    sweep_name = sweep_config.run_name or f"lora_sweep_{sweep_config.base_model.split('/')[-1]}"

    logger.info(
        "Starting LoRA rank sweep: ranks=%s, base_model=%s",
        sweep_config.ranks,
        sweep_config.base_model,
    )

    results: list[TrainingResult] = []
    for i, sft_config in enumerate(configs):
        run_label = f"{sweep_name}/rank_{sft_config.lora_r}"
        rank_callbacks = (
            callbacks_per_run
            if callbacks_per_run is not None
            else _default_sweep_callbacks(run_label)
        )

        logger.info("Sweep run %d/%d: rank=%d", i + 1, len(configs), sft_config.lora_r)

        try:
            result = run_sft(
                config=sft_config,
                callbacks=rank_callbacks,
                dry_run=dry_run,
                loader=loader,
            )
        except TrainingUnavailableError as exc:
            logger.warning("Rank %d skipped (training unavailable): %s", sft_config.lora_r, exc)
            # Create a failed result so the sweep continues
            result = TrainingResult(
                run_id=f"failed_{sft_config.lora_r}",
                method=sft_config.method_str,
                base_model=sft_config.base_model,
                hyperparams={"lora_r": sft_config.lora_r, "use_4bit": sft_config.use_4bit},
                train_set_size=0,
                train_time_minutes=0.0,
                peak_vram_gb=0.0,
                final_train_loss=float("nan"),
                final_val_loss=None,
                checkpoint_uri="",
                status="failed",
                run_name=run_label,
            )

        results.append(result)
        if persist and result.status == "completed":
            try:
                persist_training_run(result)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to persist run %s: %s", result.run_id, exc)

    # Select best by val_loss (lower is better); fall back to train_loss.
    best_result: TrainingResult | None = None
    best_val_loss: float | None = None
    for r in results:
        if r.final_val_loss is not None:
            if best_val_loss is None or r.final_val_loss < best_val_loss:
                best_val_loss = r.final_val_loss
                best_result = r
        elif r.final_train_loss < float("inf"):
            # No val loss — use train loss as fallback metric
            if best_result is None:
                best_result = r

    best_rank = best_result.hyperparams.get("lora_r") if best_result else None

    # Build per-run summaries
    summaries = [_summarize_run(r) for r in results]

    SweepReport(
        sweep_name=sweep_name,
        base_model=sweep_config.base_model,
        num_runs=len(results),
        best_rank=best_rank,
        best_val_loss=best_val_loss,
        best_checkpoint_uri=best_result.checkpoint_uri if best_result else "",
        all_runs=summaries,
    )

    logger.info("Sweep complete: best_rank=%s, best_val_loss=%.4f", best_rank, best_val_loss or 0.0)

    return SweepResult(
        base_model=sweep_config.base_model,
        sweep_name=sweep_name,
        results=results,
        best_rank=best_rank,
        best_val_loss=best_val_loss,
    )


def _default_sweep_callbacks(run_label: str) -> list:
    """Default callbacks for a single sweep run: mock-W&B + checkpoint + progress."""
    return [
        WandbCallback(run_name=run_label, mock=True),
    ]


def _summarize_run(result: TrainingResult) -> dict:
    """Produce a one-line summary dict for a sweep run."""
    return {
        "rank": result.hyperparams.get("lora_r", "?"),
        "status": result.status,
        "train_loss": (
            round(result.final_train_loss, 4) if result.final_train_loss is not None else None
        ),
        "val_loss": round(result.final_val_loss, 4) if result.final_val_loss is not None else None,
        "vram_gb": round(result.peak_vram_gb, 2),
        "minutes": round(result.train_time_minutes, 2),
        "checkpoint_uri": result.checkpoint_uri,
    }


# Re-export for convenience
__all__ = [
    "SweepReport",
    "run_lora_sweep",
]
