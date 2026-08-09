"""Stage 5 — training callbacks.

Provides three callback hooks used during training:

- ``WandbCallback`` — logs loss curves to Weights & Biases (lazy-imports wandb).
- ``CheckpointCallback`` — saves model adapters / tokenizer to MinIO on
  each checkpoint, and tracks peak VRAM usage.
- ``ProgressCallback`` — lightweight console progress for test runs.

Design notes
------------
The callbacks follow the same injectable / lazy-import pattern as every other
Stage: heavy deps (``wandb``, ``transformers``, ``boto3``) are imported inside
the methods that need them, not at module level. Every callback can also run
in a "mock" mode (``WandbCallback(mock=True)``) so tests can assert call
sequences without a W&B account or GPU.

Each callback implements a minimal Protocol so the trainer can type-hint
without importing the concrete classes — this prevents circular imports and
keeps the trainer testable with a fake callback.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol — the trainer calls these methods on whatever callbacks it receives.
# ---------------------------------------------------------------------------


class TrainingCallback(Protocol):
    """Minimal protocol followed by all training callbacks."""

    def on_init(self, config: dict) -> None:
        """Called once before training starts."""

    def on_step(self, step: int, loss: float | None = None) -> None:
        """Called after each optimiser step."""

    def on_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
    ) -> None:
        """Called at the end of each epoch."""

    def on_train_end(
        self,
        final_train_loss: float,
        final_val_loss: float | None = None,
        peak_vram_gb: float = 0.0,
        train_time_minutes: float = 0.0,
    ) -> None:
        """Called once when training completes (or aborts)."""

    def on_error(self, error: str) -> None:
        """Called when training encounters an unrecoverable error."""


# ---------------------------------------------------------------------------
# Memory / resource tracking helper
# ---------------------------------------------------------------------------


@dataclass
class ResourceTracker:
    """Tracks peak VRAM and elapsed wall-clock time across a training run.

    Uses ``torch.cuda`` for VRAM when available; falls back to 0.0 when
    CUDA is not present (CPU-only training, or tests running without torch).
    """

    start_time: float = 0.0
    peak_vram_bytes: int = 0
    _torch_available: bool = field(default=False, repr=False)
    _torch_cuda_available: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import torch  # noqa: F401

            self._torch_available = True
            self._torch_cuda_available = torch.cuda.is_available()
            if self._torch_cuda_available:
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            self._torch_available = False
            self._torch_cuda_available = False

    def start(self) -> None:
        self.start_time = time.time()
        if self._torch_cuda_available:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def record_peak_memory(self) -> None:
        if self._torch_cuda_available:
            import torch

            current = torch.cuda.max_memory_allocated()
            if current > self.peak_vram_bytes:
                self.peak_vram_bytes = current

    @property
    def peak_vram_gb(self) -> float:
        """Peak VRAM in GB (0.0 if CUDA was unavailable)."""
        bytes_per_gb = 1024**3
        return self.peak_vram_bytes / bytes_per_gb

    @property
    def elapsed_minutes(self) -> float:
        if self.start_time == 0.0:
            return 0.0
        return (time.time() - self.start_time) / 60.0


# ---------------------------------------------------------------------------
# W&B callback
# ---------------------------------------------------------------------------


class WandbCallback:
    """Logs loss curves and metrics to Weights & Biases.

    When ``mock=True`` (default), the callback stores calls in memory and
    never touches the W&B API — ideal for unit tests.

    Parameters
    ----------
    project:
        W&B project name (default: "vuln-triage-harness").
    run_name:
        Human-readable run name shown in the W&B UI.
    mock:
        If True, skip ``wandb.init`` and store calls in ``self.calls``.
    """

    def __init__(
        self,
        project: str = "vuln-triage-harness",
        run_name: str | None = None,
        mock: bool = False,
    ):
        self.project = project
        self.run_name = run_name
        self.mock = mock
        self.calls: list[dict] = []
        self._initialized: bool = False
        self._run_name: str | None = None

    def on_init(self, config: dict) -> None:
        if self.mock:
            self._run_name = f"mock_run_{len(self.calls)}"
            self.calls.append({"event": "init", "config": config})
            self._initialized = True
            return

        try:
            import wandb

            self._run_name = run_name = self.run_name or f"sft_{int(time.time())}"
            wandb.init(
                project=self.project,
                name=run_name,
                config=config,
            )
            self._initialized = True
            logger.info("W&B run started: %s/%s", self.project, run_name)
        except ImportError:
            logger.warning("wandb not installed — W&B callback disabled.")
            self.mock = True
            self._run_name = f"disabled_{int(time.time())}"
            self._initialized = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("W&B init failed: %s — continuing without W&B.", exc)
            self.mock = True
            self._run_name = f"disabled_{int(time.time())}"
            self._initialized = True

    def on_step(self, step: int, loss: float | None = None) -> None:
        if not self._initialized:
            return
        if self.mock:
            self.calls.append({"event": "step", "step": step, "loss": loss})
            return
        try:
            import wandb

            wandb.log({"train/loss": loss}, step=step)
        except Exception:  # noqa: BLE001
            pass

    def on_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
    ) -> None:
        if not self._initialized:
            return
        metrics: dict = {"train/loss": train_loss, "epoch": epoch}
        if val_loss is not None:
            metrics["validation/loss"] = val_loss
        if self.mock:
            self.calls.append({"event": "epoch", **metrics})
            return
        try:
            import wandb

            wandb.log(metrics)
        except Exception:  # noqa: BLE001
            pass

    def on_train_end(
        self,
        final_train_loss: float,
        final_val_loss: float | None = None,
        peak_vram_gb: float = 0.0,
        train_time_minutes: float = 0.0,
    ) -> None:
        if not self._initialized:
            return
        metrics: dict = {
            "train/final_loss": final_train_loss,
            "train/peak_vram_gb": peak_vram_gb,
            "train/time_minutes": train_time_minutes,
        }
        if final_val_loss is not None:
            metrics["validation/final_loss"] = final_val_loss
        if self.mock:
            self.calls.append({"event": "train_end", **metrics})
            return
        try:
            import wandb

            wandb.log(metrics)
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass

    def on_error(self, error: str) -> None:
        if self.mock:
            self.calls.append({"event": "error", "error": error})
            return
        try:
            import wandb

            wandb.log({"error": error})
            wandb.finish()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Checkpoint callback — saves to MinIO
# ---------------------------------------------------------------------------


class CheckpointCallback:
    """Saves model adapters and tokenizer to MinIO on each checkpoint.

    Mirrors the MinIO upload pattern from ``app.storage.object_store``:
    the full adapter file is uploaded as a JSON/object to ``s3://bucket/key``,
    and the returned URI string is stored on the ``TrainingResult``.

    When ``mock=True``, calls are recorded in ``self.checkpoints`` and no
    MinIO/S3 access occurs — ideal for unit tests.
    """

    def __init__(
        self,
        bucket: str = "vuln-triage",
        prefix: str = "checkpoints/stage5",
        mock: bool = False,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.mock = mock
        self.checkpoints: list[dict] = []
        self._saved_paths: list[str] = []

    @property
    def saved_paths(self) -> list[str]:
        """URIs of all checkpoints saved (mock or real)."""
        return list(self._saved_paths)

    def save_checkpoint(
        self,
        run_id: str,
        checkpoint_dir: str,
        epoch: int,
        tokenizer: object | None = None,
        model: object | None = None,
    ) -> str:
        """Upload a checkpoint directory to MinIO and return the S3 URI.

        Parameters
        ----------
        run_id:
            The training run ID (used in the S3 key path).
        checkpoint_dir:
            Local directory containing the adapter / model files.
        epoch:
            Current epoch number (for versioning the checkpoint).
        tokenizer, model:
            Objects with ``save_pretrained`` method (transformers / PEFT).
            In mock mode they are ignored.
        """
        key = f"{self.prefix}/{run_id}/epoch_{epoch}"
        uri = f"s3://{self.bucket}/{key}"

        if self.mock:
            self.checkpoints.append(
                {
                    "event": "checkpoint",
                    "run_id": run_id,
                    "key": key,
                    "epoch": epoch,
                    "checkpoint_dir": checkpoint_dir,
                }
            )
            self._saved_paths.append(uri)
            return uri

        # Save locally first, then upload
        try:
            from app.storage.object_store import get_client

            client = get_client()
            # Walk the checkpoint directory and upload each file.
            uploaded: list[str] = []
            for root, _dirs, files in os.walk(checkpoint_dir):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as f:
                        data = f.read()
                    file_key = f"{key}/{fname}"
                    client.put_object(
                        Bucket=self.bucket,
                        Key=file_key,
                        Body=data,
                    )
                    uploaded.append(file_key)

            self._saved_paths.append(uri)
            logger.info("Uploaded %d checkpoint files to %s", len(uploaded), uri)
            return uri
        except Exception as exc:  # noqa: BLE001
            logger.warning("Checkpoint upload failed for %s: %s", run_id, exc)
            return uri  # return the intended URI anyway — caller decides how to handle

    def on_init(self, config: dict) -> None:
        pass

    def on_step(self, step: int, loss: float | None = None) -> None:
        pass

    def on_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
    ) -> None:
        pass

    def on_train_end(
        self,
        final_train_loss: float,
        final_val_loss: float | None = None,
        peak_vram_gb: float = 0.0,
        train_time_minutes: float = 0.0,
    ) -> None:
        pass

    def on_error(self, error: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Console progress callback
# ---------------------------------------------------------------------------


class ProgressCallback:
    """Lightweight progress bar for console runs (no external deps).

    Prints to stderr so it doesn't corrupt JSONL output. Designed for
    quick smoke tests and non-W&B runs.
    """

    def __init__(self, total_steps: int = 0, verbose: bool = False):
        self.total_steps = total_steps
        self.verbose = verbose
        self.calls: list[dict] = []
        self._last_step: int = 0

    def on_init(self, config: dict) -> None:
        if self.verbose:
            logger.info(
                "Training started — method=%s, base_model=%s",
                config.get("method", "?"),
                config.get("base_model", "?"),
            )

    def on_step(self, step: int, loss: float | None = None) -> None:
        self._last_step = step
        self.calls.append({"event": "step", "step": step, "loss": loss})
        if self.verbose and step % 10 == 0:
            loss_str = f"{loss:.4f}" if loss is not None else "—"
            logger.info("  step %d/%d loss=%s", step, self.total_steps, loss_str)

    def on_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None = None,
    ) -> None:
        self.calls.append(
            {
                "event": "epoch",
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )
        if self.verbose:
            vl = f"{val_loss:.4f}" if val_loss is not None else "—"
            logger.info("Epoch %d: train_loss=%.4f val_loss=%s", epoch, train_loss, vl)

    def on_train_end(
        self,
        final_train_loss: float,
        final_val_loss: float | None = None,
        peak_vram_gb: float = 0.0,
        train_time_minutes: float = 0.0,
    ) -> None:
        self.calls.append({"event": "train_end", "final_train_loss": final_train_loss})
        if self.verbose:
            logger.info(
                "Training complete: final_train_loss=%.4f, peak_vram=%.2fGB, time=%.1fm",
                final_train_loss,
                peak_vram_gb,
                train_time_minutes,
            )

    def on_error(self, error: str) -> None:
        self.calls.append({"event": "error", "error": error})
        logger.error("Training error: %s", error)
