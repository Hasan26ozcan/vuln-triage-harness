"""Unit tests for Stage 5 training callbacks.

Covers:
  - WandbCallback (mock mode): call recording, no API access.
  - CheckpointCallback (mock mode): checkpoint URI generation, call recording.
  - ProgressCallback: call recording, last step tracking.
  - ResourceTracker: elapsed time, peak VRAM (mocked for CPU).
  - TrainingCallback protocol conformance.
"""

from __future__ import annotations

import time

import pytest

from app.training.callbacks import (
    CheckpointCallback,
    ProgressCallback,
    ResourceTracker,
    WandbCallback,
)

# ---------------------------------------------------------------------------
# WandbCallback (mock mode)
# ---------------------------------------------------------------------------


class TestWandbCallbackMock:
    def test_init_mock_defaults(self):
        cb = WandbCallback(mock=True)
        assert cb.mock is True
        assert cb.calls == []
        assert cb._initialized is False

    def test_on_init_records_call(self):
        cb = WandbCallback(mock=True, run_name="test_run")
        cb.on_init({"method": "sft_qlora", "base_model": "Qwen/Qwen2.5-Coder-7B"})
        assert cb._initialized is True
        assert len(cb.calls) == 1
        assert cb.calls[0]["event"] == "init"
        assert cb.calls[0]["config"]["method"] == "sft_qlora"

    def test_on_step_records_call(self):
        cb = WandbCallback(mock=True)
        cb.on_init({"method": "sft_qlora"})
        cb.on_step(1, loss=0.5)
        assert len(cb.calls) == 2
        assert cb.calls[1]["event"] == "step"
        assert cb.calls[1]["step"] == 1
        assert cb.calls[1]["loss"] == 0.5

    def test_on_step_without_init_is_noop(self):
        """on_step before on_init should be ignored (not initialized)."""
        cb = WandbCallback(mock=True)
        cb.on_step(1, loss=0.5)
        assert cb.calls == []

    def test_on_epoch_records_call(self):
        cb = WandbCallback(mock=True)
        cb.on_init({"method": "sft_qlora"})
        cb.on_epoch(epoch=1, train_loss=0.3, val_loss=0.4)
        assert len(cb.calls) == 2
        event = cb.calls[1]
        assert event["event"] == "epoch"
        assert event["train/loss"] == 0.3
        assert event["validation/loss"] == 0.4

    def test_on_epoch_without_val_loss(self):
        cb = WandbCallback(mock=True)
        cb.on_init({"method": "sft_qlora"})
        cb.on_epoch(epoch=1, train_loss=0.3)
        event = cb.calls[1]
        assert event["event"] == "epoch"
        assert event["train/loss"] == 0.3
        assert "validation/loss" not in event

    def test_on_train_end_records_call(self):
        cb = WandbCallback(mock=True)
        cb.on_init({"method": "sft_qlora"})
        cb.on_train_end(
            final_train_loss=0.05,
            final_val_loss=0.08,
            peak_vram_gb=6.2,
            train_time_minutes=12.5,
        )
        assert len(cb.calls) == 2
        event = cb.calls[1]
        assert event["event"] == "train_end"
        assert event["train/final_loss"] == 0.05
        assert event["validation/final_loss"] == 0.08
        assert event["train/peak_vram_gb"] == 6.2

    def test_on_train_end_without_val_loss(self):
        cb = WandbCallback(mock=True)
        cb.on_init({"method": "sft_qlora"})
        cb.on_train_end(final_train_loss=0.05)
        event = cb.calls[1]
        assert "validation/final_loss" not in event

    def test_on_error_records_call(self):
        cb = WandbCallback(mock=True)
        cb.on_init({"method": "sft_qlora"})
        cb.on_error("Something went wrong")
        assert len(cb.calls) == 2
        assert cb.calls[1]["event"] == "error"
        assert cb.calls[1]["error"] == "Something went wrong"


# ---------------------------------------------------------------------------
# CheckpointCallback (mock mode)
# ---------------------------------------------------------------------------


class TestCheckpointCallbackMock:
    def test_init_defaults(self):
        cb = CheckpointCallback(mock=True)
        assert cb.bucket == "vuln-triage"
        assert cb.prefix == "checkpoints/stage5"
        assert cb.mock is True
        assert cb.checkpoints == []
        assert cb.saved_paths == []

    def test_save_checkpoint_returns_s3_uri(self):
        cb = CheckpointCallback(mock=True, bucket="my-bucket", prefix="models/stage5")
        uri = cb.save_checkpoint("run_123", "/local/ckpt", epoch=3)
        assert uri == "s3://my-bucket/models/stage5/run_123/epoch_3"

    def test_save_checkpoint_records_call(self):
        cb = CheckpointCallback(mock=True)
        cb.save_checkpoint("run_123", "/local/ckpt", epoch=1)
        assert len(cb.checkpoints) == 1
        cp = cb.checkpoints[0]
        assert cp["event"] == "checkpoint"
        assert cp["run_id"] == "run_123"
        assert cp["key"] == "checkpoints/stage5/run_123/epoch_1"
        assert cp["epoch"] == 1
        assert cp["checkpoint_dir"] == "/local/ckpt"

    def test_saved_paths_returns_list(self):
        cb = CheckpointCallback(mock=True)
        cb.save_checkpoint("run_a", "/ckpt_a", epoch=1)
        cb.save_checkpoint("run_b", "/ckpt_b", epoch=1)
        paths = cb.saved_paths
        assert len(paths) == 2
        assert paths[0] == "s3://vuln-triage/checkpoints/stage5/run_a/epoch_1"
        assert paths[1] == "s3://vuln-triage/checkpoints/stage5/run_b/epoch_1"

    def test_saved_paths_returns_copy(self):
        """Mutating the returned list should not affect internal state."""
        cb = CheckpointCallback(mock=True)
        cb.save_checkpoint("run_a", "/ckpt", epoch=1)
        paths = cb.saved_paths
        paths.append("s3://fake/extra")
        assert len(cb.saved_paths) == 1  # still 1

    def test_protocol_methods_are_noops(self):
        """CheckpointCallback implements all TrainingCallback methods as noops."""
        cb = CheckpointCallback(mock=True)
        cb.on_init({"method": "sft_qlora"})
        cb.on_step(1, loss=0.5)
        cb.on_epoch(1, 0.3)
        cb.on_train_end(0.01, 0.02, 6.0, 10.0)
        cb.on_error("error")
        # No exceptions, no state changes
        assert cb.checkpoints == []


# ---------------------------------------------------------------------------
# ProgressCallback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_init_defaults(self):
        cb = ProgressCallback()
        assert cb.total_steps == 0
        assert cb.verbose is False
        assert cb.calls == []
        assert cb._last_step == 0

    def test_on_init_no_error(self):
        cb = ProgressCallback(verbose=True)
        cb.on_init({"method": "dpo"})
        # No exception — verbose logs to logger

    def test_on_step_records_call(self):
        cb = ProgressCallback(total_steps=100)
        cb.on_step(5, loss=0.42)
        assert cb._last_step == 5
        assert len(cb.calls) == 1
        assert cb.calls[0]["step"] == 5
        assert cb.calls[0]["loss"] == 0.42

    def test_on_step_without_loss(self):
        cb = ProgressCallback()
        cb.on_step(3)
        assert cb.calls[0]["step"] == 3
        assert cb.calls[0]["loss"] is None

    def test_on_epoch_records_call(self):
        cb = ProgressCallback()
        cb.on_epoch(1, 0.3, 0.4)
        assert cb.calls[-1]["event"] == "epoch"
        assert cb.calls[-1]["train_loss"] == 0.3
        assert cb.calls[-1]["val_loss"] == 0.4

    def test_on_train_end_records_call(self):
        cb = ProgressCallback()
        cb.on_train_end(0.01, 0.02, 6.0, 10.0)
        assert cb.calls[-1]["event"] == "train_end"
        assert cb.calls[-1]["final_train_loss"] == 0.01

    def test_on_error_records_call(self):
        cb = ProgressCallback()
        cb.on_error("Failed")
        assert cb.calls[-1]["event"] == "error"
        assert cb.calls[-1]["error"] == "Failed"


# ---------------------------------------------------------------------------
# ResourceTracker
# ---------------------------------------------------------------------------


class TestResourceTracker:
    def test_init_without_torch(self):
        """ResourceTracker should work even when torch is not installed."""
        tracker = ResourceTracker()
        assert tracker.start_time == 0.0
        assert tracker.peak_vram_bytes == 0
        assert tracker.peak_vram_gb == 0.0
        assert tracker.elapsed_minutes == 0.0

    def test_start_sets_start_time(self):
        tracker = ResourceTracker()
        tracker.start()
        assert tracker.start_time > 0.0
        # elapsed_minutes should be near 0 right after start
        assert tracker.elapsed_minutes >= 0.0

    def test_elapsed_minutes_after_start(self):
        tracker = ResourceTracker()
        tracker.start()
        time.sleep(0.01)  # 10ms
        elapsed = tracker.elapsed_minutes
        assert elapsed >= 0.0
        # 10ms should be < 1 minute
        assert elapsed < 0.02

    def test_record_peak_memory_noop_without_gpu(self):
        """Without CUDA, record_peak_memory is a safe no-op."""
        tracker = ResourceTracker()
        tracker.start()
        tracker.record_peak_memory()
        assert tracker.peak_vram_bytes == 0
        assert tracker.peak_vram_gb == 0.0

    def test_peak_vram_gb_conversion(self):
        """Even with bytes set, peak_vram_gb should convert correctly."""
        tracker = ResourceTracker()
        # Simulate 6 GB in bytes
        tracker.peak_vram_bytes = 6 * 1024**3
        assert tracker.peak_vram_gb == pytest.approx(6.0, rel=1e-5)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify that concrete callbacks satisfy the TrainingCallback protocol.

    Python's Protocol is structural — we just check the methods exist.
    """

    def test_wandb_callback_conforms(self):
        cb = WandbCallback(mock=True)
        assert hasattr(cb, "on_init")
        assert hasattr(cb, "on_step")
        assert hasattr(cb, "on_epoch")
        assert hasattr(cb, "on_train_end")
        assert hasattr(cb, "on_error")

    def test_checkpoint_callback_conforms(self):
        cb = CheckpointCallback(mock=True)
        assert hasattr(cb, "on_init")
        assert hasattr(cb, "on_step")
        assert hasattr(cb, "on_epoch")
        assert hasattr(cb, "on_train_end")
        assert hasattr(cb, "on_error")

    def test_progress_callback_conforms(self):
        cb = ProgressCallback()
        assert hasattr(cb, "on_init")
        assert hasattr(cb, "on_step")
        assert hasattr(cb, "on_epoch")
        assert hasattr(cb, "on_train_end")
        assert hasattr(cb, "on_error")
