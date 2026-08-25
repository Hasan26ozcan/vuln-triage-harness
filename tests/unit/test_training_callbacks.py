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
from unittest.mock import MagicMock, patch

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
        with patch("torch.cuda.is_available", return_value=False):
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


class TestResourceTrackerCudaError:
    """Tests for CUDA RuntimeError handling in ResourceTracker (lines 96-97, 109-110, 120-123)."""

    def test_post_init_cuda_reset_peak_memory_runtime_error(self):
        """__post_init__ catches RuntimeError when reset_peak_memory_stats fails."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.reset_peak_memory_stats.side_effect = RuntimeError("CUDA driver not ready")
        with patch.dict("sys.modules", {"torch": fake_torch}):
            tracker = ResourceTracker()
        assert tracker._torch_available is True
        assert tracker._torch_cuda_available is True

    def test_start_cuda_reset_peak_memory_runtime_error(self):
        """start() catches RuntimeError when reset_peak_memory_stats fails."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.reset_peak_memory_stats.side_effect = RuntimeError("driver not ready")
        with patch.dict("sys.modules", {"torch": fake_torch}):
            tracker = ResourceTracker()
            tracker._torch_cuda_available = True
            tracker.start()
        assert tracker.start_time > 0

    def test_record_peak_memory_runtime_error(self):
        """record_peak_memory() catches RuntimeError from max_memory_allocated."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.side_effect = RuntimeError("CUDA not initialized")
        with patch.dict("sys.modules", {"torch": fake_torch}):
            tracker = ResourceTracker()
            tracker._torch_cuda_available = True
            tracker.start()
            tracker.record_peak_memory()
        assert tracker.peak_vram_bytes == 0

    def test_record_peak_memory_updates_on_success(self):
        """record_peak_memory() updates peak when no error."""
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.max_memory_allocated.return_value = 5 * 1024**3
        with patch.dict("sys.modules", {"torch": fake_torch}):
            tracker = ResourceTracker()
            tracker._torch_cuda_available = True
            tracker.start()
            tracker.record_peak_memory()
        assert tracker.peak_vram_bytes == 5 * 1024**3


# ---------------------------------------------------------------------------
# ResourceTracker — torch paths (lines 94-97, 102-104, 108-112)
# ---------------------------------------------------------------------------


class TestResourceTrackerTorch:
    def test_torch_available_cuda_true(self):
        """Lines 91-94: torch import succeeds, CUDA is available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker = ResourceTracker()
            assert tracker._torch_available is True
            assert tracker._torch_cuda_available is True
            # reset_peak_memory_stats was called in __post_init__
            mock_torch.cuda.reset_peak_memory_stats.assert_called()

    def test_torch_available_cuda_false(self):
        """Lines 94-97: torch import succeeds, CUDA not available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker = ResourceTracker()
            assert tracker._torch_available is True
            assert tracker._torch_cuda_available is False

    def test_start_with_cuda(self):
        """Lines 102-104: start() calls reset_peak_memory_stats when CUDA available."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker = ResourceTracker()
            tracker.start()
            assert tracker.start_time > 0.0

    def test_record_peak_memory_with_cuda(self):
        """Lines 108-112: record_peak_memory updates peak_vram_bytes."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 8 * 1024**3  # 8 GB

        with patch.dict("sys.modules", {"torch": mock_torch}):
            tracker = ResourceTracker()
            tracker.record_peak_memory()
            assert tracker.peak_vram_bytes == 8 * 1024**3
            assert tracker.peak_vram_gb == pytest.approx(8.0, rel=1e-5)


class TestResourceTrackerNoTorch:
    def test_torch_import_error(self):
        """Lines 95-97: ImportError when torch is not installed."""
        with patch.dict("sys.modules", {"torch": None}):
            tracker = ResourceTracker()
            assert tracker._torch_available is False
            assert tracker._torch_cuda_available is False


# ---------------------------------------------------------------------------
# WandbCallback — non-mock paths (lines 168-188, 196-203, 212, 219-224,
#                                  234, 245-251, 257-263)
# ---------------------------------------------------------------------------


class TestWandbCallbackNonMock:
    def test_on_init_real_wandb(self):
        """Lines 169-178: wandb import succeeds, wandb.init is called."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False, run_name="my_run")
            cb.on_init({"method": "sft_qlora", "base_model": "test"})
            assert cb._initialized is True
            assert cb._run_name == "my_run"
            mock_wandb.init.assert_called_once()

    def test_on_init_wandb_import_error_fallback(self):
        """Lines 179-183: ImportError when wandb not installed → fallback to mock."""
        with patch.dict("sys.modules", {"wandb": None}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            assert cb.mock is True
            assert cb._initialized is True

    def test_on_init_wandb_init_exception_fallback(self):
        """Lines 184-188: wandb.init raises → fallback to mock via Exception."""
        mock_wandb = MagicMock()
        mock_wandb.init.side_effect = RuntimeError("connection refused")
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            assert cb.mock is True
            assert cb._initialized is True

    def test_on_init_real_no_run_name(self):
        """Line 171: run_name defaults to f'sft_{int(time.time())}'."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False, run_name=None)
            cb.on_init({"method": "sft_qlora"})
            assert cb._run_name is not None
            assert cb._run_name.startswith("sft_")

    def test_on_step_not_initialized(self):
        """Line 191-192: on_step returns early when not initialized."""
        cb = WandbCallback(mock=False)
        # on_step without calling on_init first
        cb.on_step(1, loss=0.5)
        assert cb.calls == []

    def test_on_step_non_mock(self):
        """Lines 196-201: non-mock on_step calls wandb.log."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_step(1, loss=0.5)
            mock_wandb.log.assert_called_with({"train/loss": 0.5}, step=1)

    def test_on_step_non_mock_log_exception(self):
        """Lines 202-203: wandb.log exception is caught and suppressed."""
        mock_wandb = MagicMock()
        mock_wandb.log.side_effect = RuntimeError("W&B down")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_step(1, loss=0.5)
            # Should not raise
            mock_wandb.log.assert_called_once()

    def test_on_epoch_not_initialized(self):
        """Line 212: on_epoch returns early when not initialized."""
        cb = WandbCallback(mock=False)
        cb.on_epoch(epoch=1, train_loss=0.3, val_loss=0.4)
        assert cb.calls == []

    def test_on_epoch_non_mock(self):
        """Lines 219-223: non-mock on_epoch calls wandb.log with metrics."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_epoch(epoch=1, train_loss=0.3, val_loss=0.4)
            call_args = mock_wandb.log.call_args
            assert call_args[0][0]["train/loss"] == 0.3
            assert call_args[0][0]["validation/loss"] == 0.4

    def test_on_epoch_non_mock_without_val_loss(self):
        """on_epoch non-mock with val_loss=None: no validation key."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_epoch(epoch=1, train_loss=0.3)
            call_args = mock_wandb.log.call_args
            assert "validation/loss" not in call_args[0][0]

    def test_on_epoch_non_mock_log_exception(self):
        """Lines 220-225: wandb.log exception is caught in on_epoch."""
        mock_wandb = MagicMock()
        mock_wandb.log.side_effect = RuntimeError("W&B down")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_epoch(epoch=1, train_loss=0.3)
            # Should not raise

    def test_on_train_end_not_initialized(self):
        """Line 234: on_train_end returns early when not initialized."""
        cb = WandbCallback(mock=False)
        cb.on_train_end(0.01, 0.02, 6.0, 10.0)
        assert cb.calls == []

    def test_on_train_end_non_mock(self):
        """Lines 245-249: non-mock on_train_end calls wandb.log and wandb.finish."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_train_end(0.05, 0.08, peak_vram_gb=6.0, train_time_minutes=12.0)
            mock_wandb.log.assert_called_once()
            mock_wandb.finish.assert_called_once()

    def test_on_train_end_non_mock_without_val_loss(self):
        """on_train_end non-mock with final_val_loss=None."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_train_end(0.05)
            call_args = mock_wandb.log.call_args
            assert "validation/final_loss" not in call_args[0][0]

    def test_on_train_end_non_mock_exception(self):
        """Lines 250-251: on_train_end exception is caught."""
        mock_wandb = MagicMock()
        mock_wandb.log.side_effect = RuntimeError("W&B down")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_train_end(0.05)
            # Should not raise

    def test_on_error_non_mock(self):
        """Lines 257-261: non-mock on_error calls wandb.log and wandb.finish."""
        mock_wandb = MagicMock()
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_error("something broke")
            mock_wandb.log.assert_called_with({"error": "something broke"})
            mock_wandb.finish.assert_called_once()

    def test_on_error_non_mock_exception(self):
        """Lines 262-263: on_error exception is caught."""
        mock_wandb = MagicMock()
        mock_wandb.log.side_effect = RuntimeError("W&B down")

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            cb = WandbCallback(mock=False)
            cb.on_init({"method": "sft_qlora"})
            cb.on_error("something broke")
            # Should not raise


# ---------------------------------------------------------------------------
# CheckpointCallback — non-mock path (lines 338-362)
# ---------------------------------------------------------------------------


class TestCheckpointCallbackNonMock:
    def test_save_checkpoint_uploads_files(self, tmp_path):
        """Lines 338-359: non-mock save walks dir, uploads each file to S3."""
        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter.safetensors").write_bytes(b"model weights")
        (ckpt_dir / "config.json").write_text("{}", encoding="utf-8")

        mock_client = MagicMock()
        with (
            patch("app.storage.object_store.get_client", return_value=mock_client),
            patch("app.training.callbacks.logger"),
        ):
            cb = CheckpointCallback(mock=False, bucket="my-bucket", prefix="models/stage5")
            uri = cb.save_checkpoint("run_123", str(ckpt_dir), epoch=1)

            assert uri == "s3://my-bucket/models/stage5/run_123/epoch_1"
            assert mock_client.put_object.call_count == 2
            assert len(cb.saved_paths) == 1

    def test_save_checkpoint_upload_exception_returns_uri(self, tmp_path):
        """Lines 360-362: if get_client() or upload raises, URI is still returned."""
        ckpt_dir = tmp_path / "checkpoint"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter.safetensors").write_bytes(b"weights")

        with (
            patch(
                "app.storage.object_store.get_client", side_effect=ConnectionError("S3 unreachable")
            ),
            patch("app.training.callbacks.logger"),
        ):
            cb = CheckpointCallback(mock=False)
            uri = cb.save_checkpoint("run_123", str(ckpt_dir), epoch=5)

            assert uri == "s3://vuln-triage/checkpoints/stage5/run_123/epoch_5"


# ---------------------------------------------------------------------------
# ProgressCallback — verbose paths (lines 421-422, 439-440, 451)
# ---------------------------------------------------------------------------


class TestProgressCallbackVerbose:
    def test_on_step_verbose_with_loss(self):
        """Lines 421-422: step % 10 == 0 with loss renders formatted string."""
        cb = ProgressCallback(total_steps=100, verbose=True)
        cb.on_step(10, loss=0.42)
        assert cb.calls[0]["step"] == 10

    def test_on_step_verbose_without_loss(self):
        """Line 421: step % 10 == 0 with loss=None renders '—'."""
        cb = ProgressCallback(total_steps=100, verbose=True)
        cb.on_step(20, loss=None)
        assert cb.calls[0]["loss"] is None

    def test_on_epoch_verbose_with_val_loss(self):
        """Line 439: verbose on_epoch with val_loss renders formatted float."""
        cb = ProgressCallback(verbose=True)
        cb.on_epoch(1, train_loss=0.3, val_loss=0.4)
        assert cb.calls[-1]["val_loss"] == 0.4

    def test_on_epoch_verbose_without_val_loss(self):
        """Line 439: verbose on_epoch with val_loss=None renders '—'."""
        cb = ProgressCallback(verbose=True)
        cb.on_epoch(1, train_loss=0.3, val_loss=None)
        assert cb.calls[-1]["val_loss"] is None

    def test_on_train_end_verbose(self):
        """Line 451: verbose on_train_end logs completion message."""
        cb = ProgressCallback(verbose=True)
        cb.on_train_end(0.01, 0.02, 6.0, 10.0)
        assert cb.calls[-1]["final_train_loss"] == 0.01

    def test_on_step_non_verbose_multiple_steps(self):
        """Ensure step tracking works without verbose for non-multiple-of-10."""
        cb = ProgressCallback(total_steps=100, verbose=False)
        cb.on_step(3, loss=0.5)
        cb.on_step(7, loss=0.4)
        assert cb._last_step == 7
        assert len(cb.calls) == 2
