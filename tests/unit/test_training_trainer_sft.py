"""Unit tests for Stage 5 SFT trainer.

Covers:
  - TrainingUnavailableError: raised by _check_can_train when ML stack is missing.
  - estimate_training_steps: pure arithmetic (no GPU / torch required).
  - training_run_id_is_valid: path existence + non-empty check.
  - run_sft(dry_run=True): returns a TrainingResult with step/memory estimates.
  - run_sft raises FileNotFoundError when train_jsonl is missing.
  - run_sft raises TrainingUnavailableError when not dry_run and ML stack missing.

All tests run without a GPU or model downloads — the dry-run path is pure
arithmetic and the real-training path is gated behind _check_can_train.
"""

from __future__ import annotations

import pytest

from app.schemas.dataset import InstructionExample
from app.schemas.training import TrainingResult
from app.training.config import SFTConfig
from app.training.trainer_sft import (
    StepEstimate,
    TrainingUnavailableError,
    estimate_training_steps,
    run_sft,
    training_run_id_is_valid,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_example(
    cwe: str = "CWE-89",
    severity: str = "high",
    explanation: str = "SQL injection.",
    prompt: str = "Classify this vulnerability.",
    token_count: int = 100,
) -> InstructionExample:
    return InstructionExample(
        id="ie_001",
        sample_id="vs_001",
        prompt=prompt,
        target_cwe=cwe,
        target_severity=severity,
        target_explanation=explanation,
        target_patch_diff=None,
        token_count_estimate=token_count,
    )


class _MockLoader:
    """Injectable loader that returns pre-built examples."""

    def __init__(
        self,
        train: list[InstructionExample],
        val: list[InstructionExample] | None = None,
    ):
        self._train = train
        self._val = val or []
        self.train_called = False
        self.val_called = False

    def load(self, path: str) -> list[InstructionExample]:
        if "train" in path:
            self.train_called = True
            return self._train
        if "val" in path:
            self.val_called = True
            return self._val
        return []


def _write_jsonl(path: str, examples: list[InstructionExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# training_run_id_is_valid
# ---------------------------------------------------------------------------


class TestTrainingRunIdIsValid:
    def test_valid_existing_path(self, tmp_path):
        path = tmp_path / "train.jsonl"
        path.write_text("{}")
        assert training_run_id_is_valid(str(path)) is True

    def test_empty_string(self):
        assert training_run_id_is_valid("") is False

    def test_nonexistent_path(self):
        assert training_run_id_is_valid("does/not/exist.jsonl") is False


# ---------------------------------------------------------------------------
# estimate_training_steps
# ---------------------------------------------------------------------------


class TestEstimateTrainingSteps:
    def test_basic_calculation(self):
        config = SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            use_4bit=True,
        )
        estimate = estimate_training_steps(config, n_train_examples=128)
        # steps_per_epoch = 128 // 1 = 128
        # optim_steps_per_epoch = 128 // 8 = 16
        # total = 16 * 3 = 48
        assert estimate.steps_per_epoch == 16
        assert estimate.num_train_steps == 48
        assert estimate.num_epochs == 3
        assert estimate.gradient_accumulation_steps == 8

    def test_qlora_fits_in_8gb(self):
        config = SFTConfig(use_4bit=True)
        estimate = estimate_training_steps(config, n_train_examples=100)
        # QLoRA: 6.0 + 1.0 = 7.0 GB → fits in 8GB
        assert estimate.estimated_vram_gb == 7.0
        assert estimate.can_fit_in_8gb is True

    def test_full_sft_does_not_fit_in_8gb(self):
        config = SFTConfig(use_4bit=False)
        estimate = estimate_training_steps(config, n_train_examples=100)
        # Full SFT: 15.0 + 1.0 = 16.0 GB → does NOT fit in 8GB
        assert estimate.estimated_vram_gb == 16.0
        assert estimate.can_fit_in_8gb is False

    def test_small_dataset(self):
        """When examples < batch_size, at least 1 step per epoch."""
        config = SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
        )
        estimate = estimate_training_steps(config, n_train_examples=2)
        # steps_per_epoch = max(1, 2 // 4) = 1
        # optim_steps_per_epoch = max(1, 1 // 2) = 1
        assert estimate.steps_per_epoch == 1
        assert estimate.num_train_steps == 1

    def test_returns_step_estimate(self):
        config = SFTConfig()
        result = estimate_training_steps(config, n_train_examples=100)
        assert isinstance(result, StepEstimate)


# ---------------------------------------------------------------------------
# TrainingUnavailableError
# ---------------------------------------------------------------------------


class TestTrainingUnavailableError:
    def test_is_runtime_error(self):
        err = TrainingUnavailableError("test message")
        assert isinstance(err, RuntimeError)
        assert "test message" in str(err)


# ---------------------------------------------------------------------------
# run_sft — dry_run mode
# ---------------------------------------------------------------------------


class TestRunSftDryRun:
    def test_dry_run_returns_completed_result(self, tmp_path):
        """In dry-run mode, run_sft returns a TrainingResult with status='dry_run'."""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        examples = [_make_example() for _ in range(5)]
        _write_jsonl(str(train_path), examples)
        _write_jsonl(str(val_path), examples[:2])

        config = SFTConfig(
            train_jsonl=str(train_path),
            val_jsonl=str(val_path),
            use_4bit=True,
            lora_r=8,
            num_train_epochs=1,
        )
        result = run_sft(config, dry_run=True)

        assert isinstance(result, TrainingResult)
        assert result.status == "dry_run"
        assert result.train_set_size == 5
        assert result.final_train_loss == 0.0
        assert result.final_val_loss is None
        assert result.checkpoint_uri == ""

    def test_dry_run_sets_peak_vram_estimate(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        examples = [_make_example() for _ in range(5)]
        _write_jsonl(str(train_path), examples)

        config = SFTConfig(train_jsonl=str(train_path), use_4bit=True)
        result = run_sft(config, dry_run=True)
        assert result.peak_vram_gb == 7.0  # QLoRA estimate

    def test_dry_run_with_full_sft(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        examples = [_make_example() for _ in range(5)]
        _write_jsonl(str(train_path), examples)

        config = SFTConfig(train_jsonl=str(train_path), use_4bit=False)
        result = run_sft(config, dry_run=True)
        assert result.peak_vram_gb == 16.0  # full SFT estimate
        assert result.method == "sft_full"

    def test_dry_run_with_injected_loader(self, tmp_path):
        """Loader is injected; the JSONL file must exist (path checked first)."""
        examples = [_make_example() for _ in range(3)]
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), examples)
        mock_loader = _MockLoader(train=examples)

        config = SFTConfig(train_jsonl=str(train_path), use_4bit=True)
        result = run_sft(config, dry_run=True, loader=mock_loader)
        assert result.train_set_size == 3
        assert result.status == "dry_run"

    def test_dry_run_generates_run_id(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = SFTConfig(train_jsonl=str(train_path))
        result = run_sft(config, dry_run=True)
        assert result.run_id.startswith("sft_qlora_")

    def test_dry_run_callbacks_notified(self, tmp_path):
        """Callbacks should receive on_init in dry-run mode."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        class _SpyCallback:
            def __init__(self):
                self.init_calls = []
                self.train_end_calls = []

            def on_init(self, config: dict) -> None:
                self.init_calls.append(config)

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
                self.train_end_calls.append(
                    {
                        "final_train_loss": final_train_loss,
                        "peak_vram_gb": peak_vram_gb,
                    }
                )

            def on_error(self, error: str) -> None:
                pass

        spy = _SpyCallback()
        config = SFTConfig(train_jsonl=str(train_path))
        run_sft(config, dry_run=True, callbacks=[spy])
        assert len(spy.init_calls) == 1
        assert spy.init_calls[0]["dry_run"] is True


# ---------------------------------------------------------------------------
# run_sft — error handling
# ---------------------------------------------------------------------------


class TestRunSftErrors:
    def test_missing_train_jsonl_raises_filenotfound(self):
        """When train_jsonl is empty and config.train_jsonl is also empty."""
        config = SFTConfig(train_jsonl="")
        with pytest.raises(FileNotFoundError, match="train_jsonl path is empty"):
            run_sft(config, dry_run=True)

    def test_missing_train_file_raises_filenotfound(self, tmp_path):
        config = SFTConfig(train_jsonl=str(tmp_path / "nonexistent.jsonl"))
        with pytest.raises(FileNotFoundError):
            run_sft(config, dry_run=True)

    def test_real_training_raises_when_ml_unavailable(self, tmp_path):
        """When not dry_run and torch/transformers are missing, raises TrainingUnavailableError.

        We simulate this by monkeypatching the lazy import check.
        """
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = SFTConfig(train_jsonl=str(train_path))

        # Monkeypatch _check_can_train to always raise (simulating no GPU/torch)
        import app.training.trainer_sft as trainer_module

        original = trainer_module._check_can_train

        def _fake_check(cfg):
            raise TrainingUnavailableError("No CUDA GPU detected (simulated).")

        trainer_module._check_can_train = _fake_check
        try:
            with pytest.raises(TrainingUnavailableError):
                run_sft(config, dry_run=False)
        finally:
            trainer_module._check_can_train = original

    def test_custom_run_id_is_respected(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = SFTConfig(train_jsonl=str(train_path))
        result = run_sft(config, dry_run=True, run_id="my_custom_run")
        assert result.run_id == "my_custom_run"
