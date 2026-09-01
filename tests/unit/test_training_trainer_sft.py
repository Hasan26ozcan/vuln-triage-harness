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

import json
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.dataset import InstructionExample
from app.schemas.training import TrainingResult
from app.training.config import SFTConfig
from app.training.trainer_sft import (
    StepEstimate,
    TrainingUnavailableError,
    _attach_callbacks,
    _check_can_train,
    _convert_for_causal_lm,
    _load_jsonl_for_training,
    _run_sft,
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
    id_: str = "ie_001",
) -> InstructionExample:
    return InstructionExample(
        id=id_,
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

        config = SFTConfig(train_jsonl=str(train_path), use_4bit=False, lora_r=0)
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


# ---------------------------------------------------------------------------
# _check_can_train (lines 106-135)
# ---------------------------------------------------------------------------


class TestCheckCanTrain:
    """Covers _check_can_train: ImportError + no-CUDA paths."""

    def test_import_error_raises_training_unavailable(self):
        """When torch/transformers are not importable, raise TrainingUnavailableError."""
        with patch.dict("sys.modules", {"torch": None, "transformers": None}):
            config = SFTConfig()
            with pytest.raises(TrainingUnavailableError, match="torch/transformers not installed"):
                _check_can_train(config)

    def test_no_cuda_4bit_raises(self):
        """No CUDA + use_4bit=True → TrainingUnavailableError (QLoRA needs CUDA)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": MagicMock()}):
            config = SFTConfig(use_4bit=True)
            with pytest.raises(TrainingUnavailableError, match="QLoRA.*requires CUDA"):
                _check_can_train(config)

    def test_no_cuda_cpu_fallback_warns(self):
        """No CUDA + use_4bit=False → CPU fallback with warning (no raise)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch, "transformers": MagicMock()}):
            _check_can_train(SFTConfig(use_4bit=False))
            # No exception raised — CPU fallback is allowed


# ---------------------------------------------------------------------------
# _load_jsonl_for_training
# ---------------------------------------------------------------------------


class TestLoadJsonlForTraining:
    """Covers _load_jsonl_for_training (lines 138-149)."""

    def test_with_default_loader(self, tmp_path):
        """When loader=None, JsonlDataLoader is used."""
        train_path = tmp_path / "train.jsonl"
        examples = [_make_example(id_=f"ie_{i}") for i in range(3)]
        _write_jsonl(str(train_path), examples)
        train, val = _load_jsonl_for_training(str(train_path), "")
        assert len(train) == 3
        assert val == []

    def test_with_val_path(self, tmp_path):
        """When val_path is provided, it is loaded too."""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(str(train_path), [_make_example(id_="t1")])
        _write_jsonl(str(val_path), [_make_example(id_="v1")])
        train, val = _load_jsonl_for_training(str(train_path), str(val_path))
        assert len(train) == 1
        assert len(val) == 1
        assert train[0].id == "t1"
        assert val[0].id == "v1"

    def test_with_injected_loader(self):
        """When loader is injected, no real file is read."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = [_make_example()]
        train, val = _load_jsonl_for_training(
            "fake_train.jsonl", "fake_val.jsonl", loader=mock_loader
        )
        assert len(train) == 1
        assert len(val) == 1
        assert mock_loader.load.call_count == 2


# ---------------------------------------------------------------------------
# _convert_for_causal_lm
# ---------------------------------------------------------------------------


class TestConvertForCausalLm:
    """Covers _convert_for_causal_lm (lines 152-170)."""

    def test_basic_conversion(self):
        ex = _make_example(
            prompt="Classify this vulnerability.",
            cwe="CWE-89",
            explanation="SQL injection.",
        )
        rows = _convert_for_causal_lm([ex])
        assert len(rows) == 1
        assert rows[0]["prompt"] == "Classify this vulnerability."
        # Completion is a JSON string with cwe_id, severity, explanation, patch_diff
        completion = json.loads(rows[0]["completion"])
        assert completion["cwe_id"] == "CWE-89"
        assert completion["severity"] == "high"
        assert completion["explanation"] == "SQL injection."
        assert completion["patch_diff"] is None
        assert rows[0]["cwe_id"] == "CWE-89"

    def test_empty_explanation_uses_blank_string(self):
        ex = _make_example(explanation="")
        rows = _convert_for_causal_lm([ex])
        # Completion is JSON; empty explanation → "" in the JSON field
        completion = json.loads(rows[0]["completion"])
        assert completion["explanation"] == ""

    def test_multiple_examples(self):
        examples = [_make_example(id_=f"ie_{i}", cwe=f"CWE-0{i + 1}") for i in range(3)]
        rows = _convert_for_causal_lm(examples)
        assert len(rows) == 3
        assert rows[0]["cwe_id"] == "CWE-01"
        assert rows[2]["cwe_id"] == "CWE-03"


# ---------------------------------------------------------------------------
# _attach_callbacks
# ---------------------------------------------------------------------------


class TestAttachCallbacks:
    """Covers _attach_callbacks (line 461 — the pass no-op)."""

    def test_attach_callbacks_is_no_op(self):
        """_attach_callbacks is a no-op (pass); it should return None."""
        tracker = MagicMock()
        config = SFTConfig()
        result = _attach_callbacks(
            MagicMock(),  # trainer
            [],  # callbacks
            tracker,  # tracker
            config,  # config
            "run_1",  # run_id
        )
        assert result is None


# ---------------------------------------------------------------------------
# _run_sft — full training path (lines 173-453)
# ---------------------------------------------------------------------------


class TestRunSftTraining:
    """Covers _run_sft with all ML imports mocked."""

    def _mock_ml_modules(self, cuda_available=True):
        """Build mock modules for torch, peft, transformers with version."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = cuda_available
        mock_torch.float16 = "float16"
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.set_num_threads = MagicMock()

        mock_peft = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.__version__ = "5.14.1"

        # transformers.trainer_callback needs its own sys.modules entry
        # because _run_sft does `from transformers.trainer_callback import TrainerCallback`
        # and then subclasses it (class _LossCallback(_TrainerCallback)).
        # We must provide a real class, not a MagicMock, for subclassing to work.
        class _MockTrainerCallback:
            """Stand-in for transformers.trainer_callback.TrainerCallback."""

        mock_trainer_callback_module = MagicMock()
        mock_trainer_callback_module.TrainerCallback = _MockTrainerCallback

        return {
            "torch": mock_torch,
            "peft": mock_peft,
            "transformers": mock_transformers,
            "transformers.trainer_callback": mock_trainer_callback_module,
        }

    def _setup_trainer_mocks(self, mock_modules):
        """Configure model, tokenizer, trainer mocks."""
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "eos_token"
        mock_tokenizer.save_pretrained = MagicMock()

        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.123}

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_train_result
        mock_trainer.evaluate.return_value = {"eval_loss": 0.456}

        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["transformers"].DataCollatorForLanguageModeling.return_value = MagicMock()
        mock_modules["transformers"].Trainer.return_value = mock_trainer
        # get_peft_model should return the same mock_model so save_pretrained
        # calls can be asserted on it.
        mock_modules["peft"].get_peft_model.return_value = mock_model

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        return mock_model, mock_tokenizer, mock_trainer, mock_tracker

    def _make_callbacks(self, include_checkpoint=False):
        """Build callback list, optionally including CheckpointCallback."""
        from app.training.callbacks import CheckpointCallback, WandbCallback

        callbacks = [WandbCallback(run_name="test", mock=True)]
        if include_checkpoint:
            callbacks.append(CheckpointCallback(mock=True))
        return callbacks

    def test_run_sft_qlora_path(self):
        """Covers _run_sft QLoRA path: use_4bit=True, CUDA available."""

        config = SFTConfig(use_4bit=True, lora_r=8, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=True)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )
        callbacks = self._make_callbacks(include_checkpoint=True)

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            train_examples = [_make_example()]
            result = _run_sft(config, train_examples, [], callbacks, "sft_qlora_1")

        # QLoRA path used prepare_model_for_kbit_training
        mock_modules["peft"].prepare_model_for_kbit_training.assert_called_once()
        assert result.status == "completed"
        assert result.train_set_size == 1
        assert result.checkpoint_uri != ""
        assert len(callbacks[1].checkpoints) == 1  # CheckpointCallback saved

    def test_run_sft_cpu_path(self):
        """Covers _run_sft CPU path: use_4bit=False, no CUDA."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=False)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )
        callbacks = self._make_callbacks(include_checkpoint=False)

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            train_examples = [_make_example()]
            result = _run_sft(config, train_examples, [], callbacks, "sft_cpu")

        assert result.status == "completed"
        assert result.train_set_size == 1
        # No CheckpointCallback → local save path
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()

    def test_run_sft_cuda_non_4bit_path(self):
        """Covers _run_sft CUDA path without 4-bit (lines 264-265, 315-317):
        use_4bit=False + cuda_available=True → fp16 + float16."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=True)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )
        callbacks = self._make_callbacks(include_checkpoint=False)

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            train_examples = [_make_example()]
            result = _run_sft(config, train_examples, [], callbacks, "sft_cuda_full")

        # CUDA + non-4bit path uses torch.float16
        assert mock_modules["torch"].float16  # float16 was accessed
        assert result.status == "completed"
        mock_model.save_pretrained.assert_called_once()

    def test_run_sft_old_transformers_version(self):
        """Covers the else branch at line 365 (transformers < 5.0.0 → tokenizer kwarg)."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=False)
        mock_modules["transformers"].__version__ = "4.46.0"
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )
        callbacks = self._make_callbacks(include_checkpoint=False)

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            train_examples = [_make_example()]
            result = _run_sft(config, train_examples, [], callbacks, "sft_old_tf")

        # Trainer was called with tokenizer kwarg (not processing_class)
        mock_trainer_init_kwargs = mock_modules["transformers"].Trainer.call_args
        assert "tokenizer" in mock_trainer_init_kwargs.kwargs
        assert "processing_class" not in mock_trainer_init_kwargs.kwargs
        assert result.status == "completed"

    def test_run_sft_with_val_examples(self):
        """Covers evaluation path (lines 394-397) when val_examples provided."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=False)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )
        callbacks = self._make_callbacks(include_checkpoint=False)

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            train_examples = [_make_example()]
            val_examples = [_make_example(id_="val_1")]
            result = _run_sft(config, train_examples, val_examples, callbacks, "sft_val")

        mock_trainer.evaluate.assert_called_once()
        assert result.final_val_loss == 0.456

    def test_run_sft_callback_on_init_raises_is_caught(self):
        """When a callback's on_init raises, the warning is logged and _run_sft continues."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=False)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )

        bad_cb = MagicMock()
        bad_cb.on_init.side_effect = RuntimeError("init failed")
        good_cb = MagicMock()

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            result = _run_sft(config, [_make_example()], [], [bad_cb, good_cb], "sft_bad_init")

        bad_cb.on_init.assert_called_once()
        good_cb.on_init.assert_called_once()
        assert result.status == "completed"

    def test_run_sft_callback_on_train_end_raises_is_caught(self):
        """When a callback's on_train_end raises, the warning is logged and
        the result is still returned."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=False)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )

        bad_cb = MagicMock()
        bad_cb.on_train_end.side_effect = RuntimeError("train_end failed")
        good_cb = MagicMock()

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            result = _run_sft(config, [_make_example()], [], [bad_cb, good_cb], "sft_bad_end")

        bad_cb.on_train_end.assert_called_once()
        good_cb.on_train_end.assert_called_once()
        assert result.status == "completed"

    def test_loss_callback_on_log_appends_loss(self):
        """Covers the internal _LossCallback.on_log method (lines 381-383)."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=False)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )
        callbacks = self._make_callbacks(include_checkpoint=False)

        captured: list = []

        def _capture(cb_class):
            captured.append(cb_class)

        mock_trainer.add_callback.side_effect = _capture

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            _run_sft(config, [_make_example()], [], callbacks, "sft_loss")

        # _attach_callbacks is a no-op (pass), so add_callback is only called once
        # — with the _LossCallback class
        assert len(captured) == 1
        loss_cb_class = captured[0]
        instance = loss_cb_class()
        instance.on_log(args=None, state=None, control=None, logs={"loss": 0.42})

    def test_loss_callback_on_log_without_loss_key(self):
        """When logs has no 'loss' key, on_log does nothing (line 382 condition False)."""
        config = SFTConfig(use_4bit=False, lora_r=16, num_train_epochs=1)
        mock_modules = self._mock_ml_modules(cuda_available=False)
        mock_model, mock_tokenizer, mock_trainer, mock_tracker = self._setup_trainer_mocks(
            mock_modules
        )
        callbacks = self._make_callbacks(include_checkpoint=False)

        captured: list = []

        def _capture(cb_class):
            captured.append(cb_class)

        mock_trainer.add_callback.side_effect = _capture

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_sft.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            _run_sft(config, [_make_example()], [], callbacks, "sft_loss2")

        assert len(captured) == 1
        instance = captured[0]()
        instance.on_log(args=None, state=None, control=None, logs={})


# ---------------------------------------------------------------------------
# run_sft — real training path (lines 562-586)
# ---------------------------------------------------------------------------


class TestRunSftRealTraining:
    """Covers run_sft's real training path: callback on_init + try/except."""

    def test_real_training_on_init_and_success(self, tmp_path):
        """Real (non-dry-run) path: on_init callback + _run_sft call."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = SFTConfig(train_jsonl=str(train_path))

        spy = MagicMock()
        mock_result = TrainingResult(
            run_id="sft_real_1",
            method="sft_qlora",
            base_model="test/model",
            hyperparams={"lora_r": 8},
            train_set_size=1,
            train_time_minutes=1.0,
            peak_vram_gb=7.0,
            final_train_loss=0.5,
            final_val_loss=None,
        )

        with (
            patch("app.training.trainer_sft._check_can_train"),
            patch("app.training.trainer_sft._run_sft", return_value=mock_result),
            patch(
                "app.training.trainer_sft._load_jsonl_for_training",
                return_value=([_make_example()], []),
            ),
        ):
            result = run_sft(config, dry_run=False, callbacks=[spy])

        spy.on_init.assert_called_once()
        assert result is mock_result

    def test_real_training_training_unavailable_is_reraised(self, tmp_path):
        """TrainingUnavailableError from _run_sft is re-raised directly (line 579-580)."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = SFTConfig(train_jsonl=str(train_path))

        spy = MagicMock()

        with (
            patch("app.training.trainer_sft._check_can_train"),
            patch(
                "app.training.trainer_sft._run_sft",
                side_effect=TrainingUnavailableError("CUDA missing"),
            ),
            patch(
                "app.training.trainer_sft._load_jsonl_for_training",
                return_value=([_make_example()], []),
            ),
        ):
            with pytest.raises(TrainingUnavailableError):
                run_sft(config, dry_run=False, callbacks=[spy])

    def test_real_training_other_exception_calls_on_error(self, tmp_path):
        """When _run_sft raises a non-TrainingUnavailableError, on_error is called."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = SFTConfig(train_jsonl=str(train_path))

        spy = MagicMock()

        with (
            patch("app.training.trainer_sft._check_can_train"),
            patch(
                "app.training.trainer_sft._run_sft",
                side_effect=RuntimeError("training crashed"),
            ),
            patch(
                "app.training.trainer_sft._load_jsonl_for_training",
                return_value=([_make_example()], []),
            ),
        ):
            with pytest.raises(RuntimeError, match="training crashed"):
                run_sft(config, dry_run=False, callbacks=[spy])

        spy.on_error.assert_called_once()
        assert "training crashed" in spy.on_error.call_args[0][0]
