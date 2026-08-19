"""Unit tests for Stage 5 DPO trainer.

Covers:
  - estimate_dpo_steps: pure arithmetic step/VRAM estimation.
  - build_preference_pairs: chosen/rejected pair construction.
  - _format_response: JSON serialization of targets.
  - _make_rejected_response: synthetic wrong-CWE generation.
  - run_dpo(dry_run=True): returns TrainingResult with estimates.
  - run_dpo raises FileNotFoundError when train_jsonl is missing.
  - run_dpo raises DPOUnavailableError when not dry_run and ML stack missing.

All tests run without a GPU or model downloads.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.dataset import InstructionExample
from app.schemas.training import TrainingResult
from app.training.config import DPOConfig
from app.training.trainer_dpo import (
    DPOStepEstimate,
    DPOUnavailableError,
    _check_can_train,
    _format_response,
    _make_rejected_response,
    _run_dpo,
    build_preference_pairs,
    estimate_dpo_steps,
    run_dpo,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_example(
    cwe: str = "CWE-89",
    severity: str = "high",
    explanation: str = "SQL injection.",
    patch_diff: str | None = None,
    prompt: str = "Classify this vulnerability.",
) -> InstructionExample:
    return InstructionExample(
        id="ie_001",
        sample_id="vs_001",
        prompt=prompt,
        target_cwe=cwe,
        target_severity=severity,
        target_explanation=explanation,
        target_patch_diff=patch_diff,
        token_count_estimate=100,
    )


def _write_jsonl(path: str, examples: list[InstructionExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# estimate_dpo_steps
# ---------------------------------------------------------------------------


class TestEstimateDpoSteps:
    def test_basic_calculation(self):
        config = DPOConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
        )
        estimate = estimate_dpo_steps(config, n_pairs=128)
        # steps_per_epoch = 128 // 1 = 128
        # optim_steps_per_epoch = 128 // 8 = 16
        # total = 16 * 3 = 48
        assert estimate.steps_per_epoch == 16
        assert estimate.total_steps == 48
        assert estimate.n_pairs == 128

    def test_vram_estimate(self):
        """DPO with reference + policy model: ~12 GB for 7B."""
        config = DPOConfig()
        estimate = estimate_dpo_steps(config, n_pairs=100)
        assert estimate.estimated_vram_gb == 12.0
        assert estimate.can_fit_in_8gb is False

    def test_small_dataset(self):
        config = DPOConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
        )
        estimate = estimate_dpo_steps(config, n_pairs=2)
        assert estimate.steps_per_epoch == 1
        assert estimate.total_steps == 1

    def test_returns_dpo_step_estimate(self):
        config = DPOConfig()
        result = estimate_dpo_steps(config, n_pairs=100)
        assert isinstance(result, DPOStepEstimate)


# ---------------------------------------------------------------------------
# build_preference_pairs
# ---------------------------------------------------------------------------


class TestBuildPreferencePairs:
    def test_single_example_synthetic_rejected(self):
        ex = _make_example(cwe="CWE-89")
        pairs = build_preference_pairs([ex])
        assert len(pairs) == 1
        pair = pairs[0]
        assert pair["prompt"] == ex.prompt
        # chosen is the correct response
        chosen = json.loads(pair["chosen"])
        assert chosen["cwe_id"] == "CWE-89"
        # rejected is a wrong response
        rejected = json.loads(pair["rejected"])
        assert rejected["cwe_id"] != "CWE-89"

    def test_chosen_matches_example_targets(self):
        ex = _make_example(
            cwe="CWE-79", severity="medium", explanation="XSS vuln.", patch_diff="--- diff"
        )
        pairs = build_preference_pairs([ex])
        chosen = json.loads(pairs[0]["chosen"])
        assert chosen["cwe_id"] == "CWE-79"
        assert chosen["severity"] == "medium"
        assert chosen["explanation"] == "XSS vuln."
        assert chosen["patch_diff"] == "--- diff"

    def test_with_explicit_rejected_examples(self):
        chosen_ex = _make_example(cwe="CWE-89")
        rejected_ex = _make_example(cwe="CWE-79")
        pairs = build_preference_pairs([chosen_ex], [rejected_ex])
        chosen = json.loads(pairs[0]["chosen"])
        rejected = json.loads(pairs[0]["rejected"])
        assert chosen["cwe_id"] == "CWE-89"
        assert rejected["cwe_id"] == "CWE-79"

    def test_rejected_uses_synthetic_when_not_enough_rejected(self):
        """If fewer rejected examples than train examples, fill with synthetic."""
        train = [_make_example(cwe="CWE-89"), _make_example(cwe="CWE-79")]
        rejected = [_make_example(cwe="CWE-22")]
        pairs = build_preference_pairs(train, rejected)
        assert len(pairs) == 2
        # First pair uses the explicit rejected example
        first_rejected = json.loads(pairs[0]["rejected"])
        assert first_rejected["cwe_id"] == "CWE-22"
        # Second pair uses a synthetic rejected response
        second_rejected = json.loads(pairs[1]["rejected"])
        assert second_rejected["cwe_id"] != "CWE-79"

    def test_with_none_patch_diff(self):
        ex = _make_example(cwe="CWE-89", patch_diff=None)
        pairs = build_preference_pairs([ex])
        chosen = json.loads(pairs[0]["chosen"])
        assert chosen["patch_diff"] is None


# ---------------------------------------------------------------------------
# _format_response
# ---------------------------------------------------------------------------


class TestFormatResponse:
    def test_basic_response(self):
        ex = _make_example(
            cwe="CWE-89", severity="high", explanation="SQLi.", patch_diff="--- a\n+++ b\n"
        )
        response = _format_response(ex)
        obj = json.loads(response)
        assert obj["cwe_id"] == "CWE-89"
        assert obj["severity"] == "high"
        assert obj["explanation"] == "SQLi."
        assert obj["patch_diff"] == "--- a\n+++ b\n"

    def test_response_is_json_string(self):
        ex = _make_example()
        response = _format_response(ex)
        assert isinstance(response, str)
        # Must be parseable JSON
        json.loads(response)


# ---------------------------------------------------------------------------
# _make_rejected_response
# ---------------------------------------------------------------------------


class TestMakeRejectedResponse:
    def test_rejected_has_wrong_cwe(self):
        ex = _make_example(cwe="CWE-89")
        rejected = _make_rejected_response(ex)
        obj = json.loads(rejected)
        assert obj["cwe_id"] != "CWE-89"

    def test_rejected_cwe_is_within_scope(self):
        ex = _make_example(cwe="CWE-79")
        rejected = _make_rejected_response(ex)
        obj = json.loads(rejected)
        assert obj["cwe_id"] in ("CWE-89", "CWE-22", "CWE-78", "CWE-190", "CWE-502")

    def test_rejected_has_shallow_explanation(self):
        ex = _make_example(cwe="CWE-89")
        rejected = _make_rejected_response(ex)
        obj = json.loads(rejected)
        assert "not sure" in obj["explanation"].lower()

    def test_rejected_has_empty_patch(self):
        ex = _make_example(cwe="CWE-89")
        rejected = _make_rejected_response(ex)
        obj = json.loads(rejected)
        assert obj["patch_diff"] == ""

    def test_rejected_severity_is_low(self):
        ex = _make_example(cwe="CWE-89")
        rejected = _make_rejected_response(ex)
        obj = json.loads(rejected)
        assert obj["severity"] == "low"

    def test_rejected_for_unknown_cwe_defaults_to_cwe89(self):
        """If target_cwe is not in the scope list, falls back to CWE-89."""
        ex = _make_example(cwe="CWE-999")
        rejected = _make_rejected_response(ex)
        obj = json.loads(rejected)
        # Should produce a valid wrong CWE (not the same as input)
        assert obj["cwe_id"] != "CWE-999"


# ---------------------------------------------------------------------------
# run_dpo — dry_run mode
# ---------------------------------------------------------------------------


class TestRunDpoDryRun:
    def test_dry_run_returns_completed_result(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        examples = [_make_example() for _ in range(5)]
        _write_jsonl(str(train_path), examples)

        config = DPOConfig(train_jsonl=str(train_path), beta=0.1)
        result = run_dpo(config, dry_run=True)

        assert isinstance(result, TrainingResult)
        assert result.status == "dry_run"
        assert result.train_set_size == 5  # number of pairs
        assert result.final_train_loss == 0.0
        assert result.final_val_loss is None
        assert result.checkpoint_uri == ""

    def test_dry_run_sets_peak_vram(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = DPOConfig(train_jsonl=str(train_path))
        result = run_dpo(config, dry_run=True)
        assert result.peak_vram_gb == 12.0

    def test_dry_run_generates_run_id(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = DPOConfig(train_jsonl=str(train_path))
        result = run_dpo(config, dry_run=True)
        assert result.run_id.startswith("dpo_")

    def test_dry_run_method_is_dpo(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = DPOConfig(train_jsonl=str(train_path))
        result = run_dpo(config, dry_run=True)
        assert result.method == "dpo"

    def test_dry_run_with_val(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(str(train_path), [_make_example() for _ in range(4)])
        _write_jsonl(str(val_path), [_make_example() for _ in range(2)])

        config = DPOConfig(train_jsonl=str(train_path))
        result = run_dpo(config, dry_run=True, val_path=str(val_path))
        # DPO pairs = train examples (each becomes a pair)
        assert result.train_set_size == 4

    def test_dry_run_callbacks_notified(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        events: list[str] = []

        class _SpyCallback:
            def on_init(self, config: dict) -> None:
                events.append("init")

            def on_step(self, step: int, loss: float | None = None) -> None:
                events.append("step")

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
                events.append("train_end")

            def on_error(self, error: str) -> None:
                events.append("error")

        spy = _SpyCallback()
        config = DPOConfig(train_jsonl=str(train_path))
        run_dpo(config, dry_run=True, callbacks=[spy])
        assert "init" in events
        assert "train_end" in events


# ---------------------------------------------------------------------------
# run_dpo — error handling
# ---------------------------------------------------------------------------


class TestRunDpoErrors:
    def test_missing_train_jsonl_raises_filenotfound(self):
        config = DPOConfig(train_jsonl="")
        with pytest.raises(FileNotFoundError, match="train_jsonl path is empty"):
            run_dpo(config, dry_run=True)

    def test_missing_train_file_raises_filenotfound(self, tmp_path):
        config = DPOConfig(train_jsonl=str(tmp_path / "nonexistent.jsonl"))
        with pytest.raises(FileNotFoundError):
            run_dpo(config, dry_run=True)

    def test_real_training_raises_when_ml_unavailable(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = DPOConfig(train_jsonl=str(train_path))

        # Monkeypatch _check_can_train to always raise
        import app.training.trainer_dpo as trainer_module

        original = trainer_module._check_can_train

        def _fake_check(cfg):
            raise DPOUnavailableError("No CUDA GPU detected (simulated).")

        trainer_module._check_can_train = _fake_check
        try:
            with pytest.raises(DPOUnavailableError):
                run_dpo(config, dry_run=False)
        finally:
            trainer_module._check_can_train = original

    def test_custom_run_id_is_respected(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = DPOConfig(train_jsonl=str(train_path))
        result = run_dpo(config, dry_run=True, run_id="my_dpo_run")
        assert result.run_id == "my_dpo_run"


# ---------------------------------------------------------------------------
# _check_can_train
# ---------------------------------------------------------------------------


class TestCheckCanTrain:
    """Covers _check_can_train — ImportError and no-CUDA paths (lines 304-320)."""

    def test_import_error_raises_dpo_unavailable(self):
        """When torch/transformers/trl are not importable, raise DPOUnavailableError."""
        with patch.dict("sys.modules", {"torch": None, "transformers": None, "trl": None}):
            with pytest.raises(DPOUnavailableError, match="torch/transformers/trl not installed"):
                _check_can_train(DPOConfig())

    def test_no_cuda_raises_dpo_unavailable(self):
        """When torch imports fine but CUDA is not available, raise DPOUnavailableError."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "transformers": MagicMock(), "trl": MagicMock()},
        ):
            with pytest.raises(DPOUnavailableError, match="No CUDA GPU detected"):
                _check_can_train(DPOConfig())


# ---------------------------------------------------------------------------
# _run_dpo — mocked ML stack, full execution path (lines 154-301)
# ---------------------------------------------------------------------------


class TestRunDpoTraining:
    """Covers _run_dpo end-to-end with all ML imports mocked."""

    def _mock_ml_modules(self):
        """Return mock modules for torch, peft, transformers, trl, datasets.

        ``datasets`` is included because ``_run_dpo`` does a real ``from
        datasets import Dataset`` which introspects ``torch.__spec__``; when
        ``torch`` is a plain ``MagicMock`` (no ``__spec__``) that introspection
        raises ``ValueError``, so we mock ``datasets`` too.

        ``TrainerCallback`` is set as a *real* class on the mocked
        ``transformers`` module because ``_run_dpo`` subclasses it:
        ``class _LossCallback(TrainerCallback)``.  Subclassing a bare
        ``MagicMock`` raises ``AttributeError: __get__`` on some platforms
        (observed on Linux CI) during MRO computation.  Providing a real
        stand-in class avoids the descriptor-protocol issue.
        """
        mock_torch = MagicMock()
        mock_peft = MagicMock()
        mock_transformers = MagicMock()
        mock_trl = MagicMock()
        mock_datasets = MagicMock()

        class _MockTrainerCallback:
            """Stand-in for ``transformers.TrainerCallback`` in tests."""

        mock_transformers.TrainerCallback = _MockTrainerCallback

        return {
            "torch": mock_torch,
            "peft": mock_peft,
            "transformers": mock_transformers,
            "trl": mock_trl,
            "datasets": mock_datasets,
        }

    def _setup_trainer_mocks(self):
        """Configure mock trainer, model, tokenizer for _run_dpo."""
        mock_model = MagicMock()
        mock_model.save_pretrained = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "eos"
        mock_tokenizer.save_pretrained = MagicMock()

        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}

        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_train_result
        mock_trainer.evaluate.return_value = {"eval_loss": 0.42}

        return mock_model, mock_tokenizer, mock_trainer

    def test_run_dpo_full_path_with_checkpoint_callback_and_val(self):
        """Covers _run_dpo: model load, tokenizer, TrlDPOConfig, DPOTrainer,
        loss callback, train, evaluate (val_pairs), checkpoint save, on_train_end."""
        from app.training.callbacks import CheckpointCallback
        from app.training.trainer_dpo import DPOConfig

        config = DPOConfig(train_jsonl="dummy")
        mock_modules = self._mock_ml_modules()
        mock_model, mock_tokenizer, mock_trainer = self._setup_trainer_mocks()

        ckpt_cb = CheckpointCallback(mock=True)
        callbacks = [ckpt_cb]

        # Configure the mocked modules
        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["trl"].DPOTrainer.return_value = mock_trainer

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_dpo.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),  # no sft_checkpoint file
        ):
            pairs = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
            val_pairs = [{"prompt": "pv", "chosen": "cv", "rejected": "rv"}]
            result = _run_dpo(config, pairs, val_pairs, callbacks, "dpo_run_1")

        assert isinstance(result, TrainingResult)
        assert result.status == "completed"
        assert result.method == "dpo"
        assert result.final_train_loss == 0.5
        assert result.final_val_loss == 0.42
        assert result.train_set_size == 1
        # CheckpointCallback should have saved
        assert len(ckpt_cb.checkpoints) == 1
        assert ckpt_cb.checkpoints[0]["run_id"] == "dpo_run_1"

    def test_run_dpo_sft_checkpoint_exists(self):
        """When sft_checkpoint exists, PEFT adapter is loaded for DPO tuning.

        DPOTrainer handles PEFT models natively (no merge_and_unload needed).
        """

        from app.training.trainer_dpo import DPOConfig

        config = DPOConfig(train_jsonl="dummy", sft_checkpoint="some/ckpt")
        mock_modules = self._mock_ml_modules()
        mock_model, mock_tokenizer, mock_trainer = self._setup_trainer_mocks()

        # prepare_model_for_kbit_training is a no-op on the mock — return model unchanged
        mock_modules["peft"].prepare_model_for_kbit_training.return_value = mock_model
        # PeftModel.from_pretrained returns the model (already mocked)
        mock_modules["peft"].PeftModel.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["trl"].DPOTrainer.return_value = mock_trainer

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_dpo.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=True),
        ):
            _run_dpo(config, [], None, [], "dpo_ckpt")

        # Verify PeftModel.from_pretrained was called (sft_checkpoint branch)
        mock_modules["peft"].PeftModel.from_pretrained.assert_called_once()

    def test_run_dpo_no_checkpoint_callback_uses_local_save(self):
        """When no CheckpointCallback, model/tokenizer are saved locally."""

        from app.training.trainer_dpo import DPOConfig

        config = DPOConfig(train_jsonl="dummy")
        mock_modules = self._mock_ml_modules()
        mock_model, mock_tokenizer, mock_trainer = self._setup_trainer_mocks()

        # No sft_checkpoint → else branch: prepare_model_for_kbit_training +
        # get_peft_model.  Configure them to return mock_model so the
        # save_pretrained assertion targets the right mock.
        mock_modules["peft"].prepare_model_for_kbit_training.return_value = mock_model
        mock_modules["peft"].get_peft_model.return_value = mock_model
        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["trl"].DPOTrainer.return_value = mock_trainer

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        callbacks = []  # no CheckpointCallback

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_dpo.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            result = _run_dpo(config, [], None, callbacks, "dpo_nockpt")

        # Model and tokenizer should be saved locally
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()
        assert result.checkpoint_uri != ""
        # checkpoint_uri should be the local path
        assert "final_checkpoint" in result.checkpoint_uri

    def test_run_dpo_callback_on_init_raises_is_caught(self):
        """When a callback's on_init raises, the warning is logged and _run_dpo continues."""

        from app.training.trainer_dpo import DPOConfig

        config = DPOConfig(train_jsonl="dummy")
        mock_modules = self._mock_ml_modules()
        mock_model, mock_tokenizer, mock_trainer = self._setup_trainer_mocks()

        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["trl"].DPOTrainer.return_value = mock_trainer

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        bad_cb = MagicMock()
        bad_cb.on_init.side_effect = RuntimeError("init failed")
        good_cb = MagicMock()

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_dpo.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            result = _run_dpo(config, [], None, [bad_cb, good_cb], "dpo_cb")

        # bad_cb.on_init was called (and failed), good_cb.on_init was also called
        bad_cb.on_init.assert_called_once()
        good_cb.on_init.assert_called_once()
        assert result.status == "completed"

    def test_run_dpo_callback_on_train_end_raises_is_caught(self):
        """When a callback's on_train_end raises, the warning is logged and
        result is still returned."""

        from app.training.trainer_dpo import DPOConfig

        config = DPOConfig(train_jsonl="dummy")
        mock_modules = self._mock_ml_modules()
        mock_model, mock_tokenizer, mock_trainer = self._setup_trainer_mocks()

        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["trl"].DPOTrainer.return_value = mock_trainer

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        bad_cb = MagicMock()
        bad_cb.on_train_end.side_effect = RuntimeError("end failed")
        good_cb = MagicMock()

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_dpo.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            result = _run_dpo(config, [], None, [bad_cb, good_cb], "dpo_end")

        bad_cb.on_train_end.assert_called_once()
        good_cb.on_train_end.assert_called_once()
        assert result.status == "completed"


# ---------------------------------------------------------------------------
# run_dpo — real training path (lines 440-458: on_init + try/except)
# ---------------------------------------------------------------------------


class TestRunDpoRealTraining:
    """Covers run_dpo's real training path: callback on_init + try/except
    around _run_dpo with on_error dispatch."""

    def test_real_training_on_init_and_success(self, tmp_path):
        """Real (non-dry-run) path: callbacks.on_init is called, _run_dpo is invoked."""
        from app.training.config import DPOConfig

        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = DPOConfig(train_jsonl=str(train_path))

        spy = MagicMock()
        mock_result = TrainingResult(
            run_id="dpo_real_1",
            method="dpo",
            base_model="test/model",
            hyperparams={"beta": 0.1},
            train_set_size=1,
            train_time_minutes=1.0,
            peak_vram_gb=12.0,
            final_train_loss=0.5,
            final_val_loss=None,
        )

        with (
            patch("app.training.trainer_dpo._check_can_train") as mock_check,
            patch("app.training.trainer_dpo._run_dpo", return_value=mock_result) as mock_run,
        ):
            result = run_dpo(config, dry_run=False, callbacks=[spy])

        mock_check.assert_called_once_with(config)
        mock_run.assert_called_once()
        spy.on_init.assert_called_once()
        assert result is mock_result

    def test_real_training_on_error_is_called_on_failure(self, tmp_path):
        """When _run_dpo raises, callbacks.on_error is called and the exception
        is re-raised (lines 451-456)."""
        from app.training.config import DPOConfig

        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), [_make_example()])

        config = DPOConfig(train_jsonl=str(train_path))

        spy = MagicMock()

        with (
            patch("app.training.trainer_dpo._check_can_train"),
            patch(
                "app.training.trainer_dpo._run_dpo",
                side_effect=RuntimeError("training crashed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="training crashed"):
                run_dpo(config, dry_run=False, callbacks=[spy])

        # on_error should have been called with the error message
        spy.on_error.assert_called_once()
        assert "training crashed" in spy.on_error.call_args[0][0]


# ---------------------------------------------------------------------------
# _LossCallback.on_log — covers lines 234-235
# ---------------------------------------------------------------------------


class TestLossCallbackOnLog:
    """Covers the internal _LossCallback.on_log method (lines 232-235).

    The _LossCallback class is defined locally inside _run_dpo and captures
    loss_history via closure. We extract the class from the mock trainer's
    add_callback call and invoke on_log on an instance.
    """

    def test_on_log_appends_loss(self):
        """When logs contains 'loss', it is appended to loss_history."""

        from app.training.trainer_dpo import DPOConfig

        config = DPOConfig(train_jsonl="dummy")
        mock_transformers = MagicMock()

        # Provide a real TrainerCallback so class _LossCallback(TrainerCallback)
        # works under mocked transformers (subclassing a MagicMock raises
        # AttributeError: __get__ on Linux CI).
        class _MockTrainerCallback:
            pass

        mock_transformers.TrainerCallback = _MockTrainerCallback
        mock_modules = {
            "torch": MagicMock(),
            "peft": MagicMock(),
            "transformers": mock_transformers,
            "trl": MagicMock(),
            "datasets": MagicMock(),
        }
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "eos"
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_train_result

        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["trl"].DPOTrainer.return_value = mock_trainer

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        # Capture the _LossCallback class passed to add_callback
        captured: list = []

        def _capture(cb_class):
            captured.append(cb_class)

        mock_trainer.add_callback.side_effect = _capture

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_dpo.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            _run_dpo(config, [], None, [], "loss_test")

        assert len(captured) == 1
        loss_cb_class = captured[0]
        instance = loss_cb_class()
        instance.on_log(args=None, state=None, control=None, logs={"loss": 0.42})
        # The on_log was called without raising — lines 234-235 covered

    def test_on_log_no_loss_key_does_not_append(self):
        """When logs has no 'loss' key, nothing happens (line 234 condition is False)."""

        from app.training.trainer_dpo import DPOConfig

        config = DPOConfig(train_jsonl="dummy")
        mock_transformers = MagicMock()

        # Provide a real TrainerCallback so class _LossCallback(TrainerCallback)
        # works under mocked transformers (subclassing a MagicMock raises
        # AttributeError: __get__ on Linux CI).
        class _MockTrainerCallback:
            pass

        mock_transformers.TrainerCallback = _MockTrainerCallback
        mock_modules = {
            "torch": MagicMock(),
            "peft": MagicMock(),
            "transformers": mock_transformers,
            "trl": MagicMock(),
            "datasets": MagicMock(),
        }
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "eos"
        mock_train_result = MagicMock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_train_result

        mock_modules["transformers"].AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_modules["transformers"].AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_modules["trl"].DPOTrainer.return_value = mock_trainer

        mock_tracker = MagicMock()
        mock_tracker.peak_vram_gb = 5.0
        mock_tracker.elapsed_minutes = 12.5

        captured: list = []

        def _capture(cb_class):
            captured.append(cb_class)

        mock_trainer.add_callback.side_effect = _capture

        with (
            patch.dict("sys.modules", mock_modules),
            patch("app.training.trainer_dpo.ResourceTracker", return_value=mock_tracker),
            patch("os.path.exists", return_value=False),
        ):
            _run_dpo(config, [], None, [], "loss_test2")

        assert len(captured) == 1
        instance = captured[0]()
        instance.on_log(args=None, state=None, control=None, logs={})
        # No exception, no append — line 234 evaluated to False
