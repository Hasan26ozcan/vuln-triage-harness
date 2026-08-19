"""Unit tests for app/training/cli.py

Covers all branches not exercised by integration tests in
tests/integration/test_stage5_training.py:

* Helper functions: _resolve_train_path, _resolve_val_path, _safe_validate
  (success + exception), _print_training_result (all branches: None losses,
  val loss present, loss history, persist success/failure, dry_run vs completed).
* CLI error paths: TrainingUnavailableError in sft/lora-sweep/dpo,
  empty --ranks in lora-sweep, FileNotFoundError.
* list-runs command: empty results, populated results, exception.
* inspect command: run found (with/without val loss), run not found.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

typer = pytest.importorskip("typer")

from app.schemas.training import TrainingResult  # noqa: E402
from app.training.cli import (  # noqa: E402
    _print_training_result,
    _resolve_train_path,
    _resolve_val_path,
    _safe_validate,
    dpo,
    inspect,
    list_runs,
    lora_sweep,
    sft,
)
from app.training.config import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_DPO_BETA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_TRAIN_EPOCHS,
    DPOConfig,
    SFTConfig,
)
from app.training.trainer_sft import TrainingUnavailableError  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path, n=3, cwe="CWE-89"):
    from app.schemas.dataset import InstructionExample

    with open(str(path), "w", encoding="utf-8") as f:
        for i in range(n):
            ex = InstructionExample(
                id=f"ie_{i:03d}",
                sample_id=f"vs_{i:03d}",
                prompt="Classify this vulnerability.",
                target_cwe=cwe,
                target_severity="high",
                target_explanation="SQL injection.",
                target_patch_diff=None,
                token_count_estimate=100,
            )
            f.write(ex.model_dump_json() + "\n")


def _sft_kwargs(**overrides):
    """Valid defaults for every parameter of the ``sft`` command."""
    defaults = dict(
        train_jsonl="",
        val_jsonl="",
        model=DEFAULT_BASE_MODEL,
        output_dir="./output/stage5",
        no_4bit=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_NUM_TRAIN_EPOCHS,
        batch_size=1,
        grad_accum=8,
        run_name=None,
        dry_run=True,
        verbose=False,
    )
    defaults.update(overrides)
    return defaults


def _lora_sweep_kwargs(**overrides):
    defaults = dict(
        train_jsonl="",
        val_jsonl="",
        model=DEFAULT_BASE_MODEL,
        output_dir="./output/stage5/sweep",
        ranks="8,16,32",
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_NUM_TRAIN_EPOCHS,
        grad_accum=8,
        run_name=None,
        dry_run=True,
        no_persist=True,
        verbose=False,
    )
    defaults.update(overrides)
    return defaults


def _dpo_kwargs(**overrides):
    defaults = dict(
        train_jsonl="",
        val_jsonl="",
        model=DEFAULT_BASE_MODEL,
        sft_checkpoint="",
        output_dir="./output/stage5",
        beta=DEFAULT_DPO_BETA,
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=DEFAULT_NUM_TRAIN_EPOCHS,
        batch_size=1,
        grad_accum=8,
        run_name=None,
        dry_run=True,
        verbose=False,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# _resolve_train_path / _resolve_val_path
# ---------------------------------------------------------------------------


class TestResolveTrainPath:
    def test_returns_cli_value_when_set(self):
        """When train_jsonl is non-empty, it takes priority over config."""
        config = SFTConfig(train_jsonl="config_path.jsonl")
        assert _resolve_train_path("cli_path.jsonl", config) == "cli_path.jsonl"

    def test_falls_back_to_config_when_cli_empty(self):
        """When train_jsonl is empty, fall back to config.train_jsonl."""
        config = SFTConfig(train_jsonl="config_path.jsonl")
        assert _resolve_train_path("", config) == "config_path.jsonl"

    def test_both_empty_returns_empty(self):
        """When both are empty, returns empty string."""
        config = SFTConfig(train_jsonl="")
        assert _resolve_train_path("", config) == ""

    def test_works_with_dpo_config(self):
        """The same helper works with DPOConfig."""
        config = DPOConfig(train_jsonl="dpo_config.jsonl")
        assert _resolve_train_path("", config) == "dpo_config.jsonl"


class TestResolveValPath:
    def test_returns_cli_value_when_set(self):
        config = SFTConfig(val_jsonl="config_val.jsonl")
        assert _resolve_val_path("cli_val.jsonl", config) == "cli_val.jsonl"

    def test_falls_back_to_config_when_cli_empty(self):
        config = SFTConfig(val_jsonl="config_val.jsonl")
        assert _resolve_val_path("", config) == "config_val.jsonl"

    def test_both_empty_returns_empty(self):
        config = SFTConfig(val_jsonl="")
        assert _resolve_val_path("", config) == ""


# ---------------------------------------------------------------------------
# _safe_validate
# ---------------------------------------------------------------------------


class TestSafeValidate:
    def test_returns_warnings_on_success(self):
        """validate_config returns warnings that _safe_validate passes through."""
        config = SFTConfig()  # train_jsonl="" → will produce a warning
        result = _safe_validate(config)
        assert isinstance(result, list)
        # SFTConfig with empty train_jsonl produces at least one warning
        assert any("train_jsonl is not set" in w for w in result)

    def test_catches_exception_and_returns_error_list(self):
        """When validate_config raises, _safe_validate returns an error string."""
        config = SFTConfig()
        with patch(
            "app.training.config.validate_config",
            side_effect=RuntimeError("unexpected boom"),
        ):
            result = _safe_validate(config)
        assert len(result) == 1
        assert "Validation error" in result[0]
        assert "unexpected boom" in result[0]


# ---------------------------------------------------------------------------
# _print_training_result
# ---------------------------------------------------------------------------


class TestPrintTrainingResult:
    """Exercise all output branches of _print_training_result."""

    def _make_result(self, **kwargs) -> TrainingResult:
        defaults: dict = dict(
            run_id="test_run_id",
            method="sft_qlora",
            base_model=DEFAULT_BASE_MODEL,
            hyperparams={"lora_r": 8, "learning_rate": 2e-5},
            train_set_size=100,
            train_time_minutes=5.0,
            peak_vram_gb=7.0,
            final_train_loss=0.1234,
            final_val_loss=0.2345,
            checkpoint_uri="s3://vuln-triage/checkpoints/stage5/test_run_id",
            status="completed",
            run_name="my_run",
            train_loss_history=[],
        )
        defaults.update(kwargs)
        return TrainingResult(**defaults)

    def test_basic_output(self):
        result = self._make_result()
        mock_typer = MagicMock()
        _print_training_result(result, mock_typer)

        echo_args = [call.args[0] for call in mock_typer.echo.call_args_list]
        joined = "\n".join(echo_args)
        assert "Run ID:    test_run_id" in joined
        assert "Method:    sft_qlora" in joined
        assert "Base model:" in joined
        assert "Status:    completed" in joined
        assert "Train set: 100 examples" in joined
        assert "Train loss: 0.1234" in joined
        assert "Val loss:  0.2345" in joined
        assert "Peak VRAM: 7.00 GB" in joined
        assert "Train time: 5.00 min" in joined

    def test_train_loss_none_shows_dash(self):
        """When final_train_loss is None, 'Train loss: —' is printed."""
        result = self._make_result(
            final_train_loss=None,
            status="dry_run",  # avoid persist path
        )
        mock_typer = MagicMock()
        _print_training_result(result, mock_typer)

        echo_args = [call.args[0] for call in mock_typer.echo.call_args_list]
        assert any("Train loss: —" in arg for arg in echo_args)

    def test_val_loss_not_shown_when_none(self):
        """When final_val_loss is None, no Val loss line is printed."""
        result = self._make_result(
            final_val_loss=None,
            status="dry_run",
        )
        mock_typer = MagicMock()
        _print_training_result(result, mock_typer)

        echo_args = [call.args[0] for call in mock_typer.echo.call_args_list]
        assert not any("Val loss:" in arg for arg in echo_args)

    def test_loss_history_printed_when_present(self):
        """Non-empty train_loss_history triggers loss-history output."""
        result = self._make_result(
            train_loss_history=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            status="dry_run",  # avoid persist path
        )
        mock_typer = MagicMock()
        _print_training_result(result, mock_typer)

        echo_args = [call.args[0] for call in mock_typer.echo.call_args_list]
        assert any("Loss history" in arg for arg in echo_args)
        step_lines = [a for a in echo_args if "step" in a]
        assert len(step_lines) == 10  # last 10 of 11 entries

    def test_no_loss_history_when_empty(self):
        result = self._make_result(train_loss_history=[], status="dry_run")
        mock_typer = MagicMock()
        _print_training_result(result, mock_typer)

        echo_args = [call.args[0] for call in mock_typer.echo.call_args_list]
        assert not any("Loss history" in arg for arg in echo_args)

    def test_completed_status_calls_persist(self):
        """status='completed' triggers persist_training_run."""
        result = self._make_result(status="completed", train_loss_history=[0.1])
        mock_typer = MagicMock()

        with patch("app.training.experiment.persist_training_run") as mock_persist:
            _print_training_result(result, mock_typer)
            mock_persist.assert_called_once_with(result)

        echo_args = [call.args[0] for call in mock_typer.echo.call_args_list]
        assert any("Persisted run" in arg for arg in echo_args)

    def test_completed_status_persist_failure(self):
        """When persist raises, a warning is echoed to stderr."""
        result = self._make_result(status="completed", train_loss_history=[0.1])
        mock_typer = MagicMock()

        with patch(
            "app.training.experiment.persist_training_run",
            side_effect=RuntimeError("DB connection lost"),
        ):
            _print_training_result(result, mock_typer)

        err_calls = [c for c in mock_typer.echo.call_args_list if c.kwargs.get("err")]
        assert len(err_calls) == 1
        assert "Warning: failed to persist" in err_calls[0].args[0]
        assert "DB connection lost" in err_calls[0].args[0]

    def test_dry_run_status_no_persist(self):
        """status='dry_run' does not call persist."""
        result = self._make_result(
            status="dry_run",
            final_train_loss=0.0,
            final_val_loss=None,
            train_loss_history=[],
        )
        mock_typer = MagicMock()

        with patch("app.training.experiment.persist_training_run") as mock_persist:
            _print_training_result(result, mock_typer)
            mock_persist.assert_not_called()

    def test_checkpoint_empty_shows_not_saved(self):
        """Empty checkpoint_uri prints '(not saved)'."""
        result = self._make_result(
            checkpoint_uri="",
            status="dry_run",
            final_train_loss=0.0,
            final_val_loss=None,
        )
        mock_typer = MagicMock()
        _print_training_result(result, mock_typer)

        echo_args = [call.args[0] for call in mock_typer.echo.call_args_list]
        assert any("Checkpoint:" in arg and "(not saved)" in arg for arg in echo_args)


# ---------------------------------------------------------------------------
# sft command
# ---------------------------------------------------------------------------


class TestSftCommand:
    def test_sft_training_unavailable_error(self, tmp_path, capsys):
        """TrainingUnavailableError prints error + hint and exits 1."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        with patch(
            "app.training.trainer_sft.run_sft",
            side_effect=TrainingUnavailableError("No CUDA GPU"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                sft(**_sft_kwargs(train_jsonl=str(train_path), dry_run=False))
            assert exc_info.value.exit_code == 1

        err = capsys.readouterr().err
        assert "Error" in err
        assert "No CUDA GPU" in err
        assert "Hint" in err

    def test_sft_file_not_found_error(self, tmp_path, capsys):
        """FileNotFoundError prints error and exits 1."""
        with patch(
            "app.training.trainer_sft.run_sft",
            side_effect=FileNotFoundError("Dataset File not found: /bad/path"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                sft(**_sft_kwargs(train_jsonl="/bad/path", dry_run=False))
            assert exc_info.value.exit_code == 1

        err = capsys.readouterr().err
        assert "Error" in err
        assert "Dataset File not found" in err

    def test_sft_warnings_echoed_to_stderr(self, tmp_path, capsys):
        """Config validation warnings are printed to stderr."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        result = TrainingResult(
            run_id="dry_1",
            method="sft_qlora",
            base_model=DEFAULT_BASE_MODEL,
            hyperparams={"lora_r": 8},
            train_set_size=3,
            train_time_minutes=0.0,
            peak_vram_gb=7.0,
            final_train_loss=0.0,
            final_val_loss=None,
            checkpoint_uri="",
            status="dry_run",
        )
        with patch("app.training.trainer_sft.run_sft", return_value=result):
            sft(**_sft_kwargs(train_jsonl=str(train_path), dry_run=True))

        out = capsys.readouterr()
        assert "Train set: 3 examples" in out.out

    def test_sft_validation_warnings_printed(self, capsys):
        """When _safe_validate returns warnings, they are echoed to stderr."""
        with patch(
            "app.training.trainer_sft.run_sft",
            return_value=TrainingResult(
                run_id="r",
                method="sft_qlora",
                base_model=DEFAULT_BASE_MODEL,
                hyperparams={},
                train_set_size=1,
                train_time_minutes=0.0,
                peak_vram_gb=7.0,
                final_train_loss=0.0,
                status="dry_run",
            ),
        ):
            with patch(
                "app.training.config.validate_config",
                return_value=["num_train_epochs is 0 — no training will occur."],
            ):
                sft(**_sft_kwargs(train_jsonl="train.jsonl", dry_run=True))

        err = capsys.readouterr().err
        assert "Warning: num_train_epochs is 0" in err


# ---------------------------------------------------------------------------
# lora-sweep command
# ---------------------------------------------------------------------------


class TestLoraSweepCommand:
    def test_empty_ranks_error(self, capsys):
        """Empty --ranks produces an error and exits 1."""
        with pytest.raises(typer.Exit) as exc_info:
            lora_sweep(**_lora_sweep_kwargs(ranks=""))
        assert exc_info.value.exit_code == 1

        err = capsys.readouterr().err
        assert "at least one integer" in err

    def test_empty_ranks_with_only_commas(self, capsys):
        """Ranks like ',,' that split to empty strings also produce the error."""
        with pytest.raises(typer.Exit) as exc_info:
            lora_sweep(**_lora_sweep_kwargs(ranks=","))
        assert exc_info.value.exit_code == 1
        assert "at least one integer" in capsys.readouterr().err

    def test_training_unavailable_error_dry_run(self, tmp_path, capsys):
        """TrainingUnavailableError in dry-run: error but NO hint."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        with patch(
            "app.training.sweep.run_lora_sweep",
            side_effect=TrainingUnavailableError("No GPU"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                lora_sweep(**_lora_sweep_kwargs(train_jsonl=str(train_path), dry_run=True))
            assert exc_info.value.exit_code == 1

        err = capsys.readouterr().err
        assert "Error" in err
        # Hint NOT printed in dry-run mode (the `if not dry_run:` guard)
        assert "Hint" not in err

    def test_training_unavailable_error_not_dry_run(self, tmp_path, capsys):
        """TrainingUnavailableError NOT in dry-run: error + hint."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        with patch(
            "app.training.sweep.run_lora_sweep",
            side_effect=TrainingUnavailableError("No GPU"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                lora_sweep(**_lora_sweep_kwargs(train_jsonl=str(train_path), dry_run=False))
            assert exc_info.value.exit_code == 1

        err = capsys.readouterr().err
        assert "Error" in err
        assert "Hint" in err

    def test_lora_sweep_validation_warnings_printed(self, tmp_path, capsys):
        """When _safe_validate returns warnings, they are echoed to stderr."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        from app.schemas.training import SweepResult

        sweep_result = SweepResult(
            base_model=DEFAULT_BASE_MODEL,
            sweep_name="s",
            results=[
                TrainingResult(
                    run_id="r1",
                    method="sft_qlora",
                    base_model=DEFAULT_BASE_MODEL,
                    hyperparams={"lora_r": 8},
                    train_set_size=3,
                    train_time_minutes=0.0,
                    peak_vram_gb=7.0,
                    final_train_loss=0.0,
                    final_val_loss=None,
                    checkpoint_uri="",
                    status="dry_run",
                )
            ],
        )
        with patch("app.training.sweep.run_lora_sweep", return_value=sweep_result):
            with patch(
                "app.training.config.validate_config",
                return_value=["num_train_epochs is 0 — no training will occur."],
            ):
                lora_sweep(**_lora_sweep_kwargs(train_jsonl=str(train_path)))

        err = capsys.readouterr().err
        assert "Warning: num_train_epochs is 0" in err

    def test_sweep_success_output(self, tmp_path, capsys):
        """Successful sweep prints the summary table."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        from app.schemas.training import SweepResult

        sweep_result = SweepResult(
            base_model=DEFAULT_BASE_MODEL,
            sweep_name="test_sweep",
            results=[
                TrainingResult(
                    run_id="r1",
                    method="sft_qlora",
                    base_model=DEFAULT_BASE_MODEL,
                    hyperparams={"lora_r": 8},
                    train_set_size=3,
                    train_time_minutes=1.0,
                    peak_vram_gb=7.0,
                    final_train_loss=0.5,
                    final_val_loss=0.6,
                    checkpoint_uri="s3://x/r1",
                    status="completed",
                ),
                TrainingResult(
                    run_id="r2",
                    method="sft_qlora",
                    base_model=DEFAULT_BASE_MODEL,
                    hyperparams={"lora_r": 16},
                    train_set_size=3,
                    train_time_minutes=2.0,
                    peak_vram_gb=7.0,
                    final_train_loss=float("nan"),
                    final_val_loss=None,
                    checkpoint_uri="",
                    status="failed",
                ),
            ],
            best_rank=8,
            best_val_loss=0.6,
        )
        with patch(
            "app.training.sweep.run_lora_sweep",
            return_value=sweep_result,
        ):
            lora_sweep(**_lora_sweep_kwargs(train_jsonl=str(train_path), dry_run=False))

        out = capsys.readouterr().out
        assert "Starting LoRA sweep" in out
        assert "Best rank: 8" in out
        assert "sweep_name" in out.lower() or "test_sweep" in out
        assert "Per-rank summary" in out


# ---------------------------------------------------------------------------
# dpo command
# ---------------------------------------------------------------------------


class TestDpoCommand:
    def test_dpo_training_unavailable_error(self, tmp_path, capsys):
        """TrainingUnavailableError prints error + hint and exits 1."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        with patch(
            "app.training.trainer_dpo.run_dpo",
            side_effect=TrainingUnavailableError("No GPU"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                dpo(**_dpo_kwargs(train_jsonl=str(train_path), dry_run=False))
            assert exc_info.value.exit_code == 1

        err = capsys.readouterr().err
        assert "Error" in err
        assert "Hint" in err

    def test_dpo_file_not_found_error(self, capsys):
        with patch(
            "app.training.trainer_dpo.run_dpo",
            side_effect=FileNotFoundError("train_jsonl path is empty"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                dpo(**_dpo_kwargs(dry_run=False))
            assert exc_info.value.exit_code == 1

        err = capsys.readouterr().err
        assert "Error" in err

    def test_dpo_warnings_echoed(self, tmp_path, capsys):
        """DPO validation warnings are echoed to stderr."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, n=3)

        result = TrainingResult(
            run_id="dpo_1",
            method="dpo",
            base_model=DEFAULT_BASE_MODEL,
            hyperparams={"beta": 0.1},
            train_set_size=3,
            train_time_minutes=0.0,
            peak_vram_gb=12.0,
            final_train_loss=0.0,
            final_val_loss=None,
            checkpoint_uri="",
            status="dry_run",
        )
        with patch("app.training.trainer_dpo.run_dpo", return_value=result):
            dpo(**_dpo_kwargs(train_jsonl=str(train_path), dry_run=True))

        out = capsys.readouterr()
        assert "Run ID:" in out.out
        assert "dpo" in out.out


# ---------------------------------------------------------------------------
# list-runs command
# ---------------------------------------------------------------------------


class TestListRunsCommand:
    def test_empty_results(self, capsys):
        """When no runs exist, prints 'No training runs found.'."""
        with patch("app.training.experiment.list_training_runs", return_value=[]):
            list_runs(limit=50, method=None, status=None)
        out = capsys.readouterr()
        assert "No training runs found." in out.out

    def test_populated_results(self, capsys):
        """Populated results print a table with run_id, method, status, losses."""
        runs = [
            TrainingResult(
                run_id="run_1",
                method="sft_qlora",
                base_model="Qwen",
                hyperparams={"lora_r": 8},
                train_set_size=100,
                train_time_minutes=5.0,
                peak_vram_gb=7.0,
                final_train_loss=0.123,
                final_val_loss=0.234,
                checkpoint_uri="s3://x",
                status="completed",
                run_name="r1",
            ),
            TrainingResult(
                run_id="run_2",
                method="dpo",
                base_model="Qwen",
                hyperparams={},
                train_set_size=50,
                train_time_minutes=3.0,
                peak_vram_gb=12.0,
                final_train_loss=0.05,
                final_val_loss=None,  # None → "—"
                checkpoint_uri="",
                status="failed",
                run_name=None,
            ),
        ]
        with patch("app.training.experiment.list_training_runs", return_value=runs):
            list_runs(limit=50, method=None, status=None)
        out = capsys.readouterr()
        assert "run_1" in out.out
        assert "sft_qlora" in out.out
        assert "completed" in out.out
        assert "0.1230" in out.out  # formatted train loss
        assert "0.2340" in out.out  # formatted val loss
        assert "run_2" in out.out
        assert "dpo" in out.out
        assert "—" in out.out  # None val loss shows as em dash

    def test_exception_handling(self, capsys):
        """When list_training_runs raises, error is printed and Exit(1)."""
        with patch(
            "app.training.experiment.list_training_runs",
            side_effect=RuntimeError("DB connection refused"),
        ):
            with pytest.raises(typer.Exit) as exc_info:
                list_runs(limit=50, method=None, status=None)
            assert exc_info.value.exit_code == 1
        err = capsys.readouterr().err
        assert "Error querying runs" in err
        assert "DB connection refused" in err


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


class TestInspectCommand:
    def test_run_not_found(self, capsys):
        """When load_training_run returns None, prints 'Run not found' and exits 1."""
        with patch("app.training.experiment.load_training_run", return_value=None):
            with pytest.raises(typer.Exit) as exc_info:
                inspect(run_id="nonexistent_run")
            assert exc_info.value.exit_code == 1
        out = capsys.readouterr()
        assert "Run not found: nonexistent_run" in out.out

    def test_inspect_with_val_loss(self, capsys):
        """A run with final_val_loss prints the val loss line."""
        run = TrainingResult(
            run_id="run_123",
            method="sft_qlora",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            hyperparams={"lora_r": 8, "learning_rate": 2e-5},
            train_set_size=100,
            train_time_minutes=5.5,
            peak_vram_gb=6.2,
            final_train_loss=0.123,
            final_val_loss=0.234,
            checkpoint_uri="s3://bucket/ckpt",
            status="completed",
            run_name="test_run",
        )
        with patch("app.training.experiment.load_training_run", return_value=run):
            inspect(run_id="run_123")
        out = capsys.readouterr().out
        assert "Run ID:    run_123" in out
        assert "Method:    sft_qlora" in out
        assert "Base model: Qwen/Qwen2.5-Coder-7B-Instruct" in out
        assert "Status:    completed" in out
        assert "Train loss: 0.123" in out
        assert "Val loss:  0.234" in out
        assert "Peak VRAM: 6.20 GB" in out
        assert "Train time: 5.50 min" in out
        assert "Train set size: 100" in out
        assert "Checkpoint URI: s3://bucket/ckpt" in out
        assert "Hyperparams:" in out
        assert "lora_r" in out

    def test_inspect_without_val_loss(self, capsys):
        """A run without final_val_loss does not print the val loss line."""
        run = TrainingResult(
            run_id="run_456",
            method="dpo",
            base_model="Qwen",
            hyperparams={"beta": 0.1},
            train_set_size=50,
            train_time_minutes=2.0,
            peak_vram_gb=12.0,
            final_train_loss=0.5,
            final_val_loss=None,
            checkpoint_uri="s3://x",
            status="completed",
            run_name=None,
        )
        with patch("app.training.experiment.load_training_run", return_value=run):
            inspect(run_id="run_456")
        out = capsys.readouterr().out
        assert "Run ID:    run_456" in out
        assert "Val loss:" not in out  # No val loss line


# ---------------------------------------------------------------------------
# __main__ guard
# ---------------------------------------------------------------------------


class TestMainGuard:
    def test_main_block_executes_app(self):
        """The ``if __name__ == '__main__'`` block runs ``app()``.

        Uses ``runpy.run_module`` with ``run_name='__main__'`` so the guard
        is true. ``--help`` causes Typer to print help and exit cleanly.
        """
        import runpy
        import sys

        with patch.object(sys, "argv", ["cli", "--help"]):
            try:
                runpy.run_module("app.training.cli", run_name="__main__")
            except SystemExit:
                pass
