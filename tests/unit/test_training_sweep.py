"""Unit tests for Stage 5 LoRA rank sweep.

Covers:
  - run_lora_sweep(dry_run=True): multiple ranks, best-rank selection.
  - SweepReport.to_dict(): serialization.
  - _summarize_run(): per-run summary dict.
  - SweepResult fields and best_rank selection logic.

All tests use dry_run mode — no GPU or model downloads required.
"""

from __future__ import annotations

from unittest.mock import patch

from app.schemas.training import SweepResult, TrainingResult
from app.training.config import SweepConfig
from app.training.sweep import (
    SweepReport,
    _default_sweep_callbacks,
    _summarize_run,
    run_lora_sweep,
)
from app.training.trainer_sft import TrainingUnavailableError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path: str, n: int = 5) -> None:
    from app.schemas.dataset import InstructionExample

    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = InstructionExample(
                id=f"ie_{i:03d}",
                sample_id=f"vs_{i:03d}",
                prompt="Classify this vulnerability.",
                target_cwe="CWE-89",
                target_severity="high",
                target_explanation="SQL injection.",
                target_patch_diff=None,
                token_count_estimate=100,
            )
            f.write(ex.model_dump_json() + "\n")


class _MockLoader:
    """Injectable loader for sweep tests."""

    def __init__(self, n: int = 5):
        self.n = n

    def load(self, path: str):
        from app.schemas.dataset import InstructionExample

        return [
            InstructionExample(
                id=f"ie_{i:03d}",
                sample_id=f"vs_{i:03d}",
                prompt="Classify.",
                target_cwe="CWE-89",
                target_severity="high",
                target_explanation="desc",
                target_patch_diff=None,
                token_count_estimate=100,
            )
            for i in range(self.n)
        ]


# ---------------------------------------------------------------------------
# _summarize_run
# ---------------------------------------------------------------------------


class TestSummarizeRun:
    def test_completed_run(self):
        result = TrainingResult(
            run_id="run_1",
            method="sft_qlora",
            base_model="Qwen/Qwen2.5-Coder-7B",
            hyperparams={"lora_r": 16, "use_4bit": True},
            train_set_size=100,
            train_time_minutes=5.0,
            peak_vram_gb=6.2,
            final_train_loss=0.123,
            final_val_loss=0.234,
            checkpoint_uri="s3://bucket/ckpt",
            status="completed",
        )
        summary = _summarize_run(result)
        assert summary["rank"] == 16
        assert summary["status"] == "completed"
        assert summary["train_loss"] == 0.123
        assert summary["val_loss"] == 0.234
        assert summary["vram_gb"] == 6.2
        assert summary["minutes"] == 5.0
        assert summary["checkpoint_uri"] == "s3://bucket/ckpt"

    def test_failed_run(self):
        result = TrainingResult(
            run_id="run_fail",
            method="sft_qlora",
            base_model="Qwen",
            hyperparams={"lora_r": 32},
            train_set_size=0,
            train_time_minutes=0.0,
            peak_vram_gb=0.0,
            final_train_loss=float("nan"),
            final_val_loss=None,
            checkpoint_uri="",
            status="failed",
        )
        summary = _summarize_run(result)
        assert summary["status"] == "failed"
        # NaN is truthy and not None, so it passes the `is not None` check
        # and gets rounded — round(nan, 4) is still nan
        assert summary["train_loss"] != summary["train_loss"]  # NaN check
        assert summary["val_loss"] is None


# ---------------------------------------------------------------------------
# _default_sweep_callbacks
# ---------------------------------------------------------------------------


class TestDefaultSweepCallbacks:
    def test_returns_list_with_wandb_mock(self):
        callbacks = _default_sweep_callbacks("my_sweep/run_8")
        assert isinstance(callbacks, list)
        assert len(callbacks) >= 1
        # At least one should be a WandbCallback in mock mode
        from app.training.callbacks import WandbCallback

        assert any(isinstance(cb, WandbCallback) and cb.mock for cb in callbacks)


# ---------------------------------------------------------------------------
# SweepReport
# ---------------------------------------------------------------------------


class TestSweepReport:
    def test_to_dict(self):
        report = SweepReport(
            sweep_name="test_sweep",
            base_model="Qwen/Qwen2.5-Coder-7B",
            num_runs=3,
            best_rank=16,
            best_val_loss=0.25,
            best_checkpoint_uri="s3://bucket/ckpt",
            all_runs=[{"rank": 8}, {"rank": 16}, {"rank": 32}],
        )
        d = report.to_dict()
        assert d["sweep_name"] == "test_sweep"
        assert d["base_model"] == "Qwen/Qwen2.5-Coder-7B"
        assert d["num_runs"] == 3
        assert d["best_rank"] == 16
        assert d["best_val_loss"] == 0.25
        assert d["best_checkpoint_uri"] == "s3://bucket/ckpt"
        assert len(d["all_runs"]) == 3


# ---------------------------------------------------------------------------
# run_lora_sweep — dry_run mode
# ---------------------------------------------------------------------------


class TestRunLoraSweepDryRun:
    def test_dry_run_multiple_ranks(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(str(train_path), n=5)
        _write_jsonl(str(val_path), n=3)

        config = SweepConfig(
            ranks=[8, 16, 32],
            train_jsonl=str(train_path),
            val_jsonl=str(val_path),
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)

        assert isinstance(result, SweepResult)
        assert result.num_runs == 3
        assert all(r.status == "dry_run" for r in result.results)
        # All runs have estimated VRAM (QLoRA → 7.0 GB)
        assert all(r.peak_vram_gb == 7.0 for r in result.results)

    def test_dry_run_ranks_have_correct_lora_r(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=5)

        config = SweepConfig(ranks=[8, 16, 32, 64, 128], train_jsonl=str(train_path))
        result = run_lora_sweep(config, dry_run=True, persist=False)

        for i, r in enumerate(result.results):
            assert r.hyperparams["lora_r"] == [8, 16, 32, 64, 128][i]

    def test_dry_run_best_rank_selected(self, tmp_path):
        """With identical val losses, the first rank should be selected as best."""
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(str(train_path), n=5)
        _write_jsonl(str(val_path), n=2)

        config = SweepConfig(
            ranks=[8, 16, 32],
            train_jsonl=str(train_path),
            val_jsonl=str(val_path),
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)

        # Dry-run has no val loss, so best_rank is the first rank (fallback)
        assert result.best_rank == 8

    def test_dry_run_with_injected_loader(self, tmp_path):
        """Loader is injected; a real JSONL file must exist (path checked first)."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=1)  # content doesn't matter — loader is injected

        config = SweepConfig(
            train_jsonl=str(train_path),
            ranks=[8, 16],
        )
        result = run_lora_sweep(config, dry_run=True, persist=False, loader=_MockLoader(n=5))
        assert result.num_runs == 2
        assert all(r.train_set_size == 5 for r in result.results)

    def test_dry_run_sweep_name(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=3)

        config = SweepConfig(
            ranks=[8],
            train_jsonl=str(train_path),
            run_name="my_experiment",
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)
        assert result.sweep_name == "my_experiment"

    def test_dry_run_default_sweep_name(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=3)

        config = SweepConfig(ranks=[8, 16], train_jsonl=str(train_path))
        result = run_lora_sweep(config, dry_run=True, persist=False)
        assert "lora_sweep_" in result.sweep_name

    def test_dry_run_all_statuses_completed(self, tmp_path):
        """In dry-run mode, all results should have status 'dry_run', not 'failed'."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=5)

        config = SweepConfig(ranks=[8, 16, 32, 64, 128], train_jsonl=str(train_path))
        result = run_lora_sweep(config, dry_run=True, persist=False)

        assert all(r.status == "dry_run" for r in result.results)
        assert result.best_rank is not None

    def test_dry_run_train_loss_history_empty(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=3)

        config = SweepConfig(ranks=[8], train_jsonl=str(train_path))
        result = run_lora_sweep(config, dry_run=True, persist=False)
        # Dry-run results have no loss history
        assert all(r.train_loss_history == [] for r in result.results)

    def test_dry_run_with_no_persist_still_works(self, tmp_path):
        """persist=False should skip PostgreSQL writes (which may not be running)."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=3)

        config = SweepConfig(ranks=[8, 16], train_jsonl=str(train_path))
        result = run_lora_sweep(config, dry_run=True, persist=False)
        assert result.num_runs == 2


# ---------------------------------------------------------------------------
# SweepResult.summary()
# ---------------------------------------------------------------------------


class TestSweepResultSummary:
    """Covers SweepResult.summary() — the summary() method in
    app/schemas/training.py (lines 76-88)."""

    def _make_results(self):
        return [
            TrainingResult(
                run_id="run-1",
                method="lora",
                base_model="test/model",
                hyperparams={"lora_r": 8, "lora_alpha": 16, "lr": 2e-4},
                train_set_size=100,
                train_time_minutes=12.345,
                peak_vram_gb=7.5,
                final_train_loss=0.123456,
                final_val_loss=0.234567,
                checkpoint_uri="s3://bucket/run-1",
                train_loss_history=[0.5, 0.3, 0.123456],
            ),
            TrainingResult(
                run_id="run-2",
                method="lora",
                base_model="test/model",
                hyperparams={"lora_r": 16, "lora_alpha": 32, "lr": 2e-4},
                train_set_size=100,
                train_time_minutes=20.0,
                peak_vram_gb=7.6,
                final_train_loss=0.098765,
                final_val_loss=None,
                checkpoint_uri="s3://bucket/run-2",
                train_loss_history=[0.4, 0.2, 0.098765],
            ),
            TrainingResult(
                run_id="run-3",
                method="lora",
                base_model="test/model",
                hyperparams={},  # no lora_r → "?" fallback
                train_set_size=100,
                train_time_minutes=5.5,
                peak_vram_gb=7.7,
                final_train_loss=0.5,
                final_val_loss=0.6,
                checkpoint_uri="",
            ),
        ]

    def test_summary_returns_list_of_dicts(self):
        sweep = SweepResult(
            base_model="test/model",
            sweep_name="sweep-1",
            results=self._make_results(),
        )
        rows = sweep.summary()
        assert isinstance(rows, list)
        assert len(rows) == 3
        assert all(isinstance(row, dict) for row in rows)

    def test_summary_correct_values(self):
        sweep = SweepResult(
            base_model="test/model",
            sweep_name="sweep-1",
            results=self._make_results(),
        )
        rows = sweep.summary()

        # Row 0: rank 8, has val_loss
        assert rows[0]["rank"] == 8
        assert rows[0]["train_loss"] == round(0.123456, 4)
        assert rows[0]["val_loss"] == round(0.234567, 4)
        assert rows[0]["vram_gb"] == round(7.5, 2)
        assert rows[0]["minutes"] == round(12.345, 2)
        assert rows[0]["checkpoint_uri"] == "s3://bucket/run-1"

        # Row 1: rank 16, no val_loss → None
        assert rows[1]["rank"] == 16
        assert rows[1]["val_loss"] is None

        # Row 2: no lora_r → "?"
        assert rows[2]["rank"] == "?"
        assert rows[2]["checkpoint_uri"] == ""

    def test_summary_empty_sweep(self):
        sweep = SweepResult(base_model="test/model", sweep_name="empty")
        assert sweep.summary() == []


# ---------------------------------------------------------------------------
# run_lora_sweep — TrainingUnavailableError branch and persist exception path
# ---------------------------------------------------------------------------


class TestRunLoraSweepErrorPaths:
    """Covers lines 105-108 (TrainingUnavailableError → failed result),
    125-128 (persist exception), and 135-137 (val-loss best selection)."""

    def test_training_unavailable_error_creates_failed_result(self, tmp_path):
        """When run_sft raises TrainingUnavailableError, a failed TrainingResult
        is created and the sweep continues."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=3)

        config = SweepConfig(ranks=[8, 16], train_jsonl=str(train_path))

        with patch(
            "app.training.trainer_sft.run_sft",
            side_effect=TrainingUnavailableError("no GPU"),
        ):
            result = run_lora_sweep(config, dry_run=False, persist=False)

        assert result.num_runs == 2
        # Both runs should have "failed" status
        assert all(r.status == "failed" for r in result.results)
        # best_rank is None because failed results have val_loss=None and
        # train_loss=NaN (NaN comparison returns False), so no best is selected
        assert result.best_rank is None
        assert result.best_val_loss is None

    def test_persist_exception_is_logged_and_continues(self, tmp_path):
        """When persist_training_run raises, the warning is logged and the sweep
        continues to the next run."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=3)

        config = SweepConfig(ranks=[8, 16], train_jsonl=str(train_path))

        # Return completed results (has val_loss) so best-selection path is hit
        def _make_completed_result(rank):
            return TrainingResult(
                run_id=f"run_{rank}",
                method="sft_qlora",
                base_model="test/model",
                hyperparams={"lora_r": rank, "use_4bit": True},
                train_set_size=10,
                train_time_minutes=1.0,
                peak_vram_gb=7.0,
                final_train_loss=0.1,
                final_val_loss=0.5 if rank == 8 else 0.3,
                checkpoint_uri=f"s3://bucket/run_{rank}",
                status="completed",
            )

        with (
            patch(
                "app.training.trainer_sft.run_sft",
                side_effect=[
                    _make_completed_result(8),
                    _make_completed_result(16),
                ],
            ),
            patch(
                "app.training.experiment.persist_training_run",
                side_effect=RuntimeError("DB down"),
            ),
        ):
            result = run_lora_sweep(config, dry_run=False, persist=True)

        assert result.num_runs == 2
        # best should be rank 16 (val_loss 0.3 < 0.5)
        assert result.best_rank == 16
        assert result.best_val_loss == 0.3

    def test_best_val_loss_selection_path(self, tmp_path):
        """Results with non-None final_val_loss select best by lowest val loss."""
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(str(train_path), n=3)

        config = SweepConfig(ranks=[8, 16, 32], train_jsonl=str(train_path))

        def _make_result(rank, val_loss):
            return TrainingResult(
                run_id=f"run_{rank}",
                method="sft_qlora",
                base_model="test/model",
                hyperparams={"lora_r": rank, "use_4bit": True},
                train_set_size=10,
                train_time_minutes=1.0,
                peak_vram_gb=7.0,
                final_train_loss=0.1,
                final_val_loss=val_loss,
                checkpoint_uri=f"s3://bucket/run_{rank}",
                status="completed",
            )

        with (
            patch(
                "app.training.trainer_sft.run_sft",
                side_effect=[
                    _make_result(8, 0.5),
                    _make_result(16, 0.3),
                    _make_result(32, 0.1),
                ],
            ),
            patch("app.training.experiment.persist_training_run"),
        ):
            result = run_lora_sweep(config, dry_run=False, persist=True)

        assert result.best_rank == 32
        assert result.best_val_loss == 0.1
