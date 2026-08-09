"""Unit tests for Stage 5 LoRA rank sweep.

Covers:
  - run_lora_sweep(dry_run=True): multiple ranks, best-rank selection.
  - SweepReport.to_dict(): serialization.
  - _summarize_run(): per-run summary dict.
  - SweepResult fields and best_rank selection logic.

All tests use dry_run mode — no GPU or model downloads required.
"""

from __future__ import annotations

from app.schemas.training import SweepResult, TrainingResult
from app.training.config import SweepConfig
from app.training.sweep import (
    SweepReport,
    _default_sweep_callbacks,
    _summarize_run,
    run_lora_sweep,
)

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
