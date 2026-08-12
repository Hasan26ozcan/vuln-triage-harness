"""Integration tests for Stage 5 training matrix.

These tests exercise the full Stage 5 flow end-to-end in dry-run mode
(no GPU / model download required):

  1. Real Stage 3 JSONL files are created (InstructionExample format).
  2. run_sft(dry_run=True) loads the data, estimates steps, returns a
     TrainingResult with correct metadata.
  3. run_dpo(dry_run=True) builds preference pairs, estimates DPO steps.
  4. run_lora_sweep(dry_run=True) runs multiple ranks, selects best.
  5. Callbacks (mock W&B, checkpoint, progress) are invoked correctly.
  6. CLI dry-run commands produce correct output via CliRunner.

The data files are generated in Stage 3's InstructionExample JSONL format
(id, sample_id, prompt, target_cwe, target_severity, target_explanation,
target_patch_diff, token_count_estimate) and written to tmp_path.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from app.schemas.dataset import InstructionExample
from app.schemas.training import SweepResult, TrainingResult
from app.training.config import DEFAULT_SWEEP_RANKS, DPOConfig, SFTConfig, SweepConfig
from app.training.sweep import run_lora_sweep
from app.training.trainer_dpo import run_dpo
from app.training.trainer_sft import run_sft

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CWES = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]


def _make_instruction_example(
    idx: int,
    cwe: str,
    severity: str = "high",
    prompt: str = "Classify this vulnerability and suggest a patch.",
    explanation: str = "This code has a vulnerability.",
    patch_diff: str | None = "--- a/file.py\n+++ b/file.py\n- old\n+ new",
) -> InstructionExample:
    return InstructionExample(
        id=f"ie_{idx:04d}",
        sample_id=f"vs_{idx:04d}",
        prompt=prompt,
        target_cwe=cwe,
        target_severity=severity,
        target_explanation=explanation,
        target_patch_diff=patch_diff,
        token_count_estimate=150,
    )


def _write_instruction_jsonl(path: str, examples: list[InstructionExample]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


@pytest.fixture
def stage3_jsonl(tmp_path):
    """Create train/val JSONL files in Stage 3 InstructionExample format."""
    train_examples = [
        _make_instruction_example(0, "CWE-89"),
        _make_instruction_example(1, "CWE-79"),
        _make_instruction_example(2, "CWE-22"),
        _make_instruction_example(3, "CWE-78"),
        _make_instruction_example(4, "CWE-190"),
        _make_instruction_example(5, "CWE-502"),
    ]
    val_examples = [
        _make_instruction_example(100, "CWE-89"),
        _make_instruction_example(101, "CWE-79"),
        _make_instruction_example(102, "CWE-22"),
    ]

    train_path = str(tmp_path / "train.jsonl")
    val_path = str(tmp_path / "val.jsonl")
    _write_instruction_jsonl(train_path, train_examples)
    _write_instruction_jsonl(val_path, val_examples)

    return {"train": train_path, "val": val_path}


# ---------------------------------------------------------------------------
# SFT dry-run integration
# ---------------------------------------------------------------------------


class TestSftDryRunIntegration:
    def test_sft_qlora_dry_run_end_to_end(self, stage3_jsonl):
        """QLoRA SFT dry-run: load data, estimate, return result."""
        config = SFTConfig(
            train_jsonl=stage3_jsonl["train"],
            val_jsonl=stage3_jsonl["val"],
            use_4bit=True,
            lora_r=16,
            num_train_epochs=3,
        )
        result = run_sft(config, dry_run=True)

        assert isinstance(result, TrainingResult)
        assert result.status == "dry_run"
        assert result.method == "sft_qlora"
        assert result.train_set_size == 6
        assert result.peak_vram_gb == 7.0  # QLoRA estimate
        assert result.final_train_loss == 0.0
        assert result.final_val_loss is None
        assert result.checkpoint_uri == ""
        assert result.run_id.startswith("sft_qlora_")

    def test_sft_full_dry_run_end_to_end(self, stage3_jsonl):
        """Full-parameter SFT dry-run: higher VRAM estimate.

        SFT_FULL requires use_4bit=False AND lora_r=0 (no LoRA adapters).
        """
        config = SFTConfig(
            train_jsonl=stage3_jsonl["train"],
            val_jsonl=stage3_jsonl["val"],
            use_4bit=False,
            lora_r=0,
            num_train_epochs=1,
        )
        result = run_sft(config, dry_run=True)

        assert result.method == "sft_full"
        assert result.peak_vram_gb == 16.0  # full SFT estimate

    def test_sft_dry_run_with_callbacks(self, stage3_jsonl):
        """Dry-run should invoke callbacks with init + train_end events."""
        from app.training.callbacks import ProgressCallback, WandbCallback

        config = SFTConfig(
            train_jsonl=stage3_jsonl["train"],
            val_jsonl=stage3_jsonl["val"],
            use_4bit=True,
        )
        wandb_cb = WandbCallback(mock=True)
        progress_cb = ProgressCallback(total_steps=100)

        result = run_sft(config, dry_run=True, callbacks=[wandb_cb, progress_cb])

        # WandbCallback should have recorded init call
        events = [c["event"] for c in wandb_cb.calls]
        assert "init" in events
        assert result.train_set_size == 6


# ---------------------------------------------------------------------------
# DPO dry-run integration
# ---------------------------------------------------------------------------


class TestDpoDryRunIntegration:
    def test_dpo_dry_run_end_to_end(self, stage3_jsonl):
        """DPO dry-run: build preference pairs, estimate steps."""
        config = DPOConfig(
            train_jsonl=stage3_jsonl["train"],
            beta=0.1,
            num_train_epochs=2,
        )
        result = run_dpo(config, dry_run=True)

        assert isinstance(result, TrainingResult)
        assert result.status == "dry_run"
        assert result.method == "dpo"
        assert result.train_set_size == 6  # 6 pairs (one per example)
        assert result.peak_vram_gb == 12.0  # DPO estimate
        assert result.final_train_loss == 0.0
        assert result.checkpoint_uri == ""

    def test_dpo_dry_run_with_val(self, stage3_jsonl):
        """DPO dry-run with validation set."""
        config = DPOConfig(
            train_jsonl=stage3_jsonl["train"],
        )
        result = run_dpo(config, dry_run=True, val_path=stage3_jsonl["val"])
        assert result.train_set_size == 6

    def test_dpo_preference_pairs_built_correctly(self, stage3_jsonl):
        """Verify that preference pairs are built from the train data."""
        from app.training.data import load_examples
        from app.training.trainer_dpo import build_preference_pairs

        train_examples = load_examples(stage3_jsonl["train"])
        pairs = build_preference_pairs(train_examples)
        assert len(pairs) == 6

        # Each pair has prompt, chosen, rejected
        for pair in pairs:
            assert "prompt" in pair
            assert "chosen" in pair
            assert "rejected" in pair
            chosen = json.loads(pair["chosen"])
            rejected = json.loads(pair["rejected"])
            assert chosen["cwe_id"] != rejected["cwe_id"]


# ---------------------------------------------------------------------------
# LoRA sweep dry-run integration
# ---------------------------------------------------------------------------


class TestLoraSweepDryRunIntegration:
    def test_sweep_dry_run_all_ranks(self, stage3_jsonl):
        """Full LoRA sweep with all default ranks in dry-run mode."""
        config = SweepConfig(
            train_jsonl=stage3_jsonl["train"],
            val_jsonl=stage3_jsonl["val"],
            ranks=list(DEFAULT_SWEEP_RANKS),
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)

        assert isinstance(result, SweepResult)
        assert result.num_runs == 5
        assert result.sweep_name is not None
        assert result.best_rank in DEFAULT_SWEEP_RANKS

        # All runs should be dry_run status
        assert all(r.status == "dry_run" for r in result.results)

        # All should have QLoRA VRAM estimate
        assert all(r.peak_vram_gb == 7.0 for r in result.results)

    def test_sweep_best_rank_selected(self, stage3_jsonl):
        """The sweep should select a best rank."""
        config = SweepConfig(
            train_jsonl=stage3_jsonl["train"],
            val_jsonl=stage3_jsonl["val"],
            ranks=[8, 16, 32],
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)
        assert result.best_rank in [8, 16, 32]

    def test_sweep_results_have_correct_ranks(self, stage3_jsonl):
        """Each result should have the correct lora_r in hyperparams."""
        config = SweepConfig(
            train_jsonl=stage3_jsonl["train"],
            ranks=[8, 16, 32],
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)

        ranks_found = [r.hyperparams["lora_r"] for r in result.results]
        assert ranks_found == [8, 16, 32]

    def test_sweep_train_set_size_correct(self, stage3_jsonl):
        """All sweep runs should see the same training set size."""
        config = SweepConfig(
            train_jsonl=stage3_jsonl["train"],
            val_jsonl=stage3_jsonl["val"],
            ranks=[8, 16],
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)

        assert all(r.train_set_size == 6 for r in result.results)

    def test_sweep_summary(self, stage3_jsonl):
        """SweepResult.summary() should produce per-run summaries."""
        config = SweepConfig(
            train_jsonl=stage3_jsonl["train"],
            ranks=[8, 16],
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)

        summaries = result.summary()
        assert len(summaries) == 2
        assert summaries[0]["rank"] == 8
        assert summaries[1]["rank"] == 16


# ---------------------------------------------------------------------------
# CLI integration (dry-run mode)
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    def test_cli_sft_dry_run(self, stage3_jsonl):
        """CLI sft --dry-run should exit 0 and print result info."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "sft",
                "--train-jsonl",
                stage3_jsonl["train"],
                "--val-jsonl",
                stage3_jsonl["val"],
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Run ID:" in result.output
        assert "sft_qlora" in result.output
        assert "Train set: 6 examples" in result.output

    def test_cli_sft_dry_run_full_param(self, stage3_jsonl):
        """CLI sft --dry-run --no-4bit --lora-r 0 should show sft_full method."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "sft",
                "--train-jsonl",
                stage3_jsonl["train"],
                "--val-jsonl",
                stage3_jsonl["val"],
                "--dry-run",
                "--no-4bit",
                "--lora-r",
                "0",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "sft_full" in result.output

    def test_cli_dpo_dry_run(self, stage3_jsonl):
        """CLI dpo --dry-run should exit 0 and print result info."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "dpo",
                "--train-jsonl",
                stage3_jsonl["train"],
                "--dry-run",
                "--beta",
                "0.2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "dpo" in result.output
        assert "Train set: 6 examples" in result.output

    def test_cli_lora_sweep_dry_run(self, stage3_jsonl):
        """CLI lora-sweep --dry-run should exit 0 and print sweep summary."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "lora-sweep",
                "--train-jsonl",
                stage3_jsonl["train"],
                "--val-jsonl",
                stage3_jsonl["val"],
                "--dry-run",
                "--ranks",
                "8,16,32",
                "--no-persist",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Starting LoRA sweep" in result.output
        assert "Best rank:" in result.output

    def test_cli_sft_dry_run_missing_train_jsonl(self):
        """CLI sft without --train-jsonl should fail with error."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["sft", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_cli_sft_dry_run_nonexistent_file(self):
        """CLI sft with nonexistent file should fail with error."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["sft", "--train-jsonl", "/nonexistent/path.jsonl", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_cli_dpo_dry_run_missing_train_jsonl(self):
        """CLI dpo without --train-jsonl should fail with error."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["dpo", "--dry-run"],
        )
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_cli_lora_sweep_dry_run_missing_train_jsonl(self):
        """CLI lora-sweep without --train-jsonl should fail with error."""
        from app.training.cli import app

        runner = CliRunner()
        result = runner.invoke(
            app,
            ["lora-sweep", "--dry-run", "--no-persist"],
        )
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Cross-module wiring
# ---------------------------------------------------------------------------


class TestCrossModuleWiring:
    """Verify that the modules can import each other cleanly and the public
    API is consistent."""

    def test_training_package_exports(self):
        """The app.training package should export all public symbols."""
        from app.training import (
            DPOConfig,
            SFTConfig,
            SweepConfig,
            SweepReport,
            SweepResult,
            TrainingMethod,
            TrainingResult,
            config_to_hyperparams,
            estimate_dpo_steps,
            estimate_training_steps,
            generate_run_id,
            run_dpo,
            run_lora_sweep,
            run_sft,
            validate_config,
        )

        assert all(
            x is not None
            for x in [
                DPOConfig,
                SweepConfig,
                SFTConfig,
                TrainingMethod,
                TrainingResult,
                SweepResult,
                SweepReport,
                run_sft,
                run_dpo,
                run_lora_sweep,
                generate_run_id,
                estimate_training_steps,
                estimate_dpo_steps,
                config_to_hyperparams,
                validate_config,
            ]
        )

    def test_run_id_format_consistency(self):
        """generate_run_id should use the method prefix consistently."""
        from app.training.experiment import generate_run_id

        sft_id = generate_run_id("sft_qlora", None)
        dpo_id = generate_run_id("dpo", None)
        assert sft_id.startswith("sft_qlora_")
        assert dpo_id.startswith("dpo_")

    def test_dry_run_result_persistable(self, stage3_jsonl):
        """A dry-run TrainingResult should be serializable via config_to_hyperparams."""
        from app.training.config import config_to_hyperparams

        config = SFTConfig(
            train_jsonl=stage3_jsonl["train"],
            use_4bit=True,
            lora_r=8,
        )
        run_sft(config, dry_run=True)
        hp = config_to_hyperparams(config)
        assert "lora_r" in hp
        assert hp["lora_r"] == 8
        assert "use_4bit" in hp
        assert hp["use_4bit"] is True

    def test_sweep_uses_default_callbacks_in_dry_run(self, stage3_jsonl):
        """When callbacks_per_run is None, default mock-W&B callbacks are used."""
        config = SweepConfig(
            train_jsonl=stage3_jsonl["train"],
            ranks=[8],
        )
        result = run_lora_sweep(config, dry_run=True, persist=False)
        assert result.num_runs == 1
