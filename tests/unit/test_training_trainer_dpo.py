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

import pytest

from app.schemas.dataset import InstructionExample
from app.schemas.training import TrainingResult
from app.training.config import DPOConfig
from app.training.trainer_dpo import (
    DPOStepEstimate,
    DPOUnavailableError,
    _format_response,
    _make_rejected_response,
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
