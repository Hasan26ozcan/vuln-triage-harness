"""Unit tests for Stage 3 pipeline (``app.data.formatting.pipeline``).

Covers:
  - ``Stage3Result`` dataclass (properties + counts)
  - ``write_jsonl`` file output
  - ``_filter_by_split`` helper
  - ``run_stage3`` — pre-loaded samples, storage fallback, empty-samples
    RuntimeError, token-budget drops, manifest writing
  - ``load_from_hf_dataset`` — iterating splits, field filtering
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.data.formatting.builder import BuildResult
from app.data.formatting.pipeline import (
    OUTPUT_SPLITS,
    Stage3Result,
    _filter_by_split,
    load_from_hf_dataset,
    run_stage3,
    write_jsonl,
)
from app.schemas.dataset import InstructionExample
from app.schemas.vuln import VulnSample


def _make_sample(sample_id="s1", cwe_id="CWE-89", split="train"):
    return VulnSample(
        id=sample_id,
        source="cve_real",
        repo_name="test/repo",
        cwe_id=cwe_id,
        severity="high",
        language="python",
        vulnerable_code="import os; os.system('rm -rf /')",
        fixed_code="pass",
        description="test",
        split=split,
    )


def _make_instruction_example(sample_id="s1"):
    return InstructionExample(
        id="ie_001",
        sample_id=sample_id,
        prompt="Fix the vulnerability:",
        target_cwe="CWE-89",
        target_severity="high",
        target_explanation="SQL injection",
        target_patch_diff="--- a/app.py\n+++ b/app.py\n- old\n+ new",
        token_count_estimate=100,
    )


# ---------------------------------------------------------------------------
# Stage3Result
# ---------------------------------------------------------------------------


class TestStage3Result:
    """Tests for the Stage3Result dataclass properties."""

    def test_total_examples(self):
        """total_examples sums example counts across splits."""
        result = Stage3Result(
            total_samples_loaded=3,
            examples_by_split={
                "train": [_make_instruction_example("s1")],
                "val": [_make_instruction_example("s2"), _make_instruction_example("s3")],
                "test": [],
            },
            dropped_by_split={"train": [], "val": [], "test": []},
            token_counter=None,
            output_dir="/tmp/out",
            max_tokens=4096,
        )
        assert result.total_examples == 3

    def test_total_dropped(self):
        """total_dropped sums drop counts across splits."""
        result = Stage3Result(
            total_samples_loaded=3,
            examples_by_split={"train": [], "val": [], "test": []},
            dropped_by_split={
                "train": [("s1", 5000), ("s2", 5100)],
                "val": [("s3", 6000)],
                "test": [],
            },
            token_counter=None,
            output_dir="/tmp/out",
            max_tokens=4096,
        )
        assert result.total_dropped == 3

    def test_counts(self):
        """counts() returns per-split kept/dropped dict."""
        result = Stage3Result(
            total_samples_loaded=3,
            examples_by_split={
                "train": [_make_instruction_example("s1")],
                "val": [_make_instruction_example("s2"), _make_instruction_example("s3")],
                "test": [],
            },
            dropped_by_split={
                "train": [("d1", 5000)],
                "val": [],
                "test": [],
            },
            token_counter=None,
            output_dir="/tmp/out",
            max_tokens=4096,
        )
        counts = result.counts()
        assert counts["train"] == {"kept": 1, "dropped": 1}
        assert counts["val"] == {"kept": 2, "dropped": 0}
        assert counts["test"] == {"kept": 0, "dropped": 0}


# ---------------------------------------------------------------------------
# write_jsonl
# ---------------------------------------------------------------------------


class TestWriteJsonl:
    """Tests for write_jsonl."""

    def test_writes_jsonl(self, tmp_path):
        """write_jsonl writes one JSON per line and returns count."""
        examples = [
            _make_instruction_example("s1"),
            _make_instruction_example("s2"),
        ]
        path = str(tmp_path / "train.jsonl")
        n = write_jsonl(examples, path)
        assert n == 2
        lines = Path(path).read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["sample_id"] == "s1"

    def test_empty_list(self, tmp_path):
        """write_jsonl with empty list writes 0 lines."""
        path = str(tmp_path / "empty.jsonl")
        n = write_jsonl([], path)
        assert n == 0


# ---------------------------------------------------------------------------
# _filter_by_split
# ---------------------------------------------------------------------------


class TestFilterBySplit:
    """Tests for _filter_by_split."""

    def test_filters_by_split(self):
        """Only samples matching the split are returned."""
        samples = [
            _make_sample("s1", split="train"),
            _make_sample("s2", split="val"),
            _make_sample("s3", split="train"),
        ]
        train = _filter_by_split(samples, "train")
        assert len(train) == 2
        assert train[0].id == "s1"
        assert train[1].id == "s3"

    def test_no_matches(self):
        """No samples match → empty list."""
        samples = [_make_sample("s1", split="train")]
        result = _filter_by_split(samples, "test")
        assert result == []

    def test_none_sample_split(self):
        """Sample with split=None is not matched."""
        s = _make_sample("s1", split="train")
        s.split = None
        result = _filter_by_split([s], "train")
        assert result == []


# ---------------------------------------------------------------------------
# run_stage3
# ---------------------------------------------------------------------------


class TestRunStage3:
    """Tests for run_stage3 — the full pipeline orchestration."""

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    @patch("app.data.formatting.pipeline.build_examples")
    def test_with_pre_loaded_samples(self, mock_build, mock_load):
        """run_stage3 with pre-loaded samples skips storage loading."""
        mock_build.return_value = BuildResult(
            examples=[_make_instruction_example("s1")],
            dropped=[],
        )
        samples = [_make_sample("s1", split="train")]
        with patch("app.data.formatting.pipeline.write_jsonl"):
            result = run_stage3(
                max_tokens=4096,
                output_dir="/tmp/stage3_test",
                samples=samples,
            )
        mock_load.assert_not_called()
        assert result.total_samples_loaded == 1
        assert result.max_tokens == 4096
        assert "train" in result.examples_by_split

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    @patch("app.data.formatting.pipeline.build_examples")
    def test_without_samples_loads_from_storage(self, mock_build, mock_load):
        """run_stage3 with samples=None loads from storage."""
        mock_load.return_value = [
            _make_sample("s1", split="train"),
            _make_sample("s2", split="val"),
        ]
        mock_build.return_value = BuildResult(
            examples=[_make_instruction_example("s1")],
            dropped=[],
        )
        with patch("app.data.formatting.pipeline.write_jsonl"):
            result = run_stage3(max_tokens=4096, samples=None)
        mock_load.assert_called_once()
        assert result.total_samples_loaded == 2

    def test_empty_samples_raises_runtime_error(self):
        """Empty sample list → RuntimeError."""
        with pytest.raises(RuntimeError, match="No samples found"):
            run_stage3(max_tokens=4096, samples=[])

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    def test_storage_returns_empty_raises(self, mock_load):
        """Storage returns empty list → RuntimeError."""
        mock_load.return_value = []
        with pytest.raises(RuntimeError, match="No samples found"):
            run_stage3(max_tokens=4096, samples=None)

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    @patch("app.data.formatting.pipeline.build_examples")
    def test_writes_jsonl_and_manifest(self, mock_build, mock_load, tmp_path):
        """run_stage3 writes JSONL files and manifest.json."""
        mock_load.return_value = [
            _make_sample("s1", split="train"),
            _make_sample("s2", split="val"),
            _make_sample("s3", split="test"),
        ]
        mock_build.return_value = BuildResult(
            examples=[_make_instruction_example("s1")],
            dropped=[],
        )
        run_stage3(
            max_tokens=4096,
            output_dir=str(tmp_path),
            samples=None,
        )
        # Check JSONL files exist
        for split in OUTPUT_SPLITS:
            assert (tmp_path / f"{split}.jsonl").exists()
        # Check manifest exists and has expected fields
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["max_tokens"] == 4096
        assert "token_counter_model" in manifest
        assert "splits" in manifest
        for split in OUTPUT_SPLITS:
            assert split in manifest["splits"]
            assert "n_examples" in manifest["splits"][split]
            assert "n_dropped" in manifest["splits"][split]

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    @patch("app.data.formatting.pipeline.build_examples")
    def test_returns_correct_result(self, mock_build, mock_load):
        """run_stage3 returns a Stage3Result with correct sample count."""
        mock_load.return_value = [
            _make_sample("s1", split="train"),
            _make_sample("s2", split="train"),
        ]
        mock_build.return_value = BuildResult(
            examples=[_make_instruction_example("s1")],
            dropped=[("s2", 5000)],
        )
        with patch("app.data.formatting.pipeline.write_jsonl"):
            result = run_stage3(max_tokens=4096, samples=None)
        # build_examples is called once per split (OUTPUT_SPLITS = train, val, test)
        # mock_build always returns 1 example + 1 dropped per call → 3 total
        assert result.total_samples_loaded == 2
        assert result.total_examples == 3
        assert result.total_dropped == 3
        assert result.output_dir == "./output/stage3"

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    @patch("app.data.formatting.pipeline.build_examples")
    def test_token_counter_model_name_in_manifest(self, mock_build, mock_load, tmp_path):
        """When token_counter has model_name attr, manifest uses it."""
        mock_counter = MagicMock()
        mock_counter.model_name = "test-tokenizer"
        mock_load.return_value = [_make_sample("s1", split="train")]
        mock_build.return_value = BuildResult(examples=[], dropped=[])
        run_stage3(
            max_tokens=4096,
            output_dir=str(tmp_path),
            samples=mock_load.return_value,
            token_counter=mock_counter,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["token_counter_model"] == "test-tokenizer"

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    @patch("app.data.formatting.pipeline.build_examples")
    def test_token_counter_no_model_name(self, mock_build, mock_load, tmp_path):
        """When token_counter lacks model_name, manifest uses 'unknown'."""
        # spec=[] prevents auto-creation of arbitrary attributes like model_name
        mock_counter = MagicMock(spec=[])
        mock_load.return_value = [_make_sample("s1", split="train")]
        mock_build.return_value = BuildResult(examples=[], dropped=[])
        run_stage3(
            max_tokens=4096,
            output_dir=str(tmp_path),
            samples=mock_load.return_value,
            token_counter=mock_counter,
        )
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["token_counter_model"] == "unknown"

    @patch("app.data.formatting.pipeline.load_samples_from_storage")
    @patch("app.data.formatting.pipeline.build_examples")
    def test_default_token_counter_created(self, mock_build, mock_load):
        """run_stage3 creates a default TokenCounter when none is provided."""
        mock_load.return_value = [_make_sample("s1", split="train")]
        mock_build.return_value = BuildResult(examples=[], dropped=[])
        with patch("app.data.formatting.pipeline.TokenCounter") as mock_tc:
            mock_instance = MagicMock()
            mock_instance.model_name = "default"
            mock_tc.return_value = mock_instance
            with patch("app.data.formatting.pipeline.write_jsonl"):
                run_stage3(max_tokens=4096, samples=mock_load.return_value)
        mock_tc.assert_called_once()


# ---------------------------------------------------------------------------
# load_from_hf_dataset
# ---------------------------------------------------------------------------


class TestLoadFromHfDataset:
    """Tests for load_from_hf_dataset."""

    @staticmethod
    def _row(sample_id="s1", split_name="train", cwe_id="CWE-89", **kwargs):
        """Build a minimal VulnSample-compatible row dict with all required fields."""
        base = {
            "id": sample_id,
            "source": "cve_real",
            "repo_name": "test/repo",
            "cwe_id": cwe_id,
            "severity": "high",
            "language": "python",
            "vulnerable_code": "import os; os.system('rm -rf /')",
            "description": "test vuln",
            "split": split_name,
        }
        base.update(kwargs)
        return base

    @patch("app.data.formatting.pipeline.load_dataset_locally")
    def test_loads_all_splits(self, mock_load_dataset):
        """load_from_hf_dataset iterates over all OUTPUT_SPLITS."""
        mock_load_dataset.return_value = {
            "train": [self._row("s1", "train", "CWE-89")],
            "val": [self._row("s2", "val", "CWE-79")],
            "test": [self._row("s3", "test", "CWE-22")],
        }
        samples = load_from_hf_dataset("/tmp/fake_hf")
        assert len(samples) == 3
        assert samples[0].id == "s1"
        assert samples[1].id == "s2"
        assert samples[2].id == "s3"
        mock_load_dataset.assert_called_once_with("/tmp/fake_hf")

    @patch("app.data.formatting.pipeline.load_dataset_locally")
    def test_filters_to_model_fields(self, mock_load_dataset):
        """Only fields matching VulnSample.model_fields are passed."""
        row = self._row("s1", "train", "CWE-89", extra_field="should be ignored")
        mock_load_dataset.return_value = {"train": [row]}
        samples = load_from_hf_dataset("/tmp/fake_hf")
        assert len(samples) == 1
        assert samples[0].id == "s1"
        # extra_field should not cause issues (VulnSample doesn't accept it)

    @patch("app.data.formatting.pipeline.load_dataset_locally")
    def test_skips_missing_splits(self, mock_load_dataset):
        """Splits not present in dataset_dict are skipped."""
        mock_load_dataset.return_value = {
            "train": [self._row("s1", "train")],
        }
        samples = load_from_hf_dataset("/tmp/fake_hf")
        assert len(samples) == 1
        assert samples[0].split == "train"

    @patch("app.data.formatting.pipeline.load_dataset_locally")
    def test_empty_dataset_dict(self, mock_load_dataset):
        """Empty dataset dict → empty samples list."""
        mock_load_dataset.return_value = {}
        samples = load_from_hf_dataset("/tmp/empty_hf")
        assert samples == []

    @patch("app.data.formatting.pipeline.load_dataset_locally")
    def test_custom_splits(self, mock_load_dataset):
        """Custom splits parameter controls which splits are loaded."""
        mock_load_dataset.return_value = {
            "train": [self._row("s1", "train")],
            "val": [self._row("s2", "val")],
            "test": [self._row("s3", "test")],
        }
        samples = load_from_hf_dataset("/tmp/fake_hf", splits=("train",))
        assert len(samples) == 1
        assert samples[0].split == "train"
