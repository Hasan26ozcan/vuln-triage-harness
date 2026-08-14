"""Unit tests for Stage 5 training data module.

Covers:
  - JsonlDataLoader: loading valid JSONL, skipping invalid lines,
    raising FileNotFoundError for missing files.
  - load_examples: injecting a mock loader.
  - compute_stats / DatasetStats: distributions, avg token estimate, to_dict.
  - examples_to_dict_list: prompt/completion/cwe_id mapping.
  - load_stage3_dataset: train + val split loading.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.schemas.dataset import InstructionExample
from app.training.data import (
    DatasetStats,
    JsonlDataLoader,
    compute_stats,
    examples_to_dict_list,
    load_examples,
    load_stage3_dataset,
    make_hf_dataset,
    make_hf_dataset_pair,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_example(
    id_: str = "ie_001",
    cwe: str = "CWE-89",
    severity: str = "high",
    explanation: str = "SQL injection.",
    patch_diff: str | None = None,
    token_count: int = 100,
    sample_id: str = "vs_001",
    prompt: str = "Classify the vulnerability.",
) -> InstructionExample:
    return InstructionExample(
        id=id_,
        sample_id=sample_id,
        prompt=prompt,
        target_cwe=cwe,
        target_severity=severity,
        target_explanation=explanation,
        target_patch_diff=patch_diff,
        token_count_estimate=token_count,
    )


def _write_jsonl(path, examples: list[InstructionExample]) -> None:
    with open(str(path), "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# JsonlDataLoader
# ---------------------------------------------------------------------------


class TestJsonlDataLoader:
    def test_load_valid_jsonl(self, tmp_path):
        path = tmp_path / "train.jsonl"
        examples = [_make_example(id_="ie_1"), _make_example(id_="ie_2")]
        _write_jsonl(path, examples)

        loader = JsonlDataLoader()
        loaded = loader.load(str(path))
        assert len(loaded) == 2
        assert all(isinstance(e, InstructionExample) for e in loaded)
        assert loaded[0].id == "ie_1"
        assert loaded[1].id == "ie_2"

    def test_load_skips_blank_lines(self, tmp_path):
        path = tmp_path / "train.jsonl"
        examples = [_make_example(id_="ie_1")]
        _write_jsonl(path, examples)
        # Append blank lines and a comment-like line
        with open(str(path), "a", encoding="utf-8") as f:
            f.write("\n\n")

        loaded = JsonlDataLoader().load(str(path))
        assert len(loaded) == 1

    def test_load_skips_invalid_json_lines(self, tmp_path):
        path = tmp_path / "train.jsonl"
        path.write_text(
            '{"id": "ie_1", "sample_id": "vs_1", "prompt": "p", '
            '"target_cwe": "CWE-89", "target_severity": "high", '
            '"target_explanation": "e", "token_count_estimate": 100}\n'
            "THIS IS NOT JSON\n"
            '{"id": "ie_2", "sample_id": "vs_2", "prompt": "p2", '
            '"target_cwe": "CWE-79", "target_severity": "medium", '
            '"target_explanation": "e2", "token_count_estimate": 200}\n'
        )
        loaded = JsonlDataLoader().load(str(path))
        assert len(loaded) == 2
        assert loaded[0].id == "ie_1"
        assert loaded[1].id == "ie_2"

    def test_load_missing_file_raises(self):
        loader = JsonlDataLoader()
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            loader.load("nonexistent/path/train.jsonl")

    def test_load_empty_file_returns_empty_list(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        loaded = JsonlDataLoader().load(str(path))
        assert loaded == []


# ---------------------------------------------------------------------------
# load_examples (with injectable loader)
# ---------------------------------------------------------------------------


class TestLoadExamples:
    def test_load_with_injected_mock_loader(self):
        """When a mock loader is injected, no real file is read."""
        mock_examples = [_make_example(id_="ie_1"), _make_example(id_="ie_2")]

        class _MockLoader:
            def load(self, path: str) -> list[InstructionExample]:
                return mock_examples

        loaded = load_examples("fake_path.jsonl", loader=_MockLoader())
        assert loaded is mock_examples
        assert len(loaded) == 2

    def test_load_uses_default_loader_when_none(self, tmp_path):
        path = tmp_path / "train.jsonl"
        examples = [_make_example(id_="ie_1")]
        _write_jsonl(path, examples)

        loaded = load_examples(str(path))
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# compute_stats / DatasetStats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_single_example(self):
        examples = [_make_example(cwe="CWE-89", severity="high", token_count=150)]
        stats = compute_stats(examples)
        assert stats.n_examples == 1
        assert stats.cwe_distribution == {"CWE-89": 1}
        assert stats.severity_distribution == {"high": 1}
        assert stats.avg_token_estimate == 150.0

    def test_multiple_examples_with_distribution(self):
        examples = [
            _make_example(cwe="CWE-89", severity="high", token_count=100),
            _make_example(cwe="CWE-89", severity="high", token_count=200),
            _make_example(cwe="CWE-79", severity="medium", token_count=300),
        ]
        stats = compute_stats(examples)
        assert stats.n_examples == 3
        assert stats.cwe_distribution == {"CWE-89": 2, "CWE-79": 1}
        assert stats.severity_distribution == {"high": 2, "medium": 1}
        assert stats.avg_token_estimate == 200.0

    def test_empty_examples(self):
        stats = compute_stats([])
        assert stats.n_examples == 0
        assert stats.cwe_distribution == {}
        assert stats.severity_distribution == {}
        assert stats.avg_token_estimate == 0.0

    def test_to_dict(self):
        examples = [_make_example(token_count=42)]
        stats = compute_stats(examples)
        d = stats.to_dict()
        assert d["n_examples"] == 1
        assert d["cwe_distribution"] == {"CWE-89": 1}
        assert d["avg_token_estimate"] == 42.0


class TestDatasetStats:
    def test_to_dict_structure(self):
        stats = DatasetStats(
            n_examples=5,
            cwe_distribution={"CWE-89": 3, "CWE-79": 2},
            severity_distribution={"high": 4, "low": 1},
            avg_token_estimate=123.456,
        )
        d = stats.to_dict()
        assert d["n_examples"] == 5
        assert d["cwe_distribution"] == {"CWE-89": 3, "CWE-79": 2}
        assert d["severity_distribution"] == {"high": 4, "low": 1}
        assert d["avg_token_estimate"] == 123.46  # rounded to 2 decimal places


# ---------------------------------------------------------------------------
# examples_to_dict_list
# ---------------------------------------------------------------------------


class TestExamplesToDictList:
    def test_basic_conversion(self):
        examples = [
            _make_example(
                prompt="Classify this:",
                cwe="CWE-89",
                severity="high",
                explanation="SQL injection.",
                patch_diff="--- a/x\n+++ b/x\n- old\n+ new",
            )
        ]
        rows = examples_to_dict_list(examples)
        assert len(rows) == 1
        row = rows[0]
        assert row["prompt"] == "Classify this:"
        assert row["cwe_id"] == "CWE-89"
        # completion is a JSON string with the targets
        completion = json.loads(row["completion"])
        assert completion["cwe_id"] == "CWE-89"
        assert completion["severity"] == "high"
        assert completion["explanation"] == "SQL injection."
        assert completion["patch_diff"] == "--- a/x\n+++ b/x\n- old\n+ new"

    def test_completion_is_valid_json(self):
        examples = [
            _make_example(
                prompt="p",
                cwe="CWE-79",
                severity="medium",
                explanation="XSS.",
                patch_diff=None,
            )
        ]
        rows = examples_to_dict_list(examples)
        completion = json.loads(rows[0]["completion"])
        assert completion["cwe_id"] == "CWE-79"
        assert completion["severity"] == "medium"
        assert completion["explanation"] == "XSS."
        assert completion["patch_diff"] is None

    def test_multiple_examples(self):
        examples = [
            _make_example(id_="ie_1", cwe="CWE-89"),
            _make_example(id_="ie_2", cwe="CWE-79"),
            _make_example(id_="ie_3", cwe="CWE-22"),
        ]
        rows = examples_to_dict_list(examples)
        assert len(rows) == 3
        assert rows[0]["cwe_id"] == "CWE-89"
        assert rows[1]["cwe_id"] == "CWE-79"
        assert rows[2]["cwe_id"] == "CWE-22"


# ---------------------------------------------------------------------------
# load_stage3_dataset
# ---------------------------------------------------------------------------


class TestLoadStage3Dataset:
    def test_load_train_and_val(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        val_path = tmp_path / "val.jsonl"
        _write_jsonl(train_path, [_make_example(id_="ie_1")])
        _write_jsonl(val_path, [_make_example(id_="ie_2")])

        data = load_stage3_dataset(str(train_path), str(val_path))
        assert "train" in data
        assert "val" in data
        assert len(data["train"]) == 1
        assert len(data["val"]) == 1

    def test_load_only_train(self, tmp_path):
        train_path = tmp_path / "train.jsonl"
        _write_jsonl(train_path, [_make_example(id_="ie_1"), _make_example(id_="ie_2")])

        data = load_stage3_dataset(str(train_path))
        assert len(data["train"]) == 2
        assert data["val"] == []

    def test_load_with_injected_loader(self):
        mock_examples = [_make_example(id_="ie_1")]

        class _MockLoader:
            def load(self, path: str) -> list[InstructionExample]:
                return mock_examples

        data = load_stage3_dataset("fake_train.jsonl", "fake_val.jsonl", loader=_MockLoader())
        assert len(data["train"]) == 1
        assert len(data["val"]) == 1


# ---------------------------------------------------------------------------
# make_hf_dataset  (requires the optional ``datasets`` package)
# ---------------------------------------------------------------------------


class TestMakeHfDataset:
    """Covers ``make_hf_dataset`` — the lazy ``datasets`` import and fallback."""

    def test_make_hf_dataset_success(self):
        """With ``datasets`` available, a ``Dataset`` is created from examples."""
        pytest.importorskip("datasets")
        examples = [
            _make_example(id_="ie_1", prompt="p1", cwe="CWE-89",
                          severity="high", explanation="e1", patch_diff=None),
        ]
        ds = make_hf_dataset(examples)
        assert ds is not None
        # The returned object should be a datasets.Dataset instance.
        assert type(ds).__module__.startswith("datasets")

    def test_make_hf_dataset_import_error_raises_runtime(self):
        """When ``datasets`` cannot be imported, ``RuntimeError`` is raised."""
        with patch.dict("sys.modules", {"datasets": None}):
            examples = [_make_example(id_="ie_1")]
            with pytest.raises(RuntimeError, match="datasets is not installed"):
                make_hf_dataset(examples)


# ---------------------------------------------------------------------------
# make_hf_dataset_pair
# ---------------------------------------------------------------------------


class TestMakeHfDatasetPair:
    """Covers ``make_hf_dataset_pair`` — the dict-of-datasets return path."""

    def test_make_hf_dataset_pair_success(self):
        """A dict with ``train`` and ``validation`` keys is returned."""
        pytest.importorskip("datasets")
        train_examples = [
            _make_example(id_="t1", prompt="p1", cwe="CWE-89",
                          severity="high", explanation="e1", patch_diff=None),
        ]
        val_examples = [_make_example(id_="v1")]
        result = make_hf_dataset_pair(train_examples, val_examples)
        assert "train" in result
        assert "validation" in result
        assert type(result["train"]).__module__.startswith("datasets")
        assert type(result["validation"]).__module__.startswith("datasets")
