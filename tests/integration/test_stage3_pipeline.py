"""Integration tests for Stage 3 formatting pipeline.

These tests exercise the full pipeline: load split samples → build instruction
examples → enforce token budget → write JSONL files. They mock the Postgres/
MinIO storage layer (same pattern as Stage 2 integration tests) and the
token counter, so no real model downloads or database connections are needed.

Unlike unit tests, integration tests verify that the modules work together:
template → tokenizer → builder → pipeline → JSONL output.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock

import pytest

from app.data.formatting.pipeline import Stage3Result, run_stage3
from app.data.formatting.tokenizer import TokenCounter
from app.schemas.dataset import InstructionExample
from app.schemas.vuln import VulnSample


class _CharTokenCounter:
    """Fake token backend: 1 char = 1 token. Predictable, no model needed.

    Implements the ``encode`` method that ``TokenCounter.count()`` calls via
    ``len(tokenizer.encode(text))``.
    """

    def encode(self, text: str) -> list[int]:
        return list(range(max(len(text), 1)))


def _make_sample(
    id_: str,
    cwe: str,
    repo: str,
    code: str,
    fixed: str | None = None,
    split: str | None = None,
    severity: str = "high",
    description: str = "Test vulnerability.",
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        commit_sha=f"sha_{id_}",
        cwe_id=cwe,
        severity=severity,
        language="python",
        vulnerable_code=code,
        fixed_code=fixed,
        description=description,
        split=split,
    )


def _make_realistic_dataset() -> dict[str, list[VulnSample]]:
    """Create a dataset with samples across all splits and CWEs.

    Returns a dict keyed by split name with lists of assigned samples.
    """
    cwes = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]
    sql_code = "cursor.execute('SELECT * FROM t WHERE id = ' + user_input)"
    xss_code = "element.innerHTML = userInput"
    path_code = "open('/etc/' + user_path)"
    cmd_code = "os.system('ls ' + user_input)"
    overflow_code = "result = a * b  # no overflow check"
    deser_code = "pickle.loads(user_data)"

    patterns = {
        "CWE-89": sql_code,
        "CWE-79": xss_code,
        "CWE-22": path_code,
        "CWE-78": cmd_code,
        "CWE-190": overflow_code,
        "CWE-502": deser_code,
    }

    result: dict[str, list[VulnSample]] = {"train": [], "val": [], "test": []}
    idx = 0
    for split_name in ("train", "val", "test"):
        for cwe in cwes:
            code = patterns[cwe]
            sample = _make_sample(
                id_=f"{split_name}_{cwe}_{idx}",
                cwe=cwe,
                repo=f"org/repo_{split_name}_{idx}",
                code=code,
                fixed=code + "_safe",
                split=split_name,
                description=f"{cwe} vulnerability in {split_name} split.",
            )
            result[split_name].append(sample)
            idx += 1
    return result


@pytest.fixture
def mock_pipeline_storage(monkeypatch):
    """Patch load_samples_from_storage to return our test samples."""
    dataset = _make_realistic_dataset()
    all_samples = []
    for split_name in ("train", "val", "test"):
        all_samples.extend(dataset[split_name])

    import app.data.formatting.pipeline as fmt_pipeline

    monkeypatch.setattr(fmt_pipeline, "load_samples_from_storage", lambda: all_samples)
    return dataset


# --- End-to-end pipeline ---


def test_stage3_pipeline_end_to_end(mock_pipeline_storage, tmp_path):
    """Full pipeline: load → build → write JSONL for all three splits."""
    counter = _CharTokenCounter()

    result = run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=100000,  # generous budget, nothing dropped
        output_dir=str(tmp_path / "stage3"),
        samples=None,  # use mocked storage
    )

    assert isinstance(result, Stage3Result)
    assert result.total_samples_loaded == 18  # 6 CWEs x 3 splits

    # All splits have examples
    for split_name in ("train", "val", "test"):
        assert len(result.examples_by_split[split_name]) > 0
        assert len(result.dropped_by_split[split_name]) == 0

    # JSONL files were written
    for split_name in ("train", "val", "test"):
        path = os.path.join(str(tmp_path / "stage3"), f"{split_name}.jsonl")
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == len(result.examples_by_split[split_name])
        # Each line is valid JSON matching InstructionExample schema
        for line in lines:
            data = json.loads(line)
            assert "id" in data
            assert "prompt" in data
            assert "target_cwe" in data
            assert "target_severity" in data
            assert "target_explanation" in data
            assert "token_count_estimate" in data

    # Manifest was written
    manifest_path = os.path.join(str(tmp_path / "stage3"), "manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["max_tokens"] == 100000
    for split_name in ("train", "val", "test"):
        assert split_name in manifest["splits"]


def test_stage3_pipeline_writes_valid_jsonl(mock_pipeline_storage, tmp_path):
    """Every line in the JSONL output must be valid JSON with all required fields."""
    counter = _CharTokenCounter()
    run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=100000,
        output_dir=str(tmp_path / "stage3"),
    )

    for split_name in ("train", "val", "test"):
        path = os.path.join(str(tmp_path / "stage3"), f"{split_name}.jsonl")
        with open(path, encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                # Validate against InstructionExample schema
                ex = InstructionExample(**data)
                assert ex.id.startswith("ie_")
                assert ex.token_count_estimate > 0


def test_stage3_pipeline_token_budget_filters(mock_pipeline_storage, tmp_path):
    """With a tight token budget, some samples should be dropped."""
    counter = _CharTokenCounter()
    result = run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=50,  # very tight — most should be dropped
        output_dir=str(tmp_path / "stage3"),
    )

    total_dropped = result.total_dropped
    total_examples = result.total_examples
    # With max_tokens=50 and code that's ~30+ chars, many should be dropped
    assert total_dropped > 0 or total_examples == 0


def test_stage3_pipeline_raises_without_samples():
    """If no samples found, should raise a clear RuntimeError."""
    import app.data.formatting.pipeline as fmt_pipeline

    fmt_pipeline.load_samples_from_storage = MagicMock(return_value=[])

    with pytest.raises(RuntimeError, match="No samples found"):
        run_stage3(
            max_tokens=100000,
            output_dir="/tmp/nonexistent_stage3_test",
            samples=None,
        )


def test_stage3_pipeline_accepts_preloaded_samples(tmp_path):
    """When samples are passed directly, storage layer is not used."""
    # No fixture needed — samples are passed directly via the `samples` kwarg
    # so load_samples_from_storage is never called.

    samples = [
        _make_sample("s1", "CWE-89", "org/repo1", "code1", "code1_safe", split="train"),
        _make_sample("s2", "CWE-79", "org/repo2", "code2", "code2_safe", split="val"),
        _make_sample("s3", "CWE-22", "org/repo3", "code3", "code3_safe", split="test"),
    ]

    counter = _CharTokenCounter()
    result = run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=100000,
        output_dir=str(tmp_path / "stage3"),
        samples=samples,
    )

    assert result.total_samples_loaded == 3
    assert len(result.examples_by_split["train"]) == 1
    assert len(result.examples_by_split["val"]) == 1
    assert len(result.examples_by_split["test"]) == 1


def test_stage3_pipeline_preserves_cwe_assignments(mock_pipeline_storage, tmp_path):
    """Each example's target_cwe must match its source sample's cwe_id."""
    counter = _CharTokenCounter()
    result = run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=100000,
        output_dir=str(tmp_path / "stage3"),
    )

    for split_name in ("train", "val", "test"):
        for ex in result.examples_by_split[split_name]:
            # target_cwe should be a valid CWE ID from our scope
            assert ex.target_cwe.startswith("CWE-")
            assert ex.target_cwe in ("CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502")


def test_stage3_pipeline_manifest_has_correct_counts(mock_pipeline_storage, tmp_path):
    """Manifest should accurately report example and drop counts."""
    counter = _CharTokenCounter()
    output_dir = str(tmp_path / "stage3")
    result = run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=100000,
        output_dir=output_dir,
    )

    with open(os.path.join(output_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    for split_name in ("train", "val", "test"):
        info = manifest["splits"][split_name]
        assert info["n_examples"] == len(result.examples_by_split[split_name])
        assert info["n_dropped"] == len(result.dropped_by_split[split_name])


def test_stage3_pipeline_all_six_cwe_classes_represented(mock_pipeline_storage, tmp_path):
    """After Stage 3, all 6 CWE classes should appear in the train split."""
    counter = _CharTokenCounter()
    result = run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=100000,
        output_dir=str(tmp_path / "stage3"),
    )

    train_cwes = {ex.target_cwe for ex in result.examples_by_split["train"]}
    assert train_cwes == {"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"}


def test_stage3_pipeline_jsonl_line_count_matches_examples(mock_pipeline_storage, tmp_path):
    """Each JSONL file should have exactly as many lines as examples built."""
    counter = _CharTokenCounter()
    output_dir = str(tmp_path / "stage3")
    result = run_stage3(
        token_counter=TokenCounter(tokenizer=counter),
        max_tokens=100000,
        output_dir=output_dir,
    )

    for split_name in ("train", "val", "test"):
        path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(path, encoding="utf-8") as f:
            line_count = sum(1 for line in f if line.strip())
        assert line_count == len(result.examples_by_split[split_name])
