"""Tests for the HuggingFace datasets integration (app/data/cleaning/hf_dataset.py).

These tests verify the conversion from VulnSample objects to HF DatasetDict
format, the column schema, and the save/load round-trip — all using mock
samples and a temporary local directory, no network or HF Hub access needed.
"""


from unittest.mock import MagicMock, patch

import pytest

from app.data.cleaning.hf_dataset import (
    HF_COLUMNS,
    pull_from_hub,
    push_to_hub,
    samples_to_hf_dataset,
)
from app.schemas.vuln import VulnSample

pytest.importorskip("datasets")


def _sample(
    id_: str,
    cwe: str = "CWE-89",
    repo: str = "org/repo",
    split: str = "train",
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        commit_sha=f"sha_{id_}",
        cve_id=f"CVE-2024-{id_}",
        cwe_id=cwe,
        severity="high",
        language="python",
        vulnerable_code=f"# vulnerable code {id_}",
        fixed_code=f"# fixed code {id_}",
        description=f"Test vuln {id_}",
        split=split,
    )


def test_hf_columns_match_vuln_sample_fields():
    """HF_COLUMNS must be a subset of VulnSample's exported fields."""
    sample = _sample("s1")
    dump = sample.model_dump()
    for col in HF_COLUMNS:
        assert col in dump, f"Column {col} not in VulnSample schema"


def test_samples_to_hf_dataset_creates_all_three_splits():
    samples = [
        _sample("s1", cwe="CWE-89", split="train"),
        _sample("s2", cwe="CWE-79", split="train"),
        _sample("s3", cwe="CWE-22", split="val"),
        _sample("s4", cwe="CWE-78", split="test"),
    ]

    ds = samples_to_hf_dataset(samples)

    assert set(ds.keys()) == {"train", "val", "test"}
    assert len(ds["train"]) == 2
    assert len(ds["val"]) == 1
    assert len(ds["test"]) == 1


def test_hf_dataset_columns_match_schema():
    samples = [_sample("s1", split="train")]
    ds = samples_to_hf_dataset(samples)

    assert set(ds["train"].column_names) == set(HF_COLUMNS)


def test_hf_dataset_preserves_cwe_id_values():
    samples = [
        _sample("s1", cwe="CWE-89", split="train"),
        _sample("s2", cwe="CWE-79", split="test"),
    ]
    ds = samples_to_hf_dataset(samples)

    train_cwes = list(ds["train"]["cwe_id"])
    test_cwes = list(ds["test"]["cwe_id"])
    assert train_cwes == ["CWE-89"]
    assert test_cwes == ["CWE-79"]


def test_hf_dataset_skips_samples_with_unset_split():
    """Samples with split=None should be skipped (they haven't been assigned
    by Stage 2's split module yet)."""
    samples = [
        _sample("s1", split="train"),
        _sample("s2", split=None),  # unassigned
        _sample("s3", split="test"),
    ]
    ds = samples_to_hf_dataset(samples)

    assert len(ds["train"]) == 1
    assert len(ds["test"]) == 1
    # val should exist but be empty (placeholder)
    assert "val" in ds


def test_hf_dataset_empty_split_creates_placeholder():
    """When a split has no samples, it should still exist as a placeholder."""
    samples = [_sample("s1", split="train")]
    ds = samples_to_hf_dataset(samples)

    assert len(ds["train"]) == 1
    assert len(ds["val"]) == 0
    assert len(ds["test"]) == 0
    # Placeholder should still have the right columns
    assert set(ds["val"].column_names) == set(HF_COLUMNS)


def test_hf_dataset_save_and_load_round_trip(tmp_path):
    """Save to disk and load back; verify data is preserved."""
    from app.data.cleaning.hf_dataset import load_dataset_locally, save_dataset_locally

    samples = [
        _sample("s1", cwe="CWE-89", split="train"),
        _sample("s2", cwe="CWE-79", split="val"),
        _sample("s3", cwe="CWE-22", split="test"),
    ]
    ds = samples_to_hf_dataset(samples)

    save_path = str(tmp_path / "test_dataset")
    save_dataset_locally(ds, save_path)

    loaded = load_dataset_locally(save_path)

    assert set(loaded.keys()) == {"train", "val", "test"}
    assert len(loaded["train"]) == 1
    assert len(loaded["val"]) == 1
    assert len(loaded["test"]) == 1
    assert list(loaded["train"]["cwe_id"]) == ["CWE-89"]


def test_hf_dataset_static_findings_serialized():
    """static_findings should be properly serialized to HF format."""
    from app.schemas.vuln import StaticFinding

    sample = _sample("s1", split="train")
    sample.static_findings = [
        StaticFinding(
            tool="semgrep",
            rule_id="python.sqli-string-concat",
            message="Possible SQL injection",
            line_range=(5, 7),
        )
    ]

    ds = samples_to_hf_dataset([sample])
    found = ds["train"][0]["static_findings"]
    assert len(found) == 1
    assert found[0]["rule_id"] == "python.sqli-string-concat"
    assert found[0]["tool"] == "semgrep"


def test_hf_dataset_check_available_raises_without_datasets(monkeypatch):
    """If datasets is not installed, the helper should raise a clear error."""
    import builtins

    from app.data.cleaning.hf_dataset import _check_datasets_available
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "datasets":
            raise ImportError("No module named 'datasets'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(RuntimeError, match="datasets"):
        _check_datasets_available()


# --- push_to_hub ---


def test_push_to_hub_raises_without_token(monkeypatch):
    """When no token is provided and HF_TOKEN env var is unset, raise RuntimeError."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mock_dataset = MagicMock()
    with pytest.raises(RuntimeError, match="No HuggingFace token"):
        push_to_hub(mock_dataset, repo_id="org/repo", token=None)


def test_push_to_hub_calls_dataset_push_and_returns_url():
    """With a valid token, delegate to dataset_dict.push_to_hub and return URL."""
    mock_dataset = MagicMock()
    result = push_to_hub(mock_dataset, repo_id="org/repo", token="fake-token")

    mock_dataset.push_to_hub.assert_called_once_with(
        "org/repo", token="fake-token", private=False
    )
    assert result == "https://huggingface.co/datasets/org/repo"


def test_push_to_hub_uses_env_token(monkeypatch):
    """When token is None but HF_TOKEN env var is set, use the env var."""
    monkeypatch.setenv("HF_TOKEN", "env-token")
    mock_dataset = MagicMock()
    result = push_to_hub(mock_dataset, repo_id="org/repo")

    mock_dataset.push_to_hub.assert_called_once_with(
        "org/repo", token="env-token", private=False
    )
    assert result == "https://huggingface.co/datasets/org/repo"


# --- pull_from_hub ---


def test_pull_from_hub_with_split():
    """When split is provided, return only that split."""
    mock_dataset = MagicMock()
    mock_split = MagicMock(name="train_split")
    mock_dataset.__getitem__ = MagicMock(return_value=mock_split)

    with patch("datasets.load_dataset", return_value=mock_dataset):
        result = pull_from_hub(
            repo_id="org/repo", token="fake-token", split="train"
        )

    mock_dataset.__getitem__.assert_called_once_with("train")
    assert result is mock_split


def test_pull_from_hub_without_split():
    """When split is None, return the full DatasetDict."""
    mock_dataset = MagicMock()

    with patch("datasets.load_dataset", return_value=mock_dataset):
        result = pull_from_hub(repo_id="org/repo", token="fake-token")

    assert result is mock_dataset
