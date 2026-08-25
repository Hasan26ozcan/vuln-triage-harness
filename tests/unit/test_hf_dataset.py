"""Tests for the HuggingFace datasets integration (app/data/cleaning/hf_dataset.py).

These tests verify the conversion from VulnSample objects to HF DatasetDict
format, the column schema, and the save/load round-trip — all using mock
samples and a temporary local directory, no network or HF Hub access needed.

On Windows the ``datasets`` package triggers a pyarrow DLL error at import
time, so a functional mock is injected into ``sys.modules`` via the
``mock_datasets`` fixture for any test that touches the real library.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from app.data.cleaning.hf_dataset import (
    HF_COLUMNS,
    pull_from_hub,
    push_to_hub,
    samples_to_hf_dataset,
)
from app.schemas.vuln import VulnSample

# ---------------------------------------------------------------------------
# Lightweight fakes that emulate enough of datasets.Dataset / DatasetDict
# for the tests below to exercise real code paths.
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Minimal stand-in for ``datasets.Dataset`` supporting len, indexing,
    and ``column_names``."""

    def __init__(self, data: dict | None = None):
        self._data = data or {}
        # Derive row count from the first column.
        first_key = next(iter(self._data), None)
        self._rows = list(range(len(self._data.get(first_key, [])))) if first_key else []

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {col: vals[idx] for col, vals in self._data.items()}
        # Column access: ds["cwe_id"] → list of values
        return list(self._data.get(idx, []))

    @property
    def column_names(self):
        return list(self._data.keys())


class _FakeDatasetDict(dict):
    """Stand-in for ``datasets.DatasetDict`` — a dict subclass so
    ``__getitem__`` returns the per-split ``_FakeDataset``."""

    def save_to_disk(self, path: str) -> None:
        """No-op save — in tests we just verify the call happened."""
        self._saved_path = path


_SAVED_DATASETS: dict[str, _FakeDatasetDict] = {}


@pytest.fixture
def mock_datasets():
    """Patch ``sys.modules['datasets']`` with a functional mock.

    Returns the mock module so tests can assert on it if needed.
    The ``save_to_disk`` / ``load_from_disk`` round-trip is simulated by
    an in-memory store keyed on the save path.
    """

    def _from_dict(data):
        return _FakeDataset(data)

    def _dataset_dict(splits):
        obj = _FakeDatasetDict()
        obj.update(splits)
        return obj

    def _save_to_disk(self, path):
        _SAVED_DATASETS[path] = self

    def _load_from_disk(path):
        return _SAVED_DATASETS.get(path, _FakeDatasetDict())

    mock_ds = MagicMock()
    mock_ds.Dataset.from_dict = MagicMock(side_effect=_from_dict)
    mock_ds.DatasetDict = MagicMock(side_effect=_dataset_dict)
    # Wire save_load round-trip through the in-memory store.
    _FakeDatasetDict.save_to_disk = _save_to_disk
    mock_ds.DatasetDict.load_from_disk = MagicMock(side_effect=_load_from_disk)
    mock_ds.load_from_disk = MagicMock(side_effect=_load_from_disk)
    mock_ds.load_dataset = MagicMock(return_value=_FakeDatasetDict())

    # Clear the store at the start of each test.
    _SAVED_DATASETS.clear()

    with patch.dict(sys.modules, {"datasets": mock_ds}):
        yield mock_ds


# ---------------------------------------------------------------------------
# Test data helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# HF_COLUMNS
# ---------------------------------------------------------------------------


def test_hf_columns_match_vuln_sample_fields():
    """HF_COLUMNS must be a subset of VulnSample's exported fields."""
    sample = _sample("s1")
    dump = sample.model_dump()
    for col in HF_COLUMNS:
        assert col in dump, f"Column {col} not in VulnSample schema"


# ---------------------------------------------------------------------------
# samples_to_hf_dataset
# ---------------------------------------------------------------------------


def test_samples_to_hf_dataset_creates_all_three_splits(mock_datasets):
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


def test_hf_dataset_columns_match_schema(mock_datasets):
    samples = [_sample("s1", split="train")]
    ds = samples_to_hf_dataset(samples)

    assert set(ds["train"].column_names) == set(HF_COLUMNS)


def test_hf_dataset_preserves_cwe_id_values(mock_datasets):
    samples = [
        _sample("s1", cwe="CWE-89", split="train"),
        _sample("s2", cwe="CWE-79", split="test"),
    ]
    ds = samples_to_hf_dataset(samples)

    train_cwes = list(ds["train"]["cwe_id"])
    test_cwes = list(ds["test"]["cwe_id"])
    assert train_cwes == ["CWE-89"]
    assert test_cwes == ["CWE-79"]


def test_hf_dataset_skips_samples_with_unset_split(mock_datasets):
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


def test_hf_dataset_empty_split_creates_placeholder(mock_datasets):
    """When a split has no samples, it should still exist as a placeholder."""
    samples = [_sample("s1", split="train")]
    ds = samples_to_hf_dataset(samples)

    assert len(ds["train"]) == 1
    assert len(ds["val"]) == 0
    assert len(ds["test"]) == 0
    # Placeholder should still have the right columns
    assert set(ds["val"].column_names) == set(HF_COLUMNS)


def test_hf_dataset_save_and_load_round_trip(mock_datasets, tmp_path):
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


def test_hf_dataset_static_findings_serialized(mock_datasets):
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


# ---------------------------------------------------------------------------
# push_to_hub
# ---------------------------------------------------------------------------


def test_push_to_hub_raises_without_token(mock_datasets, monkeypatch):
    """When no token is provided and HF_TOKEN env var is unset, raise RuntimeError."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    mock_dataset = MagicMock()
    with pytest.raises(RuntimeError, match="No HuggingFace token"):
        push_to_hub(mock_dataset, repo_id="org/repo", token=None)


def test_push_to_hub_calls_dataset_push_and_returns_url(mock_datasets):
    """With a valid token, delegate to dataset_dict.push_to_hub and return URL."""
    mock_dataset = MagicMock()
    result = push_to_hub(mock_dataset, repo_id="org/repo", token="fake-token")

    mock_dataset.push_to_hub.assert_called_once_with("org/repo", token="fake-token", private=False)
    assert result == "https://huggingface.co/datasets/org/repo"


def test_push_to_hub_uses_env_token(mock_datasets, monkeypatch):
    """When token is None but HF_TOKEN env var is set, use the env var."""
    monkeypatch.setenv("HF_TOKEN", "env-token")
    mock_dataset = MagicMock()
    result = push_to_hub(mock_dataset, repo_id="org/repo")

    mock_dataset.push_to_hub.assert_called_once_with("org/repo", token="env-token", private=False)
    assert result == "https://huggingface.co/datasets/org/repo"


def test_push_to_hub_private_flag(mock_datasets):
    """The ``private`` flag is forwarded to push_to_hub."""
    mock_dataset = MagicMock()
    push_to_hub(mock_dataset, repo_id="org/repo", token="fake-token", private=True)

    mock_dataset.push_to_hub.assert_called_once_with("org/repo", token="fake-token", private=True)


# ---------------------------------------------------------------------------
# pull_from_hub
# ---------------------------------------------------------------------------


def test_pull_from_hub_with_split(mock_datasets):
    """When split is provided, return only that split."""
    mock_dataset = _FakeDatasetDict({"train": _FakeDataset({"id": [1]})})
    mock_datasets.load_dataset = MagicMock(return_value=mock_dataset)

    result = pull_from_hub(repo_id="org/repo", token="fake-token", split="train")

    assert result is mock_dataset["train"]


def test_pull_from_hub_without_split(mock_datasets):
    """When split is None, return the full DatasetDict."""
    mock_dataset = _FakeDatasetDict({"train": _FakeDataset({"id": [1]})})
    mock_datasets.load_dataset = MagicMock(return_value=mock_dataset)

    result = pull_from_hub(repo_id="org/repo", token="fake-token")

    assert result is mock_dataset


def test_pull_from_hub_uses_env_token(mock_datasets, monkeypatch):
    """When token is None, HF_TOKEN env var is read."""
    monkeypatch.setenv("HF_TOKEN", "env-token")
    mock_dataset = _FakeDatasetDict()
    mock_datasets.load_dataset = MagicMock(return_value=mock_dataset)

    pull_from_hub(repo_id="org/repo")

    mock_datasets.load_dataset.assert_called_once()
    call_kwargs = mock_datasets.load_dataset.call_args
    assert call_kwargs.kwargs["token"] == "env-token"


def test_pull_from_hub_with_revision(mock_datasets):
    """The ``revision`` parameter is forwarded to ``load_dataset``."""
    mock_dataset = _FakeDatasetDict()
    mock_datasets.load_dataset = MagicMock(return_value=mock_dataset)

    pull_from_hub(repo_id="org/repo", token="fake-token", revision="abc123")

    call_kwargs = mock_datasets.load_dataset.call_args
    assert call_kwargs.kwargs["revision"] == "abc123"


def test_pull_from_hub_default_repo_id(mock_datasets):
    """The default ``repo_id`` is the project's HF dataset name."""
    mock_dataset = _FakeDatasetDict()
    mock_datasets.load_dataset = MagicMock(return_value=mock_dataset)

    pull_from_hub(token="fake-token")

    call_args = mock_datasets.load_dataset.call_args
    assert "vuln-triage" in call_args.args[0]
