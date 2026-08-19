"""Tests for the Stage 2 cleaning pipeline (app/data/cleaning/pipeline.py).

These tests mock Postgres (get_session) and MinIO (get_json) so the pipeline
orchestration — loading, dedup, split, contamination check, persistence —
can be verified in isolation without real infrastructure.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from app.data.cleaning.pipeline import (
    Stage2Result,
    load_samples_from_storage,
    persist_splits,
    run_stage2,
)
from app.data.cleaning.split import DEFAULT_SEED, SplitConfig, SplitResult
from app.schemas.vuln import VulnSample


def _sample(
    id_: str,
    code: str = "vulnerable code",
    repo: str = "org/repo",
    split: str | None = None,
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name=repo,
        commit_sha=f"sha_{id_}",
        cve_id=f"CVE-2024-{id_}",
        cwe_id="CWE-89",
        severity="high",
        language="python",
        vulnerable_code=code,
        fixed_code="fixed code",
        description=f"Test vuln {id_}",
        split=split,
    )


# --- load_samples_from_storage ---


@patch("app.data.cleaning.pipeline.get_json")
@patch("app.data.cleaning.pipeline.get_session")
def test_load_samples_from_storage_happy_path(mock_get_session, mock_get_json):
    """Rows from Postgres are loaded and payloads fetched from MinIO."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    mock_row1 = MagicMock()
    mock_row1.id = "s1"
    mock_row1.object_store_key = "key1"

    mock_row2 = MagicMock()
    mock_row2.id = "s2"
    mock_row2.object_store_key = "key2"

    mock_session.query.return_value.all.return_value = [mock_row1, mock_row2]

    mock_get_json.side_effect = [
        {
            "id": "s1",
            "source": "cve_real",
            "repo_name": "org/repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "code1",
            "description": "d1",
        },
        {
            "id": "s2",
            "source": "cve_real",
            "repo_name": "org/repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "code2",
            "description": "d2",
        },
    ]

    samples = load_samples_from_storage()

    assert len(samples) == 2
    assert samples[0].id == "s1"
    assert samples[1].id == "s2"
    mock_session.close.assert_called_once()


@patch("app.data.cleaning.pipeline.get_json")
@patch("app.data.cleaning.pipeline.get_session")
def test_load_samples_from_storage_skips_failed_rows(mock_get_session, mock_get_json):
    """When MinIO get_json fails for a row, it's skipped with a warning."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    mock_row1 = MagicMock()
    mock_row1.id = "s1"
    mock_row1.object_store_key = "key1"

    mock_row2 = MagicMock()
    mock_row2.id = "s2"
    mock_row2.object_store_key = "key2"

    mock_session.query.return_value.all.return_value = [mock_row1, mock_row2]

    # First row succeeds, second fails
    mock_get_json.side_effect = [
        {
            "id": "s1",
            "source": "cve_real",
            "repo_name": "org/repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "code1",
            "description": "d1",
        },
        ValueError("MinIO connection lost"),
    ]

    samples = load_samples_from_storage()

    assert len(samples) == 1
    assert samples[0].id == "s1"
    mock_session.close.assert_called_once()


# --- persist_splits ---


@patch("app.data.cleaning.pipeline.get_session")
def test_persist_splits_writes_split_for_each_sample(mock_get_session):
    """Each sample's split is written back to Postgres and committed."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session

    samples = [
        _sample("s1", split="train"),
        _sample("s2", split="val"),
    ]

    persist_splits(samples)

    # Two calls — one per sample
    assert mock_session.query.return_value.filter.return_value.update.call_count == 2
    mock_session.commit.assert_called_once()
    mock_session.close.assert_called_once()


@patch("app.data.cleaning.pipeline.get_session")
def test_persist_splits_rolls_back_on_error(mock_get_session):
    """When commit fails, rollback is called and the exception propagates."""
    mock_session = MagicMock()
    mock_get_session.return_value = mock_session
    mock_session.commit.side_effect = RuntimeError("DB write failed")

    samples = [_sample("s1", split="train")]

    with pytest.raises(RuntimeError, match="DB write failed"):
        persist_splits(samples)

    mock_session.rollback.assert_called_once()
    mock_session.close.assert_called_once()


# --- run_stage2 ---


@patch("app.data.cleaning.pipeline.load_samples_from_storage")
def test_run_stage2_raises_when_no_samples(mock_load):
    """If no samples exist in storage, a RuntimeError must be raised."""
    mock_load.return_value = []

    with pytest.raises(RuntimeError, match="No samples found"):
        run_stage2()


@patch("app.data.cleaning.pipeline.persist_splits")
@patch("app.data.cleaning.pipeline.check_contamination")
@patch("app.data.cleaning.pipeline.verify_no_leakage")
@patch("app.data.cleaning.pipeline.split_leakage_safe")
@patch("app.data.cleaning.pipeline.dedup_samples")
@patch("app.data.cleaning.pipeline.load_samples_from_storage")
def test_run_stage2_happy_path(
    mock_load, mock_dedup, mock_split, mock_verify, mock_contam, mock_persist
):
    """Full pipeline: load → dedup → split → contamination → persist."""
    mock_load.return_value = [_sample("s1"), _sample("s2")]
    mock_dedup.return_value = (["s1"], [])
    split_result = SplitResult(
        train=[_sample("s1", split="train")],
        val=[],
        test=[_sample("s2", split="test")],
        config=SplitConfig(seed=DEFAULT_SEED),
    )
    mock_split.return_value = split_result
    mock_verify.return_value = True
    mock_contam.return_value = MagicMock(
        contamination_rate=0.0,
        contaminated_samples=set(),
    )

    result = run_stage2(persist=False)

    assert isinstance(result, Stage2Result)
    assert result.samples_loaded == 2
    assert result.samples_after_dedup == 1  # ["s1"]
    assert result.contamination_ok is True
    mock_persist.assert_not_called()  # persist=False


@patch("app.data.cleaning.pipeline.persist_splits")
@patch("app.data.cleaning.pipeline.check_contamination")
@patch("app.data.cleaning.pipeline.verify_no_leakage")
@patch("app.data.cleaning.pipeline.split_leakage_safe")
@patch("app.data.cleaning.pipeline.dedup_samples")
@patch("app.data.cleaning.pipeline.load_samples_from_storage")
def test_run_stage2_persists_when_requested(
    mock_load, mock_dedup, mock_split, mock_verify, mock_contam, mock_persist
):
    """When persist=True (default), persist_splits is called."""
    mock_load.return_value = [_sample("s1")]
    deduped = [_sample("s1", split="train")]
    mock_dedup.return_value = (deduped, [])
    mock_split.return_value = SplitResult(
        train=deduped, val=[], test=[], config=SplitConfig(seed=DEFAULT_SEED)
    )
    mock_verify.return_value = True
    mock_contam.return_value = MagicMock(
        contamination_rate=0.0,
        contaminated_samples=set(),
    )

    run_stage2(persist=True)

    mock_persist.assert_called_once_with(deduped)


@patch("app.data.cleaning.pipeline.persist_splits")
@patch("app.data.cleaning.pipeline.check_contamination")
@patch("app.data.cleaning.pipeline.verify_no_leakage")
@patch("app.data.cleaning.pipeline.split_leakage_safe")
@patch("app.data.cleaning.pipeline.dedup_samples")
@patch("app.data.cleaning.pipeline.load_samples_from_storage")
def test_run_stage2_high_contamination_logs_warning(
    mock_load, mock_dedup, mock_split, mock_verify, mock_contam, mock_persist, caplog
):
    """When contamination is not acceptable, a warning is logged but the
    pipeline still returns a result."""
    mock_load.return_value = [_sample("s1")]
    deduped = [_sample("s1", split="train")]
    mock_dedup.return_value = (deduped, [])
    mock_split.return_value = SplitResult(
        train=deduped, val=[], test=[], config=SplitConfig(seed=DEFAULT_SEED)
    )
    mock_verify.return_value = True
    mock_contam.return_value = MagicMock(
        contamination_rate=0.15,
        contaminated_samples=set(),
    )

    with caplog.at_level(logging.WARNING):
        result = run_stage2(
            persist=False,
            max_contamination=0.05,
        )

    assert result.contamination_ok is False
    assert "Contamination check FAILED" in caplog.text
    mock_persist.assert_not_called()
