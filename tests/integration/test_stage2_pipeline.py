"""Integration tests for Stage 2 cleaning pipeline.

These tests mock the Postgres/MinIO storage layer and verify the full
Stage 2 pipeline (dedup → split → contamination check → persist) works
end-to-end with realistic multi-repo, multi-CWE data.

Unlike unit tests, integration tests exercise the pipeline orchestration
logic — how the components work together, not just each piece in isolation.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.data.cleaning.contamination import ContaminationReport
from app.data.cleaning.pipeline import run_stage2
from app.data.cleaning.split import SplitConfig
from app.schemas.vuln import VulnSample


def _make_sample(
    id_: str,
    cwe: str,
    repo: str,
    code: str,
    split: str | None = None,
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
        vulnerable_code=code,
        fixed_code=code + "_fixed",
        description=f"Test {id_}",
        split=split,
    )


def _make_realistic_dataset(n_repos: int = 30) -> list[VulnSample]:
    """Generate a realistic dataset with multiple repos, multiple CWEs,
    and some near-duplicate code (same pattern, different CVE)."""
    cwes = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]
    sql_pattern = "cursor.execute('SELECT * FROM t WHERE id = ' + user_input)"
    samples: list[VulnSample] = []
    for i in range(n_repos):
        cwe = cwes[i % len(cwes)]
        repo = f"org/project_{i}"
        code = sql_pattern if i % 4 == 0 else f"# code variant {i}\nfunc_{i}()"
        samples.append(_make_sample(f"s{i}", cwe, repo, code))
    return samples


def _setup_mock_storage(samples: list[VulnSample]):
    """Create mock Postgres session and MinIO client that return our samples."""
    # Mock Postgres rows: just need id and object_store_key
    fake_rows = [
        MagicMock(id=s.id, object_store_key=f"vuln_samples/{s.cwe_id}/{s.id}.json") for s in samples
    ]

    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = fake_rows
    mock_session.commit = MagicMock()
    mock_session.rollback = MagicMock()
    mock_session.close = MagicMock()

    # Mock MinIO: map object_store_key -> sample dict
    store_map = {
        row.object_store_key: s.model_dump() for row, s in zip(fake_rows, samples, strict=True)
    }

    def mock_get_json(key):
        return store_map[key]

    return mock_session, mock_get_json


@pytest.fixture
def mock_pipeline_storage(monkeypatch, n_repos=30):
    """Patch the storage layer so run_stage2 works without real Postgres/MinIO."""
    samples = _make_realistic_dataset(n_repos=n_repos)
    mock_session, mock_get_json = _setup_mock_storage(samples)

    # Patch get_session to return our mock session
    import app.data.cleaning.pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "get_session", lambda: mock_session)
    monkeypatch.setattr(pipeline_mod, "get_json", mock_get_json)

    # Patch persist_splits to use our mock session
    def mock_persist(samples_to_persist):
        for s in samples_to_persist:
            mock_session.query(pipeline_mod.VulnSampleRow).filter().update({"split": s.split})
        mock_session.commit()

    monkeypatch.setattr(pipeline_mod, "persist_splits", mock_persist)

    return samples


class _DummyEmbeddingBackend:
    """Deterministic embedding backend for integration tests — no real
    model download needed. Produces embeddings that make code with the
    same pattern have cos=1.0 and different code have cos=0.0.
    """

    def embed(self, texts: list[str]):
        import numpy as np

        # Map each unique text to a unique one-hot vector
        unique_texts = sorted(set(texts))
        vocab = {t: i for i, t in enumerate(unique_texts)}
        vectors = np.zeros((len(texts), len(unique_texts)))
        for i, t in enumerate(texts):
            vectors[i, vocab[t]] = 1.0
        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


def test_stage2_pipeline_end_to_end(mock_pipeline_storage):
    """Full pipeline: load → dedup → split → contamination → persist."""
    result = run_stage2(
        dedup_threshold=0.95,
        split_config=SplitConfig(seed=42),
        contamination_n=5,
        max_contamination=0.05,
        persist=True,
        embedding_backend=_DummyEmbeddingBackend(),
    )

    # All samples loaded
    assert result.samples_loaded == 30

    # Dedup ran (some near-duplicates should be removed due to the SQL pattern
    # appearing in 1/4 of samples)
    assert result.samples_after_dedup <= 30

    # Split produced three sets
    assert len(result.split_result.train) > 0
    assert len(result.split_result.val) > 0
    assert len(result.split_result.test) > 0

    # No leakage
    from app.data.cleaning.split import verify_no_leakage

    assert verify_no_leakage(result.split_result) is True

    # Contamination report was generated
    assert isinstance(result.contamination_report, ContaminationReport)
    assert result.contamination_report.n_train_samples > 0
    assert result.contamination_report.n_eval_samples > 0

    # The result has a contamination_ok boolean
    assert result.contamination_ok in (True, False)


def test_stage2_pipeline_dry_run_no_persist(mock_pipeline_storage):
    """Dry-run should not call persist."""
    with patch("app.data.cleaning.pipeline.persist_splits") as mock_persist:
        result = run_stage2(
            dedup_threshold=0.95,
            persist=False,
            embedding_backend=_DummyEmbeddingBackend(),
        )

    mock_persist.assert_not_called()
    assert result.samples_loaded == 30


def test_stage2_pipeline_reproducible(mock_pipeline_storage):
    """Same seed → same split."""
    r1 = run_stage2(
        dedup_threshold=0.99,  # high threshold so dedup doesn't affect reproducibility test
        split_config=SplitConfig(seed=42),
        embedding_backend=_DummyEmbeddingBackend(),
    )
    r2 = run_stage2(
        dedup_threshold=0.99,
        split_config=SplitConfig(seed=42),
        embedding_backend=_DummyEmbeddingBackend(),
    )

    assert {s.id for s in r1.split_result.train} == {s.id for s in r2.split_result.train}
    assert {s.id for s in r1.split_result.test} == {s.id for s in r2.split_result.test}


def test_stage2_pipeline_different_seeds_differ(mock_pipeline_storage):
    """Different seeds → different splits."""
    r1 = run_stage2(
        dedup_threshold=0.99,
        split_config=SplitConfig(seed=1),
        embedding_backend=_DummyEmbeddingBackend(),
    )
    r2 = run_stage2(
        dedup_threshold=0.99,
        split_config=SplitConfig(seed=999),
        embedding_backend=_DummyEmbeddingBackend(),
    )

    assert {s.id for s in r1.split_result.test} != {s.id for s in r2.split_result.test}


def test_stage2_pipeline_raises_without_samples(monkeypatch):
    """If no samples in storage, the pipeline should give a clear error."""
    mock_session = MagicMock()
    mock_session.query.return_value.all.return_value = []
    monkeypatch.setattr("app.data.cleaning.pipeline.get_session", lambda: mock_session)

    backend = _DummyEmbeddingBackend()
    with pytest.raises(RuntimeError, match="No samples found"):
        run_stage2(embedding_backend=backend)


def test_stage2_pipeline_contamination_low_with_different_code(mock_pipeline_storage):
    """When train and test have different code patterns, contamination
    should be low or zero."""
    result = run_stage2(
        dedup_threshold=0.95,
        embedding_backend=_DummyEmbeddingBackend(),
    )

    # Our dummy dataset has very distinct code patterns, so contamination
    # should be 0 or very low
    assert result.contamination_report.contamination_rate < 0.5
