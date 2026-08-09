"""Unit tests for Stage 5 experiment tracking (PostgreSQL persistence).

Covers:
  - generate_run_id: format and uniqueness.
  - persist_training_run: converts TrainingResult → TrainingRunRow fields correctly.
  - load_training_run: round-trips through DB (with mocked session).
  - list_training_runs: filtering by method and status (with mocked session).
  - _row_to_result: field mapping and type conversion (string → float/int).

DB access is mocked via monkeypatch so no PostgreSQL is needed.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from app.schemas.training import TrainingResult
from app.training.experiment import (
    _row_to_result,
    generate_run_id,
    list_training_runs,
    load_training_run,
    persist_training_run,
)

# ---------------------------------------------------------------------------
# generate_run_id
# ---------------------------------------------------------------------------


class TestGenerateRunId:
    def test_basic_format(self):
        run_id = generate_run_id("sft_qlora", None)
        assert run_id.startswith("sft_qlora_")
        # Format: {method}_{timestamp}_{short_uuid}
        # (splitting on _ is unreliable since method and timestamp both use _)
        assert len(run_id) > len("sft_qlra_")

    def test_with_run_name(self):
        run_id = generate_run_id("dpo", "my_run")
        assert run_id.startswith("dpo_my_run_")

    def test_sft_full_method(self):
        run_id = generate_run_id("sft_full", None)
        assert run_id.startswith("sft_full_")

    def test_dpo_method(self):
        run_id = generate_run_id("dpo", None)
        assert run_id.startswith("dpo_")

    def test_uniqueness(self):
        """Two calls should produce different IDs (UUID suffix)."""
        id1 = generate_run_id("sft_qlora", None)
        id2 = generate_run_id("sft_qlora", None)
        assert id1 != id2

    def test_contains_uuid_fragment(self):
        run_id = generate_run_id("sft_qlora", "test")
        # The run name should be in there
        assert "test" in run_id


# ---------------------------------------------------------------------------
# _row_to_result
# ---------------------------------------------------------------------------


class TestRowToResult:
    def test_full_conversion(self):
        from app.storage.db import TrainingRunRow

        row = TrainingRunRow(
            id="run_123",
            run_name="my_run",
            method="sft_qlora",
            base_model="Qwen/Qwen2.5-Coder-7B",
            hyperparams={"lora_r": 8, "use_4bit": True},
            train_set_size="100",
            train_time_minutes="5.5",
            peak_vram_gb="6.2",
            final_train_loss="0.123",
            final_val_loss="0.234",
            checkpoint_uri="s3://bucket/ckpt",
            status="completed",
            created_at=datetime.utcnow().isoformat(),
        )
        result = _row_to_result(row)
        assert result.run_id == "run_123"
        assert result.run_name == "my_run"
        assert result.method == "sft_qlora"
        assert result.base_model == "Qwen/Qwen2.5-Coder-7B"
        assert result.hyperparams == {"lora_r": 8, "use_4bit": True}
        assert result.train_set_size == 100
        assert result.train_time_minutes == 5.5
        assert result.peak_vram_gb == 6.2
        assert result.final_train_loss == 0.123
        assert result.final_val_loss == 0.234
        assert result.checkpoint_uri == "s3://bucket/ckpt"
        assert result.status == "completed"

    def test_null_val_loss(self):
        from app.storage.db import TrainingRunRow

        row = TrainingRunRow(
            id="run_456",
            run_name=None,
            method="dpo",
            base_model="Qwen",
            hyperparams={},
            train_set_size="50",
            train_time_minutes="2.0",
            peak_vram_gb="12.0",
            final_train_loss="0.5",
            final_val_loss=None,
            checkpoint_uri="",
            status="completed",
            created_at=datetime.utcnow().isoformat(),
        )
        result = _row_to_result(row)
        assert result.final_val_loss is None
        assert result.checkpoint_uri == ""

    def test_zero_train_loss(self):
        """final_train_loss of '0.0' in the DB should become 0.0, not None."""
        from app.storage.db import TrainingRunRow

        row = TrainingRunRow(
            id="run_0",
            run_name=None,
            method="sft_qlora",
            base_model="Qwen",
            hyperparams={},
            train_set_size="10",
            train_time_minutes="1.0",
            peak_vram_gb="7.0",
            final_train_loss="0.0",
            final_val_loss=None,
            checkpoint_uri="s3://x",
            status="dry_run",
            created_at=datetime.utcnow().isoformat(),
        )
        result = _row_to_result(row)
        assert result.final_train_loss == 0.0
        assert result.final_val_loss is None


# ---------------------------------------------------------------------------
# persist_training_run (mocked session)
# ---------------------------------------------------------------------------


class TestPersistTrainingRun:
    def _make_result(self) -> TrainingResult:
        return TrainingResult(
            run_id="run_test_123",
            method="sft_qlora",
            base_model="Qwen/Qwen2.5-Coder-7B",
            hyperparams={"lora_r": 8, "use_4bit": True, "learning_rate": 2e-5},
            train_set_size=42,
            train_time_minutes=3.5,
            peak_vram_gb=6.2,
            final_train_loss=0.123,
            final_val_loss=0.234,
            checkpoint_uri="s3://bucket/run_test_123",
            status="completed",
            run_name="test_sweep",
        )

    @pytest.mark.parametrize("persist_fn", ["persist_training_run"])
    def test_persist_with_mocked_session(self, persist_fn):
        """persist_training_run should call session.merge and session.commit."""
        result = self._make_result()

        mock_session = MagicMock()

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            run_id = persist_training_run(result)

        assert run_id == result.run_id
        assert mock_session.merge.called
        assert mock_session.commit.called
        assert mock_session.close.called

        # Verify the row was constructed correctly
        row_arg = mock_session.merge.call_args[0][0]
        assert row_arg.id == result.run_id
        assert row_arg.method == result.method
        assert row_arg.base_model == result.base_model
        assert row_arg.hyperparams == result.hyperparams
        assert row_arg.train_set_size == str(result.train_set_size)
        assert row_arg.peak_vram_gb == str(result.peak_vram_gb)
        assert row_arg.final_train_loss == str(result.final_train_loss)

    def test_persist_with_empty_checkpoint_uri(self):
        """When checkpoint_uri is empty, a placeholder S3 URI should be used."""
        result = TrainingResult(
            run_id="run_empty",
            method="sft_qlora",
            base_model="Qwen",
            hyperparams={},
            train_set_size=0,
            train_time_minutes=0.0,
            peak_vram_gb=0.0,
            final_train_loss=0.0,
            checkpoint_uri="",
            status="dry_run",
        )

        mock_session = MagicMock()
        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            persist_training_run(result)

        row_arg = mock_session.merge.call_args[0][0]
        assert "s3://vuln-triage/checkpoints/stage5/run_empty" in row_arg.checkpoint_uri

    def test_persist_rollback_on_error(self):
        """If an exception occurs, session.rollback should be called."""
        result = self._make_result()
        mock_session = MagicMock()
        mock_session.commit.side_effect = RuntimeError("DB error")

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            with pytest.raises(RuntimeError, match="DB error"):
                persist_training_run(result)

        assert mock_session.rollback.called
        assert mock_session.close.called

    def test_persist_dry_run_result(self):
        """A dry-run result (status='dry_run') should be persistable."""
        result = TrainingResult(
            run_id="run_dry",
            method="sft_qlora",
            base_model="Qwen",
            hyperparams={"lora_r": 8},
            train_set_size=5,
            train_time_minutes=0.0,
            peak_vram_gb=7.0,
            final_train_loss=0.0,
            final_val_loss=None,
            checkpoint_uri="",
            status="dry_run",
        )

        mock_session = MagicMock()
        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            run_id = persist_training_run(result)

        assert run_id == "run_dry"
        assert mock_session.merge.called


# ---------------------------------------------------------------------------
# load_training_run (mocked session)
# ---------------------------------------------------------------------------


class TestLoadTrainingRun:
    @pytest.fixture
    def mock_row(self):
        from app.storage.db import TrainingRunRow

        return TrainingRunRow(
            id="run_load_test",
            run_name="test",
            method="dpo",
            base_model="Qwen",
            hyperparams={"beta": 0.1},
            train_set_size="20",
            train_time_minutes="4.0",
            peak_vram_gb="12.0",
            final_train_loss="0.05",
            final_val_loss="0.06",
            checkpoint_uri="s3://bucket/run_load_test",
            status="completed",
            created_at=datetime.utcnow().isoformat(),
        )

    def test_load_existing_run(self, mock_row):
        mock_session = MagicMock()
        mock_session.get.return_value = mock_row

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            result = load_training_run("run_load_test")

        assert result is not None
        assert result.run_id == "run_load_test"
        assert result.method == "dpo"
        assert result.train_set_size == 20
        mock_session.get.assert_called_once()
        mock_session.close.assert_called_once()

    def test_load_nonexistent_run(self):
        mock_session = MagicMock()
        mock_session.get.return_value = None

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            result = load_training_run("does_not_exist")

        assert result is None
        mock_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# list_training_runs (mocked session)
# ---------------------------------------------------------------------------


class TestListTrainingRuns:
    @pytest.fixture
    def mock_rows(self):
        from app.storage.db import TrainingRunRow

        ts = datetime.utcnow().isoformat()
        return [
            TrainingRunRow(
                id=f"run_{i}",
                run_name=None,
                method="sft_qlora",
                base_model="Qwen",
                hyperparams={"lora_r": r},
                train_set_size="10",
                train_time_minutes="1.0",
                peak_vram_gb="7.0",
                final_train_loss="0.1",
                final_val_loss="0.2",
                checkpoint_uri="s3://x",
                status="completed",
                created_at=ts,
            )
            for i, r in enumerate([8, 16, 32])
        ]

    def test_list_all_runs(self, mock_rows):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = mock_rows
        mock_session.query.return_value = mock_query

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            runs = list_training_runs(limit=50)

        assert len(runs) == 3
        assert all(r.method == "sft_qlora" for r in runs)
        mock_session.close.assert_called_once()

    def test_list_filtered_by_method(self, mock_rows):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = mock_rows[:1]
        mock_session.query.return_value = mock_query

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            runs = list_training_runs(limit=50, method="sft_qlora")

        assert len(runs) == 1
        # Verify filter was applied on method column
        mock_query.filter.assert_called()

    def test_list_filtered_by_status(self, mock_rows):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = mock_rows
        mock_session.query.return_value = mock_query

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            runs = list_training_runs(limit=50, status="completed")

        assert len(runs) == 3
        # filter should have been called at least once (for the status filter)
        assert mock_query.filter.call_count >= 1

    def test_list_empty_session(self):
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value.all.return_value = []
        mock_session.query.return_value = mock_query

        with (
            patch("app.training.experiment.get_session", return_value=mock_session),
            patch("app.training.experiment.init_db"),
        ):
            runs = list_training_runs()

        assert runs == []
