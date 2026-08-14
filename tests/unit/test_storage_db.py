"""Unit tests for app/storage/db.py — Postgres engine/session factory.

Covers every function and branch:

* ``database_url()`` — env-var override and default fallback.
* ``get_engine()`` — first-call creation and cached return.
* ``get_session()`` — first-call sessionmaker creation and cached return.
* ``init_db()`` — table creation delegation.

``create_engine`` and ``sessionmaker`` are mocked so no real database
connection is required.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

import app.storage.db as db_module  # noqa: E402
from app.storage.db import (  # noqa: E402
    TrainingRunRow,
    VulnSampleRow,
    database_url,
    get_engine,
    get_session,
    init_db,
)

# ---------------------------------------------------------------------------
# database_url
# ---------------------------------------------------------------------------


class TestDatabaseUrl:
    def test_with_env_var(self):
        """DATABASE_URL from the environment takes priority."""
        with patch.dict("os.environ", {"DATABASE_URL": "postgresql://user:pass@host/db"}):
            assert database_url() == "postgresql://user:pass@host/db"

    def test_default_url(self):
        """Without DATABASE_URL, the default postgres connection string is used."""
        env_without_db = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
        with patch.dict("os.environ", env_without_db, clear=True):
            url = database_url()
            assert "postgresql+psycopg2://" in url
            assert "vuln_triage" in url


# ---------------------------------------------------------------------------
# get_engine
# ---------------------------------------------------------------------------


class TestGetEngine:
    def test_creates_engine_on_first_call(self):
        """First call creates the engine via create_engine (lines 81-83)."""
        with (
            patch.object(db_module, "_engine", None),
            patch("app.storage.db.create_engine") as mock_create,
        ):
            result = get_engine()
            mock_create.assert_called_once()
            assert result is not None

    def test_get_engine_passes_database_url(self):
        """get_engine calls database_url() to build the connection string."""
        with (
            patch.object(db_module, "_engine", None),
            patch("app.storage.db.create_engine") as mock_create,
            patch.dict("os.environ", {"DATABASE_URL": "postgresql://test:pass@host/db"}),
        ):
            get_engine()
            url_arg = mock_create.call_args[0][0]
            assert url_arg == "postgresql://test:pass@host/db"

    def test_returns_cached_engine_on_subsequent_calls(self):
        """After the first call, get_engine returns the cached engine
        without calling create_engine again."""
        mock_engine = MagicMock()
        with (
            patch.object(db_module, "_engine", mock_engine),
            patch("app.storage.db.create_engine") as mock_create,
        ):
            result = get_engine()
            assert result is mock_engine
            mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_creates_sessionmaker_on_first_call(self):
        """First call binds sessionmaker to get_engine() (lines 88-90)."""
        mock_engine = MagicMock()
        with (
            patch.object(db_module, "_engine", mock_engine),
            patch.object(db_module, "_SessionLocal", None),
            patch("app.storage.db.sessionmaker") as mock_sessionmaker,
        ):
            mock_factory = MagicMock()
            mock_session = MagicMock()
            mock_factory.return_value = mock_session
            mock_sessionmaker.return_value = mock_factory

            result = get_session()
            mock_sessionmaker.assert_called_once_with(bind=mock_engine)
            assert result is mock_session

    def test_returns_session_from_cached_sessionmaker(self):
        """After sessionmaker is cached, get_session calls it directly."""
        mock_engine = MagicMock()
        mock_session = MagicMock()
        mock_factory = MagicMock(return_value=mock_session)

        with (
            patch.object(db_module, "_engine", mock_engine),
            patch.object(db_module, "_SessionLocal", mock_factory),
            patch("app.storage.db.sessionmaker") as mock_sessionmaker,
        ):
            result = get_session()
            # sessionmaker should NOT be called again (cached)
            mock_sessionmaker.assert_not_called()
            mock_factory.assert_called_once_with()
            assert result is mock_session


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_calls_create_all(self):
        """init_db delegates to Base.metadata.create_all with the engine (line 97)."""
        mock_engine = MagicMock()
        with (
            patch.object(db_module, "_engine", mock_engine),
            patch.object(db_module.Base.metadata, "create_all") as mock_create_all,
        ):
            init_db()
            mock_create_all.assert_called_once_with(mock_engine)

    def test_init_db_calls_get_engine(self):
        """init_db calls get_engine to obtain the engine for create_all."""
        with (
            patch.object(db_module, "_engine", None),
            patch("app.storage.db.create_engine") as mock_create,
            patch.object(db_module.Base.metadata, "create_all") as mock_create_all,
        ):
            init_db()
            mock_create.assert_called_once()
            mock_create_all.assert_called_once()


# ---------------------------------------------------------------------------
# Table / model sanity
# ---------------------------------------------------------------------------


class TestModels:
    def test_vuln_sample_row_has_tablename(self):
        assert VulnSampleRow.__tablename__ == "vuln_samples"

    def test_training_run_row_has_tablename(self):
        assert TrainingRunRow.__tablename__ == "training_runs"
