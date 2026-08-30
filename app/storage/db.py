"""Postgres access: engine/session factory + data tables.

``VulnSampleRow`` backs the Stage 1 data-collection records. ``TrainingRunRow``
backs Stage 5 training experiments. Each row is a lightweight metadata record
that lives alongside the full payload in MinIO (for samples) or the model
checkpoint (for runs).
"""

from __future__ import annotations

import os
from typing import Any

from sqlalchemy import JSON, Column, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def database_url() -> str:
    """Return the Postgres connection string.

    ``DATABASE_URL`` should always be set explicitly in any real deployment
    (it will contain real credentials and must never be committed).

    The default below is for local dev/CI only. It is built from separate
    ``POSTGRES_*`` env vars (each independently overridable) rather than a
    single hardcoded connection-string literal, so no real credential is
    checked into source control. ``POSTGRES_PASSWORD`` still needs *some*
    default to produce a working local connection out of the box; treat the
    fallback value as a well-known local-dev placeholder, never a real
    secret, and always override it (or set ``DATABASE_URL`` directly) for
    anything beyond a throwaway local database.
    """
    if "DATABASE_URL" in os.environ:
        return os.environ["DATABASE_URL"]
    user = os.environ.get("POSTGRES_USER", "vuln_triage")
    password = os.environ.get("POSTGRES_PASSWORD", "local-dev-only-change-me")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    name = os.environ.get("POSTGRES_DB", "vuln_triage")
    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"


class Base(DeclarativeBase):
    pass


class VulnSampleRow(Base):
    """Mirrors app.schemas.vuln.VulnSample. ``static_findings`` is stored as
    JSON rather than a child table -- Stage 1 doesn't need to query into
    individual findings, only read/write the whole list per sample.
    """

    __tablename__ = "vuln_samples"

    id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    repo_name = Column(String, nullable=False, index=True)  # leakage-safe split key
    commit_sha = Column(String, nullable=True)
    cve_id = Column(String, nullable=True, index=True)
    cwe_id = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False)
    language = Column(String, nullable=False)
    description = Column(String, nullable=False)
    static_findings = Column(JSON, nullable=False, default=list)
    split = Column(String, nullable=True, index=True)
    object_store_key = Column(String, nullable=False)  # where the full code lives in MinIO


class TrainingRunRow(Base):
    """Mirrors app.schemas.training.TrainingRun / TrainingResult.

    One row per training experiment (SFT-full, SFT-QLoRA, LoRA, or DPO).
    The full model checkpoint lives in MinIO (referenced by ``checkpoint_uri``);
    this table stores just the metadata needed to find and evaluate a run.
    """

    __tablename__ = "training_runs"

    id = Column(String, primary_key=True)
    run_name = Column(String, nullable=True)
    method = Column(String, nullable=False, index=True)
    base_model = Column(String, nullable=False)
    hyperparams = Column(JSON, nullable=False, default=dict)
    train_set_size = Column(String, nullable=False)
    train_time_minutes = Column(String, nullable=False)
    peak_vram_gb = Column(String, nullable=False)
    final_train_loss = Column(String, nullable=False)
    final_val_loss = Column(String, nullable=True)
    checkpoint_uri = Column(String, nullable=True)
    status = Column(String, nullable=False, default="pending", index=True)
    created_at = Column(String, nullable=False)


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine() -> Any:
    global _engine
    if _engine is None:
        _engine = create_engine(database_url(), pool_pre_ping=True)
    return _engine


def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal()


def init_db() -> None:
    """Create tables if they don't exist. Idempotent -- safe to call on every
    pipeline run. Migrations (alembic) are future work, not needed at this scale.
    """
    Base.metadata.create_all(get_engine())
