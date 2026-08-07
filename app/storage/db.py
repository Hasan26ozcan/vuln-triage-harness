"""Postgres access: engine/session factory + the VulnSample table.

Only VulnSample is modeled here for Stage 1. TrainingRun, ModelPrediction,
etc. get their own tables in Stage 5/6 — no point modeling tables for
stages that don't exist yet.
"""

from __future__ import annotations

import os

from sqlalchemy import JSON, Column, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg2://vuln_triage:vuln_triage@localhost:5432/vuln_triage",
    )


class Base(DeclarativeBase):
    pass


class VulnSampleRow(Base):
    """Mirrors app.schemas.vuln.VulnSample. `static_findings` is stored as
    JSON rather than a child table — Stage 1 doesn't need to query into
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


_engine = None
_SessionLocal: sessionmaker | None = None


def get_engine():
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
    """Create tables if they don't exist. Idempotent — safe to call on every
    pipeline run. Migrations (alembic) are future work, not needed at this scale.
    """
    Base.metadata.create_all(get_engine())
