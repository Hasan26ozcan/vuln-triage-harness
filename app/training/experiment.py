"""Stage 5 — experiment tracking via PostgreSQL.

Persists ``TrainingResult`` records to the ``training_runs`` table in Postgres
so that Stage 6 (evaluation) and Stage 10 (CI/CD gate) can look up which
checkpoint to load and compare loss curves across runs.

This module uses the same ``get_session()`` / ``Base.metadata.create_all``
pattern as Stage 1's ``app.data.collectors.pipeline``. It is import-safe
without a live Postgres connection — queries only hit the DB when called.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict
from datetime import datetime

from app.schemas.training import TrainingResult
from app.storage.db import TrainingRunRow, get_session, init_db

logger = logging.getLogger(__name__)


def persist_training_run(
    result: TrainingResult,
    output_dir: str = "./output/stage5",
) -> str:
    """Write a ``TrainingResult`` to the ``training_runs`` Postgres table.

    Returns the run ID (same as ``result.run_id``).

    If Postgres is unavailable (e.g. connection refused in CI), the run
    metadata is written to a local JSON file at
    ``{output_dir}/training_result.json`` so that Stage 6 / Stage 10 can
    still read it.  A warning is logged in the fallback case; **no
    exception is raised** — the training run itself is the primary success
    signal, persistence is a side-effect.

    The ``checkpoint_uri`` field links to the model adapter in MinIO (written
    by ``CheckpointCallback``). If checkpoints were never uploaded
    (``checkpoint_uri == ""``), a placeholder URI is stored so the row is
    still valid.
    """
    init_db()  # idempotent — creates tables if missing

    session = get_session()
    try:
        uri = result.checkpoint_uri or f"s3://vuln-triage/checkpoints/stage5/{result.run_id}"

        row = TrainingRunRow(
            id=result.run_id,
            run_name=result.run_name,
            method=result.method,
            base_model=result.base_model,
            hyperparams=result.hyperparams,
            train_set_size=str(result.train_set_size),
            train_time_minutes=str(result.train_time_minutes),
            peak_vram_gb=str(result.peak_vram_gb),
            final_train_loss=str(result.final_train_loss),
            final_val_loss=(
                str(result.final_val_loss) if result.final_val_loss is not None else None
            ),
            checkpoint_uri=uri,
            status=result.status,
            created_at=datetime.utcnow().isoformat(),
        )
        session.merge(row)
        session.commit()
        logger.info(
            "Persisted training run %s (method=%s) to Postgres",
            result.run_id,
            result.method,
        )
        return result.run_id
    except Exception as exc:
        session.rollback()
        # Fallback: write to a local JSON file so downstream stages still
        # have metadata to read.  This is common in CI where Postgres is
        # not available.
        logger.warning(
            "Postgres persist failed for run %s (%s); writing local JSON fallback",
            result.run_id,
            exc,
        )
        try:
            os.makedirs(output_dir, exist_ok=True)
            fallback_path = os.path.join(output_dir, "training_result.json")
            with open(fallback_path, "w") as f:
                json.dump(asdict(result), f, indent=2, default=str)
            logger.info("Wrote local JSON fallback to %s", fallback_path)
        except Exception as fallback_exc:  # noqa: BLE001
            logger.error(
                "Both Postgres persist and local JSON fallback failed for run %s: %s",
                result.run_id,
                fallback_exc,
            )
        # Do not re-raise — the training run itself succeeded.
        return result.run_id
    finally:
        session.close()


def load_training_run(run_id: str) -> TrainingResult | None:
    """Load a single ``TrainingResult`` from Postgres by run ID.

    Returns ``None`` if no row with that ID exists.
    """
    init_db()
    session = get_session()
    try:
        row = session.get(TrainingRunRow, run_id)
        if row is None:
            return None
        return _row_to_result(row)
    finally:
        session.close()


def list_training_runs(
    limit: int = 50,
    method: str | None = None,
    status: str | None = None,
) -> list[TrainingResult]:
    """List training runs from Postgres, with optional filtering.

    Parameters
    ----------
    limit:
        Maximum number of rows to return (oldest first).
    method:
        If set, filter to a single method (sft_full, sft_qlora, lora, dpo).
    status:
        If set, filter to a single status (pending, running, completed, failed).
    """
    init_db()
    session = get_session()
    try:
        query = session.query(TrainingRunRow).order_by(TrainingRunRow.created_at.desc())
        if method:
            query = query.filter(TrainingRunRow.method == method)
        if status:
            query = query.filter(TrainingRunRow.status == status)
        rows = query.limit(limit).all()
        return [_row_to_result(r) for r in rows]
    finally:
        session.close()


def _row_to_result(row: TrainingRunRow) -> TrainingResult:
    """Convert a Postgres ``TrainingRunRow`` into a ``TrainingResult`` dataclass."""
    return TrainingResult(
        run_id=row.id,
        method=row.method,
        base_model=row.base_model,
        hyperparams=row.hyperparams,
        train_set_size=int(row.train_set_size),
        train_time_minutes=float(row.train_time_minutes),
        peak_vram_gb=float(row.peak_vram_gb),
        final_train_loss=float(row.final_train_loss),
        final_val_loss=float(row.final_val_loss) if row.final_val_loss else None,
        checkpoint_uri=row.checkpoint_uri or "",
        status=row.status,
        run_name=row.run_name,
    )


def generate_run_id(method: str, run_name: str | None = None) -> str:
    """Generate a deterministic-enough run ID for Stage 5 experiments.

    Format: ``{method}_{timestamp}_{short_uuid}``.
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    if run_name:
        return f"{method}_{run_name}_{ts}_{short_id}"
    return f"{method}_{ts}_{short_id}"
