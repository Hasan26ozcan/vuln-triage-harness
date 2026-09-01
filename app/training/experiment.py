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
from datetime import UTC, datetime

from app.schemas.training import TrainingResult
from app.storage.db import TrainingRunRow, get_session, init_db

logger = logging.getLogger(__name__)

# Shared fallback constants — the local JSON file paths used when Postgres is
# unavailable. Extracted as module-level constants to avoid the duplicate-literal
# smell that SonarQube (S1132) flags.
FALLBACK_OUTPUT_DIR = "./output/stage5"
FALLBACK_RESULT_FILENAME = "training_result.json"
FALLBACK_DPO_SUBDIR = "dpo"
FALLBACK_RESULT_PATH = os.path.join(FALLBACK_OUTPUT_DIR, FALLBACK_RESULT_FILENAME)
FALLBACK_DPO_PATH = os.path.join(
    FALLBACK_OUTPUT_DIR, FALLBACK_DPO_SUBDIR, FALLBACK_RESULT_FILENAME
)


def _load_training_result_from_json(path: str) -> TrainingResult | None:
    """Load a ``TrainingResult`` from a local JSON fallback file.

    Returns ``None`` when the file doesn't exist or can't be parsed.
    This helper de-duplicates the fallback-loading logic previously copied
    between :func:`load_training_run` and :func:`list_training_runs`.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse JSON fallback %s: %s", path, exc)
        return None
    from dataclasses import fields

    field_names = {f.name for f in fields(TrainingResult)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return TrainingResult(**filtered)


def _build_training_row(result: TrainingResult) -> TrainingRunRow:
    """Construct a ``TrainingRunRow`` from a ``TrainingResult``."""
    uri = result.checkpoint_uri or f"s3://vuln-triage/checkpoints/stage5/{result.run_id}"
    return TrainingRunRow(
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
        created_at=datetime.now(UTC).isoformat(),
    )


def _close_session_safely(session, run_id: str) -> None:
    """Best-effort session close, swallowing exceptions."""
    if session is None:
        return
    try:
        session.close()
    except Exception as close_exc:  # noqa: BLE001
        logger.debug("Session close failed for run %s: %s", run_id, close_exc)


def _write_json_fallback(result: TrainingResult, output_dir: str) -> None:
    """Write the training result to a local JSON file (Postgres fallback)."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        fallback_path = os.path.join(output_dir, FALLBACK_RESULT_FILENAME)
        with open(fallback_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        logger.info("Wrote local JSON fallback to %s", fallback_path)
    except Exception as fallback_exc:  # noqa: BLE001
        logger.error(
            "Both Postgres persist and local JSON fallback failed for run %s: %s",
            result.run_id,
            fallback_exc,
        )


def persist_training_run(
    result: TrainingResult,
    output_dir: str = FALLBACK_OUTPUT_DIR,
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
    # Wrap DB init + session inside the try so that a missing Postgres
    # (common in CI) falls through to the JSON fallback instead of
    # raising from init_db() / get_session() before the fallback path.
    session = None
    try:
        init_db()  # idempotent — creates tables if missing
        session = get_session()
        session.merge(_build_training_row(result))
        session.commit()
        logger.info(
            "Persisted training run %s (method=%s) to Postgres",
            result.run_id,
            result.method,
        )
        return result.run_id
    except Exception as exc:
        # Fallback: write to a local JSON file so downstream stages still
        # have metadata to read.  This is common in CI where Postgres is
        # not available.
        logger.warning(
            "Postgres persist failed for run %s (%s); writing local JSON fallback",
            result.run_id,
            exc,
        )
        if session is not None:
            try:
                session.rollback()
            except Exception as rollback_exc:  # noqa: BLE001
                logger.debug("Session rollback failed for run %s: %s", result.run_id, rollback_exc)
        _write_json_fallback(result, output_dir)
        # Do not re-raise — the training run itself succeeded.
        return result.run_id
    finally:
        _close_session_safely(session, result.run_id)


def load_training_run(run_id: str) -> TrainingResult | None:
    """Load a single ``TrainingResult`` by run ID.

    Tries Postgres first; if Postgres is unavailable, falls back to
    reading ``{output_dir}/training_result.json`` from disk (the path
    used by the JSON fallback in :func:`persist_training_run`).
    Returns ``None`` if neither source has the run.
    """
    session = None
    try:
        init_db()
        session = get_session()
        row = session.get(TrainingRunRow, run_id)
        if row is None:
            return None
        return _row_to_result(row)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Postgres query failed for run %s; trying JSON fallback: %s", run_id, exc)
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as close_exc:  # noqa: BLE001
                logger.debug("Session close failed for run %s: %s", run_id, close_exc)

    # Fallback: read from the local JSON file written by persist_training_run.
    for fallback_path in (FALLBACK_RESULT_PATH, FALLBACK_DPO_PATH):
        result = _load_training_result_from_json(fallback_path)
        if result is not None and result.run_id == run_id:
            return result

    return None


def list_training_runs(
    limit: int = 50,
    method: str | None = None,
    status: str | None = None,
) -> list[TrainingResult]:
    """List training runs, with optional filtering.

    Tries Postgres first; if Postgres is unavailable, falls back to
    scanning the local JSON files in ``output/stage5/`` and
    ``output/stage5/dpo/`` (the paths used by the JSON fallback in
    :func:`persist_training_run`).

    Parameters
    ----------
    limit:
        Maximum number of runs to return.
    method:
        If set, filter to a single method (sft_full, sft_qlora, lora, dpo).
    status:
        If set, filter to a single status (pending, running, completed, failed).
    """
    session = None
    try:
        init_db()
        session = get_session()
        query = session.query(TrainingRunRow).order_by(TrainingRunRow.created_at.desc())
        if method:
            query = query.filter(TrainingRunRow.method == method)
        if status:
            query = query.filter(TrainingRunRow.status == status)
        rows = query.limit(limit).all()
        return [_row_to_result(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Postgres query failed for list_training_runs; trying JSON fallback: %s",
            exc,
        )
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as close_exc:  # noqa: BLE001
                logger.debug("Session close failed while listing training runs: %s", close_exc)

    # Fallback: scan local JSON files for training results.
    results: list[TrainingResult] = []
    for jpath in (FALLBACK_RESULT_PATH, FALLBACK_DPO_PATH):
        run = _load_training_result_from_json(jpath)
        if run is None:
            continue
        if method and run.method != method:
            continue
        if status and run.status != status:
            continue
        results.append(run)
    return results[:limit]


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
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    if run_name:
        return f"{method}_{run_name}_{ts}_{short_id}"
    return f"{method}_{ts}_{short_id}"
