"""Celery application with Redis as broker and result backend.

This module creates the shared Celery instance used throughout the
project. Configuration is read from environment variables so it works
both locally (docker-compose) and in CI/CD.

Usage (programmatic)::

    from app.celery_app import celery_app

    # In a task or endpoint:
    result = celery_app.send_task("app.tasks.evaluation.run_evaluation", args=[...])

Usage (CLI)::

    celery -A app.celery_app worker --loglevel=info --concurrency=2

The Celery worker is started alongside the infrastructure services
via ``docker compose -f docker-compose.infra.yml up -d`` and consumed
by the FastAPI serving layer via ``/api/v1/tasks/...`` endpoints.
"""

from __future__ import annotations

import os

from celery import Celery

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_CELERY_DB", "0"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")

# When true (the default), .delay()/apply_async() executes tasks
# synchronously in the current process — no broker or worker required.
# This keeps tests and local dev working without Redis. Real deployments
# (docker-compose.infra.yml) set CELERY_TASK_ALWAYS_EAGER=false for the
# API and worker containers so tasks actually queue through Redis and
# run on the Celery workers instead of blocking the caller in-process.
_eager_env = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "true").strip().lower()
CELERY_TASK_ALWAYS_EAGER = _eager_env not in ("false", "0", "no")

# Build the Redis URL for Celery broker/backend.
def _redis_url() -> str:
    """Construct the Redis connection URL from environment variables."""
    if REDIS_PASSWORD:
        return f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
    return f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"


# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------

celery_app = Celery(
    "vuln_triage_harness",
    broker=_redis_url(),
    backend=_redis_url(),
    include=[
        "app.tasks.evaluation",
        "app.tasks.training",
        "app.tasks.collectors",
    ],
)

# Global task configuration.
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,          # Retry if worker crashes mid-task.
    worker_prefetch_multiplier=1, # One task at a time for memory-heavy work.
    result_expires=3600,          # Results expire after 1 hour.
    task_routes={
        "app.tasks.collectors.*":  {"queue": "collectors"},
        "app.tasks.evaluation.*":  {"queue": "evaluation"},
        "app.tasks.training.*":    {"queue": "training"},
        "app.tasks.health_check":  {"queue": "collectors"},
    },
    # When set, .delay()/apply_async() executes tasks synchronously
    # in the current process — no broker required. Used for tests and
    # local development where Redis may not be available. Controlled by
    # CELERY_TASK_ALWAYS_EAGER (see top of file) — real deployments turn
    # this off so tasks genuinely run on the Celery workers.
    task_always_eager=CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=False,  # Swallow task errors; return PENDING.
)

# ---------------------------------------------------------------------------
# Health check task
# ---------------------------------------------------------------------------

@celery_app.task(bind=True, name="app.tasks.health_check")
def health_check(self) -> dict:
    """Simple health check task — verifies the Celery worker is alive."""
    return {
        "status": "ok",
        "task_id": self.request.id,
    }


# ---------------------------------------------------------------------------
# Module-level import convenience
# ---------------------------------------------------------------------------

__all__ = ["celery_app", "health_check"]
