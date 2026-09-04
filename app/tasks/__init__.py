"""Async task definitions powered by Celery + Redis.

This package provides the project's background job workers:

* ``app.tasks.collectors`` — CVE data collection from NVD, CVEfixes, Semgrep
* ``app.tasks.evaluation`` — Four-tier evaluation pipeline (Stage 6)
* ``app.tasks.training`` — SFT/QLoRA/DPO training orchestration (Stage 5)

All tasks are registered with the shared ``celery_app`` instance
(``app.celery_app``) and dispatched through Redis as the message broker.

Usage (enqueue from any module)::

    from app.tasks.evaluation import run_evaluation_task

    result = run_evaluation_task.delay(samples_json, predictions_json)

Usage (via API)::

    POST /api/v1/tasks/evaluation/start
    → returns {"task_id": "...", "status": "PENDING"}
"""

from __future__ import annotations
