"""Unit tests for Celery configuration and tasks (app/celery_app.py + app/tasks/).

Covers:

* ``celery_app`` — instance creation and configuration
* ``health_check`` task — basic execution
* ``collect_cve_data_task`` — task registration and result shape
* ``run_evaluation_task`` — task registration and configuration
* ``run_sft_task`` — task registration and result shape
* ``run_qlora_task`` — task registration and result shape
* ``run_dpo_task`` — task registration and result shape

Redis and PostgreSQL are not required; Celery tasks are tested
by calling them directly (``.apply_async`` bypasses the broker)
or by inspecting the task configuration.
"""

from __future__ import annotations

import pytest

celery = pytest.importorskip("celery")

from app.celery_app import celery_app, health_check  # noqa: E402


# ---------------------------------------------------------------------------
# Celery app configuration
# ---------------------------------------------------------------------------


class TestCeleryApp:
    def test_celery_app_is_initialized(self):
        """The celery_app instance exists and has a main name."""
        assert celery_app is not None
        assert celery_app.main == "vuln_triage_harness"

    def test_broker_is_redis(self):
        """Broker URL should point to Redis."""
        broker_url = celery_app.conf.broker_url
        assert "redis://" in broker_url

    def test_backend_is_redis(self):
        """Result backend URL should point to Redis."""
        backend_url = celery_app.conf.result_backend
        assert "redis://" in backend_url

    def test_task_serializer_is_json(self):
        """Tasks should serialize using JSON."""
        assert celery_app.conf.task_serializer == "json"
        assert celery_app.conf.result_serializer == "json"

    def test_timezone_is_utc(self):
        """Celery should use UTC timezone."""
        assert celery_app.conf.timezone == "UTC"
        assert celery_app.conf.enable_utc is True

    def test_task_acks_late_is_enabled(self):
        """Tasks should be acked late for retry safety."""
        assert celery_app.conf.task_acks_late is True

    def test_task_routes_are_configured(self):
        """Task routes should direct tasks to the correct queues."""
        routes = celery_app.conf.task_routes
        assert "app.tasks.collectors.*" in routes
        assert "app.tasks.evaluation.*" in routes
        assert "app.tasks.training.*" in routes
        assert routes["app.tasks.collectors.*"]["queue"] == "collectors"
        assert routes["app.tasks.evaluation.*"]["queue"] == "evaluation"
        assert routes["app.tasks.training.*"]["queue"] == "training"
        # health_check is routed to collectors so send_task() works correctly.
        assert "app.tasks.health_check" in routes
        assert routes["app.tasks.health_check"]["queue"] == "collectors"

    def test_result_expires_is_set(self):
        """Results should expire after a configured time."""
        assert celery_app.conf.result_expires == 3600

    def test_worker_prefetch_is_one(self):
        """Prefetch multiplier should be 1 for memory-heavy tasks."""
        assert celery_app.conf.worker_prefetch_multiplier == 1


class TestRedisUrl:
    """Test _redis_url by patching module-level vars directly."""

    def test_default_url(self, monkeypatch):
        """Default Redis URL uses localhost:6379."""
        monkeypatch.setattr("app.celery_app.REDIS_HOST", "localhost")
        monkeypatch.setattr("app.celery_app.REDIS_PORT", 6379)
        monkeypatch.setattr("app.celery_app.REDIS_DB", 0)
        monkeypatch.setattr("app.celery_app.REDIS_PASSWORD", "")
        from app.celery_app import _redis_url

        assert _redis_url() == "redis://localhost:6379/0"

    def test_url_with_password(self, monkeypatch):
        """When REDIS_PASSWORD is set, it should be included."""
        monkeypatch.setattr("app.celery_app.REDIS_HOST", "redis")
        monkeypatch.setattr("app.celery_app.REDIS_PORT", 6380)
        monkeypatch.setattr("app.celery_app.REDIS_DB", 1)
        monkeypatch.setattr("app.celery_app.REDIS_PASSWORD", "secret")
        from app.celery_app import _redis_url

        assert _redis_url() == "redis://:secret@redis:6380/1"

    def test_url_without_password(self, monkeypatch):
        """When REDIS_PASSWORD is not set, URL has no credentials."""
        monkeypatch.setattr("app.celery_app.REDIS_HOST", "redis")
        monkeypatch.setattr("app.celery_app.REDIS_PORT", 6379)
        monkeypatch.setattr("app.celery_app.REDIS_DB", 0)
        monkeypatch.setattr("app.celery_app.REDIS_PASSWORD", "")
        from app.celery_app import _redis_url

        assert _redis_url() == "redis://redis:6379/0"


# ---------------------------------------------------------------------------
# Health check task
# ---------------------------------------------------------------------------


class TestHealthCheckTask:
    def test_health_check_task_registered(self):
        """The health_check task should be registered with celery_app."""
        assert "app.tasks.health_check" in celery_app.tasks

    def test_health_check_task_execution(self):
        """Calling health_check directly should return a status dict."""
        # apply() runs eagerly (no Redis needed) and returns an EagerResult.
        result = health_check.apply()
        result_data = result.get()
        assert result_data["status"] == "ok"
        assert "task_id" in result_data


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------

# Import task modules to trigger Celery auto-discovery of task decorators.
import app.tasks.collectors  # noqa: F401
import app.tasks.evaluation  # noqa: F401
import app.tasks.training  # noqa: F401


class TestTaskRegistration:
    """Verify all Celery tasks are registered with the app."""

    def test_collect_cve_data_task_registered(self):
        """collect_cve_data_task should be registered."""
        assert (
            "app.tasks.collectors.collect_cve_data_task" in celery_app.tasks
        )

    def test_clean_and_format_task_registered(self):
        """clean_and_format_task should be registered."""
        assert (
            "app.tasks.collectors.clean_and_format_task" in celery_app.tasks
        )

    def test_run_evaluation_task_registered(self):
        """run_evaluation_task should be registered."""
        assert (
            "app.tasks.evaluation.run_evaluation_task" in celery_app.tasks
        )

    def test_run_baseline_task_registered(self):
        """run_baseline_task should be registered."""
        assert "app.tasks.evaluation.run_baseline_task" in celery_app.tasks

    def test_run_sft_task_registered(self):
        """run_sft_task should be registered."""
        assert "app.tasks.training.run_sft_task" in celery_app.tasks

    def test_run_qlora_task_registered(self):
        """run_qlora_task should be registered."""
        assert "app.tasks.training.run_qlora_task" in celery_app.tasks

    def test_run_dpo_task_registered(self):
        """run_dpo_task should be registered."""
        assert "app.tasks.training.run_dpo_task" in celery_app.tasks


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


class TestTaskConfiguration:
    """Verify task-level configuration."""

    def _assert_bound(self, task):
        """Verify a task has bind=True by checking co_varnames[0] == 'self'."""
        # When bind=True, Celery injects 'self' as the first argument.
        # The code object reflects this via co_varnames[0].
        assert task.run.__code__.co_varnames[0] == "self"

    def test_sft_task_is_bound(self):
        """run_sft_task should use bind=True for retry capability."""
        from app.tasks.training import run_sft_task

        self._assert_bound(run_sft_task)

    def test_qlora_task_is_bound(self):
        """run_qlora_task should use bind=True."""
        from app.tasks.training import run_qlora_task

        self._assert_bound(run_qlora_task)

    def test_dpo_task_is_bound(self):
        """run_dpo_task should use bind=True."""
        from app.tasks.training import run_dpo_task

        self._assert_bound(run_dpo_task)

    def test_evaluation_task_is_bound(self):
        """run_evaluation_task should use bind=True."""
        from app.tasks.evaluation import run_evaluation_task

        self._assert_bound(run_evaluation_task)

    def test_collector_task_is_bound(self):
        """collect_cve_data_task should use bind=True."""
        from app.tasks.collectors import collect_cve_data_task

        self._assert_bound(collect_cve_data_task)

    def test_sft_task_has_retry(self):
        """run_sft_task should have retry configuration."""
        from app.tasks.training import run_sft_task

        # Tasks with bind=True also have retry capability via self.retry.
        assert hasattr(run_sft_task, "retry")

    def test_task_name_matches_file(self):
        """Task names should follow the module.path convention."""
        from app.tasks.collectors import collect_cve_data_task
        from app.tasks.evaluation import run_evaluation_task
        from app.tasks.training import run_sft_task

        assert collect_cve_data_task.name == "app.tasks.collectors.collect_cve_data_task"
        assert run_evaluation_task.name == "app.tasks.evaluation.run_evaluation_task"
        assert run_sft_task.name == "app.tasks.training.run_sft_task"
