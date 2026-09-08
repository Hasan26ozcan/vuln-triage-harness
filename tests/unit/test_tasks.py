"""Unit tests for Celery task bodies in app/tasks/.

Tests exercise the full task logic by calling ``.apply()`` directly
with eager Celery mode. External services (Redis, MinIO, Postgres)
and heavy ML dependencies (EvaluationRunner) are mocked to avoid
network calls.
"""

from __future__ import annotations

import importlib
import json

# ---------------------------------------------------------------------------
# Disable Celery task tracking to avoid Redis calls for update_state().
# Must be set BEFORE importing celery_app (which reads env vars).
# ---------------------------------------------------------------------------
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
os.environ["CELERY_TASK_EAGER_PROPAGATES"] = "true"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6379"
os.environ["REDIS_DB"] = "0"
os.environ["REDIS_PASSWORD"] = ""

import app.celery_app  # noqa: E401

# Ensure eager mode is active in the shared Celery config
app.celery_app.celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)


@pytest.fixture(autouse=True)
def disable_task_tracking():
    """Disable task tracking so self.update_state() doesn't hit Redis.
    Also ensure eager execution is active so tasks run in-process without Redis."""
    app.celery_app.celery_app.conf.update(task_always_eager=True, task_eager_propagates=True)
    app.celery_app.celery_app.conf.task_track_started = False
    # Patch Task.update_state so explicit self.update_state() calls in tasks
    # don't try to connect to Redis for storing progress state.
    patcher = patch("celery.app.task.Task.update_state", lambda *a, **kw: None)
    patcher.start()
    yield
    patcher.stop()
    app.celery_app.celery_app.conf.task_track_started = True

# ---------------------------------------------------------------------------
# Tasks are auto-registered when the modules are imported.
# ---------------------------------------------------------------------------
import app.tasks.collectors  # noqa: F401, E401, E402
import app.tasks.evaluation  # noqa: F401, E401, E402
import app.tasks.training  # noqa: F401, E401, E402
from app.tasks.collectors import (  # noqa: E402
    clean_and_format_task,
    collect_cve_data_task,
)
from app.tasks.evaluation import (  # noqa: E402
    run_baseline_task,
    run_evaluation_task,
)
from app.tasks.training import (  # noqa: E402
    _store_checkpoint_metadata,
    run_dpo_task,
    run_qlora_task,
    run_sft_task,
)

# ============================================================================
# collect_cve_data_task — covers lines 52-161
# ============================================================================


class TestCollectCVEDataTask:
    def _mock_module_with(self, **attrs):
        """Return a module-like object where hasattr returns True only for
        explicitly provided attributes. Missing attributes raise AttributeError."""
        class Module:
            def __getattr__(self, name):
                if name in attrs:
                    val = attrs[name]
                    return lambda *a, **kw: val
                raise AttributeError(f"module has no attribute '{name}'")
        return Module()

    def _mock_import_module(self, **attrs):
        """Return a function that intercepts collector module imports.
        All other imports fall through to the real importlib.import_module."""
        real_import = importlib.import_module
        def _import(name, **kwargs):
            if "app.data.collectors" in name:
                return self._mock_module_with(**attrs)
            return real_import(name, **kwargs)
        return _import

    def test_collect_with_all_sources(self):
        """Run collect_cve_data_task with all sources — covers collection code."""
        _import = self._mock_import_module(fetch_cves=["a"], load_cvefixes=["b"], load_rules=["c"])
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(
                args=[["nvd", "cvefixes", "semgrep"], None]
            )
        data = result.get()
        assert data["status"] == "completed"
        assert data["collected"] == 3  # len(["a"])=1 + len(["b"])=1 + len(["c"])=1
        assert data["sources"] == ["nvd", "cvefixes", "semgrep"]

    def test_collect_with_subset(self):
        """Run with only NVD source."""
        _import = self._mock_import_module(fetch_cves=["cve-1"])
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[["nvd"], None])
        data = result.get()
        assert data["status"] == "completed"
        assert data["collected"] == 1
        assert data["sources"] == ["nvd"]

    def test_collect_with_cwe_filter(self):
        """Run with a CWE filter."""
        _import = self._mock_import_module(fetch_cves=["cve-1"], load_cvefixes=["f"])
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[["nvd", "cvefixes"], ["CWE-89"]])
        data = result.get()
        assert data["status"] == "completed"

    def test_collect_with_empty_sources(self):
        """Run with empty sources list."""
        with patch("app.storage.object_store.put_json"):
            result = collect_cve_data_task.apply(args=[[], None])
        data = result.get()
        assert data["status"] == "completed"
        assert data["collected"] == 0

    def test_collect_with_none_sources(self):
        """When sources is None, defaults to all three sources — covers line 53."""
        _import = self._mock_import_module(fetch_cves=[], load_cvefixes=[], load_rules=[])
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[None, None])
        data = result.get()
        assert data["status"] == "completed"
        assert data["sources"] == ["nvd", "cvefixes", "semgrep"]

    def test_collect_nvd_with_fetch_cves(self):
        """When NVD module has fetch_cves, line 74 is covered."""
        _import = self._mock_import_module(fetch_cves=["cve-1", "cve-2"])
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[["nvd"], None])
        data = result.get()
        assert data["status"] == "completed"
        assert data["collected"] == 2

    def test_collect_cvefixes_with_load_cvefixes(self):
        """When cvefixes module has load_cvefixes, lines 96-104 are covered."""
        _import = self._mock_import_module(load_cvefixes=["fix-1", "fix-2", "fix-3"])
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[["cvefixes"], None])
        data = result.get()
        assert data["status"] == "completed"
        assert data["collected"] == 3

    def test_collect_cvefixes_load_fails(self):
        """When load_cvefixes raises, outer except triggers self.retry() —
        covers the outer except handler (retry) at lines 106-112."""
        import celery

        class BadModule:
            def load_cvefixes(self, *a):
                raise Exception("cvefixes crashed")
        real_import = importlib.import_module
        def _import(name, **kwargs):
            if "app.data.collectors" in name:
                return BadModule()
            return real_import(name, **kwargs)
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            with pytest.raises(celery.exceptions.Retry):
                collect_cve_data_task.apply(args=[["cvefixes"], None])

    def test_collect_semgrep_with_load_rules(self):
        """When semgrep module has load_rules, lines 114-117 are covered."""
        _import = self._mock_import_module(load_rules=["rule-1", "rule-2"])
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[["semgrep"], None])
        data = result.get()
        assert data["status"] == "completed"
        assert data["collected"] == 2

    def test_collect_semgrep_load_rules_fails(self):
        """When load_rules raises, outer except triggers self.retry() —
        covers the outer except handler (retry)."""
        import celery

        class BadModule:
            def load_rules(self, *a):
                raise Exception("semgrep crashed")
        real_import = importlib.import_module
        def _import(name, **kwargs):
            if "app.data.collectors" in name:
                return BadModule()
            return real_import(name, **kwargs)
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            with pytest.raises(celery.exceptions.Retry):
                collect_cve_data_task.apply(args=[["semgrep"], None])

    def test_collect_summary_put_json_fails(self):
        """When put_json raises during summary storage, the local except
        catches it; task still completes. The summary block is outside
        the outer try-except so storage failure is non-fatal."""
        _import = self._mock_import_module(fetch_cves=["cve-1"])
        with patch("app.storage.object_store.put_json", side_effect=Exception("storage down")), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[["nvd"], None])
        data = result.get()
        assert data["status"] == "completed"

    def test_collect_collection_retry(self):
        """When import_module raises during collection, the outer except
        handler calls self.retry() — covers lines 106-112 (retry handler)."""
        import celery

        def _import(name, **kwargs):
            raise ImportError("no module")
        with patch("importlib.import_module", side_effect=_import):
            with pytest.raises(celery.exceptions.Retry):
                collect_cve_data_task.apply(args=[["nvd"], None])

    def test_collect_nvd_retry(self):
        """When NVD import fails during collection, the outer except
        calls self.retry() — the task retries instead of falling back
        to a default count. This covers the retry handler (lines 106-112)."""
        import celery

        def _import(name, **kwargs):
            raise ImportError("no nvd")
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            with pytest.raises(celery.exceptions.Retry):
                collect_cve_data_task.apply(args=[["nvd"], None])

    def test_collect_fallback_defaults(self):
        """When modules exist but don't have fetch_cves/load_cvefixes/load_rules,
        the else branches (lines 74, 88, 101) use default counts."""
        class EmptyModule:
            def __getattr__(self, name):
                raise AttributeError(f"module has no attribute '{name}'")
        real_import = importlib.import_module
        def _import(name, **kwargs):
            if "app.data.collectors" in name:
                return EmptyModule()
            return real_import(name, **kwargs)
        with patch("app.storage.object_store.put_json"), \
             patch("importlib.import_module", side_effect=_import):
            result = collect_cve_data_task.apply(args=[["nvd", "cvefixes", "semgrep"], None])
        data = result.get()
        assert data["status"] == "completed"
        assert data["collected"] == 42 + 38 + 104  # default fallback counts


# ============================================================================
# clean_and_format_task — covers lines 188-220
# ============================================================================


class TestCleanAndFormatTask:
    def test_clean_and_format(self):
        """Run clean_and_format_task — covers lines 188-220."""
        with patch("app.storage.object_store.put_json"):
            result = clean_and_format_task.apply(args=["raw/data", "out/data"])
        data = result.get()
        assert data["status"] == "completed"
        assert data["clean_records"] == 100
        assert data["raw_records"] == 120
        assert data["output_key"] == "out/data"

    def test_clean_and_format_put_json_fails(self):
        """When put_json raises during storage, lines 218-220 are covered
        and the retry handler is triggered."""
        import celery

        with patch("app.storage.object_store.put_json", side_effect=Exception("storage down")):
            with pytest.raises(celery.exceptions.Retry):
                clean_and_format_task.apply(args=["raw/data", "out/data"])


# ============================================================================
# run_evaluation_task — covers lines 107-159
# ============================================================================


class TestRunEvaluationTask:
    def setup_method(self):
        """Mock EvaluationRunner and tier evaluators to avoid ML deps."""
        mock_report = MagicMock()
        mock_report.model_dump.return_value = {
            "metrics": {
                "model_cwe_macro_f1": 0.85,
                "num_samples": 1,
            },
            "num_samples": 1,
            "num_predictions": 1,
        }
        self._mock_runner = MagicMock()
        self._mock_runner.run.return_value = mock_report

    def test_run_evaluation_with_mock_sandbox(self):
        """Run the evaluation task with sandbox_mode='mock'.

        Covers the full pipeline including all tier branches
        (lines 107-159).
        """
        sample = {
            "id": "eval-sample-001",
            "source": "synthetic_injected",
            "repo_name": "test-repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "cursor.execute('SELECT * FROM users')",
            "description": "SQL injection",
            "static_findings": [],
        }
        prediction = {
            "sample_id": "eval-sample-001",
            "run_id": "run-001",
            "predicted_cwe": "CWE-89",
            "predicted_severity": "high",
            "suggested_patch_diff": "",
            "rationale": "SQL injection via string concat",
        }
        with patch("app.evaluation.runner.EvaluationRunner") as MockRunner:
            MockRunner.return_value = self._mock_runner
            result = run_evaluation_task.apply(
                args=[json.dumps([sample]), json.dumps([prediction])],
                kwargs={"sandbox_mode": "mock", "skip_tier3": True, "skip_tier4": True},
            )
        data = result.get()
        assert data["status"] == "completed"

    def test_run_evaluation_with_docker_sandbox(self):
        """Run with sandbox_mode='docker' — covers docker branch in tier3."""
        sample = {
            "id": "eval-sample-002",
            "source": "synthetic_injected",
            "repo_name": "test-repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "x = 1",
            "description": "test",
            "static_findings": [],
        }
        prediction = {
            "sample_id": "eval-sample-002",
            "run_id": "run-002",
            "predicted_cwe": "CWE-89",
            "predicted_severity": "high",
            "suggested_patch_diff": "",
            "rationale": "test",
        }
        with patch("app.evaluation.runner.EvaluationRunner") as MockRunner:
            MockRunner.return_value = self._mock_runner
            result = run_evaluation_task.apply(
                args=[json.dumps([sample]), json.dumps([prediction])],
                kwargs={"sandbox_mode": "docker", "skip_tier3": True, "skip_tier4": True},
            )
        data = result.get()
        assert data["status"] == "completed"

    def test_run_evaluation_skip_tier3_and_tier4(self):
        """Run with both tiers skipped."""
        sample = {
            "id": "eval-sample-003",
            "source": "synthetic_injected",
            "repo_name": "test-repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "x = 1",
            "description": "test",
            "static_findings": [],
        }
        prediction = {
            "sample_id": "eval-sample-003",
            "run_id": "run-003",
            "predicted_cwe": "CWE-89",
            "predicted_severity": "high",
            "suggested_patch_diff": "",
            "rationale": "test",
        }
        with patch("app.evaluation.runner.EvaluationRunner") as MockRunner:
            MockRunner.return_value = self._mock_runner
            result = run_evaluation_task.apply(
                args=[json.dumps([sample]), json.dumps([prediction])],
                kwargs={"sandbox_mode": "mock", "skip_tier3": True, "skip_tier4": True},
            )
        data = result.get()
        assert data["status"] == "completed"

    def test_run_evaluation_with_docker_sandbox_real_tier3(self):
        """Run with sandbox_mode='docker' and skip_tier3=False — covers
        the docker branch (lines 126-128) and local/mock branches (129-134)."""
        sample = {
            "id": "eval-sample-docker",
            "source": "synthetic_injected",
            "repo_name": "test-repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "x = 1",
            "description": "test",
            "static_findings": [],
        }
        prediction = {
            "sample_id": "eval-sample-docker",
            "run_id": "run-docker",
            "predicted_cwe": "CWE-89",
            "predicted_severity": "high",
            "suggested_patch_diff": "",
            "rationale": "test",
        }
        mock_exec = MagicMock()
        with patch("app.evaluation.runner.EvaluationRunner") as MockRunner:
            MockRunner.return_value = self._mock_runner
            with patch("app.evaluation.tier3_exec.ExecEvaluator", return_value=mock_exec), \
                 patch("app.evaluation.tier3_exec.DockerSandboxRunner"), \
                 patch("app.evaluation.tier3_exec.LocalSandboxRunner"), \
                 patch("app.evaluation.tier3_exec.MockSandboxRunner"):
                result = run_evaluation_task.apply(
                    args=[json.dumps([sample]), json.dumps([prediction])],
                    kwargs={"sandbox_mode": "docker", "skip_tier3": False, "skip_tier4": True},
                )
        data = result.get()
        assert data["status"] == "completed"
        mock_exec.evaluate_all.assert_called_once()

    def test_run_evaluation_with_local_sandbox_real_tier3(self):
        """Run with sandbox_mode='local' and skip_tier3=False — covers
        the local branch (lines 129-131)."""
        sample = {
            "id": "eval-sample-local",
            "source": "synthetic_injected",
            "repo_name": "test-repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "x = 1",
            "description": "test",
            "static_findings": [],
        }
        prediction = {
            "sample_id": "eval-sample-local",
            "run_id": "run-local",
            "predicted_cwe": "CWE-89",
            "predicted_severity": "high",
            "suggested_patch_diff": "",
            "rationale": "test",
        }
        mock_exec = MagicMock()
        with patch("app.evaluation.runner.EvaluationRunner") as MockRunner:
            MockRunner.return_value = self._mock_runner
            with patch("app.evaluation.tier3_exec.ExecEvaluator", return_value=mock_exec), \
                 patch("app.evaluation.tier3_exec.DockerSandboxRunner"), \
                 patch("app.evaluation.tier3_exec.LocalSandboxRunner"), \
                 patch("app.evaluation.tier3_exec.MockSandboxRunner"):
                result = run_evaluation_task.apply(
                    args=[json.dumps([sample]), json.dumps([prediction])],
                    kwargs={"sandbox_mode": "local", "skip_tier3": False, "skip_tier4": True},
                )
        data = result.get()
        assert data["status"] == "completed"
        mock_exec.evaluate_all.assert_called_once()

    def test_run_evaluation_with_mock_sandbox_real_tier3(self):
        """Run with sandbox_mode='mock' and skip_tier3=False — covers
        the else/mock branch (lines 133-134)."""
        sample = {
            "id": "eval-sample-mock",
            "source": "synthetic_injected",
            "repo_name": "test-repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "x = 1",
            "description": "test",
            "static_findings": [],
        }
        prediction = {
            "sample_id": "eval-sample-mock",
            "run_id": "run-mock",
            "predicted_cwe": "CWE-89",
            "predicted_severity": "high",
            "suggested_patch_diff": "",
            "rationale": "test",
        }
        mock_exec = MagicMock()
        with patch("app.evaluation.runner.EvaluationRunner") as MockRunner:
            MockRunner.return_value = self._mock_runner
            with patch("app.evaluation.tier3_exec.ExecEvaluator", return_value=mock_exec), \
                 patch("app.evaluation.tier3_exec.DockerSandboxRunner"), \
                 patch("app.evaluation.tier3_exec.LocalSandboxRunner"), \
                 patch("app.evaluation.tier3_exec.MockSandboxRunner"):
                result = run_evaluation_task.apply(
                    args=[json.dumps([sample]), json.dumps([prediction])],
                    kwargs={"sandbox_mode": "mock", "skip_tier3": False, "skip_tier4": True},
                )
        data = result.get()
        assert data["status"] == "completed"
        mock_exec.evaluate_all.assert_called_once()

    def test_run_evaluation_with_existing_task_id(self):
        """Verify result has expected fields."""
        sample = {
            "id": "eval-sample-004",
            "source": "synthetic_injected",
            "repo_name": "test-repo",
            "cwe_id": "CWE-89",
            "severity": "high",
            "language": "python",
            "vulnerable_code": "x = 1",
            "description": "test",
            "static_findings": [],
        }
        prediction = {
            "sample_id": "eval-sample-004",
            "run_id": "run-004",
            "predicted_cwe": "CWE-89",
            "predicted_severity": "high",
            "suggested_patch_diff": "",
            "rationale": "test",
        }
        with patch("app.evaluation.runner.EvaluationRunner") as MockRunner:
            MockRunner.return_value = self._mock_runner
            result = run_evaluation_task.apply(
                args=[json.dumps([sample]), json.dumps([prediction])],
                kwargs={"sandbox_mode": "mock", "skip_tier3": True, "skip_tier4": True},
            )
        data = result.get()
        assert "task_id" in data

    def test_run_evaluation_import_error(self):
        """When json.loads fails, the retry handler is invoked.

        With task_eager_propagates=True, celery.exceptions.Retry propagates
        from .apply(). Covers the retry error path at lines 165-167."""
        import celery

        with patch("app.tasks.evaluation.json.loads", side_effect=Exception("parse error")):
            with pytest.raises(celery.exceptions.Retry):
                run_evaluation_task.apply(
                    args=["bad_json", "bad_json"],
                    kwargs={"sandbox_mode": "mock", "skip_tier3": True, "skip_tier4": True},
                )


# ============================================================================
# run_baseline_task — covers lines 193-236
# ============================================================================


class TestRunBaselineTask:
    def test_run_baseline_task(self):
        """Run the baseline task with mock=True — covers lines 193-236."""
        from dataclasses import dataclass, field

        @dataclass
        class MockMetrics:
            model_cwe_macro_f1: float = 0.85

        @dataclass
        class MockResult:
            metrics: MockMetrics = field(default_factory=MockMetrics)
            predictions: list = ()

        with patch("app.evaluation.baseline.run_baseline") as mock_run:
            mock_run.return_value = MockResult()
            result = run_baseline_task.apply(
                args=["tests/integration/test_stage4_baseline.py"],
                kwargs={"strategy": "zero_shot", "mock": True},
            )
        data = result.get()
        assert data["status"] == "completed"
        assert "metrics" in data
        assert "task_id" in data

    def test_run_baseline_task_import_error(self):
        """When run_baseline raises, the retry handler at lines 234-236
        is exercised."""
        import celery

        with patch("app.evaluation.baseline.run_baseline", side_effect=Exception("baseline error")):
            with pytest.raises(celery.exceptions.Retry):
                run_baseline_task.apply(
                    args=["nonexistent/path"],
                    kwargs={"strategy": "zero_shot", "mock": True},
                )

    def test_run_baseline_task_non_mock_raises(self):
        """When mock=False, the ValueError at line 218 is raised
        and caught by the retry handler — covers line 218."""
        import celery

        with patch("app.evaluation.baseline") as mock_baseline:
            mock_baseline.BaselineConfig = object
            mock_baseline.run_baseline = lambda *a, **kw: None
            with pytest.raises(celery.exceptions.Retry):
                run_baseline_task.apply(
                    args=["tests/integration/test_stage4_baseline.py"],
                    kwargs={"strategy": "zero_shot", "mock": False},
                )


# ============================================================================
# _store_checkpoint_metadata — covers lines 44-76
# ============================================================================


class TestStoreCheckpointMetadata:
    def test_store_checkpoint_metadata(self):
        """Call _store_checkpoint_metadata directly to cover lines 44-76.

        The function catches all exceptions internally, so even if the
        DB or MinIO calls fail, no exception propagates."""
        result = {
            "run_id": "test-run-001",
            "method": "sft",
            "base_model": "Qwen2.5-Coder-7B-Instruct",
            "train_set_size": 100,
            "train_time_minutes": 5.0,
            "peak_vram_gb": 6.51,
            "final_train_loss": 1.038,
            "final_val_loss": 1.088,
            "checkpoint_uri": "minio://checkpoints/test",
            "hyperparams": {"epochs": 3},
            "status": "completed",
        }
        # Should not raise even if storage fails
        _store_checkpoint_metadata(
            run_id="test-run-001",
            method="sft",
            result=result,
            checkpoint_key="checkpoints/test",
        )

    def test_store_checkpoint_metadata_postgres(self):
        """When Postgres storage succeeds, lines 53-74 are covered.
        The function catches all exceptions internally, so even if
        the DB calls fail, no exception propagates."""
        from unittest.mock import MagicMock, patch

        result = {
            "run_id": "test-run-002",
            "method": "sft",
            "base_model": "Qwen2.5-Coder-7B-Instruct",
            "train_set_size": 50,
            "train_time_minutes": 3.0,
            "peak_vram_gb": 4.0,
            "final_train_loss": 1.5,
            "final_val_loss": 1.6,
            "checkpoint_uri": "minio://checkpoints/test2",
            "hyperparams": {"epochs": 2},
            "status": "completed",
        }
        mock_session = MagicMock()
        with patch("app.storage.db.init_db"), \
             patch("app.storage.db.get_session", return_value=mock_session), \
             patch("app.storage.object_store.put_json"):
            _store_checkpoint_metadata(
                run_id="test-run-002",
                method="sft",
                result=result,
                checkpoint_key="checkpoints/test2",
            )
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    def test_store_checkpoint_metadata_postgres_fails(self):
        """When Postgres storage fails, the function logs a warning
        without raising — covers the except handler at lines 75-76."""
        from unittest.mock import patch

        result = {
            "run_id": "test-run-003",
            "method": "sft",
            "base_model": "Qwen2.5-Coder-7B-Instruct",
            "train_set_size": 10,
            "train_time_minutes": 1.0,
            "peak_vram_gb": 1.0,
            "final_train_loss": 2.0,
            "final_val_loss": 2.1,
            "checkpoint_uri": "minio://checkpoints/test3",
            "hyperparams": {"epochs": 1},
            "status": "completed",
        }
        with patch("app.storage.db.init_db", side_effect=Exception("db down")), \
             patch("app.storage.object_store.put_json"):
            # Should not raise
            _store_checkpoint_metadata(
                run_id="test-run-003",
                method="sft",
                result=result,
                checkpoint_key="checkpoints/test3",
            )


# ============================================================================
# run_sft_task — covers lines 114-156
# ============================================================================


class TestRunSFTTask:
    def test_run_sft_task(self):
        """Run the SFT task — covers lines 114-156."""
        config_json = (  # noqa: E501
            '{"epochs": 3, "final_train_loss": 1.038, '
            '"peak_vram_gb": 6.51, "base_model": "Qwen2.5-Coder-7B-Instruct"}'
        )
        result = run_sft_task.apply(
            args=["data/train.jsonl", config_json, "checkpoints/sft-test"],
        )
        data = result.get()
        assert data["status"] == "completed"
        assert data["method"] == "sft"
        assert "task_id" in data

    def test_run_sft_task_custom_config(self):
        """Run with custom hyperparameters."""
        config_json = (  # noqa: E501
            '{"epochs": 5, "final_train_loss": 0.95, '
            '"peak_vram_gb": 8.0, "base_model": "Qwen2.5-Coder-7B-Instruct"}'
        )
        result = run_sft_task.apply(
            args=["data/train.jsonl", config_json, "checkpoints/sft-custom"],
        )
        data = result.get()
        assert data["status"] == "completed"
        assert data["peak_vram_gb"] == 8.0

    def test_run_sft_task_retry_on_error(self):
        """When config_json is invalid, the retry handler at
        lines 158-160 is exercised."""
        import celery

        with pytest.raises(celery.exceptions.Retry):
            run_sft_task.apply(
                args=["data/train.jsonl", "not valid json", "checkpoints/sft-test"],
            )


# ============================================================================
# run_qlora_task — covers lines 181-226
# ============================================================================


class TestRunQLoRATask:
    def test_run_qlora_task(self):
        """Run the QLoRA task — covers lines 181-226."""
        config_json = (  # noqa: E501
            '{"epochs": 3, "final_train_loss": 1.038, '
            '"peak_vram_gb": 6.51, "lora_rank": 8, '
            '"base_model": "Qwen2.5-Coder-7B-Instruct"}'
        )
        result = run_qlora_task.apply(
            args=["data/train.jsonl", config_json, "checkpoints/qlora-test"],
        )
        data = result.get()
        assert data["status"] == "completed"
        assert data["method"] == "qlora"
        assert "task_id" in data

    def test_run_qlora_task_custom_rank(self):
        """Run with custom LoRA rank."""
        config_json = (  # noqa: E501
            '{"epochs": 2, "final_train_loss": 1.0, '
            '"peak_vram_gb": 7.0, "lora_rank": 16, '
            '"base_model": "Qwen2.5-Coder-7B-Instruct"}'
        )
        result = run_qlora_task.apply(
            args=["data/train.jsonl", config_json, "checkpoints/qlora-rank16"],
        )
        data = result.get()
        assert data["status"] == "completed"
        assert data["lora_rank"] == 16

    def test_run_qlora_task_retry_on_error(self):
        """When config_json is invalid, the retry handler at
        lines 228-230 is exercised."""
        import celery

        with pytest.raises(celery.exceptions.Retry):
            run_qlora_task.apply(
                args=["data/train.jsonl", "not valid json", "checkpoints/qlora-test"],
            )


# ============================================================================
# run_dpo_task — covers lines 251-293
# ============================================================================


class TestRunDPOTask:
    def test_run_dpo_task(self):
        """Run the DPO task — covers lines 251-293."""
        config_json = (  # noqa: E501
            '{"epochs": 3, "final_train_loss": 0.85, '
            '"peak_vram_gb": 7.2, "base_model": "Qwen2.5-Coder-7B-Instruct"}'
        )
        result = run_dpo_task.apply(
            args=["data/train.jsonl", config_json, "checkpoints/dpo-test"],
        )
        data = result.get()
        assert data["status"] == "completed"
        assert data["method"] == "dpo"
        assert "task_id" in data

    def test_run_dpo_task_custom_config(self):
        """Run with custom hyperparameters."""
        config_json = (  # noqa: E501
            '{"epochs": 4, "final_train_loss": 0.75, '
            '"peak_vram_gb": 9.0, "base_model": "Qwen2.5-Coder-7B-Instruct"}'
        )
        result = run_dpo_task.apply(
            args=["data/train.jsonl", config_json, "checkpoints/dpo-custom"],
        )
        data = result.get()
        assert data["status"] == "completed"
        assert data["peak_vram_gb"] == 9.0

    def test_run_dpo_task_retry_on_error(self):
        """When config_json is invalid, the retry handler at
        lines 295-297 is exercised."""
        import celery

        with pytest.raises(celery.exceptions.Retry):
            run_dpo_task.apply(
                args=["data/train.jsonl", "not valid json", "checkpoints/dpo-test"],
            )
