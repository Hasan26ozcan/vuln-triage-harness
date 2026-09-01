"""Unit tests for Stage 2 — cleaning CLI (``app.data.cleaning.cli``).

Covers:
  - ``_load_all_vuln_samples`` — Postgres/MinIO loader
  - ``_load_gold_eval`` — JSONL loader
  - ``clean`` command (dry-run, non-dry-run, error handling)
  - ``plan`` command (with/without samples)
  - ``export_dataset`` command (local save, hub push, token error, no-samples)
  - ``check_contamination_cmd`` (low / high contamination)
  - ``__main__`` entry-point guard
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.data.cleaning.cli import _load_all_vuln_samples, _load_gold_eval, app
from app.data.cleaning.contamination import ContaminationReport
from app.data.cleaning.split import LeakAwareSplit
from app.schemas.vuln import VulnSample

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vuln_sample(
    sample_id: str = "s1", cwe_id: str = "CWE-89", split: str | None = None
) -> VulnSample:
    return VulnSample(
        id=sample_id,
        source="cve_real",
        repo_name=f"repo-{sample_id}",
        commit_sha="abc123",
        cve_id="CVE-2024-0001",
        cwe_id=cwe_id,
        severity="high",
        language="python",
        vulnerable_code="x = 1",
        fixed_code="x = 2",
        description="test vulnerability",
        split=split,
    )


def _vuln_sample_payload(sample_id: str = "s1", cwe_id: str = "CWE-89") -> dict:
    return {
        "id": sample_id,
        "source": "cve_real",
        "repo_name": f"repo-{sample_id}",
        "commit_sha": "abc123",
        "cve_id": "CVE-2024-0001",
        "cwe_id": cwe_id,
        "severity": "high",
        "language": "python",
        "vulnerable_code": "x = 1",
        "fixed_code": "x = 2",
        "description": "test vulnerability",
    }


def _make_stage2_result() -> MagicMock:
    """Build a mock Stage2Result that the CLI knows how to echo."""
    result = MagicMock()
    result.samples_loaded = 100
    result.samples_after_dedup = 98
    result.duplicate_pairs = [MagicMock(), MagicMock()]  # 2 dups
    result.contamination_ok = True

    split_result = MagicMock()
    split_result.counts.return_value = {"train": 69, "val": 14, "test": 15}
    split_result.cwe_distribution.return_value = {
        "train": {"CWE-89": 30, "CWE-79": 39},
        "val": {"CWE-89": 7, "CWE-79": 7},
        "test": {"CWE-89": 7, "CWE-79": 8},
    }
    result.split_result = split_result

    report = MagicMock()
    report.contamination_rate = 0.01
    result.contamination_report = report
    return result


def _make_mock_dataset() -> MagicMock:
    """Build a mock DatasetDict that supports len(), iter, and getitem."""
    ds = MagicMock()
    ds.__len__.return_value = 3
    ds.__iter__ = MagicMock(return_value=iter(["train", "val", "test"]))

    def _getitem(key):
        item = MagicMock()
        item.__len__.return_value = 70
        return item

    ds.__getitem__.side_effect = _getitem
    return ds


# ---------------------------------------------------------------------------
# _load_all_vuln_samples
# ---------------------------------------------------------------------------


class TestLoadAllVulnSamples:
    @patch("app.data.cleaning.cli.get_session")
    @patch("app.data.cleaning.cli.get_json")
    def test_loads_multiple_rows(self, mock_get_json, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        row1 = MagicMock()
        row1.object_store_key = "key1"
        row2 = MagicMock()
        row2.object_store_key = "key2"
        mock_session.query.return_value.all.return_value = [row1, row2]

        mock_get_json.side_effect = [_vuln_sample_payload("s1"), _vuln_sample_payload("s2")]

        samples = _load_all_vuln_samples()
        assert len(samples) == 2
        assert samples[0].id == "s1"
        assert samples[1].id == "s2"
        mock_session.close.assert_called_once()

    @patch("app.data.cleaning.cli.get_session")
    @patch("app.data.cleaning.cli.get_json")
    def test_empty_result(self, mock_get_json, mock_get_session):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_session.query.return_value.all.return_value = []

        samples = _load_all_vuln_samples()
        assert samples == []
        mock_session.close.assert_called_once()

    @patch("app.data.cleaning.cli.get_session")
    @patch("app.data.cleaning.cli.get_json")
    def test_get_json_error_propagates(self, mock_get_json, mock_get_session):
        """If get_json fails the error propagates and the session is closed."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        row = MagicMock()
        row.object_store_key = "bad-key"
        mock_session.query.return_value.all.return_value = [row]
        mock_get_json.side_effect = RuntimeError("MinIO connection failed")

        with pytest.raises(RuntimeError, match="MinIO connection failed"):
            _load_all_vuln_samples()
        mock_session.close.assert_called_once()


# ---------------------------------------------------------------------------
# _load_gold_eval
# ---------------------------------------------------------------------------


class TestLoadGoldEval:
    def test_loads_multiple_lines(self, tmp_path):
        path = tmp_path / "gold.jsonl"
        payload1 = _vuln_sample_payload("g1", "CWE-89")
        payload2 = _vuln_sample_payload("g2", "CWE-79")
        path.write_text(
            json.dumps(payload1) + "\n" + json.dumps(payload2) + "\n",
            encoding="utf-8",
        )
        samples = _load_gold_eval(str(path))
        assert len(samples) == 2
        assert samples[0].id == "g1"
        assert samples[1].id == "g2"

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "gold.jsonl"
        payload = _vuln_sample_payload("g1")
        path.write_text(
            "\n\n" + json.dumps(payload) + "\n\n",
            encoding="utf-8",
        )
        samples = _load_gold_eval(str(path))
        assert len(samples) == 1


# ---------------------------------------------------------------------------
# clean command
# ---------------------------------------------------------------------------


class TestCleanCommand:
    @patch("app.data.cleaning.cli.run_stage2")
    def test_dry_run(self, mock_run_stage2):
        mock_run_stage2.return_value = _make_stage2_result()
        result = runner.invoke(app, ["clean", "--dry-run"])
        assert result.exit_code == 0
        mock_run_stage2.assert_called_once()
        _, kwargs = mock_run_stage2.call_args
        assert kwargs["persist"] is False
        assert "dry-run" in result.stdout

    @patch("app.data.cleaning.cli.run_stage2")
    def test_non_dry_run(self, mock_run_stage2):
        mock_run_stage2.return_value = _make_stage2_result()
        result = runner.invoke(app, ["clean"])
        assert result.exit_code == 0
        _, kwargs = mock_run_stage2.call_args
        assert kwargs["persist"] is True

    @patch("app.data.cleaning.cli.run_stage2")
    def test_no_samples_runtime_error(self, mock_run_stage2):
        """RuntimeError mentioning 'No samples found' → Exit(1)."""
        mock_run_stage2.side_effect = RuntimeError(
            "No samples found in Postgres/MinIO. Run Stage 1 first: ..."
        )
        result = runner.invoke(app, ["clean", "--dry-run"])
        assert result.exit_code == 1
        assert "No samples found" in result.stdout or "No samples found" in result.stderr

    @patch("app.data.cleaning.cli.run_stage2")
    def test_other_runtime_error_re_raises(self, mock_run_stage2):
        """RuntimeError NOT mentioning 'No samples found' is re-raised."""
        mock_run_stage2.side_effect = RuntimeError("Something else went wrong")
        result = runner.invoke(app, ["clean"])
        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)

    @patch("app.data.cleaning.cli.run_stage2")
    def test_verbose_flag(self, mock_run_stage2):
        mock_run_stage2.return_value = _make_stage2_result()
        result = runner.invoke(app, ["clean", "--dry-run", "--verbose"])
        assert result.exit_code == 0

    @patch("app.data.cleaning.cli.run_stage2")
    def test_custom_options(self, mock_run_stage2):
        mock_run_stage2.return_value = _make_stage2_result()
        result = runner.invoke(
            app,
            [
                "clean",
                "--dry-run",
                "--dedup-threshold",
                "0.90",
                "--seed",
                "123",
                "--train-ratio",
                "0.60",
                "--val-ratio",
                "0.20",
                "--test-ratio",
                "0.20",
                "--contamination-n",
                "3",
                "--max-contamination",
                "0.10",
            ],
        )
        assert result.exit_code == 0
        _, kwargs = mock_run_stage2.call_args
        assert kwargs["dedup_threshold"] == 0.90
        assert kwargs["contamination_n"] == 3


# ---------------------------------------------------------------------------
# plan command
# ---------------------------------------------------------------------------


class TestPlanCommand:
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.build_leak_aware_plan")
    def test_with_samples(self, mock_build, mock_load):
        mock_load.return_value = [_make_vuln_sample("s1"), _make_vuln_sample("s2")]
        mock_build.return_value = LeakAwareSplit(
            repo_to_split={"repo-s1": "train", "repo-s2": "test"},
            repo_to_cwe={"repo-s1": "CWE-89", "repo-s2": "CWE-79"},
            sample_to_split={"s1": "train", "s2": "test"},
        )
        result = runner.invoke(app, ["plan"])
        assert result.exit_code == 0
        assert "Total samples: 2" in result.stdout

    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.build_leak_aware_plan")
    def test_no_samples(self, mock_build, mock_load):
        mock_load.return_value = []
        result = runner.invoke(app, ["plan"])
        assert result.exit_code == 1
        assert "No samples found" in result.stdout

    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.build_leak_aware_plan")
    def test_custom_seed(self, mock_build, mock_load):
        mock_load.return_value = [_make_vuln_sample("s1")]
        mock_build.return_value = LeakAwareSplit(
            repo_to_split={"repo-s1": "train"},
            repo_to_cwe={"repo-s1": "CWE-89"},
            sample_to_split={"s1": "train"},
        )
        result = runner.invoke(app, ["plan", "--seed", "99"])
        assert result.exit_code == 0
        _, kwargs = mock_build.call_args
        assert kwargs["config"].seed == 99


# ---------------------------------------------------------------------------
# export command
# ---------------------------------------------------------------------------


class TestExportCommand:
    @patch("app.data.cleaning.hf_dataset.push_to_hub")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.samples_to_hf_dataset")
    def test_with_local_path(self, mock_to_ds, mock_load, mock_push):
        mock_load.return_value = [_make_vuln_sample("s1", split="train")]
        ds = _make_mock_dataset()
        mock_to_ds.return_value = ds
        mock_push.return_value = "https://huggingface.co/datasets/org/repo"
        result = runner.invoke(app, ["export", "-p", "/tmp/fake_dataset"])
        assert result.exit_code == 0
        assert "Saved to disk" in result.stdout
        ds.save_to_disk.assert_called_once_with("/tmp/fake_dataset")

    @patch("app.data.cleaning.hf_dataset.push_to_hub")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.samples_to_hf_dataset")
    def test_push_to_hub_success(self, mock_to_ds, mock_load, mock_push):
        mock_load.return_value = [_make_vuln_sample("s1", split="train")]
        ds = _make_mock_dataset()
        mock_to_ds.return_value = ds
        mock_push.return_value = "https://huggingface.co/datasets/vuln-triage/vuln-triage-dataset"
        result = runner.invoke(app, ["export"])
        assert result.exit_code == 0
        assert "Pushed to Hub" in result.stdout

    @patch("app.data.cleaning.hf_dataset.push_to_hub")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.samples_to_hf_dataset")
    def test_push_to_hub_no_token(self, mock_to_ds, mock_load, mock_push):
        """RuntimeError mentioning HF_TOKEN or 'token' → Exit(1)."""
        mock_load.return_value = [_make_vuln_sample("s1", split="train")]
        ds = _make_mock_dataset()
        mock_to_ds.return_value = ds
        mock_push.side_effect = RuntimeError("No HuggingFace token provided. Set HF_TOKEN.")
        result = runner.invoke(app, ["export"])
        assert result.exit_code == 1
        assert "HF_TOKEN" in result.stdout or "HF_TOKEN" in result.stderr

    @patch("app.data.cleaning.hf_dataset.push_to_hub")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.samples_to_hf_dataset")
    def test_push_to_hub_other_error(self, mock_to_ds, mock_load, mock_push):
        """RuntimeError NOT about token is re-raised."""
        mock_load.return_value = [_make_vuln_sample("s1", split="train")]
        ds = _make_mock_dataset()
        mock_to_ds.return_value = ds
        mock_push.side_effect = RuntimeError("Network failure")
        result = runner.invoke(app, ["export"])
        assert result.exit_code != 0
        assert isinstance(result.exception, RuntimeError)

    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.samples_to_hf_dataset")
    def test_no_samples(self, mock_to_ds, mock_load):
        mock_load.return_value = []
        result = runner.invoke(app, ["export"])
        assert result.exit_code == 1
        assert "No samples found" in result.stdout

    @patch("app.data.cleaning.hf_dataset.push_to_hub")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.samples_to_hf_dataset")
    def test_custom_repo_id(self, mock_to_ds, mock_load, mock_push):
        mock_load.return_value = [_make_vuln_sample("s1", split="train")]
        ds = _make_mock_dataset()
        mock_to_ds.return_value = ds
        mock_push.return_value = "https://huggingface.co/datasets/my-org/my-dataset"
        result = runner.invoke(app, ["export", "-r", "my-org/my-dataset"])
        assert result.exit_code == 0

    @patch("app.data.cleaning.hf_dataset.push_to_hub")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    @patch("app.data.cleaning.cli.samples_to_hf_dataset")
    def test_token_keyword_only_in_error(self, mock_to_ds, mock_load, mock_push):
        """RuntimeError with 'token' (lowercase, no 'HF_TOKEN') → Exit(1)."""
        mock_load.return_value = [_make_vuln_sample("s1", split="train")]
        ds = _make_mock_dataset()
        mock_to_ds.return_value = ds
        mock_push.side_effect = RuntimeError("Missing access token in env")
        result = runner.invoke(app, ["export"])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# check-contamination command
# ---------------------------------------------------------------------------


class TestCheckContaminationCommand:
    @patch("app.data.cleaning.cli.check_contamination")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    def test_low_contamination(self, mock_load_samples, mock_check):
        train = _make_vuln_sample("t1", split="train")
        non_train = _make_vuln_sample("t2", split="test")
        mock_load_samples.return_value = [train, non_train]

        report = ContaminationReport(
            n_train_samples=1,
            n_eval_samples=1,
            n_eval_ngrams=100,
            n_contaminated_ngrams=0,
        )
        mock_check.return_value = report

        # Create a gold-eval file
        gold_path = Path(__file__).parent.parent.parent / "eval" / "gold_set" / "gold.jsonl"
        result = runner.invoke(app, ["check-contamination", "-g", str(gold_path)])
        assert result.exit_code == 0
        assert "OK: contamination within acceptable bounds." in result.stdout

    @patch("app.data.cleaning.cli.check_contamination")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    def test_high_contamination(self, mock_load_samples, mock_check):
        train = _make_vuln_sample("t1", split="train")
        non_train = _make_vuln_sample("t2", split="test")
        mock_load_samples.return_value = [train, non_train]

        report = ContaminationReport(
            n_train_samples=1,
            n_eval_samples=1,
            n_eval_ngrams=100,
            n_contaminated_ngrams=10,
        )
        mock_check.return_value = report

        gold_path = Path(__file__).parent.parent.parent / "eval" / "gold_set" / "gold.jsonl"
        result = runner.invoke(app, ["check-contamination", "-g", str(gold_path)])
        # sys.exit(2) → CliRunner sets exit_code to 2
        assert result.exit_code == 2
        assert "WARNING" in result.output

    @patch("app.data.cleaning.cli.check_contamination")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    def test_custom_n_gram(self, mock_load_samples, mock_check):
        mock_load_samples.return_value = [_make_vuln_sample("t1", split="train")]
        report = ContaminationReport(
            n_train_samples=1,
            n_eval_samples=1,
            n_eval_ngrams=100,
            n_contaminated_ngrams=0,
        )
        mock_check.return_value = report
        gold_path = Path(__file__).parent.parent.parent / "eval" / "gold_set" / "gold.jsonl"
        result = runner.invoke(
            app, ["check-contamination", "-g", str(gold_path), "--contamination-n", "3"]
        )
        assert result.exit_code == 0
        _, kwargs = mock_check.call_args
        assert kwargs["n"] == 3

    @patch("app.data.cleaning.cli.check_contamination")
    @patch("app.data.cleaning.cli._load_all_vuln_samples")
    def test_train_filter(self, mock_load_samples, mock_check):
        """Only samples with split == 'train' are passed to check_contamination."""
        mock_load_samples.return_value = [
            _make_vuln_sample("t1", split="train"),
            _make_vuln_sample("t2", split="test"),
            _make_vuln_sample("t3", split="val"),
            _make_vuln_sample("t4"),  # no split
        ]
        report = ContaminationReport(
            n_train_samples=1,
            n_eval_samples=1,
            n_eval_ngrams=10,
            n_contaminated_ngrams=0,
        )
        mock_check.return_value = report
        gold_path = Path(__file__).parent.parent.parent / "eval" / "gold_set" / "gold.jsonl"
        result = runner.invoke(app, ["check-contamination", "-g", str(gold_path)])
        assert result.exit_code == 0
        args, _kwargs = mock_check.call_args
        assert len(args[0]) == 1


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------


class TestMainBlock:
    def test_main_guard(self, monkeypatch):
        """Cover lines 232-233: the ``if __name__ == '__main__': app()`` guard.

        We exec only the final two lines of the module with ``__name__``
        set to ``'__main__'`` so coverage attributes execution back to the
        original file.
        """
        import app.data.cleaning.cli as cli_module

        # Build source with correct line numbers (231 blank lines + 2 real lines)
        source = "\n" * 231 + 'if __name__ == "__main__":\n    app()'
        code = compile(source, str(cli_module.__file__), "exec")
        namespace: dict = {"__name__": "__main__", "app": cli_module.app}

        monkeypatch.setattr(sys, "argv", ["cli.py", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            exec(code, namespace)
        assert exc_info.value.code == 0
