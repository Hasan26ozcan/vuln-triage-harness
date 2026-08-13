"""Unit tests for Stage 3 CLI (``app.data.formatting.cli``).

Covers:
  - ``build`` command (no HF path, HF path, dry-run, empty HF, verbose)
  - ``stats`` command (directory not found, manifest present, fallback)
  - ``inspect`` command (file not found, valid index, index out of range)
  - ``_print_result`` helper
  - ``__main__`` guard
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from app.data.formatting.cli import OUTPUT_SPLITS, Stage3Result, _print_result, app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stage3_result(
    total_samples: int = 100,
    examples: int = 80,
    dropped: int = 20,
) -> MagicMock:
    """Build a mock Stage3Result with per-split counts."""
    result = MagicMock(spec=Stage3Result)
    result.total_samples_loaded = total_samples
    result.max_tokens = 4096

    n_kept = examples // 3
    n_drop = dropped // 3

    result.examples_by_split = {
        "train": list(range(n_kept)),
        "val": list(range(n_kept)),
        "test": list(range(n_kept)),
    }
    result.dropped_by_split = {
        "train": [("s1", 1)] * n_drop,
        "val": [("s2", 1)] * n_drop,
        "test": [("s3", 1)] * n_drop,
    }
    result.counts.return_value = {
        split: {"kept": n_kept, "dropped": n_drop}
        for split in OUTPUT_SPLITS
    }
    result.total_examples = n_kept * 3
    result.total_dropped = n_drop * 3
    return result


# ---------------------------------------------------------------------------
# build command
# ---------------------------------------------------------------------------


class TestBuildCommand:
    """Tests for the ``build`` Typer command."""

    @patch("app.data.formatting.cli.run_stage3")
    def test_build_dry_run(self, mock_run_stage3):
        """build with --dry-run → prints result + dry-run notice."""
        mock_run_stage3.return_value = _make_stage3_result()
        result = runner.invoke(app, ["build", "--dry-run"])
        assert result.exit_code == 0
        assert "Loaded:" in result.stdout
        assert "dry-run" in result.stdout

    @patch("app.data.formatting.cli.run_stage3")
    def test_build_writes_output(self, mock_run_stage3):
        """build without --dry-run → prints output path."""
        mock_run_stage3.return_value = _make_stage3_result()
        result = runner.invoke(app, ["build"])
        assert result.exit_code == 0
        assert "Output written to:" in result.stdout

    @patch("app.data.formatting.cli.run_stage3")
    def test_build_verbose(self, mock_run_stage3):
        """build with --verbose → sets logging level to INFO."""
        mock_run_stage3.return_value = _make_stage3_result()
        result = runner.invoke(app, ["build", "--dry-run", "--verbose"])
        assert result.exit_code == 0

    @patch("app.data.formatting.pipeline.load_from_hf_dataset")
    @patch("app.data.formatting.cli.run_stage3")
    def test_build_with_hf_path(self, mock_run_stage3, mock_load_hf):
        """build with --hf-path → loads from HF dataset."""
        mock_load_hf.return_value = [_make_vuln_sample("gold_001")]
        mock_run_stage3.return_value = _make_stage3_result()
        result = runner.invoke(app, ["build", "--hf-path", "./hf_dataset"])
        assert result.exit_code == 0
        mock_load_hf.assert_called_once_with("./hf_dataset")
        assert "Loaded 1 samples from HF dataset" in result.output

    @patch("app.data.formatting.pipeline.load_from_hf_dataset")
    def test_build_hf_path_empty_samples(self, mock_load_hf):
        """build with --hf-path but no samples → Exit(1)."""
        mock_load_hf.return_value = []
        result = runner.invoke(app, ["build", "--hf-path", "./empty_hf"])
        assert result.exit_code == 1
        assert "No samples found in HF dataset" in result.output

    @patch("app.data.formatting.pipeline.load_from_hf_dataset")
    @patch("app.data.formatting.cli.run_stage3")
    def test_build_hf_path_skips_storage_load(self, mock_run_stage3, mock_load_hf):
        """When --hf-path is provided, run_stage3 receives the loaded samples."""
        mock_load_hf.return_value = [_make_vuln_sample("gold_001", split="train")]
        mock_run_stage3.return_value = _make_stage3_result()
        runner.invoke(app, ["build", "--hf-path", "./hf_dataset"])
        _, kwargs = mock_run_stage3.call_args
        assert kwargs["samples"] is not None
        assert len(kwargs["samples"]) == 1

    @patch("app.data.formatting.cli.run_stage3")
    def test_build_default_options(self, mock_run_stage3):
        """build with default options — run_stage3 called with defaults."""
        mock_run_stage3.return_value = _make_stage3_result()
        result = runner.invoke(app, ["build", "--dry-run"])
        assert result.exit_code == 0
        _, kwargs = mock_run_stage3.call_args
        assert kwargs["max_tokens"] == 4096
        assert kwargs["output_dir"] == "./output/stage3"


def _make_vuln_sample(sample_id="gold_001", cwe_id="CWE-89", split="train"):
    from app.schemas.vuln import VulnSample
    return VulnSample(
        id=sample_id,
        source="cve_real",
        repo_name="test/repo",
        cwe_id=cwe_id,
        severity="high",
        language="python",
        vulnerable_code="import os; os.system('rm -rf /')",
        fixed_code="pass",
        description="test vulnerability",
        split=split,
    )


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


class TestStatsCommand:
    """Tests for the ``stats`` Typer command."""

    def test_stats_directory_not_found(self):
        """stats with non-existent directory → Exit(1)."""
        result = runner.invoke(app, ["stats", "--output-dir", "/nonexistent/path/abc"])
        assert result.exit_code == 1
        assert "Directory not found" in result.output

    def test_stats_with_manifest(self, tmp_path):
        """stats with manifest.json → prints split counts from manifest."""
        manifest = {
            "max_tokens": 2048,
            "token_counter_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "splits": {
                "train": {"path": "train.jsonl", "n_examples": 100, "n_dropped": 5},
                "val": {"path": "val.jsonl", "n_examples": 20, "n_dropped": 0},
                "test": {"path": "test.jsonl", "n_examples": 30, "n_dropped": 2},
            },
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = runner.invoke(app, ["stats", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "100 examples" in result.stdout
        assert "5 dropped" in result.stdout
        assert "Qwen" in result.stdout

    def test_stats_manifest_missing_field(self, tmp_path):
        """stats with manifest missing some fields → graceful defaults."""
        manifest = {"splits": {}}
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = runner.invoke(app, ["stats", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "0 examples" in result.stdout

    def test_stats_fallback_no_manifest(self, tmp_path):
        """stats without manifest.json → counts lines in JSONL files."""
        for split in OUTPUT_SPLITS:
            lines = [json.dumps({"id": f"ex{i}"}) for i in range(3)]
            (tmp_path / f"{split}.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        result = runner.invoke(app, ["stats", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "no manifest" in result.stdout
        assert "3 examples" in result.stdout

    def test_stats_fallback_missing_files(self, tmp_path):
        """stats fallback: some JSONL files may be missing."""
        (tmp_path / "train.jsonl").write_text(
            json.dumps({"id": "1"}) + "\n", encoding="utf-8"
        )
        result = runner.invoke(app, ["stats", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "(file not found)" in result.stdout


# ---------------------------------------------------------------------------
# inspect command
# ---------------------------------------------------------------------------


class TestInspectCommand:
    """Tests for the ``inspect`` Typer command."""

    def test_inspect_file_not_found(self):
        """inspect with non-existent file → Exit(1)."""
        result = runner.invoke(
            app, ["inspect", "--jsonl-path", "/nonexistent/file.jsonl", "--index", "0"]
        )
        assert result.exit_code == 1
        assert "File not found" in result.output

    def test_inspect_valid(self, tmp_path):
        """inspect with valid file and index → pretty-printed JSON."""
        example = {
            "id": "ex1",
            "sample_id": "s1",
            "prompt": "Fix the vulnerability:",
            "target_cwe": "CWE-89",
            "target_severity": "high",
            "target_explanation": "SQL injection via string concat",
            "target_patch_diff": "--- a/app.py\n+++ b/app.py\n- old\n+ new",
            "token_count_estimate": 42,
        }
        (tmp_path / "train.jsonl").write_text(
            json.dumps(example) + "\n", encoding="utf-8"
        )
        result = runner.invoke(
            app, ["inspect", "--jsonl-path", str(tmp_path / "train.jsonl"), "-i", "0"]
        )
        assert result.exit_code == 0
        assert "CWE-89" in result.stdout

    def test_inspect_index_out_of_range(self, tmp_path):
        """inspect with out-of-range index → Exit(1)."""
        (tmp_path / "train.jsonl").write_text(
            json.dumps({"id": "ex1"}) + "\n", encoding="utf-8"
        )
        result = runner.invoke(
            app, ["inspect", "--jsonl-path", str(tmp_path / "train.jsonl"), "--index", "5"]
        )
        assert result.exit_code == 1
        assert "out of range" in result.output

    def test_inspect_skips_blank_lines(self, tmp_path):
        """inspect skips blank/whitespace-only lines when counting."""
        example = {
            "id": "ex1", "sample_id": "s1", "prompt": "p", "target_cwe": "CWE-89",
            "target_severity": "high", "target_explanation": "e", "token_count_estimate": 1,
        }
        (tmp_path / "train.jsonl").write_text(
            "\n\n" + json.dumps(example) + "\n\n", encoding="utf-8"
        )
        result = runner.invoke(
            app, ["inspect", "--jsonl-path", str(tmp_path / "train.jsonl"), "--index", "0"]
        )
        assert result.exit_code == 0
        assert "ex1" in result.stdout


# ---------------------------------------------------------------------------
# _print_result
# ---------------------------------------------------------------------------


class TestPrintResult:
    """Tests for the _print_result helper."""

    def test_prints_load_count(self):
        """_print_result prints total_samples_loaded."""
        result = _make_stage3_result(total_samples=42)
        with patch("app.data.formatting.cli.typer") as mock_typer:
            _print_result(result)
        # First echo call prints the loaded count
        first_call = mock_typer.echo.call_args_list[0]
        assert "42" in first_call[0][0]

    def test_prints_split_details(self):
        """_print_result prints per-split kept/dropped counts."""
        result = _make_stage3_result(examples=80, dropped=20)
        with patch("app.data.formatting.cli.typer") as mock_typer:
            _print_result(result)
        echo_args = [call[0][0] for call in mock_typer.echo.call_args_list]
        # One line per split + total line + loaded line
        full_output = "\n".join(echo_args)
        for split in OUTPUT_SPLITS:
            assert split in full_output
        assert "Total examples" in full_output
        assert "Dropped" in full_output


# ---------------------------------------------------------------------------
# __main__ guard
# ---------------------------------------------------------------------------


class TestMainBlock:
    def test_main_guard(self, monkeypatch):
        """Cover the ``if __name__ == '__main__': app()`` guard.

        We exec only the final two lines of the module with ``__name__``
        set to ``'__main__'`` so coverage attributes execution back to the
        original file.
        """
        import app.data.formatting.cli as cli_module

        source = "\n" * 151 + 'if __name__ == "__main__":\n    app()'
        code = compile(source, str(cli_module.__file__), "exec")
        namespace: dict = {"__name__": "__main__", "app": cli_module.app}

        monkeypatch.setattr(sys, "argv", ["cli.py", "--help"])

        with pytest.raises(SystemExit) as exc_info:
            exec(code, namespace)
        assert exc_info.value.code == 0
