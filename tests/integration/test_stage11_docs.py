"""Integration tests for Stage 11 -- documentation & interview package.

These tests exercise the Stage 11 deliverables end-to-end using mock backends
(no GPU, no Docker, no network):

1. Generate all three deliverables via ``Stage11Generator.ensure_deliverables()``.
2. Verify the model card and training report contain the expected Markdown
   sections and YAML front-matter.
3. Run the demo script (``docs/demo.py``) in mock mode and verify it completes
   successfully (exit code 0).
4. Test the ``stage11`` Typer CLI subcommand via CliRunner.
5. Verify the JSON sidecars round-trip correctly.
6. Verify the demo script produces the expected output artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from app.schemas.documentation import BASE_MODEL, CWE_SCOPE
from app.stage11.config import Stage11Config
from app.stage11.generator import Stage11Generator

# ---------------------------------------------------------------------------
# 1. Deliverables generation
# ---------------------------------------------------------------------------


class TestDeliverablesGeneration:
    """Generate all deliverables and verify they are correct."""

    @pytest.fixture
    def generated(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        results = gen.ensure_deliverables()
        return {
            "gen": gen,
            "docs_dir": docs_dir,
            "output_dir": output_dir,
            "results": results,
        }

    def test_all_files_created(self, generated):
        docs_dir = generated["docs_dir"]
        assert (docs_dir / "model_card.md").exists()
        assert (docs_dir / "training_report.md").exists()
        assert (docs_dir / "demo.py").exists()

    def test_all_files_non_empty(self, generated):
        docs_dir = generated["docs_dir"]
        for name in ["model_card.md", "training_report.md", "demo.py"]:
            assert (docs_dir / name).stat().st_size > 0

    def test_validate_returns_true(self, generated):
        assert generated["gen"].validate_deliverables() is True


# ---------------------------------------------------------------------------
# 2. Markdown content verification
# ---------------------------------------------------------------------------


class TestMarkdownContent:
    """Verify the generated Markdown documents have the expected structure."""

    @pytest.fixture
    def generated(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(
            docs_dir=str(docs_dir),
            output_dir=str(output_dir),
            model_name="test-integration-model",
        )
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()
        return docs_dir

    def test_model_card_has_frontmatter(self, generated: Path):
        content = (generated / "model_card.md").read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "title:" in content
        assert "test-integration-model" in content

    def test_model_card_has_sections(self, generated: Path):
        content = (generated / "model_card.md").read_text(encoding="utf-8")
        assert "# Model Card" in content
        assert "## Model Details" in content
        assert "## Intended Use" in content
        assert "## Evaluation" in content
        assert "## Limitations" in content
        assert "## Ethical Considerations" in content
        assert "## Out of Scope" in content
        assert "## Citation" in content

    def test_model_card_contains_base_model(self, generated: Path):
        content = (generated / "model_card.md").read_text(encoding="utf-8")
        assert BASE_MODEL in content

    def test_model_card_contains_cwe_scope(self, generated: Path):
        content = (generated / "model_card.md").read_text(encoding="utf-8")
        for cwe in CWE_SCOPE:
            assert cwe in content

    def test_training_report_has_frontmatter(self, generated: Path):
        content = (generated / "training_report.md").read_text(encoding="utf-8")
        assert content.startswith("---\n")
        assert "title:" in content

    def test_training_report_has_sections(self, generated: Path):
        content = (generated / "training_report.md").read_text(encoding="utf-8")
        assert "# Training Report" in content
        assert "## Overview" in content
        assert "## Evaluation Results" in content
        assert "## Conclusions" in content
        assert "## Recommendations" in content

    def test_demo_script_is_valid_python(self, generated: Path):
        """The generated demo script should be syntactically valid Python."""
        import ast

        content = (generated / "demo.py").read_text(encoding="utf-8")
        ast.parse(content)


# ---------------------------------------------------------------------------
# 3. Demo script execution
# ---------------------------------------------------------------------------


class TestDemoScriptExecution:
    """Run the generated demo script and verify it completes."""

    @pytest.fixture
    def generated(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()
        return docs_dir

    def test_demo_script_runs(self, generated: Path, tmp_path: Path):
        """Run docs/demo.py as a subprocess and verify it exits 0."""
        # The demo script resolves project_root as parent-of-docs = __file__.parent.parent.
        # Set up a full project-like structure in tmp_path.
        tmp_docs = tmp_path / "docs"
        tmp_docs.mkdir(exist_ok=True)
        (tmp_docs / "demo.py").write_text(
            (generated / "demo.py").read_text(encoding="utf-8"), encoding="utf-8"
        )

        # Copy the gold-eval set so the demo can find it.
        project_root = Path(__file__).resolve().parents[2]
        gold_src = project_root / "eval" / "gold_set" / "gold.jsonl"
        tmp_eval_dir = tmp_path / "eval" / "gold_set"
        tmp_eval_dir.mkdir(parents=True, exist_ok=True)
        (tmp_eval_dir / "gold.jsonl").write_bytes(gold_src.read_bytes())

        # Pass PYTHONPATH so the subprocess can import the ``app`` package
        # even when the project is not installed in editable mode. The demo
        # script also adds its own project_root (tmp_path) to sys.path, but
        # that directory does not contain the ``app`` package — it lives in
        # the real repo root.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(project_root)

        result = subprocess.run(
            [sys.executable, str(tmp_docs / "demo.py")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
            timeout=120,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        assert result.returncode == 0, (
            f"Demo script failed with return code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_demo_script_uses_mock_mode(self, generated: Path):
        """Demo script should reference mock mode via CLI flags."""
        content = (generated / "demo.py").read_text(encoding="utf-8")
        assert "mock" in content.lower()
        # The demo invokes CLI subcommands with --mock and --sandbox-mode mock
        assert "--mock" in content
        assert "sandbox-mode" in content

    def test_demo_script_no_unicode_emojis(self, generated: Path):
        """Demo script must not contain emoji or arrow characters."""
        content = (generated / "demo.py").read_text(encoding="utf-8")
        for char in ["✅", "❌", "⚠", "→", "—"]:
            assert char not in content, f"Found forbidden Unicode char {ord(char)} in demo script"


# ---------------------------------------------------------------------------
# 4. CLI integration
# ---------------------------------------------------------------------------


class TestCLIStage11:
    """Test the ``stage11`` Typer CLI subcommand via CliRunner."""

    def test_cli_generates_deliverables(self, tmp_path: Path):
        from app.evaluation.cli import app

        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage11",
                "--docs-dir",
                str(docs_dir),
                "--output-dir",
                str(output_dir),
                "--no-demo",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Documentation" in result.output or "documentation" in result.output

        assert (docs_dir / "model_card.md").exists()
        assert (docs_dir / "training_report.md").exists()
        assert (docs_dir / "demo.py").exists()

    def test_cli_with_custom_options(self, tmp_path: Path):
        from app.evaluation.cli import app

        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage11",
                "--docs-dir",
                str(docs_dir),
                "--output-dir",
                str(output_dir),
                "--model-name",
                "cli-test-model",
                "--training-method",
                "dpo",
                "--lora-rank",
                "128",
                "--no-demo",
            ],
        )
        assert result.exit_code == 0, result.output

        mc = (docs_dir / "model_card.md").read_text(encoding="utf-8")
        assert "cli-test-model" in mc
        assert "dpo" in mc

    def test_cli_validation_passes(self, tmp_path: Path):
        from app.evaluation.cli import app

        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"

        runner = CliRunner()
        result = runner.invoke(
            app,
            [
                "stage11",
                "--docs-dir",
                str(docs_dir),
                "--output-dir",
                str(output_dir),
                "--no-demo",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "valid" in result.output.lower() or "OK" in result.output


# ---------------------------------------------------------------------------
# 5. JSON sidecar round-trip
# ---------------------------------------------------------------------------


class TestJSONSidecars:
    """Verify JSON sidecars are written and parseable."""

    def test_model_card_json(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        json_path = output_dir / "model_card_data.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        assert data["model_name"] == cfg.model_name
        assert data["base_model"] == cfg.base_model
        assert data["training_method"] == cfg.training_method
        assert data["cwe_scope"] == CWE_SCOPE

    def test_training_report_json(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        json_path = output_dir / "training_report_data.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))

        assert data["model_name"] == cfg.model_name
        assert data["base_model"] == cfg.base_model
        assert data["training_runs"] == []
        assert data["quant_results"] == []
        assert isinstance(data["conclusions"], list)
        assert isinstance(data["recommendations"], list)


# ---------------------------------------------------------------------------
# 6. Demo pipeline integration (mock mode)
# ---------------------------------------------------------------------------


class TestDemoPipeline:
    """Run the Stage 11 generator's run_demo() and verify results."""

    def test_run_demo_succeeds(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        # The gold eval file is relative to project root.
        project_root = Path(__file__).resolve().parents[2]
        gold_eval = project_root / "eval" / "gold_set" / "gold.jsonl"
        if not gold_eval.exists():
            pytest.skip("gold.jsonl not found")

        result = gen.run_demo()

        assert result.succeeded is True
        assert result.error is None
        assert result.num_gold_samples == 59
        assert result.run_id.startswith("stage11-")
        assert "tuned_cwe_macro_f1" in result.metrics
        assert "gate_status" in result.metrics

    def test_run_demo_creates_artifacts(self, tmp_path: Path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        project_root = Path(__file__).resolve().parents[2]
        gold_eval = project_root / "eval" / "gold_set" / "gold.jsonl"
        if not gold_eval.exists():
            pytest.skip("gold.jsonl not found")

        gen.run_demo()
        demo_dir = Path(cfg.output_dir) / "demo"
        assert (demo_dir / "stage4" / "metrics.json").exists()
