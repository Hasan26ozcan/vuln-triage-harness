"""Unit tests for Stage 11 -- documentation & interview package.

Covers:
  - Schema defaults and validation (ModelCardData, TrainingReportData,
    EvalMetricsSnapshot, TrainingRunData, QuantResultData, DemoResult)
  - Constants (CWE_SCOPE, BASE_MODEL, TRAINING_METHODS, LANGUAGE_SCOPE)
  - Stage11Config (frozen dataclass with README defaults)
  - Markdown generation functions (model card, training report)
  - Demo script generation (non-empty, contains expected sections)
  - Stage11Generator validation (missing files, empty files, happy path)
  - run_stage11 convenience function
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from app.schemas.documentation import (
    BASE_MODEL,
    CWE_SCOPE,
    LANGUAGE_SCOPE,
    TRAINING_METHODS,
    DemoResult,
    EvalMetricsSnapshot,
    ModelCardData,
    QuantResultData,
    TrainingReportData,
    TrainingRunData,
)
from app.stage11.config import (
    DEFAULT_DOCS_DIR,
    DEFAULT_LORA_RANK,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAINING_DATA_SIZE,
    DEFAULT_TRAINING_METHOD,
    Stage11Config,
)
from app.stage11.generator import (
    Stage11Generator,
    generate_demo_script,
    generate_model_card_markdown,
    generate_training_report_markdown,
    run_stage11,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify project constants match the README."""

    def test_cwe_scope_has_six_classes(self):
        assert len(CWE_SCOPE) == 6

    def test_cwe_scope_classes(self):
        expected = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]
        assert CWE_SCOPE == expected

    def test_base_model(self):
        assert BASE_MODEL == "Qwen/Qwen2.5-Coder-7B-Instruct"

    def test_training_methods(self):
        assert "sft_qlora" in TRAINING_METHODS
        assert "dpo" in TRAINING_METHODS

    def test_language_scope(self):
        assert "python" in LANGUAGE_SCOPE
        assert "javascript" in LANGUAGE_SCOPE


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestEvalMetricsSnapshot:
    def test_defaults(self):
        snap = EvalMetricsSnapshot(stage=6, run_id="test", base_model=BASE_MODEL)
        assert snap.stage == 6
        assert snap.cwe_macro_f1 == 0.0
        assert snap.severity_accuracy == 0.0
        assert snap.forgetting_delta is None
        assert snap.per_class == {}
        assert snap.manifest == {}
        assert snap.manifest == {}

    def test_with_values(self):
        snap = EvalMetricsSnapshot(
            stage=4,
            run_id="baseline",
            base_model=BASE_MODEL,
            cwe_macro_f1=0.85,
            severity_accuracy=0.90,
            hallucination_rate=0.05,
            patch_coverage=0.95,
            exec_pass_rate=0.70,
            forgetting_delta=0.02,
        )
        assert snap.cwe_macro_f1 == 0.85
        assert snap.forgetting_delta == 0.02


class TestTrainingRunData:
    def test_defaults(self):
        run = TrainingRunData(run_id="run1", method="sft_qlora")
        assert run.run_id == "run1"
        assert run.method == "sft_qlora"
        assert run.base_model == BASE_MODEL
        assert run.hyperparams == {}
        assert run.train_set_size == 0
        assert run.train_time_minutes == 0.0
        assert run.peak_vram_gb == 0.0
        assert run.final_train_loss == 0.0
        assert run.final_val_loss is None
        assert run.checkpoint_uri == ""
        assert run.train_loss_history == []

    def test_with_values(self):
        run = TrainingRunData(
            run_id="run1",
            method="sft_qlora",
            hyperparams={"lr": 0.001, "epochs": 3},
            train_set_size=5000,
            train_time_minutes=120.0,
            final_train_loss=0.05,
            final_val_loss=0.08,
            checkpoint_uri="hf://vuln-triage-qwen2.5-coder-7b",
            train_loss_history=[0.5, 0.2, 0.1, 0.05],
        )
        assert run.train_loss_history == [0.5, 0.2, 0.1, 0.05]
        assert run.final_val_loss == 0.08


class TestQuantResultData:
    def test_defaults(self):
        q = QuantResultData(quant_method="gguf")
        assert q.quant_method == "gguf"
        assert q.bit_width is None
        assert q.quantized_model_size_gb == 0.0
        assert q.status == "skipped"

    def test_with_values(self):
        q = QuantResultData(
            quant_method="gptq",
            bit_width=4,
            quantized_model_size_gb=3.5,
            estimated_vram_gb=4.2,
            tokens_per_sec=120.5,
            model_cwe_macro_f1=0.78,
            exec_pass_rate=0.65,
            status="completed",
        )
        assert q.bit_width == 4
        assert q.tokens_per_sec == 120.5


class TestModelCardData:
    def test_defaults(self):
        card = ModelCardData()
        assert card.model_name == "vuln-triage-qwen2.5-coder-7b"
        assert card.base_model == BASE_MODEL
        assert card.fine_tuned is True
        assert card.training_method == "sft_qlora"
        assert card.lora_rank == 64
        assert card.quant_method is None
        assert card.cwe_scope == CWE_SCOPE
        assert card.language == "python"
        assert card.training_data_size == 0
        assert card.serving_backends == ["llama.cpp", "ollama", "mock"]
        assert card.limitations == []
        assert card.intended_use == []
        assert card.out_of_scope == []

    def test_cwe_scope_copy(self):
        """Default CWE scope should be a copy, not shared across instances."""
        card1 = ModelCardData()
        card2 = ModelCardData()
        card1.cwe_scope.append("CWE-999")
        assert "CWE-999" not in card2.cwe_scope

    def test_generated_at_is_isoformat(self):
        card = ModelCardData()
        assert "T" in card.generated_at
        assert "Z" in card.generated_at or "+" in card.generated_at


class TestTrainingReportData:
    def test_defaults(self):
        report = TrainingReportData()
        assert report.model_name == "vuln-triage-qwen2.5-coder-7b"
        assert report.base_model == BASE_MODEL
        assert report.training_runs == []
        assert report.report_id == ""
        assert report.conclusions == []
        assert report.recommendations == []

    def test_with_runs(self):
        runs = [TrainingRunData(run_id="run1", method="sft_qlora")]
        report = TrainingReportData(
            model_name="test-model",
            training_runs=runs,
        )
        assert len(report.training_runs) == 1
        assert report.training_runs[0].run_id == "run1"


class TestDemoResult:
    def test_defaults(self):
        result = DemoResult(run_id="demo1", model_name="test", num_gold_samples=0)
        assert result.run_id == "demo1"
        assert result.model_name == "test"
        assert result.num_gold_samples == 0
        assert result.predictions == []
        assert result.metrics == {}
        assert result.stage6_report == {}
        assert result.succeeded is True
        assert result.error is None

    def test_failure(self):
        result = DemoResult(
            run_id="demo1",
            model_name="test",
            num_gold_samples=0,
            succeeded=False,
            error="something went wrong",
        )
        assert result.succeeded is False
        assert result.error == "something went wrong"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestStage11Config:
    def test_defaults(self):
        cfg = Stage11Config()
        assert cfg.base_model == BASE_MODEL
        assert cfg.model_name == DEFAULT_MODEL_NAME
        assert cfg.training_method == DEFAULT_TRAINING_METHOD
        assert cfg.lora_rank == DEFAULT_LORA_RANK
        assert cfg.quant_method is None
        assert cfg.quant_bit_width is None
        assert cfg.cwe_scope == CWE_SCOPE
        assert cfg.language == "python"
        assert cfg.training_data_size == DEFAULT_TRAINING_DATA_SIZE
        assert cfg.training_runs == []
        assert cfg.quant_results == []
        assert cfg.output_dir == DEFAULT_OUTPUT_DIR
        assert cfg.docs_dir == DEFAULT_DOCS_DIR

    def test_frozen(self):
        """Stage11Config is a frozen dataclass — attributes cannot be mutated."""
        cfg = Stage11Config()
        with pytest.raises(FrozenInstanceError):
            cfg.model_name = "new-name"

    def test_custom_values(self):
        cfg = Stage11Config(
            model_name="custom-model",
            lora_rank=128,
            quant_method="gguf",
            quant_bit_width=4,
            training_data_size=10000,
        )
        assert cfg.model_name == "custom-model"
        assert cfg.lora_rank == 128
        assert cfg.quant_method == "gguf"
        assert cfg.quant_bit_width == 4
        assert cfg.training_data_size == 10000

    def test_lora_rank_none_for_full_sft(self):
        cfg = Stage11Config(lora_rank=None, training_method="sft_full")
        assert cfg.lora_rank is None

    def test_cwe_scope_is_independent_copy(self):
        cfg = Stage11Config()
        cfg2 = Stage11Config()
        cfg.cwe_scope.append("CWE-999")
        assert "CWE-999" not in cfg2.cwe_scope


# ---------------------------------------------------------------------------
# Markdown generation tests
# ---------------------------------------------------------------------------


class TestGenerateModelCard:
    def test_contains_yaml_frontmatter(self):
        card = ModelCardData()
        md = generate_model_card_markdown(card)
        assert md.startswith("---")
        assert "title:" in md

    def test_contains_model_name(self):
        card = ModelCardData(model_name="my-custom-model")
        md = generate_model_card_markdown(card)
        assert "my-custom-model" in md

    def test_contains_cwe_scope(self):
        card = ModelCardData()
        md = generate_model_card_markdown(card)
        for cwe in CWE_SCOPE:
            assert cwe in md

    def test_contains_intended_use(self):
        card = ModelCardData(
            intended_use=["Test use 1", "Test use 2"],
        )
        md = generate_model_card_markdown(card)
        assert "Test use 1" in md
        assert "Test use 2" in md

    def test_contains_limitations(self):
        card = ModelCardData(
            limitations=["Limitation 1", "Limitation 2"],
        )
        md = generate_model_card_markdown(card)
        assert "Limitation 1" in md
        assert "Limitation 2" in md

    def test_contains_quantization(self):
        card = ModelCardData(
            quant_method="gguf",
            quant_bit_width=4,
        )
        md = generate_model_card_markdown(card)
        assert "gguf" in md
        assert "4-bit" in md

    def test_contains_evaluation_metrics(self):
        snap = EvalMetricsSnapshot(stage=6, run_id="test", base_model=BASE_MODEL,
                                  cwe_macro_f1=0.85)
        card = ModelCardData(metrics=snap)
        md = generate_model_card_markdown(card)
        assert "0.8500" in md or "0.850" in md


class TestGenerateTrainingReport:
    def test_contains_yaml_frontmatter(self):
        report = TrainingReportData()
        md = generate_training_report_markdown(report)
        assert md.startswith("---")
        assert "title:" in md

    def test_contains_model_name(self):
        report = TrainingReportData(model_name="my-report-model")
        md = generate_training_report_markdown(report)
        assert "my-report-model" in md

    def test_contains_conclusions(self):
        report = TrainingReportData(
            conclusions=["Conclusion 1", "Conclusion 2"],
        )
        md = generate_training_report_markdown(report)
        assert "Conclusion 1" in md
        assert "Conclusion 2" in md

    def test_contains_recommendations(self):
        report = TrainingReportData(
            recommendations=["Rec 1", "Rec 2"],
        )
        md = generate_training_report_markdown(report)
        assert "Rec 1" in md
        assert "Rec 2" in md

    def test_contains_training_runs_table(self):
        runs = [
            TrainingRunData(
                run_id="r1",
                method="sft_qlora",
                train_set_size=5000,
                train_time_minutes=120.0,
                peak_vram_gb=8.0,
                final_train_loss=0.05,
                final_val_loss=0.08,
            ),
        ]
        report = TrainingReportData(training_runs=runs)
        md = generate_training_report_markdown(report)
        assert "r1" in md
        assert "sft_qlora" in md

    def test_contains_quantization_matrix(self):
        quant = [
            QuantResultData(
                quant_method="gptq",
                bit_width=4,
                quantized_model_size_gb=3.5,
                estimated_vram_gb=5.0,
                status="completed",
            ),
        ]
        report = TrainingReportData(quant_results=quant)
        md = generate_training_report_markdown(report)
        assert "gptq" in md
        assert "GPTQ" in md or "gptq" in md


# ---------------------------------------------------------------------------
# Demo script generation tests
# ---------------------------------------------------------------------------


class TestGenerateDemoScript:
    def test_non_empty(self):
        script = generate_demo_script()
        assert isinstance(script, str)
        assert len(script) > 100

    def test_contains_docstring(self):
        script = generate_demo_script()
        assert "Stage 11" in script
        assert "mock" in script.lower()

    def test_contains_clirunner_invocation(self):
        """The demo should invoke CLI subcommands via CliRunner."""
        script = generate_demo_script()
        assert "CliRunner" in script
        assert "baseline" in script
        assert "stage6" in script
        assert "stage7" in script
        assert "stage10" in script

    def test_is_valid_python(self):
        """The demo script should be syntactically valid Python."""
        import ast

        script = generate_demo_script()
        ast.parse(script)

    def test_no_unicode_emojis(self):
        """Demo script must not contain emoji or arrow characters."""
        script = generate_demo_script()
        # These characters cause UnicodeEncodeError on Windows consoles.
        for char in ["✅", "❌", "⚠", "→", "—"]:
            assert char not in script, f"Found forbidden Unicode char {ord(char)} in demo script"


# ---------------------------------------------------------------------------
# Stage11Generator tests
# ---------------------------------------------------------------------------


class TestStage11GeneratorInit:
    def test_with_default_config(self):
        cfg = Stage11Config()
        gen = Stage11Generator(cfg)
        assert gen.config is cfg
        assert gen._run_id.startswith("stage11-")

    def test_with_custom_config(self):
        cfg = Stage11Config(model_name="custom")
        gen = Stage11Generator(cfg)
        assert gen._run_id.startswith("stage11-")


class TestEnsureDeliverables:
    def test_creates_all_three_files(self, tmp_path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)

        results = gen.ensure_deliverables()

        assert "model_card" in results
        assert "training_report" in results
        assert "demo_script" in results
        assert (docs_dir / "model_card.md").exists()
        assert (docs_dir / "training_report.md").exists()
        assert (docs_dir / "demo.py").exists()

    def test_creates_json_sidecars(self, tmp_path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        assert (output_dir / "model_card_data.json").exists()
        assert (output_dir / "training_report_data.json").exists()

    def test_model_card_non_empty(self, tmp_path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        content = (docs_dir / "model_card.md").read_text(encoding="utf-8")
        assert len(content) > 100
        assert "Model Card" in content

    def test_training_report_non_empty(self, tmp_path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        content = (docs_dir / "training_report.md").read_text(encoding="utf-8")
        assert len(content) > 100
        assert "Training Report" in content

    def test_json_sidecar_roundtrip(self, tmp_path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        mc_json = json.loads((output_dir / "model_card_data.json").read_text(encoding="utf-8"))
        assert mc_json["model_name"] == cfg.model_name
        assert mc_json["base_model"] == cfg.base_model

        tr_json = json.loads((output_dir / "training_report_data.json").read_text(encoding="utf-8"))
        assert tr_json["model_name"] == cfg.model_name


class TestValidateDeliverables:
    def test_all_present_returns_true(self, tmp_path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)
        gen.ensure_deliverables()

        assert gen.validate_deliverables() is True

    def test_missing_file_returns_false(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)

        # Create only model_card.md, leave others missing.
        (docs_dir / "model_card.md").write_text("# test", encoding="utf-8")
        assert gen.validate_deliverables() is False

    def test_empty_file_returns_false(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)

        # Create empty files.
        for name in ["model_card.md", "training_report.md", "demo.py"]:
            (docs_dir / name).write_text("", encoding="utf-8")
        assert gen.validate_deliverables() is False


class TestRunStage11:
    def test_creates_and_validates(self, tmp_path):
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))

        results = run_stage11(cfg)

        assert isinstance(results, dict)
        assert "model_card" in results
        assert "training_report" in results
        assert "demo_script" in results
        assert (docs_dir / "model_card.md").exists()

    def test_with_custom_model_name(self, tmp_path):
        """run_stage11 should create deliverables with custom config."""
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(
            docs_dir=str(docs_dir),
            output_dir=str(output_dir),
            model_name="custom-test-model",
            training_method="dpo",
        )
        results = run_stage11(cfg)

        assert "model_card" in results
        assert "training_report" in results
        assert "demo_script" in results

        mc_content = (docs_dir / "model_card.md").read_text(encoding="utf-8")
        assert "custom-test-model" in mc_content
        assert "dpo" in mc_content
