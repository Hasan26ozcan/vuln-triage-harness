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
from unittest.mock import MagicMock, patch

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
    _bool_str,
    _fmt_loss_history,
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
            checkpoint_uri="hf://vuln-triage-qwen2.5-coder-1.5b",
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
        assert card.model_name == "vuln-triage-qwen2.5-coder-1.5b"
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
        assert report.model_name == "vuln-triage-qwen2.5-coder-1.5b"
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


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestBoolStr:
    def test_true(self):
        assert _bool_str(True) == "yes"

    def test_false(self):
        assert _bool_str(False) == "no"


class TestFmtLossHistory:
    def test_empty_list(self):
        assert _fmt_loss_history([]) == "_no loss history available_"

    def test_short_list(self):
        result = _fmt_loss_history([0.5, 0.3, 0.1])
        assert "| # | loss |" in result
        assert "| 0 | 0.5000 |" in result
        assert "| 1 | 0.3000 |" in result
        assert "| 2 | 0.1000 |" in result

    def test_long_list_shows_first_and_last(self):
        """When losses exceed max_display, first 5 and last 5 are shown."""
        long_losses = [float(i) / 10 for i in range(30)]
        result = _fmt_loss_history(long_losses)
        # Should contain first 5 and last 5, plus the "..." row
        assert "| ... | ... |" in result
        assert "| 0 | 0.0000 |" in result
        assert "| 4 | 0.4000 |" in result
        # After the "..." separator, last 5 losses are re-numbered by display position
        assert "| 6 | 2.5000 |" in result
        assert "| 10 | 2.9000 |" in result

    def test_exactly_at_max_display(self):
        """A list of exactly max_display (20) items shows fully without "..."."""
        losses = [float(i) / 10 for i in range(20)]
        result = _fmt_loss_history(losses)
        assert "| ... | ... |" not in result
        assert "| 19 | 1.9000 |" in result

    def test_custom_max_display(self):
        """max_display parameter controls the threshold."""
        losses = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = _fmt_loss_history(losses, max_display=3)
        assert "| ... | ... |" in result


# ---------------------------------------------------------------------------
# Model card branch tests
# ---------------------------------------------------------------------------


class TestModelCardBranches:
    def test_with_forgetting_delta(self):
        """Lines 122-123: forgetting_delta in the metrics table."""
        snap = EvalMetricsSnapshot(
            stage=7, run_id="test", base_model=BASE_MODEL, forgetting_delta=-0.05
        )
        card = ModelCardData(metrics=snap)
        md = generate_model_card_markdown(card)
        assert "Forgetting delta" in md
        assert "-0.0500" in md

    def test_with_quantization_options(self):
        """Lines 128-140: full quantization options table."""
        card = ModelCardData(
            quant_method="gguf",
            quant_bit_width=4,
            quantization_options=[
                QuantResultData(
                    quant_method="gguf",
                    bit_width=4,
                    quantized_model_size_gb=3.5,
                    estimated_vram_gb=4.2,
                    tokens_per_sec=120.5,
                    model_cwe_macro_f1=0.78,
                ),
                QuantResultData(
                    quant_method="gptq",
                    bit_width=None,
                    quantized_model_size_gb=5.0,
                    estimated_vram_gb=6.0,
                    tokens_per_sec=None,
                    model_cwe_macro_f1=None,
                ),
            ],
        )
        md = generate_model_card_markdown(card)
        assert "## Quantization Options" in md
        assert "gguf" in md
        assert "4" in md  # bit_width
        assert "—" in md  # None values rendered as em-dash
        assert "120.5" in md

    def test_default_limitations(self):
        """When limitations is empty, default limitations are rendered."""
        card = ModelCardData()
        md = generate_model_card_markdown(card)
        assert "## Limitations" in md
        assert "out-of-scope CWEs" in md

    def test_default_ethical_considerations(self):
        """When ethical_considerations is empty, defaults are rendered."""
        card = ModelCardData()
        md = generate_model_card_markdown(card)
        assert "## Ethical Considerations" in md
        assert "research artifact" in md

    def test_default_out_of_scope(self):
        """When out_of_scope is empty, defaults are rendered."""
        card = ModelCardData()
        md = generate_model_card_markdown(card)
        assert "## Out of Scope" in md
        assert "Real-time repository monitoring" in md

    def test_fine_tuned_false(self):
        """When fine_tuned is False, _bool_str renders 'no'."""
        card = ModelCardData(fine_tuned=False)
        md = generate_model_card_markdown(card)
        assert "| no |" in md

    def test_lora_rank_none_omitted(self):
        """When lora_rank is None, the LoRA rank row is omitted."""
        card = ModelCardData(lora_rank=None, fine_tuned=True)
        md = generate_model_card_markdown(card)
        assert "LoRA rank" not in md

    def test_quant_bit_width_omitted_without_quant_method(self):
        """When quant_bit_width is set but quant_method is None, no quant rows."""
        card = ModelCardData(quant_method=None, quant_bit_width=4)
        md = generate_model_card_markdown(card)
        assert "Quantized" not in md
        assert "4-bit" not in md


# ---------------------------------------------------------------------------
# Training report branch tests
# ---------------------------------------------------------------------------


class TestTrainingReportBranches:
    def test_with_hyperparams(self):
        """Lines 290-293: hyperparameters table when present."""
        run = TrainingRunData(
            run_id="r1",
            method="sft_qlora",
            hyperparams={"lr": 0.001, "epochs": 3},
        )
        report = TrainingReportData(training_runs=[run])
        md = generate_training_report_markdown(report)
        assert "| Parameter | Value |" in md
        assert "`lr`" in md

    def test_without_hyperparams(self):
        """Line 295: '_No hyperparameters recorded._' when hyperparams is empty."""
        run = TrainingRunData(run_id="r1", method="sft_qlora")
        report = TrainingReportData(training_runs=[run])
        md = generate_training_report_markdown(report)
        assert "No hyperparameters recorded" in md

    def test_with_loss_history(self):
        """Lines 298-301: loss history table when train_loss_history is non-empty."""
        run = TrainingRunData(
            run_id="r1",
            method="sft_qlora",
            train_loss_history=[0.5, 0.3, 0.1],
        )
        report = TrainingReportData(training_runs=[run])
        md = generate_training_report_markdown(report)
        assert "#### Loss history" in md
        assert "| 0 | 0.5000 |" in md

    def test_with_baseline_metrics(self):
        """Lines 308-318: baseline metrics section."""
        bm = EvalMetricsSnapshot(stage=4, run_id="baseline", base_model=BASE_MODEL,
                                cwe_macro_f1=0.7, severity_accuracy=0.8)
        report = TrainingReportData(baseline_metrics=bm)
        md = generate_training_report_markdown(report)
        assert "### Stage 4 — Pre-fine-tuning Baseline" in md
        assert "0.7000" in md

    def test_with_tuned_metrics_no_per_class(self):
        """Lines 321-331: tuned metrics section without per_class table."""
        tm = EvalMetricsSnapshot(stage=6, run_id="tuned", base_model=BASE_MODEL,
                                cwe_macro_f1=0.85)
        report = TrainingReportData(tuned_metrics=tm)
        md = generate_training_report_markdown(report)
        assert "### Stage 6 — Tuned Model Four-Tier Evaluation" in md
        assert "0.8500" in md

    def test_with_tuned_metrics_and_per_class(self):
        """Lines 332-341: per_class table rendered when populated."""
        tm = EvalMetricsSnapshot(
            stage=6, run_id="tuned", base_model=BASE_MODEL,
            cwe_macro_f1=0.85,
            per_class={
                "CWE-89": {"precision": 0.90, "recall": 0.85, "f1": 0.87},
                "CWE-79": {"precision": 0.78, "recall": 0.80, "f1": 0.79},
            },
        )
        report = TrainingReportData(tuned_metrics=tm)
        md = generate_training_report_markdown(report)
        assert "| CWE | Precision | Recall | F1 |" in md
        assert "CWE-89" in md
        assert "0.8700" in md

    def test_regression_report_with_positive_delta(self):
        """Lines 344-358: regression report with forgetting_delta >= 0 => [OK]."""
        rr = EvalMetricsSnapshot(
            stage=7, run_id="reg", base_model=BASE_MODEL,
            forgetting_delta=0.02,
        )
        report = TrainingReportData(regression_report=rr)
        md = generate_training_report_markdown(report)
        assert "### Stage 7 — Regression / Forgetting Analysis" in md
        assert "[OK] No forgetting" in md
        assert "0.0200" in md

    def test_regression_report_with_negative_delta(self):
        """Forgetting delta < 0 => [WARN] Forgetting detected."""
        rr = EvalMetricsSnapshot(
            stage=7, run_id="reg", base_model=BASE_MODEL,
            forgetting_delta=-0.03,
        )
        report = TrainingReportData(regression_report=rr)
        md = generate_training_report_markdown(report)
        assert "[WARN] Forgetting detected" in md
        assert "-0.0300" in md

    def test_regression_report_no_forgetting_delta(self):
        """Forgetting delta is None => 'N/A' and [WARN] (since delta >= 0 is False)."""
        rr = EvalMetricsSnapshot(
            stage=7, run_id="reg", base_model=BASE_MODEL,
            forgetting_delta=None,
        )
        report = TrainingReportData(regression_report=rr)
        md = generate_training_report_markdown(report)
        assert "N/A" in md
        assert "[WARN] Forgetting detected" in md

    def test_with_quant_results(self):
        """Lines 361-375: quantization matrix with various field combinations."""
        report = TrainingReportData(quant_results=[
            QuantResultData(
                quant_method="gptq", bit_width=4,
                quantized_model_size_gb=3.5, estimated_vram_gb=5.0,
                tokens_per_sec=120.5, model_cwe_macro_f1=0.78,
                exec_pass_rate=0.65, status="completed",
            ),
            QuantResultData(
                quant_method="awq", bit_width=None,
                quantized_model_size_gb=4.0, estimated_vram_gb=6.0,
                tokens_per_sec=None, model_cwe_macro_f1=None,
                exec_pass_rate=None, status="failed",
            ),
        ])
        md = generate_training_report_markdown(report)
        assert "## Stage 8 — Quantization Matrix" in md
        assert "gptq" in md
        assert "awq" in md
        assert "0.6500" in md  # exec_pass_rate
        assert "—" in md  # None values

    def test_with_gate_result_pass(self):
        """Lines 379-393: gate result table rendered when present (pass)."""
        report = TrainingReportData(
            gate_result={
                "status": "pass",
                "checks": [
                    {"name": "f1_gate", "status": "pass", "message": "CWE F1 >= 0.5"},
                    {"name": "exec_gate", "status": "pass", "message": "Exec pass >= 0.4"},
                ],
            }
        )
        md = generate_training_report_markdown(report)
        assert "## Stage 10 — Regression Gate" in md
        assert "[PASS]" in md
        assert "f1_gate" in md
        assert "exec_gate" in md

    def test_with_gate_result_fail(self):
        """Gate result with 'fail' status renders [FAIL]."""
        report = TrainingReportData(
            gate_result={
                "status": "fail",
                "checks": [
                    {"name": "f1_gate", "status": "fail", "message": "CWE F1 < 0.5"},
                ],
            }
        )
        md = generate_training_report_markdown(report)
        assert "[FAIL]" in md
        assert "f1_gate" in md

    def test_with_conclusions_and_recommendations(self):
        """Lines 396-409: conclusions and recommendations sections."""
        report = TrainingReportData(
            conclusions=["Conclusion A", "Conclusion B"],
            recommendations=["Rec 1", "Rec 2"],
        )
        md = generate_training_report_markdown(report)
        assert "## Conclusions" in md
        assert "Conclusion A" in md
        assert "## Recommendations" in md
        assert "Rec 1" in md


# ---------------------------------------------------------------------------
# Stage11Generator._model_card_data tests
# ---------------------------------------------------------------------------


class TestModelCardDataGeneration:
    def test_with_quantization_config(self):
        """_model_card_data propagates quant_method and quant_bit_width."""
        cfg = Stage11Config(quant_method="gguf", quant_bit_width=4)
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert data.quant_method == "gguf"
        assert data.quant_bit_width == 4

    def test_with_lora_rank_none(self):
        """_model_card_data passes lora_rank=None through."""
        cfg = Stage11Config(lora_rank=None, training_method="sft_full")
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert data.lora_rank is None

    def test_with_tuned_metrics(self):
        """_model_card_data uses tuned_metrics when available."""
        snap = EvalMetricsSnapshot(stage=6, run_id="tuned", base_model=BASE_MODEL,
                                    cwe_macro_f1=0.9)
        cfg = Stage11Config(tuned_metrics=snap)
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert data.metrics.cwe_macro_f1 == 0.9

    def test_with_baseline_only_metrics(self):
        """When tuned_metrics is None, baseline_metrics is used."""
        snap = EvalMetricsSnapshot(stage=4, run_id="base", base_model=BASE_MODEL,
                                    cwe_macro_f1=0.7)
        cfg = Stage11Config(baseline_metrics=snap)
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert data.metrics.stage == 4

    def test_no_tuned_no_baseline_metrics(self):
        """Without any metrics, _model_card_data creates a default snapshot."""
        cfg = Stage11Config()
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert data.metrics.cwe_macro_f1 == 0.0
        assert data.metrics.stage == 6


# ---------------------------------------------------------------------------
# Stage11Generator._training_report_data tests
# ---------------------------------------------------------------------------


class TestTrainingReportDataGeneration:
    def test_without_training_runs(self):
        """When training_runs is empty, default conclusions and recommendations
        are used (lines 659-670, 683-688)."""
        cfg = Stage11Config()
        gen = Stage11Generator(cfg)
        data = gen._training_report_data()
        assert len(data.conclusions) > 0
        assert len(data.recommendations) > 0
        assert "No real training runs" in data.conclusions[0]

    def test_with_training_runs(self):
        """When training_runs is present, _conclusions_from_runs is called
        (lines 657, 673) and real-run recommendations are used."""
        snap = EvalMetricsSnapshot(stage=6, run_id="tuned", base_model=BASE_MODEL,
                                    cwe_macro_f1=0.9)
        bm = EvalMetricsSnapshot(stage=4, run_id="base", base_model=BASE_MODEL,
                                  cwe_macro_f1=0.7)
        run = TrainingRunData(run_id="r1", method="sft_qlora", final_train_loss=0.1,
                              final_val_loss=0.15)
        cfg = Stage11Config(training_runs=[run], tuned_metrics=snap, baseline_metrics=bm)
        gen = Stage11Generator(cfg)
        data = gen._training_report_data()
        assert "r1" in data.conclusions[0]
        assert any("Scale to a larger training dataset" in r for r in data.recommendations)


# ---------------------------------------------------------------------------
# Stage11Generator._conclusions_from_runs tests
# ---------------------------------------------------------------------------


class TestConclusionsFromRuns:
    def test_val_loss_present(self):
        run = TrainingRunData(run_id="r1", method="sft_qlora",
                              final_train_loss=0.1, final_val_loss=0.15)
        cfg = Stage11Config(training_runs=[run])
        gen = Stage11Generator(cfg)
        conclusions = gen._conclusions_from_runs()
        assert any("val loss = 0.1500" in c for c in conclusions)

    def test_val_loss_absent(self):
        run = TrainingRunData(run_id="r2", method="sft_full",
                              final_train_loss=0.1, final_val_loss=None)
        cfg = Stage11Config(training_runs=[run])
        gen = Stage11Generator(cfg)
        conclusions = gen._conclusions_from_runs()
        assert any("No validation loss was recorded" in c for c in conclusions)

    def test_method_empty_uses_unknown(self):
        run = TrainingRunData(run_id="r3", method="", final_train_loss=0.1)
        cfg = Stage11Config(training_runs=[run])
        gen = Stage11Generator(cfg)
        conclusions = gen._conclusions_from_runs()
        assert any("unknown" in c for c in conclusions)

    def test_tuned_metrics_low_f1(self):
        """When tuned cwe_macro_f1 < 0.5, an extra conclusion is appended."""
        run = TrainingRunData(run_id="r1", method="sft_qlora", final_train_loss=0.1,
                              final_val_loss=0.15)
        snap = EvalMetricsSnapshot(stage=6, run_id="t", base_model=BASE_MODEL,
                                   cwe_macro_f1=0.3)
        cfg = Stage11Config(training_runs=[run], tuned_metrics=snap)
        gen = Stage11Generator(cfg)
        conclusions = gen._conclusions_from_runs()
        assert any("low" in c.lower() for c in conclusions)

    def test_tuned_metrics_high_f1(self):
        """When tuned cwe_macro_f1 >= 0.5, the low-F1 extra conclusion is omitted."""
        run = TrainingRunData(run_id="r1", method="sft_qlora", final_train_loss=0.1,
                              final_val_loss=0.15)
        snap = EvalMetricsSnapshot(stage=6, run_id="t", base_model=BASE_MODEL,
                                   cwe_macro_f1=0.8)
        cfg = Stage11Config(training_runs=[run], tuned_metrics=snap)
        gen = Stage11Generator(cfg)
        conclusions = gen._conclusions_from_runs()
        # Should NOT contain the low-F1 warning
        assert not any("low" in c.lower() for c in conclusions)
        # But should contain the Stage 6 eval summary
        assert any("CWE Macro-F1" in c for c in conclusions)

    def test_baseline_metrics_present(self):
        run = TrainingRunData(run_id="r1", method="sft_qlora", final_train_loss=0.1,
                              final_val_loss=0.15)
        bm = EvalMetricsSnapshot(stage=4, run_id="b", base_model=BASE_MODEL,
                                  cwe_macro_f1=0.6)
        cfg = Stage11Config(training_runs=[run], baseline_metrics=bm)
        gen = Stage11Generator(cfg)
        conclusions = gen._conclusions_from_runs()
        assert any("Pre-fine-tuning baseline" in c for c in conclusions)

    def test_no_metrics(self):
        """With only training runs, no metric conclusions are added."""
        run = TrainingRunData(run_id="r1", method="sft_qlora", final_train_loss=0.1,
                              final_val_loss=0.15)
        cfg = Stage11Config(training_runs=[run])
        gen = Stage11Generator(cfg)
        conclusions = gen._conclusions_from_runs()
        # Only the run conclusion, no metric conclusions
        assert len(conclusions) == 1


# ---------------------------------------------------------------------------
# Stage11Generator._model_card_data full integration
# ---------------------------------------------------------------------------


class TestModelCardDataFull:
    def test_intended_use_populated(self):
        cfg = Stage11Config()
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert len(data.intended_use) == 4
        assert any("CWE category" in u for u in data.intended_use)

    def test_limitations_populated(self):
        cfg = Stage11Config()
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert len(data.limitations) == 6
        assert any("research artifact" not in lim for lim in data.limitations)

    def test_ethical_considerations_populated(self):
        cfg = Stage11Config()
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert len(data.ethical_considerations) == 3

    def test_out_of_scope_populated(self):
        cfg = Stage11Config()
        gen = Stage11Generator(cfg)
        data = gen._model_card_data()
        assert len(data.out_of_scope) == 5


# ---------------------------------------------------------------------------
# Stage11Generator.run_demo tests
# ---------------------------------------------------------------------------


class TestRunDemo:
    def test_run_demo_success(self, tmp_path):
        """Lines 825-942: full mock-mode demo pipeline succeeds."""
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)

        mock_backend = MagicMock()
        mock_backend_cls = MagicMock(return_value=mock_backend)

        # Baseline result
        mock_baseline_result = MagicMock()
        mock_baseline_result.predictions = [
            MagicMock(sample_id="s1", predicted_cwe="CWE-89", predicted_severity="high"),
            MagicMock(sample_id="s2", predicted_cwe="CWE-79", predicted_severity="low"),
        ]
        mock_baseline_result.metrics.cwe_macro_f1 = 0.5

        # Eval report
        mock_eval_report = MagicMock()
        mock_eval_report.metrics.model_cwe_macro_f1 = 0.85
        mock_eval_report.metrics.exec_pass_rate = 0.90
        mock_eval_report.metrics.hallucination_rate = 0.05
        mock_eval_report.metrics.avg_patch_coverage = 0.95
        mock_eval_report.model_dump.return_value = {"test": True}
        mock_eval_report.model_dump_json.return_value = '{"test": true}'

        # Regression report
        mock_regression_report = MagicMock()
        mock_regression_report.forgetting_delta = 0.01
        mock_regression_report.model_dump_json.return_value = '{"fd": 0.01}'

        # Gate result
        mock_gate_result = MagicMock()
        mock_gate_result.passed = True
        mock_gate_result.status.value = "pass"

        with (
            patch("app.ci.gate.run_gate", return_value=mock_gate_result),
            patch("app.evaluation.baseline.run_baseline",
                  return_value=mock_baseline_result),
            patch("app.evaluation.backends.MockBackend", mock_backend_cls),
            patch("app.evaluation.runner.EvaluationRunner") as mock_runner_cls,
            patch("app.evaluation.runner.load_samples", return_value=["s1", "s2"]),
            patch("app.evaluation.runner.load_predictions", return_value=[]),
            patch("app.evaluation.general_capability.run_regression_analysis",
                  return_value=mock_regression_report),
            patch("app.evaluation.general_capability.MockCodeTestRunner") as mock_code_runner_cls,
        ):
            mock_runner = MagicMock()
            mock_runner.run.return_value = mock_eval_report
            mock_runner_cls.return_value = mock_runner
            mock_code_runner_cls.return_value = MagicMock()

            result = gen.run_demo()

        assert isinstance(result, DemoResult)
        assert result.succeeded is True
        assert result.num_gold_samples == 2
        assert len(result.predictions) == 2
        assert result.predictions[0]["sample_id"] == "s1"
        assert "gate_status" in result.metrics
        assert result.metrics["gate_status"] == "pass"
        assert result.metrics["base_cwe_macro_f1"] == 0.5
        assert result.metrics["tuned_cwe_macro_f1"] == 0.85
        assert result.metrics["forgetting_delta"] == 0.01
        assert result.succeeded is True
        assert result.stage6_report == {"test": True}

    def test_run_demo_stage4_failure(self, tmp_path):
        """If run_baseline raises, the except block catches it (line 944)."""
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))
        gen = Stage11Generator(cfg)

        with patch("app.evaluation.baseline.run_baseline",
                    side_effect=RuntimeError("baseline crashed")):
            result = gen.run_demo()

        assert isinstance(result, DemoResult)
        assert result.succeeded is False
        assert "baseline crashed" in result.error
        assert result.num_gold_samples == 0


# ---------------------------------------------------------------------------
# run_stage11 validation failure test
# ---------------------------------------------------------------------------


class TestRunStage11ValidationFailure:
    def test_validation_failure_raises(self, tmp_path):
        """Line 964: run_stage11 raises RuntimeError when validation fails."""
        docs_dir = tmp_path / "docs"
        output_dir = tmp_path / "output" / "stage11"
        cfg = Stage11Config(docs_dir=str(docs_dir), output_dir=str(output_dir))

        with patch.object(Stage11Generator, "validate_deliverables", return_value=False):
            with pytest.raises(RuntimeError, match="Stage 11 deliverables validation failed"):
                run_stage11(cfg)
