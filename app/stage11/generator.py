"""Stage 11 - documentation & interview deliverable generator.

This module turns ``Stage11Config`` + evaluation/training artefacts into the
three Stage 11 deliverables:

1. ``docs/model_card.md`` - human-readable model card
2. ``docs/training_report.md`` - technical training report
3. ``docs/demo.py`` - runnable demo script

The generator is designed to work **without** a GPU, without a model
download, and without network access - it uses mock mode (the same pattern as
Stage 4-10 mock backends) so CI can validate the deliverables on every push.

Usage (CLI)::

    python -m app.evaluation.cli stage11 --docs-dir docs --output-dir ./output/stage11

Usage (programmatic)::

    from app.stage11.generator import Stage11Generator

    gen = Stage11Generator(config)
    gen.ensure_deliverables()   # creates all docs/ artefacts on disk
    assert gen.validate_deliverables()  # verifies they exist and are well-formed
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path

from app.schemas.documentation import (
    BASE_MODEL,
    DemoResult,
    EvalMetricsSnapshot,
    ModelCardData,
    QuantResultData,
    TrainingReportData,
    TrainingRunData,
)
from app.stage11.config import Stage11Config

logger = logging.getLogger(__name__)

# Markdown table separator / header literals reused across generators.
_MD_TABLE_SEP = "|---|---|"
_MD_METRIC_HEADER = "| Metric | Value |"
_MD_RUN_TABLE_SEP = "|---|---|---|---|---|---|---|"

# File-name constants for artifact loading.
_STAGE6_REPORT_NAME = "eval_report.json"
_STAGE7_REPORT_NAME = "regression_report.json"

# Reusable log message format for artifact parse failures.
_PARSE_WARN = "Could not parse %s: %s"


def _fmt_severity(value: float | None) -> str:
    """Render severity accuracy, distinguishing "not scored" from 0.0."""
    if value is None:
        return "N/A (not scored at this stage)"
    return f"{value:.4f}"


def _bool_str(b: bool) -> str:
    """Render a bool as ``yes`` / ``no`` for Markdown tables."""
    return "yes" if b else "no"


def _model_card_metadata(data: ModelCardData) -> list[str]:
    """YAML front matter + title (low complexity)."""
    return [
        "---",
        f'title: "{data.model_name} — Vulnerability Triage Model Card"',
        f'date: "{data.generated_at}"',
        f"base_model: {data.base_model}",
        f"training_method: {data.training_method}",
        "license: mit",
        "tags:",
        "  - code-generation",
        "  - vulnerability-detection",
        "  - security",
        "---",
        "",
        f"# Model Card: {data.model_name}",
        "",
    ]


def _model_card_details(data: ModelCardData) -> list[str]:
    """Model Details table with conditional rows (low complexity)."""
    lines: list[str] = ["## Model Details", "", "| Field | Value |", _MD_TABLE_SEP]
    lines.append(f"| Model name | `{data.model_name}` |")
    lines.append(f"| Base model | `{data.base_model}` |")
    lines.append(f"| Fine-tuned | {_bool_str(data.fine_tuned)} |")
    lines.append(f"| Training method | {data.training_method} |")
    if data.lora_rank is not None:
        lines.append(f"| LoRA rank | {data.lora_rank} |")
    if data.quant_method:
        lines.append(f"| Quantized | {data.quant_method} |")
        if data.quant_bit_width:
            lines.append(f"| Quant bit width | {data.quant_bit_width}-bit |")
    lines.append(f"| Language | {data.language} |")
    lines.append(f"| CWE scope | {', '.join(data.cwe_scope)} |")
    if data.training_data_size:
        lines.append(f"| Training data size | {data.training_data_size:,} samples |")
    lines.append("")
    return lines


def _model_card_intended_use(data: ModelCardData) -> list[str]:
    """Intended Use section with fallback defaults."""
    lines: list[str] = ["## Intended Use", ""]
    if data.intended_use:
        for u in data.intended_use:
            lines.append(f"- {u}")
    else:
        lines.extend(_DEFAULT_INTENDED_USE)
    lines.append("")
    return lines


def _model_card_evaluation(data: ModelCardData) -> list[str]:
    """Evaluation metrics table."""
    m = data.metrics
    lines: list[str] = ["## Evaluation", "", _MD_METRIC_HEADER, _MD_TABLE_SEP]
    lines.append(f"| Stage | {m.stage} |")
    lines.append(f"| CWE Macro-F1 | {m.cwe_macro_f1:.4f} |")
    lines.append(f"| Severity accuracy | {_fmt_severity(m.severity_accuracy)} |")
    lines.append(f"| Hallucination rate | {m.hallucination_rate:.4f} |")
    lines.append(f"| Patch coverage | {m.patch_coverage:.4f} |")
    lines.append(f"| Exec pass rate | {m.exec_pass_rate:.4f} |")
    if m.forgetting_delta is not None:
        lines.append(f"| Forgetting delta | {m.forgetting_delta:+.4f} |")
    lines.append("")
    return lines


_DEFAULT_INTENDED_USE = [
    "- Classifying the CWE category of a vulnerable code snippet.",
    "- Suggesting a minimal, working patch for the vulnerability.",
    "- Batch analysis of code repositories for triage prioritization.",
]

_DEFAULT_LIMITATIONS = [
    "- Trained on {n} CWE classes; out-of-scope CWEs are treated as hallucinations.",
    "- Not a general-purpose scanner - does not detect logic bugs or configuration issues.",
    "- The exec-based evaluation runs proposed patches in a sandboxed subprocess.",
]

_DEFAULT_ETHICAL = [
    "- This model is a research artifact, not a production SOC tool.",
    "- Proposed patches should be reviewed by a human before merging.",
]

_DEFAULT_OUT_OF_SCOPE = [
    "- Real-time repository monitoring in CI pipelines.",
    "- Network-based vulnerability scanning (no port scanning, no HTTP fuzzing).",
    "- Supply-chain security / third-party dependency auditing.",
    "- Legal or compliance assessment of software.",
]


def _model_card_quantization(data: ModelCardData) -> list[str]:
    """Quantization options table (only when present)."""
    lines: list[str] = []
    if not data.quantization_options:
        return lines
    lines.extend(
        [
            "## Quantization Options",
            "",
            "| Method | Bits | Size (GB) | VRAM (GB) | Tokens/s | CWE-F1 |",
            _MD_RUN_TABLE_SEP,
        ]
    )
    for q in data.quantization_options:
        bits = str(q.bit_width) if q.bit_width is not None else "—"
        f1 = f"{q.model_cwe_macro_f1:.4f}" if q.model_cwe_macro_f1 is not None else "—"
        tps = f"{q.tokens_per_sec:.1f}" if q.tokens_per_sec is not None else "—"
        lines.append(
            f"| {q.quant_method} | {bits} | {q.quantized_model_size_gb:.2f} | "
            f"{q.estimated_vram_gb:.2f} | {tps} | {f1} |"
        )
    lines.append("")
    return lines


def _model_card_serving(data: ModelCardData) -> list[str]:
    """Serving backends list."""
    lines: list[str] = ["## Serving", "", "The model can be served air-gapped via:", ""]
    for b in data.serving_backends:
        lines.append(f"- `{b}`")
    lines.append("")
    return lines


def _model_card_limitations(data: ModelCardData) -> list[str]:
    """Limitations section with fallback defaults."""
    lines: list[str] = ["## Limitations", ""]
    if data.limitations:
        for lim in data.limitations:
            lines.append(f"- {lim}")
    else:
        lines.append(_DEFAULT_LIMITATIONS[0].format(n=len(data.cwe_scope)))
        lines.extend(_DEFAULT_LIMITATIONS[1:])
    lines.append("")
    return lines


def _model_card_ethical(data: ModelCardData) -> list[str]:
    """Ethical Considerations section with fallback defaults."""
    lines: list[str] = ["## Ethical Considerations", ""]
    if data.ethical_considerations:
        for ec in data.ethical_considerations:
            lines.append(f"- {ec}")
    else:
        lines.extend(_DEFAULT_ETHICAL)
    lines.append("")
    return lines


def _model_card_scope(data: ModelCardData) -> list[str]:
    """Out of Scope section with fallback defaults."""
    lines: list[str] = ["## Out of Scope", ""]
    if data.out_of_scope:
        for oos in data.out_of_scope:
            lines.append(f"- {oos}")
    else:
        lines.extend(_DEFAULT_OUT_OF_SCOPE)
    lines.append("")
    return lines


def _model_card_citation() -> list[str]:
    """Static citation block."""
    return [
        "## Citation",
        "",
        "If you use this model in your research, please cite:",
        "",
        "```",
        "@misc{vuln-triage-harness,",
        "  title={Vulnerability Triage & Patch-Suggestion Harness},",
        "  author={Ozcan, Hasan},",
        "  year={2026},",
        "  url={https://github.com/Hasan26ozcan/vuln-triage-harness}",
        "}",
        "```",
        "",
    ]


def generate_model_card_markdown(data: ModelCardData) -> str:
    """Render a ``ModelCardData`` instance into a Markdown model card.

    The output follows the structure documented in the project README and
    mirrors the HuggingFace model-card convention (metadata YAML front
    matter + Markdown sections).
    """
    lines: list[str] = []
    lines.extend(_model_card_metadata(data))
    lines.extend(_model_card_details(data))
    lines.extend(_model_card_intended_use(data))
    lines.extend(_model_card_evaluation(data))
    lines.extend(_model_card_quantization(data))
    lines.extend(_model_card_serving(data))
    lines.extend(_model_card_limitations(data))
    lines.extend(_model_card_ethical(data))
    lines.extend(_model_card_scope(data))
    lines.extend(_model_card_citation())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Training-report markdown generation
# ---------------------------------------------------------------------------


def _fmt_loss_history(losses: list[float], max_display: int = 20) -> str:
    """Render a loss-history list as a compact table."""
    if not losses:
        return "_no loss history available_"
    step_count = len(losses)
    lines = ["| # | loss |", "|---|------|"]
    if step_count <= max_display:
        for i, loss in enumerate(losses):
            lines.append(f"| {i} | {loss:.4f} |")
    else:
        # Show first 5 and last 5 with a "..." separator line.
        # Display positions: 0–4 for first 5, 5 for "...", 6–10 for last 5.
        for i, loss in enumerate(losses[:5]):
            lines.append(f"| {i} | {loss:.4f} |")
        lines.append("| ... | ... |")
        for i, loss in enumerate(losses[-5:]):
            lines.append(f"| {i + 6} | {loss:.4f} |")
    return "\n".join(lines)


_MD_RUN_TABLE_HEADER = (
    "| Run ID | Method | Train set | Time (min) | VRAM (GB) | Train Loss | Val Loss |"
)
_MD_QUANT_TABLE_HEADER = "| Method | Bits | Size (GB) | VRAM (GB) | Tokens/s | CWE-F1 | Exec pass |"
_MD_PER_CLASS_TABLE_SEP = "|---|---|---|---|"


def _training_report_overview(data: TrainingReportData) -> list[str]:
    """YAML front matter + overview table."""
    num_runs = len(data.training_runs)
    return [
        "---",
        f'title: "{data.model_name} - Training Report"',
        f'date: "{data.generated_at}"',
        f"base_model: {data.base_model}",
        "license: mit",
        "---",
        "",
        f"# Training Report: {data.model_name}",
        "",
        f"_Generated: {data.generated_at}_",
        "",
        "## Overview",
        "",
        "| Field | Value |",
        _MD_TABLE_SEP,
        f"| Model | `{data.model_name}` |",
        f"| Base model | `{data.base_model}` |",
        f"| Report ID | `{data.report_id or 'auto'}` |",
        f"| Training runs | {num_runs} |",
        "",
    ]


def _training_report_runs(data: TrainingReportData) -> list[str]:
    """Training runs summary + hyperparameters + loss history per run."""
    lines: list[str] = []
    if not data.training_runs:
        return lines
    lines.append("## Training Runs")
    lines.append("")
    lines.append(_MD_RUN_TABLE_HEADER)
    lines.append(_MD_RUN_TABLE_SEP)
    for run in data.training_runs:
        vl = f"{run.final_val_loss:.4f}" if run.final_val_loss is not None else "—"
        lines.append(
            f"| `{run.run_id}` | {run.method} | {run.train_set_size} | "
            f"{run.train_time_minutes:.1f} | {run.peak_vram_gb:.1f} | "
            f"{run.final_train_loss:.4f} | {vl} |"
        )
    lines.append("")
    # Detailed hyperparameters + loss history for each run.
    for run in data.training_runs:
        lines.append(f"**Run `{run.run_id}` ({run.method})**")
        lines.append("")
        if run.hyperparams:
            lines.append("| Parameter | Value |")
            lines.append(_MD_TABLE_SEP)
            for k, v in sorted(run.hyperparams.items()):
                lines.append(f"| `{k}` | {v} |")
        else:
            lines.append("_No hyperparameters recorded._")
        lines.append("")
        if run.train_loss_history:
            lines.append("#### Loss history")
            lines.append("")
            lines.append(_fmt_loss_history(run.train_loss_history))
            lines.append("")
    return lines


def _training_report_evaluation(data: TrainingReportData) -> list[str]:
    """Evaluation Results: baseline, tuned, and regression sub-sections."""
    lines: list[str] = ["## Evaluation Results", ""]
    if not (data.baseline_metrics or data.tuned_metrics or data.regression_report):
        return lines
    if data.baseline_metrics:
        bm = data.baseline_metrics
        lines.extend(
            [
                "### Stage 4 — Pre-fine-tuning Baseline",
                "",
                _MD_METRIC_HEADER,
                _MD_TABLE_SEP,
            ]
        )
        lines.append(f"| Run ID | `{bm.run_id}` |")
        lines.append(f"| CWE Macro-F1 | {bm.cwe_macro_f1:.4f} |")
        lines.append(f"| Severity accuracy | {_fmt_severity(bm.severity_accuracy)} |")
        lines.append(f"| Hallucination rate | {bm.hallucination_rate:.4f} |")
        lines.append(f"| Patch coverage | {bm.patch_coverage:.4f} |")
        lines.append("")
    if data.tuned_metrics:
        tm = data.tuned_metrics
        lines.extend(
            [
                "### Stage 6 — Tuned Model Four-Tier Evaluation",
                "",
                _MD_METRIC_HEADER,
                _MD_TABLE_SEP,
            ]
        )
        lines.append(f"| Run ID | `{tm.run_id}` |")
        lines.append(f"| CWE Macro-F1 | {tm.cwe_macro_f1:.4f} |")
        lines.append(f"| Severity accuracy | {_fmt_severity(tm.severity_accuracy)} |")
        lines.append(f"| Hallucination rate | {tm.hallucination_rate:.4f} |")
        lines.append(f"| Patch coverage | {tm.patch_coverage:.4f} |")
        lines.append(f"| Exec pass rate | {tm.exec_pass_rate:.4f} |")
        if tm.per_class:
            lines.append("")
            lines.append("| CWE | Precision | Recall | F1 |")
            lines.append(_MD_PER_CLASS_TABLE_SEP)
            for cwe, stats in sorted(tm.per_class.items()):
                p = stats.get("precision", 0.0)
                r = stats.get("recall", 0.0)
                f1 = stats.get("f1", 0.0)
                lines.append(f"| {cwe} | {p:.4f} | {r:.4f} | {f1:.4f} |")
        lines.append("")
    if data.regression_report:
        rr = data.regression_report
        lines.extend(["### Stage 7 — Regression / Forgetting Analysis", ""])
        delta = rr.forgetting_delta
        delta_status = (
            "[OK] No forgetting"
            if (delta is not None and delta >= 0)
            else "[WARN] Forgetting detected"
        )
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        lines.extend([_MD_METRIC_HEADER, _MD_TABLE_SEP])
        lines.append(f"| Forgetting delta | {delta_str} |")
        lines.append(f"| Status | {delta_status} |")
        lines.append("")
    return lines


def _training_report_quantization(data: TrainingReportData) -> list[str]:
    """Stage 8 quantization matrix table."""
    lines: list[str] = []
    if not data.quant_results:
        return lines
    lines.append("## Stage 8 — Quantization Matrix")
    lines.append("")
    lines.append(_MD_QUANT_TABLE_HEADER)
    lines.append(_MD_RUN_TABLE_SEP)
    for q in data.quant_results:
        bits = str(q.bit_width) if q.bit_width is not None else "—"
        tps = f"{q.tokens_per_sec:.1f}" if q.tokens_per_sec is not None else "—"
        f1_str = f"{q.model_cwe_macro_f1:.4f}" if q.model_cwe_macro_f1 is not None else "—"
        er = f"{q.exec_pass_rate:.4f}" if q.exec_pass_rate is not None else "—"
        lines.append(
            f"| {q.quant_method} | {bits} | {q.quantized_model_size_gb:.2f} | "
            f"{q.estimated_vram_gb:.2f} | {tps} | {f1_str} | {er} |"
        )
    lines.append("")
    return lines


def _training_report_gate(data: TrainingReportData) -> list[str]:
    """Stage 10 regression gate status table."""
    lines: list[str] = []
    if not data.gate_result:
        return lines
    lines.extend(["## Stage 10 — Regression Gate", ""])
    status = data.gate_result.get("status", "unknown")
    icon = "[PASS]" if status == "pass" else "[FAIL]"
    lines.append(f"**Overall status: {icon}**")
    lines.append("")
    lines.append("| Check | Status | Message |")
    lines.append("|---|---|---|")
    checks = data.gate_result.get("checks", [])
    for c in checks:
        name = c.get("name", "?")
        cstatus = c.get("status", "?")
        msg = c.get("message", "")
        lines.append(f"| {name} | {cstatus} | {msg} |")
    lines.append("")
    return lines


def _training_report_conclusions(data: TrainingReportData) -> list[str]:
    """Conclusions bullet list."""
    lines: list[str] = []
    if data.conclusions:
        lines.append("## Conclusions")
        lines.append("")
        for c in data.conclusions:
            lines.append(f"- {c}")
        lines.append("")
    return lines


def _training_report_recommendations(data: TrainingReportData) -> list[str]:
    """Recommendations bullet list."""
    lines: list[str] = []
    if data.recommendations:
        lines.append("## Recommendations")
        lines.append("")
        for rec in data.recommendations:
            lines.append(f"- {rec}")
        lines.append("")
    return lines


def generate_training_report_markdown(data: TrainingReportData) -> str:
    """Render a ``TrainingReportData`` instance into a Markdown training report.

    Includes training methodology, hyperparameter tables, per-run metrics,
    quantization trade-offs, and the Stage 10 gate result.
    """
    lines: list[str] = []
    lines.extend(_training_report_overview(data))
    lines.extend(_training_report_runs(data))
    lines.extend(_training_report_evaluation(data))
    lines.extend(_training_report_quantization(data))
    lines.extend(_training_report_gate(data))
    lines.extend(_training_report_conclusions(data))
    lines.extend(_training_report_recommendations(data))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo script generation
# ---------------------------------------------------------------------------

_DEMO_TEMPLATE = '''"""Stage 11 - Demo script for the vulnerability triage harness.

This script demonstrates the project's capabilities in **mock mode** (no model
download, no GPU, no Docker required).  It runs the four-tier evaluation
harness on the gold-eval set and prints a summary of results.

Run::

    python docs/demo.py
    python docs/demo.py --gold-eval eval/gold_set/gold.jsonl
    python docs/demo.py --verbose
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    if "--gold-eval" in sys.argv:
        gold_eval = sys.argv[sys.argv.index("--gold-eval") + 1]
    else:
        gold_eval = os.path.join("eval", "gold_set", "gold.jsonl")
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Ensure the project root is on the path when run from docs/.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import logging
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # --- Step 1: Stage 4 baseline (mock) ---
    print("=" * 60)
    print("Stage 11 Demo - Mock Mode")
    print("=" * 60)
    print("\\n1. Stage 4 - Baseline evaluation (mock backend)")
    print(f"   Gold-eval: {gold_eval}")

    from typer.testing import CliRunner as _CliRunner

    import app.evaluation.cli as cli_mod  # noqa: F811

    runner = _CliRunner()
    result = runner.invoke(
        cli_mod.app,
        [
            "baseline",
            "--gold-eval", str(project_root / gold_eval),
            "--strategy", "zero_shot",
            "--mock",
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage4"),
        ],
    )
    if result.exit_code != 0:
        print(f"   ERROR: {result.output}")
        sys.exit(1)
    print("   [OK] Baseline complete")
    if verbose:
        print(result.output)

    # --- Step 2: Stage 6 four-tier evaluation (mock) ---
    print("\\n2. Stage 6 - Four-tier evaluation (mock sandbox)")
    stage4_dir = project_root / "output" / "stage11_demo" / "stage4"
    result = runner.invoke(
        cli_mod.app,
        [
            "stage6",
            "--gold-eval", str(project_root / gold_eval),
            "--predictions", str(stage4_dir / "predictions.jsonl"),
            "--sandbox-mode", "mock",
            "--skip-tier4",
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage6"),
        ],
    )
    if result.exit_code != 0:
        print(f"   ERROR: {result.output}")
        sys.exit(1)
    print("   [OK] Evaluation complete")
    if verbose:
        print(result.output)

    # --- Step 3: Stage 7 regression (mock) ---
    print("\\n3. Stage 7 - Regression / forgetting analysis (mock)")
    result = runner.invoke(
        cli_mod.app,
        [
            "stage7",
            "--mock",
            "--base-model", "Qwen/Qwen2.5-Coder-7B-Instruct",
            "--tuned-model", "stage11-demo-checkpoint",
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage7"),
        ],
    )
    if result.exit_code != 0:
        print(f"   ERROR: {result.output}")
        sys.exit(1)
    print("   [OK] Regression analysis complete")
    if verbose:
        print(result.output)

    # --- Step 4: Stage 10 gate ---
    print("\\n4. Stage 10 - Regression gate")
    stage6_dir = project_root / "output" / "stage11_demo" / "stage6"
    stage7_dir = project_root / "output" / "stage11_demo" / "stage7"
    result = runner.invoke(
        cli_mod.app,
        [
            "stage10",
            "--baseline-metrics", str(stage4_dir / "metrics.json"),
            "--predictions", str(stage4_dir / "predictions.jsonl"),
            "--stage6-report", str(stage6_dir / "eval_report.json"),
            "--stage7-report", str(stage7_dir / "regression_report.json"),
            "--output-dir", str(project_root / "output" / "stage11_demo" / "stage10"),
        ],
    )
    print(f"   [OK] Gate result (exit code {result.exit_code})")
    if verbose:
        print(result.output)

    # --- Step 5: Print final summary ---
    print(f"\\n{'=' * 60}")
    print("Demo complete! All stages ran successfully in mock mode.")
    print(f"{'=' * 60}")
    print(f"\\nArtifacts written to: {project_root / 'output' / 'stage11_demo'}")
    print("\\nDocumentation deliverables:")
    print("  - docs/model_card.md")
    print("  - docs/training_report.md")
    print("  - docs/demo.py (this file)")


if __name__ == "__main__":
    main()
'''


def generate_demo_script() -> str:
    """Return the demo script source code as a string.

    The demo script is self-contained and runs the full mock-mode pipeline
    (Stages 4->6->7->10) on the gold-eval set.  It is written to
    ``docs/demo.py`` by ``ensure_deliverables``.
    """
    return _DEMO_TEMPLATE


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


class Stage11Generator:
    """Generates and validates Stage 11 documentation deliverables.

    Creates the three deliverables on disk:

    * ``docs/model_card.md``
    * ``docs/training_report.md``
    * ``docs/demo.py``

    and optionally writes machine-readable JSON versions under
    ``output/stage11/`` for CI archival.
    """

    def __init__(self, config: Stage11Config):
        self.config = config
        self._run_id = f"stage11-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Artifact loading (Task 5)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_training_runs(stage_base: Path) -> list[TrainingRunData]:
        """Read Stage 5 SFT + DPO training_result.json files."""
        runs: list[TrainingRunData] = []
        stage5_files = [
            stage_base / "stage5" / "training_result.json",
            stage_base / "stage5" / "dpo" / "training_result.json",
        ]
        for stage5_file in stage5_files:
            if not stage5_file.exists():
                continue
            try:
                data = json.loads(stage5_file.read_text(encoding="utf-8"))
                runs.append(
                    TrainingRunData(
                        run_id=data.get("run_id", "unknown"),
                        method=data.get("method", "unknown"),
                        base_model=data.get("base_model", BASE_MODEL),
                        hyperparams=data.get("hyperparams", {}),
                        train_set_size=data.get("train_set_size", 0),
                        train_time_minutes=data.get("train_time_minutes", 0.0),
                        peak_vram_gb=data.get("peak_vram_gb", 0.0),
                        final_train_loss=data.get("final_train_loss", 0.0),
                        final_val_loss=data.get("final_val_loss"),
                        checkpoint_uri=data.get("checkpoint_uri", ""),
                        train_loss_history=data.get("train_loss_history", []),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(_PARSE_WARN, stage5_file, exc)
        return runs

    @staticmethod
    def _load_baseline_metrics(stage_base: Path) -> EvalMetricsSnapshot | None:
        """Read Stage 4 baseline metrics.json if it exists."""
        stage4_path = stage_base / "stage4" / "metrics.json"
        if not stage4_path.exists():
            return None
        try:
            data = json.loads(stage4_path.read_text(encoding="utf-8"))
            return EvalMetricsSnapshot(
                stage=4,
                run_id=data.get("run_id", "stage4"),
                base_model=data.get("base_model", BASE_MODEL),
                cwe_macro_f1=data.get("cwe_macro_f1", 0.0),
                severity_accuracy=data.get("severity_accuracy", 0.0),
                hallucination_rate=data.get("hallucination_rate", 0.0),
                patch_coverage=data.get("patch_coverage", 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(_PARSE_WARN, stage4_path, exc)
            return None

    @staticmethod
    def _load_tuned_metrics(stage_base: Path) -> EvalMetricsSnapshot | None:
        """Read Stage 6 eval_report.json (tuned-model four-tier metrics)."""
        stage6_path = stage_base / "stage6" / _STAGE6_REPORT_NAME
        if not stage6_path.exists():
            return None
        try:
            data = json.loads(stage6_path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", data)
            return EvalMetricsSnapshot(
                stage=6,
                run_id=data.get("run_id", "stage6"),
                base_model=data.get("base_model", BASE_MODEL),
                cwe_macro_f1=metrics.get("model_cwe_macro_f1", 0.0),
                # The four-tier Stage 6 harness does not score severity;
                # None (not 0.0) signals "not measured here" rather than
                # a real zero accuracy.
                severity_accuracy=metrics.get("severity_accuracy"),
                hallucination_rate=metrics.get("hallucination_rate", 0.0),
                patch_coverage=metrics.get("avg_patch_coverage", 0.0),
                exec_pass_rate=metrics.get("exec_pass_rate", 0.0),
                per_class=metrics.get("per_class", {}),
                manifest=data.get("manifest", {}),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(_PARSE_WARN, stage6_path, exc)
            return None

    @staticmethod
    def _load_regression_report(
        stage_base: Path, tuned_metrics: EvalMetricsSnapshot | None
    ) -> EvalMetricsSnapshot | None:
        """Read Stage 7 regression_report.json (forgetting delta)."""
        stage7_path = stage_base / "stage7" / _STAGE7_REPORT_NAME
        if not stage7_path.exists():
            return None
        try:
            data = json.loads(stage7_path.read_text(encoding="utf-8"))
            forgetting_delta = data.get("forgetting_delta", 0.0)
            # Merge the forgetting delta into the existing tuned metrics
            # (stage 6 snapshot) rather than replacing it with a stage-7
            # snapshot — the regression deltas augment the tuned evaluation,
            # they don't replace it.
            if tuned_metrics is not None:
                tuned_metrics = tuned_metrics.model_copy(
                    update={"forgetting_delta": forgetting_delta}
                )
            # Build a standalone regression_report snapshot so the
            # training report's "Stage 7 — Regression / Forgetting
            # Analysis" section is populated.
            return EvalMetricsSnapshot(
                stage=7,
                run_id=data.get("run_id", "stage7"),
                base_model=data.get("base_model", BASE_MODEL),
                forgetting_delta=forgetting_delta,
                exec_pass_rate=data.get("tuned_metrics", {}).get("execution_accuracy", 0.0),
                manifest=data.get("manifest", {}),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(_PARSE_WARN, stage7_path, exc)
            return None

    @staticmethod
    def _load_quant_results(stage_base: Path) -> list[QuantResultData]:
        """Read Stage 8 quantization result JSON files."""
        results: list[QuantResultData] = []
        stage8_dir = stage_base / "stage8"
        if not stage8_dir.exists():
            return results
        for qfile in stage8_dir.glob("quant_results_*.json"):
            try:
                data = json.loads(qfile.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for item in data:
                        results.append(QuantResultData(**item))
                elif isinstance(data, dict):
                    results.append(QuantResultData(**data))
            except Exception as exc:  # noqa: BLE001
                logger.warning(_PARSE_WARN, qfile, exc)
        return results

    def load_artifacts(self) -> Stage11Config:
        """Load training & evaluation artifacts from disk into a new config.

        Reads real outputs from the Stage 4/5/6/7 output directories and
        returns a **new** ``Stage11Config`` with those values populated.
        When a file is missing the corresponding field falls back to the
        existing config value (mock mode).

        Reads:
        - ``<stage_base>/stage5/training_result.json``      → SFT TrainingRunData
        - ``<stage_base>/stage5/dpo/training_result.json``  → DPO TrainingRunData
        - ``<stage_base>/stage4/metrics.json``               → baseline EvalMetricsSnapshot
        - ``<stage_base>/stage6/eval_report.json``           → tuned EvalMetricsSnapshot
        - ``<stage_base>/stage7/regression_report.json``     → regression metrics

        ``<stage_base>`` is the parent of ``self.config.output_dir`` (e.g.
        ``output/stage11`` → ``output``).
        """
        stage_base = Path(self.config.output_dir).resolve().parent
        training_runs = self._load_training_runs(stage_base)
        baseline_metrics = self._load_baseline_metrics(stage_base)
        if baseline_metrics is None:
            baseline_metrics = self.config.baseline_metrics
        tuned_metrics = self._load_tuned_metrics(stage_base)
        if tuned_metrics is None:
            tuned_metrics = self.config.tuned_metrics
        regression_report = self._load_regression_report(stage_base, tuned_metrics)
        if regression_report is None:
            regression_report = self.config.regression_report
        # Merge the forgetting delta from the Stage 7 regression report into
        # the tuned (Stage 6) metrics snapshot so downstream consumers see it.
        if regression_report is not None and tuned_metrics is not None:
            tuned_metrics = tuned_metrics.model_copy(
                update={"forgetting_delta": regression_report.forgetting_delta}
            )
        quant_results = self._load_quant_results(stage_base)
        quant_results = quant_results or list(self.config.quant_results)

        logger.info(
            "load_artifacts: %d training runs, baseline=%s, tuned=%s, %d quant results",
            len(training_runs),
            baseline_metrics is not None,
            tuned_metrics is not None,
            len(quant_results),
        )

        return replace(
            self.config,
            training_runs=training_runs,
            baseline_metrics=baseline_metrics,
            tuned_metrics=tuned_metrics,
            regression_report=regression_report,
            quant_results=quant_results,
        )

    # ------------------------------------------------------------------
    # Deliverable creation
    # ------------------------------------------------------------------

    def _model_card_data(self) -> ModelCardData:
        """Build ``ModelCardData`` from the config + any metrics provided."""
        return ModelCardData(
            model_name=self.config.model_name,
            base_model=self.config.base_model,
            fine_tuned=True,
            training_method=self.config.training_method,
            lora_rank=self.config.lora_rank,
            quant_method=self.config.quant_method,
            quant_bit_width=self.config.quant_bit_width,
            cwe_scope=self.config.cwe_scope,
            language=self.config.language,
            training_data_size=self.config.training_data_size,
            intended_use=[
                "Classifying the CWE category of a vulnerable code snippet",
                "Suggesting a minimal, working patch for the vulnerability",
                "Batch analysis of code repositories for triage prioritization",
                "Interactive vulnerability analysis via the air-gapped serving layer",
            ],
            metrics=(
                self.config.tuned_metrics
                or self.config.baseline_metrics
                or EvalMetricsSnapshot(stage=6, run_id="unknown", base_model=self.config.base_model)
            ),
            limitations=[
                f"Trained on {len(self.config.cwe_scope)} CWE classes; "
                "out-of-scope CWEs are treated as hallucinations.",
                "Not a general-purpose security scanner — does not detect logic bugs, "
                "configuration issues, or CWE classes outside the listed scope.",
                "The exec-based evaluation runs proposed patches in a sandboxed subprocess. "
                "Docker isolation is implemented (see `app/evaluation/tier3_exec.py`), "
                "providing read-only filesystem, no network, and memory limits.",
                "Proposed patches should be reviewed by a human before merging into production.",
                f"Trained on a small subset ({self.config.training_data_size} samples) "
                f"using {self.config.execution_environment.upper()} execution; "
                f"the full training pipeline supports GPU/QLoRA for larger datasets.",
                "This model predicts CWE-89 for most samples due to the small training set; "
                "additional training data and epochs are needed for multi-class accuracy.",
            ],
            ethical_considerations=[
                "This model is a research artifact, not a production SOC tool.",
                "Do not use model predictions as the sole basis for security decisions.",
                "The model may produce incorrect patches — verify all suggestions before use.",
            ],
            out_of_scope=[
                "Real-time repository monitoring in CI pipelines",
                "Network-based vulnerability scanning (no port scanning, no HTTP fuzzing)",
                "Supply-chain security / third-party dependency auditing (use pip-audit/Safety)",
                "Legal or compliance assessment of software",
                "Production incident response",
            ],
        )

    def _training_report_data(self) -> TrainingReportData:
        """Build ``TrainingReportData`` from the config + any saved data.

        When no real training runs are available (``training_runs`` is empty),
        the conclusions are phrased as *expected outcomes* based on the
        training methodology, not as measured results. When real training runs
        *are* present, concrete conclusions are generated from them.
        """
        if self.config.training_runs:
            conclusions = self._conclusions_from_runs()
        else:
            conclusions = [
                "No real training runs have been executed yet. The conclusions below "
                "describe the intended methodology, not measured results — training "
                "results will be populated once Stage 5 is run on a GPU.",
                "QLoRA (4-bit NF4) enables parameter-efficient fine-tuning on consumer "
                "GPUs with 8 GB VRAM (estimated, not measured).",
                "SFT full-parameter training requires >=16 GB VRAM.",
                "The LoRA rank sweep (ranks 8—128) is designed to identify the smallest "
                "adapter that preserves quality.",
                "DPO preference alignment is intended to reduce hallucination rate "
                "without sacrificing classification accuracy.",
            ]

        if self.config.training_runs:
            recommendations = [
                f"Scale to a larger training dataset (current: {self.config.training_data_size} "
                "samples) to improve multi-class CWE discrimination.",
                "Increase training epochs or try a higher LoRA rank (current: r=8) "
                "to reduce underfitting on the small dataset.",
                "Re-run the Stage 10 regression gate after any model update.",
                "Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.",
            ]
        else:
            recommendations = [
                "Run Stage 5 training on a CUDA GPU before publishing real metrics.",
                "Re-run the Stage 10 regression gate after any model update.",
                "Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.",
            ]

        return TrainingReportData(
            report_id=self._run_id,
            model_name=self.config.model_name,
            base_model=self.config.base_model,
            training_runs=self.config.training_runs,
            baseline_metrics=self.config.baseline_metrics,
            tuned_metrics=self.config.tuned_metrics,
            regression_report=self.config.regression_report,
            quant_results=self.config.quant_results,
            conclusions=conclusions,
            recommendations=recommendations,
        )

    def _conclusions_from_runs(self) -> list[str]:
        """Generate conclusions from actual training run data."""
        conclusions = []
        for run in self.config.training_runs:
            method = run.method or "unknown"
            train_loss = run.final_train_loss
            val_loss = run.final_val_loss
            if val_loss is not None:
                conclusions.append(
                    f"Run `{run.run_id}` ({method}): train loss = {train_loss:.4f}, "
                    f"val loss = {val_loss:.4f}."
                )
            else:
                conclusions.append(
                    f"Run `{run.run_id}` ({method}): train loss = {train_loss:.4f}. "
                    f"No validation loss was recorded."
                )

        # Add evaluation insights if metrics are available
        if self.config.tuned_metrics:
            tm = self.config.tuned_metrics
            conclusions.append(
                f"Tuned model Stage 6 evaluation: CWE Macro-F1 = {tm.cwe_macro_f1:.4f}, "
                f"Severity accuracy = {_fmt_severity(tm.severity_accuracy)}, "
                f"Patch coverage = {tm.patch_coverage:.4f}."
            )
            if tm.cwe_macro_f1 < 0.5:
                conclusions.append(
                    f"CWE Macro-F1 is low ({tm.cwe_macro_f1:.4f}) — the small training set "
                    f"({self.config.training_data_size} samples) limits multi-class "
                    "discrimination. The model defaults to CWE-89 for most inputs, "
                    "which inflates recall but not precision."
                )
        if self.config.baseline_metrics:
            bm = self.config.baseline_metrics
            conclusions.append(
                f"Pre-fine-tuning baseline: CWE Macro-F1 = {bm.cwe_macro_f1:.4f}, "
                f"Severity accuracy = {_fmt_severity(bm.severity_accuracy)}."
            )

        return conclusions

    def ensure_deliverables(self) -> dict[str, str]:
        """Create / refresh all Stage 11 deliverables on disk.

        Before generating, ``load_artifacts()`` is called to pull real
        training / evaluation numbers from the Stage 4–8 output directories.
        When those files are absent the generator falls back to the config's
        existing (mock-mode) values.

        Returns a dict mapping deliverable name → file path.
        """
        # Load real artifacts into a fresh config; falls back to mock values.
        self.config = self.load_artifacts()

        docs_dir = Path(self.config.docs_dir)
        docs_dir.mkdir(parents=True, exist_ok=True)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: dict[str, str] = {}

        # 1. Model card
        model_card = generate_model_card_markdown(self._model_card_data())
        mc_path = docs_dir / "model_card.md"
        mc_path.write_text(model_card, encoding="utf-8")
        results["model_card"] = str(mc_path)
        logger.info("Wrote model card to %s", mc_path)

        # JSON sidecar for CI archival
        mc_json_path = output_dir / "model_card_data.json"
        mc_json_path.write_text(self._model_card_data().model_dump_json(indent=2), encoding="utf-8")
        results["model_card_json"] = str(mc_json_path)

        # 2. Training report
        report = generate_training_report_markdown(self._training_report_data())
        tr_path = docs_dir / "training_report.md"
        tr_path.write_text(report, encoding="utf-8")
        results["training_report"] = str(tr_path)
        logger.info("Wrote training report to %s", tr_path)

        # JSON sidecar for CI archival
        tr_json_path = output_dir / "training_report_data.json"
        tr_json_path.write_text(
            self._training_report_data().model_dump_json(indent=2), encoding="utf-8"
        )
        results["training_report_json"] = str(tr_json_path)

        # 3. Demo script
        demo_src = generate_demo_script()
        demo_path = docs_dir / "demo.py"
        demo_path.write_text(demo_src, encoding="utf-8")
        results["demo_script"] = str(demo_path)
        logger.info("Wrote demo script to %s", demo_path)

        return results

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_deliverables(self) -> bool:
        """Verify that all Stage 11 deliverables exist and are non-empty.

        Returns ``True`` if all checks pass, ``False`` otherwise.
        """
        docs_dir = Path(self.config.docs_dir)
        required = ["model_card.md", "training_report.md", "demo.py"]

        for name in required:
            path = docs_dir / name
            if not path.exists():
                logger.error("Missing deliverable: %s", path)
                return False
            if path.stat().st_size == 0:
                logger.error("Empty deliverable: %s", path)
                return False
        return True

    # ------------------------------------------------------------------
    # Demo execution (mock mode - no GPU / ML deps required)
    # ------------------------------------------------------------------

    def run_demo(self) -> DemoResult:
        """Run the full mock-mode pipeline (Stages 4->6->7->10) as a demo.

        This mirrors what ``docs/demo.py`` does but is callable programmatically.
        Uses ``MockBackend`` so no model download or GPU is required.
        """
        from app.ci.config import RegressionGateConfig
        from app.ci.gate import run_gate
        from app.evaluation.backends import MockBackend
        from app.evaluation.baseline import BaselineConfig, run_baseline
        from app.evaluation.general_capability import (
            LocalCodeTestRunner,
            RegressionConfig,
            run_regression_analysis,
        )
        from app.evaluation.runner import (
            EvalConfig,
            EvaluationRunner,
            load_predictions,
            load_samples,
        )

        gold_eval_path = Path("eval/gold_set/gold.jsonl")
        output_dir = Path(self.config.output_dir) / "demo"
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_data: list[dict] = []

        try:
            # Step 1: Stage 4 baseline (mock)
            logger.info("Stage 11 demo: Step 1 - Stage 4 baseline (mock)")
            mock_backend = MockBackend(
                responses={
                    "CWE-89": '{"cwe_id": "CWE-89", "severity": "high", '
                    '"explanation": "SQL injection via string concatenation.", '
                    '"patch_diff": "--- a/app.py\\n+++ b/app.py\\n- old\\n+ new"}',
                },
                default='{"cwe_id": "CWE-89", "severity": "high", '
                '"explanation": "Demo mock response.", "patch_diff": ""}',
            )
            baseline_result = run_baseline(
                gold_eval_path=str(gold_eval_path),
                output_dir=str(output_dir / "stage4"),
                config=BaselineConfig(strategy="zero_shot", base_model=self.config.base_model),
                backend=mock_backend,
            )
            for p in baseline_result.predictions:
                predictions_data.append(
                    {
                        "sample_id": p.sample_id,
                        "predicted_cwe": p.predicted_cwe,
                        "predicted_severity": p.predicted_severity,
                    }
                )

            # Step 2: Stage 6 four-tier evaluation (mock sandbox)
            logger.info("Stage 11 demo: Step 2 - Stage 6 evaluation (mock sandbox)")
            eval_config = EvalConfig(
                base_model=self.config.model_name,
                sandbox_mode="mock",
                skip_tier4=True,
            )
            runner = EvaluationRunner(config=eval_config)
            samples = load_samples(str(gold_eval_path))
            preds = load_predictions(str(output_dir / "stage4" / "predictions.jsonl"))
            eval_report = runner.run(samples, preds)

            stage6_metrics = {
                "model_cwe_macro_f1": eval_report.metrics.model_cwe_macro_f1,
                "exec_pass_rate": eval_report.metrics.exec_pass_rate,
                "hallucination_rate": eval_report.metrics.hallucination_rate,
                "patch_coverage": eval_report.metrics.avg_patch_coverage,
            }

            # Step 3: Stage 7 regression (mock)
            logger.info("Stage 11 demo: Step 3 - Stage 7 regression (mock)")
            from app.evaluation.backends import MockBackend as MB

            reg_config = RegressionConfig(
                base_model=self.config.base_model,
                tuned_model=self.config.model_name,
            )
            # Use the same backend for both base and tuned (mock mode)
            base_backend = MB(default="def solution():\n    return None\n")
            tuned_backend = MB(default="def solution():\n    return None\n")
            code_runner = LocalCodeTestRunner()
            regression_report = run_regression_analysis(
                config=reg_config,
                base_backend=base_backend,
                tuned_backend=tuned_backend,
                runner=code_runner,
            )

            # Step 4: Stage 10 gate
            logger.info("Stage 11 demo: Step 4 - Stage 10 gate")
            gate_config = RegressionGateConfig(
                baseline_metrics_path=str(output_dir / "stage4" / "metrics.json"),
                stage6_report_path=str(output_dir / "stage6" / _STAGE6_REPORT_NAME),
                stage7_report_path=str(output_dir / "stage7" / _STAGE7_REPORT_NAME),
            )
            # Write stage6 + stage7 reports
            (output_dir / "stage6").mkdir(parents=True, exist_ok=True)
            (output_dir / "stage6" / _STAGE6_REPORT_NAME).write_text(
                eval_report.model_dump_json(indent=2), encoding="utf-8"
            )
            (output_dir / "stage7").mkdir(parents=True, exist_ok=True)
            (output_dir / "stage7" / _STAGE7_REPORT_NAME).write_text(
                regression_report.model_dump_json(indent=2), encoding="utf-8"
            )
            gate_result = run_gate(gate_config)

            return DemoResult(
                run_id=self._run_id,
                model_name=self.config.model_name,
                num_gold_samples=len(samples),
                predictions=predictions_data,
                metrics={
                    **stage6_metrics,
                    "base_cwe_macro_f1": baseline_result.metrics.cwe_macro_f1,
                    "tuned_cwe_macro_f1": eval_report.metrics.model_cwe_macro_f1,
                    "forgetting_delta": regression_report.forgetting_delta,
                    "gate_status": gate_result.status.value,
                },
                stage6_report=eval_report.model_dump(),
                succeeded=gate_result.passed,
            )

        except Exception as exc:
            logger.exception("Stage 11 demo failed: %s", exc)
            return DemoResult(
                run_id=self._run_id,
                model_name=self.config.model_name,
                num_gold_samples=0,
                succeeded=False,
                error=str(exc),
            )


def run_stage11(config: Stage11Config) -> dict[str, str]:
    """Convenience function: create all deliverables and validate them.

    This is the primary entry point for CLI usage.
    """
    generator = Stage11Generator(config)
    results = generator.ensure_deliverables()
    valid = generator.validate_deliverables()
    if not valid:
        raise RuntimeError("Stage 11 deliverables validation failed")
    return results
