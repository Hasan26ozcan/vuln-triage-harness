"""Stage 4 CLI: pre-fine-tuning baseline evaluation.

Usage:
    # Run zero-shot baseline on the gold-eval set
    python -m app.evaluation.cli baseline --gold-eval eval/gold_set/gold.jsonl

    # Run few-shot baseline with 3 in-context examples from Stage 3 output
    python -m app.evaluation.cli baseline --gold-eval eval/gold_set/gold.jsonl \\
        --strategy few-shot --num-shots 3 --few-shot-examples output/stage3/train.jsonl

    # Re-evaluate saved predictions without re-running inference
    python -m app.evaluation.cli evaluate --predictions output/stage4/predictions.jsonl \\
        --gold-eval eval/gold_set/gold.jsonl

    # Use a smaller model for fast iteration
    python -m app.evaluation.cli baseline --gold-eval eval/gold_set/gold.jsonl \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct
"""

from __future__ import annotations

import json
import logging
import os

import typer

from app.ci.config import (
    DEFAULT_FORGETTING_THRESHOLD,
    DEFAULT_MAX_F1_DROP_PERCENT,
    DEFAULT_MAX_HALLUCINATION_RATE,
    DEFAULT_MIN_EXEC_PASS_RATE,
)
from app.evaluation.backends import MockBackend, ModelBackend
from app.evaluation.baseline import (
    BaselineConfig,
    run_baseline,
)
from app.evaluation.metrics import compute_metrics
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample
from app.serving.cli import app as _serving_app

app = typer.Typer(help="Evaluation tools for the vuln-triage-harness.")

# Default base model shared across Stage 4/6/7 subcommands, so it's defined
# once instead of repeated as a literal at every call site.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Default 7B base model used by Stage 8 and Stage 4 baseline (extracted to
# avoid the duplicate-string-literal smell, SonarQube S1132).
DEFAULT_BASE_MODEL_7B = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Section header echoed before a metrics block in several subcommands.
_METRICS_HEADER = "Metrics:"

# Maps Stage 10 check status strings to terminal glyphs.
_STATUS_GLYPHS: dict[str, str] = {
    "pass": "[OK]",  # nosec B105
    "fail": "[FAIL]",
    "skip": "[SKIP]",
}

# Stage 9 subcommands — delegate to app.serving.cli directly (the serving
# CLI has no heavy ML imports at module level, so importing it here is safe).
app.add_typer(_serving_app, name="stage9")
del _serving_app


# ---------------------------------------------------------------------------
# Stage 6 helpers
# ---------------------------------------------------------------------------


def _setup_local_llm_judge(
    llm_judge_model: str | None,
    base_model: str,
    checkpoint: str | None,
) -> tuple[object | None, str | None]:
    """Set up a local LLM judge backend when ``--llm-judge-model local``.

    Returns ``(tier4_evaluator, config_llm_judge_model)`` — when the judge is
    not local, both are ``None`` / the original value respectively.
    """
    if llm_judge_model != "local":
        return None, llm_judge_model

    from app.evaluation.backends import QwenBackend
    from app.evaluation.tier4_llm_judge import LlmJudge, LocalLlmJudgeBackend

    judge_model = base_model or DEFAULT_BASE_MODEL
    typer.echo(f"Loading local LLM judge model: {judge_model}")
    if checkpoint:
        typer.echo(f"  + LoRA checkpoint: {checkpoint}")
        judge_backend = QwenBackend(
            model_name=checkpoint,
            base_model=judge_model,
        )
    else:
        judge_backend = QwenBackend(model_name=judge_model)
    pipe = judge_backend._load()
    tier4_evaluator = LlmJudge(
        backend=LocalLlmJudgeBackend(
            model=pipe.model,
            tokenizer=pipe.tokenizer,
        ),
        model=str(checkpoint) if checkpoint else judge_model,
    )
    # Don't set llm_judge_model in config — the injected evaluator is used instead.
    return tier4_evaluator, None


@app.command(name="stage6")
def stage6(
    gold_eval: str = typer.Option(
        ...,
        "--gold-eval",
        "-g",
        help="Path to gold-eval JSONL file (one VulnSample per line).",
    ),
    predictions: str = typer.Option(
        ...,
        "--predictions",
        "-p",
        help="Path to ModelPrediction JSONL file.",
    ),
    output_dir: str = typer.Option(
        "./output/stage6",
        "--output-dir",
        "-o",
        help="Directory to write the EvalReport JSON.",
    ),
    base_model: str = typer.Option(
        "unknown",
        "--base-model",
        "-m",
        help="Model name being evaluated.",
    ),
    embedding_model: str = typer.Option(
        None,
        "--embedding-model",
        "-e",
        help="Sentence-transformers model for Tier 2 embedding similarity (optional).",
    ),
    sandbox_mode: str = typer.Option(
        "mock",
        "--sandbox-mode",
        help="Sandbox mode: mock | local | docker.",
    ),
    llm_judge_model: str = typer.Option(
        None,
        "--llm-judge-model",
        help="LLM model for Tier 4 judge (e.g. 'gpt-4o-mini', or 'local' for a "
        "local HuggingFace model). Requires OPENAI_API_KEY for OpenAI models.",
    ),
    checkpoint: str = typer.Option(
        None,
        "--checkpoint",
        "-c",
        help=(
            "Path to a LoRA/DPO checkpoint directory. When "
            "--llm-judge-model local is used, this loads the checkpoint "
            "(via base_model + PEFT) as the judge. When set, "
            "--base-model should point to the base model (e.g. "
            f"{DEFAULT_BASE_MODEL})."
        ),
    ),
    skip_tier3: bool = typer.Option(
        False,
        "--skip-tier3/--no-skip-tier3",
        help="Skip exec-based evaluation (Tier 3).",
    ),
    skip_tier4: bool = typer.Option(
        False,
        "--skip-tier4/--no-skip-tier4",
        help="Skip LLM judge evaluation (Tier 4). Saves LLM cost.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the Stage 6 four-tier evaluation harness."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.evaluation.runner import EvalConfig, EvaluationRunner, load_predictions, load_samples

    tier4_evaluator, config_llm_judge_model = _setup_local_llm_judge(
        llm_judge_model, base_model, checkpoint
    )

    config = EvalConfig(
        base_model=base_model,
        embedding_model=embedding_model,
        sandbox_mode=sandbox_mode,
        llm_judge_model=config_llm_judge_model,
        skip_tier3=skip_tier3,
        skip_tier4=skip_tier4,
    )

    if tier4_evaluator is not None:
        runner = EvaluationRunner(config=config, tier4_evaluator=tier4_evaluator)
    else:
        runner = EvaluationRunner(config=config)
    samples = load_samples(gold_eval)
    preds = load_predictions(predictions)

    typer.echo("Running Stage 6 evaluation")
    typer.echo(f"Samples:     {len(samples)}")
    typer.echo(f"Predictions: {len(preds)}")
    typer.echo(f"Sandbox:     {sandbox_mode}")
    typer.echo(f"Skip Tier 3: {skip_tier3}")
    typer.echo(f"Skip Tier 4: {skip_tier4}")

    report = runner.run(samples, preds)

    # Write report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Print summary
    typer.echo("")
    typer.echo(f"Run ID: {report.run_id}")
    typer.echo(f"Stage:  {report.stage}")
    typer.echo("")
    m = report.metrics
    typer.echo(_METRICS_HEADER)
    typer.echo(f"  Tier1 CWE Macro-F1:     {m.tier1_cwe_macro_f1:.4f}")
    typer.echo(f"  Tier1 Coverage:         {m.tier1_coverage:.4f}")
    typer.echo(f"  Tier2 CWE Macro-F1:     {m.tier2_cwe_macro_f1:.4f}")
    typer.echo(f"  Tier2 Coverage:         {m.tier2_coverage:.4f}")
    typer.echo(f"  Model CWE Macro-F1:     {m.model_cwe_macro_f1:.4f}")
    typer.echo(f"  Exec Pass Rate:         {m.exec_pass_rate:.4f}")
    typer.echo(f"  Patch Applies:          {m.patch_applies_rate:.4f}")
    typer.echo(f"  Build Succeeds:         {m.build_succeeds_rate:.4f}")
    typer.echo(f"  Hallucination Rate:     {m.hallucination_rate:.4f}")
    typer.echo(f"  Patch Coverage:         {m.avg_patch_coverage:.4f}")
    if m.avg_explanation_quality is not None:
        typer.echo(f"  Avg Explanation Quality: {m.avg_explanation_quality:.4f}")
    if m.avg_patch_minimality is not None:
        typer.echo(f"  Avg Patch Minimality:   {m.avg_patch_minimality:.4f}")
    typer.echo("")
    typer.echo("Per-class F1 (model):")
    for cwe, stats in sorted(m.per_class.items()):
        typer.echo(
            f"  {cwe:10s}  P={stats['precision']:.4f}  "
            f"R={stats['recall']:.4f}  F1={stats['f1']:.4f}"
        )
    typer.echo("")
    typer.echo(f"Report written to: {report_path}")


# Stage 7 subcommands


@app.command(name="stage7")
def stage7(
    base_model: str = typer.Option(
        DEFAULT_BASE_MODEL,
        "--base-model",
        "-b",
        help="Base (pre-fine-tuning) model name or HuggingFace path.",
    ),
    tuned_model: str = typer.Option(
        ...,
        "--tuned-model",
        "-t",
        help="Tuned (post-fine-tuning) checkpoint name or path.",
    ),
    output_dir: str = typer.Option(
        "./output/stage7",
        "--output-dir",
        "-o",
        help="Directory to write the RegressionReport JSON.",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use MockBackend + MockCodeTestRunner (no model download, no subprocess).",
    ),
    timeout_seconds: int = typer.Option(
        30,
        "--timeout",
        help="Per-task test execution timeout in seconds (local/docker runner only).",
    ),
    runner: str = typer.Option(
        "local",
        "--runner",
        "-r",
        help="Code test runner: local | docker | mock. 'docker' uses DockerSandboxRunner "
        "for containerised isolation; 'local' uses subprocess.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run Stage 7 regression / forgetting analysis.

    Evaluates a base model and a fine-tuned model on a set of general
    (non-security) code-generation tasks. The forgetting delta is the
    difference in execution accuracy:

        delta = tuned_exec_accuracy - base_exec_accuracy

    A negative delta means the fine-tuned model forgot general coding ability.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.evaluation.backends import MockBackend
    from app.evaluation.general_capability import (
        DEFAULT_GENERAL_TASKS,
        CodeTestRunner,
        DockerCodeTestRunner,
        LocalCodeTestRunner,
        MockCodeTestRunner,
        RegressionConfig,
        run_regression_analysis,
    )

    config = RegressionConfig(
        base_model=base_model,
        tuned_model=tuned_model,
        timeout_seconds=timeout_seconds,
    )

    if mock:
        # MockBackend returns a trivial function for every prompt;
        # MockCodeTestRunner returns canned pass/fail without subprocess.
        base_backend: ModelBackend = MockBackend(
            default="def solution():\n    return None\n",
            responses={},
        )
        tuned_backend: ModelBackend = MockBackend(
            default="def solution():\n    return None\n",
            responses={},
        )
        code_runner: CodeTestRunner = MockCodeTestRunner(default_passed=True)
    else:
        from app.evaluation.backends import QwenBackend

        base_backend = QwenBackend(model_name=base_model)
        tuned_backend = QwenBackend(model_name=tuned_model, base_model=base_model)
        if runner == "docker":
            code_runner = DockerCodeTestRunner(timeout_seconds=timeout_seconds)
        else:
            code_runner = LocalCodeTestRunner(timeout_seconds=timeout_seconds)

    typer.echo("Running Stage 7: regression / forgetting analysis")
    typer.echo(f"Base model:  {base_model}")
    typer.echo(f"Tuned model: {tuned_model}")
    typer.echo(f"Tasks:       {len(DEFAULT_GENERAL_TASKS)}")
    typer.echo(f"Mock mode:   {mock}")
    typer.echo(f"Runner:      {runner}")

    report = run_regression_analysis(
        config=config,
        base_backend=base_backend,
        tuned_backend=tuned_backend,
        runner=code_runner,
    )

    # Write report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "regression_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Print summary
    typer.echo("")
    typer.echo(f"Run ID:                    {report.run_id}")
    typer.echo(f"Base exec accuracy:       {report.base_metrics.execution_accuracy:.4f}")
    typer.echo(f"Tuned exec accuracy:      {report.tuned_metrics.execution_accuracy:.4f}")
    typer.echo(f"Forgetting delta:         {report.forgetting_delta:+.4f}")
    typer.echo("")
    if report.forgetting_delta >= 0:
        typer.echo("[OK] No forgetting - tuned model maintains or improves general capability.")
    else:
        typer.echo("[WARN] Forgetting detected - tuned model lost general coding ability.")
    typer.echo("")
    typer.echo(f"Report written to: {report_path}")


# Stage 8 subcommands


@app.command(name="stage8")
def stage8(
    source_checkpoint: str = typer.Option(
        ...,
        "--source-checkpoint",
        "-s",
        help="Path to the Stage 5 trained checkpoint to quantize.",
    ),
    base_model: str = typer.Option(
        DEFAULT_BASE_MODEL_7B,
        "--base-model",
        "-b",
        help="Base model name (for reporting / metadata).",
    ),
    output_dir: str = typer.Option(
        "./output/stage8",
        "--output-dir",
        "-o",
        help="Directory to write the QuantReport JSON.",
    ),
    methods: str = typer.Option(
        "gptq,awq,gguf",
        "--methods",
        "-m",
        help="Comma-separated list of quantization methods (gptq, awq, gguf, none).",
    ),
    bit_widths: str = typer.Option(
        "4",
        "--bits",
        help="Comma-separated bit-widths (for GPTQ/AWQ; GGUF uses its own quant_types).",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use MockQuantizer (deterministic, no ML deps, no GPU).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Use heuristic estimates only (no quantization, no ML deps).",
    ),
    target_vram_gb: float = typer.Option(
        None,
        "--target-vram",
        help="VRAM budget in GB - filters results in select_best_config.",
    ),
    target_size_gb: float = typer.Option(
        None,
        "--target-size",
        help="On-disk size budget in GB - filters results in select_best_config.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the Stage 8 quantization matrix (GPTQ / AWQ / GGUF).

    Produces a ``QuantReport`` JSON with per-method×bit-width results
    and a best-config recommendation based on quality vs. VRAM / size.

    In ``--mock`` mode, no heavy ML dependencies are required.
    In ``--dry-run`` mode, heuristic estimates are used instead of
    calling any quantizer.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.quantization.config import QuantConfig
    from app.quantization.quantizer import run_quantization_matrix

    resolved_methods, resolved_bits = _parse_stage8_options(methods, bit_widths)

    config = QuantConfig(
        base_model=base_model,
        source_checkpoint=source_checkpoint,
        output_base=output_dir,
        methods=resolved_methods,
        bit_widths=resolved_bits,
        dry_run=dry_run,
        mock=mock,
        target_vram_gb=target_vram_gb,
        target_size_gb=target_size_gb,
    )

    for warning in config.all_warnings():
        typer.echo(f"Warning: {warning}", err=True)

    typer.echo("Running Stage 8: quantization matrix")
    typer.echo(f"Base model:    {base_model}")
    typer.echo(f"Checkpoint:    {source_checkpoint}")
    typer.echo(f"Methods:       {[m.value for m in resolved_methods]}")
    typer.echo(f"Bit widths:    {resolved_bits}")
    typer.echo(f"Mock mode:     {mock}")
    typer.echo(f"Dry run:       {dry_run}")

    report = run_quantization_matrix(config)
    report_path = _write_quant_report(report, output_dir)
    _print_quant_summary(report, report_path)


def _parse_stage8_options(
    methods: str,
    bit_widths: str,
) -> tuple[list, list[int]]:
    """Parse comma-separated method and bit-width strings into validated lists."""
    from app.schemas.quantization import QuantMethod

    method_map = {m.value: m for m in QuantMethod}
    resolved_methods: list[QuantMethod] = []
    for m_str in methods.split(","):
        key = m_str.strip().lower()
        if not key:
            continue
        if key not in method_map:
            typer.echo(
                f"Error: unknown quantization method '{m_str}'. "
                f"Valid: {', '.join(method_map.keys())}",
                err=True,
            )
            raise typer.Exit(1)
        resolved_methods.append(method_map[key])

    resolved_bits = [int(b) for b in bit_widths.split(",")]
    return resolved_methods, resolved_bits


def _write_quant_report(report, output_dir: str) -> str:
    """Write the QuantReport JSON and return the report path."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "quant_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
    return report_path


def _print_quant_summary(report, report_path: str) -> None:
    """Print the human-readable Stage 8 quantization summary."""
    from app.schemas.quantization import QuantStatus

    typer.echo("")
    typer.echo(f"Run ID:            {report.run_id}")
    typer.echo(f"Total results:     {len(report.results)}")
    if report.best_result:
        best = report.best_result
        typer.echo(f"Best config:       {best.quant_method.value} @ {best.bit_width}-bit")
        typer.echo(f"  Estimated VRAM:  {best.estimated_vram_gb} GB")
        typer.echo(f"  Est. size:       {best.quantized_model_size_gb} GB")
        if best.model_cwe_macro_f1 is not None:
            typer.echo(f"  Est. CWE-F1:     {best.model_cwe_macro_f1:.4f}")
        if best.tokens_per_sec is not None:
            typer.echo(f"  Est. tokens/sec: {best.tokens_per_sec:.1f}")
    else:
        typer.echo("Best config:       none (no completed results)")
    typer.echo("")
    typer.echo("Per-result summary:")
    for r in report.results:
        status_icon = "[OK]" if r.status == QuantStatus.COMPLETED else "[XX]"
        typer.echo(
            f"  {status_icon} {r.quant_method.value:5s} @ {str(r.bit_width or '?'):>2s}-bit  "
            f"VRAM={r.estimated_vram_gb:>4.1f}GB  "
            f"size={r.quantized_model_size_gb:>4.1f}GB  "
            f"[{r.status.value}]"
        )
    typer.echo("")
    typer.echo(f"Report written to: {report_path}")


@app.command(name="stage10")
def stage10(
    baseline_metrics: str = typer.Option(
        ...,
        "--baseline-metrics",
        "-b",
        help="Path to Stage 4 metrics.json (contains cwe_macro_f1).",
    ),
    predictions: str = typer.Option(
        ...,
        "--predictions",
        "-p",
        help="Path to ModelPrediction JSONL (for exec-eval pass rate, etc.).",
    ),
    stage6_report: str = typer.Option(
        None,
        "--stage6-report",
        "-6",
        help="Path to Stage 6 eval_report.json (optional - loads from predictions if absent).",
    ),
    stage7_report: str = typer.Option(
        None,
        "--stage7-report",
        "-7",
        help="Path to Stage 7 regression_report.json (optional).",
    ),
    output_dir: str = typer.Option(
        "./output/stage10",
        "--output-dir",
        "-o",
        help="Directory to write the RegressionGateResult JSON.",
    ),
    max_f1_drop_percent: float = typer.Option(
        DEFAULT_MAX_F1_DROP_PERCENT,
        "--max-f1-drop-percent",
        help="Max permitted % drop in CWE Macro-F1 below the Stage 4 baseline.",
    ),
    min_exec_pass_rate: float = typer.Option(
        DEFAULT_MIN_EXEC_PASS_RATE,
        "--min-exec-pass-rate",
        help="Minimum exec pass rate (0.0 = no floor).",
    ),
    forgetting_threshold: float = typer.Option(
        DEFAULT_FORGETTING_THRESHOLD,
        "--forgetting-threshold",
        help="Forgetting-delta floor (Stage 7). Gate fails below this.",
    ),
    max_hallucination_rate: float = typer.Option(
        DEFAULT_MAX_HALLUCINATION_RATE,
        "--max-hallucination-rate",
        help="Maximum hallucination rate before the gate fails.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the Stage 10 regression gate.

    Compares the current model's CWE Macro-F1 against the Stage 4 baseline
    and checks Stage 7 forgetting, exec pass rate, and hallucination rate.
    The gate fails (exit code 1) if any check exceeds its threshold.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.ci.config import RegressionGateConfig
    from app.ci.gate import run_gate

    # If a Stage 6 report was not provided, build one from predictions +
    # gold-eval using the mock sandbox (no Docker / GPU needed).
    if stage6_report is None:
        from app.evaluation.runner import (
            EvalConfig,
            EvaluationRunner,
            load_predictions,
            load_samples,
        )

        gold_eval_path = predictions.replace("predictions", "gold_eval")
        if not os.path.exists(gold_eval_path):
            # Fall back to the bundled gold set.
            gold_eval_path = "eval/gold_set/gold.jsonl"

        config = EvalConfig(
            base_model="ci-regression-gate",
            sandbox_mode="mock",
            skip_tier4=True,
        )
        runner = EvaluationRunner(config=config)
        samples = load_samples(gold_eval_path)
        preds = load_predictions(predictions)

        typer.echo(f"Running Stage 6 (mock sandbox) on {len(preds)} predictions …")
        report = runner.run(samples, preds)

        os.makedirs(output_dir, exist_ok=True)
        stage6_report = os.path.join(output_dir, "eval_report.json")
        with open(stage6_report, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        typer.echo(f"Wrote Stage 6 report to: {stage6_report}")

    config = RegressionGateConfig(
        baseline_metrics_path=baseline_metrics,
        stage6_report_path=stage6_report,
        stage7_report_path=stage7_report,
        max_f1_drop_percent=max_f1_drop_percent,
        min_exec_pass_rate=min_exec_pass_rate,
        forgetting_threshold=forgetting_threshold,
        max_hallucination_rate=max_hallucination_rate,
    )

    result = run_gate(config)

    # Write gate result.
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, "gate_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result.model_dump_json(indent=2))

    # Print summary.
    typer.echo("")
    typer.echo("Stage 10 - Regression Gate")
    typer.echo(f"Run ID:             {result.run_id}")
    typer.echo(f"Timestamp:          {result.timestamp}")
    typer.echo(f"Overall status:     {result.status.value.upper()}")
    typer.echo("")
    typer.echo("Checks:")
    for check in result.checks:
        # Maps check status strings to terminal glyphs
        icon = _STATUS_GLYPHS.get(check.status.value, "[?]")
        typer.echo(f"  {icon} [{check.status.value.upper():>4s}] {check.name}: {check.message}")
    typer.echo("")
    typer.echo("Key metrics:")
    typer.echo(f"  Baseline CWE Macro-F1:   {result.baseline_cwe_macro_f1:.4f}")
    typer.echo(f"  Current CWE Macro-F1:    {result.current_cwe_macro_f1:.4f}")
    typer.echo(
        f"  F1 drop:                 {result.f1_drop_percent:+.2f}%  "
        f"(max allowed: {result.max_allowed_f1_drop_percent:.1f}%)"
    )
    if result.forgetting_delta is not None:
        typer.echo(
            f"  Forgetting delta:        {result.forgetting_delta:+.4f}  "
            f"(threshold: >={result.forgetting_threshold:+.4f})"
        )
    typer.echo(
        f"  Exec pass rate:          {result.exec_pass_rate:.4f}  "
        f"(min: {result.min_exec_pass_rate:.4f})"
    )
    typer.echo(
        f"  Hallucination rate:      {result.hallucination_rate:.4f}  "
        f"(max: {result.max_hallucination_rate:.4f})"
    )
    typer.echo("")
    typer.echo(f"Result written to: {result_path}")

    if not result.passed:
        typer.echo("")
        typer.echo(
            "[FAIL] Regression gate FAILED - checkpoint does not pass the quality bar.",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo("[OK] Regression gate PASSED - checkpoint is eligible for promotion.")


# -----------------------------------------------------------------------
# Stage 11 subcommands
# -----------------------------------------------------------------------


@app.command(name="stage11")
def stage11(
    docs_dir: str = typer.Option(
        "docs",
        "--docs-dir",
        "-d",
        help="Directory containing model_card.md, training_report.md, demo.py.",
    ),
    output_dir: str = typer.Option(
        "./output/stage11",
        "--output-dir",
        "-o",
        help="Directory to write Stage 11 artifacts (JSON sidecars, demo output).",
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        "-m",
        help="Name for the fine-tuned model (model card / report title). "
        "If omitted, derived from --base-model (e.g. 1.5B → vuln-triage-qwen2.5-coder-1.5b).",
    ),
    base_model: str = typer.Option(
        DEFAULT_BASE_MODEL,
        "--base-model",
        "-b",
        help="Base model that was fine-tuned.",
    ),
    training_method: str = typer.Option(
        "sft_qlora",
        "--training-method",
        "-t",
        help="Training method (sft_qlora, sft_full, lora, dpo).",
    ),
    lora_rank: int = typer.Option(
        64,
        "--lora-rank",
        "-r",
        help="LoRA rank used during training (0 = full-parameter SFT).",
    ),
    quant_method: str = typer.Option(
        None,
        "--quant-method",
        "-q",
        help="Quantization method (gptq, awq, gguf, none) or empty for unquantized.",
    ),
    quant_bit_width: int = typer.Option(
        None,
        "--quant-bits",
        help="Bit-width of the quantized model (e.g. 4).",
    ),
    training_data_size: int = typer.Option(
        5000,
        "--training-data-size",
        help="Number of samples in the training set.",
    ),
    run_demo: bool = typer.Option(
        True,
        "--run-demo/--no-demo",
        help="Run the mock-mode demo pipeline (Stages 4-6-7-10).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate and validate Stage 11 documentation deliverables.

    Creates the model card (``docs/model_card.md``), training report
    (``docs/training_report.md``), and demo script (``docs/demo.py``).
    Optionally runs the mock-mode demo pipeline (Stages 4-6-7-10) to
    populate the documents with real evaluation numbers.

    This command works **without** a GPU or model download - all evaluation
    uses mock backends (the same pattern as Stage 4-10 mock mode).
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from app.stage11.config import Stage11Config, _derive_model_name
    from app.stage11.generator import Stage11Generator

    # Resolve quant options
    quant_method_val = quant_method if quant_method else None
    quant_bit_width_val = quant_bit_width if quant_bit_width else None

    # Derive model name from base_model if not explicitly provided
    if model_name is None:
        model_name = _derive_model_name(base_model)

    config = Stage11Config(
        base_model=base_model,
        model_name=model_name,
        training_method=training_method,
        lora_rank=lora_rank if lora_rank > 0 else None,
        quant_method=quant_method_val,
        quant_bit_width=quant_bit_width_val,
        training_data_size=training_data_size,
        docs_dir=docs_dir,
        output_dir=output_dir,
    )

    gen = Stage11Generator(config)

    # Generate all deliverables
    typer.echo("Stage 11 - Documentation & Interview Package")
    typer.echo(f"Model name:     {model_name}")
    typer.echo(f"Base model:     {base_model}")
    typer.echo(f"Training method: {training_method}")
    typer.echo(f"Docs dir:       {docs_dir}")
    typer.echo(f"Output dir:     {output_dir}")
    typer.echo("")

    # Ensure deliverables exist
    results = gen.ensure_deliverables()
    typer.echo("Generated deliverables:")
    for name, path in results.items():
        typer.echo(f"  {name}: {path}")
    typer.echo("")

    # Optionally run demo
    if run_demo:
        typer.echo("Running mock-mode demo pipeline (Stages 4->6->7->10)...")
        demo_result = gen.run_demo()
        if demo_result.succeeded:
            typer.echo(f"Demo completed - {demo_result.num_gold_samples} gold samples evaluated")
            f1 = demo_result.metrics.get("tuned_cwe_macro_f1", "N/A")
            typer.echo(f"   CWE Macro-F1:     {f1}")
            typer.echo(f"   Exec pass rate:   {demo_result.metrics.get('exec_pass_rate', 'N/A')}")
            typer.echo(f"   Forgetting delta: {demo_result.metrics.get('forgetting_delta', 'N/A')}")
            typer.echo(f"   Gate status:      {demo_result.metrics.get('gate_status', 'N/A')}")
        else:
            typer.echo(f"[FAIL] Demo failed: {demo_result.error}", err=True)
        typer.echo("")

    # Validate
    if gen.validate_deliverables():
        typer.echo("[OK] All Stage 11 deliverables validated - present and non-empty.")
    else:
        typer.echo("[FAIL] Stage 11 deliverables validation FAILED.", err=True)
        raise typer.Exit(1)


# Legacy Stage 4 commands - keep existing behavior.


@app.command()
def baseline(
    gold_eval: str = typer.Option(
        ...,
        "--gold-eval",
        "-g",
        help="Path to gold-eval JSONL file (one VulnSample per line).",
    ),
    output_dir: str = typer.Option(
        "./output/stage4",
        "--output-dir",
        "-o",
        help="Directory to write predictions and metrics to.",
    ),
    strategy: str = typer.Option(
        "zero_shot",
        "--strategy",
        "-s",
        help="Prompting strategy: zero_shot or few_shot.",
    ),
    num_shots: int = typer.Option(
        3,
        "--num-shots",
        "-n",
        help="Number of in-context examples (few-shot only).",
    ),
    model: str = typer.Option(
        DEFAULT_BASE_MODEL_7B,
        "--model",
        "-m",
        help="Base model to evaluate.",
    ),
    temperature: float = typer.Option(
        0.2,
        "--temperature",
        "-t",
        help="Sampling temperature (lower = more deterministic).",
    ),
    max_new_tokens: int = typer.Option(
        2048,
        "--max-new-tokens",
        help="Maximum new tokens to generate per sample.",
    ),
    few_shot_examples: str = typer.Option(
        None,
        "--few-shot-examples",
        "-f",
        help="Path to Stage 3 train JSONL for few-shot examples (few-shot strategy only).",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use MockBackend (for testing - produces deterministic fake predictions).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run baseline evaluation: zero-shot or few-shot on the gold-eval set."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if strategy not in ("zero_shot", "few_shot"):
        typer.echo(
            f"Error: --strategy must be 'zero_shot' or 'few_shot', got '{strategy}'",
            err=True,
        )
        raise typer.Exit(1)

    config = BaselineConfig(
        strategy=strategy,
        num_shots=num_shots,
        base_model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    if mock:
        backend = MockBackend(
            responses={
                "CWE-89": '{"cwe_id": "CWE-89", "severity": "high", '
                '"explanation": "SQL injection via string concatenation.", '
                '"patch_diff": "--- a/app.py\\n+++ b/app.py\\n- old\\n+ new"}',
            },
            default='{"cwe_id": "CWE-89", "severity": "high", '
            '"explanation": "Mock explanation.", "patch_diff": ""}',
        )
    else:
        backend = None  # run_baseline will create a QwenBackend

    typer.echo(f"Running Stage 4 baseline (strategy={strategy}, model={model})")
    typer.echo(f"Gold-eval: {gold_eval}")
    typer.echo(f"Output:    {output_dir}")
    if strategy == "few_shot":
        if not few_shot_examples:
            typer.echo(
                "Warning: --few-shot-examples not provided, falling back to zero-shot",
                err=True,
            )
        else:
            typer.echo(f"Examples:  {few_shot_examples} ({num_shots} shots)")

    try:
        result = run_baseline(
            gold_eval_path=gold_eval,
            output_dir=output_dir,
            config=config,
            backend=backend,
            few_shot_examples_path=few_shot_examples,
        )
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc

    # Print summary
    typer.echo("")
    typer.echo(f"Run ID:       {result.run_id}")
    typer.echo(f"Predictions:  {result.num_predictions}")
    typer.echo(f"Parse failures: {result.num_parse_failures}")
    typer.echo(f"Total attempted: {result.total_attempted}")
    typer.echo("")
    typer.echo(_METRICS_HEADER)
    typer.echo(f"  CWE Macro-F1:          {result.metrics.cwe_macro_f1:.4f}")
    typer.echo(f"  CWE Micro Accuracy:    {result.metrics.cwe_micro_accuracy:.4f}")
    typer.echo(f"  Severity Accuracy:     {result.metrics.severity_accuracy:.4f}")
    typer.echo(f"  Hallucination Rate:    {result.metrics.hallucination_rate:.4f}")
    typer.echo(f"  Patch Coverage:        {result.metrics.patch_coverage:.4f}")
    typer.echo("")
    typer.echo("Per-class F1:")
    for cwe, stats in sorted(result.metrics.per_class.items()):
        typer.echo(
            f"  {cwe:10s}  P={stats['precision']:.4f}  "
            f"R={stats['recall']:.4f}  F1={stats['f1']:.4f}  n={stats['support']}"
        )
    typer.echo("")
    typer.echo(f"Predictions written to: {os.path.join(output_dir, 'predictions.jsonl')}")
    typer.echo(f"Metrics written to:     {os.path.join(output_dir, 'metrics.json')}")
    typer.echo(f"Manifest written to:    {os.path.join(output_dir, 'manifest.json')}")


@app.command()
def evaluate(
    predictions: str = typer.Option(
        ...,
        "--predictions",
        "-p",
        help="Path to predictions.jsonl from a previous baseline run.",
    ),
    gold_eval: str = typer.Option(
        ...,
        "--gold-eval",
        "-g",
        help="Path to gold-eval JSONL file for ground truth.",
    ),
    output_dir: str = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Optional directory to write re-computed metrics.json.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Re-compute metrics from saved predictions without re-running inference."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    # Load predictions
    preds: list[ModelPrediction] = []
    with open(predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            preds.append(ModelPrediction(**data))

    # Load gold-eval
    gold_samples: list[VulnSample] = []
    with open(gold_eval, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            gold_samples.append(VulnSample(**data))

    run_id = preds[0].run_id if preds else "re-evaluated"
    metrics = compute_metrics(preds, gold_samples, run_id=run_id)

    typer.echo(f"Loaded {len(preds)} predictions and {len(gold_samples)} gold-eval samples")
    typer.echo(f"Run ID: {run_id}")
    typer.echo("")
    typer.echo(_METRICS_HEADER)
    typer.echo(f"  CWE Macro-F1:          {metrics.cwe_macro_f1:.4f}")
    typer.echo(f"  CWE Micro Accuracy:    {metrics.cwe_micro_accuracy:.4f}")
    typer.echo(f"  Severity Accuracy:     {metrics.severity_accuracy:.4f}")
    typer.echo(f"  Hallucination Rate:    {metrics.hallucination_rate:.4f}")
    typer.echo(f"  Patch Coverage:        {metrics.patch_coverage:.4f}")
    typer.echo("")
    typer.echo("Per-class F1:")
    for cwe, stats in sorted(metrics.per_class.items()):
        typer.echo(
            f"  {cwe:10s}  P={stats['precision']:.4f}  "
            f"R={stats['recall']:.4f}  F1={stats['f1']:.4f}  n={stats['support']}"
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics.__dict__, f, indent=2)
        typer.echo(f"\nMetrics written to: {metrics_path}")


if __name__ == "__main__":
    app()
