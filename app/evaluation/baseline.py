"""Stage 4 orchestration: pre-fine-tuning baseline evaluation.

This module ties together the four pieces of Stage 4 — gold-eval loading,
prompt building (zero-shot / few-shot), model generation, response parsing,
and metric computation — into a single ``run_baseline`` function.

The flow:
  1. Load gold-eval ``VulnSample`` records from a JSONL file.
  2. Optionally load few-shot ``InstructionExample`` records from a Stage 3
     output directory (only used when ``strategy="few_shot"``).
  3. For each gold sample:
     a. Build the inference prompt (zero-shot or few-shot).
     b. Call the model backend to generate a response.
     c. Parse the response into a ``ModelPrediction`` (or record a parse
        failure as a prediction with ``predicted_cwe=""``).
  4. Compute baseline metrics (CWE Macro-F1, Hallucination Rate, etc.).
  5. Write ``predictions.jsonl`` and ``metrics.json`` to the output directory.

The model backend is injectable — production runs use ``QwenBackend``
(``transformers``), tests use ``MockBackend``. This follows the same
injectable-backend pattern as ``TokenCounter`` (Stage 3) and
``EmbeddingBackend`` (Stage 2).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from app.evaluation.backends import ModelBackend
from app.evaluation.metrics import BaselineMetrics, compute_metrics
from app.evaluation.parser import ParseError, parse_prediction
from app.evaluation.prompt import build_few_shot_prompt, build_zero_shot_prompt
from app.schemas.dataset import InstructionExample
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for a baseline evaluation run.

    Attributes
    ----------
    strategy:
        ``"zero_shot"`` or ``"few_shot"``.
    num_shots:
        Number of in-context examples (few-shot only).
    base_model:
        HuggingFace model ID used for this run (for the manifest / provenance).
    max_new_tokens:
        Generation parameter passed to the backend.
    temperature:
        Generation parameter passed to the backend.
    """

    strategy: str = "zero_shot"
    num_shots: int = 3
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.2


@dataclass
class BaselineResult:
    """Full output of a Stage 4 baseline run."""

    run_id: str
    config: BaselineConfig
    predictions: list[ModelPrediction]
    parse_errors: list[ParseError]
    metrics: BaselineMetrics
    output_dir: str
    gold_samples: list[VulnSample]

    @property
    def total_attempted(self) -> int:
        return len(self.predictions) + len(self.parse_errors)

    @property
    def num_predictions(self) -> int:
        return len(self.predictions)

    @property
    def num_parse_failures(self) -> int:
        return len(self.parse_errors)


def load_gold_eval(path: str) -> list[VulnSample]:
    """Load gold-eval ``VulnSample`` records from a local JSONL file.

    Each line must be a JSON object matching the ``VulnSample`` schema.
    This follows the same pattern used by Stage 2's CLI
    (``_load_gold_eval`` in ``app/data/cleaning/cli.py``).
    """
    samples: list[VulnSample] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                samples.append(VulnSample(**payload))
            except Exception as exc:
                logger.warning("Skipping invalid gold-eval line %d: %s", line_num, exc)
    logger.info("Loaded %d gold-eval samples from %s", len(samples), path)
    return samples


def load_few_shot_examples(path: str, num_shots: int = 3) -> list[InstructionExample]:
    """Load instruction examples for few-shot prompting from a JSONL file.

    Typically this is the Stage 3 ``train.jsonl`` output. The first ``num_shots``
    examples are used as in-context demonstrations.
    """
    examples: list[InstructionExample] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                examples.append(InstructionExample(**payload))
            except Exception as exc:
                logger.warning("Skipping invalid example at line %d: %s", line_num, exc)
            if len(examples) >= num_shots:
                break
    logger.info("Loaded %d few-shot examples from %s", len(examples), path)
    return examples


def build_prompt(
    sample: VulnSample,
    config: BaselineConfig,
    few_shot_examples: list[InstructionExample] | None = None,
) -> str:
    """Dispatch to the correct prompt builder based on strategy."""
    if config.strategy == "few_shot":
        return build_few_shot_prompt(
            sample,
            examples=few_shot_examples or [],
            num_shots=config.num_shots,
        )
    return build_zero_shot_prompt(sample)


def run_baseline(
    gold_eval_path: str,
    *,
    output_dir: str = "./output/stage4",
    config: BaselineConfig | None = None,
    backend: ModelBackend | None = None,
    few_shot_examples_path: str | None = None,
) -> BaselineResult:
    """Run the complete Stage 4 baseline evaluation.

    Parameters
    ----------
    gold_eval_path:
        Path to the gold-eval JSONL file (one ``VulnSample`` per line).
    output_dir:
        Directory where ``predictions.jsonl``, ``metrics.json``, and
        ``manifest.json`` are written.
    config:
        ``BaselineConfig`` controlling strategy, model, and generation params.
        Defaults to zero-shot with the project's base model.
    backend:
        Injectable ``ModelBackend``. If None, a ``QwenBackend`` is created
        from the config's ``base_model``. Tests should inject ``MockBackend``.
    few_shot_examples_path:
        Path to a Stage 3 train JSONL for few-shot examples (few-shot strategy
        only). If None and strategy is few-shot, falls back to zero-shot.

    Returns
    -------
    ``BaselineResult`` with all predictions, parse errors, and metrics.
    """
    config = config or BaselineConfig()
    run_id = f"stage4_{config.strategy}_{uuid.uuid4().hex[:8]}"
    backend = backend or QwenBackend_stub(config)  # lazy import to avoid hard dep

    # Step 1: Load gold-eval samples
    gold_samples = load_gold_eval(gold_eval_path)
    if not gold_samples:
        raise RuntimeError(
            f"No gold-eval samples found at {gold_eval_path}. "
            "Expected a JSONL file with one VulnSample per line."
        )

    # Step 2: Load few-shot examples (if applicable)
    few_shot_examples: list[InstructionExample] | None = None
    if config.strategy == "few_shot":
        if few_shot_examples_path:
            few_shot_examples = load_few_shot_examples(
                few_shot_examples_path, num_shots=config.num_shots
            )
        if not few_shot_examples:
            logger.warning("No few-shot examples loaded — falling back to zero-shot for this run.")
            config.strategy = "zero_shot"

    # Step 3: Run inference on each gold sample
    predictions: list[ModelPrediction] = []
    parse_errors: list[ParseError] = []

    for i, sample in enumerate(gold_samples):
        prompt = build_prompt(sample, config, few_shot_examples)

        try:
            raw_output = backend.generate(prompt)
        except Exception as exc:
            logger.warning("Backend failed on sample %s: %s", sample.id, exc)
            parse_errors.append(
                ParseError(
                    sample_id=sample.id,
                    reason=f"Backend error: {exc}",
                    raw_output="",
                )
            )
            continue

        result = parse_prediction(raw_output, sample_id=sample.id, run_id=run_id)

        if isinstance(result, ParseError):
            parse_errors.append(result)
            logger.warning("Parse error on %s: %s", sample.id, result.reason)
            # Record a minimal prediction so the sample is counted
            predictions.append(
                ModelPrediction(
                    sample_id=sample.id,
                    run_id=run_id,
                    predicted_cwe="",  # empty = parse failure
                    predicted_severity="low",
                    suggested_patch_diff="",
                    rationale=f"[PARSE FAILURE: {result.reason}]",
                )
            )
        else:
            predictions.append(result)

        if (i + 1) % 10 == 0:
            logger.info("Stage 4: processed %d/%d samples", i + 1, len(gold_samples))

    # Step 4: Compute metrics
    metrics = compute_metrics(predictions, gold_samples, run_id=run_id)

    # Step 5: Write output
    os.makedirs(output_dir, exist_ok=True)

    # predictions.jsonl
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(p.model_dump_json() + "\n")
    logger.info("Wrote %d predictions to %s", len(predictions), pred_path)

    # parse_errors.jsonl
    if parse_errors:
        err_path = os.path.join(output_dir, "parse_errors.jsonl")
        with open(err_path, "w", encoding="utf-8") as f:
            for e in parse_errors:
                f.write(
                    json.dumps(
                        {
                            "sample_id": e.sample_id,
                            "reason": e.reason,
                            "raw_output": e.raw_output,
                        }
                    )
                    + "\n"
                )
        logger.info("Wrote %d parse errors to %s", len(parse_errors), err_path)

    # metrics.json
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2, default=str)
    logger.info("Wrote metrics to %s", metrics_path)

    # manifest.json
    manifest = {
        "run_id": run_id,
        "stage": 4,
        "strategy": config.strategy,
        "num_shots": config.num_shots,
        "base_model": config.base_model,
        "gold_eval_path": gold_eval_path,
        "few_shot_examples_path": few_shot_examples_path,
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "num_gold_samples": len(gold_samples),
        "num_predictions": len(predictions),
        "num_parse_failures": len(parse_errors),
        "metrics": asdict(metrics),
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Wrote manifest to %s", manifest_path)

    return BaselineResult(
        run_id=run_id,
        config=config,
        predictions=predictions,
        parse_errors=parse_errors,
        metrics=metrics,
        output_dir=output_dir,
        gold_samples=gold_samples,
    )


def run_baseline_on_predictions(
    predictions: list[ModelPrediction],
    gold_samples: list[VulnSample],
    run_id: str = "baseline",
) -> BaselineMetrics:
    """Compute metrics from predictions that were already generated.

    Useful for re-evaluating saved predictions without re-running inference.
    """
    return compute_metrics(predictions, gold_samples, run_id=run_id)


# ---------------------------------------------------------------------------
# Lazy import for the Qwen backend — avoids pulling in transformers at module
# import time (mirrors TokenCounter's lazy-load pattern in Stage 3).
# ---------------------------------------------------------------------------


def QwenBackend_stub(config: BaselineConfig) -> Any:
    """Create a QwenBackend from a BaselineConfig, importing lazily."""
    from app.evaluation.backends import QwenBackend

    return QwenBackend(
        model_name=config.base_model,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )
