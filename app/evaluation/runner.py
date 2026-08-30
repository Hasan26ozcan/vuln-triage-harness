"""Stage 6 — four-tier evaluation harness orchestrator.

This module ties together all four evaluation tiers into a single pipeline:

    Tier 1 (deterministic) → Tier 2 (static + embedding) → Tier 3 (exec) → Tier 4 (LLM judge)

The ``EvaluationRunner`` class orchestrates the pipeline. It accepts
injectable backends for each tier so that tests can verify integration
without spawning subprocesss or hitting LLM APIs.

Usage (CLI)::

    vuln-triage-harness eval-stage6 --samples data/gold/eval.jsonl --predictions preds.jsonl

Usage (programmatic)::

    runner = EvaluationRunner(runner_config)
    report = runner.run(samples, predictions)
    report.dump_json("report.json")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.evaluation.tier1_deterministic import DeterministicEvaluator
from app.evaluation.tier2_embedding_static import StaticSignalEvaluator
from app.evaluation.tier3_exec import ExecEvaluator, LocalSandboxRunner, MockSandboxRunner
from app.evaluation.tier4_llm_judge import LlmJudge, MockLlmJudgeBackend
from app.schemas.prediction_eval import (
    EvalMetrics,
    EvalReport,
    LlmJudgeScore,
    ModelPrediction,
)
from app.schemas.vuln import VulnSample
from app.security.paths import safe_read_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration for the four-tier evaluation runner.

    Attributes
    ----------
    base_model:
        The model being evaluated (e.g. ``"meta-llama/Meta-Llama-3.1-8B-Instruct"``).
    embedding_model:
        Optional sentence-transformers model name for Tier 2 embedding
        similarity. When ``None``, Tier 2 runs in static-only mode.
    sandbox_mode:
        ``"local"`` uses ``LocalSandboxRunner`` (subprocess, no Docker).
        ``"docker"`` uses ``DockerSandboxRunner`` (isolated container with
        read-only filesystem, no network, memory limit). Requires the
        ``docker`` Python package and a running Docker daemon.
        ``"mock"`` uses ``MockSandboxRunner`` for testing.
    llm_judge_model:
        Model name for Tier 4 LLM judge. When ``None``, uses mock backend.
    max_concurrent:
        Maximum number of concurrent sandbox tests (Tier 3).
    skip_tier4:
        If ``True``, skip Tier 4 (LLM judge) to save cost. Results will
        have empty ``llm_judge_scores``.
    skip_tier3:
        If ``True``, skip Tier 3 (exec) — useful for dry-run mode.
    """

    base_model: str = "unknown"
    embedding_model: str | None = None
    sandbox_mode: str = "mock"  # "local" | "docker" | "mock"
    llm_judge_model: str | None = None
    max_concurrent: int = 4
    skip_tier4: bool = False
    skip_tier3: bool = False


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _classify_prediction(
    true: str,
    pred: str | None,
    valid_cwes: Sequence[str],
) -> str | None:
    """Classify a single prediction as 'tp', 'fp', or None (miss).

    - 'tp' → pred matches true and pred is a valid CWE
    - 'fp' → pred is a valid CWE but ≠ true
    - None → pred is None, hallucinated, or unmatched
    """
    if pred is not None and pred in valid_cwes:
        return "tp" if pred == true else "fp"
    return None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision / recall / F1 from scalar counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _compute_cwe_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    valid_cwes: Sequence[str],
) -> float:
    """Compute macro-F1 for CWE classification.

    Only samples whose true label is in ``valid_cwes`` are scored.
    ``None`` predictions count as a miss for every class.
    """
    from collections import defaultdict

    tp: defaultdict[str, int] = defaultdict(int)
    fp: defaultdict[str, int] = defaultdict(int)
    fn: defaultdict[str, int] = defaultdict(int)

    for true, pred in zip(y_true, y_pred, strict=False):
        kind = _classify_prediction(true, pred, valid_cwes)
        if kind == "tp":
            tp[pred] += 1  # type: ignore[index]
        elif kind == "fp":
            fp[pred] += 1  # type: ignore[index]
            fn[true] += 1
        else:
            # pred is None or hallucinated
            fn[true] += 1

    f1s = [_prf(tp[cwe], fp[cwe], fn[cwe])[2] for cwe in valid_cwes]

    return sum(f1s) / len(f1s) if f1s else 0.0


def _compute_coverage(y_pred: list[str | None]) -> float:
    """Fraction of predictions that are non-None."""
    if not y_pred:
        return 0.0
    return sum(1 for p in y_pred if p is not None) / len(y_pred)


def _exec_rates(exec_results: list) -> dict[str, float]:
    """Compute pass/apply/build/hallucination rates from execution results."""
    n = len(exec_results) if exec_results else 1
    return {
        "exec_pass": sum(1 for r in exec_results if r.tests_pass_after_patch) / n,
        "patch_applies": sum(1 for r in exec_results if r.patch_applies_cleanly) / n,
        "build_ok": sum(1 for r in exec_results if r.build_succeeds) / n,
        "hallucination": sum(1 for r in exec_results if r.hallucinated_cwe) / n,
    }


def _count_tp_fp_fn(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    cwe: str,
) -> tuple[int, int, int]:
    """Count true-positives, false-positives, and false-negatives for one CWE."""
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == cwe and p == cwe)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != cwe and p == cwe)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == cwe and p != cwe)
    return tp, fp, fn


def _per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str | None],
    valid_cwes: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute per-class precision / recall / F1 for each CWE in *valid_cwes*."""
    result: dict[str, dict[str, float]] = {}
    for cwe in valid_cwes:
        tp, fp, fn = _count_tp_fp_fn(y_true, y_pred, cwe)
        prec, rec, f1 = _prf(tp, fp, fn)
        result[cwe] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    return result


def _llm_judge_averages(
    scores: list[LlmJudgeScore],
) -> tuple[float | None, float | None]:
    """Return (avg_explanation_quality, avg_patch_minimality) or (None, None)."""
    if not scores:
        return None, None
    eq = sum(s.explanation_quality for s in scores) / len(scores)
    pm = sum(s.patch_minimality for s in scores) / len(scores)
    return eq, pm


def compute_metrics(
    samples: list[VulnSample],
    tier1_results: list,
    tier2_results: list,
    exec_results: list,
    llm_judge_scores: list[LlmJudgeScore],
    predictions: list[ModelPrediction],
) -> EvalMetrics:
    """Compute aggregate ``EvalMetrics`` from per-tier results."""

    valid_cwes = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]

    y_true = [s.cwe_id for s in samples]
    tier1_preds = [r.predicted_cwe for r in tier1_results]
    tier2_preds = [r.predicted_cwe for r in tier2_results]
    model_preds = [p.predicted_cwe for p in predictions]

    t1_f1 = _compute_cwe_macro_f1(y_true, tier1_preds, valid_cwes)
    t1_cov = _compute_coverage(tier1_preds)
    t2_f1 = _compute_cwe_macro_f1(y_true, tier2_preds, valid_cwes)
    t2_cov = _compute_coverage(tier2_preds)
    model_f1 = _compute_cwe_macro_f1(y_true, model_preds, valid_cwes)

    rates = _exec_rates(exec_results)

    n_preds = len(predictions) if predictions else 1
    patch_cov = sum(1 for p in predictions if p.suggested_patch_diff.strip()) / n_preds

    avg_eq, avg_pm = _llm_judge_averages(llm_judge_scores)

    per_class = _per_class_f1(y_true, model_preds, valid_cwes)

    return EvalMetrics(
        num_samples=len(samples),
        num_predictions=len(predictions),
        tier1_cwe_macro_f1=round(t1_f1, 4),
        tier1_coverage=round(t1_cov, 4),
        tier2_cwe_macro_f1=round(t2_f1, 4),
        tier2_coverage=round(t2_cov, 4),
        model_cwe_macro_f1=round(model_f1, 4),
        exec_pass_rate=round(rates["exec_pass"], 4),
        patch_applies_rate=round(rates["patch_applies"], 4),
        build_succeeds_rate=round(rates["build_ok"], 4) if rates["build_ok"] else 0.0,
        hallucination_rate=round(rates["hallucination"], 4),
        avg_patch_coverage=round(patch_cov, 4),
        avg_explanation_quality=round(avg_eq, 4) if avg_eq is not None else None,
        avg_patch_minimality=round(avg_pm, 4) if avg_pm is not None else None,
        per_class=per_class,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Orchestrates the four-tier evaluation pipeline.

    Parameters
    ----------
    config:
        ``EvalConfig`` with runner settings.
    tier1_evaluator:
        Injected ``DeterministicEvaluator``. Defaults to one with
        ``DEFAULT_TIER1_RULES``.
    tier2_evaluator:
        Injected ``StaticSignalEvaluator``. Defaults to static-only (no
        embeddings).
    tier3_evaluator:
        Injected ``ExecEvaluator``. Defaults to ``MockSandboxRunner`` for
        testing; pass a real runner for production.
    tier4_evaluator:
        Injected ``LlmJudge``. Defaults to ``MockLlmJudgeBackend``.
    """

    def __init__(
        self,
        config: EvalConfig | None = None,
        tier1_evaluator: DeterministicEvaluator | None = None,
        tier2_evaluator: StaticSignalEvaluator | None = None,
        tier3_evaluator: ExecEvaluator | None = None,
        tier4_evaluator: LlmJudge | None = None,
    ):
        self.config = config or EvalConfig()

        # Tier 1 — deterministic
        self._tier1 = tier1_evaluator or DeterministicEvaluator()

        # Tier 2 — static + optional embedding
        if tier2_evaluator is not None:
            self._tier2 = tier2_evaluator
        else:
            self._tier2 = StaticSignalEvaluator(embedding_model=self.config.embedding_model)

        # Tier 3 — exec (sandbox)
        if tier3_evaluator is not None:
            self._tier3 = tier3_evaluator
        elif self.config.sandbox_mode == "local":
            self._tier3 = ExecEvaluator(sandbox_runner=LocalSandboxRunner())
        elif self.config.sandbox_mode == "docker":
            from app.evaluation.tier3_exec import DockerSandboxRunner

            self._tier3 = ExecEvaluator(sandbox_runner=DockerSandboxRunner())
        elif self.config.sandbox_mode == "mock" or self.config.skip_tier3:
            self._tier3 = ExecEvaluator(
                sandbox_runner=MockSandboxRunner(
                    default_result=None  # will produce defaults
                )
            )
        else:
            raise ValueError(
                f"Unknown sandbox_mode={self.config.sandbox_mode!r}. "
                "Use 'local', 'docker', or 'mock'."
            )

        # Tier 4 — LLM judge
        if tier4_evaluator is not None:
            self._tier4 = tier4_evaluator
        elif self.config.llm_judge_model:
            self._tier4 = LlmJudge(model=self.config.llm_judge_model)
        else:
            self._tier4 = LlmJudge(backend=MockLlmJudgeBackend())

    def run(
        self,
        samples: list[VulnSample],
        predictions: list[ModelPrediction],
    ) -> EvalReport:
        """Run the full four-tier evaluation pipeline.

        Parameters
        ----------
        samples:
            Gold-eval samples to evaluate against.
        predictions:
            Model predictions (one per sample, matched by ``sample_id``).

        Returns
        -------
        EvalReport
            Complete report with per-tier results, aggregate metrics, and
            run manifest.
        """
        run_id = f"stage6-{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        # ---------- Tier 1: Deterministic ----------
        logger.info("[%s] Tier 1: deterministic CWE classification", run_id)
        tier1_results = self._tier1.evaluate_all(samples)

        # ---------- Tier 2: Static signal + embedding ----------
        pred_map = {p.sample_id: p for p in predictions}
        logger.info("[%s] Tier 2: static signal + embedding similarity", run_id)
        tier2_results = self._tier2.evaluate_all(samples, predictions=pred_map)

        # ---------- Tier 3: Exec (sandbox) ----------
        if self.config.skip_tier3:
            logger.info("[%s] Tier 3: SKIPPED (skip_tier3=True)", run_id)
            exec_results = []
        else:
            logger.info("[%s] Tier 3: exec-based sandbox evaluation", run_id)
            exec_results = self._tier3.evaluate_all(samples, predictions)

        # ---------- Tier 4: LLM judge ----------
        if self.config.skip_tier4:
            logger.info("[%s] Tier 4: SKIPPED (skip_tier4=True)", run_id)
            llm_judge_scores = []
        else:
            logger.info("[%s] Tier 4: LLM-judge scoring", run_id)
            llm_judge_scores = self._tier4.evaluate_all(samples, predictions)

        # ---------- Metrics ----------
        metrics = compute_metrics(
            samples,
            tier1_results,
            tier2_results,
            exec_results,
            llm_judge_scores,
            predictions,
        )

        elapsed = time.time() - start_time
        manifest = {
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
            "elapsed_seconds": round(elapsed, 2),
            "config": {
                "base_model": self.config.base_model,
                "embedding_model": self.config.embedding_model or "none",
                "sandbox_mode": self.config.sandbox_mode,
                "llm_judge_model": (
                    self.config.llm_judge_model or getattr(self._tier4, "_model", None) or "mock"
                ),
                "skip_tier3": self.config.skip_tier3,
                "skip_tier4": self.config.skip_tier4,
            },
            "tier_order": [
                "tier1_deterministic",
                "tier2_embedding_static",
                "tier3_exec",
                "tier4_llm_judge",
            ],
        }

        return EvalReport(
            run_id=run_id,
            base_model=self.config.base_model,
            num_samples=len(samples),
            num_predictions=len(predictions),
            tier1_results=tier1_results,
            tier2_results=tier2_results,
            exec_results=exec_results,
            llm_judge_scores=llm_judge_scores,
            metrics=metrics,
            manifest=manifest,
        )


# ---------------------------------------------------------------------------
# Convenience: load samples and predictions from files
# ---------------------------------------------------------------------------


def load_samples(path: str | Path) -> list[VulnSample]:
    """Load VulnSamples from a JSONL file (one sample per line)."""
    samples: list[VulnSample] = []
    text = safe_read_text(path, allow_temp=True)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        samples.append(VulnSample(**data))
    return samples


def load_predictions(path: str | Path) -> list[ModelPrediction]:
    """Load ModelPredictions from a JSONL file (one prediction per line)."""
    preds: list[ModelPrediction] = []
    text = safe_read_text(path, allow_temp=True)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        preds.append(ModelPrediction(**data))
    return preds
