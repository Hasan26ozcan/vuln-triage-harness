"""Stage 4 baseline metric computation.

These metrics are computed on the gold-eval set and establish the "before"
baseline that Stage 5 (fine-tuning) and Stage 6 (full evaluation) compare
against. They are designed to be computable *without* exec-based testing —
you need only the model's structured output (``ModelPrediction``) and the
ground-truth (``VulnSample``) records.

All metrics here mirror the definitions in the README "Evaluation metrics"
table, adapted to the baseline case (no training cost, no exec-eval pass):

| Metric | Definition |
|---|---|
| CWE Macro-F1 | Per-class F1 averaged across CWE classes |
| Severity Accuracy | Share of predictions with correct severity label |
| Hallucination Rate | Share of predictions with a fabricated CWE ID |
| Patch Coverage | Share of predictions that produced a non-empty patch diff |

The full four-tier evaluation (deterministic → static → exec → LLM-judge) is
deferred to Stage 6 — Stage 4 only computes the metrics that don't require
running the proposed patches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """All metrics computed for a baseline run.

    Attributes
    ----------
    run_id:
        Identifier for this baseline run (same as ``ModelPrediction.run_id``).
    num_predictions:
        Total predictions attempted (includes parse failures with predicted_cwe="").
    num_parsed:
        Predictions that successfully parsed into a ``ModelPrediction``.
    num_parse_failures:
        Predictions where the model output couldn't be parsed.
    cwe_macro_f1:
        Per-class F1 averaged across all CWE classes present in the gold set.
    cwe_micro_accuracy:
        Overall fraction of predictions with the correct CWE.
    severity_accuracy:
        Fraction of predictions with the correct severity label.
    hallucination_rate:
        Fraction of predictions where the predicted CWE is not a valid
        target CWE (i.e., fabricated).
    patch_coverage:
        Fraction of parsed predictions that produced a non-empty patch diff.
    per_class:
        Per-CWE breakdown: {cwe_id: {"precision": f, "recall": f, "f1": f, "support": n}}
    """

    run_id: str
    num_predictions: int = 0
    num_parsed: int = 0
    num_parse_failures: int = 0
    cwe_macro_f1: float = 0.0
    cwe_micro_accuracy: float = 0.0
    severity_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    patch_coverage: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


# The set of valid CWE IDs for hallucination checking. A predicted CWE that
# is not in this set is considered hallucinated.
_VALID_CWE_IDS = frozenset({"CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"})


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision / recall / F1 from scalar counts."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def _class_counts(
    gold_samples: list[VulnSample],
    pred_lookup: dict[str, str],
    cwe: str,
) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, support) for a single CWE class."""
    tp = sum(1 for s in gold_samples if s.cwe_id == cwe and pred_lookup.get(s.id) == cwe)
    fp = sum(1 for s in gold_samples if s.cwe_id != cwe and pred_lookup.get(s.id) == cwe)
    fn = sum(1 for s in gold_samples if s.cwe_id == cwe and pred_lookup.get(s.id) != cwe)
    support = sum(1 for s in gold_samples if s.cwe_id == cwe)
    return tp, fp, fn, support


def compute_cwe_macro_f1(
    predictions: list[ModelPrediction],
    gold_samples: list[VulnSample],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute macro-averaged F1 across CWE classes.

    For each CWE class present in the gold set, computes precision, recall,
    and F1. Classes with no predictions get F1=0 (not excluded), so
    macro-averaging penalises models that ignore rare classes.

    Returns
    -------
    (macro_f1, per_class) — macro_f1 is the mean of per-class F1 scores,
    and per_class is ``{cwe_id: {"precision", "recall", "f1", "support"}}``.
    """
    if not gold_samples:
        return 0.0, {}

    pred_by_sample = {p.sample_id: p.predicted_cwe for p in predictions}
    gold_classes = {s.cwe_id for s in gold_samples}

    per_class: dict[str, dict[str, float]] = {}
    for cwe in sorted(gold_classes):
        tp, fp, fn, support = _class_counts(gold_samples, pred_by_sample, cwe)
        prec, rec, f1 = _prf(tp, fp, fn)
        per_class[cwe] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    f1_values = [per_class[c]["f1"] for c in per_class]
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    return round(macro_f1, 4), per_class


def compute_cwe_micro_accuracy(
    predictions: list[ModelPrediction],
    gold_samples: list[VulnSample],
) -> float:
    """Overall fraction of predictions with the correct CWE class."""
    if not gold_samples:
        return 0.0

    pred_by_sample: dict[str, str] = {p.sample_id: p.predicted_cwe for p in predictions}
    correct = sum(1 for s in gold_samples if pred_by_sample.get(s.id) == s.cwe_id)
    accuracy = correct / len(gold_samples)
    return round(accuracy, 4)


def compute_severity_accuracy(
    predictions: list[ModelPrediction],
    gold_samples: list[VulnSample],
) -> float:
    """Fraction of predictions where predicted severity matches ground truth."""
    if not gold_samples:
        return 0.0

    pred_by_sample: dict[str, str] = {p.sample_id: p.predicted_severity for p in predictions}
    correct = sum(1 for s in gold_samples if pred_by_sample.get(s.id) == s.severity)
    accuracy = correct / len(gold_samples)
    return round(accuracy, 4)


def compute_hallucination_rate(predictions: list[ModelPrediction]) -> float:
    """Fraction of predictions with a fabricated (out-of-scope) CWE ID.

    A CWE ID is hallucinated if it does not match any of the 6 target classes
    in the project's CWE scope. Note: a CWE ID that IS in scope but wrong
    (e.g. predicting CWE-79 when the truth is CWE-89) is a *misclassification*,
    not a hallucination — it's a real CWE class.
    """
    if not predictions:
        return 0.0

    hallucinated = sum(1 for p in predictions if p.predicted_cwe not in _VALID_CWE_IDS)
    return round(hallucinated / len(predictions), 4)


def compute_patch_coverage(predictions: list[ModelPrediction]) -> float:
    """Fraction of predictions that produced a non-empty patch diff.

    A model that can't produce a patch is still useful for classification,
    but patch coverage tells you how often it actually attempted a fix.
    """
    if not predictions:
        return 0.0

    has_patch = sum(1 for p in predictions if p.suggested_patch_diff.strip())
    return round(has_patch / len(predictions), 4)


def compute_metrics(
    predictions: list[ModelPrediction],
    gold_samples: list[VulnSample],
    run_id: str = "baseline",
) -> BaselineMetrics:
    """Compute all baseline metrics for a run.

    Parameters
    ----------
    predictions:
        All ``ModelPrediction`` records from the run. Predictions that failed
        to parse may still appear here with ``predicted_cwe=""`` — they are
        counted in ``num_parse_failures`` but their CWE is counted as
        hallucinated (not a valid CWE class).
    gold_samples:
        The ground-truth ``VulnSample`` records from the gold-eval set.
    run_id:
        Identifier for this baseline run.

    Returns
    -------
    A ``BaselineMetrics`` dataclass with all computed metrics.
    """
    # Separate parse failures from successful parses.
    # Parse failures are represented as ModelPredictions with predicted_cwe="".
    parsed = [p for p in predictions if p.predicted_cwe]
    parse_failures = [p for p in predictions if not p.predicted_cwe]

    macro_f1, per_class = compute_cwe_macro_f1(parsed, gold_samples)

    return BaselineMetrics(
        run_id=run_id,
        num_predictions=len(predictions),
        num_parsed=len(parsed),
        num_parse_failures=len(parse_failures),
        cwe_macro_f1=macro_f1,
        cwe_micro_accuracy=compute_cwe_micro_accuracy(parsed, gold_samples),
        severity_accuracy=compute_severity_accuracy(parsed, gold_samples),
        hallucination_rate=compute_hallucination_rate(parsed),
        patch_coverage=compute_patch_coverage(parsed),
        per_class=per_class,
    )
