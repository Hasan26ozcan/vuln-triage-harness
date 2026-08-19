"""Unit tests for Stage 4 baseline metrics.

These verify:
  - CWE Macro-F1 is correctly computed per-class and macro-averaged.
  - Classes with zero predictions get F1=0 (not excluded from the average).
  - Severity accuracy is the fraction of correct severity predictions.
  - Hallucination rate counts out-of-scope CWE IDs (not just wrong-but-valid).
  - Patch coverage measures non-empty patch diffs.
  - Edge cases: empty inputs, all-correct, all-wrong, single class.
"""

from app.evaluation.metrics import (
    compute_cwe_macro_f1,
    compute_cwe_micro_accuracy,
    compute_hallucination_rate,
    compute_metrics,
    compute_patch_coverage,
    compute_severity_accuracy,
)
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample


def _gold_sample(
    id_: str = "s1",
    cwe: str = "CWE-89",
    severity: str = "high",
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name="org/repo",
        commit_sha="abc123",
        cve_id="CVE-2024-0001",
        cwe_id=cwe,
        severity=severity,
        language="python",
        vulnerable_code="vuln code",
        fixed_code="fixed code",
        description="Test vulnerability.",
    )


def _pred(
    sample_id: str,
    cwe: str = "CWE-89",
    severity: str = "high",
    patch: str = "--- diff",
    run_id: str = "r1",
) -> ModelPrediction:
    return ModelPrediction(
        sample_id=sample_id,
        run_id=run_id,
        predicted_cwe=cwe,
        predicted_severity=severity,
        suggested_patch_diff=patch,
        rationale="Test prediction.",
    )


# --- CWE Macro-F1 ---


def test_macro_f1_perfect_predictions():
    """All predictions correct → macro-F1 = 1.0."""
    gold = [
        _gold_sample(id_=f"s{i}", cwe=c, severity="high")
        for i, c in enumerate(["CWE-89", "CWE-79", "CWE-22"])
    ]
    preds = [_pred(f"s{i}", cwe=c) for i, c in enumerate(["CWE-89", "CWE-79", "CWE-22"])]

    f1, per_class = compute_cwe_macro_f1(preds, gold)
    assert f1 == 1.0
    for cwe in ["CWE-89", "CWE-79", "CWE-22"]:
        assert per_class[cwe]["f1"] == 1.0
        assert per_class[cwe]["precision"] == 1.0
        assert per_class[cwe]["recall"] == 1.0


def test_macro_f1_all_wrong():
    """All predictions wrong (different CWE) → macro-F1 = 0.0."""
    gold = [_gold_sample(id_="s1", cwe="CWE-89"), _gold_sample(id_="s2", cwe="CWE-79")]
    preds = [_pred("s1", cwe="CWE-79"), _pred("s2", cwe="CWE-89")]  # swapped

    f1, per_class = compute_cwe_macro_f1(preds, gold)
    assert f1 == 0.0
    for cwe in ["CWE-89", "CWE-79"]:
        assert per_class[cwe]["f1"] == 0.0
        assert per_class[cwe]["precision"] == 0.0
        assert per_class[cwe]["recall"] == 0.0


def test_macro_f1_class_with_no_predictions():
    """A class that exists in gold but has zero predictions should get F1=0
    (not excluded from the macro average)."""
    gold = [_gold_sample(id_="s1", cwe="CWE-89"), _gold_sample(id_="s2", cwe="CWE-79")]
    preds = [_pred("s1", cwe="CWE-89")]  # only CWE-89 predicted, CWE-79 has none

    f1, per_class = compute_cwe_macro_f1(preds, gold)
    assert "CWE-79" in per_class
    assert per_class["CWE-79"]["recall"] == 0.0
    assert per_class["CWE-79"]["f1"] == 0.0
    # Macro-F1 = (1.0 + 0.0) / 2 = 0.5
    assert f1 == 0.5


def test_macro_f1_partial():
    """Mixed correct/incorrect predictions → intermediate F1."""
    gold = [
        _gold_sample(id_="s1", cwe="CWE-89"),
        _gold_sample(id_="s2", cwe="CWE-89"),
        _gold_sample(id_="s3", cwe="CWE-79"),
    ]
    preds = [
        _pred("s1", cwe="CWE-89"),  # correct
        _pred("s2", cwe="CWE-89"),  # correct
        _pred("s3", cwe="CWE-89"),  # wrong (should be CWE-79)
    ]
    f1, per_class = compute_cwe_macro_f1(preds, gold)
    # CWE-89: tp=2, fp=1, fn=0 → P=2/3, R=1.0, F1=0.8
    assert per_class["CWE-89"]["precision"] == 0.6667
    assert per_class["CWE-89"]["recall"] == 1.0
    assert per_class["CWE-89"]["f1"] == 0.8
    # CWE-79: tp=0, fp=0, fn=1 → P=0, R=0, F1=0
    assert per_class["CWE-79"]["f1"] == 0.0
    # Macro-F1 = (0.8 + 0.0) / 2 = 0.4
    assert f1 == 0.4


def test_macro_f1_empty_gold():
    """Empty gold set → F1=0.0, empty per_class."""
    f1, per_class = compute_cwe_macro_f1([], [])
    assert f1 == 0.0
    assert per_class == {}


# --- CWE Micro Accuracy ---


def test_micro_accuracy_perfect():
    gold = [_gold_sample(id_="s1", cwe="CWE-89"), _gold_sample(id_="s2", cwe="CWE-79")]
    preds = [_pred("s1", cwe="CWE-89"), _pred("s2", cwe="CWE-79")]
    assert compute_cwe_micro_accuracy(preds, gold) == 1.0


def test_micro_accuracy_half():
    gold = [_gold_sample(id_="s1", cwe="CWE-89"), _gold_sample(id_="s2", cwe="CWE-79")]
    preds = [_pred("s1", cwe="CWE-89"), _pred("s2", cwe="CWE-89")]
    assert compute_cwe_micro_accuracy(preds, gold) == 0.5


def test_micro_accuracy_empty():
    assert compute_cwe_micro_accuracy([], []) == 0.0


# --- Severity Accuracy ---


def test_severity_accuracy_perfect():
    gold = [
        _gold_sample(id_="s1", cwe="CWE-89", severity="high"),
        _gold_sample(id_="s2", cwe="CWE-79", severity="medium"),
    ]
    preds = [
        _pred("s1", cwe="CWE-89", severity="high"),
        _pred("s2", cwe="CWE-79", severity="medium"),
    ]
    assert compute_severity_accuracy(preds, gold) == 1.0


def test_severity_accuracy_half():
    gold = [
        _gold_sample(id_="s1", cwe="CWE-89", severity="high"),
        _gold_sample(id_="s2", cwe="CWE-79", severity="medium"),
    ]
    preds = [
        _pred("s1", cwe="CWE-89", severity="high"),
        _pred("s2", cwe="CWE-79", severity="critical"),  # wrong severity
    ]
    assert compute_severity_accuracy(preds, gold) == 0.5


def test_severity_accuracy_empty():
    assert compute_severity_accuracy([], []) == 0.0


# --- Hallucination Rate ---


def test_hallucination_rate_no_hallucinations():
    """All predictions use valid CWE IDs → rate = 0.0."""
    preds = [
        _pred("s1", cwe="CWE-89"),
        _pred("s2", cwe="CWE-79"),
    ]
    assert compute_hallucination_rate(preds) == 0.0


def test_hallucination_rate_all_hallucinated():
    """All predictions use invalid CWE IDs → rate = 1.0."""
    preds = [
        _pred("s1", cwe="CWE-999"),
        _pred("s2", cwe="FAKE-123"),
    ]
    assert compute_hallucination_rate(preds) == 1.0


def test_hallucination_rate_half():
    """Half hallucinated, half valid."""
    preds = [
        _pred("s1", cwe="CWE-89"),  # valid
        _pred("s2", cwe="CWE-999"),  # hallucination
    ]
    assert compute_hallucination_rate(preds) == 0.5


def test_hallucination_rate_empty():
    assert compute_hallucination_rate([]) == 0.0


def test_hallucination_distinguishes_valid_vs_invalid_cwe():
    """Wrong-but-valid CWE is NOT a hallucination; made-up CWE IS."""
    preds = [
        _pred("s1", cwe="CWE-89"),  # wrong but valid → not hallucination
        _pred("s2", cwe="CWE-999"),  # not in scope → hallucination
    ]
    rate = compute_hallucination_rate(preds)
    assert rate == 0.5


# --- Patch Coverage ---


def test_patch_coverage_all_have_patches():
    preds = [
        _pred("s1", patch="--- diff ---"),
        _pred("s2", patch="--- diff ---"),
    ]
    assert compute_patch_coverage(preds) == 1.0


def test_patch_coverage_none_have_patches():
    preds = [
        _pred("s1", patch=""),
        _pred("s2", patch=""),
    ]
    assert compute_patch_coverage(preds) == 0.0


def test_patch_coverage_half():
    preds = [
        _pred("s1", patch="--- diff ---"),
        _pred("s2", patch=""),
    ]
    assert compute_patch_coverage(preds) == 0.5


def test_patch_coverage_empty():
    assert compute_patch_coverage([]) == 0.0


# --- compute_metrics (aggregate) ---


def test_compute_metrics_full_run():
    """End-to-end: 4 gold samples, 3 correct predictions, 1 hallucination."""
    gold = [
        _gold_sample(id_="s1", cwe="CWE-89", severity="high"),
        _gold_sample(id_="s2", cwe="CWE-79", severity="medium"),
        _gold_sample(id_="s3", cwe="CWE-22", severity="high"),
        _gold_sample(id_="s4", cwe="CWE-78", severity="critical"),
    ]
    preds = [
        _pred("s1", cwe="CWE-89", severity="high"),  # fully correct
        _pred("s2", cwe="CWE-79", severity="high"),  # CWE correct, severity wrong
        _pred("s3", cwe="CWE-89", severity="high"),  # CWE wrong (not CWE-22)
        _pred("s4", cwe="CWE-999", severity="high"),  # hallucination
    ]
    metrics = compute_metrics(preds, gold, run_id="baseline_test")

    assert metrics.run_id == "baseline_test"
    assert metrics.num_predictions == 4
    assert metrics.num_parsed == 4
    assert metrics.num_parse_failures == 0
    assert metrics.cwe_micro_accuracy == 0.5  # 2/4 correct (s1, s2)
    # s1: high=high correct, s3: high=high correct; s2 and s4 wrong
    assert metrics.severity_accuracy == 0.5
    assert metrics.hallucination_rate == 0.25  # 1/4 (s4: CWE-999 not in scope)
    assert metrics.patch_coverage == 1.0  # all have patches
    # Macro-F1:
    #   CWE-89: tp=1 (s1 correct), fp=1 (s3 predicted CWE-89 but gold is CWE-22),
    #           fn=0 → P=0.5, R=1.0, F1=0.6667
    #   CWE-79: tp=1 (s2 correct), fp=0, fn=0 → P=1.0, R=1.0, F1=1.0
    #   CWE-22: tp=0, fp=0, fn=1 (s3 gold CWE-22, pred CWE-89) → F1=0.0
    #   CWE-78: tp=0, fp=0, fn=1 (s4 gold CWE-78, pred CWE-999) → F1=0.0
    #   Macro = (0.6667 + 1.0 + 0.0 + 0.0) / 4 = 0.4167
    assert metrics.cwe_macro_f1 == 0.4167


def test_compute_metrics_with_parse_failures():
    """Parse failures (predicted_cwe='') are counted separately."""
    gold = [
        _gold_sample(id_="s1", cwe="CWE-89"),
        _gold_sample(id_="s2", cwe="CWE-79"),
    ]
    preds = [
        _pred("s1", cwe="CWE-89"),  # correct
        # s2 is a parse failure — predicted_cwe is empty
        ModelPrediction(
            sample_id="s2",
            run_id="r1",
            predicted_cwe="",
            predicted_severity="low",
            suggested_patch_diff="",
            rationale="[PARSE FAILURE: no JSON]",
        ),
    ]
    metrics = compute_metrics(preds, gold, run_id="test")

    assert metrics.num_predictions == 2
    assert metrics.num_parsed == 1
    assert metrics.num_parse_failures == 1
    # Only s1 was parsed and correct → micro accuracy = 0.5
    assert metrics.cwe_micro_accuracy == 0.5


def test_compute_metrics_empty():
    metrics = compute_metrics([], [], run_id="empty")
    assert metrics.cwe_macro_f1 == 0.0
    assert metrics.cwe_micro_accuracy == 0.0
    assert metrics.hallucination_rate == 0.0
