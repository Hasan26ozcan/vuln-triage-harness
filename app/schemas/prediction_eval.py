"""Model prediction + four-tier evaluation contracts (Stage 6 / Stage 7)."""

from pydantic import BaseModel


class ModelPrediction(BaseModel):
    sample_id: str
    run_id: str
    predicted_cwe: str
    predicted_severity: str
    suggested_patch_diff: str
    rationale: str


class Tier1Result(BaseModel):
    """Tier 1 — deterministic pattern-based CWE classification.

    Uses regex rules on the vulnerable code to classify the CWE without
    any model. A ``None`` ``predicted_cwe`` means no rule matched.
    """

    sample_id: str
    predicted_cwe: str | None
    confidence: float  # 0.0-1.0
    matched_pattern: str | None  # the rule description that fired
    num_patterns_matched: int


class Tier2Result(BaseModel):
    """Tier 2 — static-analysis signal + optional embedding similarity.

    Maps Semgrep findings to CWE IDs. When an embedding model is available,
    ``embedding_similarity`` compares the predicted patch against the gold
    fix (cosine similarity, 0.0-1.0). May be ``None`` when embeddings are
    not configured.
    """

    sample_id: str
    predicted_cwe: str | None
    confidence: float  # 0.0-1.0
    signal_sources: list[str]  # e.g. ["semgrep:python.sqli-string-concat"]
    embedding_similarity: float | None = None  # None when embeddings not used


class ExecEvalResult(BaseModel):
    """Tier 3 — exec-based evaluation result (Docker sandbox)."""

    prediction_id: str
    patch_applies_cleanly: bool
    build_succeeds: bool | None = None
    tests_pass_after_patch: bool | None = None
    cwe_classification_correct: bool
    hallucinated_cwe: bool  # made up a CWE ID that doesn't exist
    hallucinated_function_ref: bool  # referenced a function/file not present in the code


class LlmJudgeScore(BaseModel):
    """Tier 4 — LLM-judge score, used only for explanation quality / patch minimality."""

    prediction_id: str
    explanation_quality: float  # 0-1
    patch_minimality: float  # did it avoid unnecessary changes
    evaluator_model: str
    rationale: str


class EvalMetrics(BaseModel):
    """Aggregate metrics across all four evaluation tiers."""

    num_samples: int
    num_predictions: int
    tier1_cwe_macro_f1: float  # deterministic classification macro-F1
    tier1_coverage: float  # fraction of samples tier-1 could classify
    tier2_cwe_macro_f1: float  # static signal classification macro-F1
    tier2_coverage: float  # fraction of samples tier-2 could classify
    model_cwe_macro_f1: float  # model prediction macro-F1 (recomputed in Stage 6)
    exec_pass_rate: float  # fraction of patches that pass exec-eval tests
    patch_applies_rate: float  # fraction of patches that apply cleanly
    build_succeeds_rate: float  # fraction of patches that build/import
    hallucination_rate: float  # fraction of predicted CWEs not in scope
    avg_patch_coverage: float  # fraction of predictions with non-empty patch
    avg_explanation_quality: float | None = None  # Tier 4 LLM judge
    avg_patch_minimality: float | None = None  # Tier 4 LLM judge
    per_class: dict[str, dict[str, float]] = {}  # per-CWE F1 for model predictions


class EvalReport(BaseModel):
    """Full report from the Stage 6 four-tier evaluation harness.

    Aggregates per-sample/prediction results from all four tiers into a
    single report with computed metrics and a run manifest.
    """

    run_id: str
    base_model: str
    stage: int = 6
    num_samples: int
    num_predictions: int
    tier1_results: list[Tier1Result]
    tier2_results: list[Tier2Result]
    exec_results: list[ExecEvalResult]
    llm_judge_scores: list[LlmJudgeScore]
    metrics: EvalMetrics
    manifest: dict  # run provenance (config, paths, etc.)


class GeneralCapabilityResult(BaseModel):
    """Result of evaluating the model on a single general-capability task.

    These tasks are intentionally *not* security-related — they measure
    general code generation / reasoning ability (e.g. implement a
    well-known algorithm) to detect catastrophic forgetting after
    fine-tuning on vulnerability data.
    """

    task_id: str
    name: str
    passed: bool
    model_response: str  # raw model output
    exec_output: str | None = None
    exec_error: str | None = None


class GeneralCapabilityMetrics(BaseModel):
    """Aggregate metrics across all general-capability tasks (Stage 7)."""

    num_tasks: int
    num_passed: int
    execution_accuracy: float  # num_passed / num_tasks
    task_results: list[GeneralCapabilityResult] = []


class RegressionReport(BaseModel):
    """Stage 7 — regression / forgetting analysis report.

    Compares general-code-capability scores before (base model) and after
    (fine-tuned model) to detect catastrophic forgetting.

    A negative ``forgetting_delta`` means the fine-tuned model lost
    general coding ability relative to the base model.

    Attributes
    ----------
    run_id:
        Unique identifier for this regression analysis run.
    base_model:
        The pre-fine-tuning model name.
    tuned_model:
        The fine-tuned checkpoint identifier.
    base_metrics:
        ``GeneralCapabilityMetrics`` for the base model.
    tuned_metrics:
        ``GeneralCapabilityMetrics`` for the tuned model.
    forgetting_delta:
        ``tuned_metrics.execution_accuracy - base_metrics.execution_accuracy``.
        A negative value signals forgetting.
    manifest:
        Run provenance (config, task set, timestamps, etc.).
    """

    run_id: str
    base_model: str
    tuned_model: str
    base_metrics: GeneralCapabilityMetrics
    tuned_metrics: GeneralCapabilityMetrics
    forgetting_delta: float
    manifest: dict = {}


class RegressionSummary(BaseModel):
    """One row per checkpoint, aggregating all four eval tiers plus forgetting/cost.

    Combines Stage 6 evaluation metrics (``EvalMetrics``) with the Stage 7
    forgetting delta and a cost-per-patch estimate into a single summary
    row. This is the primary output consumed by the Stage 10 regression gate.
    """

    run_id: str
    cwe_macro_f1: float
    exec_pass_rate: float
    hallucination_rate: float
    general_capability_delta: float  # forgetting check (Stage 7)
    cost_per_accepted_patch_usd: float
