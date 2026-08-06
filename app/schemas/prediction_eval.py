"""Model prediction + four-tier evaluation contracts (Stage 6 / Stage 7)."""

from pydantic import BaseModel


class ModelPrediction(BaseModel):
    sample_id: str
    run_id: str
    predicted_cwe: str
    predicted_severity: str
    suggested_patch_diff: str
    rationale: str


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


class RegressionSummary(BaseModel):
    """One row per checkpoint, aggregating all four eval tiers plus forgetting/cost."""

    run_id: str
    cwe_macro_f1: float
    exec_pass_rate: float
    hallucination_rate: float
    general_capability_delta: float  # forgetting check (Stage 7)
    cost_per_accepted_patch_usd: float
