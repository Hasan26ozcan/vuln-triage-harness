from app.evaluation.backends import (
    DEFAULT_BASE_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MockBackend,
    ModelBackend,
    QwenBackend,
)
from app.evaluation.baseline import (
    BaselineConfig,
    BaselineResult,
    run_baseline,
    run_baseline_on_predictions,
)

# Stage 7 — regression / forgetting analysis
from app.evaluation.general_capability import (
    DEFAULT_GENERAL_TASKS,
    CodeTestResult,
    CodeTestRunner,
    GeneralCapabilityEvaluator,
    GeneralCapabilityTask,
    LocalCodeTestRunner,
    MockCodeTestRunner,
    RegressionConfig,
    build_capability_prompt,
    build_regression_summary,
    estimate_cost_per_accepted_patch_usd,
    run_regression_analysis,
)
from app.evaluation.metrics import (
    BaselineMetrics,
    compute_cwe_macro_f1,
    compute_cwe_micro_accuracy,
    compute_hallucination_rate,
    compute_metrics,
    compute_patch_coverage,
    compute_severity_accuracy,
)
from app.evaluation.parser import ParseError, parse_prediction
from app.evaluation.prompt import (
    RESPONSE_FORMAT_INSTRUCTION,
    build_few_shot_prompt,
    build_zero_shot_prompt,
)

# Stage 6 — four-tier evaluation harness
from app.evaluation.runner import (
    EvalConfig,
    EvaluationRunner,
    load_predictions,
    load_samples,
)
from app.evaluation.runner import (
    compute_metrics as compute_stage6_metrics,
)
from app.evaluation.tier1_deterministic import (
    DEFAULT_TIER1_RULES,
    DeterministicEvaluator,
    PatternRule,
    classify_deterministic,
)
from app.evaluation.tier2_embedding_static import (
    DEFAULT_RULE_TO_CWE,
    EmbeddingBackend,
    StaticSignalEvaluator,
)
from app.evaluation.tier3_exec import (
    ExecEvaluator,
    LocalSandboxRunner,
    MockSandboxRunner,
    SandboxResult,
    SandboxRunner,
    TestGenerator,
    apply_unified_diff,
    check_hallucinated_function_ref,
)
from app.evaluation.tier4_llm_judge import (
    JUDGE_PROMPT,
    LlmJudge,
    LlmJudgeBackend,
    MockLlmJudgeBackend,
)

__all__ = [
    # Backends (Stage 4)
    "DEFAULT_BASE_MODEL",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "MockBackend",
    "ModelBackend",
    "QwenBackend",
    # Baseline orchestration (Stage 4)
    "BaselineConfig",
    "BaselineResult",
    "run_baseline",
    "run_baseline_on_predictions",
    # Metrics (Stage 4)
    "BaselineMetrics",
    "compute_cwe_macro_f1",
    "compute_cwe_micro_accuracy",
    "compute_hallucination_rate",
    "compute_metrics",
    "compute_patch_coverage",
    "compute_severity_accuracy",
    # Parser
    "ParseError",
    "parse_prediction",
    # Prompts
    "RESPONSE_FORMAT_INSTRUCTION",
    "build_few_shot_prompt",
    "build_zero_shot_prompt",
    # Stage 6 — Tier 1: Deterministic
    "DEFAULT_TIER1_RULES",
    "DeterministicEvaluator",
    "PatternRule",
    "classify_deterministic",
    # Stage 6 — Tier 2: Static + Embedding
    "DEFAULT_RULE_TO_CWE",
    "EmbeddingBackend",
    "StaticSignalEvaluator",
    # Stage 6 — Tier 3: Exec
    "ExecEvaluator",
    "LocalSandboxRunner",
    "MockSandboxRunner",
    "SandboxResult",
    "SandboxRunner",
    "TestGenerator",
    "apply_unified_diff",
    "check_hallucinated_function_ref",
    # Stage 6 — Tier 4: LLM Judge
    "JUDGE_PROMPT",
    "LlmJudge",
    "LlmJudgeBackend",
    "MockLlmJudgeBackend",
    # Stage 6 — Runner
    "EvalConfig",
    "EvaluationRunner",
    "compute_stage6_metrics",
    "load_predictions",
    "load_samples",
    # Stage 7 — Regression / forgetting analysis
    "DEFAULT_GENERAL_TASKS",
    "CodeTestResult",
    "CodeTestRunner",
    "GeneralCapabilityEvaluator",
    "GeneralCapabilityTask",
    "LocalCodeTestRunner",
    "MockCodeTestRunner",
    "RegressionConfig",
    "build_capability_prompt",
    "build_regression_summary",
    "estimate_cost_per_accepted_patch_usd",
    "run_regression_analysis",
]
