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

__all__ = [
    # Backends
    "DEFAULT_BASE_MODEL",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "MockBackend",
    "ModelBackend",
    "QwenBackend",
    # Baseline orchestration
    "BaselineConfig",
    "BaselineResult",
    "run_baseline",
    "run_baseline_on_predictions",
    # Metrics
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
]
