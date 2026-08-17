"""Stage 11 — documentation & interview package.

Generates the three Stage 11 deliverables:

1. ``docs/model_card.md`` — model card (intended use, evaluation results,
   limitations, ethical considerations).
2. ``docs/training_report.md`` — technical training report (methodology,
   hyperparameters, loss curves, quantisation trade-offs, conclusions).
3. ``docs/demo.py`` — runnable demo script (mock mode, no GPU / ML deps).

The module works **without** a GPU or model download — it uses mock backends
(the same pattern as Stages 4–10) so CI can validate the deliverables on every
push.  The CLI subcommand ``stage11`` creates and validates all deliverables::

    python -m app.evaluation.cli stage11 \\
        --docs-dir docs --output-dir ./output/stage11 \\
        --model-name vuln-triage-qwen2.5-coder-1.5b \\
        --training-method sft_qlora --lora-rank 64

Usage (programmatic)::

    from app.stage11.config import Stage11Config
    from app.stage11.generator import Stage11Generator

    config = Stage11Config(model_name="my-finetuned-model", ...)
    gen = Stage11Generator(config)
    gen.ensure_deliverables()
    assert gen.validate_deliverables()
"""

from app.stage11.config import Stage11Config
from app.stage11.generator import (
    Stage11Generator,
    generate_demo_script,
    generate_model_card_markdown,
    generate_training_report_markdown,
    run_stage11,
)

__all__ = [
    "Stage11Config",
    "Stage11Generator",
    "generate_model_card_markdown",
    "generate_training_report_markdown",
    "generate_demo_script",
    "run_stage11",
]
