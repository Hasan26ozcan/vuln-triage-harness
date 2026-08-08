"""Stage 3: instruction-format dataset builder.

Converts ``VulnSample`` records (post-Stage 2 dedup + leakage-safe split)
into ``InstructionExample`` records ready for SFT/DPO training — applying
the prompt template, computing ground-truth targets, and enforcing a token
budget so no example overflows the model's context window.
"""

from app.data.formatting.builder import BuildResult, build_examples, build_instruction_example
from app.data.formatting.template import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    format_prompt,
    format_static_findings,
    make_patch_diff,
)
from app.data.formatting.tokenizer import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, TokenCounter

__all__ = [
    "PROMPT_TEMPLATE",
    "SYSTEM_PROMPT",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "TokenCounter",
    "format_prompt",
    "format_static_findings",
    "make_patch_diff",
    "build_instruction_example",
    "build_examples",
    "BuildResult",
]
