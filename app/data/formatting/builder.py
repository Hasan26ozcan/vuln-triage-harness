"""Stage 3, step 3: instruction-example builder with token budget enforcement.

Takes ``VulnSample`` records (already deduped and split by Stage 2) and turns
each into an ``InstructionExample`` — the format needed for SFT/DPO training.

The builder applies a single, deterministic prompt template to every sample,
extracts the ground-truth targets (CWE, severity, explanation, patch diff),
and estimates the total token count (prompt + targets so the model sees both
during training).

Samples whose estimated token count exceeds ``max_tokens`` are **dropped**
from the output — they would either be truncated (losing the fix) or
overflow the model's context window. The dropped samples are returned
separately so the caller can report coverage loss.
"""

from __future__ import annotations

import logging
import textwrap
import uuid
from dataclasses import dataclass

from app.data.formatting.template import PROMPT_TEMPLATE, format_prompt, make_patch_diff
from app.data.formatting.tokenizer import DEFAULT_MAX_TOKENS, TokenCounter
from app.schemas.dataset import InstructionExample
from app.schemas.vuln import VulnSample

logger = logging.getLogger(__name__)


@dataclass
class BuildResult:
    """Output of ``build_examples``.

    Attributes
    ----------
    examples:
        The ``InstructionExample`` records that fit within the token budget.
    dropped:
        Samples whose prompt + target exceeded ``max_tokens``. Each entry is
        a tuple of (sample_id, token_count) so the caller can report which
        samples were dropped and why.
    """

    examples: list[InstructionExample]
    dropped: list[tuple[str, int]]


def _build_explanation(sample: VulnSample) -> str:
    """Construct a brief explanation from the sample's description.

    Uses the ``description`` field set by Stage 1's NVD enrichment. Falls back
    to a generic CWE-based explanation if the description is empty.
    """
    desc = sample.description.strip()
    if desc:
        return desc
    # Fallback: describe the CWE without a specific detail
    return textwrap.dedent(f"""\
        This code exhibits {sample.cwe_id} ({sample.severity} severity).
        The vulnerable code fails to properly sanitize or validate input.""").strip()


def build_instruction_example(
    sample: VulnSample,
    token_counter: TokenCounter | None = None,
    template: str = PROMPT_TEMPLATE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> InstructionExample | None:
    """Build a single ``InstructionExample`` from a ``VulnSample``.

    Returns ``None`` if the estimated token count (prompt + all targets)
    exceeds ``max_tokens``. The caller should treat ``None`` as "dropped
    for budget" — not as an error.
    """
    counter = token_counter or TokenCounter()

    # --- Targets ---
    target_cwe = sample.cwe_id
    target_severity = sample.severity
    target_explanation = _build_explanation(sample)
    target_patch_diff = make_patch_diff(sample.vulnerable_code, sample.fixed_code)

    # --- Prompt ---
    prompt = format_prompt(sample, template=template)

    # --- Token budget check (prompt + all non-None targets) ---
    target_strs = [target_cwe, target_severity, target_explanation]
    if target_patch_diff is not None:
        target_strs.append(target_patch_diff)

    token_count = counter.count_prompt_and_target(prompt, *target_strs)

    if token_count > max_tokens:
        logger.debug(
            "Sample %s dropped: %d tokens > %d limit",
            sample.id,
            token_count,
            max_tokens,
        )
        return None

    return InstructionExample(
        id=f"ie_{uuid.uuid4().hex[:12]}",
        sample_id=sample.id,
        prompt=prompt,
        target_cwe=target_cwe,
        target_severity=target_severity,
        target_explanation=target_explanation,
        target_patch_diff=target_patch_diff,
        token_count_estimate=token_count,
    )


def build_examples(
    samples: list[VulnSample],
    token_counter: TokenCounter | None = None,
    template: str = PROMPT_TEMPLATE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BuildResult:
    """Build instruction examples for all samples, enforcing a token budget.

    Parameters
    ----------
    samples:
        ``VulnSample`` records (typically already split by Stage 2).
    token_counter:
        Optional injectable ``TokenCounter``. Defaults to a real
        ``TokenCounter`` (Qwen tokenizer) with heuristic fallback.
    template:
        Custom prompt template string.
    max_tokens:
        Maximum prompt + target tokens. Samples exceeding this are dropped.

    Returns
    -------
    ``BuildResult`` with ``examples`` (within budget) and ``dropped``
    (sample_id + token_count for each dropped sample).
    """
    counter = token_counter or TokenCounter()
    examples: list[InstructionExample] = []
    dropped: list[tuple[str, int]] = []

    for s in samples:
        example = build_instruction_example(
            s,
            token_counter=counter,
            template=template,
            max_tokens=max_tokens,
        )
        if example is None:
            # We don't have the token count here (it was computed inside
            # build_instruction_example). Re-compute for the drop record.
            prompt = format_prompt(s, template=template)
            targets = [s.cwe_id, s.severity, _build_explanation(s)]
            patch = make_patch_diff(s.vulnerable_code, s.fixed_code)
            if patch is not None:
                targets.append(patch)
            count = counter.count_prompt_and_target(prompt, *targets)
            dropped.append((s.id, count))
        else:
            examples.append(example)

    logger.info(
        "Built %d instruction examples (%d dropped for token budget > %d)",
        len(examples),
        len(dropped),
        max_tokens,
    )
    return BuildResult(examples=examples, dropped=dropped)
