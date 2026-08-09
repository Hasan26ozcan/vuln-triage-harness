"""Stage 4 inference prompt builders.

There are two prompting strategies for the pre-fine-tuning baseline:

- **Zero-shot**: give the model the vulnerable code and ask it to classify +
  patch, with no in-context examples. The prompt is the same structure as
  Stage 3's training prompt (``SYSTEM_PROMPT`` + ``PROMPT_TEMPLATE``) — the
  only difference is we add an explicit "respond with JSON" suffix so the
  model knows the expected output format.

- **Few-shot**: prepend 1–3 ``InstructionExample`` records (from the train
  split) as in-context examples, then ask the model to respond to the
  target sample. This tests whether a handful of demonstrations can boost
  the base model's zero-shot performance — the key question Stage 4 answers
  before we invest in full fine-tuning.

Both strategies reuse Stage 3's ``format_prompt`` from ``app.data.formatting.template``
so the inference prompt is structurally identical to what the fine-tuned model
will see during training — making the before/after comparison apples-to-apples.
"""

from __future__ import annotations

from app.data.formatting.template import format_prompt
from app.schemas.dataset import InstructionExample
from app.schemas.vuln import VulnSample

# An explicit instruction appended to every inference prompt, telling the model
# exactly how to respond. This mirrors the ``### Response (JSON):`` block in
# Stage 3's PROMPT_TEMPLATE but is phrased as an instruction (not part of the
# training format).
RESPONSE_FORMAT_INSTRUCTION = (
    '\n\n'
    'Respond with a JSON object with these exact keys: '
    '"cwe_id" (e.g. "CWE-89"), '
    '"severity" (low, medium, high, or critical), '
    '"explanation" (1-2 sentences), '
    '"patch_diff" (a unified diff string, or null if no fix is possible).'
    '\n'
    'Put only the JSON inside a ```json code block and nothing else.'
)


def build_zero_shot_prompt(sample: VulnSample) -> str:
    """Build a zero-shot inference prompt for a single ``VulnSample``.

    Uses Stage 3's ``format_prompt`` (same system + task prompt as training)
    and appends the response-format instruction.
    """
    base_prompt = format_prompt(sample)
    return base_prompt + RESPONSE_FORMAT_INSTRUCTION


def build_few_shot_prompt(
    sample: VulnSample,
    examples: list[InstructionExample],
    num_shots: int = 3,
) -> str:
    """Build a few-shot inference prompt by prepending ``num_shots`` examples.

    Each example is rendered as an "Input:" / "Output:" pair so the model can
    see the expected format. The target sample is appended at the end with
    only "Input:" (no output), prompting the model to generate the response.

    Parameters
    ----------
    sample:
        The target ``VulnSample`` to classify and patch.
    examples:
        ``InstructionExample`` records (from the train split) to use as
        demonstrations. These should have ``target_cwe``, ``target_severity``,
        ``target_explanation``, and ``target_patch_diff`` populated.
    num_shots:
        Number of examples to prepend (clamped to ``len(examples)``).
    """
    if not examples:
        return build_zero_shot_prompt(sample)

    n = min(num_shots, len(examples))
    selected = examples[:n]

    parts: list[str] = []

    for i, ex in enumerate(selected):
        parts.append(f"--- Example {i + 1} ---")
        parts.append(f"Input:\n{ex.prompt}")
        parts.append("Output:")
        parts.append(_format_example_output(ex))

    # The actual target: just the input, no output — the model generates it.
    parts.append("--- Your Turn ---")
    parts.append(f"Input:\n{format_prompt(sample)}")
    parts.append("Output:")

    return "\n".join(parts) + RESPONSE_FORMAT_INSTRUCTION


def _format_example_output(ex: InstructionExample) -> str:
    """Format an ``InstructionExample``'s targets as a JSON response string.

    This is what the model should have produced for the example's prompt —
    it's placed after "Output:" in the few-shot context.
    """
    import json as json_mod

    output: dict[str, str | None] = {
        "cwe_id": ex.target_cwe,
        "severity": ex.target_severity,
        "explanation": ex.target_explanation,
        "patch_diff": ex.target_patch_diff,
    }
    return "```json\n" + json_mod.dumps(output, ensure_ascii=False, indent=2) + "\n```"
