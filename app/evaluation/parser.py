"""Stage 4 model-response parser.

Given the raw text output of a code LLM (likely containing a JSON object,
possibly wrapped in `` ```json `` markdown fences, possibly with stray text
before or after), extract and validate the structured prediction into a
``ModelPrediction`` record.

Parse failures are non-fatal — a single bad response shouldn't abort the
entire baseline run. ``parse_prediction`` returns ``None`` and logs the
reason when parsing fails, so the caller can count failures and report
them. This mirrors how Stage 1's pipeline collects skip reasons rather than
aborting on a single bad sample.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from app.schemas.prediction_eval import ModelPrediction

logger = logging.getLogger(__name__)

# CWE ID pattern — matches things like CWE-89, CWE-79, etc.
_CWE_RE = re.compile(r"\bCWE-(\d{2,4})\b", re.IGNORECASE)

# Valid severity values (must match the Literal in VulnSample).
_VALID_SEVERITIES = {"low", "medium", "high", "critical"}

# Markdown code fence that might wrap the JSON block.
_JSON_FENCE_RE = re.compile(
    r"(?:```(?:json)?\s*)([\s\S]*?)(?:\s*```)",
    re.IGNORECASE,
)


@dataclass
class ParseError:
    """Details about why a model response failed to parse."""

    sample_id: str
    reason: str
    raw_output: str


def parse_prediction(
    raw_output: str,
    sample_id: str,
    run_id: str,
) -> ModelPrediction | ParseError:
    """Parse a raw LLM response into a ``ModelPrediction`` or ``ParseError``.

    The model is expected to output a JSON object with keys ``cwe_id``,
    ``severity``, ``explanation``, and ``patch_diff``. The parser is lenient:

    - It first looks for a `` ```json ... ``` `` block and extracts the JSON
      inside it. If none is found, it falls back to scanning the entire
      output for a JSON object.
    - Missing or null fields are handled: ``patch_diff`` defaults to an empty
      string (an empty patch is still a valid ``ModelPrediction``).
    - ``cwe_id`` and ``severity`` are validated against known formats.

    Returns
    -------
    ``ModelPrediction`` on success, ``ParseError`` on failure (not None —
    the caller can distinguish "failed to parse" from "valid prediction
    with empty patch").
    """
    if not raw_output or not raw_output.strip():
        return ParseError(
            sample_id=sample_id,
            reason="Empty model response",
            raw_output="",
        )

    # Step 1: Try to extract a JSON block from markdown fences.
    json_str = _extract_json(raw_output)

    if json_str is None:
        return ParseError(
            sample_id=sample_id,
            reason="No JSON object found in response",
            raw_output=raw_output[:500],
        )

    # Step 2: Parse the JSON.
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return ParseError(
            sample_id=sample_id,
            reason=f"Invalid JSON: {exc}",
            raw_output=raw_output[:500],
        )

    if not isinstance(data, dict):
        return ParseError(
            sample_id=sample_id,
            reason=f"Expected JSON object, got {type(data).__name__}",
            raw_output=raw_output[:500],
        )

    # Step 3: Extract and validate fields.
    predicted_cwe = data.get("cwe_id")
    if predicted_cwe is None or not str(predicted_cwe).strip():
        # Try to find a CWE mention in the raw text as a last resort.
        match = _CWE_RE.search(raw_output)
        if match:
            predicted_cwe = f"CWE-{match.group(1)}"
        else:
            return ParseError(
                sample_id=sample_id,
                reason="Missing or empty 'cwe_id' field and no CWE mentioned in text",
                raw_output=raw_output[:500],
            )

    predicted_cwe = str(predicted_cwe).strip()

    predicted_severity = data.get("severity")
    if predicted_severity is None:
        return ParseError(
            sample_id=sample_id,
            reason="Missing 'severity' field",
            raw_output=raw_output[:500],
        )
    predicted_severity = str(predicted_severity).strip().lower()
    if predicted_severity not in _VALID_SEVERITIES:
        return ParseError(
            sample_id=sample_id,
            reason=f"Invalid severity '{predicted_severity}', expected one of {_VALID_SEVERITIES}",
            raw_output=raw_output[:500],
        )

    explanation = data.get("explanation", "")
    if explanation is None:
        explanation = ""

    patch_diff = data.get("patch_diff", "")
    if patch_diff is None:
        patch_diff = ""
    patch_diff = str(patch_diff)

    return ModelPrediction(
        sample_id=sample_id,
        run_id=run_id,
        predicted_cwe=predicted_cwe,
        predicted_severity=predicted_severity,
        suggested_patch_diff=patch_diff,
        rationale=str(explanation),
    )


def _extract_json(text: str) -> str | None:
    """Extract a JSON string from a possibly-messy LLM response.

    Strategy:
    1. Look for a `` ```json ... ``` `` block — if found, return the inner text.
    2. If no fence, try to find the first ``{`` and match balanced braces.
    3. If neither works, return None.
    """
    # 1. Markdown fenced block
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. Raw JSON object with brace matching
    start = text.find("{")
    if start == -1:
        return None

    # Find the matching closing brace by counting depth.
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    return text[start:end].strip()
