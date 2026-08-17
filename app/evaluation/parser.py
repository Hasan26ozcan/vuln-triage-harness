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

    # Step 2: Parse the JSON.  If strict parsing fails (e.g. unescaped quotes
    # inside a patch_diff string), fall back to regex-based field extraction.
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        fallback = _try_fallback_extract(json_str, raw_output)
        if fallback is not None:
            data = fallback
        else:
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


def _is_template_json(json_str: str) -> bool:
    """Check if a JSON string looks like a template with placeholder values.

    The prompt template includes ``"cwe_id": "..."`` etc. as placeholders.
    The model sometimes echoes these literally instead of substituting real
    values.  We detect this so the caller can skip the template and try the
    next candidate.
    """
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    cwe = data.get("cwe_id")
    return isinstance(cwe, str) and cwe.strip() == "..."


def _extract_json(text: str) -> str | None:
    """Extract a JSON string from a possibly-messy LLM response.

    Strategy:
    1. Strip leading standalone backtick fences (``` without a language tag)
       that the model echoes from the prompt template.  Crucially, do NOT
       consume `` ```json `` fences — doing so was destroying the fence
       marker and causing "No JSON object found" errors.
    2. Find all `` ```json ... ``` `` blocks and try each one, skipping any
       that look like template placeholders (``"cwe_id": "..."``).
    3. If no fenced block yields valid content, search for all ``{...}``
       brace-matched objects and try each one (skipping templates).
    4. If nothing works, return None.
    """
    # 0. Strip leading standalone ``` blocks (e.g. the model echoes an empty
    #    backtick block before emitting its own ```json block).  We must NOT
    #    strip ```json — check for a language tag after the backticks.
    text = text.lstrip()
    while text.startswith("```"):
        rest = text[3:].lstrip()
        # If this was a ```json or other language-tagged fence, stop stripping.
        if re.match(r"[a-zA-Z]", rest):
            break
        text = rest
    text = text.strip()

    # 1. Try all ```json fenced blocks, preferring the first non-template one.
    fence_matches = list(_JSON_FENCE_RE.finditer(text))
    for fm in fence_matches:
        result = fm.group(1).strip()
        if not result:
            continue
        if not _is_template_json(result):
            return result
    # If every fence block was a template, fall through to brace matching.

    # 2. Try all {…} brace-matched objects (handles non-fenced JSON and
    #    template-first-then-real-JSON cases).
    for candidate in _find_json_objects(text):
        if not _is_template_json(candidate):
            return candidate

    # 3. Last resort: return the first fenced block even if it looks like a
    #    template (the caller will get a parse error with a helpful reason).
    if fence_matches:
        result = fence_matches[0].group(1).strip()
        if result:
            return result

    # 4. Last resort: return the first brace-matched object.
    candidates = _find_json_objects(text)
    if candidates:
        return candidates[0]

    return None


def _find_json_objects(text: str) -> list[str]:
    """Find all top-level ``{...}`` objects in *text* using brace counting.

    Unlike a naive ``find("{")``, this iterates over every ``{`` position so
    that multiple JSON objects in the same text are all discovered (e.g. when
    the model outputs a template followed by the real data).  Returns the
    candidate substrings; validity is checked by the caller.
    """
    candidates: list[str] = []
    for i, ch in enumerate(text):
        if ch == "{":
            depth = 0
            end = -1
            for j in range(i, len(text)):
                c = text[j]
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        end = j + 1
                        break
            if end > i:
                candidate = text[i:end].strip()
                if candidate:
                    candidates.append(candidate)
    return candidates


# Regex patterns for fallback field extraction when JSON is malformed.
_FALLBACK_CWE_RE = re.compile(r'"cwe_id"\s*:\s*"(CWE-\d+[A-Za-z0-9-]*)"', re.IGNORECASE)
_FALLBACK_SEVERITY_RE = re.compile(
    r'"severity"\s*:\s*"(low|medium|high|critical)"', re.IGNORECASE
)
_FALLBACK_EXPLANATION_RE = re.compile(
    r'"explanation"\s*:\s*"((?:[^"\\]|\\.|[^\n])*)"', re.IGNORECASE
)
_FALLBACK_PATCH_RE = re.compile(
    r'"patch_diff"\s*:\s*"((?:[^"\\]|\\.|[^\n])*)"', re.IGNORECASE
)


def _try_fallback_extract(json_str: str, raw_output: str) -> dict | None:
    """Extract fields via regex when strict JSON parsing fails.

    Models trained on code-patch data often produce ``patch_diff`` strings
    containing unescaped double quotes (from source code), which breaks
    ``json.loads``.  This helper uses targeted regexes to rescue the
    CWE ID, severity, explanation, and patch diff from the raw text.

    Returns a dict with the extracted fields, or ``None`` if the minimum
    required fields (cwe_id + severity) cannot be found.
    """
    text = json_str if json_str else raw_output

    cwe_match = _FALLBACK_CWE_RE.search(text)
    sev_match = _FALLBACK_SEVERITY_RE.search(text)

    # The fallback is only for JSON that was *close* — i.e. it had both
    # "cwe_id": "CWE-XX" and "severity": "low|medium|..." in quoted-key format.
    # If either is missing, the text is too malformed to salvage; let the
    # caller report the parse error.
    if not cwe_match or not sev_match:
        return None

    cwe = cwe_match.group(1)
    severity = sev_match.group(1).lower()

    data: dict[str, str | None] = {
        "cwe_id": cwe,
        "severity": severity,
        "explanation": "",
        "patch_diff": "",
    }

    # Try to extract explanation (may be truncated or malformed).
    exp_match = _FALLBACK_EXPLANATION_RE.search(text)
    if exp_match:
        data["explanation"] = exp_match.group(1)

    # Try to extract patch_diff (handles unescaped quotes by scanning
    # from the first patch_diff value quote to the end of the object).
    patch_match = _FALLBACK_PATCH_RE.search(text)
    if patch_match:
        data["patch_diff"] = patch_match.group(1)
    else:
        # Fallback: grab everything after patch_diff": up to the closing }
        m = re.search(r'"patch_diff"\s*:\s*"', text)
        if m:
            rest = text[m.end():]
            # Find the last } or \n}\n that terminates the JSON object.
            brace_pos = rest.rfind("}")
            if brace_pos > 0:
                data["patch_diff"] = rest[:brace_pos].rstrip()

    return data
