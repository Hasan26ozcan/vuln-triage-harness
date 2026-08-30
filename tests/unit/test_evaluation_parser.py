"""Unit tests for Stage 4's response parser.

These verify:
  - parse_prediction correctly extracts JSON from clean model output.
  - Markdown code fences (```json) are handled.
  - JSON with surrounding text before/after is handled.
  - Missing fields fall back to defaults (empty patch_diff, etc.).
  - Invalid JSON returns ParseError.
  - Empty responses return ParseError.
  - Missing CWE ID with no CWE mention in text returns ParseError.
  - CWE extraction from text when 'cwe_id' field is missing but text mentions a CWE.
  - Invalid severity values return ParseError.
"""

from app.evaluation.parser import ParseError, parse_prediction


def _clean_response(
    cwe: str = "CWE-89",
    severity: str = "high",
    explanation: str = "SQL injection via string concatenation.",
    patch: str = "--- a/app.py\n+++ b/app.py\n- old\n+ new",
) -> str:
    """Generate a clean JSON model response."""
    import json

    return json.dumps(
        {
            "cwe_id": cwe,
            "severity": severity,
            "explanation": explanation,
            "patch_diff": patch,
        }
    )


# --- Clean JSON responses ---


def test_parse_clean_json_response():
    raw = _clean_response()
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError) is False
    assert result.sample_id == "s1"
    assert result.run_id == "r1"
    assert result.predicted_cwe == "CWE-89"
    assert result.predicted_severity == "high"
    assert "SQL injection" in result.rationale
    assert "--- a/app.py" in result.suggested_patch_diff


def test_parse_json_in_markdown_fence():
    """Model wraps JSON in ```json fences — parser should extract it."""
    raw = "```json\n" + _clean_response() + "\n```"
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"


def test_parse_json_with_surrounding_text():
    """Model adds explanatory text before/after the JSON block."""
    raw = "Here is my analysis:\n" + _clean_response() + "\n\nThat's my conclusion."
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"


# --- Missing / optional fields ---


def test_parse_missing_patch_diff_defaults_to_empty():
    """If patch_diff is omitted, it should default to empty string."""
    import json

    data = {
        "cwe_id": "CWE-79",
        "severity": "medium",
        "explanation": "XSS vulnerability.",
    }
    raw = json.dumps(data)
    result = parse_prediction(raw, sample_id="s2", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.suggested_patch_diff == ""


def test_parse_null_patch_diff_defaults_to_empty():
    """If patch_diff is explicitly null, it should become empty string."""
    import json

    data = {
        "cwe_id": "CWE-79",
        "severity": "medium",
        "explanation": "XSS vulnerability.",
        "patch_diff": None,
    }
    raw = json.dumps(data)
    result = parse_prediction(raw, sample_id="s2", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.suggested_patch_diff == ""


# --- Error cases ---


def test_parse_empty_response_returns_parse_error():
    result = parse_prediction("", sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "Empty" in result.reason


def test_parse_whitespace_only_response_returns_parse_error():
    result = parse_prediction("   \n  \t  ", sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "Empty" in result.reason


def test_parse_no_json_returns_parse_error():
    result = parse_prediction(
        "I couldn't find any vulnerability here.", sample_id="s1", run_id="r1"
    )
    assert isinstance(result, ParseError)
    assert "No JSON" in result.reason


def test_parse_invalid_json_returns_parse_error():
    result = parse_prediction("{cwe_id: CWE-89}", sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "JSON" in result.reason or "json" in result.reason.lower()


def test_parse_missing_cwe_id_no_cwe_in_text_returns_error():
    """If cwe_id field is missing and no CWE appears in text, it's an error."""
    import json

    data = {
        "severity": "high",
        "explanation": "Some vulnerability.",
    }
    raw = json.dumps(data)
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "cwe_id" in result.reason.lower()


def test_parse_missing_severity_returns_error():
    """If severity field is missing, it's an error."""
    import json

    data = {
        "cwe_id": "CWE-89",
        "explanation": "SQL injection.",
    }
    raw = json.dumps(data)
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "severity" in result.reason.lower()


def test_parse_invalid_severity_returns_error():
    """If severity is not one of the valid values, it's an error."""
    import json

    data = {
        "cwe_id": "CWE-89",
        "severity": "extreme",
        "explanation": "SQL injection.",
    }
    raw = json.dumps(data)
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "severity" in result.reason.lower()


def test_parse_cwe_extraction_from_text_when_field_missing():
    """If cwe_id field is missing but CWE-XX appears in explanation text,
    extract it as a fallback."""
    import json

    data = {
        "severity": "high",
        "explanation": "This is a CWE-89 vulnerability.",
    }
    raw = json.dumps(data)
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"


def test_parse_unbalanced_json_returns_error():
    """If the JSON has unbalanced braces, parser should detect it."""
    result = parse_prediction(
        '{"cwe_id": "CWE-89", "severity": "high"', sample_id="s1", run_id="r1"
    )
    assert isinstance(result, ParseError)


def test_parse_json_not_object_returns_error():
    """If the top-level JSON is an array or string, not an object."""
    result = parse_prediction('["CWE-89", "high"]', sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "object" in result.reason.lower() or "dict" in result.reason.lower()


# --- All valid CWE classes ---


def test_parse_all_valid_cwe_classes():
    """Parser should handle all 6 target CWE classes."""
    for cwe in ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]:
        raw = _clean_response(cwe=cwe)
        result = parse_prediction(raw, sample_id="s1", run_id="r1")
        assert hasattr(result, "predicted_cwe")
        assert result.predicted_cwe == cwe


def test_parse_all_valid_severities():
    """Parser should handle all severity levels."""
    for sev in ["low", "medium", "high", "critical"]:
        raw = _clean_response(severity=sev)
        result = parse_prediction(raw, sample_id="s1", run_id="r1")
        assert hasattr(result, "predicted_cwe")
        assert result.predicted_severity == sev


def test_parse_fenced_block_without_json_keyword():
    """````` without the 'json' language tag should still work."""
    raw = "```\n" + _clean_response() + "\n```"
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"


def test_parse_double_fence_echoed_fences():
    """Model echoes both the prompt template's closing fence and opening fence.

    The response looks like: ``` ```json { ... } ``` ```json
    The parser should strip the echoed fences and extract the valid JSON.
    """
    response = _clean_response()
    raw = "``` ```json\n" + response + "\n``` ```json"
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"
    assert result.predicted_severity == "high"


def test_parse_leading_empty_backtick_block_then_json():
    """Model starts with an empty ``` block before ```json (from eval observations).

    Previously the leading-``` stripper would consume the ``` from ```json,
    destroying the fence marker.  The parser should stop stripping when it
    encounters a language tag.
    """
    import json as json_mod

    payload = json_mod.dumps(
        {
            "cwe_id": "CWE-89",
            "severity": "high",
            "explanation": "SQL injection.",
            "patch_diff": "--- a/app.py\n+++ b/app.py\n- old\n+ new",
        }
    )
    raw = f"```\n\n```json\n{payload}\n```"
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "Predicted_cwe") or hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"
    assert result.predicted_severity == "high"


def test_parse_skip_template_placeholder_json():
    """If the model outputs a template (cwe_id: "...") followed by real JSON,
    skip the template and use the real data.
    """
    import json as json_mod

    template = json_mod.dumps(
        {
            "cwe_id": "...",
            "severity": "...",
            "explanation": "...",
            "patch_diff": "...",
        }
    )
    real = json_mod.dumps(
        {
            "cwe_id": "CWE-79",
            "severity": "medium",
            "explanation": "XSS vulnerability.",
            "patch_diff": "--- a/app.py\n+++ b/app.py\n- old\n+ new",
        }
    )
    raw = f"```json\n{template}\n```\n{real}"
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-79"
    assert result.predicted_severity == "medium"


def test_parse_fallback_for_unescaped_quotes_in_patch():
    """JSON with unescaped quotes in patch_diff should still parse via fallback.

    This happens when patch content contains code like ``'" + str(id)``
    where the unescaped ``"`` breaks JSON parsing.
    """
    # Manually craft malformed JSON: the patch_diff contains an unescaped quote
    raw = (
        "```json\n"
        "{\n"
        '  "cwe_id": "CWE-89",\n'
        '  "severity": "high",\n'
        '  "explanation": "SQL injection via string concat.",\n'
        '  "patch_diff": "--- a/app.py\\n- query = " + str(id)\\n+ query = %s\\n"\n'
        "}\n"
        "```"
    )
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"
    assert result.predicted_severity == "high"


# --- Edge cases not covered by the happy path ---


def test_parse_json_array_in_fence_returns_error():
    """Line 99: JSON array inside a fence parses to a list, not a dict."""
    raw = "Analysis:\n```json\n[1, 2, 3]\n```"
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert isinstance(result, ParseError)
    assert "object" in result.reason.lower() or "dict" in result.reason.lower()


def test_parse_null_explanation_defaults_to_empty():
    """Line 138: 'explanation': null should become empty string."""
    import json

    data = {
        "cwe_id": "CWE-89",
        "severity": "high",
        "explanation": None,
        "patch_diff": "patch",
    }
    raw = json.dumps(data)
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.rationale == ""


def test_parse_json_in_fence_after_text():
    """Line 177: fence in the middle of text reaches the return in _extract_json.

    When the fence is NOT at the start of the text, the leading-backtick
    stripping loop does not consume it, so the markdown-fence regex path is
    taken directly (rather than falling through to brace-matching).
    """
    import json

    payload = json.dumps(
        {
            "cwe_id": "CWE-89",
            "severity": "high",
            "explanation": "test",
            "patch_diff": "",
        }
    )
    raw = "Here is my analysis:\n```json\n" + payload + "\n```\nConclusion."
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"
    assert result.predicted_severity == "high"


# --- Edge cases for _extract_json fallback paths ---


def test_parse_empty_json_in_fence_skipped():
    """An empty ```json\n``` block is skipped (continue), then real JSON is found."""
    payload = '{"cwe_id": "CWE-89", "severity": "high", "explanation": "x", "patch_diff": ""}'
    raw = "```json\n\n```\n" + payload
    result = parse_prediction(raw, sample_id="s1", run_id="r1")
    assert hasattr(result, "predicted_cwe")
    assert result.predicted_cwe == "CWE-89"


def test_parse_all_template_fences_returns_first_anyway():
    """When ALL fenced blocks are templates, fall through to returning the
    first fenced block (line 223-225) so the caller gets a parse error with
    a helpful message."""
    from app.evaluation.parser import _extract_json

    # All fences contain template JSON
    raw = '```json\n{"cwe_id": "...", "severity": "..."}\n```'
    result = _extract_json(raw)
    # Returns the fenced content even though it's a template
    assert result is not None
    assert "..." in result


def test_extract_json_returns_first_brace_match_when_no_fences():
    """When there are no fenced blocks and no valid candidates pass template
    check, the last resort (line 228-230) returns the first brace-matched
    object."""
    from app.evaluation.parser import _extract_json

    # No fenced blocks, just raw JSON with braces — but it's a template
    raw = '{"cwe_id": "...", "severity": "..."}'
    result = _extract_json(raw)
    # Falls through to last resort: return first brace-matched object
    assert result is not None


def test_extract_json_returns_none_when_nothing_found():
    """When no JSON can be extracted at all, return None."""
    from app.evaluation.parser import _extract_json

    raw = "Just some random text with no JSON or braces at all"
    result = _extract_json(raw)
    assert result is None


# --- Fallback patch_diff extraction (lines 335-341) ---


def test_parse_fallback_extract_patch_diff_after_keyword():
    r"""When the _FALLBACK_PATCH_RE regex doesn't match but 'patch_diff':'"
    appears in the text, extract everything after it up to the closing brace.

    This covers line 335-341: the simpler regex ``r'"patch_diff"\s*:\s*"'"``
    finds the opening, then content is taken up to the last ``}`` in the text.
    """
    from app.evaluation.parser import _try_fallback_extract

    # patch_diff value has no closing quote → _FALLBACK_PATCH_RE won't match
    # (it requires both opening and closing "), but the simpler pattern finds it.
    text = '{"cwe_id": "CWE-89", "severity": "high", "patch_diff": "broken value\n}'
    result = _try_fallback_extract(text, text)
    assert result is not None
    assert result["cwe_id"] == "CWE-89"
    assert result["severity"] == "high"
    assert result["patch_diff"] == "broken value"


def test_parse_fallback_no_patch_match_returns_empty():
    """When patch_diff regex doesn't match and no closing brace found,
    patch_diff remains empty."""
    from app.evaluation.parser import _try_fallback_extract

    # Text with cwe_id and severity but patch_diff value with no closing brace
    text = (
        '{"cwe_id": "CWE-79", "severity": "medium", '
        '"patch_diff": "--- a/file.py no closing brace here'
    )
    result = _try_fallback_extract(text, text)
    assert result is not None
    assert result["cwe_id"] == "CWE-79"
    assert result["patch_diff"] == ""


def test_parse_fallback_no_cwe_or_severity_returns_none():
    """When neither cwe_id nor severity can be found, return None."""
    from app.evaluation.parser import _try_fallback_extract

    text = "This text has no CWE or severity fields"
    result = _try_fallback_extract(None, text)
    assert result is None


def test_parse_fallback_with_none_json_str():
    """When json_str is None, use raw_output for extraction."""
    from app.evaluation.parser import _try_fallback_extract

    text = '{"cwe_id": "CWE-89", "severity": "high", "explanation": "ok"}'
    result = _try_fallback_extract(None, text)
    assert result is not None
    assert result["cwe_id"] == "CWE-89"
