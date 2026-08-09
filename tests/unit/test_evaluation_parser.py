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

    return json.dumps({
        "cwe_id": cwe,
        "severity": severity,
        "explanation": explanation,
        "patch_diff": patch,
    })


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
    raw = (
        "Here is my analysis:\n"
        + _clean_response()
        + "\n\nThat's my conclusion."
    )
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
