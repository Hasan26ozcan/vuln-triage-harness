"""Tests for Stage 3's prompt template and patch-diff generation.

Verifies:
  - Prompt formatting includes the vulnerable code, language, and system prompt.
  - Static findings are rendered correctly (with and without findings).
  - Patch diff generation produces valid unified diffs.
  - make_patch_diff returns None when fixed_code is None.
  - The prompt template is injectable (custom template support).
"""

import re

from app.data.formatting.template import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    format_prompt,
    format_static_findings,
    make_patch_diff,
)
from app.schemas.vuln import StaticFinding, VulnSample


def _sample(
    id_: str = "vs_001",
    cwe: str = "CWE-89",
    language: str = "python",
    code: str = "cursor.execute('SELECT * FROM t WHERE id = ' + user_id)",
    fixed: str | None = "cursor.execute('SELECT * FROM t WHERE id = %s', (user_id,))",
    findings: list[StaticFinding] | None = None,
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name="org/repo",
        commit_sha="abc123",
        cve_id=f"CVE-2024-{id_}",
        cwe_id=cwe,
        severity="high",
        language=language,
        vulnerable_code=code,
        fixed_code=fixed,
        static_findings=findings if findings is not None else [],
        description="SQL injection via string concatenation.",
    )


# --- format_static_findings ---


def test_format_static_findings_renders_findings():
    findings = [
        StaticFinding(
            tool="semgrep",
            rule_id="python.sqli-string-concat",
            message="Possible SQL injection via string formatting.",
            line_range=(10, 12),
        )
    ]
    result = format_static_findings(findings)
    assert "python.sqli-string-concat" in result
    assert "Possible SQL injection" in result
    assert "lines 10-12" in result


def test_format_static_findings_empty_list_returns_placeholder():
    result = format_static_findings([])
    assert result == "_No static findings available._"


def test_format_static_findings_multiple_findings():
    findings = [
        StaticFinding(
            tool="semgrep",
            rule_id="rule_a",
            message="Finding A.",
            line_range=(1, 3),
        ),
        StaticFinding(
            tool="semgrep",
            rule_id="rule_b",
            message="Finding B.",
            line_range=(5, 7),
        ),
    ]
    result = format_static_findings(findings)
    assert "rule_a" in result
    assert "rule_b" in result
    # Two bullet points
    assert result.count("- [semgrep:") == 2


# --- make_patch_diff ---


def test_make_patch_diff_generates_unified_diff():
    vulnerable = "x = 1\ny = 2\nz = x + y\n"
    fixed = "x = 1\ny = 2\nz = x - y\n"
    diff = make_patch_diff(vulnerable, fixed)
    assert diff is not None
    assert "--- a/vulnerable_code.py" in diff
    assert "+++ b/fixed_code.py" in diff
    assert "-z = x + y" in diff
    assert "+z = x - y" in diff


def test_make_patch_diff_returns_none_when_fixed_code_is_none():
    result = make_patch_diff("some code", None)
    assert result is None


def test_make_patch_diff_empty_when_codes_identical():
    code = "def foo():\n    return 1\n"
    diff = make_patch_diff(code, code)
    # When there are no differences, unified_diff returns an empty iterator
    assert diff == ""


def test_make_patch_diff_preserves_new_function_addition():
    vulnerable = "def add(a, b):\n    return a + b\n"
    fixed = (
        "def add(a, b):\n    return a + b\n\n"
        "def safe_add(a, b):\n    if a < 0 or b < 0:\n        raise ValueError('negative')\n"
        "    return a + b\n"
    )
    diff = make_patch_diff(vulnerable, fixed)
    assert diff is not None
    assert "+" in diff  # has additions


# --- format_prompt ---


def test_format_prompt_includes_system_prompt():
    sample = _sample()
    prompt = format_prompt(sample)
    assert SYSTEM_PROMPT in prompt


def test_format_prompt_includes_vulnerable_code():
    code = "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)"
    sample = _sample(code=code)
    prompt = format_prompt(sample)
    assert code in prompt


def test_format_prompt_includes_language():
    sample = _sample(language="python")
    prompt = format_prompt(sample)
    assert "python" in prompt


def test_format_prompt_includes_static_findings():
    finding = StaticFinding(
        tool="semgrep",
        rule_id="python.sqli",
        message="Possible SQL injection.",
        line_range=(5, 7),
    )
    sample = _sample(findings=[finding])
    prompt = format_prompt(sample)
    assert "python.sqli" in prompt
    assert "Possible SQL injection" in prompt


def test_format_prompt_includes_placeholder_when_no_findings():
    sample = _sample(findings=[])
    prompt = format_prompt(sample)
    assert "_No static findings available._" in prompt


def test_format_prompt_uses_custom_template():
    custom_template = "Custom prompt for {language}: {vulnerable_code}"
    sample = _sample(language="javascript", code="const x = 1;")
    prompt = format_prompt(sample, template=custom_template)
    assert prompt.startswith(SYSTEM_PROMPT)
    assert "Custom prompt for javascript" in prompt
    assert "const x = 1;" in prompt


def test_format_prompt_includes_json_response_format():
    """The prompt should include the JSON response format hint so the model
    knows what structure to output."""
    sample = _sample()
    prompt = format_prompt(sample)
    assert '"cwe_id"' in prompt
    assert '"severity"' in prompt
    assert '"explanation"' in prompt
    assert '"patch_diff"' in prompt


def test_prompt_template_has_all_placeholders():
    """The default PROMPT_TEMPLATE must have placeholders for language,
    vulnerable_code, and static_findings."""
    placeholders = re.findall(r"\{(\w+)\}", PROMPT_TEMPLATE)
    assert "language" in placeholders
    assert "vulnerable_code" in placeholders
    assert "static_findings" in placeholders


def test_format_prompt_includes_task_description():
    """The prompt should mention the task (classification + patch)."""
    sample = _sample()
    prompt = format_prompt(sample)
    assert "Vulnerability Classification" in prompt
    assert "Patch Generation" in prompt


def test_format_prompt_javascript_language():
    code = "element.innerHTML = userInput;"
    sample = _sample(
        language="javascript",
        cwe="CWE-79",
        code=code,
        fixed="element.textContent = userInput;",
    )
    prompt = format_prompt(sample)
    assert "javascript" in prompt
    assert code in prompt
    # Code block fence should use javascript
    assert "```javascript" in prompt or "```\n" in prompt  # at minimum the code is in a fence
