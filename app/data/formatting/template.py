"""Stage 3, step 1: instruction prompt template and patch-diff generation.

Converts a ``VulnSample`` (raw vulnerable/fixed code pair + CWE + severity +
static findings) into the two parts an instruction-tuning model expects:

- **prompt**: the fully-formatted instruction that gets sent to the model
  (task description, the vulnerable code, and any static-analysis signal).
- **targets**: the expected outputs (CWE ID, severity, explanation, patch diff).

The prompt template follows the project's design philosophy: it is a
plain Python string (no Jinja2 dependency), and every component of the prompt
is a separate function so tests can verify each part in isolation.
"""

from __future__ import annotations

import difflib
import textwrap
from typing import Protocol

from app.schemas.vuln import StaticFinding, VulnSample

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# The system-level task description given to the model. This is appended to
# the prompt as a "system" prefix — in the Alpaca / OpenAI chat format it
# would be the system message.
SYSTEM_PROMPT = textwrap.dedent("""\
    You are a security-focused code assistant. Given a code snippet that
    contains a vulnerability, you will:
    1. Identify the CWE ID (e.g. CWE-89).
    2. Assess the severity (low, medium, high, or critical).
    3. Explain the vulnerability in 1-2 sentences.
    4. Provide a unified diff that fixes the vulnerability.
""").strip()


# Per-sample prompt template. ``{language}`` is filled from the sample's
# ``language`` field; ``{vulnerable_code}`` and ``{static_findings}`` are
# pre-rendered blocks inserted by ``format_prompt``.
PROMPT_TEMPLATE = textwrap.dedent("""\
    ### Task: Vulnerability Classification and Patch Generation

    ### Language: {language}

    ### Vulnerable Code ({language}):
    ```{language}
    {vulnerable_code}
    ```

    ### Static Analysis Findings:
    {static_findings}

    ### Instructions:
    1. Identify the CWE ID of the vulnerability.
    2. Assess the severity: low, medium, high, or critical.
    3. Explain the vulnerability in 1-2 sentences.
    4. Output a unified diff that fixes the vulnerability.

    ### Response (JSON):
    ```json
    {{
      "cwe_id": "...",
      "severity": "...",
      "explanation": "...",
      "patch_diff": "..."
    }}
    ```
    """).strip()


class PromptRenderer(Protocol):
    """Protocol for objects that can render a prompt from a VulnSample."""

    def render(self, sample: VulnSample) -> str: ...


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def format_static_findings(findings: list[StaticFinding]) -> str:
    """Render a list of static-analysis findings as a human-readable block.

    If the list is empty, a placeholder line is emitted so the model knows
    the section was considered, not omitted.
    """
    if not findings:
        return "_No static findings available._"

    lines: list[str] = []
    for f in findings:
        lines.append(
            f"- [{f.tool}:{f.rule_id}] {f.message} "
            f"(lines {f.line_range[0]}-{f.line_range[1]})"
        )
    return "\n".join(lines)


def make_patch_diff(vulnerable_code: str, fixed_code: str | None) -> str | None:
    """Generate a unified diff between vulnerable and fixed code.

    Uses ``difflib.unified_diff`` so there is no dependency on ``git`` or
    any external diff tool. Returns ``None`` when ``fixed_code`` is ``None``
    (some VulnSample records may not have a fix available).

    The diff has ``---`` / ``+++`` headers with ``a/`` and ``b/`` prefixes
    so it can be applied with ``git apply`` or ``patch``.
    """
    if fixed_code is None:
        return None

    vuln_lines = vulnerable_code.splitlines(keepends=False)
    fixed_lines = fixed_code.splitlines(keepends=False)

    diff = difflib.unified_diff(
        vuln_lines,
        fixed_lines,
        fromfile="a/vulnerable_code.py",
        tofile="b/fixed_code.py",
        lineterm="",
    )

    return "\n".join(diff)


def format_prompt(sample: VulnSample, template: str = PROMPT_TEMPLATE) -> str:
    """Render the full prompt for a single ``VulnSample``.

    Combines the ``SYSTEM_PROMPT`` prefix, the task description, the vulnerable
    code, and the static-analysis findings into a single string suitable for
    feeding to a code LLM.

    Parameters
    ----------
    sample:
        The ``VulnSample`` to render.
    template:
        Optional custom prompt template. Must contain ``{language}``,
        ``{vulnerable_code}``, and ``{static_findings}`` placeholders.
    """
    findings_str = format_static_findings(sample.static_findings)
    prompt_body = template.format(
        language=sample.language,
        vulnerable_code=sample.vulnerable_code,
        static_findings=findings_str,
    )
    return f"{SYSTEM_PROMPT}\n\n{prompt_body}"
