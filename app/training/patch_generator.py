"""Stage 5+ — Separate Patch Generation Strategy.

This module implements a two-stage approach where CWE classification
and patch generation are handled by separate strategies:

1. **Stage 1 — Classification**: The fine-tuned model identifies the CWE
   ID and severity (trained in the existing LoRA experiments).

2. **Stage 2 — Patch Generation**: A separate, dedicated approach
   generates the code patch based on the classification result and
   the vulnerable code context.

Two patch generation strategies are provided:

- **Template-based**: Uses a structured template with the CWE type
  to guide the LLM to generate a specific patch.
- **Diff-injection**: Takes the model's own patch generation and
  applies verification to ensure correctness.

This separation matters because:
- Classification requires understanding *what* the vulnerability is
- Patch generation requires understanding *how* to fix it
- A single model can struggle with both tasks simultaneously
- Dedicated patch generation allows for more targeted training data

Usage::

    from app.training.patch_generator import PatchGenerator, PatchResult

    generator = PatchGenerator(strategy="template")
    result = generator.generate_patch(vulnerable_code, cwe_id="CWE-89")

    # Or with RAG context:
    result = generator.generate_patch_with_rag(
        vulnerable_code, cwe_id="CWE-89", rag_results=[...]
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PatchResult:
    """Result of a patch generation attempt."""

    cwe_id: str
    vulnerable_code: str
    patch_diff: str | None
    confidence: float
    strategy: str
    explanation: str
    verification_passed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cwe_id": self.cwe_id,
            "patch_diff": self.patch_diff,
            "confidence": self.confidence,
            "strategy": self.strategy,
            "explanation": self.explanation,
            "verification_passed": self.verification_passed,
            "metadata": self.metadata,
        }


class PatchGenerator:
    """Generates code patches for identified vulnerabilities.

    This class separates patch generation from CWE classification,
    allowing each task to use specialized prompting and training.

    Parameters
    ----------
    strategy:
        Patch generation strategy:
        - "template": Use structured CWE-specific templates
        - "rag": Use Retrieval-Augmented Generation
        - "both": Combine templates with RAG context
    cwe_templates:
        Custom CWE-specific prompt templates.
    """

    # CWE-specific patch generation templates
    # Each template guides the model to fix a specific type of vulnerability
    CWE_TEMPLATES: dict[str, str] = {
        "CWE-89": """### Task: Fix SQL Injection Vulnerability

**Vulnerability Type**: SQL Injection (CWE-89)
**Description**: User input is concatenated into SQL queries without parameterization.

**Fix Strategy**:
- Replace string concatenation with parameterized queries (prepared statements)
- Use query parameters (`?` or named parameters) instead of string interpolation
- Ensure ALL user-controlled inputs are parameterized, including in JOIN/WHERE clauses

**Code to fix**:
```python
{vulnerable_code}
```

**Expected fix pattern**: Replace `cursor.execute(f"SELECT ... {user_input}")`
with `cursor.execute("SELECT ... ?", (user_input,))`

Generate ONLY the patch diff (unified diff format):
```diff
--- a/vulnerable.py
+++ b/fixed.py
```
""",
        "CWE-79": """### Task: Fix Cross-Site Scripting (XSS) Vulnerability

**Vulnerability Type**: Cross-Site Scripting (CWE-79)
**Description**: Untrusted data is rendered in HTML without proper escaping.

**Fix Strategy**:
- Use HTML escaping (`html.escape()` or framework auto-escaping)
- For React/Vue/Angular: rely on framework auto-escaping by default
- Never use `innerHTML`, `dangerouslySetInnerHTML`, or `v-html` with user data
- Use `textContent` or `{{ variable }}` instead of HTML injection

**Code to fix**:
```python
{vulnerable_code}
```

**Expected fix pattern**: Wrap user input in escaping function or use framework's
auto-escaping mechanism.

Generate ONLY the patch diff (unified diff format):
```diff
--- a/vulnerable.py
+++ b/fixed.py
```
""",
        "CWE-22": """### Task: Fix Path Traversal Vulnerability

**Vulnerability Type**: Path Traversal (CWE-22)
**Description**: User-controlled input is used as a file path without sanitization.

**Fix Strategy**:
- Validate file paths against an allowlist of permitted directories
- Use `os.path.basename()` to strip directory components
- Use `os.path.realpath()` and check the resolved path starts with the base directory
- Never concatenate user input into file paths

**Code to fix**:
```python
{vulnerable_code}
```

**Expected fix pattern**: Validate that the resolved path stays within the allowed directory.
Use `os.path.realpath(path)` and check `startswith(allowed_dir)`.

Generate ONLY the patch diff (unified diff format):
```diff
--- a/vulnerable.py
+++ b/fixed.py
```
""",
        "CWE-78": """### Task: Fix OS Command Injection Vulnerability

**Vulnerability Type**: OS Command Injection (CWE-78)
**Description**: User input is passed to system commands without validation.

**Fix Strategy**:
- Never use `os.system()`, `os.popen()`, or `subprocess` with `shell=True`
- Use `subprocess.run()` with a list of arguments (no shell)
- Validate and sanitize all arguments
- Use explicit command paths and avoid shell metacharacters

**Code to fix**:
```python
{vulnerable_code}
```

**Expected fix pattern**: Replace shell commands with `subprocess.run(['cmd', arg1, arg2])`
where arguments are validated.

Generate ONLY the patch diff (unified diff format):
```diff
--- a/vulnerable.py
+++ b/fixed.py
```
""",
        "CWE-502": """### Task: Fix Deserialization of Untrusted Data (CWE-502)

**Vulnerability Type**: Insecure Deserialization (CWE-502)
**Description**: Untrusted data is deserialized using unsafe methods.

**Fix Strategy**:
- Replace `pickle.loads()` with `json.loads()` or other safe formats
- Use `yaml.safe_load()` instead of `yaml.load()`
- If pickle is required, use digital signatures to verify data integrity
- Never deserialize data from unauthenticated sources

**Code to fix**:
```python
{vulnerable_code}
```

**Expected fix pattern**: Replace unsafe deserialization with safe alternatives.
Use `json.loads()`, `yaml.safe_load()`, or add signature verification.

Generate ONLY the patch diff (unified diff format):
```diff
--- a/vulnerable.py
+++ b/fixed.py
```
""",
        "CWE-190": """### Task: Fix Integer Overflow Vulnerability

**Vulnerability Type**: Integer Overflow (CWE-190)
**Description**: Arithmetic operations can overflow without bounds checking.

**Fix Strategy**:
- Add bounds checking before arithmetic operations
- Use `math.add`/`math.multiply` with overflow checking where available
- Validate input ranges and clamp values
- Use larger integer types if available (Python's int is arbitrary precision,
  but check for logical overflow in downstream operations)
- Use assertions or explicit checks for boundary conditions

**Code to fix**:
```python
{vulnerable_code}
```

**Expected fix pattern**: Add bounds checking before the arithmetic operation.
Example: `if value > MAX_VALUE: value = MAX_VALUE`.

Generate ONLY the patch diff (unified diff format):
```diff
--- a/vulnerable.py
+++ b/fixed.py
```
""",
    }

    def __init__(
        self,
        strategy: str = "template",
        use_rag: bool = False,
        cwe_templates: dict[str, str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.use_rag = use_rag
        self.cwe_templates = cwe_templates or self.CWE_TEMPLATES

    def generate_patch(
        self,
        vulnerable_code: str,
        cwe_id: str,
        explanation: str = "",
        rag_results: list[Any] | None = None,
    ) -> PatchResult:
        """Generate a patch for a vulnerability.

        Parameters
        ----------
        vulnerable_code:
            The vulnerable code snippet.
        cwe_id:
            The CWE classification (e.g., "CWE-89").
        explanation:
            Explanation of the vulnerability (optional).
        rag_results:
            Retrieved similar examples for RAG context (optional).

        Returns
        -------
        A ``PatchResult`` containing the generated patch diff.
        """
        template = self.cwe_templates.get(cwe_id)

        if template is None:
            logger.warning("No template for %s — using generic patch prompt", cwe_id)
            prompt = self._generic_patch_prompt(vulnerable_code, cwe_id, explanation)
        elif self.use_rag and rag_results:
            prompt = self._build_rag_prompt(template, vulnerable_code, rag_results)
        else:
            prompt = self._build_template_prompt(template, vulnerable_code, explanation)

        # The actual LLM call would happen here — this method returns
        # the prompt that should be sent to the LLM.
        # In production, this would call the fine-tuned model.
        patch_diff = self._generate_patch_diff(vulnerable_code, cwe_id, prompt)

        return PatchResult(
            cwe_id=cwe_id,
            vulnerable_code=vulnerable_code,
            patch_diff=patch_diff,
            confidence=self._estimate_confidence(cwe_id, rag_results),
            strategy=self.strategy,
            explanation=explanation or f"Auto-generated patch for {cwe_id}",
            verification_passed=False,  # Would be verified by tier3_exec
        )

    def generate_patch_with_rag(
        self,
        vulnerable_code: str,
        cwe_id: str,
        retrieval_results: list[Any],
    ) -> PatchResult:
        """Generate a patch using RAG-augmented context.

        Parameters
        ----------
        vulnerable_code:
            The vulnerable code snippet.
        cwe_id:
            The CWE classification.
        retrieval_results:
            List of ``RetrievalResult`` from the RAG retriever.

        Returns
        -------
        A ``PatchResult`` with the generated patch.
        """
        prompt = self._build_rag_prompt(
            self.cwe_templates.get(cwe_id, ""),
            vulnerable_code,
            retrieval_results,
        )
        patch_diff = self._generate_patch_diff(vulnerable_code, cwe_id, prompt)

        return PatchResult(
            cwe_id=cwe_id,
            vulnerable_code=vulnerable_code,
            patch_diff=patch_diff,
            confidence=self._estimate_confidence(cwe_id, retrieval_results),
            strategy="rag",
            explanation=f"RAG-augmented patch for {cwe_id}",
            verification_passed=False,
            metadata={
                "rag_results_count": len(retrieval_results),
                "rag_scores": [r.score for r in retrieval_results],
            },
        )

    def _build_template_prompt(self, template: str, vulnerable_code: str, explanation: str) -> str:
        """Fill in a CWE-specific template with the vulnerable code."""
        return template.format(
            vulnerable_code=vulnerable_code,
            explanation=explanation,
        )

    def _build_rag_prompt(self, template: str, vulnerable_code: str, rag_results: list[Any]) -> str:
        """Build a prompt with RAG context appended to the CWE template."""
        base = self._build_template_prompt(template, vulnerable_code, "")

        rag_section = "\n\n### Similar Vulnerability Examples:\n"
        for i, result in enumerate(rag_results, 1):
            text = getattr(result, "text", str(result))
            score = getattr(result, "score", 0.0)
            rag_section += f"--- Example {i} (similarity: {score:.3f}) ---\n{text[:500]}\n\n"

        rag_section += "### Use these examples as reference for your patch.\n"
        return base + rag_section

    def _generic_patch_prompt(self, vulnerable_code: str, cwe_id: str, explanation: str) -> str:
        """Fallback prompt for CWE types without a specific template."""
        return (
            f"### Task: Fix {cwe_id} vulnerability\n\n"
            f"### Vulnerability: {explanation}\n\n"
            f"### Vulnerable Code:\n```python\n{vulnerable_code}\n```\n\n"
            f"### Instructions:\n"
            f"Generate a unified diff patch that fixes this {cwe_id} vulnerability. "
            f"Focus on removing the unsafe pattern and replacing it with a secure alternative.\n\n"
            f"### Patch:\n```diff\n"
        )

    def _generate_patch_diff(self, vulnerable_code: str, cwe_id: str, prompt: str) -> str | None:
        """Generate the actual patch diff.

        In production, this would call the fine-tuned model.
        For now, it returns a structured prompt that can be sent
        to the LLM for patch generation.

        Returns
        -------
        The patch diff string (or None if generation fails).
        The caller should send ``prompt`` to the LLM and parse the response.
        """
        # This is a stub — in production, the prompt would be sent to
        # the fine-tuned model and the response parsed for a diff.
        # For the template-based strategy, the model is expected to
        # generate a diff within the ```diff ... ``` block.
        logger.info(
            "Patch generation prompt ready for %s. Send prompt to LLM for patch diff generation.",
            cwe_id,
        )
        return None  # Placeholder — actual generation happens at LLM call time

    def _estimate_confidence(self, cwe_id: str, rag_results: list[Any] | None) -> float:
        """Estimate confidence in the generated patch."""
        base_confidence = 0.75 if cwe_id in self.cwe_templates else 0.50
        if rag_results:
            base_confidence += 0.1 * min(len(rag_results), 5)
        return min(base_confidence, 0.95)

    def get_cwe_template(self, cwe_id: str) -> str | None:
        """Get the CWE-specific patch template."""
        return self.cwe_templates.get(cwe_id)

    def add_cwe_template(self, cwe_id: str, template: str) -> None:
        """Add or update a CWE-specific patch template."""
        self.cwe_templates[cwe_id] = template
        logger.info("Added/updated template for %s", cwe_id)


# ------------------------------------------------------------------
# Patch verification helper (used after patch generation)
# ------------------------------------------------------------------


def verify_patch_pattern(patch_diff: str, cwe_id: str) -> dict[str, Any]:
    """Basic pattern verification of a generated patch.

    Checks that the patch contains expected safe patterns for the
    given CWE type. This is a lightweight check before running
    the full test suite (tier3_exec).

    Returns
    -------
    Dict with ``passed`` (bool) and ``issues`` (list of strings).
    """
    issues: list[str] = []
    patch_lower = patch_diff.lower()

    checks: dict[str, list[str]] = {
        "CWE-89": ["execute(", "?", "parameter"],  # Should use parameterized queries
        "CWE-79": ["escape", "textContent", "safe"],  # Should have escaping
        "CWE-22": ["basename", "realpath", "allowlist", "startswith"],  # Should validate paths
        "CWE-78": ["shell=False", "subprocess.run", "subprocess.call"],  # Should avoid shell=True
        "CWE-502": ["safe_load", "json.loads", "sign"],  # Should use safe deserialization
        "CWE-190": ["max", "min", "bound", "check", "clamp"],  # Should have bounds checks
    }

    if cwe_id in checks:
        required_patterns = checks[cwe_id]
        found_patterns = [p for p in required_patterns if p in patch_lower]
        if not found_patterns:
            issues.append(
                f"No security-specific pattern found for {cwe_id}. "
                f"Expected one of: {required_patterns}"
            )

    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "cwe_id": cwe_id,
        "checks_performed": len(checks.get(cwe_id, [])),
    }
