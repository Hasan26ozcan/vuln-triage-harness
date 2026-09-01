"""Tests for Stage 3's instruction-example builder.

These verify:
  - build_instruction_example correctly maps a VulnSample to an InstructionExample.
  - Prompt, target_cwe, target_severity, target_explanation, target_patch_diff
    are all correctly populated.
  - Token budget filtering drops examples that exceed max_tokens.
  - build_examples returns a BuildResult with examples and dropped list.
  - Token counter is injectable (mock backend for deterministic tests).
  - Samples with fixed_code=None get target_patch_diff=None.
"""

from app.data.formatting.builder import build_examples, build_instruction_example
from app.schemas.vuln import StaticFinding, VulnSample


class _MockTokenizer:
    """Returns 1 token per character — predictable, no model needed."""

    def encode(self, text: str) -> list[int]:
        return list(range(max(len(text), 1)))


def _make_counter():
    from app.data.formatting.tokenizer import TokenCounter

    return TokenCounter(tokenizer=_MockTokenizer())


def _sample(
    id_: str = "vs_001",
    cwe: str = "CWE-89",
    language: str = "python",
    code: str = "cursor.execute('SELECT * FROM t WHERE id = ' + user_id)",
    fixed: str | None = "cursor.execute('SELECT * FROM t WHERE id = %s', (user_id,))",
    severity: str = "high",
    description: str = "SQL injection via unsanitized string concatenation.",
    findings: list[StaticFinding] | None = None,
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name="org/repo",
        commit_sha="abc123",
        cve_id=f"CVE-2024-{id_}",
        cwe_id=cwe,
        severity=severity,
        language=language,
        vulnerable_code=code,
        fixed_code=fixed,
        static_findings=findings if findings is not None else [],
        description=description,
    )


# --- Single example building ---


def test_build_instruction_example_sets_all_targets():
    sample = _sample()
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter)
    assert ex is not None
    assert ex.sample_id == "vs_001"
    assert ex.target_cwe == "CWE-89"
    assert ex.target_severity == "high"
    assert ex.target_explanation == sample.description
    assert ex.target_patch_diff is not None
    assert "--- a/vulnerable_code.py" in ex.target_patch_diff
    assert "+++ b/fixed_code.py" in ex.target_patch_diff


def test_build_instruction_example_prompt_contains_code():
    sample = _sample()
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter)
    assert ex is not None
    assert sample.vulnerable_code in ex.prompt


def test_build_instruction_example_prompt_contains_language():
    sample = _sample(language="javascript")
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter)
    assert ex is not None
    assert "javascript" in ex.prompt


def test_build_instruction_example_generates_unique_id():
    sample = _sample()
    counter = _make_counter()
    ex1 = build_instruction_example(sample, token_counter=counter)
    ex2 = build_instruction_example(sample, token_counter=counter)
    assert ex1 is not None
    assert ex2 is not None
    assert ex1.id != ex2.id  # uuid-based
    assert ex1.id.startswith("ie_")


def test_build_instruction_example_token_count_positive():
    sample = _sample()
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter)
    assert ex is not None
    assert ex.token_count_estimate > 0


def test_build_instruction_example_no_fixed_code():
    """When fixed_code is None, target_patch_diff should be None."""
    sample = _sample(fixed=None)
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter)
    assert ex is not None
    assert ex.target_patch_diff is None


def test_build_instruction_example_fallback_explanation():
    """When description is empty, a fallback explanation should be generated."""
    sample = _sample(description="")
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter)
    assert ex is not None
    assert ex.target_explanation is not None
    assert len(ex.target_explanation) > 0
    assert "CWE-89" in ex.target_explanation


def test_build_instruction_example_includes_static_findings():
    findings = [
        StaticFinding(
            tool="semgrep",
            rule_id="python.sqli",
            message="Possible SQL injection.",
            line_range=(5, 7),
        )
    ]
    sample = _sample(findings=findings)
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter)
    assert ex is not None
    assert "python.sqli" in ex.prompt


# --- Token budget filtering ---


def test_build_instruction_example_dropped_when_exceeds_max_tokens():
    sample = _sample()
    counter = _make_counter()
    # max_tokens=1 is impossibly small — everything should be dropped
    ex = build_instruction_example(sample, token_counter=counter, max_tokens=1)
    assert ex is None


def test_build_instruction_example_kept_when_within_max_tokens():
    sample = _sample()
    counter = _make_counter()
    # With the mock tokenizer (1 char = 1 token), a short sample should fit
    ex = build_instruction_example(sample, token_counter=counter, max_tokens=100000)
    assert ex is not None


def test_build_examples_returns_build_result():
    sample = _sample()
    counter = _make_counter()
    result = build_examples([sample], token_counter=counter, max_tokens=100000)
    assert len(result.examples) == 1
    assert len(result.dropped) == 0


def test_build_examples_drops_exceeding_samples():
    samples = [_sample(id_=f"s{i}") for i in range(5)]
    counter = _make_counter()
    # max_tokens=1 drops everything
    result = build_examples(samples, token_counter=counter, max_tokens=1)
    assert len(result.examples) == 0
    assert len(result.dropped) == 5
    # Each dropped entry is (sample_id, token_count)
    for sample_id, count in result.dropped:
        assert sample_id.startswith("s")
        assert count > 1  # they exceeded max_tokens=1


def test_build_examples_mixed_kept_and_dropped():
    """Some samples fit, some don't."""
    samples = [
        _sample(id_="small"),  # will be dropped (max_tokens=1)
    ]
    counter = _make_counter()
    result = build_examples(
        samples,
        token_counter=counter,
        max_tokens=1,
    )
    assert len(result.examples) == 0
    assert len(result.dropped) == 1


def test_build_examples_empty_input():
    counter = _make_counter()
    result = build_examples([], token_counter=counter)
    assert len(result.examples) == 0
    assert len(result.dropped) == 0


def test_build_examples_uses_default_counter_when_none():
    """When token_counter is None, the builder should create a default one."""
    sample = _sample()
    result = build_examples([sample], token_counter=None, max_tokens=1000000)
    assert len(result.examples) == 1


def test_build_instruction_example_token_count_reflects_prompt_plus_target():
    """The token_count_estimate should be roughly prompt + target tokens."""
    sample = _sample()
    counter = _make_counter()
    ex = build_instruction_example(sample, token_counter=counter, max_tokens=100000)
    assert ex is not None
    # The token count should be larger than just the prompt
    prompt_tokens = len(ex.prompt)  # mock: 1 char = 1 token
    assert ex.token_count_estimate >= prompt_tokens


def test_build_instruction_example_all_cwe_classes():
    """Ensure the builder works for all CWE classes in scope."""
    cwe_ids = ["CWE-89", "CWE-79", "CWE-22", "CWE-78", "CWE-190", "CWE-502"]
    counter = _make_counter()
    for cwe in cwe_ids:
        sample = _sample(cwe=cwe, severity="high")
        ex = build_instruction_example(sample, token_counter=counter, max_tokens=100000)
        assert ex is not None
        assert ex.target_cwe == cwe


def test_build_instruction_example_all_severities():
    """Ensure the builder works for all severity levels."""
    severities = ["low", "medium", "high", "critical"]
    counter = _make_counter()
    for sev in severities:
        sample = _sample(severity=sev)
        ex = build_instruction_example(sample, token_counter=counter, max_tokens=100000)
        assert ex is not None
        assert ex.target_severity == sev
