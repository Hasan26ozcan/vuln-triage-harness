"""Unit tests for Stage 4 inference prompt builders.

These verify:
  - Zero-shot prompt includes SYSTEM_PROMPT, vulnerable code, and the
    JSON response format instruction.
  - Few-shot prompt includes examples as Input/Output pairs.
  - Few-shot with 0 examples falls back to zero-shot.
  - Few-shot respects num_shots (clamps to available examples).
  - The prompt reuses Stage 3's format_prompt (same vulnerable code in output).
  - System prompt is consistent between zero-shot and few-shot.
"""

from app.data.formatting.template import SYSTEM_PROMPT
from app.evaluation.prompt import (
    RESPONSE_FORMAT_INSTRUCTION,
    build_few_shot_prompt,
    build_zero_shot_prompt,
)
from app.schemas.dataset import InstructionExample
from app.schemas.vuln import VulnSample


def _sample(
    id_: str = "s1",
    cwe: str = "CWE-89",
    language: str = "python",
    code: str = "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)",
) -> VulnSample:
    return VulnSample(
        id=id_,
        source="cve_real",
        repo_name="org/repo",
        commit_sha="abc123",
        cve_id="CVE-2024-0001",
        cwe_id=cwe,
        severity="high",
        language=language,
        vulnerable_code=code,
        fixed_code=code + "_safe",
        description="SQL injection vulnerability.",
    )


def _example(
    cwe: str = "CWE-79",
    severity: str = "medium",
    prompt: str = "### Task: ...",
) -> InstructionExample:
    return InstructionExample(
        id="ie_001",
        sample_id="vs_001",
        prompt=prompt,
        target_cwe=cwe,
        target_severity=severity,
        target_explanation="XSS vulnerability.",
        target_patch_diff="--- a/app.py\n+++ b/app.py\n- old\n+ new",
        token_count_estimate=100,
    )


# --- Zero-shot ---


def test_zero_shot_prompt_includes_system_prompt():
    prompt = build_zero_shot_prompt(_sample())
    assert SYSTEM_PROMPT in prompt


def test_zero_shot_prompt_includes_vulnerable_code():
    code = "cursor.execute('SELECT * FROM users WHERE id = ' + user_id)"
    sample = _sample(code=code)
    prompt = build_zero_shot_prompt(sample)
    assert code in prompt


def test_zero_shot_prompt_includes_language():
    sample = _sample(language="python")
    prompt = build_zero_shot_prompt(sample)
    assert "python" in prompt


def test_zero_shot_prompt_has_json_response_instruction():
    """The response format instruction must be present to guide the model."""
    prompt = build_zero_shot_prompt(_sample())
    assert RESPONSE_FORMAT_INSTRUCTION.strip() in prompt
    assert "cwe_id" in prompt
    assert "severity" in prompt
    assert "explanation" in prompt
    assert "patch_diff" in prompt


def test_zero_shot_prompt_has_markdown_fence_hint():
    """The instruction should tell the model to use ```json fences."""
    prompt = build_zero_shot_prompt(_sample())
    assert "json" in prompt.lower()


# --- Few-shot ---


def test_few_shot_prompt_includes_examples():
    """Few-shot prompt should contain the example prompts and outputs."""
    sample = _sample()
    examples = [
        _example(cwe="CWE-89", prompt="Example 1 prompt"),
        _example(cwe="CWE-79", prompt="Example 2 prompt"),
    ]
    prompt = build_few_shot_prompt(sample, examples, num_shots=2)
    assert "Example 1" in prompt or "Example 1 prompt" in prompt
    assert "Example 2" in prompt or "Example 2 prompt" in prompt
    # Target sample's code should appear in the "Your Turn" section
    assert sample.vulnerable_code in prompt


def test_few_shot_prompt_falls_back_to_zero_shot_with_no_examples():
    """When no examples are provided, few-shot should equal zero-shot."""
    sample = _sample()
    prompt_fs = build_few_shot_prompt(sample, [], num_shots=3)
    prompt_zs = build_zero_shot_prompt(sample)
    assert prompt_fs == prompt_zs


def test_few_shot_prompt_respects_num_shots():
    """num_shots should clamp to the number of available examples."""
    sample = _sample()
    examples = [
        _example(cwe="CWE-89", prompt="Ex 1"),
        _example(cwe="CWE-79", prompt="Ex 2"),
        _example(cwe="CWE-22", prompt="Ex 3"),
    ]
    prompt = build_few_shot_prompt(sample, examples, num_shots=3)
    assert "Example 1" in prompt
    assert "Example 2" in prompt
    assert "Example 3" in prompt
    # Only 3 examples → no "Example 4"
    assert "Example 4" not in prompt


def test_few_shot_prompt_clamps_num_shots():
    """If num_shots > available examples, use all available."""
    sample = _sample()
    examples = [
        _example(cwe="CWE-89", prompt="Ex 1"),
    ]
    prompt = build_few_shot_prompt(sample, examples, num_shots=5)
    assert "Example 1" in prompt
    assert "Example 2" not in prompt


def test_few_shot_prompt_includes_system_prompt():
    """Few-shot prompt should also include the system prompt (via format_prompt)."""
    sample = _sample()
    examples = [_example()]
    prompt = build_few_shot_prompt(sample, examples, num_shots=1)
    assert SYSTEM_PROMPT in prompt


def test_few_shot_prompt_ends_with_your_turn():
    """The target sample should be in a 'Your Turn' section."""
    sample = _sample()
    examples = [_example()]
    prompt = build_few_shot_prompt(sample, examples, num_shots=1)
    assert "Your Turn" in prompt


def test_few_shot_prompt_includes_example_output_json():
    """Each example should show the expected JSON output format."""
    sample = _sample()
    examples = [_example(cwe="CWE-89")]
    prompt = build_few_shot_prompt(sample, examples, num_shots=1)
    # The example output should contain the cwe_id from the target
    assert "CWE-89" in prompt


# --- Cross-strategy consistency ---


def test_zero_shot_and_few_shot_both_contain_target_code():
    """Both strategies must include the vulnerable code from the sample."""
    sample = _sample()
    examples = [_example()]

    zs = build_zero_shot_prompt(sample)
    fs = build_few_shot_prompt(sample, examples, num_shots=1)

    assert sample.vulnerable_code in zs
    assert sample.vulnerable_code in fs


def test_response_format_instruction_present_in_both():
    """Both strategies should end with the JSON response instruction."""
    sample = _sample()
    examples = [_example()]

    zs = build_zero_shot_prompt(sample)
    fs = build_few_shot_prompt(sample, examples, num_shots=1)

    assert RESPONSE_FORMAT_INSTRUCTION.strip() in zs
    assert RESPONSE_FORMAT_INSTRUCTION.strip() in fs
