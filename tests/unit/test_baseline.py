"""Unit tests for Stage 4 baseline evaluation — covering edge cases not
exercised by the integration tests.

Covers:
  - Blank line handling in load_gold_eval and load_few_shot_examples.
  - Invalid JSON line handling in load_gold_eval and load_few_shot_examples.
  - num_shots early break in load_few_shot_examples.
  - Backend failure (exception in generate) in run_baseline.
  - Parse error (no JSON in response) in run_baseline.
  - Empty gold-eval file → RuntimeError in run_baseline.
  - run_baseline_on_predictions delegation to compute_metrics.
  - QwenBackend_stub lazy import and construction.
"""

import json

import pytest

from app.evaluation.backends import MockBackend
from app.evaluation.baseline import (
    BaselineConfig,
    _qwen_backend,
    load_few_shot_examples,
    load_gold_eval,
    run_baseline,
    run_baseline_on_predictions,
)

# --- load_gold_eval: blank lines ---


def test_load_gold_eval_skips_blank_lines(tmp_path):
    """Blank and whitespace-only lines are silently skipped."""
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"id": "s1", "source": "cve_real", "repo_name": "r1", '
        '"cwe_id": "CWE-89", "severity": "high", "language": "python", '
        '"vulnerable_code": "vuln", "description": "desc"}\n'
        "\n"
        "   \n"
        '{"id": "s2", "source": "cve_real", "repo_name": "r2", '
        '"cwe_id": "CWE-79", "severity": "medium", "language": "javascript", '
        '"vulnerable_code": "vuln2", "description": "desc2"}\n'
    )
    samples = load_gold_eval(str(path))
    assert len(samples) == 2
    assert samples[0].id == "s1"
    assert samples[1].id == "s2"


# --- load_few_shot_examples: blank lines, invalid JSON, num_shots break ---


def _example_json(idx: int) -> str:
    return json.dumps(
        {
            "id": f"ex{idx}",
            "sample_id": f"s{idx}",
            "prompt": f"prompt{idx}",
            "target_cwe": "CWE-89",
            "target_severity": "high",
            "target_explanation": f"expl{idx}",
            "token_count_estimate": 10,
        }
    )


def test_load_few_shot_examples_skips_blank_lines(tmp_path):
    """Blank and whitespace-only lines are silently skipped in few-shot loading."""
    path = tmp_path / "examples.jsonl"
    path.write_text(f"{_example_json(1)}\n\n   \n{_example_json(2)}\n")
    examples = load_few_shot_examples(str(path), num_shots=5)
    assert len(examples) == 2


def test_load_few_shot_examples_skips_invalid_lines(tmp_path):
    """Invalid JSON lines are skipped (not fatal) in few-shot loading."""
    path = tmp_path / "examples.jsonl"
    path.write_text(f"{_example_json(1)}\nNOT VALID JSON\n{_example_json(2)}\n")
    examples = load_few_shot_examples(str(path), num_shots=5)
    assert len(examples) == 2


def test_load_few_shot_examples_stops_at_num_shots(tmp_path):
    """The loader returns at most num_shots examples (early break)."""
    path = tmp_path / "examples.jsonl"
    lines = [_example_json(i) for i in range(5)]
    path.write_text("\n".join(lines) + "\n")

    examples = load_few_shot_examples(str(path), num_shots=2)
    assert len(examples) == 2


# --- run_baseline: backend failure ---


def test_run_baseline_handles_backend_failure(tmp_path):
    """When backend.generate raises, the sample becomes a ParseError (not a crash)."""

    class _FailingBackend:
        """Backend whose generate() always raises."""

        def generate(self, prompt: str) -> str:
            raise RuntimeError("model crashed")

    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        '{"id": "s1", "source": "cve_real", "repo_name": "r1", '
        '"cwe_id": "CWE-89", "severity": "high", "language": "python", '
        '"vulnerable_code": "vuln", "description": "desc"}\n'
    )

    result = run_baseline(
        gold_eval_path=str(gold_path),
        output_dir=str(tmp_path / "out"),
        config=BaselineConfig(strategy="zero_shot"),
        backend=_FailingBackend(),
    )

    assert result.num_predictions == 0
    assert result.num_parse_failures == 1
    assert result.parse_errors[0].sample_id == "s1"
    assert "Backend error" in result.parse_errors[0].reason


# --- load_gold_eval: invalid lines (lines 110-111) ---


def test_load_gold_eval_skips_invalid_lines(tmp_path):
    """Line 110-111: invalid JSON / wrong schema lines are skipped with a warning."""
    path = tmp_path / "gold.jsonl"
    path.write_text(
        '{"id": "s1", "source": "cve_real", "repo_name": "r1", '
        '"cwe_id": "CWE-89", "severity": "high", "language": "python", '
        '"vulnerable_code": "vuln", "description": "desc"}\n'
        "NOT VALID JSON\n"
        '{"id": "s2", "source": "cve_real", "repo_name": "r2", '
        '"cwe_id": "CWE-79", "severity": "medium", "language": "javascript", '
        '"vulnerable_code": "vuln2", "description": "desc2"}\n'
    )
    samples = load_gold_eval(str(path))
    assert len(samples) == 2
    assert samples[0].id == "s1"
    assert samples[1].id == "s2"


# --- run_baseline: empty gold-eval (line 192) ---


def test_run_baseline_empty_gold_eval_raises(tmp_path):
    """Line 192: empty gold-eval file → RuntimeError."""
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("")

    config = BaselineConfig(strategy="zero_shot")
    backend = MockBackend(default="fallback")
    with pytest.raises(RuntimeError, match="No gold-eval samples found"):
        run_baseline(
            gold_eval_path=str(empty_path),
            output_dir=str(tmp_path / "out"),
            config=config,
            backend=backend,
        )


# --- run_baseline: parse error (lines 231-234) ---


def test_run_baseline_parse_error_records_failure(tmp_path):
    """Lines 231-234: when parse_prediction returns a ParseError,
    the error is recorded and a minimal prediction appended."""
    gold_path = tmp_path / "gold.jsonl"
    gold_path.write_text(
        '{"id": "s1", "source": "cve_real", "repo_name": "r1", '
        '"cwe_id": "CWE-89", "severity": "high", "language": "python", '
        '"vulnerable_code": "vuln", "description": "desc"}\n'
    )

    # Backend returns text with no JSON → parse failure
    backend = MockBackend(default="No JSON here, just text.")

    result = run_baseline(
        gold_eval_path=str(gold_path),
        output_dir=str(tmp_path / "out"),
        config=BaselineConfig(strategy="zero_shot"),
        backend=backend,
    )

    assert result.num_predictions == 1
    assert result.num_parse_failures == 1
    assert result.parse_errors[0].sample_id == "s1"
    assert "No JSON" in result.parse_errors[0].reason


# --- run_baseline_on_predictions (line 320) ---


def test_run_baseline_on_predictions_computes_metrics(tmp_path):
    """Line 320: run_baseline_on_predictions delegates to compute_metrics."""
    from app.evaluation.metrics import BaselineMetrics
    from app.schemas.prediction_eval import ModelPrediction
    from app.schemas.vuln import VulnSample

    gold_samples = [
        VulnSample(
            id="s1",
            source="cve_real",
            repo_name="r1",
            cve_id="CVE-2024-1",
            cwe_id="CWE-89",
            severity="high",
            language="python",
            vulnerable_code="vuln",
            description="d",
        ),
    ]

    predictions = [
        ModelPrediction(
            sample_id="s1",
            run_id="test",
            predicted_cwe="CWE-89",
            predicted_severity="high",
            suggested_patch_diff="",
            rationale="test",
        ),
    ]

    metrics = run_baseline_on_predictions(predictions, gold_samples, run_id="test")

    assert isinstance(metrics, BaselineMetrics)
    assert metrics.run_id == "test"
    assert metrics.num_predictions == 1
    assert metrics.num_parsed == 1
    assert metrics.num_parse_failures == 0
    assert metrics.cwe_macro_f1 == 1.0  # perfect prediction


# --- _qwen_backend ---


def test_qwen_backend_creates_qwen_backend():
    """_qwen_backend lazily imports and constructs QwenBackend from config."""
    config = BaselineConfig(
        base_model="test/model",
        max_new_tokens=512,
        temperature=0.5,
    )
    backend = _qwen_backend(config)

    from app.evaluation.backends import QwenBackend

    assert isinstance(backend, QwenBackend)
    assert backend.model_name == "test/model"
    assert backend.max_new_tokens == 512
    assert backend.temperature == 0.5
