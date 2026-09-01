"""Integration tests for Stage 4 baseline evaluation pipeline.

These tests exercise the full baseline flow: gold-eval loading → prompt building
→ mock model inference → parsing → metrics computation → output file writing.

They mirror the integration test patterns from Stage 2 and Stage 3:
- Mock backends are injected (no real model downloads).
- The full pipeline (baseline.py) is tested end-to-end.
- Output files (predictions.jsonl, metrics.json, manifest.json) are validated.
- Real gold-eval JSONL data is used from eval/gold_set/gold.jsonl.
"""

from __future__ import annotations

import json
import os

import pytest

from app.evaluation.backends import MockBackend
from app.evaluation.baseline import (
    BaselineConfig,
    BaselineResult,
    load_gold_eval,
    run_baseline,
    run_baseline_on_predictions,
)
from app.evaluation.metrics import BaselineMetrics
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample

# Path to the bundled gold-eval set
GOLD_EVAL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "eval",
    "gold_set",
    "gold.jsonl",
)


# --- Helpers ---


def _make_mock_backend_always_correct(gold_samples: list[VulnSample]) -> MockBackend:
    """Create a MockBackend that returns the correct CWE for each sample.

    The MockBackend matches on prompt substrings, so we key on a unique
    snippet from each sample's vulnerable_code (which is included in the
    format_prompt output). Since each gold sample has distinct code, the
    first line serves as a reliable key.
    """
    import json as json_mod

    responses: dict[str, str] = {}
    for sample in gold_samples:
        # Use the first line of vulnerable code as the key — it's unique
        # per sample and appears in the rendered prompt.
        key = sample.vulnerable_code.split("\n")[0].strip()
        responses[key] = json_mod.dumps(
            {
                "cwe_id": sample.cwe_id,
                "severity": sample.severity,
                "explanation": f"Detected {sample.cwe_id} vulnerability.",
                "patch_diff": "--- a/code.py\n+++ b/code.py\n- old\n+ new",
            }
        )

    return MockBackend(responses=responses)


class _SequentialMockBackend:
    """Backend that returns pre-set responses in call order.

    Used in few-shot tests where prompt-substring matching is ambiguous
    (the few-shot prompt contains code from multiple samples)."""

    def __init__(self, responses: list[str]):
        self._responses = responses
        self._index = 0
        self.call_count = 0
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        self.calls.append(prompt[:80] + "..." if len(prompt) > 80 else prompt)
        if self._index < len(self._responses):
            resp = self._responses[self._index]
        else:
            resp = self._responses[-1] if self._responses else "{}"
        self._index += 1
        return resp


def _make_sequential_backend_always_correct(
    gold_samples: list[VulnSample],
) -> _SequentialMockBackend:
    """Create a sequential backend that returns the correct CWE for each sample
    in order (matching the gold sample order)."""
    import json as json_mod

    responses = []
    for sample in gold_samples:
        responses.append(
            json_mod.dumps(
                {
                    "cwe_id": sample.cwe_id,
                    "severity": sample.severity,
                    "explanation": f"Detected {sample.cwe_id} vulnerability.",
                    "patch_diff": "--- a/code.py\n+++ b/code.py\n- old\n+ new",
                }
            )
        )
    return _SequentialMockBackend(responses=responses)


def _make_mock_backend_hallucinating() -> MockBackend:
    """Create a MockBackend that always returns a hallucinated CWE."""
    import json as json_mod

    return MockBackend(
        default=json_mod.dumps(
            {
                "cwe_id": "CWE-4242",
                "severity": "critical",
                "explanation": "I found a vulnerability.",
                "patch_diff": "",
            }
        )
    )


# --- Gold-eval loading ---


def test_load_gold_eval_existing_file():
    """The bundled gold.jsonl should load 59 samples."""
    samples = load_gold_eval(GOLD_EVAL_PATH)
    assert len(samples) == 59
    assert all(isinstance(s, VulnSample) for s in samples)
    assert all(s.split == "gold_eval" for s in samples)


def test_load_gold_eval_all_cwe_classes():
    """Gold set should cover all 6 CWE classes with the expected distribution."""
    samples = load_gold_eval(GOLD_EVAL_PATH)
    cwe_counts = {}
    for s in samples:
        cwe_counts[s.cwe_id] = cwe_counts.get(s.cwe_id, 0) + 1
    assert len(cwe_counts) == 6
    expected = {"CWE-89": 14, "CWE-79": 14, "CWE-22": 14, "CWE-78": 8, "CWE-190": 4, "CWE-502": 5}
    for cwe, count in cwe_counts.items():
        assert count == expected[cwe], f"Expected {expected[cwe]} samples for {cwe}, got {count}"


def test_load_gold_eval_skips_invalid_lines(tmp_path):
    """Invalid JSON lines should be skipped, valid ones loaded."""
    path = tmp_path / "bad_gold.jsonl"
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
    assert samples[0].cwe_id == "CWE-89"
    assert samples[1].cwe_id == "CWE-79"


# --- End-to-end baseline run (zero-shot) ---


def test_baseline_zero_shot_end_to_end(tmp_path):
    """Full pipeline: load gold-eval, run zero-shot, parse, compute metrics."""
    # Use sequential backend — with 59 samples some share first code lines,
    # causing substring-match collisions in _make_mock_backend_always_correct.
    gold = load_gold_eval(GOLD_EVAL_PATH)
    backend = _make_sequential_backend_always_correct(gold)
    config = BaselineConfig(strategy="zero_shot", base_model="mock/model")

    result = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4"),
        config=config,
        backend=backend,
    )

    assert isinstance(result, BaselineResult)
    assert result.run_id.startswith("stage4_zero_shot_")
    assert result.num_predictions == 59
    assert result.num_parse_failures == 0
    assert result.total_attempted == 59

    # Metrics: with perfect predictions, F1 should be high
    assert result.metrics.cwe_macro_f1 == 1.0
    assert result.metrics.cwe_micro_accuracy == 1.0
    assert result.metrics.hallucination_rate == 0.0
    assert result.metrics.patch_coverage == 1.0


def test_baseline_zero_shot_writes_predictions_jsonl(tmp_path):
    """predictions.jsonl should have one valid ModelPrediction per line."""
    backend = _make_mock_backend_always_correct(load_gold_eval(GOLD_EVAL_PATH))
    result = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4"),
        config=BaselineConfig(strategy="zero_shot", base_model="mock/model"),
        backend=backend,
    )

    pred_path = os.path.join(str(tmp_path / "stage4"), "predictions.jsonl")
    assert result.num_predictions == 59  # noqa: F841 — keep for clarity
    assert os.path.exists(pred_path)

    with open(pred_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == result.num_predictions

    for line in lines:
        data = json.loads(line)
        pred = ModelPrediction(**data)
        assert pred.sample_id  # non-empty
        assert pred.run_id == result.run_id
        assert pred.predicted_cwe.startswith("CWE-")


def test_baseline_writes_metrics_json(tmp_path):
    """metrics.json should contain all expected metric fields."""
    backend = _make_mock_backend_always_correct(load_gold_eval(GOLD_EVAL_PATH))
    result = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4"),
        config=BaselineConfig(strategy="zero_shot", base_model="mock/model"),
        backend=backend,
    )

    metrics_path = os.path.join(str(tmp_path / "stage4"), "metrics.json")
    assert os.path.exists(metrics_path)

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    assert metrics["run_id"] == result.run_id
    assert "cwe_macro_f1" in metrics
    assert "cwe_micro_accuracy" in metrics
    assert "hallucination_rate" in metrics
    assert "severity_accuracy" in metrics
    assert "patch_coverage" in metrics
    assert "per_class" in metrics
    assert metrics["num_predictions"] == 59


def test_baseline_writes_manifest_json(tmp_path):
    """manifest.json should contain run provenance."""
    backend = _make_mock_backend_always_correct(load_gold_eval(GOLD_EVAL_PATH))
    run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4"),
        config=BaselineConfig(strategy="zero_shot", base_model="mock/model"),
        backend=backend,
    )

    manifest_path = os.path.join(str(tmp_path / "stage4"), "manifest.json")
    assert os.path.exists(manifest_path)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["stage"] == 4
    assert manifest["strategy"] == "zero_shot"
    assert manifest["base_model"] == "mock/model"
    assert manifest["num_gold_samples"] == 59
    assert manifest["num_predictions"] == 59


# --- End-to-end baseline run (few-shot) ---


def test_baseline_few_shot_end_to_end(tmp_path):
    """Few-shot baseline: 2 examples from gold set as demonstrations."""
    # Create a small few-shot example file from gold samples
    gold = load_gold_eval(GOLD_EVAL_PATH)
    from app.data.formatting.builder import build_instruction_example
    from app.data.formatting.tokenizer import TokenCounter

    class _MockTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(max(len(text), 1)))

    counter = TokenCounter(tokenizer=_MockTokenizer())
    examples = [
        build_instruction_example(s, token_counter=counter, max_tokens=100000) for s in gold[:2]
    ]

    examples_path = tmp_path / "few_shot_examples.jsonl"
    with open(examples_path, "w") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")

    backend = _make_sequential_backend_always_correct(gold)
    config = BaselineConfig(strategy="few_shot", num_shots=2, base_model="mock/model")

    result = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4_fs"),
        config=config,
        backend=backend,
        few_shot_examples_path=str(examples_path),
    )

    assert result.num_predictions == 59
    assert result.metrics.cwe_macro_f1 == 1.0
    assert backend.call_count == 59
    # Few-shot prompt should be longer (includes examples)
    # (we can't easily check this from the result, but call_count confirms
    #  the backend was actually called for each sample)


def test_baseline_few_shot_falls_back_to_zero_shot_without_examples(tmp_path):
    """If few_shot_examples_path is None for few-shot strategy, it falls back."""
    backend = _make_mock_backend_always_correct(load_gold_eval(GOLD_EVAL_PATH))
    config = BaselineConfig(strategy="few_shot", num_shots=3, base_model="mock/model")

    result = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4_fb"),
        config=config,
        backend=backend,
        few_shot_examples_path=None,
    )

    # Should still run (zero-shot fallback)
    assert result.num_predictions == 59
    # Config strategy was modified to zero_shot
    assert result.config.strategy == "zero_shot"


# --- Hallucination detection ---


def test_baseline_hallucination_rate_with_bad_backend(tmp_path):
    """When the model always returns a fake CWE, hallucination rate = 1.0."""
    backend = _make_mock_backend_hallucinating()
    result = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4_hall"),
        config=BaselineConfig(strategy="zero_shot", base_model="mock/model"),
        backend=backend,
    )

    assert result.metrics.hallucination_rate == 1.0
    assert result.metrics.cwe_macro_f1 < 0.1  # very low, since all CWEs wrong


# --- Parse failure handling ---


def test_baseline_handles_unparseable_responses(tmp_path):
    """When the model returns garbage, it's recorded as a parse failure."""
    backend = MockBackend(default="I'm sorry I can't help with that")
    config = BaselineConfig(strategy="zero_shot", base_model="mock/model")

    result = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "stage4_bad"),
        config=config,
        backend=backend,
    )

    assert result.num_parse_failures == 59
    assert result.metrics.num_parsed == 0
    # Parse failures still appear as predictions (with empty CWE)
    assert result.num_predictions == 59

    # parse_errors.jsonl was written
    err_path = os.path.join(str(tmp_path / "stage4_bad"), "parse_errors.jsonl")
    assert os.path.exists(err_path)


# --- run_baseline_on_predictions ---


def test_run_baseline_on_predictions_without_inference(tmp_path):
    """Re-compute metrics from saved predictions without running the model."""
    gold = load_gold_eval(GOLD_EVAL_PATH)
    # Create perfect predictions
    predictions = [
        ModelPrediction(
            sample_id=s.id,
            run_id="test_run",
            predicted_cwe=s.cwe_id,
            predicted_severity=s.severity,
            suggested_patch_diff="--- a/x\n+++ b/x\n",
            rationale="Perfect.",
        )
        for s in gold
    ]

    metrics = run_baseline_on_predictions(predictions, gold, run_id="test_run")
    assert isinstance(metrics, BaselineMetrics)
    assert metrics.cwe_macro_f1 == 1.0
    assert metrics.cwe_micro_accuracy == 1.0
    assert metrics.hallucination_rate == 0.0


# --- Reproducibility ---


def test_baseline_is_reproducible_with_mock_backend(tmp_path):
    """Two runs with the same mock backend should produce identical predictions."""
    backend1 = _make_mock_backend_always_correct(load_gold_eval(GOLD_EVAL_PATH))
    backend2 = _make_mock_backend_always_correct(load_gold_eval(GOLD_EVAL_PATH))
    config = BaselineConfig(strategy="zero_shot", base_model="mock/model")

    r1 = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "r1"),
        config=config,
        backend=backend1,
    )
    r2 = run_baseline(
        gold_eval_path=GOLD_EVAL_PATH,
        output_dir=str(tmp_path / "r2"),
        config=config,
        backend=backend2,
    )

    # Same CWE predictions
    cwe1 = sorted(p.predicted_cwe for p in r1.predictions)
    cwe2 = sorted(p.predicted_cwe for p in r2.predictions)
    assert cwe1 == cwe2

    # Same metrics
    assert r1.metrics.cwe_macro_f1 == r2.metrics.cwe_macro_f1


# --- Gold-eval loading from file ---


def test_baseline_raises_on_missing_gold_eval(tmp_path):
    """If the gold-eval file doesn't exist, should raise FileNotFoundError."""
    config = BaselineConfig(strategy="zero_shot")
    backend = MockBackend(default="{}")
    with pytest.raises(FileNotFoundError):
        run_baseline(
            gold_eval_path=str(tmp_path / "nonexistent.jsonl"),
            output_dir=str(tmp_path / "out"),
            config=config,
            backend=backend,
        )


def test_baseline_raises_on_empty_gold_eval(tmp_path):
    """If the gold-eval file is empty, should raise RuntimeError."""
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("")

    config = BaselineConfig(strategy="zero_shot")
    backend = MockBackend(default="{}")
    with pytest.raises(RuntimeError, match="No gold-eval samples found"):
        run_baseline(
            gold_eval_path=str(empty_path),
            output_dir=str(tmp_path / "out"),
            config=config,
            backend=backend,
        )
