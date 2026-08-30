"""Stage 6 — rerun only the four-tier evaluation using saved Stage 4 predictions.

This avoids regenerating model predictions (which takes ~18 min on GPU)
by loading them from output/stage4/predictions.jsonl, then running
Tiers 1-4 (Docker sandbox + local LLM judge).

Usage::

    python scripts/run_stage6_only.py [--checkpoint PATH] [--sandbox-mode docker]

The script loads the DPO checkpoint for the local LLM judge (Tier 4),
loads saved predictions from output/stage4/, and writes the Stage 6
report to output/stage6/eval_report.json.
"""

import json
import logging
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the project root is on sys.path when run as a standalone script.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.security.paths import safe_read_text, validate_output_path, validate_path  # noqa: E402
from scripts.verify_checkpoint import verify_checkpoint  # noqa: E402

# Ensure output is unbuffered so progress is visible in background runs.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_CHECKPOINT = "./output/stage5/dpo/final_checkpoint"


def load_trained_model(base_model: str, checkpoint: str):
    """Load the base model + LoRA adapter for inference (Tier 4 judge)."""
    fingerprint = verify_checkpoint(checkpoint)
    print(f"Checkpoint verified: {fingerprint}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({base_model})...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        base_model,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    print(f"Base model loaded in {time.time() - start:.1f}s")

    print(f"Loading LoRA adapter from {checkpoint}...")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    print("Model ready for inference")

    return model, tokenizer


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Stage 6 only (using saved Stage 4 predictions)")
    ap.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Path to LoRA checkpoint (default: {DEFAULT_CHECKPOINT})",
    )
    ap.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model (default: {DEFAULT_BASE_MODEL})",
    )
    ap.add_argument(
        "--sandbox-mode",
        default="docker",
        choices=["mock", "local", "docker"],
        help="Tier 3 sandbox mode",
    )
    ap.add_argument(
        "--predictions",
        default="output/stage4/predictions.jsonl",
        help="Path to saved Stage 4 predictions JSONL",
    )
    ap.add_argument(
        "--gold-set", default="eval/gold_set/gold.jsonl", help="Path to gold-eval JSONL"
    )
    ap.add_argument(
        "--skip-tier4",
        action="store_true",
        default=False,
        help="Skip LLM judge evaluation (Tier 4)",
    )
    args = ap.parse_args()

    # Step 1: Load saved predictions
    from app.evaluation.runner import load_predictions

    predictions = load_predictions(args.predictions)
    print(f"Loaded {len(predictions)} saved predictions from {args.predictions}")

    # Step 2: Load gold eval samples
    from app.evaluation.runner import load_samples

    samples = load_samples(args.gold_set)
    print(f"Loaded {len(samples)} gold-eval samples")

    # Step 3: Load model for Tier 4 LLM judge (if not skipping)
    tier4_evaluator = None
    if not args.skip_tier4:
        print("\n=== Loading model for Tier 4 LLM judge ===")
        model, tokenizer = load_trained_model(args.base_model, args.checkpoint)
        from app.evaluation.tier4_llm_judge import LlmJudge, LocalLlmJudgeBackend

        tier4_evaluator = LlmJudge(
            backend=LocalLlmJudgeBackend(
                model=model,
                tokenizer=tokenizer,
                max_tokens=128,
            ),
            model="local-qwen-judge",
            max_tokens=128,
        )

        # Wrap evaluate_all with progress logging
        _orig_eval_all = tier4_evaluator.evaluate_all

        def _progress_eval_all(samples, predictions):
            pred_map = {p.sample_id: p for p in predictions}
            matched = [s for s in samples if s.id in pred_map]
            print(f"  Judge: scoring {len(matched)} predictions...", flush=True)
            results = []
            for i, sample in enumerate(samples):
                pred = pred_map.get(sample.id)
                if pred is None:
                    continue
                results.append(tier4_evaluator.evaluate(sample, pred))
                if (i + 1) % 5 == 0 or i + 1 == len(samples):
                    print(f"  Judge: {i + 1}/{len(samples)} samples done", flush=True)
            return results

        tier4_evaluator.evaluate_all = _progress_eval_all

    # Step 4: Run Stage 6 four-tier evaluation
    print(f"\n=== Stage 6 Four-Tier Evaluation (sandbox_mode={args.sandbox_mode}) ===", flush=True)
    from app.evaluation.runner import EvalConfig, EvaluationRunner

    eval_config = EvalConfig(
        base_model=args.base_model,
        embedding_model="jinaai/jina-embeddings-v2-base-code",
        sandbox_mode=args.sandbox_mode,
        skip_tier4=args.skip_tier4,
    )
    runner = EvaluationRunner(config=eval_config, tier4_evaluator=tier4_evaluator)

    print("Starting evaluation...", flush=True)
    eval_report = runner.run(samples, predictions)
    print("Evaluation complete.", flush=True)

    # Step 5: Save results
    m = eval_report.metrics
    print("\nStage 6 Metrics:")
    print(f"  Tier1 CWE Macro-F1: {m.tier1_cwe_macro_f1:.4f}")
    print(f"  Tier1 Coverage: {m.tier1_coverage:.4f}")
    print(f"  Tier2 CWE Macro-F1: {m.tier2_cwe_macro_f1:.4f}")
    print(f"  Tier2 Coverage: {m.tier2_coverage:.4f}")
    print(f"  Model CWE Macro-F1: {m.model_cwe_macro_f1:.4f}")
    print(f"  Exec Pass Rate: {m.exec_pass_rate:.4f}")
    print(f"  Patch Applies Rate: {m.patch_applies_rate:.4f}")
    print(f"  Build Succeeds Rate: {m.build_succeeds_rate:.4f}")
    print(f"  Hallucination Rate: {m.hallucination_rate:.4f}")
    print(f"  Patch Coverage: {m.avg_patch_coverage:.4f}")
    if m.avg_explanation_quality:
        print(f"  Avg Explanation Quality (LLM judge): {m.avg_explanation_quality:.4f}")
    else:
        print("  Avg Explanation Quality: N/A")
    if m.avg_patch_minimality:
        print(f"  Avg Patch Minimality (LLM judge): {m.avg_patch_minimality:.4f}")
    else:
        print("  Avg Patch Minimality: N/A")
    print("  Per-class:")
    for cwe, stats in m.per_class.items():
        print(f"    {cwe}: P={stats['precision']:.4f} R={stats['recall']:.4f} F1={stats['f1']:.4f}")

    # Save Stage 6 report
    stage6_path = validate_output_path("output/stage6/eval_report.json", allow_temp=True)
    stage6_path.parent.mkdir(parents=True, exist_ok=True)
    stage6_path.write_text(json.dumps(eval_report.model_dump(), indent=2, default=str))
    print(f"\nStage 6 report saved to {stage6_path}")

    # Save combined output (Stage 4 + Stage 6)
    # Load the training result for completeness
    ckpt_dir = validate_path(args.checkpoint, allow_temp=True)
    local_tr = ckpt_dir.parent / "training_result.json"
    fallback_tr = validate_path("output/stage5/training_result.json", allow_temp=True)
    tr_path = local_tr if local_tr.exists() else fallback_tr
    training_result = json.loads(safe_read_text(tr_path, allow_temp=True))

    # Load Stage 4 metrics
    stage4_metrics_path = validate_path("output/stage4/metrics.json", allow_temp=True)
    stage4_metrics = (
        json.loads(safe_read_text(stage4_metrics_path, allow_temp=True))
        if stage4_metrics_path.exists()
        else {}
    )

    output = {
        "run_id": eval_report.run_id,
        "base_model": args.base_model,
        "lora_checkpoint": args.checkpoint,
        "method": training_result.get("method", "sft_qlora"),
        "training_result": training_result,
        "baseline_metrics": stage4_metrics,
        "stage6_report": eval_report.model_dump(),
        "stage6_metrics": {
            "cwe_macro_f1": m.model_cwe_macro_f1,
            "exec_pass_rate": m.exec_pass_rate,
            "hallucination_rate": m.hallucination_rate,
            "patch_coverage": m.avg_patch_coverage,
            "severity_accuracy": stage4_metrics.get("severity_accuracy"),
            "per_class": m.per_class,
            "tier1_cwe_macro_f1": m.tier1_cwe_macro_f1,
            "tier1_coverage": m.tier1_coverage,
            "tier2_cwe_macro_f1": m.tier2_cwe_macro_f1,
            "tier2_coverage": m.tier2_coverage,
        },
        "sandbox_mode": args.sandbox_mode,
    }

    out_path = validate_output_path("output/stage5/eval_results.json", allow_temp=True)
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results saved to {out_path}")

    print("\n=== Summary ===")
    print(f"Model: {args.base_model} + LoRA(r=8, alpha=16)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sandbox mode: {args.sandbox_mode}")
    print(f"Baseline CWE Macro-F1: {stage4_metrics.get('cwe_macro_f1', 'N/A')}")
    print(f"Model CWE Macro-F1: {m.model_cwe_macro_f1:.4f}")


if __name__ == "__main__":
    main()
