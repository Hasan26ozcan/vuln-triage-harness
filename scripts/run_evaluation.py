"""Run real evaluation with the trained Qwen 1.5B + LoRA model on gold-eval set.

This script:
1. Loads the base Qwen 1.5B model + LoRA adapter from a checkpoint
2. Generates predictions on the gold-eval set (59 real samples, 6 CWE classes)
3. Parses predictions into ModelPrediction records
4. Computes baseline metrics (Stage 4)
5. Runs the four-tier evaluation (Stage 6) with configurable Tier 3 sandbox
6. Saves all results as JSON for doc regeneration

Usage::

    python scripts/run_evaluation.py [--checkpoint PATH] [--base-model MODEL]
        [--sandbox-mode {mock,local,docker}] [--gold-set PATH]
"""
import argparse
import json
import logging
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.evaluation.metrics import compute_metrics
from app.evaluation.parser import ParseError, parse_prediction
from app.evaluation.prompt import build_zero_shot_prompt
from app.evaluation.runner import EvalConfig, EvaluationRunner
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample

torch.set_num_threads(8)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_CHECKPOINT = "./output/stage5/qwen_lora_gpu/final_checkpoint"


def load_trained_model(base_model: str, checkpoint: str):
    """Load the base model + LoRA adapter for inference."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({base_model})...")
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Base model loaded in {time.time()-start:.1f}s")

    print(f"Loading LoRA adapter from {checkpoint}...")
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    print("Model ready for inference")

    return model, tokenizer


def generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a prediction from the model."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (exclude the prompt)
    new_tokens = output[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Return raw response; the parser's _extract_json handles fence detection
    # and stripping. Stripping backticks piecemeal here would break the
    # ```json ... ``` fence pattern the parser expects.
    return response


def main():
    ap = argparse.ArgumentParser(description="Real evaluation of trained LoRA model")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                    help=f"Path to LoRA checkpoint (default: {DEFAULT_CHECKPOINT})")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                    help=f"Base model (default: {DEFAULT_BASE_MODEL})")
    ap.add_argument("--sandbox-mode", default="mock",
                    choices=["mock", "local", "docker"],
                    help="Tier 3 sandbox mode: mock (fast), local (exec), docker (isolated)")
    ap.add_argument("--gold-set", default="eval/gold_set/gold.jsonl",
                    help="Path to gold-eval JSONL")
    args = ap.parse_args()

    # Step 1: Load model
    model, tokenizer = load_trained_model(args.base_model, args.checkpoint)

    # Step 2: Load gold-eval samples
    gold_path = Path(args.gold_set)
    samples = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            samples.append(VulnSample(**json.loads(line)))
    print(f"\nLoaded {len(samples)} gold-eval samples")

    # Step 3: Generate predictions
    run_id = f"stage4_real_{time.strftime('%Y%m%d_%H%M%S')}"
    predictions = []
    parse_errors = []

    for i, sample in enumerate(samples):
        prompt = build_zero_shot_prompt(sample)
        print(f"\n[{i+1}/{len(samples)}] {sample.id} ({sample.cwe_id})...")

        try:
            raw_output = generate_prediction(model, tokenizer, prompt)
            print(f"  Response (first 200 chars): {raw_output[:200]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            parse_errors.append(ParseError(
                sample_id=sample.id,
                reason=str(e),
                raw_output="",
            ))
            predictions.append(ModelPrediction(
                sample_id=sample.id,
                run_id=run_id,
                predicted_cwe="",
                predicted_severity="low",
                suggested_patch_diff="",
                rationale=f"[ERROR: {e}]",
            ))
            continue

        result = parse_prediction(raw_output, sample_id=sample.id, run_id=run_id)
        if isinstance(result, ParseError):
            parse_errors.append(result)
            print(f"  Parse error: {result.reason}")
            predictions.append(ModelPrediction(
                sample_id=sample.id,
                run_id=run_id,
                predicted_cwe="",
                predicted_severity="low",
                suggested_patch_diff="",
                rationale=f"[PARSE FAILURE: {result.reason}]",
            ))
        else:
            predictions.append(result)
            print(f"  -> CWE: {result.predicted_cwe}, Severity: {result.predicted_severity}")

    # Step 4: Compute metrics
    metrics = compute_metrics(predictions, samples, run_id=run_id)
    print("\n=== Baseline Metrics (Stage 4) ===")
    print(f"Num predictions: {metrics.num_predictions}")
    print(f"Num parsed: {metrics.num_parsed}")
    print(f"Num parse failures: {metrics.num_parse_failures}")
    print(f"CWE Macro-F1: {metrics.cwe_macro_f1:.4f}")
    print(f"CWE Micro Accuracy: {metrics.cwe_micro_accuracy:.4f}")
    print(f"Severity Accuracy: {metrics.severity_accuracy:.4f}")
    print(f"Hallucination Rate: {metrics.hallucination_rate:.4f}")
    print(f"Patch Coverage: {metrics.patch_coverage:.4f}")
    print("Per-class:")
    for cwe, stats in metrics.per_class.items():
        print(
            f"  {cwe}: P={stats['precision']:.4f} "
            f"R={stats['recall']:.4f} F1={stats['f1']:.4f} "
            f"(support={stats['support']})"
        )

    # Step 5: Run Stage 6 four-tier evaluation
    print(f"\n=== Stage 6 Four-Tier Evaluation (sandbox_mode={args.sandbox_mode}) ===")
    eval_config = EvalConfig(
        base_model=args.checkpoint,
        embedding_model="jinaai/jina-embeddings-v2-base-code",
        sandbox_mode=args.sandbox_mode,
        skip_tier4=True,  # skip LLM judge (expensive)
    )
    runner = EvaluationRunner(config=eval_config)
    eval_report = runner.run(samples, predictions)

    print("Stage 6 Metrics:")
    m = eval_report.metrics
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
    print("  Per-class:")
    for cwe, stats in m.per_class.items():
        print(f"    {cwe}: P={stats['precision']:.4f} R={stats['recall']:.4f} F1={stats['f1']:.4f}")

    # Step 6: Save results
    training_result = json.loads(Path("output/stage5/training_result.json").read_text())
    output = {
        "run_id": run_id,
        "base_model": args.base_model,
        "lora_checkpoint": args.checkpoint,
        "method": training_result.get("method", "sft_qlora"),
        "lora_r": training_result.get("hyperparams", {}).get("lora_r", 8),
        "num_epochs": training_result.get("hyperparams", {}).get("num_train_epochs", 3),
        "learning_rate": training_result.get("hyperparams", {}).get("learning_rate", 2e-4),
        "training_result": training_result,
        "baseline_metrics": {
            "run_id": metrics.run_id,
            "num_predictions": metrics.num_predictions,
            "num_parsed": metrics.num_parsed,
            "num_parse_failures": metrics.num_parse_failures,
            "cwe_macro_f1": metrics.cwe_macro_f1,
            "cwe_micro_accuracy": metrics.cwe_micro_accuracy,
            "severity_accuracy": metrics.severity_accuracy,
            "hallucination_rate": metrics.hallucination_rate,
            "patch_coverage": metrics.patch_coverage,
            "per_class": metrics.per_class,
        },
        "stage6_report": eval_report.model_dump(),
        "stage6_metrics": {
            "cwe_macro_f1": m.model_cwe_macro_f1,
            "exec_pass_rate": m.exec_pass_rate,
            "hallucination_rate": m.hallucination_rate,
            "patch_coverage": m.avg_patch_coverage,
            "severity_accuracy": metrics.severity_accuracy,
            "per_class": m.per_class,
            "tier1_cwe_macro_f1": m.tier1_cwe_macro_f1,
            "tier1_coverage": m.tier1_coverage,
            "tier2_cwe_macro_f1": m.tier2_cwe_macro_f1,
            "tier2_coverage": m.tier2_coverage,
        },
        "predictions": [p.model_dump() for p in predictions],
        "parse_errors": [
            {
                "sample_id": e.sample_id,
                "reason": e.reason,
                "raw_output": e.raw_output[:500],
            }
            for e in parse_errors
        ],
        "sandbox_mode": args.sandbox_mode,
    }

    out_path = Path("output/stage5/eval_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {out_path}")
    print("\n=== Summary ===")
    print(f"Model: {args.base_model} + LoRA(r=8, alpha=16)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sandbox mode: {args.sandbox_mode}")

    # Read training result for display
    tr = output["training_result"]
    print(f"Train loss: {tr['final_train_loss']:.4f}")
    print(f"Val loss: {tr['final_val_loss']}")
    print(f"Train set size: {tr['train_set_size']}")
    print(f"CWE Macro-F1 (baseline): {metrics.cwe_macro_f1:.4f}")
    print(f"CWE Macro-F1 (model): {m.model_cwe_macro_f1:.4f}")
    print(f"Severity Accuracy: {metrics.severity_accuracy:.4f}")
    print(f"Patch Coverage: {metrics.patch_coverage:.4f}")


if __name__ == "__main__":
    main()
