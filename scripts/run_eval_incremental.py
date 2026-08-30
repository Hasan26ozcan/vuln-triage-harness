"""Incremental evaluation: generate predictions, save after each sample, skip Stage 6.

This script generates predictions for all 59 gold-eval samples using the trained
LoRA checkpoint. It saves predictions incrementally to a JSON file after each
sample, so progress can be monitored and resumed if the process is interrupted.

Usage:
    python -u scripts/run_eval_incremental.py [--checkpoint PATH]
"""

import argparse
import json
import threading
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.evaluation.metrics import compute_metrics
from app.evaluation.parser import ParseError, parse_prediction
from app.evaluation.prompt import build_zero_shot_prompt
from app.schemas.prediction_eval import ModelPrediction
from app.schemas.vuln import VulnSample
from app.security.paths import validate_path

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_CHECKPOINT = "./output/stage5/dpo/final_checkpoint"
PROGRESS_FILE = Path("output/stage5/predictions_progress.json")
RESULTS_FILE = Path("output/stage5/eval_results_dpo.json")


def load_trained_model(base_model: str, checkpoint: str):
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)  # nosec B615
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({base_model})...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        base_model,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    print(f"Base model loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"Loading LoRA adapter from {checkpoint}...", flush=True)
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    print("Model ready for inference", flush=True)

    return model, tokenizer


def generate_prediction(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def main():
    ap = argparse.ArgumentParser(description="Incremental evaluation")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--gold-set", default="eval/gold_set/gold.jsonl")
    args = ap.parse_args()

    model, tokenizer = load_trained_model(args.base_model, args.checkpoint)

    # Load gold samples
    gold_path = validate_path(args.gold_set, allow_temp=True)
    samples = []
    for line in gold_path.read_text(encoding="utf-8").splitlines():  # NOSONAR
        line = line.strip()
        if line:
            samples.append(VulnSample(**json.loads(line)))
    print(f"\nLoaded {len(samples)} gold-eval samples", flush=True)

    run_id = f"dpo_real_{time.strftime('%Y%m%d_%H%M%S')}"
    predictions = []
    parse_errors = []
    completed = 0

    # Resume from progress file if it exists
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
            predictions = progress.get("predictions", [])
            parse_errors = progress.get("parse_errors", [])
            completed = len(predictions)
            print(f"Resuming from sample {completed}...", flush=True)
        except Exception:  # nosec B110 — gracefully resume from missing/corrupt progress
            pass

    for i, sample in enumerate(samples):
        if i < completed:
            continue

        prompt = build_zero_shot_prompt(sample)
        print(f"\n[{i + 1}/{len(samples)}] {sample.id} ({sample.cwe_id})...", flush=True)

        t0 = time.time()
        try:
            raw_output = generate_prediction(model, tokenizer, prompt)
            t_gen = time.time() - t0
            print(f"  Gen: {t_gen:.1f}s, {len(raw_output)} chars", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            parse_errors.append({"sample_id": sample.id, "reason": str(e), "raw_output": ""})
            predictions.append(
                {
                    "sample_id": sample.id,
                    "run_id": run_id,
                    "predicted_cwe": "",
                    "predicted_severity": "low",
                    "suggested_patch_diff": "",
                    "rationale": f"[ERROR: {e}]",
                }
            )
            _save_progress(predictions, parse_errors, run_id)
            continue

        print(f"  Raw (first 500): {raw_output[:500]}", flush=True)
        t0 = time.time()
        result = None
        # Run parse_prediction in a thread with a 30s timeout
        result_holder = {}

        def _parse(
            _raw: str = raw_output,
            _sid: str = sample.id,
            _rid: str = run_id,
            _holder: dict = result_holder,
        ):
            _holder["result"] = parse_prediction(_raw, sample_id=_sid, run_id=_rid)

        parse_thread = threading.Thread(target=_parse, daemon=True)
        parse_thread.start()
        parse_thread.join(timeout=30.0)
        t_parse = time.time() - t0
        if parse_thread.is_alive():
            print(f"  Parse: TIMEOUT after {t_parse:.1f}s", flush=True)
            # Fallback: use regex to extract CWE and severity
            import re

            cwe_match = re.search(r'"cwe_id"\s*:\s*"(CWE-\d+)"', raw_output, re.IGNORECASE)
            sev_match = re.search(
                r'"severity"\s*:\s*"(low|medium|high|critical)"',
                raw_output,
                re.IGNORECASE,
            )
            if cwe_match and sev_match:
                pred = ModelPrediction(
                    sample_id=sample.id,
                    run_id=run_id,
                    predicted_cwe=cwe_match.group(1),
                    predicted_severity=sev_match.group(1).lower(),
                    suggested_patch_diff="",
                    rationale="Parse timed out; used regex fallback",
                )
                predictions.append(pred.model_dump())
                print(f"  -> CWE (fallback): {pred.predicted_cwe}", flush=True)
            else:
                parse_errors.append(
                    {
                        "sample_id": sample.id,
                        "reason": "Parse timeout and no regex match",
                        "raw_output": raw_output[:500],
                    }
                )
                predictions.append(
                    {
                        "sample_id": sample.id,
                        "run_id": run_id,
                        "predicted_cwe": "",
                        "predicted_severity": "low",
                        "suggested_patch_diff": "",
                        "rationale": "[PARSE TIMEOUT]",
                    }
                )
                print("  Parse: failed (timeout + regex)", flush=True)
            _save_progress(predictions, parse_errors, run_id)
            continue
        else:
            result = result_holder.get("result")
            print(f"  Parse: {t_parse:.2f}s", flush=True)

        if isinstance(result, ParseError):
            parse_errors.append(
                {
                    "sample_id": result.sample_id,
                    "reason": result.reason,
                    "raw_output": raw_output[:500],
                }
            )
            print(f"  Parse error: {result.reason}", flush=True)
            predictions.append(
                {
                    "sample_id": sample.id,
                    "run_id": run_id,
                    "predicted_cwe": "",
                    "predicted_severity": "low",
                    "suggested_patch_diff": "",
                    "rationale": f"[PARSE FAILURE: {result.reason}]",
                }
            )
        else:
            # ModelPrediction
            predictions.append(result.model_dump())
            print(
                f"  -> CWE: {result.predicted_cwe}, Severity: {result.predicted_severity}",
                flush=True,
            )

        _save_progress(predictions, parse_errors, run_id)

    # Compute metrics
    metrics = compute_metrics(predictions, samples, run_id=run_id)
    print("\n=== Metrics ===", flush=True)
    print(f"Num predictions: {metrics.num_predictions}", flush=True)
    print(f"Num parsed: {metrics.num_parsed}", flush=True)
    print(f"CWE Macro-F1: {metrics.cwe_macro_f1:.4f}", flush=True)
    print(f"CWE Micro Accuracy: {metrics.cwe_micro_accuracy:.4f}", flush=True)
    print(f"Severity Accuracy: {metrics.severity_accuracy:.4f}", flush=True)
    print("Per-class:", flush=True)
    for cwe, stats in metrics.per_class.items():
        print(
            f"  {cwe}: P={stats['precision']:.4f}"
            f" R={stats['recall']:.4f} F1={stats['f1']:.4f}"
            f" (support={stats['support']})",
            flush=True,
        )

    # Save final results
    output = {
        "run_id": run_id,
        "predictions": predictions,
        "parse_errors": parse_errors,
    }
    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


def _save_progress(predictions, parse_errors, run_id):
    """Save progress after each sample."""
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "predictions": predictions,
                "parse_errors": parse_errors,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
