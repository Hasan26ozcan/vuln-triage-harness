"""Merge a Stage 5 LoRA checkpoint into its base model and save the result.

This exists so Stage 9 (air-gapped serving) can serve the *actual
fine-tuned* model rather than the pre-fine-tuning base model. Before this
script, the only place that merged a LoRA adapter into the base model was
``QwenBackend._load()`` in ``app/evaluation/backends.py`` — useful for
in-process evaluation (Stage 6/7), but there was no standalone way to
produce a merged model directory on disk for ``scripts/convert_to_gguf.py``
/ ``scripts/run_stage9_serve.py`` to pick up. This script extracts that
same merge logic (``PeftModel.from_pretrained`` + ``merge_and_unload()``)
into a reusable CLI.

Usage::

    python scripts/merge_lora_for_export.py \\
        --lora-checkpoint output/stage5/qwen_lora_gpu/final_checkpoint \\
        --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --output output/stage9/tuned_merged

Requires the ``[ml]`` extra (``transformers``, ``peft``, ``torch``) and a
real Hugging Face download of the base model — run this on a machine with
GPU + internet access (e.g. your RTX 4060 box), not in a network-restricted
sandbox.
"""

import argparse
import logging
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from app.security.paths import validate_path  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge a LoRA checkpoint into its base model")
    ap.add_argument(
        "--lora-checkpoint",
        default="output/stage5/qwen_lora_gpu/final_checkpoint",
        help="Directory with adapter_config.json + adapter_model.safetensors",
    )
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model the adapter was trained on",
    )
    ap.add_argument(
        "--output",
        default="output/stage9/tuned_merged",
        help="Directory to write the merged full model + tokenizer to",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    lora_path = validate_path(args.lora_checkpoint, allow_temp=True)

    adapter_config = lora_path / "adapter_config.json"
    adapter_weights = lora_path / "adapter_model.safetensors"
    if not adapter_config.exists():
        print(f"ERROR: {adapter_config} not found — is --lora-checkpoint correct?")
        sys.exit(1)
    if not adapter_weights.exists():
        print(
            f"ERROR: {adapter_weights} not found. The adapter checkpoint is "
            "incomplete (weights missing)."
        )
        sys.exit(1)

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("ERROR: transformers/peft not installed. Run: pip install -e '.[ml]'")
        sys.exit(1)

    logger.info("Loading base model %s ...", args.base_model)
    model = AutoModelForCausalLM.from_pretrained(  # nosec B615
        args.base_model,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Applying LoRA adapter from %s ...", lora_path)
    model = PeftModel.from_pretrained(model, str(lora_path))

    logger.info("Merging adapter into base weights (merge_and_unload) ...")
    model = model.merge_and_unload()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(  # nosec B615
        args.base_model, trust_remote_code=True
    )

    output_dir = validate_path(args.output, allow_temp=True)
    output_dir.mkdir(parents=True, exist_ok=True)  # NOSONAR
    logger.info("Saving merged model + tokenizer to %s ...", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nDone. Merged model written to: {output_dir}")
    print("Next step — convert to GGUF:")
    print(
        f"  python scripts/convert_to_gguf.py --model-dir {output_dir} "
        f"--tokenizer-dir {output_dir} --output output/stage9/tuned_model.gguf"
    )


if __name__ == "__main__":
    main()
