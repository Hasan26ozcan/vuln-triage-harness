"""Convert a HuggingFace Qwen2 safetensors checkpoint to GGUF format.

Uses the standalone ``gguf`` Python package (not ``llama-cpp-python``) so it
works in environments where the C-extension build is blocked by AppLocker
policies.  The converted file can be served directly by ``llama-server.exe``.

Usage::

    python scripts/convert_to_gguf.py \
        --model-dir output/stage8/_diag_model \
        --tokenizer-dir output/stage5/qwen_lora_cpu/final_checkpoint \
        --output output/stage8/qwen2_gguf_f32.gguf
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import safetensors.torch
import torch
from gguf import (
    MODEL_ARCH,
    BpeVocab,
    GGMLQuantizationType,
    GGUFWriter,
    LlamaFileType,
    SpecialVocab,
    get_tensor_name_map,
)

from app.security.paths import validate_output_path, validate_path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert HF Qwen2 safetensors → GGUF")
    ap.add_argument(
        "--model-dir",
        default="output/stage8/_diag_model",
        help="Directory with config.json + model.safetensors",
    )
    ap.add_argument(
        "--tokenizer-dir",
        default="output/stage5/qwen_lora_cpu/final_checkpoint",
        help="Directory with vocab.json / tokenizer.json + merges.txt",
    )
    ap.add_argument(
        "--output",
        default="output/stage8/qwen2_gguf_f32.gguf",
        help="Output .gguf file path",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = validate_path(args.model_dir, allow_temp=True)
    tokenizer_dir = validate_path(args.tokenizer_dir, allow_temp=True)
    output_path = validate_output_path(args.output, allow_temp=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Load HF config -------------------------------------------------
    with open(model_dir / "config.json", encoding="utf-8") as f:  # NOSONAR
        config = json.load(f)

    n_layers: int = config["num_hidden_layers"]
    hidden_size: int = config["hidden_size"]
    intermediate_size: int = config["intermediate_size"]
    n_heads: int = config["num_attention_heads"]
    n_kv_heads: int = config["num_key_value_heads"]
    vocab_size: int = config["vocab_size"]
    rms_norm_eps: float = config["rms_norm_eps"]
    # transformers >= 5.x nests rope_theta inside a "rope_parameters" sub-dict;
    # older configs expose it as a top-level key. Support both.
    rope_theta: float = config.get(
        "rope_theta",
        config.get("rope_parameters", {}).get("rope_theta", 1000000.0),
    )
    max_pos: int = config["max_position_embeddings"]

    print(
        f"[cfg] Qwen2: {n_layers} layers, hidden={hidden_size}, "
        f"ffn={intermediate_size}, heads={n_heads}/{n_kv_heads}, "
        f"vocab={vocab_size}, ctx={max_pos}"
    )

    # --- 2. Load tensors ---------------------------------------------------
    print("[load] Reading safetensors weights ...")
    t0 = time.perf_counter()
    sd = safetensors.torch.load_file(str(model_dir / "model.safetensors"))
    print(f"[load] {len(sd)} tensors in {time.perf_counter() - t0:.1f}s")

    # --- 3. Set up tokenizer / vocab --------------------------------------
    vocab = BpeVocab(tokenizer_dir)
    special = SpecialVocab(tokenizer_dir, load_merges=True, n_vocab=vocab_size)
    print(
        f"[vocab] BPE vocab base={vocab.vocab_size_base}, "
        f"added={len(vocab.added_tokens_list)}, total={vocab.vocab_size}"
    )

    # --- 4. GGUF writer + metadata ----------------------------------------
    writer = GGUFWriter(str(output_path), arch="qwen2")

    writer.add_architecture()
    writer.add_block_count(n_layers)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(intermediate_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_context_length(max_pos)
    writer.add_vocab_size(vocab_size)
    writer.add_layer_norm_rms_eps(rms_norm_eps)
    writer.add_rope_freq_base(rope_theta)
    writer.add_rope_dimension_count(hidden_size // n_heads)  # head_dim
    writer.add_file_type(LlamaFileType.ALL_F32)

    # --- 5. Tokenizer data -------------------------------------------------
    writer.add_tokenizer_model(vocab.tokenizer_model)  # "gpt2" / "bpe"

    # Add all BPE tokens — pad to vocab_size so the token list matches the
    # embedding weight dimension (HF config often sets vocab_size to a padded
    # value while the tokenizer has fewer actual tokens).
    all_tokens = list(vocab.all_tokens())
    print(
        f"[vocab] Writing {len(all_tokens)} tokens to GGUF (padding to vocab_size={vocab_size}) ..."
    )

    token_list: list[bytes] = []
    token_scores: list[float] = []
    for token_bytes, score, _tok_type in all_tokens:
        token_list.append(token_bytes)
        token_scores.append(score)

    # Pad with empty tokens so the GGUF token list length equals vocab_size,
    # matching the embedding weight's row count.  These extra entries are
    # never emitted during normal generation.
    padding_needed = vocab_size - len(token_list)
    if padding_needed > 0:
        token_list.extend(b"" for _ in range(padding_needed))
        token_scores.extend(0.0 for _ in range(padding_needed))
    elif padding_needed < 0:
        raise ValueError(
            f"Tokenizer has more tokens ({len(token_list)}) than config vocab_size ({vocab_size})"
        )

    writer.add_token_list(token_list)
    writer.add_token_scores(token_scores)

    # Add special token ids from config
    writer.add_bos_token_id(config["bos_token_id"])
    writer.add_eos_token_id(config["eos_token_id"])
    if "pad_token_id" in config:
        writer.add_pad_token_id(config["pad_token_id"])

    # Add merges + special-token flags from tokenizer dir
    special.add_to_gguf(writer)
    writer.add_string("tokenizer.pre", "qwen2")

    # --- 6. Write weights --------------------------------------------------
    print("[write] Mapping and converting tensors ...")
    tns = get_tensor_name_map(MODEL_ARCH.QWEN2, n_blocks=n_layers)

    skipped: list[str] = []
    written = 0
    t1 = time.perf_counter()

    for hf_name, tensor in sd.items():
        tensor_type, gguf_name = tns.get_type_and_name(hf_name, try_suffixes=[".weight", ".bias"])
        if gguf_name is None:
            skipped.append(hf_name)
            continue

        # Convert BFloat16 tensors to float32 before numpy (numpy can't
        # handle BFloat16 directly). Write as float32 to avoid F16 CUDA
        # kernel bugs (binbcast.cu assertion) in older llama-cpp-python.
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        arr = tensor.detach().cpu().numpy()
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        # Ensure C-contiguous memory layout.
        arr = np.ascontiguousarray(arr)

        # GGUF expects the "natural" shape (not transposed); add_tensor
        # stores shape as given and llama.cpp reads it correctly.
        writer.add_tensor(gguf_name, arr, raw_dtype=GGMLQuantizationType.F32)
        written += 1

    print(
        f"[write] {written} tensors mapped, "
        f"{len(skipped)} skipped, "
        f"elapsed={time.perf_counter() - t1:.1f}s"
    )
    if skipped:
        for s in skipped:
            print(f"  [skip] {s}")

    # --- 7. Finalize file --------------------------------------------------
    print("[write] Writing GGUF file ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = output_path.stat().st_size / (1024**3)
    print(f"\n[done] GGUF written to {output_path}")
    print(f"[done] File size: {size_gb:.2f} GB")


if __name__ == "__main__":
    main()
