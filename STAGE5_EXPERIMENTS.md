# Stage 5+ LoRA Experiments — Summary of Changes

## Overview

This document describes the changes made to the vuln-triage-harness project
to explore different LoRA configurations, RAG-based retrieval, and separate
patch generation strategies.

## Previous Run Results (Baseline)

- **Config**: r=8, alpha=16, lr=2e-4, 50 epochs, patience=3
- **Problem**: Early stopping kicked in at epoch 3 (stopped_early=True)
- **Result**: train_loss=0.4384, val_loss=0.3183
- **Root Cause**: patience=3 was too short — model hadn't converged by epoch 3

## New Experiment Configurations

Four LoRA presets have been added in `app/training/config.py`:

| Preset Name | r | alpha | LR | Epochs | Patience | Description |
|---|---|---|---|---|---|---|
| `lora_r16_alpha32_lr5e5` | 16 | 32 | 5e-5 | 50 | 7 | Moderate rank, higher alpha, lower LR |
| `lora_r32_alpha64_lr5e5` | 32 | 64 | 5e-5 | 50 | 10 | Higher capacity, extended convergence |
| `lora_r32_alpha64_lr3e5_pat10_e75` | 32 | 64 | 3e-5 | 75 | 10 | Best-of-breed: lowest LR, most epochs |
| `lora_r16_alpha32_lr5e5_pat5_e50` | 16 | 32 | 5e-5 | 50 | 5 | Conservative quick-converge variant |

### Key Changes from Baseline

1. **Higher LoRA Ranks** (16, 32 instead of 8): More adapter parameters for greater
   representational capacity
2. **Proportionally Higher Alpha** (32, 64 instead of 16): Alpha should scale with
   rank — this provides a more stable training signal
3. **Lower Learning Rate** (5e-5 instead of 2e-4): Prevents overshooting and allows
   more stable convergence over many epochs
4. **More Epochs** (50-75 instead of the previous early stop at 3): Gives the model
   time to converge before early stopping can kick in
5. **Longer Early Stopping Patience** (5-10 instead of 3): Prevents premature stopping
   when the model is still improving

## Files Created/Modified

### New Files

| File | Purpose |
|---|---|
| `scripts/run_multi_config_training.py` | Runs all LoRA presets sequentially with results comparison |
| `app/training/rag.py` | RAG retrieval pipeline for vulnerability patch generation |
| `app/training/patch_generator.py` | Separate patch generation strategy with CWE-specific templates |
| `STAGE5_EXPERIMENTS.md` | This summary document |

### Modified Files

| File | Changes |
|---|---|
| `app/training/config.py` | Added `LoRAPreset` dataclass, `LORA_PRESETS`, `LORA_PRESETS_EXTENDED`, `preset_to_sft_config()`, `get_preset()` |
| `app/training/__init__.py` | Exported new presets, RAG, and patch_generator modules |
| `output/stage5/eval_results.json` | Added `new_experiments` section documenting the configurations |

## Usage

### Run All Experiments

```bash
# Run all LoRA presets with actual training
python scripts/run_multi_config_training.py --all --train-jsonl output/stage3/train.jsonl --val-jsonl output/stage3/val.jsonl

# Run specific presets
python scripts/run_multi_config_training.py --presets lora_r16_alpha32_lr5e5,lora_r32_alpha64_lr5e5

# Dry run (estimate steps/VRAM without training)
python scripts/run_multi_config_training.py --dry-run --all
```

### Use Individual Presets with CLI

```bash
# Via Typer CLI
python -m app.training.cli sft --train-jsonl output/stage3/train.jsonl \
    --val-jsonl output/stage3/val.jsonl \
    --lora-r 16 --lora-alpha 32 --learning-rate 5e-5 --epochs 50 \
    --early-stopping --early-stopping-patience 7
```

### RAG Retrieval

```python
from app.training.rag import RAGRetriever, build_rag_patch_prompt

# Build index from training examples
retriever = RAGRetriever()
retriever.build_index(train_examples)

# Retrieve similar examples
results = retriever.retrieve(vulnerable_code, top_k=5)

# Build patch-generation prompt with RAG context
prompt = build_rag_patch_prompt(vulnerable_code, cwe_id="CWE-89", retrieval_results=results)
```

### Separate Patch Generation

```python
from app.training.patch_generator import PatchGenerator

generator = PatchGenerator(strategy="template")

# Generate patch for a CWE-89 vulnerability
result = generator.generate_patch(vulnerable_code, cwe_id="CWE-89")

# With RAG context
result = generator.generate_patch_with_rag(vulnerable_code, cwe_id="CWE-89", rag_results=results)

# Verify patch pattern
verification = verify_patch_pattern(result.patch_diff, cwe_id="CWE-89")
```

## Architecture

### Two-Stage Pipeline

```
Vulnerable Code → Stage 1: CWE Classification → Stage 2: Patch Generation
                         ↑                          ↑
                    LoRA Fine-tuned            CWE-specific templates
                    Model (LoRA)               + RAG Context
```

### Why Separate Classification and Patch Generation?

1. **Different Skills**: Classification requires identifying vulnerability types;
   patch generation requires knowing how to fix them
2. **Specialization**: A model can be better at one task than both simultaneously
3. **RAG Integration**: Patch generation benefits from seeing similar vulnerability
   fixes; classification benefits from seeing similar vulnerability patterns
4. **Iterative Refinement**: The classification output feeds the patch generator,
   allowing for feedback loops

### RAG Pipeline Architecture

```
Training Data → Embedding Model → Vector Index
                                         ↑
Query: Vulnerable Code → Embedding → Similarity Search → Top-K Results
                                                       ↓
                                              Few-Shot Prompt Builder
                                                       ↓
                                              LLM → Patch Generation
```

## CWE-Specific Patch Templates

The `PatchGenerator` includes templates for 6 CWE types:

| CWE | Fix Strategy |
|-----|-------------|
| CWE-89 | Parameterized queries (prepared statements) |
| CWE-79 | HTML escaping, framework auto-escaping |
| CWE-22 | Path validation, `realpath()` + allowlist |
| CWE-78 | `subprocess.run()` without `shell=True` |
| CWE-502 | `json.loads()`, `yaml.safe_load()` |
| CWE-190 | Bounds checking before arithmetic |

## Expected Improvements Over Baseline

1. **Higher LoRA ranks** should capture more complex vulnerability patterns
2. **Lower LR** should provide more stable convergence without oscillation
3. **Longer patience** should allow the model to fully converge before stopping
4. **RAG** should improve patch quality by providing similar examples
5. **Separate patch generation** should produce more targeted, correct patches

## Next Steps

1. Run `scripts/run_multi_config_training.py --all` to train all presets
2. Compare results in `output/stage5/multi_config_results.json`
3. Evaluate best model on Stage 6 test set
4. Integrate RAG into the inference pipeline
5. Add the separate patch generator to Stage 6 evaluation
