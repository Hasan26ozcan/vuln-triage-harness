---
title: "vuln-triage-qwen2.5-coder-1.5b - Training Report"
date: "2026-08-17T08:49:50.501436+00:00"
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
license: mit
---

# Training Report: vuln-triage-qwen2.5-coder-1.5b

_Generated: 2026-08-17T08:49:50.501436+00:00_

## Overview

| Field | Value |
|---|---|
| Model | `vuln-triage-qwen2.5-coder-1.5b` |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Report ID | `stage11-df8b41a5` |
| Training runs | 1 |

## Training Runs

| Run ID | Method | Train set | Time (min) | VRAM (GB) | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| `sft_qlora_qwen-1.5b-qlora-gpu_20260816_125651_9b39fcd6` | sft_qlora | 404 | 36.4 | 8.8 | 0.8419 | 0.6996 |

### Hyperparameters

**Run `sft_qlora_qwen-1.5b-qlora-gpu_20260816_125651_9b39fcd6` (sft_qlora)**

| Parameter | Value |
|---|---|
| `learning_rate` | 0.0002 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `lora_r` | 8 |
| `num_train_epochs` | 3 |
| `use_4bit` | True |

#### Loss history

| # | loss |
|---|------|
| 0 | 1.0454 |
| 1 | 1.1344 |
| 2 | 1.2830 |
| 3 | 1.0724 |
| 4 | 1.0583 |
| ... | ... |
| 6 | 0.8451 |
| 7 | 0.8701 |
| 8 | 0.6331 |
| 9 | 0.8241 |
| 10 | 0.6831 |

## Evaluation Results

### Stage 4 — Pre-fine-tuning Baseline

| Metric | Value |
|---|---|
| Run ID | `stage4_real_20260816_122922` |
| CWE Macro-F1 | 0.1667 |
| Severity accuracy | 0.2500 |
| Hallucination rate | 0.5000 |
| Patch coverage | 1.0000 |

### Stage 6 — Tuned Model Four-Tier Evaluation

| Metric | Value |
|---|---|
| Run ID | `stage6-ac23dfa2` |
| CWE Macro-F1 | 0.1667 |
| Severity accuracy | 0.2500 |
| Hallucination rate | 0.8333 |
| Patch coverage | 0.3333 |
| Exec pass rate | 0.0000 |

| CWE | Precision | Recall | F1 |
|---|---|---|---|
| CWE-190 | 0.0000 | 0.0000 | 0.0000 |
| CWE-22 | 0.0000 | 0.0000 | 0.0000 |
| CWE-502 | 0.0000 | 0.0000 | 0.0000 |
| CWE-78 | 0.0000 | 0.0000 | 0.0000 |
| CWE-79 | 0.0000 | 0.0000 | 0.0000 |
| CWE-89 | 1.0000 | 1.0000 | 1.0000 |

## Conclusions

- Run `sft_qlora_qwen-1.5b-qlora-gpu_20260816_125651_9b39fcd6` (sft_qlora): train loss = 0.8419, val loss = 0.6996.
- Tuned model Stage 6 evaluation: CWE Macro-F1 = 0.1667, Severity accuracy = 0.2500, Patch coverage = 0.3333.
- CWE Macro-F1 is low (0.1667) — the small training set (404 samples) limits multi-class discrimination. The model defaults to CWE-89 for most inputs, which inflates recall but not precision.
- Pre-fine-tuning baseline: CWE Macro-F1 = 0.1667, Severity accuracy = 0.2500.

## Recommendations

- Scale to a larger training dataset (current: 404 samples) to improve multi-class CWE discrimination.
- Increase training epochs or try a higher LoRA rank (current: r=8) to reduce underfitting on the small dataset.
- Re-run the Stage 10 regression gate after any model update.
- Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.
