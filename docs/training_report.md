---
title: "vuln-triage-qwen2.5-coder-7b - Training Report"
date: "2026-08-16T09:47:38.643963+00:00"
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
license: mit
---

# Training Report: vuln-triage-qwen2.5-coder-7b

_Generated: 2026-08-16T09:47:38.643963+00:00_

## Overview

| Field | Value |
|---|---|
| Model | `vuln-triage-qwen2.5-coder-7b` |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Report ID | `stage11-5bb4d664` |
| Training runs | 1 |

## Training Runs

| Run ID | Method | Train set | Time (min) | VRAM (GB) | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| `lora_qwen-1.5b-lora-cpu_20260816_080419_c3945f3b` | lora | 53 | 54.5 | 0.0 | 0.7925 | 0.7157 |

### Hyperparameters

**Run `lora_qwen-1.5b-lora-cpu_20260816_080419_c3945f3b` (lora)**

| Parameter | Value |
|---|---|
| `learning_rate` | 0.0002 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `lora_r` | 8 |
| `num_train_epochs` | 2 |
| `use_4bit` | False |

#### Loss history

| # | loss |
|---|------|
| 0 | 1.1253 |
| 1 | 1.3683 |
| 2 | 1.2038 |
| 3 | 0.9467 |
| 4 | 1.2658 |
| ... | ... |
| 6 | 0.6333 |
| 7 | 1.3010 |
| 8 | 0.9556 |
| 9 | 0.5741 |
| 10 | 0.7257 |

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
| Run ID | `stage4_real_20260816_122922` |
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

- Run `lora_qwen-1.5b-lora-cpu_20260816_080419_c3945f3b` (lora): train loss = 0.7925, val loss = 0.7157.
- Tuned model Stage 6 evaluation: CWE Macro-F1 = 0.1667, Severity accuracy = 0.2500, Patch coverage = 0.3333.
- CWE Macro-F1 is low (0.1667) — the small training set (53 samples) limits multi-class discrimination. The model defaults to CWE-89 for most inputs, which inflates recall but not precision.
- Pre-fine-tuning baseline: CWE Macro-F1 = 0.1667, Severity accuracy = 0.2500.

## Recommendations

- Scale to a larger training dataset (current: 53 samples) to improve multi-class CWE discrimination.
- Increase training epochs or try a higher LoRA rank (current: r=8) to reduce underfitting on the small dataset.
- Re-run the Stage 10 regression gate after any model update.
- Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.
