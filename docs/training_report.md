---
title: "vuln-triage-qwen2.5-coder-7b - Training Report"
date: "2026-08-12T11:50:14.198243+00:00"
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
license: mit
---

# Training Report: vuln-triage-qwen2.5-coder-7b

_Generated: 2026-08-12T11:50:14.198243+00:00_

## Overview

| Field | Value |
|---|---|
| Model | `vuln-triage-qwen2.5-coder-7b` |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Report ID | `stage11-543c2d5b` |
| Training runs | 1 |

## Training Runs

| Run ID | Method | Train set | Time (min) | VRAM (GB) | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| `lora_qwen-1.5b-lora-cpu_20260812_094446_df452891` | lora | 6 | 0.5 | 0.0 | 2.6842 | 2.9778 |

### Hyperparameters

**Run `lora_qwen-1.5b-lora-cpu_20260812_094446_df452891` (lora)**

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
| 0 | 2.5081 |
| 1 | 2.7991 |
| 2 | 3.0871 |
| 3 | 3.0212 |
| 4 | 2.5726 |
| 5 | 3.1438 |
| 6 | 2.7639 |
| 7 | 2.1323 |
| 8 | 2.9259 |
| 9 | 2.3439 |
| 10 | 2.6062 |
| 11 | 2.3060 |

## Evaluation Results

### Stage 4 — Pre-fine-tuning Baseline

| Metric | Value |
|---|---|
| Run ID | `stage4_real_20260812_143510` |
| CWE Macro-F1 | 0.0667 |
| Severity accuracy | 0.3333 |
| Hallucination rate | 0.0000 |
| Patch coverage | 1.0000 |

### Stage 6 — Tuned Model Four-Tier Evaluation

| Metric | Value |
|---|---|
| Run ID | `stage4_real_20260812_143510` |
| CWE Macro-F1 | 0.0667 |
| Severity accuracy | 0.3333 |
| Hallucination rate | 0.3333 |
| Patch coverage | 0.6667 |
| Exec pass rate | 0.0000 |

| CWE | Precision | Recall | F1 |
|---|---|---|---|
| CWE-190 | 0.0000 | 0.0000 | 0.0000 |
| CWE-22 | 0.0000 | 0.0000 | 0.0000 |
| CWE-502 | 0.0000 | 0.0000 | 0.0000 |
| CWE-78 | 0.0000 | 0.0000 | 0.0000 |
| CWE-79 | 0.0000 | 0.0000 | 0.0000 |
| CWE-89 | 0.2500 | 1.0000 | 0.4000 |

## Conclusions

- Run `lora_qwen-1.5b-lora-cpu_20260812_094446_df452891` (lora): train loss = 2.6842, val loss = 2.9778.
- Tuned model Stage 6 evaluation: CWE Macro-F1 = 0.0667, Severity accuracy = 0.3333, Patch coverage = 0.6667.
- CWE Macro-F1 is low (0.0667) — the small training set (6 samples) limits multi-class discrimination. The model defaults to CWE-89 for most inputs, which inflates recall but not precision.
- Pre-fine-tuning baseline: CWE Macro-F1 = 0.0667, Severity accuracy = 0.3333.

## Recommendations

- Scale to a larger training dataset (current: 6 samples) to improve multi-class CWE discrimination.
- Increase training epochs or try a higher LoRA rank (current: r=8) to reduce underfitting on the small dataset.
- Re-run the Stage 10 regression gate after any model update.
- Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.
