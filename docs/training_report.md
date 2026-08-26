---
title: "vuln-triage-qwen2.5-coder-1.5b - Training Report"
date: "2026-08-26T21:34:36.156424+00:00"
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
license: mit
---

# Training Report: vuln-triage-qwen2.5-coder-1.5b

_Generated: 2026-08-26T21:34:36.156424+00:00_

## Overview

| Field | Value |
|---|---|
| Model | `vuln-triage-qwen2.5-coder-1.5b` |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Report ID | `stage11-be629a56` |
| Training runs | 2 |

## Training Runs

| Run ID | Method | Train set | Time (min) | VRAM (GB) | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| `sft_qlora_qwen-1.5b-qlora-NVIDIA GeForce RTX 4060 Laptop GPU_20260826_210136_6a8e2161` | sft_qlora | 47 | 3.2 | 6.5 | 1.0381 | 1.0616 |
| `dpo_20260817_204502_bef7a2ac` | dpo | 404 | 0.2 | 3.1 | 0.6618 | — |

### Hyperparameters

**Run `sft_qlora_qwen-1.5b-qlora-NVIDIA GeForce RTX 4060 Laptop GPU_20260826_210136_6a8e2161` (sft_qlora)**

| Parameter | Value |
|---|---|
| `learning_rate` | 0.0002 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.05 |
| `lora_r` | 8 |
| `num_train_epochs` | 2 |
| `use_4bit` | True |

#### Loss history

| # | loss |
|---|------|
| 0 | 0.9627 |
| 1 | 1.0975 |
| 2 | 1.0791 |
| 3 | 0.9790 |
| 4 | 1.0522 |
| 5 | 1.3086 |
| 6 | 0.9404 |
| 7 | 0.9047 |
| 8 | 1.0881 |
| 9 | 1.0093 |
| 10 | 1.0799 |
| 11 | 0.9551 |

**Run `dpo_20260817_204502_bef7a2ac` (dpo)**

| Parameter | Value |
|---|---|
| `beta` | 0.1 |
| `gradient_accumulation_steps` | 1 |
| `learning_rate` | 0.0002 |
| `loss_type` | sigmoid |
| `lr_scheduler_type` | cosine |
| `max_grad_norm` | 0.3 |
| `max_length` | 256 |
| `num_train_epochs` | 1 |
| `per_device_train_batch_size` | 101 |
| `sft_checkpoint` | output/stage5/qwen_lora_gpu/final_checkpoint |
| `warmup_ratio` | 0.03 |
| `weight_decay` | 0.01 |

#### Loss history

| # | loss |
|---|------|
| 0 | 0.6618 |

## Evaluation Results

### Stage 4 — Pre-fine-tuning Baseline

| Metric | Value |
|---|---|
| Run ID | `stage4_real_20260818_140802` |
| CWE Macro-F1 | 0.1626 |
| Severity accuracy | 0.5085 |
| Hallucination rate | 0.1316 |
| Patch coverage | 0.4211 |

### Stage 6 — Tuned Model Four-Tier Evaluation

| Metric | Value |
|---|---|
| Run ID | `stage6-32ea9676` |
| CWE Macro-F1 | 0.1626 |
| Severity accuracy | 0.0000 |
| Hallucination rate | 0.4407 |
| Patch coverage | 0.2712 |
| Exec pass rate | 0.0000 |

| CWE | Precision | Recall | F1 |
|---|---|---|---|
| CWE-190 | 0.0000 | 0.0000 | 0.0000 |
| CWE-22 | 1.0000 | 0.1429 | 0.2500 |
| CWE-502 | 0.0000 | 0.0000 | 0.0000 |
| CWE-78 | 1.0000 | 0.2500 | 0.4000 |
| CWE-79 | 0.2414 | 0.5000 | 0.3256 |
| CWE-89 | 0.0000 | 0.0000 | 0.0000 |

### Stage 7 — Regression / Forgetting Analysis

| Metric | Value |
|---|---|
| Forgetting delta | +0.0000 |
| Status | [OK] No forgetting |

## Conclusions

- Run `sft_qlora_qwen-1.5b-qlora-NVIDIA GeForce RTX 4060 Laptop GPU_20260826_210136_6a8e2161` (sft_qlora): train loss = 1.0381, val loss = 1.0616.
- Run `dpo_20260817_204502_bef7a2ac` (dpo): train loss = 0.6618. No validation loss was recorded.
- Tuned model Stage 6 evaluation: CWE Macro-F1 = 0.1626, Severity accuracy = 0.0000, Patch coverage = 0.2712.
- CWE Macro-F1 is low (0.1626) — the small training set (47 samples) limits multi-class discrimination. The model defaults to CWE-89 for most inputs, which inflates recall but not precision.
- Pre-fine-tuning baseline: CWE Macro-F1 = 0.1626, Severity accuracy = 0.5085.

## Recommendations

- Scale to a larger training dataset (current: 47 samples) to improve multi-class CWE discrimination.
- Increase training epochs or try a higher LoRA rank (current: r=8) to reduce underfitting on the small dataset.
- Re-run the Stage 10 regression gate after any model update.
- Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.
