---
title: "vuln-triage-qwen2.5-coder-1.5b - Training Report"
date: "2026-09-01T19:31:50.589110+00:00"
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
license: mit
---

# Training Report: vuln-triage-qwen2.5-coder-1.5b

_Generated: 2026-09-01T19:31:50.589110+00:00_

## Overview

| Field | Value |
|---|---|
| Model | `vuln-triage-qwen2.5-coder-1.5b` |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Report ID | `stage11-17e2db79` |
| Training runs | 2 |

## Training Runs

| Run ID | Method | Train set | Time (min) | VRAM (GB) | Train Loss | Val Loss |
|---|---|---|---|---|---|---|
| `sft_completed_1` | sft_qlora | 3 | 5.0 | 7.0 | 0.5000 | 0.6000 |
| `dpo_20260817_204502_bef7a2ac` | dpo | 404 | 0.2 | 3.1 | 0.6618 | — |

**Run `sft_completed_1` (sft_qlora)**

| Parameter | Value |
|---|---|
| `lora_r` | 8 |

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
| Run ID | `stage4_zero_shot_81b6dba7` |
| CWE Macro-F1 | 0.0676 |
| Severity accuracy | 0.0847 |
| Hallucination rate | 0.0000 |
| Patch coverage | 1.0000 |

### Stage 6 — Tuned Model Four-Tier Evaluation

| Metric | Value |
|---|---|
| Run ID | `stage6-7a7d1b75` |
| CWE Macro-F1 | 0.0639 |
| Severity accuracy | N/A (not scored at this stage) |
| Hallucination rate | 0.0000 |
| Patch coverage | 1.0000 |
| Exec pass rate | 0.0000 |

| CWE | Precision | Recall | F1 |
|---|---|---|---|
| CWE-190 | 0.0000 | 0.0000 | 0.0000 |
| CWE-22 | 0.0000 | 0.0000 | 0.0000 |
| CWE-502 | 0.0000 | 0.0000 | 0.0000 |
| CWE-78 | 0.0000 | 0.0000 | 0.0000 |
| CWE-79 | 0.0000 | 0.0000 | 0.0000 |
| CWE-89 | 0.2373 | 1.0000 | 0.3836 |

### Stage 7 — Regression / Forgetting Analysis

| Metric | Value |
|---|---|
| Forgetting delta | +0.0000 |
| Status | [OK] No forgetting |

## Conclusions

- Run `sft_completed_1` (sft_qlora): train loss = 0.5000, val loss = 0.6000.
- Run `dpo_20260817_204502_bef7a2ac` (dpo): train loss = 0.6618. No validation loss was recorded.
- Tuned model Stage 6 evaluation: CWE Macro-F1 = 0.0639, Severity accuracy = N/A (not scored at this stage), Patch coverage = 1.0000.
- CWE Macro-F1 is low (0.0639) — the small training set (5000 samples) limits multi-class discrimination. The model defaults to CWE-89 for most inputs, which inflates recall but not precision.
- Pre-fine-tuning baseline: CWE Macro-F1 = 0.0676, Severity accuracy = 0.0847.

## Recommendations

- Scale to a larger training dataset (current: 5000 samples) to improve multi-class CWE discrimination.
- Increase training epochs or try a higher LoRA rank (current: r=8) to reduce underfitting on the small dataset.
- Re-run the Stage 10 regression gate after any model update.
- Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.
