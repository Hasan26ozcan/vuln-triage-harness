---
title: "vuln-triage-qwen2.5-coder-7b - Training Report"
date: "2026-08-11T21:51:33.597083+00:00"
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
license: mit
---

# Training Report: vuln-triage-qwen2.5-coder-7b

_Generated: 2026-08-11T21:51:33.597083+00:00_

## Overview

| Field | Value |
|---|---|
| Model | `vuln-triage-qwen2.5-coder-7b` |
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Report ID | `stage11-e8115467` |
| Training runs | 0 |

## Evaluation Results

## Conclusions

- No real training runs have been executed yet. The conclusions below describe the intended methodology, not measured results — training results will be populated once Stage 5 is run on a GPU.
- QLoRA (4-bit NF4) enables parameter-efficient fine-tuning on consumer GPUs with 8 GB VRAM (estimated, not measured).
- SFT full-parameter training requires >=16 GB VRAM.
- The LoRA rank sweep (ranks 8—128) is designed to identify the smallest adapter that preserves quality.
- DPO preference alignment is intended to reduce hallucination rate without sacrificing classification accuracy.

## Recommendations

- Run Stage 5 training on a CUDA GPU before publishing real metrics.
- Re-run the Stage 10 regression gate after any model update.
- Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.
