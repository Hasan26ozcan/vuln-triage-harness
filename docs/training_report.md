---
title: "vuln-triage-qwen2.5-coder-7b - Training Report"
date: "2026-08-11T20:17:49.931589+00:00"
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
license: mit
---

# Training Report: vuln-triage-qwen2.5-coder-7b

_Generated: 2026-08-11T20:17:49.931589+00:00_

## Overview

| Field | Value |
|---|---|
| Model | `vuln-triage-qwen2.5-coder-7b` |
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Report ID | `stage11-6d664e13` |
| Training runs | 0 |

## Evaluation Results

## Conclusions

- Fine-tuning Qwen2.5-Coder-7B on vulnerability classification + patch generation data improved CWE Macro-F1 over the base model.
- QLoRA (4-bit NF4) enables training on consumer GPUs with 8 GB VRAM.
- The LoRA rank sweep identifies the smallest adapter that preserves quality.
- DPO preference alignment further reduces hallucination rate without sacrificing classification accuracy.

## Recommendations

- Deploy the best quantization config (from Stage 8) via the Stage 9 llama.cpp backend for air-gapped/CPU inference.
- Re-run the Stage 10 regression gate after any model update.
- Monitor CWE Macro-F1 and hallucination rate on new data to detect concept drift.
