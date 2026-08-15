---
title: "vuln-triage-qwen2.5-coder-7b — Vulnerability Triage Model Card"
date: "2026-08-15T09:17:53.464121+00:00"
base_model: Qwen/Qwen2.5-Coder-7B-Instruct
training_method: sft_qlora
license: mit
tags:
  - code-generation
  - vulnerability-detection
  - security
---

# Model Card: vuln-triage-qwen2.5-coder-7b

## Model Details

| Field | Value |
|---|---|
| Model name | `vuln-triage-qwen2.5-coder-7b` |
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Fine-tuned | yes |
| Training method | sft_qlora |
| LoRA rank | 64 |
| Language | python |
| CWE scope | CWE-89, CWE-79, CWE-22, CWE-78, CWE-190, CWE-502 |
| Training data size | 131 samples |

## Intended Use

- Classifying the CWE category of a vulnerable code snippet
- Suggesting a minimal, working patch for the vulnerability
- Batch analysis of code repositories for triage prioritization
- Interactive vulnerability analysis via the air-gapped serving layer

## Evaluation

| Metric | Value |
|---|---|
| Stage | 6 |
| CWE Macro-F1 | 0.0000 |
| Severity accuracy | 0.0000 |
| Hallucination rate | 0.0000 |
| Patch coverage | 0.0000 |
| Exec pass rate | 0.0000 |

## Serving

The model can be served air-gapped via:

- `llama.cpp`
- `ollama`
- `mock`

## Limitations

- Trained on 6 CWE classes; out-of-scope CWEs are treated as hallucinations.
- Not a general-purpose security scanner — does not detect logic bugs, configuration issues, or CWE classes outside the listed scope.
- The exec-based evaluation runs proposed patches in a sandboxed subprocess. Docker isolation is implemented (see `app/evaluation/tier3_exec.py`), providing read-only filesystem, no network, and memory limits.
- Proposed patches should be reviewed by a human before merging into production.
- Trained on a small subset (131 samples) using MOCK execution; the full training pipeline supports GPU/QLoRA for larger datasets.
- This model predicts CWE-89 for most samples due to the small training set; additional training data and epochs are needed for multi-class accuracy.

## Ethical Considerations

- This model is a research artifact, not a production SOC tool.
- Do not use model predictions as the sole basis for security decisions.
- The model may produce incorrect patches — verify all suggestions before use.

## Out of Scope

- Real-time repository monitoring in CI pipelines
- Network-based vulnerability scanning (no port scanning, no HTTP fuzzing)
- Supply-chain security / third-party dependency auditing (use pip-audit/Safety)
- Legal or compliance assessment of software
- Production incident response

## Citation

If you use this model in your research, please cite:

```
@misc{vuln-triage-harness,
  title={Vulnerability Triage & Patch-Suggestion Harness},
  author={Ozcan, Hasan},
  year={2024},
  url={https://github.com/hasanozcan/vuln-triage-harness}
}
```
