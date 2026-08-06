# Vulnerability Triage & Patch-Suggestion Fine-Tuning Harness

An end-to-end post-training harness that fine-tunes an open-weight code LLM
(Qwen2.5-Coder-7B-Instruct) on real CVE-patch pairs plus static-analysis
signal, so it can **classify** a vulnerability (CWE + severity) and
**propose a working patch** — validated by a four-tier evaluation harness
that includes exec-based sandbox testing rather than relying on an LLM
judge alone.

> **Scope, stated up front:** this is not a general-purpose "AI security
> scanner." It targets a narrow set of 5-8 CWE classes on a small/mid-size
> open model, with a measured, reproducible before/after comparison at
> every stage — full fine-tune, LoRA rank sweep, DPO preference alignment,
> and quantization trade-offs.

## Status

🚧 **Stage 0 — environment & repo skeleton.** Data collection, training,
and evaluation stages are not implemented yet. See the roadmap below for
what's coming and in what order.

## Why this project

- A real training loop (SFT + LoRA/QLoRA + DPO), not just prompting a base model.
- Leakage-safe data discipline: repo-based splits, embedding dedup, n-gram
  contamination checks on the gold-eval set.
- **Exec-based evaluation**: proposed patches are actually applied and run
  against the project's test suite in a sandboxed container — LLM-judge
  scoring is used only for explanation quality, not pass/fail.
- Explicit quantization/deployment constraint: the final checkpoint has to
  run air-gapped, on consumer hardware.
- CI/CD and security scanning from day one (pytest, ruff, Bandit, Gitleaks, Trivy).

## Architecture

```
STAGE 0  Environment & repo skeleton         (this stage)
STAGE 1  Data collection & labeling          CVEfixes/BigVul/OSV -> VulnSample
STAGE 2  Cleaning, dedup, leakage-safe split repo-based split, contamination check
STAGE 3  Instruction-format dataset build    prompt template, token budget, JSONL splits
STAGE 4  Pre-fine-tuning baseline            zero-shot / few-shot base model on gold-eval
STAGE 5  Training matrix                     SFT (full/QLoRA) · LoRA rank sweep · DPO
STAGE 6  Four-tier evaluation harness        deterministic -> embedding/static -> exec -> LLM-judge
STAGE 7  Regression / forgetting analysis    general code-capability delta, before/after
STAGE 8  Quantization matrix                 GPTQ / AWQ / GGUF, quality vs. speed/VRAM
STAGE 9  Air-gapped serving                  llama.cpp/Ollama, network-isolated Docker, CLI + API
STAGE 10 CI/CD & regression gate             pytest, Bandit, Gitleaks, Trivy, automated eval gate
STAGE 11 Documentation & interview package   README, model card, training report, demo
```

Cross-cutting infrastructure: **PostgreSQL** for experiment/metric state,
**Redis + Celery** for long-running jobs (training, quantization), **W&B**
for loss curves and eval tracking, **MinIO/S3** for model checkpoint and
dataset artifact storage.

## Repo layout

```
vuln-triage-harness/
├── app/
│   ├── schemas/          # Pydantic v2 data contracts (all stages)
│   ├── data/
│   │   ├── collectors/   # CVEfixes/BigVul/NVD/OSV downloaders   (Stage 1)
│   │   ├── cleaning/     # dedup, leakage-safe split              (Stage 2)
│   │   └── formatting/   # instruction-format dataset builder     (Stage 3)
│   ├── training/         # train_sft.py, train_lora_sweep.py, train_dpo.py (Stage 5)
│   ├── evaluation/       # tier1_deterministic.py ... tier4_llm_judge.py   (Stage 6-7)
│   ├── quantization/     # export_gptq.py, export_awq.py, export_gguf.py  (Stage 8)
│   ├── serving/          # cli.py, api.py                                 (Stage 9)
│   └── storage/          # Postgres models, MinIO client
├── eval/gold_set/        # 40-60 manually verified examples
├── sandbox/              # per-language Docker images for exec-based eval
├── tests/{unit,integration}/
├── docker-compose.yml    # Postgres + Redis + MinIO
├── .github/workflows/ci.yml
└── pyproject.toml
```

## Tech stack

| Layer | Choice |
|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct (primary), 1.5B (fast iteration) |
| PEFT | `peft` (LoRA/QLoRA), `bitsandbytes` 4-bit NF4 |
| Preference tuning | `trl` `DPOTrainer` |
| Data source | CVEfixes / BigVul (CVE→commit→diff mapped), NVD API, OSV.dev |
| Static signal | Semgrep |
| Exec eval | Docker sandbox + language-specific test runner |
| Dedup | `sentence-transformers` code-embedding model |
| Experiment tracking | Weights & Biases |
| Quantization | AutoGPTQ, AutoAWQ, llama.cpp (GGUF) |
| Serving | llama.cpp server (air-gapped/CPU), vLLM (GPU) |
| Orchestration | Celery + Redis |
| State/metrics DB | PostgreSQL |
| CI/CD | GitHub Actions — pytest, ruff, Bandit, Gitleaks, Trivy |

## Quickstart (Stage 0)

```bash
# 1. Install dependencies (ML extras are optional — not needed for Stage 0)
pip install -e ".[dev]"

# 2. Bring up Postgres + Redis + MinIO
docker compose up -d

# 3. Run the test suite
pytest tests/unit -v
```

## Evaluation metrics (defined up front, measured at every checkpoint)

| Metric | Definition |
|---|---|
| CWE Macro-F1 | Per-class F1 averaged across CWE classes (accuracy is misleading on imbalanced data) |
| Exec Pass Rate | Share of predictions where `tests_pass_after_patch = True` |
| Hallucination Rate | Share of predictions with a fabricated CWE ID or a reference to nonexistent code |
| Cost per Accepted Patch | (inference $ + amortized training $) / patches passing exec-eval |
| Forgetting Delta | general-capability-score(tuned) − general-capability-score(base) |

## Out of scope (stated explicitly, not claimed)

Full fine-tuning of the 7B model, multi-GPU distributed training, and
quantization of very large models are out of budget for this project on a
single 8GB-VRAM GPU. These are listed as future work, not claimed as done.

## License

MIT — see [LICENSE](LICENSE).
