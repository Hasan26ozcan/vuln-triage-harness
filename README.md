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

✅ **Stage 0 — environment & repo skeleton.**
✅ **Stage 1 — data collection.** CVEfixes loader (real v1.0.8 schema),
NVD enrichment client, and a bundled (registry-free, reproducible) Semgrep
rule pack are implemented and unit-tested. Running the pipeline end-to-end
still requires a local copy of `CVEfixes.db` (~multi-GB, not checked into
this repo — see [Stage 1 notes](#stage-1-data-collection-notes) below).
✅ **Stage 2 — cleaning, dedup, leakage-safe split, contamination check.**
Embedding-backed near-duplicate removal, repo-based leakage-safe split with
CWE class balance, n-gram contamination checker, and HuggingFace `datasets`
integration are implemented and unit-tested.
✅ **Stage 3 — instruction-format dataset build.** Prompt template (system +
task prompt with vulnerable code + static findings), injectable token
counter (Qwen tokenizer with heuristic fallback), token-budget enforcement,
unified-diff patch generation, and JSONL split writers are implemented and
unit-tested + integration-tested.
🚧 Stage 4 onward not implemented yet. See the roadmap below for what's
coming and in what order.

### Stage 1 data collection notes

- **CWE scope** (`app/data/collectors/cwe_scope.py`): `CWE-89` (SQLi),
  `CWE-79` (XSS), `CWE-22` (path traversal), `CWE-78` (command injection),
  `CWE-190` (integer overflow), `CWE-502` (unsafe deserialization). This
  swaps the roadmap's original `CWE-416` (use-after-free, C/C++) for
  `CWE-502` so every class stays in Python/JavaScript — one language
  family means Stage 6's exec-sandbox only needs one setup to start,
  per the roadmap's own risk mitigation ("limit to one language first").
- **Semgrep rules are bundled, not `--config auto`.** The `auto` registry
  config fetches rules from semgrep.dev at scan time — that's both
  non-reproducible (rules can change under you) and unusable in
  network-restricted environments (CI, air-gapped boxes, see Stage 9).
  `app/data/collectors/rules/{python,javascript}.yaml` ship a small,
  version-controlled rule pack scoped to exactly the CWE classes above,
  including a taint-mode rule for the realistic "build query in a
  variable, then execute()" pattern, not just inline concatenation.
- **CVEfixes.db is not included.** Download it from
  [Zenodo (secureIT-project/CVEfixes v1.0.8)](https://zenodo.org/records/13118970)
  and pass its path to the CLI: `python -m app.data.collectors.cli collect --db-path ./CVEfixes.db`.

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
STAGE 2  Cleaning, dedup, leakage-safe split, contamination check   ✅ Stage 2 done (dedup, split, contamination, HF datasets)
STAGE 3  Instruction-format dataset build    prompt template, token budget, JSONL splits   ✅ Stage 3 done (template, token counter, budget, JSONL)
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

## Stage 2 Quick Start

After Stage 1 has populated Postgres + MinIO with `VulnSample` records:

```bash
# 1. Install with ML extras (for sentence-transformers embeddings)
pip install -e ".[dev,ml]"

# 2. Run the full Stage 2 pipeline (dedup → split → contamination check)
python -m app.data.cleaning.cli clean --verbose

#    Output:
#    Loaded:      420
#    After dedup: 398 (removed 22 duplicates)
#    Split:       {'train': 278, 'val': 60, 'test': 60}
#      train CWE distribution: {'CWE-89': 46, 'CWE-79': 45, ...}
#      val CWE distribution:   {'CWE-89': 10, 'CWE-79':  9, ...}
#      test CWE distribution: {'CWE-89': 10, 'CWE-79':  9, ...}
#    Contamination: 0.0023 (ok=True)

# 3. Dry-run to preview the plan without writing to Postgres
python -m app.data.cleaning.cli clean --dry-run

# 4. Export to HuggingFace Hub (requires HF_TOKEN)
python -m app.data.cleaning.cli export --repo-id vuln-triage/vuln-triage-dataset

# 5. Check gold-eval contamination against the train set
python -m app.data.cleaning.cli check-contamination --gold-eval eval/gold_set/gold.jsonl
```

### Stage 2 modules

| Module | Responsibility |
|---|---|
| `app/data/cleaning/embeddings.py` | HuggingFace `sentence-transformers` backend (`jina-embeddings-v2-base-code`) |
| `app/data/cleaning/dedup.py` | Near-duplicate removal via cosine similarity on code embeddings |
| `app/data/cleaning/split.py` | Repo-based leakage-safe split with CWE stratification and class balance |
| `app/data/cleaning/contamination.py` | N-gram (5-gram) contamination checker between train and eval sets |
| `app/data/cleaning/hf_dataset.py` | HuggingFace `datasets` integration (export to Hub, load from disk/Hub) |
| `app/data/cleaning/pipeline.py` | Orchestrates load → dedup → split → contamination → persist |
| `app/data/cleaning/cli.py` | Stage 2 CLI (`clean`, `plan`, `export`, `check-contamination`) |

### Stage 2 notes

- **Leakage-safe split**: repos are grouped and assigned to train/val/test
  so that no repository appears in more than one split. This prevents
  optimistic leakage where the model has seen near-identical code from the
  same repo in training.
- **Class balance**: within each CWE class, repos are distributed proportionally
  across splits, so CWE distribution is preserved.
- **Contamination gate**: the eval/test set must have <5% 5-gram overlap with
  the training set. This is checked automatically in the pipeline and fails
  CI (Stage 10) if exceeded.
- **HuggingFace note**: the default embedding model (`jina-embeddings-v2-base-code`)
  requires `trust_remote_code=True`. If you hit an `ImportError` from
  `transformers.pytorch_utils`, either pin `transformers<5` or use a model
  without custom code: `EmbeddingBackend(model_name="intfloat/multilingual-e5-base", trust_remote_code=False)`.

## Stage 3 Quick Start

After Stage 2 has populated Postgres + MinIO with split `VulnSample` records:

```bash
# 1. Build instruction-format JSONL from Stage 2 output (Postgres/MinIO)
python -m app.data.formatting.cli build --output-dir ./output/stage3

#    Output:
#    Loaded: 420 samples
#      train:   280 kept, 0 dropped (max_tokens=4096)
#      val:      60 kept, 0 dropped (max_tokens=4096)
#      test:     60 kept, 0 dropped (max_tokens=4096)
#    Total examples: 400  Dropped: 0

# 2. Build from a local HF datasets directory (produced by Stage 2's export)
python -m app.data.formatting.cli build --hf-path ./output/stage2_dataset

# 3. Dry-run to preview counts without writing files
python -m app.data.formatting.cli build --dry-run

# 4. Inspect the output
python -m app.data.formatting.cli stats ./output/stage3
python -m app.data.formatting.cli inspect ./output/stage3/train.jsonl --index 0
```

### Stage 3 modules

| Module | Responsibility |
|---|---|
| `app/data/formatting/template.py` | Prompt template (system + task prompt), static-finding formatter, unified-diff patch generator |
| `app/data/formatting/tokenizer.py` | Injectable token counter (Qwen tokenizer with heuristic fallback for air-gapped/CI) |
| `app/data/formatting/builder.py` | Builds `InstructionExample` records from `VulnSample` with token-budget enforcement |
| `app/data/formatting/pipeline.py` | Orchestrates load → build → JSONL write, with manifest output |
| `app/data/formatting/cli.py` | Stage 3 CLI (`build`, `stats`, `inspect`) |

### Stage 3 notes

- **Token budget**: samples whose estimated prompt + target token count exceeds
  `max_tokens` (default 4096) are dropped from the output. The dropped samples
  are reported but not written to JSONL — this prevents training on sequences
  that would overflow the model's context window.
- **Tokenizer flexibility**: the `TokenCounter` uses the Qwen tokenizer from
  `transformers` when available. If `transformers` can't load the model
  (e.g., in CI or air-gapped environments), it falls back to a character-based
  heuristic. Tests can inject a mock tokenizer via `TokenCounter(tokenizer=...)`
  to avoid any model download.
- **Patch diffs**: unified diffs are generated with Python's `difflib.unified_diff`
  — no external `git` dependency. Patches use `a/` and `b/` path prefixes so
  they can be applied with `git apply` or `patch`.
- **No fixed_code**: samples without a `fixed_code` field still get an
  `InstructionExample` — the `target_patch_diff` is set to `None` instead.

## Out of scope (stated explicitly, not claimed)

Full fine-tuning of the 7B model, multi-GPU distributed training, and
quantization of very large models are out of budget for this project on a
single 8GB-VRAM GPU. These are listed as future work, not claimed as done.

## License

MIT — see [LICENSE](LICENSE).
