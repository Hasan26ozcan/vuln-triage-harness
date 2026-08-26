# Vulnerability Triage & Patch-Suggestion Fine-Tuning Harness

An end-to-end post-training harness that fine-tunes an open-weight code LLM
(Qwen2.5-Coder-7B-Instruct — the **designed-for** model; CPU-only validation runs
use the 1.5B-Instruct variant due to hardware constraints) on real CVE-patch
pairs plus static-analysis signal, so it can **classify** a vulnerability (CWE +
severity) and **propose a working patch** — validated by a four-tier evaluation
harness that includes exec-based sandbox testing rather than relying on an LLM
judge alone.

> **Scope, stated up front:** this is not a general-purpose "AI security
> scanner." It targets a narrow set of 6 CWE classes on a small/mid-size
> open model, with a measured, reproducible before/after comparison at
> every stage — full fine-tune, LoRA rank sweep, DPO preference alignment,
> and quantization trade-offs.

## Table of Contents

- [Status](#status)
- [Why this project](#why-this-project)
- [Out of Scope](#out-of-scope)
- [Architecture](#architecture)
- [Repo Layout](#repo-layout)
- [Tech Stack](#tech-stack)
- [Quickstart (Stage 0)](#quickstart-stage-0)
- [Evaluation Metrics](#evaluation-metrics)
- [Stage 1 — Data Collection](#stage-1--data-collection)
- [Stage 2 — Cleaning, Dedup, Leakage-Safe Split](#stage-2--cleaning-dedup-leakage-safe-split)
- [Stage 3 — Instruction-Format Dataset Build](#stage-3--instruction-format-dataset-build)
- [Stage 4 — Pre-Fine-Tuning Baseline](#stage-4--pre-fine-tuning-baseline)
- [Stage 5 — Training Matrix](#stage-5--training-matrix)
- [Stage 6 — Four-Tier Evaluation Harness](#stage-6--four-tier-evaluation-harness)
- [Stage 7 — Regression / Forgetting Analysis](#stage-7--regression--forgetting-analysis)
- [Stage 8 — Quantization Matrix](#stage-8--quantization-matrix)
- [Stage 9 — Air-Gapped Serving](#stage-9--air-gapped-serving)
- [Stage 10 — CI/CD & Regression Gate](#stage-10--cicd--regression-gate)
- [Stage 11 — Documentation & Interview Package](#stage-11--documentation--interview-package)
- [Testing](#testing)
- [Makefile Targets](#makefile-targets)
- [Contributing](#contributing)
- [License](#license)

## Status

- ✅ **Stage 0** — environment & repo skeleton.
- ✅ **Stage 1** — data collection.
  - ✅ **Run end-to-end on 2026-08-16** via `scripts/run_stage1_real.py` (deterministic mock NVD client + real bundled Semgrep rules, CVEfixes v1.0.8 schema). Results: 992 raw pairs processed, 621 kept after token-budget filter (404 train / 114 val / 103 test), 371 dropped — see `output/stage3/manifest.json` and [Stage 1 notes](#stage-1-notes).
- ✅ **Stage 2** — cleaning, dedup, leakage-safe split, contamination check.
  - ✅ **Run end-to-end** — Stage 1 output was deduped, split, and token-budget filtered, producing the 404/114/103 instruction-format dataset in `output/stage3/`.
- ✅ **Stage 3** — instruction-format dataset build. Prompt template (system + task prompt with vulnerable code + static findings), injectable token counter (Qwen tokenizer with heuristic fallback), token-budget enforcement, unified-diff patch generation, and JSONL split writers are implemented and unit-tested + integration-tested.
- ✅ **Stage 4** — pre-fine-tuning baseline.
  - ✅ **Real baseline run on 2026-08-16** (zero-shot evaluation of Qwen2.5-Coder-1.5B-Instruct on the 59-sample gold-eval set). Zero-shot and few-shot evaluation with CWE Macro-F1, severity accuracy, hallucination rate, and patch coverage metrics. Fully implemented and tested.
- ✅ **Stage 5** — training matrix.
  - ✅ **Real GPU QLoRA training run on 2026-08-17** (1.5B, LoRA r=8, 4-bit NF4, 3 epochs, 404 train samples, peak VRAM 9.26 GB on RTX 4060 Laptop GPU — see `scripts/run_gpu_training.py`). CPU-compatible training also available via `scripts/run_cpu_training.py`. All modes support `--dry-run`.
- ✅ **Stage 6** — four-tier evaluation harness.
  - ✅ **Real eval run on 2026-08-16** (Tier 3 uses Docker sandbox; 59 gold samples, 12 model predictions). Deterministic (Tier 1) → static+embedding (Tier 2) → exec sandbox (Tier 3) → LLM-judge (Tier 4).
- ✅ **Stage 7** — regression / forgetting analysis. The execution layer is real: `LocalCodeTestRunner` spawns an actual `python -m pytest` subprocess per task and the committed `output/stage7/regression_report.json` contains genuine pytest stdout per task (`platform win32 ... 1 passed in 0.03s`, etc.) — the `"mock test result for task gc_N"` string only ever lives inside `MockCodeTestRunner`, a deliberate test double used in unit tests, not in the real pipeline. The model backend was also switched to the real path: `run_stage7_only.py` drives `QwenBackend` against the actual Stage 5 fine-tuned checkpoint (LoRA adapter merged on top of the base model), producing a genuinely measured (not simulated-solution) forgetting delta. A real bug blocking the real-backend path was found and fixed on 2026-08-26: `QwenBackend._load()` passed `framework="pt"` to `transformers.pipeline()`, which is not a valid keyword in `transformers` 5.x and raises `ValueError` when forwarded to the model's `generate()` method — the parameter was removed to restore the real-backend path. `output/stage7/manifest.json` now records `"script": "scripts/run_stage7_only.py"` as proof the real (non-simulated) backend was used.
- ✅ **Stage 8** — quantization matrix.
  - GPTQ / AWQ / GGUF with quality-vs-VRAM trade-off scoring.
  - ✅ **Real run on 2026-08-20** (GPTQ 4-bit on Qwen2.5-Coder-1.5B LoRA checkpoint). Mock and dry-run modes supported.
- ✅ **Stage 9** — air-gapped serving. llama.cpp / Ollama / mock backends behind a FastAPI service + Typer CLI (serve / analyze / batch / dry-run modes).
- ✅ **Stage 10** — CI/CD & regression gate.
  - GitHub Actions workflow with ruff, Bandit, pytest, eval gate (Stage 4→6→7→10 mock pipeline), Gitleaks (secret scanning), and Trivy (vuln + config scanning).
- ✅ **Stage 11** — documentation & interview package.
  - `Stage11Generator.load_artifacts()` is wired to the real Stage 4/5/6/7 output files (`ensure_deliverables()` calls it before rendering) and this is now confirmed working: `docs/training_report.md` lists **2 real training runs** (`sft_qlora` and `dpo`, both from the 2026-08-17 GPU run, with real loss/VRAM/time figures) instead of the old *"No real training runs have been executed yet"* placeholder. Model card (`docs/model_card.md`), training report, and demo script (`docs/demo.py`) are all generated and validated via the `stage11` CLI subcommand.

> **Test suite (verified 2026-08-26):** **1,641 tests** — 1,464 unit tests
> across 54 files in `tests/unit/`, plus 177 integration tests across 12
> files in `tests/integration/`. Running unit + integration together:
> **1,640 passed**, 1 failed — the single failure
> (`test_record_peak_memory_noop_without_gpu`) only reproduces in an
> environment without the `[ml]` extras (no `torch` installed) and is not a
> real bug; it passes wherever `pip install -e ".[dev,data,ml]"` was run,
> as CI does. `ruff check .` is clean (0 issues). `bandit -r app -q` is
> clean (0 issues) — `bandit_report.json` in the repo root was regenerated
> with the exact CI command/scope on 2026-08-26 and now matches (previously
> it held a stale, wider-scope scan with 260 `B101 assert_used` findings
> from `tests/`, which aren't part of the CI security-scan scope). Coverage
> on `tests/unit` alone (no `[ml]` extras): **98%** (5,945 statements, 121
> missed) — re-run `pytest --cov=app --cov-report=term-missing` with the
> `[ml]` extras installed for the full-coverage figure. Everything above
> runs in mock/dry-run mode — no GPU, Docker, or network required; the
> Stage 5/7/8 *real*-mode runs referenced elsewhere in this README were done
> separately, on the author's own GPU machine.

### Stage 1 Notes

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
  `app/data/collectors/rules/python.yaml` and `app/data/collectors/rules/javascript.yaml`
  ship a small, version-controlled rule pack scoped to exactly the CWE
  classes above, including a taint-mode rule for the realistic "build query
  in a variable, then execute()" pattern, not just inline concatenation.
- **CVEfixes.db is not included.** Download it from
  [Zenodo (secureIT-project/CVEfixes v1.0.8)](https://zenodo.org/records/13118970)
  and pass its path to the CLI: `python -m app.data.collectors.cli collect --db-path ./CVEfixes.db`.
- **Stage 1 was run end-to-end on 2026-08-16** via `scripts/run_stage1_real.py`
  against a local copy of `CVEfixes.db` (v1.0.8 schema). Results: 992 raw pairs
  processed, 621 kept after the 4096-token budget filter (404 train / 114 val /
  103 test), 371 dropped for exceeding the token budget. See `output/stage3/manifest.json`.
  The NVD enrichment uses a **deterministic mock client** (`_MockNvdClient` in
  `scripts/run_stage1_real.py`) that derives severity from CVE year — this avoids
  NVD API rate limits while keeping the run reproducible. Real Semgrep rules
  (bundled in `app/data/collectors/rules/`) were used for static findings.

## Why this project

- A real training loop (SFT + LoRA/QLoRA + DPO), not just prompting a base model.
- Leakage-safe data discipline: repo-based splits, embedding dedup, n-gram
  contamination checks on the gold-eval set.
- **Exec-based evaluation**: proposed patches are actually applied and run
  against the project's test suite in a sandboxed subprocess — LLM-judge
  scoring is used only for explanation quality, not pass/fail.
- Explicit quantization/deployment constraint: the final checkpoint has to
  run air-gapped, on consumer hardware.
- CI/CD and security scanning from day one (pytest, ruff, Bandit, Gitleaks, Trivy).

## Out of Scope

This project does **not** do the following. Be clear about these boundaries:

- **General-purpose vulnerability scanner.** Targets only the 6 CWE classes in
  scope (CWE-89, CWE-79, CWE-22, CWE-78, CWE-190, CWE-502). Does not claim to
  detect logic bugs, configuration issues, or CWE classes outside the listed scope.
- **Real-time scanning.** The serving layer (Stage 9) is for interactive /
  batch vulnerability analysis of isolated code snippets, not for continuous
  monitoring of repositories in CI.
- **Supply-chain security.** Does not audit third-party packages or perform
  dependency-graph analysis. Use `pip-audit` / `Safety` for that.
- **Network-based scanning.** No network port scanning, no HTTP fuzzing, no
  live system exploitation. All evaluation is offline against curated CVE data.
- **Legal / compliance assessment.** The model classifies CWE IDs and proposes
  patches from training data patterns — it does not provide legal opinions on
  liability, regulatory compliance, or licensing.
- **Production incident response.** This is a research / training artifact,
  not a SOC tool. Do not use it as a sole decision-maker for production
  security alerts.

## Architecture

```
STAGE 0  Environment & repo skeleton         ✅ Done
STAGE 1  Data collection & labeling          ✅ Done (CVEfixes/BigVul/OSV → VulnSample, Semgrep rules bundled)
STAGE 2  Cleaning, dedup, leakage-safe split, contamination check   ✅ Done
STAGE 3  Instruction-format dataset build    ✅ Done (prompt template, token budget, JSONL splits)
STAGE 4  Pre-fine-tuning baseline            ✅ Done (zero-shot / few-shot base model on gold-eval)
STAGE 5  Training matrix                     ✅ Done (SFT full/QLoRA · LoRA rank sweep · DPO)
STAGE 6  Four-tier evaluation harness        ✅ Done (deterministic → embedding/static → exec → LLM-judge)
STAGE 7  Regression / forgetting analysis    ✅ Done (general code-capability delta, before/after)
STAGE 8  Quantization matrix                 ✅ Done (GPTQ / AWQ / GGUF, quality vs. speed/VRAM)
STAGE 9  Air-gapped serving                  ✅ Done (llama.cpp/Ollama/mock, FastAPI + CLI)
STAGE 10 CI/CD & regression gate             ✅ Done (ruff/Bandit/pytest + eval gate + Gitleaks + Trivy)
STAGE 11 Documentation & interview package   ✅ Done (model card, training report, demo script)
```

Cross-cutting infrastructure: **PostgreSQL** for experiment/metric state,
**Redis + Celery** for long-running jobs (training, quantization), **W&B**
for loss curves and eval tracking, **MinIO/S3** for model checkpoint and
dataset artifact storage.

## Repo Layout

```
vuln-triage-harness/
├── app/
│   ├── schemas/          # Pydantic v2 data contracts (all stages)
│   │   ├── vuln.py            # VulnSample
│   │   ├── dataset.py         # InstructionExample
│   │   ├── prediction_eval.py # ModelPrediction, EvalMetrics, RegressionReport
│   │   ├── training.py        # TrainingResult, SweepReport
│   │   ├── quantization.py    # QuantReport, QuantResult
│   │   ├── serving.py         # ServeRequest, ServeResponse, BatchServeResponse
│   │   ├── ci.py              # GateStatus, GateCheck, RegressionGateResult
│   │   ├── documentation.py   # ModelCardData, TrainingReportData, etc.
│   │   └── __init__.py
│   ├── data/
│   │   ├── collectors/   # CVEfixes/BigVul/NVD/OSV downloaders + Semgrep      (Stage 1)
│   │   │   └── rules/    #   └── bundled Semgrep rule packs (python.yaml, javascript.yaml)
│   │   ├── cleaning/     # dedup, leakage-safe split, contamination check     (Stage 2)
│   │   └── formatting/   # instruction-format dataset builder, token counter   (Stage 3)
│   ├── training/         # sft/qlora/lora-sweep/dpo trainers, CLI              (Stage 5)
│   ├── evaluation/       # tier1→tier4 evaluators, baseline, regression        (Stage 4-6-7)
│   ├── quantization/     # GPTQ/AWQ/GGUF quantizers, matrix runner, CLI        (Stage 8)
│   ├── serving/          # FastAPI app, Typer CLI, backends, config           (Stage 9)
│   │   ├── Dockerfile.gpu    # Multi-stage CUDA image (GGML_CUDA=on) for GPU serving (Stage 9)
│   ├── storage/          # Postgres models, MinIO client
│   ├── ci/               # regression gate, security scanner parsers            (Stage 10)
│   └── stage11/          # documentation generator (model card, report, demo)   (Stage 11)
├── docs/                 # Generated deliverables (model card, training report, demo.py)
│   ├── model_card.md
│   ├── training_report.md
│   ├── demo.py
├── eval/
│   └── gold_set/         # 59 manually verified gold-eval examples (6 CWE classes)
├── sandbox/              # Docker sandbox for exec-based eval (Stage 6): Dockerfile + Python 3.11 image
├── tests/unit/           # 50 unit test files covering all 11 stages
├── tests/integration/    # One file per stage — end-to-end pipeline tests in mock mode
├── .github/workflows/ci.yml    # ruff, Bandit, pytest, eval-gate, Gitleaks, Trivy
├── .gitleaks.toml      # Gitleaks config with allowlist for test fixtures
├── docker-compose.yml    # Postgres + Redis + MinIO + GPU serving profile
├── Makefile              # install, test, lint, security, up, down
├── scripts/              # Real-mode runner scripts (all support --dry-run)
│   ├── convert_to_gguf.py  # HF safetensors → GGUF (BFloat16-aware, standalone `gguf` package)
│   ├── convert_cvefixes.py  # CVEfixes full schema → reduced schema for Stage 1
│   ├── expand_gold_set.py  # Expand gold-eval set with LLM-generated variants
│   ├── generate_docs.py    # Stage 11 standalone doc generator
│   ├── run_cpu_training.py  # CPU-only training (Stage 5)
│   ├── run_gpu_training.py  # GPU QLoRA training (Stage 5)
│   ├── run_stage1_real.py   # Real Stage 1 data collection
│   ├── run_stage7_only.py  # Regression analysis from saved checkpoint (Stage 7)
│   ├── run_stage8_real.py  # Real GPTQ quantization on GPU (Stage 8)
│   ├── run_stage9_serve.py  # Real GPU serving with llama-server.exe (Stage 9)
│   ├── run_stage10_real.py  # CI regression gate on real artifacts (Stage 10)
│   └── run_evaluation.py  # Run evaluation with configurable backends
├── requirements-lock.txt    # Pinned transitive dependencies for reproducibility
└── pyproject.toml           # Project metadata, dependencies, ruff/pytest/coverage config
```

## Tech Stack

| Layer | Choice |
|---|---|
| Base model | Qwen2.5-Coder-7B-Instruct (primary), 1.5B (fast iteration) |
| PEFT | `peft` (LoRA/QLoRA), `bitsandbytes` 4-bit NF4 |
| Preference tuning | `trl` `DPOTrainer` |
| Data source | CVEfixes / BigVul (CVE→commit→diff mapped), NVD API, OSV.dev |
| Static signal | Semgrep |
| Exec eval | Docker sandbox (`sandbox/Dockerfile`) + subprocess fallback |
| Dedup | `sentence-transformers` code-embedding model |
| Experiment tracking | Weights & Biases |
| Quantization | AutoGPTQ, AutoAWQ, llama.cpp (GGUF — Q4_K, Q8_0, F32) |
| Serving | llama.cpp (`llama-cpp-python` w/ CUDA), `llama-server.exe` (HTTP), Ollama, mock |
| GPU serving | Docker + NVIDIA Container Toolkit (`--gpus all`, CUDA 12.4.1) |
| Orchestration | Celery + Redis |
| State/metrics DB | PostgreSQL |
| CI/CD | GitHub Actions — pytest, ruff, Bandit, Gitleaks, Trivy |

> **Model size note:** Qwen2.5-Coder-7B-Instruct is the **designed-for** model.
> CPU-only validation runs (no CUDA GPU) use the 1.5B-Instruct variant — the
> same inference code, smaller checkpoint. The real GPU QLoRA training run on
> RTX 4060 Laptop GPU (8.19 GB VRAM) used the 1.5B-Instruct variant with 4-bit NF4 quantization
> (`scripts/run_gpu_training.py`). The 7B model can be used via the same CLI
> with `--base-model Qwen/Qwen2.5-Coder-7B-Instruct` if more VRAM is available.

## Quickstart (Stage 0)

```bash
# 1. Install dependencies (ML extras are optional — not needed for Stage 0)
pip install -e ".[dev]"

# For reproducible installs across environments, pin transitive deps:
#   pip install -e ".[dev,data,ml]" -c requirements-lock.txt

# 2. Bring up Postgres + Redis + MinIO
docker compose up -d

# 3. Run the test suite (1464 unit tests, 98%+ coverage, no GPU/network needed)
pytest tests/unit -v --cov=app --cov-report=term-missing
```

## Evaluation Metrics (Defined Up Front, Measured at Every Checkpoint)

| Metric | Definition |
|---|---|
| CWE Macro-F1 | Per-class F1 averaged across CWE classes (accuracy is misleading on imbalanced data) |
| Exec Pass Rate | Share of predictions where `tests_pass_after_patch = True` |
| Hallucination Rate | Share of predictions with a fabricated CWE ID or a reference to nonexistent code |
| Cost per Accepted Patch | (inference $ + amortized training $) / patches passing exec-eval |
| Forgetting Delta | general-capability-score(tuned) − general-capability-score(base) |

---

## Stage 1 — Data Collection

Stage 1 collects CVE-patch pairs from CVEfixes (with BigVul/OSV as fallbacks),
enriches them with NVD metadata, runs bundled Semgrep rules to extract static
findings, and persists `VulnSample` records to Postgres + MinIO.

**CLI:**

```bash
# Collect CVE-patch pairs from a local CVEfixes.db
python -m app.data.collectors.cli collect --db-path ./CVEfixes.db

# Run the full pipeline (download → load → NVD enrich → Semgrep → persist)
python -m app.data.collectors.cli collect \
  --db-path ./CVEfixes.db \
  --nvd-api-key $NVD_API_KEY \
  --output-format jsonl \
  --output-dir ./output/stage1
```

### Stage 1 Modules

| Module | Responsibility |
|---|---|
| `cwe_scope.py` | `CWE_SCOPE` constant (6 classes), helper functions |
| `cvefixes_loader.py` | `CveFixesLoader` — loads CVEfixes SQLite v1.0.8 schema |
| `cvefixes_reduced_loader.py` | `ReducedCveFixesLoader` — loads reduced-schema SQLite (smaller footprint) |
| `nvd_client.py` | `NvdClient` — NVD API enrichment client |
| `semgrep_runner.py` | `run_semgrep()` — runs bundled Semgrep rules on a code snippet |
| `rules/` | Bundled Semgrep rule packs (python.yaml, javascript.yaml) |
| `pipeline.py` | Orchestrates collection → enrichment → scanning → persistence |
| `cli.py` | Stage 1 CLI (`collect` subcommand) |

### Stage 1 Notes

- **CWE scope**: `CWE-89` (SQLi), `CWE-79` (XSS), `CWE-22` (path traversal),
  `CWE-78` (command injection), `CWE-190` (integer overflow), `CWE-502`
  (unsafe deserialization).
- **Semgrep rules are bundled** — see the top-level [Stage 1 Notes](#stage-1-notes) for rationale.
- **CVEfixes.db is not included** — download from
  [Zenodo (secureIT-project/CVEfixes v1.0.8)](https://zenodo.org/records/13118970).
- **✅ Run end-to-end on 2026-08-16** — see [Stage 1 Notes](#stage-1-notes) for
  details. Results: 621 instruction examples built (404/114/103 train/val/test, 371 dropped).
  Uses `scripts/run_stage1_real.py` with a deterministic mock NVD client + real Semgrep.

---

## Stage 2 — Cleaning, Dedup, Leakage-Safe Split

After Stage 1 has populated Postgres + MinIO with `VulnSample` records, Stage 2
performs embedding-backed near-duplicate removal, a repo-based leakage-safe
split with CWE stratification, and an n-gram contamination check.

```bash
# 1. Install with ML extras (for sentence-transformers embeddings)
pip install -e ".[dev,ml]"

# 2. Run the full Stage 2 pipeline (dedup → split → contamination check)
python -m app.data.cleaning.cli clean --verbose

# 3. Dry-run to preview the plan without writing to Postgres
python -m app.data.cleaning.cli clean --dry-run

# 4. Export to HuggingFace Hub (requires HF_TOKEN)
python -m app.data.cleaning.cli export --repo-id vuln-triage/vuln-triage-dataset

# 5. Check gold-eval contamination against the train set
python -m app.data.cleaning.cli check-contamination --gold-eval eval/gold_set/gold.jsonl
```

### Stage 2 Modules

| Module | Responsibility |
|---|---|
| `embeddings.py` | HuggingFace `sentence-transformers` backend (`jina-embeddings-v2-base-code`) |
| `dedup.py` | Near-duplicate removal via cosine similarity on code embeddings |
| `split.py` | Repo-based leakage-safe split with CWE stratification and class balance |
| `contamination.py` | N-gram (5-gram) contamination checker between train and eval sets |
| `hf_dataset.py` | HuggingFace `datasets` integration (export to Hub, load from disk/Hub) |
| `pipeline.py` | Orchestrates load → dedup → split → contamination → persist |
| `cli.py` | Stage 2 CLI (`clean`, `plan`, `export`, `check-contamination`) |

### Stage 2 Notes

- **Leakage-safe split**: repos are grouped and assigned to train/val/test
  so that no repository appears in more than one split.
- **Class balance**: within each CWE class, repos are distributed proportionally
  across splits, so CWE distribution is preserved.
- **Contamination gate**: the eval/test set must have < 5% 5-gram overlap with
  the training set. Checked automatically in the pipeline and fails CI
  (Stage 10) if exceeded.
- **HuggingFace note**: the default embedding model
  (`jina-embeddings-v2-base-code`) requires `trust_remote_code=True`. With
  `transformers>=5.0`, this works out of the box. For models without custom
  code, set `EmbeddingBackend(model_name="intfloat/multilingual-e5-base",
  trust_remote_code=False)`.

---

## Stage 3 — Instruction-Format Dataset Build

After Stage 2 has produced split `VulnSample` records, Stage 3 builds
instruction-format JSONL with prompt templates, token-budget enforcement,
and unified-diff patch generation.

```bash
# 1. Build instruction-format JSONL from Stage 2 output (Postgres/MinIO)
python -m app.data.formatting.cli build --output-dir ./output/stage3

# 2. Build from a local HF datasets directory (produced by Stage 2's export)
python -m app.data.formatting.cli build --hf-path ./output/stage2_dataset

# 3. Dry-run to preview counts without writing files
python -m app.data.formatting.cli build --dry-run

# 4. Inspect the output
python -m app.data.formatting.cli stats ./output/stage3
python -m app.data.formatting.cli inspect ./output/stage3/train.jsonl --index 0
```

### Stage 3 Modules

| Module | Responsibility |
|---|---|
| `template.py` | Prompt template (system + task prompt), static-finding formatter, unified-diff patch generator |
| `tokenizer.py` | Injectable token counter (Qwen tokenizer with heuristic fallback for air-gapped/CI) |
| `builder.py` | Builds `InstructionExample` records from `VulnSample` with token-budget enforcement |
| `pipeline.py` | Orchestrates load → build → JSONL write, with manifest output |
| `cli.py` | Stage 3 CLI (`build`, `stats`, `inspect`) |

### Stage 3 Notes

- **Token budget**: samples whose estimated prompt + target token count exceeds
  `max_tokens` (default 4096) are dropped from the output.
- **Tokenizer flexibility**: the `TokenCounter` uses the Qwen tokenizer from
  `transformers` when available; falls back to a character-based heuristic.
  Tests can inject a mock tokenizer via `TokenCounter(tokenizer=...)`.
- **Patch diffs**: unified diffs are generated with Python's `difflib.unified_diff`
  — no external `git` dependency. Patches use `a/` and `b/` path prefixes.
- **No fixed_code**: samples without a `fixed_code` field still get an
  `InstructionExample` — the `target_patch_diff` is set to `None`.

---

## Stage 4 — Pre-Fine-Tuning Baseline

Stage 4 evaluates the **base** (pre-fine-tuning) model on the gold-eval set
to establish a "before" baseline. Supports zero-shot and few-shot prompting.
The default model is `Qwen/Qwen2.5-Coder-7B-Instruct`. For fast iteration or
testing without model downloads, use `--mock` (deterministic `MockBackend`).

```bash
# 1. Zero-shot baseline on the gold-eval set (uses real Qwen model)
python -m app.evaluation.cli baseline \
  --gold-eval eval/gold_set/gold.jsonl \
  --strategy zero-shot \
  --output-dir ./output/stage4

# 2. Few-shot baseline (3-shot, uses nearest-neighbor from training set)
python -m app.evaluation.cli baseline \
  --gold-eval eval/gold_set/gold.jsonl \
  --strategy few-shot \
  --train-jsonl ./output/stage3/train.jsonl \
  --output-dir ./output/stage4_fewshot

# 3. Mock mode — deterministic, no model download needed
python -m app.evaluation.cli baseline \
  --gold-eval eval/gold_set/gold.jsonl \
  --strategy zero-shot \
  --mock \
  --output-dir ./output/stage4_mock
```

### Output Files

`output/stage4/` contains:

| File | Contents |
|---|---|
| `predictions.jsonl` | One `ModelPrediction` per gold sample |
| `metrics.json` | Aggregate `EvalMetrics` (CWE Macro-F1, exec pass rate, halluc rate, patch coverage) |

### Stage 4 Modules

| Module | Responsibility |
|---|---|
| `baseline.py` | `BaselineConfig`, `ZeroShotBackend`, `FewShotBackend`, `MockBackend` |
| `backends.py` | `ModelBackend` Protocol, `QwenBackend` (HuggingFace transformers), `MockBackend` |
| `metrics.py` | `compute_tier4_metrics()`, `compute_tier1_tier2_metrics()`, aggregation helpers |
| `parser.py` | `parse_model_output()`, `parse_json_response()` — extracts CWE/severity/patch from raw LLM output |
| `prompt.py` | `build_zero_shot_prompt()`, `build_few_shot_prompt()` — prompt templates |
| `cli.py` | `baseline` Typer subcommand |

---

## Stage 5 — Training Matrix

Stage 5 implements three training modes, all with `--dry-run` support:

1. **SFT (Supervised Fine-Tuning)** — full-parameter or QLoRA (4-bit NF4)
2. **LoRA Rank Sweep** — trains across ranks [8, 16, 32, 64, 128], selects best by val loss
3. **DPO (Direct Preference Optimization)** — preference-aligns the SFT checkpoint

```bash
# QLoRA (4-bit NF4) — fits 8GB VRAM, recommended starting point
python -m app.training.cli sft \
  --train-jsonl ./output/stage3/train.jsonl \
  --val-jsonl   ./output/stage3/val.jsonl \
  --output-dir  ./output/stage5/sft_qlora \
  --lora-r 64 --lora-alpha 16 --lora-dropout 0.05 \
  --learning-rate 2e-5 --epochs 3 \
  --grad-accum 8

# Full-parameter SFT (no quantization) — needs >=16 GB VRAM
python -m app.training.cli sft \
  --train-jsonl ./output/stage3/train.jsonl \
  --val-jsonl   ./output/stage3/val.jsonl \
  --no-4bit \
  --output-dir ./output/stage5/sft_full

# Dry-run: estimate steps/VRAM without a GPU
python -m app.training.cli sft \
  --train-jsonl ./output/stage3/train.jsonl \
  --dry-run
```

### LoRA Rank Sweep

```bash
# Full 5-rank sweep (dry-run — no GPU needed)
python -m app.training.cli lora-sweep \
  --train-jsonl ./output/stage3/train.jsonl \
  --val-jsonl   ./output/stage3/val.jsonl \
  --dry-run \
  --no-persist

# Real training (remove --dry-run, ensure GPU is available)
python -m app.training.cli lora-sweep \
  --train-jsonl ./output/stage3/train.jsonl \
  --val-jsonl   ./output/stage3/val.jsonl \
  --ranks 8,16,32,64,128
```

### DPO Preference Alignment

```bash
# DPO from an SFT checkpoint (recommended)
python -m app.training.cli dpo \
  --train-jsonl    ./output/stage3/train.jsonl \
  --sft-checkpoint ./output/stage5/sft_qlora/final_checkpoint \
  --beta 0.1 \
  --output-dir ./output/stage5/dpo

# DPO from the base model (not recommended — no SFT warmup)
python -m app.training.cli dpo \
  --train-jsonl ./output/stage3/train.jsonl \
  --beta 0.1

# Dry-run
python -m app.training.cli dpo --train-jsonl ./output/stage3/train.jsonl --dry-run
```

### Inspecting Runs

Training metadata is persisted to PostgreSQL (when available):

```bash
# List all training runs
python -m app.training.cli list-runs
python -m app.training.cli list-runs --limit 10 --method sft_qlora --status completed

# Inspect a specific run
python -m app.training.cli inspect --run-id dpo_20260817_202000_abc12345
```

### Stage 5 Modules

| Module | Responsibility |
|---|---|
| `config.py` | `TrainingMethod` enum, `SFTConfig`, `DPOConfig`, `SweepConfig` dataclasses |
| `data.py` | `JsonlDataLoader` (injectable), `load_examples()`, `compute_stats()`, `make_hf_dataset()` (lazy `datasets` import) |
| `callbacks.py` | `TrainingCallback` Protocol, `WandbCallback` (mock mode), `CheckpointCallback` (MinIO upload), `ProgressCallback`, `ResourceTracker` (peak VRAM) |
| `experiment.py` | `persist_training_run()`, `load_training_run()`, `list_training_runs()` (PostgreSQL via SQLAlchemy), `generate_run_id()` |
| `trainer_sft.py` | `run_sft()` (full + QLoRA), `estimate_training_steps()` (pure arithmetic), `TrainingUnavailableError` |
| `trainer_dpo.py` | `run_dpo()` with TRL `DPOTrainer`, `estimate_dpo_steps()`, `build_preference_pairs()` |
| `sweep.py` | `run_lora_sweep()` — orchestrates multiple `run_sft` calls across ranks, `SweepReport` summary |
| `cli.py` | Typer CLI: `sft`, `lora-sweep`, `dpo`, `list-runs`, `inspect` subcommands |

### Stage 5 Notes

- **No GPU needed for development.** All training modes support `--dry-run`.
  Real training is gated behind `_check_can_train()`, which raises
  `TrainingUnavailableError` if torch/transformers/trl are missing or no CUDA GPU.
- **Lazy ML imports.** Heavy dependencies (`torch`, `transformers`, `peft`,
  `bitsandbytes`, `trl`, `datasets`, `wandb`) are imported inside functions.
- **Injectable backends for testing.** The `loader` parameter on `run_sft`,
  `run_dpo`, and `run_lora_sweep` accepts any object implementing the
  `DataLoadable` Protocol, so tests can inject pre-built `InstructionExample`
  lists without touching the filesystem.
- **QLoRA defaults.** By default, SFT uses 4-bit NF4 quantization via
  `bitsandbytes` (with `bnb_4bit_use_double_quant=True`) so a 7B model fits in
  8 GB VRAM.
- **LoRA rank range.** The sweep tests ranks `[8, 16, 32, 64, 128]`, bracketing
  the "useful parameter-efficient range" from the QLoRA paper
  (Dettmers et al., 2023, arXiv:2305.14168).
- **PostgreSQL tracking.** When `persist=True` (default), each completed
  run is written to the `training_runs` table via SQLAlchemy.
- **Experiment tracking via W&B.** `WandbCallback` logs loss curves in real
  mode; in mock mode it stores calls in memory.

---

## Stage 6 — Four-Tier Evaluation Harness

Stage 6 implements the **four-tier evaluation harness** that validates model
predictions across multiple dimensions:

```
         ┌──────────────────────────────────────────────────────────┐
         │  Four-tier evaluation harness (Stage 6)                   │
         │                                                          │
  Gold   │  Tier 1: deterministic regex classifier (CWE only)      │
  Eval → │  → Tier 2: static Semgrep findings + embedding          │
  Sample │  → Tier 3: exec — apply patch, run tests in sandbox    │
  +      │  → Tier 4: LLM-judge — explanation quality/minimality │
  Model  └──────────────────────────────────────────────────────────┘
```

### Tiers

| Tier | Method | Backend | What it measures |
|---|---|---|---|
| 1 — deterministic | Regex-based CWE classifier | `tier1_deterministic.py` | Baseline CWE Macro-F1 (upper bound) |
| 2 — static + embedding | Semgrep findings + cosine similarity | `tier2_embedding_static.py` | Static signal quality |
| 3 — exec sandbox | Apply patch, run tests | `tier3_exec.py` | Exec Pass Rate (patches that actually work) |
| 4 — LLM judge | LLM scores explanation quality | `tier4_llm_judge.py` | Hallucination rate, explanation minimality |

### Running Stage 6

```bash
# Mock mode — all tiers, deterministic, no GPU or Docker needed
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions ./output/stage4/predictions.jsonl \
  --sandbox-mode mock \
  --skip-tier4 \
  --output-dir ./output/stage6

# Real exec sandbox (requires Docker — see sandbox/)
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions ./output/stage4/predictions.jsonl \
  --sandbox-mode docker \
  --output-dir ./output/stage6

# Tier 4 LLM-judge (requires Ollama or similar)
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions ./output/stage4/predictions.jsonl \
  --sandbox-mode docker \
  --run-tier4 \
  --output-dir ./output/stage6
```

### Stage 6 Modules

| Module | Responsibility |
|---|---|
| `tier1_deterministic.py` | Regex-based CWE classifier with static pattern matching |
| `tier2_embedding_static.py` | Embedding similarity (sentence-transformers) + static findings |
| `tier3_exec.py` | Docker sandbox executor (applies patch, runs pytest) + mock executor |
| `tier4_llm_judge.py` | LLM-judge for explanation quality and hallucination detection |
| `backends.py` | `ModelBackend` Protocol, `QwenBackend`, `MockBackend`, `Tier4JudgeBackend` |
| `baseline.py` | `ZeroShotBackend`, `FewShotBackend` for Stage 4 baselines |
| `metrics.py` | `compute_tier4_metrics()`, `EvalMetrics`, `compute_hallucination_rate()` |
| `parser.py` | `parse_model_output()` — extracts CWE, severity, explanation, patch |
| `prompt.py` | Prompt templates for zero-shot, few-shot, and exec-sandbox |
| `runner.py` | `EvaluationRunner` — orchestrates all four tiers, `EvaluationConfig` |
| `cli.py` | `stage6` Typer subcommand |

---

## Stage 7 — Regression / Forgetting Analysis

Stage 7 measures whether fine-tuning the model on vulnerability classification
causes "forgetting" of general code-capability. It runs a set of HumanEval-style
tasks on both the base and tuned models, then computes a **forgetting delta**:

```
forgetting_delta = accuracy(tuned) − accuracy(base)
```

A negative delta means the model got worse on general code tasks after
fine-tuning.

```bash
# Mock mode — deterministic, no GPU needed
python -m app.evaluation.cli stage7 \
  --mock \
  --base-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-model "ci-checkpoint" \
  --output-dir ./output/stage7

# Real mode (requires model access; GPU preferred, CPU works with --max-new-tokens)
python scripts/run_stage7_only.py \
  --base-model "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --checkpoint ./output/stage5/qwen_lora_gpu/final_checkpoint \
  --stage6-report ./output/stage6/eval_report.json \
  --output-dir ./output/stage7 \
  --max-new-tokens 256  # optional: lower for faster CPU inference
```

### Stage 7 Modules

| Module | Responsibility |
|---|---|
| `general_capability.py` | 12 HumanEval-style tasks, `GeneralCapabilityTask`, `CodeTestRunner` Protocol, `LocalCodeTestRunner`, `MockCodeTestRunner`, `run_regression_analysis()` |
| `cli.py` | `stage7` Typer subcommand with `--mock`, `--base-model`, `--tuned-model`, `--timeout`, `--output-dir`, `--verbose` |
| `ci/gate.py` | `load_stage7_report()` — loads `regression_report.json`, checks `forgetting_delta` |
| `schemas/prediction_eval.py` | `GeneralCapabilityMetrics`, `RegressionReport`, `RegressionSummary` |
| `scripts/run_stage7_only.py` | Real-mode runner for Stage 7 |

### Stage 7 Notes

- **12 default tasks**: factorial, palindrome, fibonacci, binary search,
  two-sum, vowel counting, integer reversal, anagram, longest common prefix,
  valid parentheses, remove duplicates, and max subarray sum — all pure-Python.
- **No Docker needed for real eval** — `LocalCodeTestRunner` spawns isolated
  `python -m pytest` subprocesses. This part is genuinely real — the
  committed `output/stage7/regression_report.json` contains actual pytest
  stdout per task, not a canned string.
- **Forgetting delta = `tuned_acc − base_acc`**. Negative = forgetting.
  Feeds into `RegressionSummary`, consumed by the Stage 10 regression gate.
- **Real model backend.** `run_stage7_only.py` drives the real `QwenBackend`
  (Qwen2.5-Coder-1.5B-Instruct base + LoRA adapter merged on top) against the
  Stage 5 `qwen_lora_gpu` checkpoint — `output/stage7/manifest.json` records
  `"script": "scripts/run_stage7_only.py"` as proof. A `--max-new-tokens` flag
  is available for CPU inference (e.g. `--max-new-tokens 256`).
  `run_stage7_local.py` + `FastSolutionBackend` remain available as a fast,
  no-GPU/no-download fallback for local iteration.

---

## Stage 8 — Quantization Matrix

Stage 8 quantizes a trained Stage 5 checkpoint with GPTQ, AWQ, and GGUF, then
selects the best configuration by a quality-vs-size-vs-speed score.

```bash
# 1. Mock mode — deterministic, no GPU, no ML deps (fast)
python -m app.evaluation.cli stage8 \
  --source-checkpoint ./output/stage5/sft_qlora \
  --mock \
  --output-dir ./output/stage8

# 2. Dry-run mode — heuristic estimates, no actual quantization
python -m app.evaluation.cli stage8 \
  --source-checkpoint ./output/stage5/sft_qlora \
  --dry-run \
  --methods gptq,awq,gguf \
  --bits 2,3,4 \
  --target-vram-gb 8.0     # filter to configs that fit in 8 GB VRAM

# 3. Real quantization via standalone script
python scripts/run_stage8_real.py \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --checkpoint ./output/stage5/qwen_lora_gpu/final_checkpoint \
  --output-dir ./output/stage8 \
  --methods gptq \
  --bits 4 \
  --calib-dataset output/stage3/train.jsonl \
  --skip-eval
```

### Stage 8 Real-Run Results (2026-08-20)

GPTQ 4-bit quantization of the Stage 5 LoRA checkpoint (Qwen2.5-Coder-1.5B
base + LoRA adapter merged) on an RTX 4060 Laptop GPU (8.19 GB VRAM):

| Config | File Size | Measured VRAM | Throughput | Quality (F1) | Time |
|---|---|---|---|---|---|
| GPTQ 4-bit (g=128) | 1.51 GB | 1.10 GB | N/A¹ | — | 852s |

¹ Throughput measurement skipped due to ExLlama kernel not being compiled on
Windows; the model loads and quantizes correctly but CUDA inference kernels
are unavailable in the pre-built auto_gptq wheel.

### Stage 8 Modules

| Module | Responsibility |
|---|---|
| `schemas/quantization.py` | `QuantMethod`, `QuantReport`, `QuantResult`, `QuantStatus` Pydantic models |
| `quantization/config.py` | `QuantConfig`, `GPTQConfig`, `AWQConfig`, `GGUFConfig` dataclasses + heuristic estimators |
| `quantization/quantizer.py` | `Quantizer` Protocol, `MockQuantizer`, `quantize_single()`, `select_best_config()`, `run_quantization_matrix()` |
| `quantization/export_gptq.py` | `GPTQQuantizer` (AutoGPTQ wrapper) |
| `quantization/export_awq.py` | `AWQQuantizer` (AutoAWQ wrapper) |
| `quantization/export_gguf.py` | `GGUFQuantizer` (llama.cpp / llama-cpp-python wrapper) |
| `quantization/cli.py` | Stage 8 CLI |

### Stage 8 Notes

- **Mock & dry-run modes** — no GPU or ML dependencies required.
- **Quality scoring** — `select_best_config()` weights quality (0.6),
  size (0.2), and speed (0.2).
- **GGUF quant types** — GGUF iterates over `Q2_K` through `Q8_0` rather
  than bit-widths.
- **Lazy imports** — `auto_gptq`, `autoawq`, `llama_cpp` are imported inside
  the quantizer classes' methods.
- **GGUF conversion** — `scripts/convert_to_gguf.py` converts HF safetensors
  → GGUF using the standalone `gguf` package (BFloat16 → float32 aware). The
  resulting GGUF checkpoint is served by Stage 9's `llama-server` backend.

---

## Stage 9 — Air-Gapped Serving

Stage 9 provides air-gapped serving via a FastAPI app + Typer CLI with five
backend options:

| Backend | Transport | Use Case |
|---|---|---|
| `mock` | In-process | Testing, CI, dry-run (no model needed) |
| `llama.cpp` | In-process (`llama-cpp-python`) | CPU or GPU inference via GGUF checkpoint |
| `llama-server` | HTTP subprocess | Air-gapped GPU serving via `llama-server.exe` |
| `ollama` | HTTP | Local Ollama daemon |
| `none` | — | Config validation only |

### Quick Start — GPU Serving (Recommended)

The GPU serving path uses `llama-cpp-python` compiled with CUDA support
(`GGML_CUDA=on`) inside a Docker container. The quantized GGUF checkpoint
(Q4_K, 786 MB) runs entirely on the GPU.

#### Prerequisites

```bash
# 1. Install NVIDIA Container Toolkit (Linux)
#    (Windows users — Docker Desktop handles this automatically)
#    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# 2. Verify GPU access:
docker run --rm --gpus all nvidia/cuda:12.4.1-devel-ubuntu22.04 nvidia-smi
```

#### Build & Run

```bash
# 1. Place your GGUF checkpoint in the model directory
#    F32:   output/quantized/model.gguf       (824 MB, full precision)
#    Q4_K:  output/quantized_q4/model_q4_k.gguf (824 MB, 4-bit quantized)
cp output/quantized/model.gguf output/quantized/model.gguf

# 2. Build the GPU serving image (multi-stage: CUDA devel → runtime)
docker compose --profile gpu build serving-gpu

# 3. Start the container with GPU access
docker compose --profile gpu up serving-gpu -d

# 4. Verify health
curl -s http://localhost:8000/healthz
# {"status":"ok","backend":"llama.cpp"}
```

#### Serve a Real Example

```bash
# Single vulnerability analysis (real-time GPU inference)
curl -s -X POST http://localhost:8000/api/v1/serve \
  -H "Content-Type: application/json" \
  -d '{
    "sample_id": "gold_001",
    "vulnerable_code": "def get_user(user_id):\n    query = \"SELECT * FROM users WHERE id = \" + str(user_id)\n    cursor.execute(query)\n    return cursor.fetchone()",
    "language": "python",
    "cwe_id": "CWE-89",
    "severity": "critical",
    "description": "SQL injection via string concatenation in user lookup query."
  }' | python3 -m json.tool
```

#### CLI (Local / Non-Docker)

```bash
# 1. Dry-run — print config and warnings without loading a model
python -m app.evaluation.cli stage9 serve --dry-run --backend mock

# 2. Analyze a single sample from a JSON file (no server needed)
echo '{"vulnerable_code": "cursor.execute(\"SELECT * FROM users WHERE id = \" + user_id)", "language": "python"}' > /tmp/sample.json
python -m app.evaluation.cli stage9 serve --backend mock --analyze -i /tmp/sample.json

# 3. Batch analysis from a JSON array
python -m app.evaluation.cli stage9 serve --backend mock --batch -i /tmp/samples.json -o /tmp/results.json

# 4. Start the FastAPI server (mock backend — no model needed)
python -m app.evaluation.cli stage9 serve --backend mock --host 127.0.0.1 --port 8000

# 5. Start with a real GGUF checkpoint + GPU offloading
python -m app.evaluation.cli stage9 serve \
  -m ./output/quantized/model.gguf \
  --backend llama.cpp \
  --n-gpu-layers -1  # -1 = offload all layers to GPU
```

### API Endpoints

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/api/v1/serve` | `ServeRequest` | Analyze a single vulnerability |
| `POST` | `/api/v1/serve/batch` | `BatchServeRequest` | Analyze multiple samples |
| `GET` | `/api/v1/manifest` | — | Run provenance (run_id, backend, request count) |
| `GET` | `/healthz` | — | Health check |

### Stage 9 Modules

| Module | Responsibility |
|---|---|
| `schemas/serving.py` | `ServeRequest`, `ServeResponse`, `BatchServeRequest`, `BatchServeResponse` Pydantic models |
| `serving/config.py` | `ServingConfig` dataclass (backend, model_path, ports, generation params) |
| `serving/backends.py` | `ServingBackend` Protocol, `LlamaCppBackend`, `LlamaServerBackend`, `OllamaBackend`, `MockServingBackend` |
| `serving/serve.py` | `VulnerabilityServer` — ties backend to Stage 4 prompt/parser |
| `serving/api.py` | `create_app()` FastAPI factory with `/serve`, `/serve/batch`, `/manifest`, `/healthz` |
| `serving/cli.py` | Typer `stage9 serve` subcommand (serve / analyze / batch / dry-run modes) |
| `serving/Dockerfile.gpu` | Multi-stage CUDA Docker build for GPU serving |

### Stage 9 Real-Run Results (2026-08-21)

✅ **Real serving run on 2026-08-21** via `scripts/run_stage9_serve.py` using
`llama-server.exe` on Windows with the F32 GGUF checkpoint
(`output/quantized/model.gguf`, 824 MB). See `output/stage9/serve_result.json`.

| Metric | Value |
|---|---|
| Backend | `llama-server` (HTTP subprocess via `llama-server.exe`) |
| Model | Qwen2.5-Coder-1.5B-Instruct, F32 GGUF (824 MB) |
| Port | 8082 |
| Latency | 50,019 ms (first-token + generation) |
| Predicted CWE | CWE-78 (model output) |
| Actual CWE | CWE-89 |
| Parse | ✅ JSON parsed successfully |

> **Note**: The model misclassified the CWE (predicted CWE-78 "command injection"
> instead of CWE-89 "SQL injection"). This is the **pre-fine-tuning base model** —
> the vulnerability triage fine-tune (Stage 5) is what improves this. The serving
> pipeline itself (prompt → HTTP request → JSON parse) works end-to-end.

### Stage 9 Notes

- **Backend Protocol**: all backends implement `ServingBackend`
  (`generate(prompt) → str` + `model_info` property). `LlamaCppBackend` uses
  `llama-cpp-python`'s `Llama` class; `LlamaServerBackend` communicates via HTTP
  to `llama-server.exe`; `OllamaBackend` uses HTTP to Ollama; `MockBackend`
  returns deterministic fake JSON for testing.
- **Lazy imports** — `llama_cpp` and `httpx` are imported inside the backend
  classes' `_load()` methods.
- **Dry-run mode** — validates config and prints warnings without starting a
  server or backend.
- **Analyze / batch modes** — run the server's pipeline on a JSON file
  without starting uvicorn. Useful for CI or one-off batch processing.
- **GPU offloading** — `--n-gpu-layers -1` (or `N_GPU_LAYERS=-1` env var)
  offloads every transformer layer to the GPU. Any positive number N offloads
  only the first N layers (hybrid CPU+GPU mode).
- **Real-serving script** — `scripts/run_stage9_serve.py` starts
  `llama-server.exe` with a GGUF checkpoint, sends a real vulnerability-
  analysis request via HTTP `/completion`, parses the model's JSON response,
  and saves results to `output/stage9/serve_result.json`.

---

## Stage 10 — CI/CD & Regression Gate

Stage 10 is the CI/CD pipeline that gates every push with lint, security
scan, and automated tests. The workflow is defined at `.github/workflows/ci.yml`.

### Current Coverage

| Check | Tool | Status |
|---|---|---|
| Lint | `ruff check .` | ✅ Passing |
| Security scan | `bandit -r app -q` | ✅ Passing (0 issues in `app/`) |
| Unit tests | `pytest tests/unit --cov=app --cov-report=term-missing` | ✅ 1,464 tests, 98% coverage (no `[ml]` extras) |
| Integration tests (Stages 1–11) | `pytest tests/integration -v -k "stage..."` | ✅ Implemented |
| **Eval gate** — regression gate on CWE Macro-F1 / forgetting | `app.evaluation.cli stage10` | ✅ Implemented |
| Gitleaks (secret scanning) | `gitleaks/gitleaks-action@v2` (full git history) | ✅ Configured (`.gitleaks.toml`) |
| Trivy (vuln + config + secret scanning) | `aquasecurity/trivy-action` (`severity: CRITICAL,HIGH`) | ✅ Implemented |
| pip-audit (dependency vulnerabilities) | `pip-audit` | ✅ 0 vulnerabilities after upgrading torch ≥2.10, transformers ≥5.0 |

### CI Pipeline

The workflow (`.github/workflows/ci.yml`) is a 4-job pipeline:

```yaml
# .github/workflows/ci.yml — four-job pipeline
# Runs on: push, pull_request
# Python: 3.11
# Install: pip install -e ".[dev,data,ml]"
#
# test — ruff, bandit, unit tests, integration tests
# eval-gate (needs: test) — Stage 4→6→7→10 mock-mode pipeline + regression gate
# gitleaks (needs: test) — secret scan on full git history
# trivy (needs: test) — filesystem scan: vuln + misconfig + secret, CRITICAL/HIGH severity only
```

### Stage 10 Modules

| Module | Description |
|---|---|
| `ci/config.py` | `RegressionGateConfig` — frozen dataclass with artifact paths and thresholds |
| `ci/gate.py` | `RegressionGate` class, `run_gate()` convenience function, and artifact loaders |
| `ci/security_scanners.py` | `parse_gitleaks_output()`, `parse_trivy_output()` — defensive JSON parsers |
| `schemas/ci.py` | `GateStatus`, `GateCheck`, `RegressionGateResult`, `SecurityScanSummary`, `CiReport` |
| `.github/workflows/ci.yml` | 4-job workflow: `test`, `eval-gate`, `gitleaks`, `trivy` |
| `.gitleaks.toml` | Gitleaks config with allowlist for test fixtures |

### Quick Start (Mock Pipeline)

```bash
# Install with all extras
pip install -e ".[dev,data,ml]"

# Stage 4 baseline (mock, deterministic)
python -m app.evaluation.cli baseline \
  --gold-eval eval/gold_set/gold.jsonl --strategy zero_shot --mock \
  --output-dir ./output/stage4_baseline

# Stage 6 eval (mock sandbox)
python -m app.evaluation.cli stage6 \
  --gold-eval     eval/gold_set/gold.jsonl \
  --predictions   ./output/stage4_baseline/predictions.jsonl \
  --sandbox-mode  mock \
  --skip-tier4 \
  --output-dir    ./output/stage6

# Stage 7 regression (mock)
python -m app.evaluation.cli stage7 --mock \
  --base-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-model "ci-checkpoint" \
  --output-dir ./output/stage7

# Stage 10 gate — passes if F1 drop ≤5%, forgetting ≥-0.10, exec ≥0.0, halluc ≤0.50
python -m app.evaluation.cli stage10 \
  --baseline-metrics  ./output/stage4_baseline/metrics.json \
  --predictions       ./output/stage4_baseline/predictions.jsonl \
  --stage6-report     ./output/stage6/eval_report.json \
  --stage7-report     ./output/stage7/regression_report.json \
  --output-dir ./output/stage10
```

### Gate Checks

The regression gate (`app/ci/gate.py`) evaluates four checks:

1. **CWE F1 regression** — `model_cwe_macro_f1` (Stage 6) must not drop more than
   `max_f1_drop_percent` (default 5%) below `cwe_macro_f1` (Stage 4 baseline).
2. **Forgetting** — `forgetting_delta` (Stage 7) must not fall below
   `forgetting_threshold` (default -0.10). Skipped if no Stage 7 report provided.
3. **Exec pass rate** — `exec_pass_rate` must meet `min_exec_pass_rate`
   (default 0.0).
4. **Hallucination rate** — must not exceed `max_hallucination_rate` (default 0.50).

---

## Stage 11 — Documentation & Interview Package

Stage 11 generates the documentation deliverables (model card, training report,
and demo script) that accompany the project.

### Deliverables

| Deliverable | Status |
|---|---|
| Model card (`docs/model_card.md`) | ✅ Generated |
| Training report (`docs/training_report.md`) | ✅ Generated |
| Demo script (`docs/demo.py`) | ✅ Generated |
| Mock evaluation dashboard (`output/mock_eval_dashboard.html`) | ✅ Available |

### Generating the Deliverables

```bash
# Generate all three deliverables (model card, training report, demo script)
python -m app.evaluation.cli stage11 --docs-dir docs --output-dir ./output/stage11

# Optionally run the mock-mode demo pipeline (Stages 4 -> 6 -> 7 -> 10)
python -m app.evaluation.cli stage11 --run-demo

# Programmatic usage
python -c "
from app.stage11.config import Stage11Config
from app.stage11.generator import Stage11Generator
gen = Stage11Generator(Stage11Config())
gen.ensure_deliverables()
assert gen.validate_deliverables()
if True:
    gen.run_demo()
"
```

### Stage 11 Modules

| Module | Description |
|---|---|
| `schemas/documentation.py` | Pydantic contracts (`ModelCardData`, `TrainingReportData`, etc.) and project constants (`CWE_SCOPE`, `BASE_MODEL`, `TRAINING_METHODS`, `LANGUAGE_SCOPE`) |
| `stage11/config.py` | `Stage11Config`, a frozen dataclass with README defaults |
| `stage11/generator.py` | `Stage11Generator` class — creates and validates deliverables |
| `evaluation/cli.py` | The `stage11` Typer CLI subcommand |

---

## Testing

The test suite is ruff-clean and Bandit-clean for the CI-scoped run
(`bandit -r app -q`). Verified on 2026-08-26: **1,641 tests** total
(1,464 unit + 177 integration); running unit + integration together,
**1,640 pass**. The one failure without the `[ml]` extras installed
(`test_record_peak_memory_noop_without_gpu`) is an environment gap, not a
real bug — it passes once `torch` is available, as it is in CI. All tests
run in mock/dry-run mode — no GPU, Docker, or network required.

```bash
# Full suite (recommended)
pytest tests/ -v

# Unit tests only (fast, with coverage report)
pytest tests/unit -v --cov=app --cov-report=term-missing

# Integration tests only (Stage 1–11, mock mode)
pytest tests/integration -v

# Per-stage focused runs
pytest tests/unit/test_tier1_deterministic.py        # Stage 6 Tier 1
pytest tests/integration/test_stage4_baseline.py     # Stage 4 baseline
pytest tests/integration/test_stage6_four_tier.py    # Stage 6 full pipeline
pytest tests/integration/test_stage7_regression.py   # Stage 7 regression
pytest tests/integration/test_stage8_quantization.py # Stage 8 quant
pytest tests/integration/test_stage9_serving.py      # Stage 9 serving
pytest tests/integration/test_stage10_ci.py          # Stage 10 CI gate
pytest tests/integration/test_stage11_docs.py        # Stage 11 docs

# Linting & security
ruff check .
bandit -r app -q
pip-audit                                    # dependency vulnerability scan
trivy fs --skip-dirs .venv,output --severity CRITICAL,HIGH .  # requires trivy install
```

### Test Structure

| Directory | Contents |
|---|---|
| `tests/unit/` | One file per module — 54 unit test files, 1,464 tests, covering all 11 stages |
| `tests/integration/` | One file per stage — 12 files, 177 tests, end-to-end pipeline tests in mock mode |

### Design Principles in Tests

- **Lazy ML imports**: Tests use `patch.dict(sys.modules, ...)` to inject
  mock modules for torch, transformers, peft, etc. — no real ML deps needed.
- **Injectable backends**: `Protocol`-based backends (`ModelBackend`,
  `ServingBackend`, `Quantizer`, `DataLoadable`) are mocked in tests.
- **Typer OptionInfo**: Direct calls to Typer CLI functions use helper kwargs
  builders since `typer.Option` creates `OptionInfo` objects, not real values.

---

## Makefile Targets

| Target | Description |
|---|---|
| `make install` | Install dependencies: `pip install -e ".[dev]"` |
| `make test` | Run unit tests with coverage: `pytest tests/unit -v --cov=app --cov-report=term-missing` |
| `make lint` | Run linters: `ruff check .` |
| `make security` | Run security scanner: `bandit -r app -q` |
| `make scan` | Run Trivy filesystem scan: `trivy fs --skip-dirs .venv,output --severity CRITICAL,HIGH .` |
| `make up` | Start infra services: `docker compose up -d` |
| `make down` | Stop infra services: `docker compose down` |

---

## Contributing

This project follows a test-first approach with mock/dry-run modes for every
stage. To contribute:

1. **Install dependencies:** `pip install -e ".[dev,data,ml]"`
2. **Run tests:** `pytest tests/ -v` (all should pass without GPU/network)
3. **Check lint:** `ruff check .`
4. **Check security:** `bandit -r app -q`
5. **Scan for vulns:** `pip-audit` (or `trivy fs --skip-dirs .venv,output --severity CRITICAL,HIGH .`)
6. **Make your changes** — follow the lazy-import pattern for ML deps,
   implement `Protocol` interfaces for injectable backends, and add unit
   tests that use mock backends.
7. **Run tests again** to ensure nothing regresses.

> **Note:** The `CWE_SCOPE`, `TRAINING_METHODS`, and `BASE_MODEL` constants
> live in `app/schemas/documentation.py` and should be treated as the
> source of truth for any documentation or model card generation.

---

## License

MIT — see [LICENSE](LICENSE).