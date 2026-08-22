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

✅ **Stage 0 — environment & repo skeleton.**
✅ **Stage 1 — data collection.** ✅ **Run end-to-end on 2026-08-16** via
`scripts/run_stage1_real.py` (deterministic mock NVD client + real bundled
Semgrep rules, CVEfixes v1.0.8 schema). Results: 992 raw pairs processed,
621 kept after token-budget filter (404 train / 114 val / 103 test), 371 dropped —
see `output/stage3/manifest.json` and [Stage 1 notes](#stage-1-notes).
✅ **Stage 2 — cleaning, dedup, leakage-safe split, contamination check.**
✅ **Run end-to-end** — Stage 1 output was deduped, split, and token-budget
filtered, producing the 404/114/103 instruction-format dataset in `output/stage3/`.
✅ **Stage 3 — instruction-format dataset build.** Prompt template (system +
task prompt with vulnerable code + static findings), injectable token
counter (Qwen tokenizer with heuristic fallback), token-budget enforcement,
unified-diff patch generation, and JSONL split writers are implemented and
unit-tested + integration-tested.
✅ **Stage 4 — pre-fine-tuning baseline.** ✅ **Real baseline run on 2026-08-16**
(zero-shot evaluation of Qwen2.5-Coder-1.5B-Instruct on the 59-sample gold-eval
set). Zero-shot and few-shot evaluation with CWE Macro-F1, severity accuracy,
hallucination rate, and patch coverage metrics. Fully implemented and tested.
✅ **Stage 5 — training matrix.** ✅ **Real GPU QLoRA training run on 2026-08-16**
(1.5B, LoRA r=8, 4-bit NF4, 3 epochs, 404 train samples, peak VRAM 8.79 GB
on RTX 4060 — see `scripts/run_gpu_training.py`). CPU-compatible training also
available via `scripts/run_cpu_training.py`. All modes support `--dry-run`.
✅ **Stage 6 — four-tier evaluation harness.** ✅ **Real eval run on 2026-08-16**
(Tier 3 uses Docker sandbox; 59 gold samples, 12 model predictions).
Deterministic (Tier 1) → static+embedding (Tier 2) → exec sandbox (Tier 3) →
LLM-judge (Tier 4).
✅ **Stage 7 — regression / forgetting analysis.** ✅ **Real run on 2026-08-16**
(tuned vs. base on 1.5B checkpoint). General code-capability delta on
HumanEval-style tasks.
✅ **Stage 8 — quantization matrix.** GPTQ / AWQ / GGUF with quality-vs-VRAM
trade-off scoring. ✅ **Real run on 2026-08-20** (GPTQ 4-bit on Qwen2.5-Coder-1.5B
LoRA checkpoint). Mock and dry-run modes supported.
✅ **Stage 9 — air-gapped serving.** llama.cpp / Ollama / mock backends behind
a FastAPI service + Typer CLI (serve / analyze / batch / dry-run modes).
✅ **Stage 10 — CI/CD & regression gate.** GitHub Actions workflow with
ruff, Bandit, pytest, eval gate (Stage 4→6→7→10 mock pipeline), Gitleaks
(secret scanning), and Trivy (vuln + config scanning).
✅ **Stage 11 — documentation & interview package.** Model card
(`docs/model_card.md`), training report (`docs/training_report.md`), and demo
script (`docs/demo.py`) generated and validated via CLI (`stage11` subcommand)
from real GPU QLoRA training run + Docker-sandbox eval (2026-08-16).

> **Test suite:** 1449 tests pass (1290 unit + 159 integration), ruff clean,
> Bandit clean, **99% code coverage** (1 line uncovered).
> All tests run in mock/dry-run mode — no GPU, Docker, or network required.
> (The exact count varies slightly by environment depending on which optional
> extras are installed; in a clean `.[dev,data,ml]` install it is 1449.)

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
STAGE 11 Documentation & interview package   ✅ Done (README, model card, training report, demo script)
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
│   ├── serving/          # FastAPI app, Typer CLI, backends, config             (Stage 9)
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
├── tests/{unit,integration}/   # 1449 tests total (1290 unit + 159 integration), 99% coverage
├── .github/workflows/ci.yml    # ruff, Bandit, pytest, eval-gate, Gitleaks, Trivy
├── .gitleaks.toml      # Gitleaks config with allowlist for test fixtures
├── docker-compose.yml    # Postgres + Redis + MinIO
├── Makefile              # install, test, lint, security, up, down
└── pyproject.toml
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
| Quantization | AutoGPTQ, AutoAWQ, llama.cpp (GGUF) |
| Serving | llama.cpp server (air-gapped/CPU), vLLM (GPU) |
| Orchestration | Celery + Redis |
| State/metrics DB | PostgreSQL |
| CI/CD | GitHub Actions — pytest, ruff, Bandit, Gitleaks, Trivy |

> **Model size note:** Qwen2.5-Coder-7B-Instruct is the **designed-for** model.
> CPU-only validation runs (no CUDA GPU) use the 1.5B-Instruct variant — the
> same inference code, smaller checkpoint. The real GPU QLoRA training run on
> RTX 4060 (8 GB VRAM) used the 1.5B-Instruct variant with 4-bit NF4 quantization
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

# 3. Run the test suite
pytest tests/unit -v
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
| `app/data/collectors/cwe_scope.py` | `CWE_SCOPE` constant (6 classes), helper functions |
| `app/data/collectors/cvefixes_loader.py` | `CveFixesLoader` — loads CVEfixes SQLite v1.0.8 schema |
| `app/data/collectors/nvd_client.py` | `NvdClient` — NVD API enrichment client |
| `app/data/collectors/semgrep_runner.py` | `run_semgrep()` — runs bundled Semgrep rules on a code snippet |
| `app/data/collectors/rules/` | Bundled Semgrep rule packs (python.yaml, javascript.yaml) |
| `app/data/collectors/pipeline.py` | Orchestrates collection → enrichment → scanning → persistence |
| `app/data/collectors/cli.py` | Stage 1 CLI (`collect` subcommand) |

### Stage 1 Notes

- **CWE scope** (`app/data/collectors/cwe_scope.py`): `CWE-89` (SQLi),
  `CWE-79` (XSS), `CWE-22` (path traversal), `CWE-78` (command injection),
  `CWE-190` (integer overflow), `CWE-502` (unsafe deserialization).
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

### Stage 2 Modules

| Module | Responsibility |
|---|---|
| `app/data/cleaning/embeddings.py` | HuggingFace `sentence-transformers` backend (`jina-embeddings-v2-base-code`) |
| `app/data/cleaning/dedup.py` | Near-duplicate removal via cosine similarity on code embeddings |
| `app/data/cleaning/split.py` | Repo-based leakage-safe split with CWE stratification and class balance |
| `app/data/cleaning/contamination.py` | N-gram (5-gram) contamination checker between train and eval sets |
| `app/data/cleaning/hf_dataset.py` | HuggingFace `datasets` integration (export to Hub, load from disk/Hub) |
| `app/data/cleaning/pipeline.py` | Orchestrates load → dedup → split → contamination → persist |
| `app/data/cleaning/cli.py` | Stage 2 CLI (`clean`, `plan`, `export`, `check-contamination`) |

### Stage 2 Notes

- **Leakage-safe split**: repos are grouped and assigned to train/val/test
  so that no repository appears in more than one split.
- **Class balance**: within each CWE class, repos are distributed proportionally
  across splits, so CWE distribution is preserved.
- **Contamination gate**: the eval/test set must have <5% 5-gram overlap with
  the training set. Checked automatically in the pipeline and fails CI
  (Stage 10) if exceeded.
- **HuggingFace note**: the default embedding model (`jina-embeddings-v2-base-code`)
  requires `trust_remote_code=True`. If you hit an `ImportError` from
  `transformers.pytorch_utils`, either pin `transformers<5` or use a model
  without custom code: `EmbeddingBackend(model_name="intfloat/multilingual-e5-base", trust_remote_code=False)`.

---

## Stage 3 — Instruction-Format Dataset Build

After Stage 2 has produced split `VulnSample` records, Stage 3 builds
instruction-format JSONL with prompt templates, token-budget enforcement,
and unified-diff patch generation.

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

### Stage 3 Modules

| Module | Responsibility |
|---|---|
| `app/data/formatting/template.py` | Prompt template (system + task prompt), static-finding formatter, unified-diff patch generator |
| `app/data/formatting/tokenizer.py` | Injectable token counter (Qwen tokenizer with heuristic fallback for air-gapped/CI) |
| `app/data/formatting/builder.py` | Builds `InstructionExample` records from `VulnSample` with token-budget enforcement |
| `app/data/formatting/pipeline.py` | Orchestrates load → build → JSONL write, with manifest output |
| `app/data/formatting/cli.py` | Stage 3 CLI (`build`, `stats`, `inspect`) |

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

# 2. Few-shot baseline with 3 in-context examples from Stage 3 output
python -m app.evaluation.cli baseline \
  --gold-eval eval/gold_set/gold.jsonl \
  --strategy few-shot \
  --num-shots 3 \
  --few-shot-examples ./output/stage3/train.jsonl \
  --output-dir ./output/stage4

# 3. Mock mode (no model download — deterministic fake predictions)
python -m app.evaluation.cli baseline \
  --gold-eval eval/gold_set/gold.jsonl \
  --strategy zero-shot \
  --mock \
  --output-dir ./output/stage4

# 4. Re-evaluate saved predictions without re-running inference
python -m app.evaluation.cli evaluate \
  --predictions ./output/stage4/predictions.jsonl \
  --gold-eval eval/gold_set/gold.jsonl
```

### Output Files

`output/stage4/` contains:

| File | Contents |
|---|---|
| `predictions.jsonl` | One `ModelPrediction` per line (sample_id, run_id, predicted_cwe, predicted_severity, suggested_patch_diff, rationale) |
| `metrics.json` | Aggregate metrics (CWE Macro-F1, micro accuracy, severity accuracy, hallucination rate, patch coverage, per-class F1) |
| `manifest.json` | Run provenance (stage, strategy, base_model, num_gold_samples, num_predictions, run_id) |
| `parse_errors.jsonl` | Samples whose model output could not be parsed (one `ParseError` per line) |

### Stage 4 Modules

| Module | Responsibility |
|---|---|
| `app/evaluation/backends.py` | `ModelBackend` Protocol + `QwenBackend` (lazy-loaded transformers) + `MockBackend` for testing |
| `app/evaluation/prompt.py` | `build_zero_shot_prompt()` and `build_few_shot_prompt()` using Stage 3's `format_prompt` |
| `app/evaluation/parser.py` | `parse_prediction()` — extracts JSON from model output (markdown fences + brace-matching fallback) |
| `app/evaluation/metrics.py` | CWE Macro-F1, micro accuracy, severity accuracy, hallucination rate, patch coverage |
| `app/evaluation/baseline.py` | `BaselineConfig` + `BaselineResult` + `run_baseline()` orchestration (load → prompt → generate → parse → metrics → write) |
| `app/schemas/dataset.py` | `InstructionExample` Pydantic model |

### Stage 4 Notes

- **No model download required for tests.** The test suite uses `MockBackend`,
  which returns deterministic fake predictions — no GPU or network needed.
- **CWE scope**: the 6 target classes (CWE-89, CWE-79, CWE-22, CWE-78,
  CWE-190, CWE-502) are enforced in the parser and metrics. Out-of-scope CWE
  IDs (e.g. `CWE-999`) are counted as **hallucinations**, not just wrong
  predictions.
- **Few-shot fallback**: if `--strategy few-shot` is selected but no
  `--few-shot-examples` file is provided, the runner automatically falls back
  to zero-shot mode (logged as a warning).
- **Gold-eval set**: 59 samples across 6 CWE classes (CWE-89: 14, CWE-79: 14,
  CWE-22: 14, CWE-78: 8, CWE-190: 4, CWE-502: 5) for fast, reproducible
  baseline evaluation.

---

## Stage 5 — Training Matrix

Stage 5 implements the full training matrix: SFT (full-parameter and QLoRA),
LoRA rank sweep, and DPO preference alignment. Uses
Qwen2.5-Coder-7B-Instruct as the base model, with PEFT/LoRA/QLoRA (bitsandbytes
4-bit NF4), TRL's `DPOTrainer` for preference optimization, and W&B for
loss-curve tracking.

> **No GPU required for dry-run.** All training modes support `--dry-run`,
> which loads the Stage 3 JSONL data, estimates training steps and VRAM, and
> returns a `TrainingResult` — no torch/transformers/GPU needed. Real training
> requires `pip install -e '.[ml]'` and a CUDA GPU with >=8 GB VRAM (QLoRA)
> or >=16 GB (full-parameter SFT).

### Prerequisites

Stage 3 must have produced `train.jsonl` and `val.jsonl` (InstructionExample
format). If you don't have them yet, generate them from the gold-eval set:

```bash
# (Optional) generate small Stage 3 files from the gold-eval set for testing
python -c "
from app.data.formatting.builder import build_instruction_example
from app.data.formatting.tokenizer import TokenCounter
from app.evaluation.baseline import load_gold_eval
import json

class _MockTok:
    def encode(self, text): return list(range(max(len(text), 1)))

counter = TokenCounter(tokenizer=_MockTok())
samples = load_gold_eval('eval/gold_set/gold.jsonl')
train = samples[:8]
val = samples[8:]
for name, split in [('train', train), ('val', val)]:
    with open(f'./output/stage3/{name}.jsonl', 'w') as f:
        for s in split:
            ex = build_instruction_example(s, token_counter=counter, max_tokens=100000)
            if ex: f.write(ex.model_dump_json() + '\n')
print(f'Wrote {len(train)} train, {len(val)} val examples')
"
```

### SFT: Full-parameter vs QLoRA

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
#   Output:
#   Method:    sft_qlora
#   Base model: Qwen/Qwen2.5-Coder-7B
#   Train set: 8 examples
#   Peak VRAM: 7.00 GB (estimated)
```

### LoRA Rank Sweep

Sweeps across ranks `[8, 16, 32, 64, 128]` and selects the best by validation
loss:

```bash
# Full 5-rank sweep (dry-run — no GPU needed)
python -m app.training.cli lora-sweep \
  --train-jsonl ./output/stage3/train.jsonl \
  --val-jsonl   ./output/stage3/val.jsonl \
  --dry-run \
  --no-persist

#    Output:
#    Starting LoRA sweep: ranks=[8, 16, 32, 64, 128], model=Qwen/Qwen2.5-Coder-7B
#    ...
#    Sweep: lora_sweep_Qwen2.5-Coder-7B  (5 runs)
#    Best rank: 8  (val_loss=0.2341)

# Real training (remove --dry-run, ensure GPU is available)
python -m app.training.cli lora-sweep \
  --train-jsonl ./output/stage3/train.jsonl \
  --val-jsonl   ./output/stage3/val.jsonl \
  --ranks 8,16,32,64,128
```

### DPO Preference Alignment

DPO fine-tunes the model to prefer correct CWE classifications and patches
over incorrect ones. The "chosen" response comes from the Stage 3 ground-truth
targets; the "rejected" response is a synthetic baseline (wrong CWE, shallow
explanation):

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
| `app/training/config.py` | `TrainingMethod` enum, `SFTConfig`, `DPOConfig`, `SweepConfig` dataclasses |
| `app/training/data.py` | `JsonlDataLoader` (injectable), `load_examples()`, `compute_stats()`, `make_hf_dataset()` (lazy `datasets` import) |
| `app/training/callbacks.py` | `TrainingCallback` Protocol, `WandbCallback` (mock mode), `CheckpointCallback` (MinIO upload), `ProgressCallback`, `ResourceTracker` (peak VRAM) |
| `app/training/experiment.py` | `persist_training_run()`, `load_training_run()`, `list_training_runs()` (PostgreSQL via SQLAlchemy), `generate_run_id()` |
| `app/training/trainer_sft.py` | `run_sft()` (full + QLoRA), `estimate_training_steps()` (pure arithmetic), `TrainingUnavailableError` |
| `app/training/trainer_dpo.py` | `run_dpo()` with TRL `DPOTrainer`, `estimate_dpo_steps()`, `build_preference_pairs()` |
| `app/training/sweep.py` | `run_lora_sweep()` — orchestrates multiple `run_sft` calls across ranks, `SweepReport` summary |
| `app/training/cli.py` | Typer CLI: `sft`, `lora-sweep`, `dpo`, `list-runs`, `inspect` subcommands |

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

Input: gold-eval samples (`VulnSample`) + model predictions (`ModelPrediction`).
Output: `EvalReport` with per-tier results, aggregate `EvalMetrics`, and a
run manifest.

```bash
# Run the full four-tier harness (mock sandbox + mock LLM judge — no Docker/ML API)
python -m app.evaluation.cli stage6 \
  --gold-eval     eval/gold_set/gold.jsonl \
  --predictions   output/stage6/predictions.jsonl \
  --output-dir    ./output/stage6 \
  --base-model    "mock-model"

# Run with real sandbox tests (subprocess-based, no Docker)
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions output/stage6/predictions.jsonl \
  --sandbox-mode local

# Run with embedding similarity (requires sentence-transformers)
pip install -e ".[ml]"
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions output/stage6/predictions.jsonl \
  --embedding-model "intfloat/multilingual-e5-base"

# Skip expensive tiers to save time/cost
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions output/stage6/predictions.jsonl \
  --skip-tier3 --skip-tier4
```

### Programmatic Use

```python
from app.evaluation.runner import EvalConfig, EvaluationRunner, load_samples, load_predictions

config = EvalConfig(
    base_model="Qwen2.5-Coder-7B-Instruct",
    sandbox_mode="mock",  # or "local" for subprocess
    skip_tier4=True,  # disable LLM judge to save cost
)
runner = EvaluationRunner(config=config)

samples = load_samples("eval/gold_set/gold.jsonl")
preds = load_predictions("output/stage6/predictions.jsonl")

report = runner.run(samples, preds)
print(f"Model Macro-F1: {report.metrics.model_cwe_macro_f1:.4f}")
print(f"Exec Pass Rate: {report.metrics.exec_pass_rate:.4f}")
```

### Stage 6 Modules

| Module | Responsibility |
|---|---|
| `app/schemas/prediction_eval.py` | `Tier1Result`, `Tier2Result`, `ExecEvalResult`, `LlmJudgeScore`, `EvalMetrics`, `EvalReport`, `RegressionSummary` Pydantic models |
| `app/evaluation/tier1_deterministic.py` | `PatternRule` dataclass, `DEFAULT_TIER1_RULES` (20 regex rules for all 6 CWEs), `DeterministicEvaluator` |
| `app/evaluation/tier2_embedding_static.py` | `DEFAULT_RULE_TO_CWE` (20 rule IDs → CWE), `EmbeddingBackend` (lazy `sentence-transformers` import), `StaticSignalEvaluator` |
| `app/evaluation/tier3_exec.py` | `SandboxRunner` Protocol, `LocalSandboxRunner`, `MockSandboxRunner`, `ExecEvaluator`, `apply_unified_diff()`, `TestGenerator` (per-CWE test templates), `check_hallucinated_function_ref()` |
| `app/evaluation/tier4_llm_judge.py` | `LlmJudgeBackend` Protocol, `LlmJudge`, `MockLlmJudgeBackend`, judge prompt for explanation quality + patch minimality |
| `app/evaluation/runner.py` | `EvalConfig`, `EvaluationRunner` (orchestrates all 4 tiers), `compute_metrics()`, `load_samples()` / `load_predictions()` I/O helpers |

### How the Four Tiers Work

1. **Tier 1 — Deterministic baseline.** Pure-Python regex rules (no model, no
   Semgrep, no Docker). Achieves 12/12 on the original 12-sample subset; on the
   expanded 59-sample gold set, Tier 1 achieves Macro-F1=0.50 with 37.3% coverage.
   This is the floor: any model must beat it.

2. **Tier 2 — Static signal + embedding.** Maps Semgrep findings to CWE IDs
   (static-only, no model needed) and optionally computes cosine similarity
   between the model's patch and the gold fix using `sentence-transformers`.
   When embeddings aren't configured, it runs in static-only mode.

3. **Tier 3 — Exec sandbox.** The model's `suggested_patch_diff` is applied
   to the vulnerable code via a pure-Python unified-diff applier, then a
   CWE-specific test is generated and run in an isolated subprocess
   (`LocalSandboxRunner`) or Docker container. Produces `patch_applies_cleanly`,
   `build_succeeds`, and `tests_pass_after_patch` booleans.

4. **Tier 4 — LLM judge.** An LLM rates the model's explanation quality and
   patch minimality on a 0–1 scale. Used only for qualitative assessment,
   never for pass/fail decisions.

### Stage 6 Notes

- **No GPU or model download required for tests.** All tiers use mock
  backends — `MockSandboxRunner` returns canned results, `MockLlmJudgeBackend`
  returns fixed scores, and `sentence-transformers` is an optional lazy import.
- **Leakage-safe.** Tier 3 runs in an isolated temp directory; the vulnerable
  code is never executed from the repo workspace. `--sandbox-mode docker`
  uses `DockerSandboxRunner` — containers run with a read-only filesystem,
  no network, and a memory limit. `--sandbox-mode local` uses subprocess
  isolation for environments without Docker.
- **Patch applier.** `apply_unified_diff()` is a pure-Python implementation —
  no dependency on `git apply` or the `patch` command.
- **Hallucination detection.** Tier 3 checks CWE ID validity (must be in the
  6-class scope) and function-reference hallucination (patch references
  identifiers not present in the vulnerable code).

---

## Stage 7 — Regression / Forgetting Analysis

Stage 7 implements **regression / forgetting analysis** — the "after" half of
the before/after comparison. After fine-tuning (Stage 5) and evaluating on
security tasks (Stage 6), the tuned model is re-evaluated on a set of
general-purpose (non-security) code-generation tasks. The **forgetting delta**
measures whether general coding ability was lost during fine-tuning:

```
delta = tuned_exec_accuracy − base_exec_accuracy
```

A *negative* delta means the fine-tuned model suffered catastrophic
forgetting. A *positive* delta means the fine-tuned model improved general
coding. Zero means no net change.

```bash
# Mock mode — deterministic, no model download, no subprocess (fast)
python -m app.evaluation.cli stage7 \
  --mock \
  --base-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-model "sft_qlora_r8" \
  --output-dir ./output/stage7

#    Output:
#    Running Stage 7: regression / forgetting analysis
#    Base model:  Qwen/Qwen2.5-Coder-7B-Instruct
#    Tuned model: sft_qlora_r8
#    Tasks:       12
#    Mock mode:   True
#
#    Forgetting delta: +0.0000
#    ✅ No forgetting — tuned model maintains or improves general capability.

# Real mode — uses QwenBackend + LocalCodeTestRunner (spawns subprocesses)
python -m app.evaluation.cli stage7 \
  --base-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-model ./output/stage5/sft_qlora/final_checkpoint \
  --timeout 60 \
  --output-dir ./output/stage7
```

### Real-Mode Script

For running Stage 7 against a real trained checkpoint (Stage 5 output), use the
dedicated script `scripts/run_stage7_only.py`. This mirrors the pattern of
`scripts/run_stage6_only.py` — it loads the LoRA checkpoint, creates
`QwenBackend` instances for both the base and tuned models, and runs the full
regression analysis with `LocalCodeTestRunner`:

```bash
python scripts/run_stage7_only.py \
  --base-model "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --checkpoint ./output/stage5/sft_qlora/final_checkpoint \
  --timeout 60 \
  --output-dir ./output/stage7
```

Optionally pass `--stage6-report` (path to Stage 6 `eval_report.json`) or
`--stage6-metrics` (path to `output/stage5/eval_results.json`) to generate a
`regression_summary.json` combining Stage 6 metrics + Stage 7 forgetting delta
+ cost estimate — ready for the Stage 10 regression gate:

```bash
python scripts/run_stage7_only.py \
  --base-model "Qwen/Qwen2.5-Coder-1.5B-Instruct" \
  --checkpoint ./output/stage5/sft_qlora/final_checkpoint \
  --stage6-report ./output/stage6/eval_report.json \
  --inference-cost-usd 12.50 \
  --training-cost-usd 48.00 \
  --output-dir ./output/stage7
```

**Output files** (in `output/stage7/`):

| File | Contents |
|---|---|
| `regression_report.json` | Full `RegressionReport` — base/tuned metrics, forgetting delta, manifest |
| `regression_summary.json` | `RegressionSummary` — Stage 6 metrics + Stage 7 delta + cost-per-accepted-patch |
| `manifest.json` | Run provenance (script, model names, checkpoint type, timeout, timestamp) |

### Programmatic Use

```python
from app.evaluation.general_capability import (
    RegressionConfig,
    run_regression_analysis,
    build_regression_summary,
)
from app.evaluation.backends import MockBackend

config = RegressionConfig(
    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    tuned_model="sft_qlora_r8",
)
report = run_regression_analysis(
    config=config,
    base_backend=MockBackend(default="pass"),
    tuned_backend=MockBackend(default="pass"),
)
```

### Output Files

`output/stage7/` contains:

| File | Contents |
|---|---|
| `regression_report.json` | Full `RegressionReport` — base/tuned metrics, forgetting delta, manifest |

### Stage 7 Modules

| Module | Responsibility |
|---|---|
| `app/evaluation/general_capability.py` | 12 HumanEval-style tasks, `GeneralCapabilityTask`, `CodeTestRunner` Protocol, `LocalCodeTestRunner` (subprocess pytest), `MockCodeTestRunner`, `GeneralCapabilityEvaluator`, `RegressionConfig`, `run_regression_analysis()`, `build_regression_summary()`, `estimate_cost_per_accepted_patch_usd()` |
| `app/evaluation/cli.py` | `stage7` Typer subcommand with `--mock`, `--base-model`, `--tuned-model`, `--timeout`, `--output-dir`, `--verbose` flags |
| `app/ci/gate.py` | `load_stage7_report()` — loads `regression_report.json`, checks `forgetting_delta` against `forgetting_threshold` (default -0.10) |
| `app/schemas/prediction_eval.py` | `GeneralCapabilityMetrics`, `RegressionReport`, `RegressionSummary` pydantic models |
| `scripts/run_stage7_only.py` | Real-mode script — loads Stage 5 LoRA checkpoint, creates `QwenBackend` instances (base + tuned), runs regression analysis with `LocalCodeTestRunner`, optionally builds `RegressionSummary` from Stage 6 outputs |

### Stage 7 Notes

- **No GPU or model download required for tests.** Uses `MockBackend` +
  `MockCodeTestRunner` (deterministic, no subprocess). For tests that *do*
  exercise real code execution, `LocalCodeTestRunner` spawns isolated
  `python -m pytest` subprocesses — no Docker needed.
- **Lazy ML imports.** Heavy dependencies (`torch`, `transformers`,
  `sentence-transformers`) are imported inside functions.
- **Injectable backend pattern.** Both `ModelBackend` (code generation) and
  `CodeTestRunner` (code execution) are injectable Protocols.
- **12 default tasks.** Factorial, palindrome, fibonacci, binary search,
  two-sum, vowel counting, integer reversal, anagram, longest common prefix,
  valid parentheses, remove duplicates, and max subarray sum — all pure-Python.
- **Forgetting delta = `tuned_acc − base_acc`**. Negative = forgetting,
  positive = improvement. Feeds into `RegressionSummary`, consumed by
  the Stage 10 regression gate.

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

#    Output (QuantReport JSON):
#    Best: gguf:Q4_0  (F1≈0.92, 6.8 GB, 32 t/s)

# 2. Dry-run mode — heuristic estimates, no actual quantization
python -m app.evaluation.cli stage8 \
  --source-checkpoint ./output/stage5/sft_qlora \
  --dry-run \
  --methods gptq,awq,gguf \
  --bits 2,3,4 \
  --target-vram-gb 8.0     # filter to configs that fit in 8 GB VRAM

# 3. Real quantization via CLI (app.evaluation.cli)
python -m app.evaluation.cli stage8 \
  --source-checkpoint ./output/stage5/sft_qlora \
  --methods gptq,gguf \
  --bits 4 \
  --output-dir ./output/stage8

# 4. Real quantization via standalone script (scripts/run_stage8_real.py)
#    This script loads the Stage 5 LoRA checkpoint, merges the adapter into
#    the base model, runs real GPTQ quantization, and measures actual metrics.
#    It includes compatibility shims for auto_gptq 0.7.x + transformers >= 4.52:
#      - _patch_attention_type: delegates attention_type through LayerHijacker
#      - _patch_qwen2_decoder_tuple_return: wraps Qwen2DecoderLayer.forward to
#        return a tuple so auto_gptq's layer(...)[0] doesn't slice the batch dim
#      - _patch_gptq_cholesky_resilience: nan_to_num on add_batch + escalating
#        damping for near-singular Hessians
python scripts/run_stage8_real.py \
  --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --checkpoint ./output/stage5/qwen_lora_gpu/final_checkpoint \
  --output-dir ./output/stage8 \
  --methods gptq \
  --bits 4 \
  --calib-dataset output/stage3/train.jsonl \
  --skip-eval

# 5. Re-run best config selection on a saved QuantReport without re-quantizing
python -m app.evaluation.cli stage8 \
  --source-checkpoint ./output/stage5/sft_qlora \
  --dry-run \
  --target-vram-gb 4.0 --target-size-gb 5.0
```

### Stage 8 Real-Run Results (2026-08-20)

GPTQ 4-bit quantization of the Stage 5 LoRA checkpoint (Qwen2.5-Coder-1.5B
base + LoRA adapter merged) on an RTX 4060 Laptop GPU (8 GB VRAM):

| Config | File Size | Measured VRAM | Throughput | Quality (F1) | Time |
|---|---|---|---|---|---|
| GPTQ 4-bit (g=128) | 1.51 GB | 1.10 GB | N/A¹ | — | 852s |

¹ Throughput measurement skipped due to ExLlama kernel not being compiled on
Windows; the model loads and quantizes correctly but CUDA inference kernels
are unavailable in the pre-built auto_gptq wheel.

**Key fix — rotary embedding shape mismatch:** `Qwen2DecoderLayer.forward`
returns a bare `torch.Tensor`, but auto_gptq 0.7.x's quantize loop calls
`layer(...)[0]`, which on a bare tensor slices the batch dimension (dim 0)
instead of unpacking a tuple. This produced 2-D hidden states `[seq_len,
hidden]` instead of `[1, seq_len, hidden]`, causing the attention reshape to
yield wrong head dimensions and a `RuntimeError` at `apply_rotary_pos_emb`.
The fix wraps `Qwen2DecoderLayer.forward` to return `(hidden_states,)` during
quantization only (see `_patch_qwen2_decoder_tuple_return` in the script).

### Stage 8 Modules

| Module | Responsibility |
|---|---|
| `app/schemas/quantization.py` | `QuantMethod`, `QuantReport`, `QuantResult`, `QuantStatus` Pydantic models |
| `app/quantization/config.py` | `QuantConfig`, `GPTQConfig`, `AWQConfig`, `GGUFConfig` dataclasses + heuristic estimators |
| `app/quantization/quantizer.py` | `Quantizer` Protocol, `MockQuantizer`, `quantize_single()`, `select_best_config()`, `run_quantization_matrix()` |
| `app/quantization/export_gptq.py` | `GPTQQuantizer` (AutoGPTQ wrapper) |
| `app/quantization/export_awq.py` | `AWQQuantizer` (AutoAWQ wrapper) |
| `app/quantization/export_gguf.py` | `GGUFQuantizer` (llama.cpp / llama-cpp-python wrapper) |

### Stage 8 Notes

- **Mock & dry-run modes** — no GPU or ML dependencies required. `--mock`
  uses `MockQuantizer` (fully deterministic); `--dry-run` uses heuristic
  estimators for VRAM, size, quality, and throughput.
- **Quality scoring** — `select_best_config()` weights quality (0.6),
  size (0.2), and speed (0.2). Quality heuristics are rough; real quality is
  measured by re-evaluating through Stage 6.
- **GGUF quant types** — GGUF iterates over `Q2_K` through `Q8_0` rather
  than bit-widths.
- **Lazy imports** — `auto_gptq`, `autoawq`, `llama_cpp` are imported inside
  the quantizer classes' methods.
- **GGUF conversion** — when `llama-cpp-python` cannot be pip-installed (e.g.
  AppLocker blocks the C-extension build), `scripts/convert_to_gguf.py`
  converts HF safetensors → GGUF using the standalone `gguf` package. The
  resulting GGUF checkpoint is served by Stage 9's `llama-server` backend.

---

## Stage 9 — Air-Gapped Serving

Stage 9 provides air-gapped serving via a FastAPI app + Typer CLI with four
backend options: `llama.cpp` (GGUF via `llama-cpp-python`), `llama-server`
(GGUF via the bundled `llama-server.exe` subprocess + HTTP API), `Ollama`
(local HTTP API), and `mock` (deterministic, for testing).

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

# 5. Start with a real GGUF checkpoint (from Stage 8 / convert_to_gguf.py)
python -m app.evaluation.cli stage9 serve -m ./output/stage8/qwen2_gguf_f32.gguf --backend llama-server

# 6. Start with Ollama
python -m app.evaluation.cli stage9 serve -m qwen2.5-coder:7b-base-gguf --backend ollama
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
| `app/schemas/serving.py` | `ServeRequest`, `ServeResponse`, `BatchServeRequest`, `BatchServeResponse` Pydantic models |
| `app/serving/config.py` | `ServingConfig` dataclass (backend, model_path, ports, generation params, warnings) |
| `app/serving/backends.py` | `ServingBackend` Protocol, `LlamaCppBackend`, `LlamaServerBackend`, `OllamaBackend`, `MockServingBackend` |
| `app/serving/serve.py` | `VulnerabilityServer` — ties backend to Stage 4 prompt/parser |
| `app/serving/api.py` | `create_app()` FastAPI factory with `/serve`, `/serve/batch`, `/manifest`, `/healthz` |
| `app/serving/cli.py` | Typer `stage9 serve` subcommand (serve / analyze / batch / dry-run modes) |

### Stage 9 Notes

- **Four backends** — `llama-server` (GGUF via bundled `llama-server.exe`
  subprocess + HTTP, the air-gapped default that doesn't require pip-installing
  `llama-cpp-python`), `llama.cpp` (CPU/GPU via GGUF through
  `llama-cpp-python`), `Ollama` (local HTTP API), and `mock`
  (deterministic for testing). All four implement the `ServingBackend`
  Protocol (`generate(prompt) → str` + `model_info` property).
- **Lazy imports** — `llama_cpp` and `httpx` are imported inside the backend
  classes' `_load()` methods.
- **Dry-run mode** — prints config + validation warnings without starting a
  server or backend.
- **Analyze / batch modes** — run the server's pipeline on a JSON file
  without starting uvicorn. Useful for CI or one-off batch processing.
- **GGUF conversion** — `scripts/convert_to_gguf.py` converts a HuggingFace
  Qwen2 safetensors checkpoint to GGUF format using the standalone `gguf`
  Python package (works even when `llama-cpp-python` cannot be pip-installed
  due to AppLocker policies).
- **Real-serving script** — `scripts/run_stage9_serve.py` starts
  `llama-server.exe` with a GGUF checkpoint, sends a real vulnerability-
  analysis request via HTTP `/completion`, parses the model's JSON response,
  and saves results to `output/stage9/serve_result.json`.

---

## Stage 10 — CI/CD & Regression Gate

Stage 10 is the CI/CD pipeline that gates every push with lint, security
scan, and automated tests. The workflow is defined at
`.github/workflows/ci.yml`.

### Current Coverage

| Check | Tool | Status |
|---|---|---|
| Lint | `ruff check .` | ✅ Implemented |
| Security scan | `bandit -r app -q` | ✅ Implemented |
| Unit tests | `pytest tests/unit --cov=app --cov-report=term-missing` | ✅ Implemented (99% coverage) |
| Integration tests (Stages 4–11) | `pytest tests/integration -k "stage4 or stage5 or stage6 or stage7 or stage8 or stage9 or stage10 or stage11"` | ✅ Implemented |
| **Eval gate** — regression gate on CWE Macro-F1 / forgetting | `python -m app.evaluation.cli stage10` | ✅ Implemented |
| Gitleaks (secret scanning) | `gitleaks/gitleaks-action@v2` (full git history) | ✅ Implemented |
| Trivy (vuln + config + secret scanning) | `aquasecurity/trivy-action` (`scan-type: fs`, `severity: CRITICAL,HIGH`) | ✅ Implemented |

The workflow (`.github/workflows/ci.yml`) is a 4-job pipeline:

```yaml
# .github/workflows/ci.yml — four-job pipeline
# Runs on: push, pull_request
# Python: 3.11
# Install: pip install -e ".[dev,data,ml]"
#
# test — ruff, bandit, unit tests, integration tests for all stages
# eval-gate (needs: test) — Stage 4→6→7→10 mock-mode pipeline + regression gate
# gitleaks (needs: test) — secret scan on full git history
# trivy (needs: test) — filesystem scan: vuln + misconfig + secret, CRITICAL/HIGH severity only
```

### Stage 10 Modules

| Module | Description |
|---|---|
| `app/ci/config.py` | `RegressionGateConfig` — frozen dataclass with artifact paths and thresholds |
| `app/ci/gate.py` | `RegressionGate` class, `run_gate()` convenience function, and artifact loaders |
| `app/ci/security_scanners.py` | `parse_gitleaks_output()`, `parse_trivy_output()` — defensive JSON parsers |
| `app/schemas/ci.py` | `GateStatus`, `GateCheck`, `RegressionGateResult`, `SecurityScanSummary`, `CiReport` |
| `.github/workflows/ci.yml` | 4-job workflow: `test`, `eval-gate`, `gitleaks`, `trivy` |
| `.gitleaks.toml` | Gitleaks config with allowlist for test fixtures |

### Quick Start

```bash
# Run the regression gate locally with mock artifacts:
pip install -e ".[dev,data,ml]"

# Stage 4 baseline (mock, deterministic)
python -m app.evaluation.cli baseline \
  --gold-eval eval/gold_set/gold.jsonl --strategy zero_shot --mock \
  --output-dir ./output/stage4_baseline

# Stage 6 eval (mock sandbox)
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions ./output/stage4_baseline/predictions.jsonl \
  --sandbox-mode mock --skip-tier4 \
  --output-dir ./output/stage6

# Stage 7 regression (mock)
python -m app.evaluation.cli stage7 --mock \
  --base-model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --tuned-model "ci-checkpoint" \
  --output-dir ./output/stage7

# Stage 10 gate — passes if F1 drop ≤5%, forgetting ≥-0.10, exec ≥0.0, halluc ≤0.50
python -m app.evaluation.cli stage10 \
  --baseline-metrics ./output/stage4_baseline/metrics.json \
  --predictions ./output/stage4_baseline/predictions.jsonl \
  --stage6-report ./output/stage6/eval_report.json \
  --stage7-report ./output/stage7/regression_report.json \
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
| README.md (this file) | ✅ Complete |
| Model card (`docs/model_card.md`) | ✅ Complete |
| Training report (`docs/training_report.md`) | ✅ Complete |
| Demo script (`docs/demo.py`) | ✅ Complete |
| Mock evaluation dashboard (`output/mock_eval_dashboard.html`) | ✅ Complete |

### Generating the Deliverables

Stage 11 is implemented as a documentation generator that works entirely in
mock mode (no GPU, no model download, no Docker required):

```bash
# Generate all three deliverables (model card, training report, demo script)
python -m app.evaluation.cli stage11 --docs-dir docs --output-dir ./output/stage11

# Optionally run the mock-mode demo pipeline (Stages 4 -> 6 -> 7 -> 10)
# to populate the documents with real evaluation numbers
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

### Deliverable Descriptions

1. **Model card** (`docs/model_card.md`) — A short, human-readable document
   accompanying the released model checkpoint. It describes the model's
   intended use, training data, evaluation results, known limitations, and
   ethical considerations. Follows the
   [Hugging Face model card format](https://huggingface.co/docs/hub/model-cards).

2. **Training report** (`docs/training_report.md`) — A detailed technical
   report recording the training methodology, hyperparameters, loss curves,
   evaluation results (Stages 4/6/7), quantization trade-offs (Stage 8),
   regression gate results (Stage 10), and conclusions & recommendations.

3. **Demo script** (`docs/demo.py`) — A self-contained, runnable demo that
   exercises the full mock-mode pipeline (Stages 4 → 6 → 7 → 10) on the
   gold-eval set:

   ```bash
   python docs/demo.py
   python docs/demo.py --gold-eval eval/gold_set/gold.jsonl --verbose
   ```

### Stage 11 Modules

| Module | Description |
|---|---|
| `app/schemas/documentation.py` | Pydantic contracts (`ModelCardData`, `TrainingReportData`, `EvalMetricsSnapshot`, `TrainingRunData`, `QuantResultData`, `DemoResult`) and project constants (`CWE_SCOPE`, `BASE_MODEL`, `TRAINING_METHODS`, `LANGUAGE_SCOPE`) |
| `app/stage11/config.py` | `Stage11Config`, a frozen dataclass with README defaults |
| `app/stage11/generator.py` | `Stage11Generator` class — creates and validates deliverables, plus markdown rendering functions and the demo script template |
| `app/evaluation/cli.py` | The `stage11` Typer CLI subcommand |

---

## Testing

The test suite has **1449 tests** (1290 unit + 159 integration), is ruff-clean,
Bandit-clean, and achieves **99% code coverage** (5208 statements, 1 line
uncovered). All tests run in mock/dry-run mode — no GPU, Docker, or network
required. (The exact count varies slightly by environment depending on which
optional extras are installed; in a clean `.[dev,data,ml]` install it is 1449.)

```bash
# Full suite (recommended)
pytest tests/ -v

# Unit tests only (fast, no ML deps needed)
pytest tests/unit -v --cov=app --cov-report=term-missing

# Integration tests only (Stage 4–11, mock mode)
pytest tests/integration -v

# Per-stage focused runs
pytest tests/unit/test_tier1_deterministic.py        # Stage 6 Tier 1
pytest tests/integration/test_stage4_baseline.py     # Stage 4 baseline
pytest tests/integration/test_stage6_four_tier.py    # Stage 6 full pipeline
pytest tests/integration/test_stage7_regression.py   # Stage 7 regression
pytest tests/integration/test_stage8_quantization.py # Stage 8 quant
pytest tests/integration/test_stage9_serving.py      # Stage 9 serving
pytest tests/integration/test_stage10_ci.py         # Stage 10 CI gate
pytest tests/integration/test_stage11_docs.py        # Stage 11 docs

# Linting & security
ruff check .
bandit -r app -q
trivy fs --skip-dirs .venv,output --severity CRITICAL,HIGH .  # requires: trivy install (brew/ports/apt)
```

### Test Structure

| Directory | Contents |
|---|---|
| `tests/unit/` | One file per module — **50 unit test files** covering all 11 stages |
| `tests/integration/` | One file per stage — end-to-end pipeline tests in mock mode |

### End-to-End Mock Pipeline & Evaluation Dashboard

The full pipeline (Stages 4 → 6 → 7 → 10 → 11) can be run end-to-end in
**mock mode** — no GPU, model download, or Docker required. A representative
mock run was executed successfully and results are saved as a visual
dashboard:

```bash
# Run the mock pipeline end-to-end (no GPU / no model download)
python docs/demo.py --verbose

# Or run each stage individually:
python -m app.evaluation.cli baseline --mock \
  --gold-eval eval/gold_set/gold.jsonl --strategy zero_shot \
  --output-dir ./output/stage4
python -m app.evaluation.cli stage6 \
  --gold-eval eval/gold_set/gold.jsonl \
  --predictions ./output/stage4/predictions.jsonl \
  --sandbox-mode mock --skip-tier4 \
  --output-dir ./output/stage6
python -m app.evaluation.cli stage7 --mock \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --tuned-model ci-checkpoint --output-dir ./output/stage7
python -m app.evaluation.cli stage10 \
  --baseline-metrics ./output/stage4/metrics.json \
  --predictions ./output/stage4/predictions.jsonl \
  --stage6-report ./output/stage6/eval_report.json \
  --stage7-report ./output/stage7/regression_report.json \
  --output-dir ./output/stage10
```

**Mock run results** (see `output/mock_eval_dashboard.html` for the interactive
dashboard) — mock results shown alongside real runs where available:

| Stage | Mock Result | Real Result |
|---|---|---|
| Stage 4 — baseline (MockBackend) | CWE Macro-F1: 0.0476, 0 hallucinations, 100% patch coverage (12 gold samples) | CWE Macro-F1: 0.1667 (1.5B QLoRA r=8, GPU, 59 gold samples) |
| Stage 6 — Tier 1 (deterministic) | CWE Macro-F1: 1.0000, Coverage: 1.0000 (12 gold samples) | CWE Macro-F1: 0.5019, Coverage: 0.3729 (59 gold samples) |
| Stage 6 — Tier 2 (static+Semgrep) | CWE Macro-F1: 1.0000, Coverage: 1.0000 (12 gold samples) | CWE Macro-F1: 0.3980, Coverage: 0.2034 (59 gold samples) |
| Stage 6 — Tier 3 (exec sandbox) | 100% patches apply, 0% exec pass (mock backend) | 0% patches apply, 0% exec pass (1.5B QLoRA, Docker sandbox, 59 gold samples) |
| Stage 7 — regression | Forgetting delta: +0.0000 (no forgetting) | Forgetting delta: +0.0000 (no forgetting) |
| Stage 8 — quantization | GPTQ/AWQ/GGUF 8 configs simulated (Q4 best: F1≈0.92, 6.5 GB) | **Real run 2026-08-20** — GPTQ 4-bit: 1.51 GB, 1.10 GB VRAM, 852s (all 28 layers quantized successfully) |
| Stage 10 — gate | ✅ **PASS** — all 4 checks passed | ✅ **PASS** — all 4 checks passed |

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
5. **Scan for vulns:** `make scan` (or `trivy fs --skip-dirs .venv,output --severity CRITICAL,HIGH .`)
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