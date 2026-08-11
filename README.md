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
✅ **Stage 4 — pre-fine-tuning baseline.** Zero-shot and few-shot evaluation of
the base Qwen2.5-Coder-7B-Instruct model on the gold-eval set, with CWE
Macro-F1, severity accuracy, hallucination rate, and patch coverage metrics.
Fully implemented and tested.
✅ **Stage 5 — training matrix.** SFT (full-parameter + QLoRA), LoRA rank
sweep, and DPO preference alignment. All modes support `--dry-run` (no GPU).
✅ **Stage 6 — four-tier evaluation harness.** Deterministic (Tier 1) →
static+embedding (Tier 2) → exec sandbox (Tier 3) → LLM-judge (Tier 4).
✅ **Stage 7 — regression / forgetting analysis.** General code-capability
delta (tuned vs. base) on HumanEval-style tasks.
✅ **Stage 8 — quantization matrix.** GPTQ / AWQ / GGUF with quality-vs-VRAM
trade-off scoring. Mock and dry-run modes supported.
✅ **Stage 9 — air-gapped serving.** llama.cpp / Ollama / mock backends behind
a FastAPI service + Typer CLI (serve / analyze / batch / dry-run modes).
✅ **Stage 10 — CI/CD & regression gate.** GitHub Actions workflow with
ruff, Bandit, pytest, eval gate (Stage 4→6→7→10 mock pipeline), Gitleaks
(secret scanning), and Trivy (vuln + config scanning).
🔄 **Stage 11 — documentation & interview package.** README complete; model
card and training report not yet written.

> **Test suite:** 807 tests pass (unit + integration), ruff clean, Bandit clean.

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
STAGE 10 CI/CD & regression gate             🔄 Partial (ruff/Bandit/pytest only; no eval gate yet)
STAGE 11 Documentation & interview package   🔄 Partial (README done; model card + report pending)
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
│   │   ├── vuln.py            # VulnSample
│   │   ├── dataset.py         # InstructionExample
│   │   ├── prediction_eval.py # ModelPrediction, EvalMetrics, RegressionReport
│   │   ├── training.py        # TrainingResult, SweepReport
│   │   ├── quantization.py    # QuantReport, QuantResult
│   │   ├── serving.py         # ServeRequest, ServeResponse, BatchServeResponse
│   │   └── __init__.py
│   ├── data/
│   │   ├── collectors/   # CVEfixes/BigVul/NVD/OSV downloaders + Semgrep      (Stage 1)
│   │   ├── cleaning/     # dedup, leakage-safe split, contamination check     (Stage 2)
│   │   └── formatting/   # instruction-format dataset builder, token counter   (Stage 3)
│   ├── training/         # sft/qlora/lora-sweep/dpo trainers, CLI              (Stage 5)
│   ├── evaluation/       # tier1→tier4 evaluators, baseline, regression        (Stage 4-6-7)
│   ├── quantization/     # GPTQ/AWQ/GGUF quantizers, matrix runner, CLI        (Stage 8)
│   ├── serving/          # FastAPI app, Typer CLI, backends, config             (Stage 9)
│   └── storage/          # Postgres models, MinIO client
├── eval/gold_set/        # 12 manually verified gold-eval examples (2 per CWE)
├── sandbox/              # per-language Docker images for exec-based eval
├── tests/{unit,integration}/   # 873 tests total, ruff clean
├── .github/workflows/ci.yml    # ruff, Bandit, pytest, eval-gate, Gitleaks, Trivy
├── docker-compose.yml    # Postgres + Redis + MinIO
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

## Stage 4 Quick Start

Stage 4 evaluates the **base** (pre-fine-tuning) model on the gold-eval set
to establish a "before" baseline. It supports two prompting strategies:

- **Zero-shot** — the model classifies the vulnerability with no examples.
- **Few-shot** — N in-context examples from Stage 3 output are prepended.

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

# 5. Run unit + integration tests for Stage 4
pytest tests/unit/test_evaluation_parser.py tests/unit/test_evaluation_metrics.py \
       tests/unit/test_evaluation_prompt.py tests/unit/test_evaluation_backends.py \
       tests/integration/test_stage4_baseline.py -v
```

### Output files

`output/stage4/` contains:

| File | Contents |
|---|---|
| `predictions.jsonl` | One `ModelPrediction` per line (sample_id, run_id, predicted_cwe, predicted_severity, suggested_patch_diff, rationale) |
| `metrics.json` | Aggregate metrics (CWE Macro-F1, micro accuracy, severity accuracy, hallucination rate, patch coverage, per-class F1) |
| `manifest.json` | Run provenance (stage, strategy, base_model, num_gold_samples, num_predictions, run_id) |
| `parse_errors.jsonl` | Samples whose model output could not be parsed (one `ParseError` per line) |

### Stage 4 modules

| Module | Responsibility |
|---|---|
| `app/evaluation/backends.py` | `ModelBackend` Protocol + `QwenBackend` (lazy-loaded transformers) + `MockBackend` for testing |
| `app/evaluation/prompt.py` | `build_zero_shot_prompt()` and `build_few_shot_prompt()` using Stage 3's `format_prompt` |
| `app/evaluation/parser.py` | `parse_prediction()` — extracts JSON from model output (markdown fences + brace-matching fallback) |
| `app/evaluation/metrics.py` | CWE Macro-F1, micro accuracy, severity accuracy, hallucination rate, patch coverage |
| `app/evaluation/baseline.py` | `BaselineConfig` + `BaselineResult` + `run_baseline()` orchestration (load → prompt → generate → parse → metrics → write) |
| `app/evaluation/cli.py` | Typer CLI with `baseline` and `evaluate` subcommands |
| `eval/gold_set/gold.jsonl` | 12 manually-verified gold-eval examples (2 per CWE class × 6 classes) |

### Stage 4 notes

- **No model download required for tests.** The test suite uses `MockBackend`,
  which returns deterministic fake predictions — no GPU or network needed.
  The `QwenBackend` is only instantiated when no `--mock` flag is passed to
  the CLI and `transformers` is installed.
- **CWE scope**: the 6 target classes (CWE-89, CWE-79, CWE-22, CWE-78,
  CWE-190, CWE-502) are enforced in the parser and metrics. Out-of-scope CWE
  IDs (e.g. `CWE-999`) are counted as **hallucinations**, not just wrong
  predictions.
- **Few-shot fallback**: if `--strategy few-shot` is selected but no
  `--few-shot-examples` file is provided, the runner automatically falls back
  to zero-shot mode (logged as a warning).
- **Gold-eval set**: 12 samples (2 per CWE class) for fast, reproducible
  baseline evaluation. This is the "small, manually-verifiable eval set"
  described in the architecture diagram.

## Stage 5 Quick Start

Stage 5 implements the full training matrix: SFT (full-parameter and QLoRA),
LoRA rank sweep, and DPO preference alignment. It uses
Qwen2.5-Coder-7B-Instruct as the base model, with PEFT/LoRA/QLoRA (bitsandbytes
4-bit NF4) for parameter-efficient training, TRL's `DPOTrainer` for preference
optimization, and W&B for loss-curve tracking.

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
from app.schemas.vuln import VulnSample
from app.evaluation.baseline import load_gold_eval
import uuid, json

class _MockTok:
    def encode(self, text): return list(range(max(len(text), 1)))

counter = TokenCounter(tokenizer=_MockTok())
samples = load_gold_eval('eval/gold_set/gold.jsonl')
# Split into train/val
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

### LoRA rank sweep

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

### DPO preference alignment

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

### Inspecting runs

Training metadata is persisted to PostgreSQL (when available):

```bash
# List all training runs
python -m app.training.cli list-runs
python -m app.training.cli list-runs --limit 10 --method sft_qlora --status completed

# Inspect a specific run
python -m app.training.cli inspect --run-id dpo_20240809_120000_a1b2c3d4
```

### Stage 5 modules

| Module | Responsibility |
|---|---|
| `app/training/config.py` | `TrainingMethod` enum, `SFTConfig`, `DPOConfig`, `SweepConfig` dataclasses with defaults from the README tech-stack table |
| `app/training/data.py` | `JsonlDataLoader` (injectable), `load_examples()`, `compute_stats()`, `examples_to_dict_list()`, `make_hf_dataset()` (lazy `datasets` import) |
| `app/training/callbacks.py` | `TrainingCallback` Protocol, `WandbCallback` (mock mode), `CheckpointCallback` (MinIO upload), `ProgressCallback`, `ResourceTracker` (peak VRAM) |
| `app/training/experiment.py` | `persist_training_run()`, `load_training_run()`, `list_training_runs()` (PostgreSQL via SQLAlchemy), `generate_run_id()` |
| `app/training/trainer_sft.py` | `run_sft()` (full + QLoRA), `estimate_training_steps()` (pure arithmetic), `TrainingUnavailableError`, 4-bit NF4 model loading |
| `app/training/trainer_dpo.py` | `run_dpo()` with TRL `DPOTrainer`, `estimate_dpo_steps()`, `build_preference_pairs()`, synthetic rejected-response generation |
| `app/training/sweep.py` | `run_lora_sweep()` — orchestrates multiple `run_sft` calls across ranks, `SweepReport` summary |
| `app/training/cli.py` | Typer CLI: `sft`, `lora-sweep`, `dpo`, `list-runs`, `inspect` subcommands |

### Stage 5 notes

- **No GPU needed for development**. All training modes support `--dry-run`,
  which loads Stage 3 JSONL data, estimates training steps and VRAM, and
  returns a `TrainingResult`. The real training path (`_run_sft` / `_run_dpo`)
  is gated behind `_check_can_train()`, which raises `TrainingUnavailableError`
  if `torch`/`transformers`/`trl` are missing or no CUDA GPU is detected.
- **Lazy ML imports**. Heavy dependencies (`torch`, `transformers`, `peft`,
  `bitsandbytes`, `trl`, `datasets`, `wandb`) are imported inside functions,
  never at module level. This follows the same pattern as Stage 4's
  `QwenBackend` and Stage 3's `TokenCounter`.
- **Injectable backends for testing**. The `loader` parameter on `run_sft`,
  `run_dpo`, and `run_lora_sweep` accepts any object implementing the
  `DataLoadable` Protocol, so tests can inject pre-built `InstructionExample`
  lists without touching the filesystem. Callbacks are also injectable.
- **QLoRA defaults**. By default, SFT uses 4-bit NF4 quantization via
  `bitsandbytes` (with `bnb_4bit_use_double_quant=True`) so a 7B model fits in
  8 GB VRAM. Full-parameter SFT (`--no-4bit`) is available as a baseline but
  requires ~16 GB.
- **LoRA rank range**. The sweep tests ranks `[8, 16, 32, 64, 128]`, bracketing
  the "useful parameter-efficient range" from the QLoRA paper (Dettmers et al.,
  2023, arXiv:2305.14168). Rank 8 gives the smallest adapter; rank 128 is
  closest to full fine-tuning quality.
- **PostgreSQL tracking**. When `persist=True` (the default), each completed
  run is written to the `training_runs` table via SQLAlchemy. Use
  `--no-persist` in dry-run mode to skip DB writes. Runs are retrievable via
  `list-runs` / `inspect` for Stage 6 (evaluation) to find the best checkpoint.
- **Experiment tracking via W&B**. When `wandb` is installed and W&B is
  configured, `WandbCallback` logs loss curves in real training mode. In mock
  mode (default in tests), it stores calls in memory. The `--dry-run` path does
  not touch W&B.
- **Checkpoint storage**. `CheckpointCallback` uploads model adapters to MinIO
  under `s3://vuln-triage/checkpoints/stage5/{run_id}/epoch_N`. In mock mode,
  the S3 URI is returned without an upload — useful for testing the wiring.

### Stage 5 test suite

```bash
# Unit tests (no GPU, no ML deps)
pytest tests/unit/test_training_config.py \
       tests/unit/test_training_data.py \
       tests/unit/test_training_callbacks.py \
       tests/unit/test_training_trainer_sft.py \
       tests/unit/test_training_trainer_dpo.py \
       tests/unit/test_training_sweep.py \
       tests/unit/test_training_experiment.py -v

# Integration tests (dry-run end-to-end, CLI via CliRunner)
pytest tests/integration/test_stage5_training.py -v
```

## Stage 6 Quick Start

Stage 6 implements the **four-tier evaluation harness** that validates model
predictions across multiple dimensions:

```
         ┌────────────────────────────────────────────────────────┐
         │  Four-tier evaluation harness (Stage 6)                │
         │                                                        │
  Gold   │  Tier 1: deterministic regex classifier (CWE only)   │
  Eval → │  → Tier 2: static Semgrep findings + embedding      │
  Sample │  → Tier 3: exec — apply patch, run tests in sandbox  │
  +      │  → Tier 4: LLM-judge — explanation quality/minimality │
  Model  └────────────────────────────────────────────────────────┘
```

Input: gold-eval samples (`VulnSample`) + model predictions (`ModelPrediction`).
Output: `EvalReport` with per-tier results, aggregate `EvalMetrics`, and a
run manifest.

### CLI

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

### Programmatic use

```python
from app.evaluation.runner import EvalConfig, EvaluationRunner, load_samples, load_predictions

config = EvalConfig(
    base_model="Qwen2.5-Coder-7B-Instruct",
    sandbox_mode="mock",      # or "local" for subprocess
    skip_tier4=True,          # disable LLM judge to save cost
)
runner = EvaluationRunner(config=config)

samples = load_samples("eval/gold_set/gold.jsonl")
preds   = load_predictions("output/stage6/predictions.jsonl")

report = runner.run(samples, preds)
print(f"Model Macro-F1: {report.metrics.model_cwe_macro_f1:.4f}")
print(f"Exec Pass Rate: {report.metrics.exec_pass_rate:.4f}")
```

### Stage 6 modules

| Module | Responsibility |
|---|---|
| `app/schemas/prediction_eval.py` | `Tier1Result`, `Tier2Result`, `ExecEvalResult`, `LlmJudgeScore`, `EvalMetrics`, `EvalReport`, `RegressionSummary` Pydantic models |
| `app/evaluation/tier1_deterministic.py` | `PatternRule` dataclass, `DEFAULT_TIER1_RULES` (20 regex rules for all 6 CWEs), `DeterministicEvaluator` with `evaluate()`/`evaluate_all()` |
| `app/evaluation/tier2_embedding_static.py` | `DEFAULT_RULE_TO_CWE` (20 rule IDs → CWE), `EmbeddingBackend` (lazy `sentence-transformers` import), `StaticSignalEvaluator` |
| `app/evaluation/tier3_exec.py` | `SandboxRunner` Protocol, `LocalSandboxRunner`, `MockSandboxRunner`, `ExecEvaluator`, `apply_unified_diff()`, `TestGenerator` (per-CWE test templates), `check_hallucinated_function_ref()` |
| `app/evaluation/tier4_llm_judge.py` | `LlmJudgeBackend` Protocol, `LlmJudge`, `MockLlmJudgeBackend`, judge prompt for explanation quality + patch minimality |
| `app/evaluation/runner.py` | `EvalConfig`, `EvaluationRunner` (orchestrates all 4 tiers), `compute_metrics()`, `load_samples()` / `load_predictions()` I/O helpers |
| `app/evaluation/cli.py` | Typer `stage6` subcommand (`--gold-eval`, `--predictions`, `--sandbox-mode`, `--skip-tier3`, `--skip-tier4`, `--embedding-model`) |

### How the four tiers work

1. **Tier 1 — Deterministic baseline.** Pure-Python regex rules (no model,
   no Semgrep, no Docker). Achieves 12/12 on the gold eval set. This is the
   floor: any model must beat it.

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

### Stage 6 notes

- **No GPU or model download required for tests.** All tiers use mock
  backends in the test suite — `MockSandboxRunner` returns canned results,
  `MockLlmJudgeBackend` returns fixed scores, and `sentence-transformers`
  is an optional lazy import.
- **Leakage-safe.** Tier 3 runs in an isolated temp directory; the vulnerable
  code is never executed from the repo workspace. For production CI, pass
  `--sandbox-mode docker` to use Docker isolation (see `sandbox/` directory).
- **CWE scope.** The 6 target classes (CWE-89, CWE-79, CWE-22, CWE-78,
  CWE-190, CWE-502) are enforced. Predictions with out-of-scope CWE IDs
  are counted as **hallucinations**.
- **Patch applier.** `apply_unified_diff()` is a pure-Python implementation
  — no dependency on `git apply` or the `patch` command. It validates context
  lines before applying hunks and returns an error message on mismatch.
- **Hallucination detection.** Tier 3 checks both CWE ID validity (must be
  in the 6-class scope) and function-reference hallucination (patch references
  identifiers not present in the vulnerable code).

### Stage 6 test suite

```bash
# Unit tests (one file per tier)
pytest tests/unit/test_tier1_deterministic.py \
       tests/unit/test_tier2_embedding_static.py \
       tests/unit/test_tier3_exec.py \
       tests/unit/test_tier4_llm_judge.py -v

# Integration test (full pipeline end-to-end with gold-eval set)
pytest tests/integration/test_stage6_four_tier.py -v
```

## Stage 7 Quick Start

Stage 7 implements **regression / forgetting analysis** — the "after" half of
the before/after comparison. After fine-tuning (Stage 5) and evaluating on
security tasks (Stage 6), the tuned model is re-evaluated on a set of
general-purpose (non-security) code-generation tasks. The **forgetting delta**
measures whether general coding ability was lost during fine-tuning:

```
delta = tuned_exec_accuracy − base_exec_accuracy
```

A *negative* delta means the fine-tuned model suffered catastrophic
forgetting — it got good at vulnerability tasks but lost general coding
ability. A *positive* delta means the fine-tuned model improved general
coding. Zero means no net change.

### CLI

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

### Programmatic use

```python
from app.evaluation.general_capability import (
    RegressionConfig,
    run_regression_analysis,
    build_regression_summary,
)
from app.evaluation.backends import MockBackend
from app.schemas.prediction_eval import EvalMetrics

# Configure
config = RegressionConfig(
    base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
    tuned_model="sft_qlora_r8",
)

# Run forgetting analysis (use MockBackend + MockCodeTestRunner for tests)
report = run_regression_analysis(
    config=config,
    base_backend=MockBackend(default="pass"),
    tuned_backend=MockBackend(default="pass"),
)

# Combine with Stage 6 metrics into a single summary row
summary = build_regression_summary(
    run_id="checkpoint_001",
    stage6_metrics=EvalMetrics(...),  # from Stage 6
    regression_report=report,
    inference_cost_usd=6.0,
    training_cost_usd=4.0,
)
```

### Output files

`output/stage7/` contains:

| File | Contents |
|---|---|
| `regression_report.json` | Full `RegressionReport` — base/tuned metrics, forgetting delta, manifest |

### Stage 7 modules

| Module | Responsibility |
|---|---|
| `app/evaluation/general_capability.py` | 12 HumanEval-style tasks, `GeneralCapabilityTask`, `CodeTestRunner` Protocol, `LocalCodeTestRunner` (subprocess pytest), `MockCodeTestRunner`, `GeneralCapabilityEvaluator`, `RegressionConfig`, `run_regression_analysis()`, `build_regression_summary()`, `estimate_cost_per_accepted_patch_usd()` |
| `app/evaluation/cli.py` | Typer `stage7` subcommand (`--base-model`, `--tuned-model`, `--mock`, `--timeout`, `--output-dir`) |
| `app/schemas/prediction_eval.py` | `GeneralCapabilityResult`, `GeneralCapabilityMetrics`, `RegressionReport`, `RegressionSummary` Pydantic models |

### Stage 7 notes

- **No GPU or model download required for tests.** The test suite uses
  `MockBackend` + `MockCodeTestRunner` (deterministic, no subprocess). For
  tests that *do* exercise real code execution, `LocalCodeTestRunner` spawns
  isolated `python -m pytest` subprocesses — no Docker needed.
- **Lazy ML imports.** Heavy dependencies (`torch`, `transformers`,
  `sentence-transformers`) are imported inside functions, never at module
  level. The `QwenBackend` is only instantiated in real mode (no `--mock`).
- **Injectable backend pattern.** Both `ModelBackend` (code generation) and
  `CodeTestRunner` (code execution) are injectable Protocols, so every code
  path is testable without model downloads.
- **12 default tasks.** `DEFAULT_GENERAL_TASKS` covers factorial, palindrome,
  fibonacci, binary search, two-sum, vowel counting, integer reversal,
  anagram, longest common prefix, valid parentheses, remove duplicates, and
  max subarray sum — all pure-Python with no external dependencies.
- **Security.** `LocalCodeTestRunner` uses `subprocess` with the same
  `# nosec B603` pattern as `tier3_exec.py`. Inputs are trusted (system
  `sys.executable` + temp file paths). For untrusted code, use Docker
  isolation (see `sandbox/`).
- **Forgetting delta = `tuned_acc − base_acc`**. Negative = forgetting,
  positive = improvement. This value feeds into `RegressionSummary`, the
  primary output consumed by the Stage 10 regression gate.

### Stage 7 test suite

```bash
# Unit tests
pytest tests/unit/test_general_capability.py -v

# Integration tests (mock mode, local subprocess, CLI, RegressionSummary)
pytest tests/integration/test_stage7_regression.py -v

# With Stage 6 tests for full pipeline
pytest tests/unit/test_general_capability.py tests/integration/test_stage7_regression.py -v
```

## Stage 8 Quick Start

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

# 3. Real quantization (requires GPU + torch + auto-gptq/autoawq/llama-cpp-python)
python -m app.evaluation.cli stage8 \
  --source-checkpoint ./output/stage5/sft_qlora \
  --methods gptq,gguf \
  --bits 4 \
  --output-dir ./output/stage8

# 4. Re-run best config selection on a saved QuantReport without re-quantizing
python -m app.evaluation.cli stage8 \
  --source-checkpoint ./output/stage5/sft_qlora \
  --dry-run \
  --target-vram-gb 4.0 --target-size-gb 5.0
```

### Stage 8 modules

| Module | Responsibility |
|---|---|
| `app/schemas/quantization.py` | `QuantMethod`, `QuantReport`, `QuantResult`, `QuantStatus` Pydantic models |
| `app/quantization/config.py` | `QuantConfig`, `GPTQConfig`, `AWQConfig`, `GGUFConfig` dataclasses + heuristic estimators |
| `app/quantization/quantizer.py` | `Quantizer` Protocol, `MockQuantizer`, `quantize_single()`, `select_best_config()`, `run_quantization_matrix()` |
| `app/quantization/export_gptq.py` | `GPTQQuantizer` (AutoGPTQ wrapper) |
| `app/quantization/export_awq.py` | `AWQQuantizer` (AutoAWQ wrapper) |
| `app/quantization/export_gguf.py` | `GGUFQuantizer` (llama.cpp / llama-cpp-python wrapper) |
| `app/evaluation/cli.py` | Typer `stage8` subcommand registered on the shared CLI app |

### Stage 8 notes

- **Mock & dry-run modes** — no GPU or ML dependencies required. `--mock`
  uses `MockQuantizer` (fully deterministic); `--dry-run` uses heuristic
  estimators for VRAM, size, quality, and throughput. Real quantization is
  only attempted when neither flag is set and the method-specific library
  (`auto_gptq`, `autoawq`, or `llama-cpp-python`) is importable.
- **Quality scoring** — `select_best_config()` weights quality (0.6),
  size (0.2), and speed (0.2). Quality heuristics are rough; real quality is
  measured by re-evaluating the quantized checkpoint through Stage 6.
- **GGUF quant types** — GGUF iterates over `Q2_K` through `Q8_0` rather
  than bit-widths, since each type has a different bytes-per-parameter ratio.
- **Lazy imports** — `auto_gptq`, `autoawq`, `llama_cpp` are imported inside
  the quantizer classes' methods, never at module level.

### Stage 8 test suite

```bash
# Unit tests
pytest tests/unit/test_quantization.py -v

# Integration tests (mock matrix + CLI)
pytest tests/integration/test_stage8_quantization.py -v
```

## Stage 9 Quick Start

Stage 9 provides air-gapped serving via a FastAPI app + Typer CLI with three
backend options: `llama.cpp` (GGUF via `llama-cpp-python`), `Ollama`
(local HTTP API), and `mock` (deterministic, for testing).

```bash
# 1. Dry-run — print config and warnings without loading a model
python -m app.evaluation.cli stage9 serve --dry-run --backend mock

# 2. Analyze a single sample from a JSON file (no server needed)
echo '{"vulnerable_code": "cursor.execute(\"SELECT * FROM users WHERE id = \" + user_id)", "language": "python"}' > /tmp/sample.json
python -m app.evaluation.cli stage9 serve --backend mock --analyze -i /tmp/sample.json

#    Output: JSON with predicted_cwe, severity, explanation, patch_diff

# 3. Batch analysis from a JSON array
python -m app.evaluation.cli stage9 serve --backend mock --batch -i /tmp/samples.json -o /tmp/results.json

# 4. Start the FastAPI server (mock backend — no model needed)
python -m app.evaluation.cli stage9 serve --backend mock --host 127.0.0.1 --port 8000

# 5. Start with a real GGUF checkpoint (from Stage 8)
python -m app.evaluation.cli stage9 serve -m ./output/stage8/gguf_bits4/q4_0.gguf --backend llama.cpp

# 6. Start with Ollama
python -m app.evaluation.cli stage9 serve -m qwen2.5-coder:7b-base-gguf --backend ollama
```

### API endpoints

| Method | Path | Body | Description |
|---|---|---|---|
| `POST` | `/api/v1/serve` | `ServeRequest` | Analyze a single vulnerability |
| `POST` | `/api/v1/serve/batch` | `BatchServeRequest` | Analyze multiple samples |
| `GET` | `/api/v1/manifest` | — | Run provenance (run_id, backend, request count) |
| `GET` | `/healthz` | — | Health check |

### Stage 9 modules

| Module | Responsibility |
|---|---|
| `app/schemas/serving.py` | `ServeRequest`, `ServeResponse`, `BatchServeRequest`, `BatchServeResponse` Pydantic models |
| `app/serving/config.py` | `ServingConfig` dataclass (backend, model_path, ports, generation params, warnings) |
| `app/serving/backends.py` | `ServingBackend` Protocol, `LlamaCppBackend`, `OllamaBackend`, `MockServingBackend` |
| `app/serving/serve.py` | `VulnerabilityServer` — ties backend to Stage 4 prompt/parser |
| `app/serving/api.py` | `create_app()` FastAPI factory with `/serve`, `/serve/batch`, `/manifest`, `/healthz` |
| `app/serving/cli.py` | Typer `stage9 serve` subcommand (serve / analyze / batch / dry-run modes) |

### Stage 9 notes

- **Three backends** — `llama.cpp` (CPU/GPU via GGUF, the air-gapped default),
  `Ollama` (local HTTP API), and `mock` (deterministic for testing). All three
  implement the `ServingBackend` Protocol (`generate(prompt) → str` +
  `model_info` property).
- **Lazy imports** — `llama_cpp` and `httpx` are imported inside the backend
  classes' `_load()` methods, never at module import. This makes the CLI and
  API import-safe without those packages installed.
- **Dry-run mode** — prints config + validation warnings without starting a
  server or backend. Use this to verify configuration before loading a model.
- **Analyze / batch modes** — run the server's pipeline on a JSON file
  without starting uvicorn. Useful for CI or one-off batch processing.
- **No Docker dependency** for `local` or `ollama` backends. The `llama.cpp`
  backend uses `llama-cpp-python` directly (no Docker needed). For hardened
  isolation, the `sandbox/` directory contains per-language Docker images
  used by Stage 6's exec eval.

### Stage 9 test suite

```bash
# Unit tests (per-module)
pytest tests/unit/test_serving_backends.py \
       tests/unit/test_serving_config.py \
       tests/unit/test_serving_api.py \
       tests/unit/test_serving_schemas.py \
       tests/unit/test_vulnerability_server.py -v

# Integration test (CLI + end-to-end server)
pytest tests/integration/test_stage9_serving.py -v
```

## Stage 10 Quick Start

Stage 10 is the CI/CD pipeline that gates every push with lint, security
scan, and automated tests. The workflow is defined at
`.github/workflows/ci.yml`.

### Current coverage

| Check | Tool | Status |
|---|---|---|
| Lint | `ruff check .` | ✅ Implemented |
| Security scan | `bandit -r app -q` | ✅ Implemented |
| Unit tests | `pytest tests/unit --cov=app` | ✅ Implemented |
| Integration tests (Stages 4–10) | `pytest tests/integration -k "stage4 or stage5 or stage6 or stage7 or stage8 or stage9 or stage10"` | ✅ Implemented |
| **Eval gate** — regression gate on CWE Macro-F1 / forgetting | `python -m app.evaluation.cli stage10` | ✅ Implemented |
| Gitleaks (secret scanning) | `gitleaks detect` via GitHub Action | ✅ Implemented |
| Trivy (vuln + config scanning) | `trivy fs` via GitHub Action | ✅ Implemented |

```yaml
# .github/workflows/ci.yml — four-job pipeline
# Runs on: push, pull_request
# Python: 3.11
# Install: pip install -e ".[dev,data,ml]"
#
# test — ruff, bandit, unit tests, integration tests for all stages
# eval-gate (needs: test) — Stage 4→6→7→10 mock-mode pipeline + regression gate
# gitleaks (needs: test) — secret scan on full git history
# trivy (needs: test) — filesystem vulnerability + misconfiguration scan
```

### Stage 10 modules

| Module | Description |
|---|---|
| `app/ci/config.py` | `RegressionGateConfig` — frozen dataclass with artifact paths and thresholds |
| `app/ci/gate.py` | `RegressionGate` class, `run_gate()` convenience function, and artifact loaders |
| `app/ci/security_scanners.py` | `parse_gitleaks_output()`, `parse_trivy_output()` — defensive JSON parsers |
| `app/schemas/ci.py` | `GateStatus`, `GateCheck`, `RegressionGateResult`, `SecurityScanSummary`, `CiReport` |
| `.github/workflows/ci.yml` | 4-job workflow: `test`, `eval-gate`, `gitleaks`, `trivy` |
| `.gitleaks.toml` | Gitleaks config with allowlist for test fixtures |

### Quick start

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

### Gate checks

The regression gate (`app/ci/gate.py`) evaluates four checks:

1. **CWE F1 regression** — `model_cwe_macro_f1` (Stage 6) must not drop more than
   `max_f1_drop_percent` (default 5%) below `cwe_macro_f1` (Stage 4 baseline).
2. **Forgetting** — `forgetting_delta` (Stage 7, `tuned_acc − base_acc`) must
   not fall below `forgetting_threshold` (default -0.10). Skipped if no
   Stage 7 report is provided.
3. **Exec pass rate** — `exec_pass_rate` must meet `min_exec_pass_rate`
   (default 0.0).
4. **Hallucination rate** — must not exceed `max_hallucination_rate` (default 0.50).

### CI workflow

The `eval-gate` job in `.github/workflows/ci.yml` runs the full mock-mode
pipeline end-to-end (Stages 4→6→7→10) on every push and pull request. This
ensures the pipeline math is verified on every commit without requiring a GPU
or Docker. The `gitleaks` and `trivy` jobs run as separate parallel jobs
(wired with `needs: test`) and fail the workflow on any finding.

## Stage 11 Quick Start

Stage 11 is the documentation & interview deliverables.

### Current deliverables

| Deliverable | Status |
|---|---|
| README.md (this file) | ✅ Complete |
| Model card (`docs/model_card.md`) | 🔄 Not started |
| Training report (`docs/training_report.md`) | 🔄 Not started |
| Demo script / notebook | 🔄 Not started |

### Out of scope (stated explicitly, not claimed)

- **General-purpose vulnerability scanner.** This harness targets the 6 CWE
  classes in scope (CWE-89, CWE-79, CWE-22, CWE-78, CWE-190, CWE-502). It does
  not claim to detect logic bugs, configuration issues, or CWE classes outside
  the listed scope.
- **Real-time scanning.** The serving layer (Stage 9) is for interactive /
  batch vulnerability analysis of isolated code snippets, not for continuous
  monitoring of repositories in CI.
- **Supply-chain security.** This project does not audit third-party packages
  or perform dependency-graph analysis. Use `pip-audit` / `Safety` for that.
- **Network-based scanning.** No network port scanning, no HTTP fuzzing, no
  live system exploitation. All evaluation is offline against curated CVE data.
- **Legal / compliance assessment.** The model classifies CWE IDs and proposes
  patches from training data patterns — it does not provide legal opinions on
  liability, regulatory compliance, or licensing.
- **Production incident response.** This is a research / training artifact,
  not a SOC tool. Do not use it as a sole decision-maker for production
  security alerts.

## License

MIT — see [LICENSE](LICENSE).
