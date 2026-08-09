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
Fully implemented and tested: 16 prompt tests, 23 metric tests, 18 parser
tests, 16 backend tests (unit) + 15 end-to-end tests (integration).
All 234 tests pass (150 existing + 84 new), ruff clean.

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
STAGE 4  Pre-fine-tuning baseline            zero-shot / few-shot base model on gold-eval   ✅ Stage 4 done
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
│   ├── evaluation/       # tier1_deterministic.py ... tier4_llm_judge.py   (Stage 4-6-7)
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

## Out of scope (stated explicitly, not claimed)

Full fine-tuning of the 7B model, multi-GPU distributed training, and
quantization of very large models are out of budget for this project on a
single 8GB-VRAM GPU. These are listed as future work, not claimed as done.

## License

MIT — see [LICENSE](LICENSE).
