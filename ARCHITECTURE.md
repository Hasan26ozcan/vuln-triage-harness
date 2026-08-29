# Vuln-Triage-Harness — Complete Architecture

> 11-stage vulnerability triage & fine-tuning pipeline.
> **Core principle:** Every external dependency is a `Protocol` with a mock implementation. Heavy ML deps are lazy-loaded. Zero-dependency at import time.

---

## 📊 1. Overall Pipeline Flow (11 Stages)

```mermaid
flowchart LR
    subgraph "Inputs"
        CVE["CVE / NVD API"]
        SEMGREP["Semgrep Rules<br/>(app/data/collectors/rules/)"]
        GH["GitHub Repos"]
        HF["HuggingFace Hub"]
    end

    subgraph S1["Stage 1: Data Collection<br/>app/data/collectors/"]
        S1A["collect_pipeline.py"] --> S1B["nvd_client.py"]
        S1A --> S1C["semgrep_runner.py"]
        S1A --> S1D["cvefixes_loader.py<br/>+ cvefixes_reduced_loader.py"]
        S1A --> S1E["cwe_scope.py — 6 CWE enum"]
    end

    subgraph S2["Stage 2: Cleaning & Split<br/>app/data/cleaning/"]
        S2A["pipeline.py — orchestrator"]
        S2B["dedup.py — cosine similarity"]
        S2C["split.py — repo-grouped leak-safe"]
        S2D["contamination.py — n-gram overlap"]
        S2E["embeddings.py — lazy sentence-transformers"]
        S2F["hf_dataset.py — datasets.DatasetDict"]
        S2G["cli.py — Typer CLI"]
    end

    subgraph S3["Stage 3: Formatting<br/>app/data/formatting/"]
        S3A["builder.py — build_examples"]
        S3B["template.py — SYSTEM_PROMPT<br/>+ PROMPT_TEMPLATE<br/>+ format_prompt<br/>+ make_patch_diff"]
        S3C["tokenizer.py — TokenCounter<br/>(TokenBackend Protocol<br/>+ regex heuristic)"]
        S3D["pipeline.py + cli.py"]
    end

    subgraph S4["Stage 4: Baseline Eval<br/>app/evaluation/"]
        S4A["baseline.py — run_baseline()"]
        S4B["prompt.py — build_zero_shot_prompt<br/>build_few_shot_prompt"]
        S4C["backends.py — ModelBackend Protocol<br/>QwenBackend (lazy transformers)<br/>MockBackend"]
        S4D["parser.py — parse_prediction<br/>4-step extraction<br/>_template_detection"]
        S4E["metrics.py — BaselineMetrics<br/>+ compute_metrics<br/>(cwe_macro_f1,<br/>hallucination_rate,<br/>patch_coverage)"]
        S4F["cli.py — Typer CLI"]
    end

    subgraph S5["Stage 5: Training<br/>app/training/"]
        S5A["trainer_sft.py — SFT/QLoRA"]
        S5B["trainer_dpo.py — DPO via TRL"]
        S5C["sweep.py — LoRA rank sweep"]
        S5D["experiment.py — PostgreSQL + JSON fallback"]
        S5E["config.py — SFTConfig/DPOConfig/SweepConfig"]
        S5F["callbacks.py — TrainingCallback Protocol<br/>ResourceTracker / CheckpointCallback<br/>WandbCallback / ProgressCallback"]
        S5G["data.py + cli.py"]
    end

    subgraph S6["Stage 6: 4-Tier Eval<br/>app/evaluation/"]
        S6A["runner.py — EvaluationRunner<br/>EvalConfig<br/>compute_metrics → EvalMetrics"]
        S6B["tier1_deterministic.py<br/>22 PatternRules → 6 CWEs<br/>DeterministicEvaluator"]
        S6C["tier2_embedding_static.py<br/>19 rule_id→CWE mappings<br/>EmbeddingBackend (lazy)<br/>StaticSignalEvaluator"]
        S6D["tier3_exec.py<br/>apply_unified_diff (pure Python)<br/>TestGenerator (6 CWE templates)<br/>SandboxRunner Protocol:<br/>  LocalSandboxRunner<br/>  DockerSandboxRunner<br/>  MockSandboxRunner<br/>check_hallucinated_function_ref"]
        S6E["tier4_llm_judge.py<br/>LlmJudgeBackend Protocol<br/>MockLlmJudgeBackend<br/>LocalLlmJudgeBackend<br/>_OpenAiBackend<br/>_parse_judge_response (3-tier)"]
    end

    subgraph S7["Stage 7: Regression<br/>app/evaluation/"]
        S7A["general_capability.py<br/>12 non-security tasks<br/>CodeTestRunner Protocol:<br/>  LocalCodeTestRunner<br/>  DockerCodeTestRunner<br/>  MockCodeTestRunner<br/>_sanitize_paths"]
    end

    subgraph S8["Stage 8: Quantization<br/>app/quantization/"]
        S8A["quantizer.py — Quantizer Protocol<br/>MockQuantizer<br/>run_quantization_matrix<br/>quantize_single<br/>select_best_config<br/>score_quality_size_speed (0.6/0.2/0.2)"]
        S8B["config.py — QuantConfig<br/>GPTQConfig / AWQConfig / GGUFConfig"]
        S8C["export_gptq.py — GPTQQuantizer"]
        S8D["export_awq.py — AWQQuantizer"]
        S8E["export_gguf.py — GGUFQuantizer<br/>+ gguf_type_to_bits"]
        S8F["cli.py"]
    end

    subgraph S9["Stage 9: Serving<br/>app/serving/"]
        S9A["serve.py — VulnerabilityServer<br/>from_config + serve_sample + serve_batch<br/>(reuses Stage 4 prompt+parser)"]
        S9B["backends.py — ServingBackend Protocol<br/>LlamaCppBackend (llama-cpp-python)<br/>LlamaServerBackend (HTTP subprocess)<br/>TransformersBackend (HF format)<br/>OllamaBackend (HTTP API)<br/>MockServingBackend"]
        S9C["config.py — ServingConfig (frozen)<br/>all_warnings()"]
        S9D["api.py — FastAPI 3 endpoints<br/>/serve /serve/batch /manifest<br/>/healthz"]
        S9E["cli.py — Typer CLI"]
    end

    subgraph S10["Stage 10: CI Gate<br/>app/ci/"]
        S10A["gate.py — RegressionGate<br/>load_baseline_metrics<br/>load_stage6_report<br/>load_stage7_report<br/>4 checks:<br/>  f1_regression (≤5% drop)<br/>  forgetting (≥-0.10)<br/>  exec_pass_rate (≥0.0)<br/>  hallucination (≤50%)<br/>parse_gitleaks_output<br/>parse_trivy_output"]
        S10B["config.py — RegressionGateConfig (frozen)"]
        S10C["security_scanners.py — SecurityScanSummary"]
    end

    subgraph S11["Stage 11: Docs<br/>app/stage11/"]
        S11A["generator.py — Stage11Generator<br/>load_artifacts (fallback)<br/>ensure_deliverables<br/>validate_deliverables<br/>run_demo"]
        S11B["config.py — Stage11Config (frozen)<br/>_derive_model_name"]
    end

    subgraph ST["Storage Layer"]
        PG[("PostgreSQL<br/>app/storage/db.py<br/>VulnSampleRow<br/>TrainingRunRow")]
        S3[("MinIO / S3<br/>app/storage/object_store.py<br/>get_client / put_json / get_json")]
        JF["Local JSON Fallback<br/>output/stage*/ dir"]
    end

    CVE --> S1A
    SEMGREP --> S1A
    GH --> S1A
    HF --> S5A

    S1A -->|"VulnSample JSONL"| S2A
    S2F -->|"train/val/test/gold_eval"| S3A
    S3D -->|"InstructionExample JSONL"| S4A
    S4A -->|"predictions.jsonl<br/>metrics.json"| S5A
    S4A -->|"baseline metrics"| S10A
    S5A -->|"checkpoint"| S3
    S5D --> PG
    S5A --> S8A
    S5A --> S11A
    S4A -->|"metrics"| S6A
    S6A -->|"eval_report.json"| S10A
    S6A -->|"EvalReport"| S7A
    S7A -->|"regression_report.json"| S10A
    S8A -->|"quant_report.json"| S10A
    S8E --> S9A
    S10A -->|"gate_result.json<br/>ci_report.json"| S11A
    PG -.-> S2A
    S3 -.-> S5A
    PG -.-> S6A
    S3 -.-> S9A

    style S1 fill:#e1f5fe
    style S2 fill:#e8f5e9
    style S3 fill:#fff3e0
    style S4 fill:#fce4ec
    style S5 fill:#f3e5f5
    style S6 fill:#ede7f6
    style S7 fill:#f1f8e9
    style S8 fill:#fff8e1
    style S9 fill:#e0f7fa
    style S10 fill:#fff3e0
    style S11 fill:#fce4ec
    style ST fill:#fafafa
```

---

## 🔌 2. Injectable Backend Pattern (Across All Stages)

```mermaid
flowchart TB
    subgraph Pattern["The Injectable Backend Pattern"]
        direction TB
        P["Protocol"] -->|"implements"| Prod["Production Impl<br/>(e.g. QwenBackend)"]
        P -->|"implements"| Mock["Mock Impl<br/>(e.g. MockBackend)"]

        Prod -->|"lazy import inside method"| Deps["torch / transformers / peft<br/>sentence-transformers / auto_gptq / autoawq<br/>llama_cpp / httpx / docker / boto3"]
        Mock -->|"no deps"| Clean["Tests run without GPU<br/>or model downloads"]
    end

    subgraph Backends["Backend Implementations by Stage"]
        direction TB
        B1["Stage 2: EmbeddingBackend<br/>(lazy sentence-transformers)"]
        B2["Stage 3: TokenBackend Protocol<br/>(TokenCounter)"]
        B3["Stage 4-7: ModelBackend Protocol<br/>QwenBackend / MockBackend"]
        B4["Stage 6: SandboxRunner Protocol<br/>LocalSandboxRunner / DockerSandboxRunner / MockSandboxRunner"]
        B5["Stage 6: LlmJudgeBackend Protocol<br/>MockLlmJudgeBackend / LocalLlmJudgeBackend / _OpenAiBackend"]
        B6["Stage 6: CodeTestRunner Protocol<br/>LocalCodeTestRunner / DockerCodeTestRunner / MockCodeTestRunner"]
        B7["Stage 8: Quantizer Protocol<br/>GPTQQuantizer / AWQQuantizer / GGUFQuantizer / MockQuantizer"]
        B8["Stage 9: ServingBackend Protocol<br/>LlamaCpp / LlamaServer / Transformers<br/>Ollama / MockServingBackend"]
    end

    P -.->|"structural typing"| B1
    P -.-> B2
    P -.-> B3
    P -.-> B4
    P -.-> B5
    P -.-> B6
    P -.-> B7
    P -.-> B8
```

---

## ⚖️ 3. Four-Tier Evaluation Escalation (Stage 6)

```mermaid
flowchart TD
    T1["Tier 1: Deterministic<br/>(app/evaluation/tier1_deterministic.py)<br/><br/>DeterministicEvaluator<br/>• PatternRule dataclass (frozen)<br/>• DEFAULT_TIER1_RULES: 22 rules across 6 CWEs<br/>  — CWE-89: 5 patterns (f-string, concat, printf-style, SELECT, f'SELECT)<br/>  — CWE-79: 4 patterns (innerHTML, outerHTML, document.write, .send concat)<br/>  — CWE-22: 2 patterns (open() with +, open(var,))<br/>  — CWE-78: 4 patterns (shell=True, os.system, os.popen, subprocess+concat)<br/>  — CWE-190: 4 patterns (bytearray, (0)*, <<, multiplication)<br/>  — CWE-502: 3 patterns (pickle.loads, yaml.load, yaml.load+Loader)<br/>• Confidence: 0.55–0.99<br/>• evaluate(sample) → Tier1Result<br/>  (predicted_cwe, confidence, matched_pattern, num_matched)<br/><br/>Output: Tier1Result"]

    T2["Tier 2: Static Signal + Embedding<br/>(app/evaluation/tier2_embedding_static.py)<br/><br/>StaticSignalEvaluator<br/>• DEFAULT_RULE_TO_CWE: 19 semgrep rule→CWE mappings<br/>• Vote across findings → predicted_cwe<br/>• EmbeddingBackend (lazy sentence-transformers)<br/>  intfloat/multilingual-e5-base<br/>• cosine similarity: predicted patch vs gold fix<br/>• _cosine_similarity (pure Python)<br/>• evaluate(sample, prediction) → Tier2Result<br/>  (signal_sources, embedding_similarity)<br/><br/>Output: Tier2Result"]

    T3["Tier 3: Exec Sandbox<br/>(app/evaluation/tier3_exec.py)<br/><br/>ExecEvaluator<br/>• apply_unified_diff(source, diff)<br/>  — Pure Python (no git/patch dep)<br/>  — Brace-matching parser for @@ hunks<br/>  — Context verification + error messages<br/>• _TEST_TEMPLATES: 6 per-CWE test templates (AST-based)<br/>  — CWE-89: ast.parse, check no JoinedStr in .execute()<br/>  — CWE-79: check no innerHTML/outerHTML/document.write<br/>  — CWE-22: check realpath/abspath + startswith/commonpath<br/>  — CWE-78: check no shell=True/os.system/os.popen<br/>  — CWE-190: check OverflowError/raise/if guard<br/>  — CWE-502: ast.walk, check no pickle/unsafe yaml.load<br/>• SandboxRunner(Protocol)<br/>  — LocalSandboxRunner: subprocess, temp dir<br/>  — DockerSandboxRunner: --read-only --network none<br/>    user UID 1000, 512MB mem limit<br/>  — MockSandboxRunner: canned results<br/>• check_hallucinated_function_ref(code, patch)<br/>• evaluate(sample, prediction) → ExecEvalResult<br/>  (patch_applies, build_succeeds,<br/>tests_pass, hallucinated_cwe,<br/>hallucinated_function_ref)<br/><br/>Output: ExecEvalResult"]

    T4["Tier 4: LLM Judge<br/>(app/evaluation/tier4_llm_judge.py)<br/><br/>LlmJudge<br/>• JUDGE_PROMPT template:<br/>  CWE + description + vuln_code<br/>  + patch_diff + rationale →<br/>  JSON: {explanation_quality, patch_minimality, rationale}<br/>• _parse_judge_response (3-tier fallback):<br/>  1. strict json.loads<br/>  2. _find_json_objects brace-matching<br/>  3. regex extraction for truncated JSON<br/>• LlmJudgeBackend(Protocol)<br/>  — MockLlmJudgeBackend (default 0.5/0.5)<br/>  — LocalLlmJudgeBackend (HF model)<br/>  — _OpenAiBackend (OpenAI-compatible API)<br/>• evaluate(sample, prediction) → LlmJudgeScore<br/><br/>Output: LlmJudgeScore"]

    T1 -->|"samples"| T2
    T2 -->|"samples + predictions"| T3
    T3 -->|"samples + predictions"| T4
    T4 -->|"aggregate all tiers"| METRICS["compute_metrics (runner.py)<br/>→ EvalMetrics / EvalReport<br/><br/>Metrics:<br/>• tier1_cwe_macro_f1 / coverage<br/>• tier2_cwe_macro_f1 / coverage<br/>• model_cwe_macro_f1<br/>• exec_pass_rate / patch_applies_rate<br/>• build_succeeds_rate<br/>• hallucination_rate<br/>• avg_patch_coverage<br/>• avg_explanation_quality (T4)<br/>• avg_patch_minimality (T4)<br/>• per_class F1<br/><br/>Output: eval_report.json"]

    style T1 fill:#e3f2fd
    style T2 fill:#e8f5e9
    style T3 fill:#fff3e0
    style T4 fill:#f3e5f5
    style METRICS fill:#ede7f6
```

---

## 🛠️ 4. Stage 4: Baseline Evaluation (Detailed)

```mermaid
flowchart LR
    subgraph Stage4["Stage 4: Baseline Evaluation"]
        A["VulnSample JSONL<br/>(gold_eval split)"] --> B["run_baseline()"]
        C["InstructionExample JSONL<br/>(train split)"] --> D["build_few_shot_prompt<br/>(prompt.py)"]

        B --> E{"Strategy?"}
        E -->|"zero_shot"| F["build_zero_shot_prompt<br/>(prompt.py)"]
        E -->|"few_shot"| D
        F --> G["backend.generate<br/>(ModelBackend Protocol)"]
        D --> G
        G --> H["parse_prediction<br/>(parser.py)<br/>4-step JSON extraction<br/>→ ModelPrediction | ParseError"]
        H --> I["compute_metrics<br/>(metrics.py)<br/>→ BaselineMetrics"]
        I --> J["predictions.jsonl<br/>metrics.json<br/>manifest.json"]
        G -.->|"MockBackend"| K["Tests"]
    end

    subgraph "Parser 4-Step Extraction"
        P1["Strip bare fenced code blocks<br/>(not language-tagged)"]
        P2["Find json-tagged fenced blocks<br/>→ _JSON_FENCE_RE<br/>Skip template<br/>(_is_template_json)"]
        P3["Brace-match all {…}<br/>via _find_json_objects<br/>(brace counting<br/>with string-aware nesting)"]
        P4["Regex fallback<br/>_TTLBACK_*_RE regexes<br/>for unescaped quotes<br/>in patch_diff"]
    end

    H -.->|uses| P1
    P1 --> P2
    P2 --> P3
    P3 --> P4
```

**Files:**
| File | Key Functions/Classes |
|------|----------------------|
| `app/evaluation/baseline.py` | `run_baseline()`, `BaselineConfig`, `BaselineResult` |
| `app/evaluation/prompt.py` | `build_zero_shot_prompt()`, `build_few_shot_prompt()`, `RESPONSE_FORMAT_INSTRUCTION` (reuses `format_prompt` from Stage 3) |
| `app/evaluation/backends.py` | `ModelBackend` Protocol, `QwenBackend` (lazy `transformers`+`peft`), `MissingAdapterWeightsError`, `MockBackend` |
| `app/evaluation/parser.py` | `parse_prediction()`, `ParseError`, `_extract_json`, `_find_json_objects`, `_is_template_json`, `_try_fallback_extract` |
| `app/evaluation/metrics.py` | `BaselineMetrics`, `compute_metrics()` (baseline-specific: `cwe_macro_f1`, `cwe_micro_accuracy`, `severity_accuracy`, `hallucination_rate`, `patch_coverage`) |
| `app/evaluation/cli.py` | Typer CLI for baseline runs |

**Critical Fix — `MissingAdapterWeightsError`:**
When a LoRA adapter directory has `adapter_config.json` but no `adapter_model.safetensors`/`.bin`, the `QwenBackend` raises this error by default instead of silently falling back to the base model. This prevents a false `forgetting_delta == 0.0` in Stage 7. Callers must explicitly set `allow_base_fallback=True` to skip this guard.

---

## 🏋️ 5. Stage 5: Fine-Tuning (SFT / QLoRA / LoRA / DPO)

```mermaid
flowchart LR
    subgraph Stage5["Stage 5: Fine-Tuning"]
        A["InstructionExample JSONL"] --> B["run_sft / run_dpo"]
        B --> C["Lazy Import<br/>torch + transformers + peft +<br/>BitsAndBytesConfig + trl"]
        C --> D["_check_can_train()<br/>RuntimeError if missing<br/>CUDA required for QLoRA<br/>CPU allowed for LoRA"]
        D --> E["Training Loop<br/>with Callbacks:"]
        E --> E1["WandbCallback<br/>(mock=True or wandb)"]
        E --> E2["CheckpointCallback<br/>→ MinIO upload<br/>per-file boto3 put_object"]
        E --> E3["ProgressCallback<br/>(console)"]
        E --> E4["ResourceTracker<br/>peak VRAM + wall time"]
        E --> F["TrainingResult<br/>+ loss history"]
        F --> G["persist_training_run<br/>→ PostgreSQL<br/>+ JSON fallback"]
        F --> H["CheckpointCallback.save_checkpoint<br/>→ s3://bucket/checkpoints/stage5/RUN_ID/epoch_N/"]
    end

    subgraph "Training Methods"
        M1["SFT Full<br/>(4-bit QLoRA<br/>bitsandbytes nf4)<br/>BitsAndBytesConfig"]
        M2["SFT LoRA<br/>(bfloat16/float32<br/>CPU fallback)<br/>PEFT LoraConfig"]
        M3["LoRA Sweep<br/> ranks: 8,16,32,64,128<br/>select best by val loss"]
        M4["DPO<br/>TRL DPOTrainer<br/>preference pairs<br/>beta=0.1, sigmoid loss<br/>4-bit QLoRA loading<br/>FSDP compat shim"]
    end

    B --> M1
    B --> M2
    B --> M3
    M2 --> M4
```

**Files:**
| File | Key Classes/Functions |
|------|----------------------|
| `app/training/trainer_sft.py` | `SFTConfig`, `_run_sft`, `run_sft`, QLoRA with `BitsAndBytesConfig(nf4)`, CPU fallback, `dry_run` mode |
| `app/training/trainer_dpo.py` | `DPOConfig`, `_run_dpo`, `run_dpo` (TRL `DPOTrainer`), preference pair construction |
| `app/training/sweep.py` | `SweepConfig`, `run_sweep` (LoRA ranks 8-128), `SweepResult` |
| `app/training/experiment.py` | `persist_training_run()` (PostgreSQL `TrainingRunRow` + JSON fallback) |
| `app/training/config.py` | `SFTConfig`, `DPOConfig`, `SweepConfig` dataclasses |
| `app/training/callbacks.py` | `TrainingCallback` Protocol, `ResourceTracker`, `CheckpointCallback`→MinIO, `WandbCallback`, `ProgressCallback` |
| `app/training/data.py` | `load_examples()`, `JsonlDataLoader` |
| `app/training/cli.py` | Typer CLI for SFT/DPO/sweep |

---

## 📦 6. Storage Layer — PostgreSQL + MinIO / S3 Split

```mermaid
erDiagram
    VulnSampleRow {
        string id PK
        string source "cve_real, synthetic, or ctf"
        string repo_name "leakage-safe split key"
        string commit_sha
        string cwe_id "CWE-89 through CWE-502"
        string severity "low, medium, high, or critical"
        string language
        string description
        json static_findings
        string split "train, val, test, or gold_eval"
        string object_store_key "MinIO pointer"
        string created_at
    }

    TrainingRunRow {
        string id PK
        string run_name
        string method "sft_full, sft_qlora, lora, or dpo"
        string base_model
        json hyperparams
        string train_set_size
        string train_time_minutes
        string peak_vram_gb
        string final_train_loss
        string final_val_loss
        string checkpoint_uri "S3 path"
        string status "pending, running, completed, or failed"
        string created_at
    }

    ObjectStore {
        string default_bucket "vuln-triage"
        string put_json "key, payload"
        string get_json "key"
    }

    JSONFallback {
        string stage2_splits "output/stage2/splits.json"
        string stage5_training_result "output/stage5/training_result.json"
        string stage6_eval_report "output/stage6/eval_report.json"
        string stage7_regression_report "output/stage7/regression_report.json"
        string stage8_quant_report "output/stage8/quant_report.json"
        string stage10_gate_result "output/stage10/gate_result.json"
        string stage10_ci_report "output/stage10/ci_report.json"
    }

    VulnSampleRow ||--o{ ObjectStore : "object_store_key references"
    TrainingRunRow ||--o{ ObjectStore : "checkpoint_uri references"
    VulnSampleRow }o--|| JSONFallback : "fallback when Postgres offline"
    TrainingRunRow }o--|| JSONFallback : "fallback when Postgres offline"
```

**Schema Contracts** (`app/schemas/`):

| File | Models |
|------|--------|
| `vuln.py` | `VulnSample`, `StaticFinding` |
| `dataset.py` | `InstructionExample` |
| `prediction_eval.py` | `ModelPrediction`, `Tier1Result`, `Tier2Result`, `ExecEvalResult`, `LlmJudgeScore`, `EvalMetrics`, `EvalReport`, `GeneralCapabilityResult`, `GeneralCapabilityMetrics`, `RegressionReport`, `RegressionSummary` |
| `training.py` | `TrainingRun`, `TrainingResult`, `SweepResult` |
| `quantization.py` | `QuantMethod` (StrEnum), `QuantStatus`, `QuantResult`, `QuantReport`, `QuantRecommendation` |
| `serving.py` | `ServeRequest`, `ServeResponse`, `BatchServeRequest`, `BatchServeResponse`, `ServeManifest` |
| `ci.py` | `GateStatus`, `GateCheck`, `RegressionGateResult`, `SecurityScanSummary`, `CiReport` |
| `documentation.py` | `CWE_SCOPE`, `LANGUAGE_SCOPE`, `BASE_MODEL`, `EvalMetricsSnapshot`, `TrainingRunData`, `QuantResultData`, `ModelCardData`, `TrainingReportData`, `DemoResult` |

**Key Design Decisions:**
- **Metadata in Postgres, payloads in S3:** Only queryable metadata (CWE, severity, repo_name, split) lives in the DB; full code payloads and model checkpoints live in MinIO/S3
- **JSON fallback everywhere:** Every stage writes JSON to `output/stageN/` as a file-based fallback when Postgres is unavailable
- **Leakage-safe splits:** Grouping by `repo_name` (not individual samples) prevents the same repository appearing in both train and test

---

## 🎯 7. Stage 7: Regression / Forgetting Analysis

```mermaid
graph LR
    subgraph Stage7["Stage 7: Regression Analysis"]
        A["Base Model<br/>(QwenBackend, allow_base_fallback=True)"] --> B["GeneralCapabilityEvaluator"]
        C["Tuned Model<br/>(QwenBackend with LoRA)"] --> B
        B --> D["12 Algorithm Tasks<br/>(app/evaluation/general_capability.py)<br/>DEFAULT_GENERAL_TASKS:<br/>• factorial<br/>• is_palindrome<br/>• fibonacci<br/>• binary_search<br/>• two_sum<br/>• reverse_int<br/>• longest_common_prefix<br/>• max_subarray_sum<br/>• is_valid_parentheses<br/>• count_vowels<br/>• is_anagram<br/>• merge_sorted_arrays<br/><br/>All stdlib-only — no external deps<br/>to keep sandbox self-contained"]
        D --> E["CodeTestRunner Protocol"]
        E --> E1["LocalCodeTestRunner<br/>subprocess + temp dir"]
        E --> E2["DockerCodeTestRunner<br/>isolated container"]
        E --> E3["MockCodeTestRunner<br/>(canned results)"]
        E --> F["Solution: generate → write<br/>to temp dir → run pytest<br/>→ pass/fail"]
        F --> G["GeneralCapabilityMetrics<br/>execution_accuracy = passed/total"]
        F --> H["_sanitize_paths<br/>(redacts /tmp/ paths<br/>from pytest output)"]
        G --> I["forgetting_delta =<br/>tuned_acc - base_acc<br/>(negative = forgetting)"]
        I --> J["RegressionReport<br/>+ RegressionSummary<br/>→ regression_report.json"]
        J -->|"→ Stage 10 Gate"| K
    end

    style Stage7 fill:#f1f8e9
```

---

## ⚙️ 8. Stage 8: Quantization Matrix

```mermaid
graph LR
    subgraph Stage8["Stage 8: Quantization Matrix"]
        A["Stage 5 Checkpoint<br/>full-precision"] --> B["run_quantization_matrix<br/>(quantizer.py)"]
        B --> C["QuantConfig<br/>methods × bit_widths"]
        C --> D1["GPTQQuantizer<br/>auto-gptq 2-4 bit<br/>(group_size 128)"]
        C --> D2["AWQQuantizer<br/>autoawq 2-5 bit<br/>(group_size 128)"]
        C --> D3["GGUFQuantizer<br/>llama.cpp: Q2_K, Q3_K,<br/>Q4_0, Q4_K, Q5_K,<br/>Q8_0, F16"]
        D1 --> E["QuantResult<br/>per method × bits"]
        D2 --> E
        D3 --> E
        E --> F["select_best_config<br/>filters: COMPLETED<br/>target_vram_gb / target_size_gb<br/>score = 0.6×quality<br/>+ 0.2×size<br/>+ 0.2×speed<br/><br/>• size_score = 1 - size_gb/14.0<br/>• speed_score = min(1, tps/30.0)<br/>• quality = model_cwe_macro_f1<br/>  (fallback: estimate_quality)<br/><br/>Return: max(score)"]
        F --> G["QuantReport<br/>+ best_result<br/>+ manifest<br/>→ quant_report.json"]
        G -->|"→ Stage 10 Gate"| H
        G -->|"best GGUF"| I["Stage 9 Serving"]
        B -->|"mock=True"| JM["MockQuantizer<br/>deterministic<br/>heuristic estimates"]
        B -->|"dry_run=True"| DR["heuristic estimates<br/>(no real quant)"]
    end

    style Stage8 fill:#fff8e1
```

**Scoring Constants** (`quantizer.py`):
- `_QUALITY_WEIGHT = 0.6`
- `_SIZE_WEIGHT = 0.2`
- `_SPEED_WEIGHT = 0.2`

**Baseline normalization:** FP16 = 14 GB / ~30 t/s (7B model on GPU)

---

## 🚀 9. Stage 9: Air-Gapped Serving

```mermaid
graph LR
    subgraph Stage9["Stage 9: Serving"]
        A["GGUF checkpoint<br/>(from Stage 8 best)"] --> B["VulnerabilityServer<br/>(serve.py)"]
        B --> C["build_zero_shot_prompt<br/>(reuse Stage 4 prompt.py)<br/>→ model.generate<br/>→ parse_prediction<br/>→ ServeResponse"]
        C --> D["ServingBackend Protocol"]
        D --> E1["LlamaCppBackend<br/>llama-cpp-python<br/>(lazy import)<br/>CPU/GPU n_gpu_layers"]
        D --> E2["LlamaServerBackend<br/>llama-server.exe subprocess<br/>+ HTTP /completion<br/>(lazy httpx import)"]
        D --> E3["TransformersBackend<br/>HF format model.safetensors<br/>(torch + transformers)<br/>GPU fallback<br/>when llama.cpp unavailable"]
        D --> E4["OllamaBackend<br/>local HTTP API<br/>http://localhost:11434<br/>(lazy httpx)"]
        D --> E5["MockServingBackend<br/>(deterministic)<br/>for tests"]
        B --> F["from_config(ServingConfig)<br/>backend factory:<br/>  llama.cpp → LlamaCppBackend<br/>  llama-server → LlamaServerBackend<br/>  ollama → OllamaBackend<br/>  mock → MockServingBackend"]

        subgraph "API Layer"
            G["create_app<br/>(api.py)<br/>FastAPI<br/>3 endpoints:<br/>  POST /api/v1/serve<br/>  POST /api/v1/serve/batch<br/>  GET /api/v1/manifest<br/>  GET /healthz"]
            H["cli.py<br/>Typer CLI<br/>serve / serve-batch"]
        end

        F --> G
        G --> H
    end

    style Stage9 fill:#e0f7fa
```

---

## 🚦 10. Stage 10: CI Regression Gate

```mermaid
graph LR
    subgraph Stage10["Stage 10: CI Regression Gate"]
        A["Stage 4<br/>metrics.json"] --> B["RegressionGate"]
        C["Stage 6<br/>eval_report.json"] --> B
        D["Stage 7<br/>regression_report.json"] --> B
        E["Stage 8<br/>quant_report.json"] --> B
        B --> F["4 Gate Checks:<br/><br/>① check_f1_regression<br/>  baseline_cwe_macro_f1<br/>  vs current_cwe_macro_f1<br/>  FAIL if drop > 5.0%<br/><br/>② check_forgetting<br/>  forgetting_delta < -0.10<br/>  SKIP if Stage 7 absent<br/><br/>③ check_exec_pass_rate<br/>  FAIL if exec_pass_rate < 0.0<br/>  (configurable floor)<br/><br/>④ check_hallucination_rate<br/>  FAIL if > 0.50<br/>  (50% threshold)"]
        B --> G["Artifact Loaders:<br/>  load_baseline_metrics<br/>    → FileNotFoundError<br/>    → RuntimeError (missing key)<br/>  load_stage6_report<br/>    → tolerates nested<br/>      or flattened structure<br/>  load_stage7_report<br/>    → requires forgetting_delta<br/>  load_quant_report (optional)"]
        F --> H["RegressionGateResult<br/>status: PASS|FAIL<br/>checks: 4x GateCheck<br/>manifest: config + paths"]
        B --> I["Security Scans:<br/>parse_gitleaks_output<br/>→ SecurityScanSummary<br/>parse_trivy_output<br/>→ SecurityScanSummary"]
        H --> J["CiReport<br/>overall_status: PASS|FAIL<br/>(gate + gitleaks + trivy)<br/>→ ci_report.json"]
        J -->|"CI verdict"| CI["CI/CD Pipeline"]
        J -->|"→ Stage 11"| K
    end

    style Stage10 fill:#fff3e0
```

**Files:**
| File | Key Functions/Classes |
|------|----------------------|
| `app/ci/gate.py` | `RegressionGate` class, `run_gate()`, `load_baseline_metrics()`, `load_stage6_report()`, `load_stage7_report()`, `parse_gitleaks_output()`, `parse_trivy_output()` |
| `app/ci/config.py` | `RegressionGateConfig` (frozen dataclass with threshold defaults) |
| `app/ci/security_scanners.py` | Gitleaks + Trivy output parsers → `SecurityScanSummary` |

**Schema** (`app/schemas/ci.py`): `GateStatus` (StrEnum: PASS/FAIL/SKIP), `GateCheck` (name, status, message, details), `RegressionGateResult` (status, run_id, timestamp, baseline/current F1, f1_drop_percent, exec_pass_rate, hallucination_rate, forgetting_delta, checks, manifest), `SecurityScanSummary` (tool, status, findings_count, severity_counts), `CiReport` (run_id, gate, gitleaks, trivy, overall_status)

---

## 📄 11. Stage 11: Documentation Generation

```mermaid
graph LR
    subgraph Stage11["Stage 11: Documentation"]
        A["Stage 4<br/>metrics.json"] --> B["Stage11Generator<br/>(generator.py)"]
        C["Stage 5<br/>TrainingRunRow"] --> B
        D["Stage 6<br/>eval_report.json"] --> B
        E["Stage 7<br/>regression_report.json"] --> B
        F["Stage 8<br/>quant_report.json"] --> B
        G["Stage 10<br/>gate_result.json<br/>ci_report.json"] --> B
        B --> H["load_artifacts<br/>(gold/silver/bronze<br/>fallback chain)<br/><br/>If Stage 6 report missing:<br/>  gold → eval_report.json<br/>  silver → metrics.json<br/>  bronze → mock defaults<br/><br/>If Stage 5 record missing:<br/>  gold → postgres TrainingRunRow<br/>  silver → training_result.json<br/>  bronze → config defaults"]

        H --> I["ensure_deliverables:"]
        I --> I1["docs/model_card.md<br/>generate_model_card_markdown<br/>(ModelCardData)<br/><br/>Fields:<br/>  model_name, base_model<br/>  training_method, lora_rank<br/>  quant_method, quant_bit_width<br/>  cwe_scope, eval metrics<br/>  limitations, ethical_considerations"]

        I --> I2["docs/training_report.md<br/>generate_training_report<br/>(TrainingReportData)<br/><br/>Fields:<br/>  training_runs (list),<br/>  baseline/tuned/quant metrics<br/>  gate_result<br/>  conclusions, recommendations"]

        I --> I3["docs/demo.py<br/>_DEMO_TEMPLATE<br/><br/>Executable demo:<br/>  1. MockServingBackend<br/>  2. Sample vulnerable code<br/>  3. Run inference<br/>  4. Print prediction<br/>  5. Run 4-tier eval<br/>  6. Print metrics"]

        B --> J["validate_deliverables<br/>(exists + non-empty)"]
        B --> K["run_demo<br/>(mock subprocess)"]
    end

    style Stage11 fill:#fce4ec
```

---

## 📋 12. Test Coverage Matrix

```mermaid
graph LR
    subgraph Tests["Test Suites"]
        subgraph Unit["tests/unit/"]
            U1["test_evaluation_backends.py<br/>MockBackend response mapping<br/>MockBackend call tracking<br/>QwenBackend constructor storage<br/>Lazy loading behavior<br/>MissingAdapterWeightsError<br/>adapter_applied flag<br/>Protocol compatibility"]

            U2["test_evaluation_runner.py<br/>_compute_cwe_macro_f1 edge cases<br/>_compute_coverage<br/>EvaluationRunner tier injection<br/>Sandbox modes: local/docker/mock<br/>load_samples / load_predictions<br/>run() end-to-end with mocks"]

            U3["test_evaluation_parser.py<br/>parse_prediction<br/>_is_template_json<br/>_extract_json<br/>_find_json_objects<br/>_try_fallback_extract<br/>ParseError handling"]

            U4["test_evaluation_metrics.py<br/>BaselineMetrics fields<br/>compute_metrics<br/>per-class F1<br/>hallucination rate<br/>patch coverage"]

            U5["test_evaluation_prompt.py<br/>build_zero_shot_prompt<br/>build_few_shot_prompt<br/>RESPONSE_FORMAT_INSTRUCTION<br/>format_prompt reuse"]

            U6["test_tier1_deterministic.py<br/>PatternRule matching<br/>confidence ranking<br/>no-match → None<br/>DEFAULT_TIER1_RULES count<br/>classify_deterministic"]

            U7["test_tier2_embedding_static.py<br/>DEFAULT_RULE_TO_CWE mapping<br/>Semgrep vote → CWE<br/>EmbeddingBackend lazy load<br/>_cosine_similarity<br/>StaticSignalEvaluator.evaluate"]

            U8["test_tier3_exec.py<br/>apply_unified_diff<br/>_parse_diff_hunks<br/>_find_first_mismatch<br/>_TEST_TEMPLATES (6 CWEs)<br/>TestGenerator<br/>sandbox modes<br/>check_hallucinated_function_ref<br/>LocalSandboxRunner<br/>MockSandboxRunner"]

            U9["test_tier4_llm_judge.py<br/>JUDGE_PROMPT formatting<br/>_parse_judge_response<br/>3-tier fallback:<br/>  strict JSON → brace-match → regex<br/>MockLlmJudgeBackend<br/>LlmJudge.invoke / evaluate<br/>_OpenAiBackend fallback"]

            U10["test_baseline.py<br/>run_baseline<br/>BaselineConfig<br/>BaselineResult<br/>parse_errors<br/>MockBackend injection"]

            U11["test_vulnerability_server.py<br/>VulnerabilityServer.serve_sample<br/>serve_batch<br/>from_config<br/>MockServingBackend<br/>parse_prediction integration"]

            U12["test_serving_backends.py<br/>LlamaCppBackend._load<br/>mock mode<br/>generate output parsing<br/>LlamaServerBackend lifecycle<br/>OllamaBackend<br/>TransformersBackend<br/>MockServingBackend<br/>ServingBackend Protocol<br/>_import_httpx<br/>_find_hf_model_dir<br/>_find_llama_server"]

            U13["test_serving_config.py<br/>ServingConfig validation<br/>all_warnings<br/>boundary conditions<br/>backend_type validation"]

            U14["test_quantization.py<br/>Quantizer Protocol<br/>MockQuantizer<br/>quantize_single<br/>dry_run / mock modes<br/>select_best_config filters<br/>score_quality_size_speed<br/>_NoOpQuantizer"]

            U15["test_stage10_gate.py<br/>RegressionGate<br/>4 checks:<br/>  check_f1_regression<br/>  check_forgetting (SKIP)<br/>  check_exec_pass_rate<br/>  check_hallucination_rate<br/>load_baseline/stage6/stage7<br/>parse_gitleaks_output<br/>parse_trivy_output<br/>RegressionGateConfig (frozen)<br/>run_gate end-to-end"]

            U16["test_training_trainer_sft.py<br/>_check_can_train<br/>dry_run mode<br/>TrainingResult<br/>ResourceTracker<br/>Config validation<br/>Callback injection"]

            U17["test_training_trainer_dpo.py<br/>DPOTrainer integration<br/>Preference pair construction<br/>FSDP shim<br/>TrainingResult fields"]

            U18["test_training_callbacks.py<br/>WandbCallback mock mode<br/>CheckpointCallback mock → MinIO<br/>ProgressCallback<br/>ResourceTracker VRAM<br/>TrainingCallback Protocol"]

            U19["test_training_experiment.py<br/>persist_training_run<br/>PostgreSQL + JSON fallback<br/>TrainingRunRow mapping<br/>load_training_run"]

            U20["test_general_capability.py<br/>DEFAULT_GENERAL_TASKS (12)<br/>_gc_task constructor<br/>CodeTestRunner Protocol<br/>LocalCodeTestRunner<br/>subprocess safety (_sanitize_paths)<br/>MockCodeTestRunner<br/>forgetting_delta calc"]

            U21["test_stage11_documentation.py<br/>Stage11Config (frozen)<br/>_derive_model_name<br/>generate_model_card_markdown<br/>generate_training_report<br/>generate_demo_script<br/>load_artifacts fallback chain<br/>ensure_deliverables<br/>validate_deliverables<br/>run_demo (mocked<br/>subprocess)"]
        end

        subgraph Integration["tests/integration/"]
            I1["test_stage1_pipeline.py<br/>CVE → Semgrep → DB"]
            I2["test_stage2_pipeline.py<br/>Dedup + split + contaminate"]
            I3["test_stage3_pipeline.py<br/>VulnSample → InstructionExample"]
            I4["test_stage4_baseline.py<br/>End-to-end baseline<br/>with MockBackend"]
            I5["test_stage5_training.py<br/>dry_run SFT/QLoRA/DPO<br/>training_result.json<br/>Loss curve generation"]
            I6["test_stage6_four_tier.py<br/>EvaluationRunner<br/>all 4 tiers<br/>EvalReport output<br/>compute_metrics"]
            I7["test_stage7_regression.py<br/>GeneralCapabilityEvaluator<br/>forgetting analysis<br/>LocalCodeTestRunner"]
            I8["test_stage8_quantization.py<br/>QuantizationMatrix<br/>QuantReport<br/>select_best_config<br/>dry_run estimation"]
            I9["test_stage9_serving.py<br/>VulnerabilityServer<br/>end-to-end<br/>MockServingBackend<br/>ServeResponse"]
            I10["test_stage10_ci.py<br/>RegressionGate end-to-end<br/>CiReport<br/>SecurityScanSummary<br/>GateCheck statuses"]
            I11["test_stage11_docs.py<br/>Stage11Generator<br/>all 3 deliverables<br/>File existence + content"]
        end

        subgraph CodeQuality["tests/code_quality/"]
            CQ1["test_mypy_types.py<br/>mypy strict checks"]
            CQ2["test_type_coverage.py<br/>type annotation coverage"]
        end
    end

    style Tests fill:#f5f5f5
```

---

## 🔒 13. Security Patterns

| Pattern | Location | `# nosec` Code | Why it's safe |
|---------|----------|---------------|---------------|
| Subprocess with `sys.executable` + temp files | `tier3_exec.py`, `general_capability.py` | `B603` (run) / `B404` (import) | Inputs are `sys.executable` (system Python) and temp file paths created by the code itself — attacker doesn't control the path or args |
| `random.Random` for split seeding | `split.py` | `B311` | Deterministic MD5 hash-based split, not security-sensitive |
| Enum string values flagged as hardcoded | `ci.py` | `B105` | These are enum literals for `PASS`/`FAIL`/`SKIP`, not passwords |
| `open()` on local paths | `object_store.py`, templates | `B108` | Paths are under `/tmp/` with test-controlled names |
| `transformers` remote model load | `backends.py` | `B615` | Models loaded from pinned HuggingFace tags, not arbitrary URLs |

---

## 🧩 14. Lazy Import Map

```mermaid
graph LR
    subgraph LazyImports["Heavy ML Dependencies — All Lazy-Loaded"]
        subgraph L1["Stage 2"]
            LI1["embeddings.py<br/>→ sentence-transformers"]
        end
        subgraph L2["Stage 3"]
            LI2["tokenizer.py<br/>→ transformers<br/>(AutoTokenizer)"]
        end
        subgraph L3["Stage 4"]
            LI3["backends.py<br/>→ transformers + peft<br/>(QwenBackend._load)"]
        end
        subgraph L4["Stage 5"]
            LI4A["trainer_sft.py<br/>→ torch + transformers<br/>+ peft + bitsandbytes"]
            LI4B["trainer_dpo.py<br/>→ trl + transformers<br/>+ peft"]
            LI4C["callbacks.py<br/>→ torch.cuda<br/>(wandb lazy)"]
            LI4D["experiment.py<br/>→ sqlalchemy (always)<br/>+ psycopg2 (always)"]
        end
        subgraph L5["Stage 6 Tier 2"]
            LI5["tier2_embedding_static.py<br/>→ sentence-transformers<br/>(EmbeddingBackend)"]
        end
        subgraph L6["Stage 6 Tier 4"]
            LI6["tier4_llm_judge.py<br/>→ torch (LocalLlmJudgeBackend)<br/>+ openai (_OpenAiBackend)"]
        end
        subgraph L7["Stage 6 Tier 3"]
            LI7a["tier3_exec.py<br/>→ docker (DockerSandboxRunner)"]
            LI7b["tier3_exec.py<br/>→ subprocess (always)<br/>+ pytest (runtime)"]
        end
        subgraph L8["Stage 7"]
            LI8a["general_capability.py<br/>→ subprocess (always)<br/>+ pytest (runtime)"]
            LI8b["general_capability.py<br/>→ docker (DockerCodeTestRunner)"]
        end
        subgraph L9["Stage 9"]
            LI9a["backends.py<br/>→ llama_cpp (LlamaCppBackend)"]
            LI9b["backends.py<br/>→ httpx (LlamaServerBackend<br/>+ OllamaBackend)"]
            LI9c["backends.py<br/>→ torch + transformers<br/>(TransformersBackend)"]
        end
        subgraph L10["Stage 8"]
            LI10a["export_gptq.py<br/>→ auto_gptq + torch"]
            LI10b["export_awq.py<br/>→ autoawq + torch"]
            LI10c["export_gguf.py<br/>→ llama_cpp or<br/>subprocess + llama.cpp CLI"]
        end
    end

    subgraph AlwaysLoaded["Always Loaded (at import time)"]
        AL1["boto3 (storage/object_store.py)"]
        AL2["pydantic (all schemas)"]
        AL3["fastapi + uvicorn (api.py)"]
        AL4["sqlalchemy (storage/db.py)"]
        AL5["typer (all CLI modules)"]
    end

    style LazyImports fill:#fff9c4
    style AlwaysLoaded fill:#bbdefb
```

---

## 🗺️ 15. Dependency Flow Diagram

```mermaid
graph LR
    subgraph "External → Stage 1"
        EXT["CVE API • Semgrep • GitHub • HF Hub"]
    end

    subgraph "Data Plane"
        S1["Stage 1<br/>Collectors"]
        S2["Stage 2<br/>Cleaning"]
        S3["Stage 3<br/>Formatting"]
        S4["Stage 4<br/>Baseline"]
        S5["Stage 5<br/>Training"]
        S6["Stage 6<br/>Evaluation"]
        S7["Stage 7<br/>Regression"]
        S8["Stage 8<br/>Quantize"]
    end

    subgraph "Control Plane"
        S10["Stage 10<br/>CI Gate"]
        S11["Stage 11<br/>Docs"]
    end

    subgraph "Runtime"
        S9["Stage 9<br/>Serving"]
    end

    subgraph "Shared Infra"
        PG["PostgreSQL"]
        S3S3["S3/MinIO"]
        SCHEMA["app/schemas/"]
    end

    EXT --> S1
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 -->|"baseline metrics"| S5
    S4 -->|"baseline metrics"| S10
    S5 --> S6
    S5 --> S8
    S5 --> S11
    S6 --> S7
    S6 --> S10
    S7 --> S10
    S8 --> S10
    S8 --> S9
    S10 --> S11

    S1 <--> PG
    S1 <--> S3S3
    S5 <--> PG
    S5 <--> S3S3
    S6 <--> PG
    S9 <--> S3S3
    S4 <--> SCHEMA
    S5 <--> SCHEMA
    S6 <--> SCHEMA
    S7 <--> SCHEMA
    S8 <--> SCHEMA
    S9 <--> SCHEMA
    S10 <--> SCHEMA
    S11 <--> SCHEMA

    style EXT fill:#e3f2fd
    style S1 fill:#e1f5fe
    style S2 fill:#e8f5e9
    style S3 fill:#fff3e0
    style S4 fill:#fce4ec
    style S5 fill:#f3e5f5
    style S6 fill:#ede7f6
    style S7 fill:#f1f8e9
    style S8 fill:#fff8e1
    style S9 fill:#e0f7fa
    style S10 fill:#fff3e0
    style S11 fill:#fce4ec
    style PG fill:#fafafa
    style S3S3 fill:#fafafa
    style SCHEMA fill:#fafafa
```
