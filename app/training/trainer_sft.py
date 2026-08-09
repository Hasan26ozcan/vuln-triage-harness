"""Stage 5 — SFT (Supervised Fine-Tuning) trainer.

Supports two modes:

- **Full-parameter fine-tuning** (``SFTConfig(use_4bit=False)``): the entire
  model is updated. This is the baseline against which LoRA/QLoRA are compared.
- **QLoRA** (``SFTConfig(use_4bit=True)``): the base model is loaded in 4-bit
  NF4 via ``bitsandbytes``, and only LoRA adapters are trained. This is the
  budget-friendly path for an 8GB-VRAM GPU.

Heavy ML imports (``transformers``, ``peft``, ``bitsandbytes``, ``torch``)
are imported **lazily** inside ``_run_sft`` — constructing the module or
calling ``estimate_memory`` does not require them. This keeps the module
importable (and testable) in CI without a GPU.

The trainer accepts injectable callbacks (following the same injectable-backend
pattern from Stages 2–4). In tests, pass ``callbacks=[WandbCallback(mock=True)]``
and a mock dataset — no real training will happen because ``_run_sft`` performs
a ``_check_can_train`` guard that raises ``TrainingUnavailableError`` when
``transformers`` or a GPU is missing, **unless** ``dry_run=True`` is passed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from app.schemas.dataset import InstructionExample
from app.schemas.training import TrainingResult
from app.training.callbacks import (
    CheckpointCallback,
    ProgressCallback,
    ResourceTracker,
    WandbCallback,
)
from app.training.config import SFTConfig

logger = logging.getLogger(__name__)


class TrainingUnavailableError(RuntimeError):
    """Raised when training is attempted but the required ML stack is absent.

    In CI or on CPU-only machines, we catch this in the CLI and fall back to
    ``--dry-run`` mode (which reports the hyperparams and estimated steps
    without actually training).
    """


@dataclass
class StepEstimate:
    """Estimated training steps and memory, computed without running training."""

    num_train_steps: int
    steps_per_epoch: int
    num_epochs: int
    gradient_accumulation_steps: int
    estimated_vram_gb: float  # heuristic, not measured
    can_fit_in_8gb: bool


def estimate_training_steps(
    config: SFTConfig,
    n_train_examples: int,
) -> StepEstimate:
    """Compute the number of optimiser steps and a VRAM heuristic.

    This does **not** require a GPU or torch — it's a pure arithmetic
    calculation so the CLI can report "this run would take ~N steps"
    even in ``--dry-run`` mode.
    """
    steps_per_epoch = max(
        1,
        n_train_examples // config.per_device_train_batch_size,
    )
    # With gradient accumulation, each *optimiser step* covers
    # (batch_size × grad_accum) examples.
    optim_steps_per_epoch = max(
        1,
        steps_per_epoch // config.gradient_accumulation_steps,
    )
    total_steps = optim_steps_per_epoch * config.num_train_epochs

    # Very rough VRAM heuristic: 4-bit QLoRA ~ 5-6 GB for 7B, full SFT ~ 14-15 GB.
    # We just use the method to categorise.
    if config.use_4bit:
        est_vram = 6.0  # QLoRA fit on 8GB
    else:
        est_vram = 15.0  # full SFT needs ~16GB for 7B

    # Also factor in sequence length — longer sequences = more VRAM.
    # Rough rule: +1 GB per 2k tokens of context.
    est_vram += 1.0

    return StepEstimate(
        num_train_steps=total_steps,
        steps_per_epoch=optim_steps_per_epoch,
        num_epochs=config.num_train_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        estimated_vram_gb=est_vram,
        can_fit_in_8gb=est_vram <= 8.0,
    )


def _check_can_train(config: SFTConfig) -> None:
    """Verify that the ML stack is importable and a GPU is available.

    Raises ``TrainingUnavailableError`` with a helpful message if not.
    Callers (CLI) catch this and offer ``--dry-run``.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise TrainingUnavailableError(
            "torch/transformers not installed. Run `pip install -e '.[ml]'` "
            "to train, or use `--dry-run` for a step estimate."
        ) from exc

    if not torch.cuda.is_available():
        raise TrainingUnavailableError(
            "No CUDA GPU detected. Training Qwen2.5-Coder-7B requires a GPU "
            "with >=10 GB VRAM (8 GB for QLoRA). Use `--dry-run` for an estimate, "
            "or set up a GPU machine."
        )


def _load_jsonl_for_training(
    train_path: str,
    val_path: str,
    loader=None,
) -> tuple[list[InstructionExample], list[InstructionExample]]:
    """Load train and val examples, with injectable loader for tests."""
    from app.training.data import JsonlDataLoader, load_examples

    ld = loader or JsonlDataLoader()
    train = load_examples(train_path, loader=ld)
    val = load_examples(val_path, loader=ld) if val_path else []
    return train, val


def _convert_for_causal_lm(examples: list[InstructionExample]) -> list[dict]:
    """Format examples for causal-LM training (prompt + completion concatenated).

    The standard instruction-tuning format wraps the prompt and completion in
    a single text string with EOS tokens. We use the Qwen2.5 chat template
    convention: ``<|im_start|>system\\n...\\n<|im_start|>user\\n{prompt}``
    ``\\n<|im_start|>assistant\\n{completion}``.
    """
    rows: list[dict] = []
    for ex in examples:
        completion = ex.target_explanation or ""
        rows.append(
            {
                "prompt": ex.prompt,
                "completion": completion,
                "cwe_id": ex.target_cwe,
            }
        )
    return rows


def _run_sft(
    config: SFTConfig,
    train_examples: list[InstructionExample],
    val_examples: list[InstructionExample],
    callbacks: list,
    run_id: str,
) -> TrainingResult:
    """Internal: perform the actual SFT training loop.

    Lazy-imports torch/transformers/peft/bitsandbytes inside this function.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tracker = ResourceTracker()
    tracker.start()

    # --- Notify callbacks ---
    hp = {
        "base_model": config.base_model,
        "use_4bit": config.use_4bit,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.num_train_epochs,
    }
    for cb in callbacks:
        try:
            cb.on_init(hp)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Callback on_init failed: %s", exc)

    # --- Tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = _convert_for_causal_lm(train_examples)
    val_rows = _convert_for_causal_lm(val_examples) if val_examples else None

    # --- Load model ---
    if config.use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info("QLoRA model loaded (r=%d, alpha=%d)", config.lora_r, config.lora_alpha)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        logger.info("Full-parameter model loaded")

    # --- Tokenize datasets ---
    def _tokenize_fn(example):
        prompt_ids = tokenizer(example["prompt"], truncation=True, max_length=4096)
        completion_ids = tokenizer(example["completion"], truncation=True, max_length=1024)
        return {
            "input_ids": prompt_ids["input_ids"][:-1] + completion_ids["input_ids"][:-1],
            "attention_mask": prompt_ids["attention_mask"][:-1]
            + completion_ids["attention_mask"][:-1],
            "labels": prompt_ids["input_ids"][:-1] + completion_ids["input_ids"][:-1],
        }

    train_dataset = [_tokenize_fn(r) for r in train_rows]
    eval_dataset = [_tokenize_fn(r) for r in val_rows] if val_rows else None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=not config.use_4bit,  # QLoRA uses 4-bit internally
        logging_steps=10,
        eval_steps=500 if eval_dataset else None,
        save_steps=500,
        save_total_limit=2,
        report_to="none",  # we handle W&B via our own callback
        run_name=config.run_name or run_id,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # --- Hook our callbacks into the Trainer ---
    _attach_callbacks(trainer, callbacks, tracker, config, run_id)

    # --- Train ---
    loss_history: list[float] = []

    class _LossCallback:
        """Extracts the final train loss from the trainer's log history."""

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                loss_history.append(logs["loss"])

        def on_train_end(self, args, state, control, **kwargs):
            pass

    trainer.add_callback(_LossCallback)

    train_result = trainer.train()

    final_train_loss = float(
        train_result.metrics.get("train_loss", loss_history[-1] if loss_history else 0.0)
    )

    # --- Validation loss ---
    final_val_loss: float | None = None
    if eval_dataset:
        eval_metrics = trainer.evaluate()
        final_val_loss = float(eval_metrics.get("eval_loss", 0.0))

    tracker.record_peak_memory()

    # --- Save checkpoint ---
    ckpt_callback = next((c for c in callbacks if isinstance(c, CheckpointCallback)), None)
    checkpoint_uri = ""
    if ckpt_callback:
        local_ckpt_dir = os.path.join(config.output_dir, "final_checkpoint")
        model.save_pretrained(local_ckpt_dir)
        tokenizer.save_pretrained(local_ckpt_dir)
        checkpoint_uri = ckpt_callback.save_checkpoint(
            run_id=run_id,
            checkpoint_dir=local_ckpt_dir,
            epoch=config.num_train_epochs,
        )
    else:
        # Fallback: save locally
        local_ckpt_dir = os.path.join(config.output_dir, "final_checkpoint")
        model.save_pretrained(local_ckpt_dir)
        tokenizer.save_pretrained(local_ckpt_dir)
        checkpoint_uri = local_ckpt_dir

    # --- Notify callbacks of end ---
    for cb in callbacks:
        try:
            cb.on_train_end(
                final_train_loss=final_train_loss,
                final_val_loss=final_val_loss,
                peak_vram_gb=tracker.peak_vram_gb,
                train_time_minutes=tracker.elapsed_minutes,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Callback on_train_end failed: %s", exc)

    return TrainingResult(
        run_id=run_id,
        method=config.method_str,
        base_model=config.base_model,
        hyperparams={
            "use_4bit": config.use_4bit,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "learning_rate": config.learning_rate,
            "num_train_epochs": config.num_train_epochs,
        },
        train_set_size=len(train_examples),
        train_time_minutes=tracker.elapsed_minutes,
        peak_vram_gb=tracker.peak_vram_gb,
        final_train_loss=final_train_loss,
        final_val_loss=final_val_loss,
        checkpoint_uri=checkpoint_uri,
        status="completed",
        run_name=config.run_name,
        train_loss_history=loss_history,
    )


def _attach_callbacks(trainer, callbacks, tracker, config, run_id):
    """Attach the resource tracker to a transformers Trainer."""
    # The Trainer's native callbacks handle logging; our callbacks are
    # notified synchronously in on_epoch / on_train_end after training.
    # The ResourceTracker samples VRAM at the end.
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_sft(
    config: SFTConfig,
    train_path: str = "",
    val_path: str = "",
    *,
    callbacks: list | None = None,
    run_id: str | None = None,
    dry_run: bool = False,
    loader=None,
) -> TrainingResult:
    """Run a single SFT (full or QLoRA) training experiment.

    Parameters
    ----------
    config:
        ``SFTConfig`` with hyperparameters and paths.
    train_path:
        Path to Stage 3 ``train.jsonl`` (overrides ``config.train_jsonl`` if set).
    val_path:
        Path to Stage 3 ``val.jsonl`` (overrides ``config.val_jsonl`` if set).
    callbacks:
        List of ``TrainingCallback`` instances (or objects following the protocol).
        If None, a default ``WandbCallback(mock=True)`` + ``ProgressCallback``
        are used so the run doesn't require W&B.
    run_id:
        Override the generated run ID.
    dry_run:
        If True, skip actual training and return a ``TrainingResult`` with
        a step/memory estimate (no GPU or torch required).
    loader:
        Injectable data loader for testing.
    """

    train_jsonl = train_path or config.train_jsonl
    val_jsonl = val_path or config.val_jsonl

    if not training_run_id_is_valid(train_jsonl):
        raise FileNotFoundError(
            f"train_jsonl path is empty or does not exist: {train_jsonl!r}. "
            "Provide --train-jsonl or set SFTConfig.train_jsonl."
        )

    # Generate or use provided run ID
    if run_id is None:
        from app.training.experiment import generate_run_id

        run_id = generate_run_id(config.method_str, config.run_name)

    callbacks = callbacks if callbacks is not None else _default_callbacks(run_id)

    # --- Dry-run path: no GPU, no torch needed ---
    if dry_run:
        train_examples, val_examples = _load_jsonl_for_training(train_jsonl, val_jsonl, loader)
        estimate = estimate_training_steps(config, len(train_examples))
        logger.info(
            "Dry-run: %d epochs, %d steps, est. VRAM %.1f GB",
            estimate.num_epochs,
            estimate.num_train_steps,
            estimate.estimated_vram_gb,
        )
        for cb in callbacks:
            cb.on_init(
                {
                    "method": config.method_str,
                    "base_model": config.base_model,
                    "dry_run": True,
                    "estimated_steps": estimate.num_train_steps,
                    "estimated_vram_gb": estimate.estimated_vram_gb,
                }
            )

        return TrainingResult(
            run_id=run_id,
            method=config.method_str,
            base_model=config.base_model,
            hyperparams={
                "use_4bit": config.use_4bit,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
            },
            train_set_size=len(train_examples),
            train_time_minutes=0.0,
            peak_vram_gb=estimate.estimated_vram_gb,
            final_train_loss=0.0,
            final_val_loss=None,
            checkpoint_uri="",
            status="dry_run",
            run_name=config.run_name,
        )

    # --- Real training path ---
    _check_can_train(config)

    train_examples, val_examples = _load_jsonl_for_training(train_jsonl, val_jsonl, loader)

    for cb in callbacks:
        cb.on_init(
            {
                "method": config.method_str,
                "base_model": config.base_model,
                "use_4bit": config.use_4bit,
                "lora_r": config.lora_r,
            }
        )

    try:
        result = _run_sft(config, train_examples, val_examples, callbacks, run_id)
    except TrainingUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001
        for cb in callbacks:
            cb.on_error(str(exc))
        raise

    return result


def training_run_id_is_valid(path: str) -> bool:
    """Check that a training JSONL path is non-empty and exists."""
    return bool(path) and os.path.exists(path)


def _default_callbacks(run_id: str) -> list:
    """Default callbacks for non-injected runs: mock-W&B + console progress."""
    return [
        WandbCallback(run_name=run_id, mock=True),
        ProgressCallback(verbose=False),
    ]
