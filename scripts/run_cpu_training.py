"""Run real CPU-based LoRA training on Qwen 1.5B and save results."""
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
import torch
torch.set_num_threads(8)

from app.training.config import SFTConfig
from app.training.trainer_sft import run_sft

def main():
    config = SFTConfig(
        base_model='Qwen/Qwen2.5-Coder-1.5B-Instruct',
        output_dir='./output/stage5/qwen_lora_cpu',
        use_4bit=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=2e-4,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        train_jsonl='output/stage3/train.jsonl',
        val_jsonl='output/stage3/val.jsonl',
        run_name='qwen-1.5b-lora-cpu',
    )

    print('Starting real CPU training...')
    start = time.time()
    result = run_sft(config, dry_run=False)
    elapsed = time.time() - start

    print(f'Training complete in {elapsed:.1f}s!')
    print(f'  Status: {result.status}')
    print(f'  Train loss: {result.final_train_loss:.4f}')
    print(f'  Val loss: {result.final_val_loss}')
    print(f'  Train time: {result.train_time_minutes:.2f} min')
    print(f'  Loss history: {result.train_loss_history}')
    print(f'  Checkpoint: {result.checkpoint_uri}')

    # Save training result as JSON
    result_dict = {
        'run_id': result.run_id,
        'method': result.method,
        'base_model': result.base_model,
        'hyperparams': result.hyperparams,
        'train_set_size': result.train_set_size,
        'train_time_minutes': result.train_time_minutes,
        'peak_vram_gb': result.peak_vram_gb,
        'final_train_loss': result.final_train_loss,
        'final_val_loss': result.final_val_loss,
        'checkpoint_uri': result.checkpoint_uri,
        'status': result.status,
        'run_name': result.run_name,
        'train_loss_history': result.train_loss_history,
    }
    out_path = Path('output/stage5/training_result.json')
    out_path.write_text(json.dumps(result_dict, indent=2))
    print(f'Saved result to {out_path}')

    return result_dict

if __name__ == '__main__':
    main()
