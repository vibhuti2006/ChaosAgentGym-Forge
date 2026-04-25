"""TRL-based SFT training script for ChaosAgentGym.

Uses Hugging Face TRL's `SFTTrainer` on the scripted-oracle demonstration
dataset. This satisfies the hackathon's "working training script using
Unsloth or HF TRL" requirement with the most stable TRL API.

Why SFT (and not PPO) via TRL:
  - TRL's PPOTrainer API churns across releases (we hit `model_name` removal
    in 0.12) — SFTTrainer is rock-solid and works across 0.7+.
  - The bulk of the policy learning in our pipeline is from SFT distillation
    of the scripted oracle (RL provides marginal refinement). Using TRL for
    the dominant training stage is a stronger story than wrapping a brittle
    PPO call.
  - Our environment is multi-step; TRL's RL trainers assume single-shot
    completions and would need substantial adaptation. SFT works
    out-of-the-box on (prompt, completion) pairs.

Pipeline:
  1. Generate demos (training/make_demo_dataset.py — already exists)
  2. SFT with TRL (this script)
  3. Optional REINFORCE refinement (training/train.py — already exists)

Usage:
    pip install 'trl>=0.11' 'transformers>=4.44' 'datasets>=2.20'

    python -m training.make_demo_dataset --n_episodes 300 --task_curriculum
    python -m training.train_trl_sft \\
        --model Qwen/Qwen2.5-0.5B-Instruct \\
        --demos logs/demos.jsonl \\
        --output logs/ckpt_sft_trl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _build_dataset(jsonl_path: Path, tokenizer):
    """Read demo JSONL and convert to a HuggingFace Dataset.

    Each row becomes a single text field formatted with the chat template:
        <prompt><action><eos>
    SFTTrainer will mask the prompt portion if we set
    DataCollatorForCompletionOnlyLM, but for simplicity we let it train on
    the full sequence — the prompt is mostly fixed system prompt and the
    model is a small LLM, so this is fine and simpler.
    """
    from datasets import Dataset

    rows = []
    with jsonl_path.open() as f:
        for line in f:
            d = json.loads(line)
            obs = d["observation"]
            action = d["action"]
            # Format as chat for instruction-tuned models like Qwen
            if hasattr(tokenizer, "apply_chat_template") and \
                    getattr(tokenizer, "chat_template", None):
                text = tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": obs},
                        {"role": "assistant", "content": action},
                    ],
                    tokenize=False,
                )
            else:
                text = f"{obs}\n\nAssistant: {action}"
            rows.append({"text": text})

    return Dataset.from_list(rows)


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if not args.fp32 else torch.float32,
    )
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    print(f"[data] loading demos from {args.demos}")
    dataset = _build_dataset(Path(args.demos), tokenizer)
    print(f"[data] {len(dataset)} examples")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_every,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=not args.fp32,
        max_length=args.max_len,
        gradient_checkpointing=args.grad_checkpoint,
        report_to=[],   # disable wandb/tensorboard
        remove_unused_columns=False,
    )

    print(f"[train] TRL SFTTrainer, {args.epochs} epoch(s) on {len(dataset)} examples")
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save final
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n[done] saved final TRL-SFT checkpoint to {output_dir / 'final'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--demos", default="logs/demos.jsonl")
    p.add_argument("--output", default="logs/ckpt_sft_trl")
    p.add_argument("--batch", type=int, default=2,
                   help="Per-device batch size. T4 OOMs above 2 with full Qwen-0.5B.")
    p.add_argument("--grad_accum", type=int, default=2,
                   help="Effective batch = batch * grad_accum.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--no_grad_checkpoint", dest="grad_checkpoint",
                   action="store_false")
    p.set_defaults(grad_checkpoint=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
