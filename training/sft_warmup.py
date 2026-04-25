"""Optional SFT warmup before REINFORCE.

Why this exists: a 0.5B base model often emits malformed JSON or refuses to
follow our action grammar on the first turn, leaving REINFORCE without any
positive trajectories to amplify. ~1 epoch over a few hundred scripted
demonstrations is enough to make the model emit parseable actions, after
which RL can do its job.

Loss: standard causal LM cross-entropy on the action tokens only (the long
prompt is masked out — we don't want to make the model better at reproducing
the system prompt).

Usage:
    # 1. generate demos (CPU, ~5s)
    python -m training.make_demo_dataset --n_episodes 300

    # 2. warmup (T4: ~5 min for 1 epoch on 1290 pairs)
    python -m training.sft_warmup \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --demos logs/demos.jsonl \
        --epochs 1 --lr 2e-5 \
        --out logs/ckpt_sft

    # 3. REINFORCE *from the warmup checkpoint*, not the base
    python -m training.train --model logs/ckpt_sft --episodes 200 --batch 8
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


class DemoDataset(Dataset):
    def __init__(self, path: Path):
        with path.open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, str]:
        return self.rows[i]


def _format_prompt(tokenizer, observation: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and \
            getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": observation}],
            tokenize=False, add_generation_prompt=True,
        )
    return observation + "\nAction: "


def _collate(batch, tokenizer, max_len: int):
    """Build (input_ids, labels) where labels are -100 on prompt tokens."""
    input_ids_list, labels_list = [], []
    for row in batch:
        prompt = _format_prompt(tokenizer, row["observation"])
        action = row["action"].strip()
        # action tokens get appended; we want to learn to predict THEM only.
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        action_ids = tokenizer(action, add_special_tokens=False).input_ids
        eos = tokenizer.eos_token_id
        if eos is not None:
            action_ids = action_ids + [eos]
        ids = (prompt_ids + action_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + action_ids)[:max_len]
        input_ids_list.append(ids)
        labels_list.append(labels)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    max_in_batch = max(len(x) for x in input_ids_list)
    input_ids = torch.full((len(batch), max_in_batch), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_in_batch), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), max_in_batch), dtype=torch.long)
    for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
        attn[i, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[load] {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.fp32 else torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    # Gradient checkpointing trades ~30% step time for ~50% less activation
    # memory — required to fit a 0.5B model + AdamW + ~700-tok contexts on
    # a free T4 (15GB). Set use_cache=False or it conflicts with checkpointing.
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    ds = DemoDataset(Path(args.demos))
    print(f"[data] {len(ds)} examples")
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: _collate(b, tok, args.max_len),
    )

    optim = AdamW(model.parameters(), lr=args.lr)
    n_steps = args.epochs * len(loader)
    print(f"[train] {args.epochs} epoch(s) x {len(loader)} batches = {n_steps} steps")

    step = 0
    losses_window = []
    for epoch in range(args.epochs):
        for batch in loader:
            t0 = time.time()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            losses_window.append(loss.item())
            losses_window = losses_window[-50:]
            if step % args.log_every == 0:
                avg = sum(losses_window) / len(losses_window)
                print(f"epoch {epoch} step {step:04d}/{n_steps} | "
                      f"loss={loss.item():.4f} (avg{len(losses_window)}={avg:.4f}) "
                      f"| {time.time() - t0:.2f}s")
            step += 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved warmup checkpoint to {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--demos", default="logs/demos.jsonl")
    p.add_argument("--out", default="logs/ckpt_sft")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=2,
                   help="T4 with 0.5B model OOMs at batch=4 due to vocab-sized "
                        "logits tensor. Default 2 fits comfortably.")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_len", type=int, default=768)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--no_grad_checkpoint", dest="grad_checkpoint",
                   action="store_false",
                   help="Disable gradient checkpointing (faster on big GPUs).")
    p.set_defaults(grad_checkpoint=True)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
