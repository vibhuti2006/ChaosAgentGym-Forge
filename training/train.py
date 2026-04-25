"""REINFORCE-with-baseline trainer for ChaosAgentGym.

Why not TRL? PPOTrainer's API has churned across TRL versions (0.7 -> 0.11 ->
0.12 are all incompatible) and is fragile under Colab session resets. A
hand-rolled REINFORCE loop is ~150 lines, has identical learning dynamics to
PPO without clipping for our small action set, and we can read every gradient
update. The reward and rollout interfaces are TRL-shaped (prompt/completion/
reward), so dropping TRL in later is mechanical.

Run on Colab T4:
    python -m training.train \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --episodes 200 --batch 8 --lr 1e-6
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW

from env import ChaosEnv, TaskDistribution

from .policies import LLMPolicy
from .rollout import Episode, rollout_episode


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

@dataclass
class TrainLog:
    csv_path: Path
    transcript_path: Path

    def __post_init__(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["update", "episode", "seed", "return", "length", "success",
                 "baseline", "loss"]
            )
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.write_text("")

    def log_episode(self, update: int, episode_idx: int, ep: Episode,
                    baseline: float, loss: float):
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                update, episode_idx, ep.seed,
                f"{ep.episode_return:.4f}", ep.length, int(ep.succeeded),
                f"{baseline:.4f}", f"{loss:.6f}",
            ])

    def log_transcript(self, update: int, ep: Episode):
        with self.transcript_path.open("a") as f:
            f.write(f"\n=== update {update} seed {ep.seed} "
                    f"return={ep.episode_return:+.3f} success={ep.succeeded} ===\n")
            for i, s in enumerate(ep.steps, 1):
                f.write(f"  [{i}] r={s.reward:+.2f} action={s.action_text[:140]}\n")


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------

def _reward_to_go(rewards: list[float], gamma: float = 1.0) -> list[float]:
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def _policy_loss(model, tokenizer, episode: Episode, advantages: list[float],
                 device) -> torch.Tensor:
    """Sum over steps of -advantage * sum(logpi(a_t | s_t)).

    We re-score the actions under the *current* policy (not the sampling
    policy) so gradients flow. The original sampling log-probs in `step.aux`
    are kept only for diagnostics.
    """
    losses = []
    for step, adv in zip(episode.steps, advantages):
        prompt_ids = step.aux["prompt_ids"].to(device)
        gen_ids = step.aux["gen_ids"].to(device)
        if gen_ids.numel() == 0:
            continue
        full_ids = torch.cat([prompt_ids, gen_ids], dim=0).unsqueeze(0)
        out = model(full_ids)
        # logits[t] predicts token t+1; we want logits at positions covering gen_ids
        prompt_len = prompt_ids.shape[0]
        # tokens we want to score live at indices [prompt_len, prompt_len + len(gen)-1]
        # the logits that *predict* those tokens are at indices [prompt_len-1, ..., prompt_len + len(gen) - 2]
        gen_len = gen_ids.shape[0]
        logits_slice = out.logits[0, prompt_len - 1: prompt_len - 1 + gen_len, :]
        logp = torch.log_softmax(logits_slice, dim=-1)
        token_logp = logp[range(gen_len), gen_ids]
        seq_logp = token_logp.sum()
        losses.append(-adv * seq_logp)
    if not losses:
        return torch.zeros((), device=device, requires_grad=True)
    return torch.stack(losses).mean()


def train(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32 if args.fp32 else torch.bfloat16,
    )
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.train()
    print(f"[load] device={device}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    optim = AdamW(model.parameters(), lr=args.lr)
    policy = LLMPolicy(model, tokenizer, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature)
    task_dist = TaskDistribution(seed=args.seed) if args.task_curriculum else None
    env = ChaosEnv(task_distribution=task_dist)
    if task_dist is not None:
        print("[env] task curriculum enabled (update_email + rollback_partial)")

    log = TrainLog(
        csv_path=Path(args.log_dir) / "rewards.csv",
        transcript_path=Path(args.log_dir) / "transcripts.txt",
    )

    baseline = 0.0
    baseline_alpha = 0.05  # EMA on episode returns
    n_updates = args.episodes // args.batch

    for upd in range(n_updates):
        t0 = time.time()
        episodes: list[Episode] = []
        # Difficulty ramp: start easier, get harder. Caps at 1.0.
        difficulty = min(1.0, 0.4 + 0.6 * upd / max(1, n_updates - 1)) \
            if args.difficulty_ramp else 1.0

        # ---- rollout ----
        model.eval()
        for b in range(args.batch):
            seed = upd * args.batch + b + args.rollout_seed_offset
            ep = rollout_episode(env, policy, seed=seed, difficulty=difficulty)
            episodes.append(ep)

        # ---- compute advantages (reward-to-go - baseline) ----
        all_advs = []
        for ep in episodes:
            rtg = _reward_to_go([s.reward for s in ep.steps], gamma=args.gamma)
            advs = [r - baseline for r in rtg]
            all_advs.append(advs)

        # normalize advantages across the batch for stability
        flat = [a for advs in all_advs for a in advs]
        if len(flat) > 1:
            mu = sum(flat) / len(flat)
            sigma = math.sqrt(sum((a - mu) ** 2 for a in flat) / len(flat)) + 1e-6
            all_advs = [[(a - mu) / sigma for a in advs] for advs in all_advs]

        # ---- gradient step ----
        model.train()
        optim.zero_grad()
        total_loss = torch.zeros((), device=device)
        for ep, advs in zip(episodes, all_advs):
            loss = _policy_loss(model, tokenizer, ep, advs, device)
            (loss / len(episodes)).backward()
            total_loss = total_loss + loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optim.step()

        loss_val = (total_loss / len(episodes)).item()

        # ---- update baseline & log ----
        for i, ep in enumerate(episodes):
            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * ep.episode_return
            log.log_episode(upd, upd * args.batch + i, ep, baseline, loss_val)
        # save one transcript per update for inspection
        log.log_transcript(upd, episodes[0])

        succ = sum(e.succeeded for e in episodes) / len(episodes)
        mean_ret = sum(e.episode_return for e in episodes) / len(episodes)
        dt = time.time() - t0
        print(f"upd {upd:03d} | diff={difficulty:.2f} | "
              f"mean_return={mean_ret:+.3f} | success={succ:.0%} | "
              f"baseline={baseline:+.3f} | loss={loss_val:+.4f} | {dt:.1f}s")

        # checkpoint
        if args.ckpt_every and (upd + 1) % args.ckpt_every == 0:
            ckpt_dir = Path(args.log_dir) / f"ckpt_upd{upd + 1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  saved checkpoint to {ckpt_dir}")

    # final save
    final = Path(args.log_dir) / "ckpt_final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"done. final checkpoint at {final}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout_seed_offset", type=int, default=10_000)
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--ckpt_every", type=int, default=5)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--difficulty_ramp", action="store_true")
    p.add_argument("--task_curriculum", action="store_true",
                   help="Sample tasks from {update_email, rollback_partial} "
                        "with varied targets — tests generalisation.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
