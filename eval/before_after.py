"""Generate side-by-side transcripts: untrained vs trained model on fixed seeds.

Usage:
    python -m eval.before_after \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --trained logs/ckpt_final \
        --seeds 7 13 21 34 \
        --out logs/before_after.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

from env import ChaosEnv

from training.policies import LLMPolicy
from training.rollout import Episode, rollout_episode


def _load(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16
    )
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()
    return model, tok


def _format(label: str, ep: Episode) -> str:
    out = [f"### {label} (seed={ep.seed}, return={ep.episode_return:+.3f}, "
           f"success={ep.succeeded})\n"]
    for i, s in enumerate(ep.steps, 1):
        out.append(
            f"  step {i} | r={s.reward:+.2f} | failure={s.info.get('failure'):<22} "
            f"| action={s.action_text[:140]}"
        )
        if s.done:
            out.append(f"           terminal: {s.info.get('terminal_obs', '')}")
    out.append("")
    return "\n".join(out)


def main(args):
    base_model, base_tok = _load(args.base)
    base_policy = LLMPolicy(base_model, base_tok, temperature=args.temperature)

    trained_model, trained_tok = _load(args.trained)
    trained_policy = LLMPolicy(trained_model, trained_tok, temperature=args.temperature)

    env = ChaosEnv()
    out_lines = ["# Before vs After — ChaosAgentGym\n",
                 f"Base model: `{args.base}`  ",
                 f"Trained checkpoint: `{args.trained}`  ",
                 f"Sampling temperature: {args.temperature}\n"]

    base_succ = trained_succ = 0
    for seed in args.seeds:
        ep_base = rollout_episode(env, base_policy, seed=seed)
        ep_trained = rollout_episode(env, trained_policy, seed=seed)
        base_succ += int(ep_base.succeeded)
        trained_succ += int(ep_trained.succeeded)

        out_lines.append(f"\n---\n## Seed {seed}\n")
        out_lines.append(_format("BEFORE (untrained)", ep_base))
        out_lines.append(_format("AFTER (trained)", ep_trained))

    n = len(args.seeds)
    out_lines.insert(4, f"Success rate: base **{base_succ}/{n}** vs trained "
                        f"**{trained_succ}/{n}**\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out_lines))
    print(f"wrote {args.out}")
    print(f"base success: {base_succ}/{n}, trained success: {trained_succ}/{n}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--trained", default="logs/ckpt_final")
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[7, 13, 21, 34, 55, 89])
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--out", default="logs/before_after.md")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
