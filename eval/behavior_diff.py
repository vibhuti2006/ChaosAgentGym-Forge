"""Compare *how* the base and trained models behave, not just whether they succeed.

This produces the per-policy behavioral fingerprint that goes into the demo:

  - frequency of valid JSON emissions
  - frequency of premature VERIFY (before any PUT)
  - frequency of "defended" VERIFY (>= 2 PUTs before VERIFY)
  - rate of action-change after a failed step (vs. blind retry)
  - top-3 most common full action sequences

The story we want the numbers to tell: trained model emits valid JSON, defends
against partial writes by double-PUTting, and changes strategy after failures.

Usage:
    python -m eval.behavior_diff \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --trained logs/ckpt_final \
        --n_seeds 50 \
        --out logs/behavior_diff.md
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from env import ChaosEnv, parse_action

from training.policies import RandomPolicy, ScriptedPolicy
from training.rollout import Episode, rollout_episode


@dataclass
class BehaviorReport:
    label: str
    n_episodes: int
    valid_json_rate: float          # fraction of actions that parsed
    premature_verify_rate: float    # fraction of episodes that VERIFY before any PUT
    defended_verify_rate: float     # fraction of episodes that VERIFY after >= 2 PUTs
    recovery_rate: float            # P(action_t+1 != action_t | step t failed)
    success_rate: float
    mean_return: float
    top_sequences: list[tuple[str, int]]  # (action sequence, count)


def _ops_of(ep: Episode) -> list[str]:
    return [parse_action(s.action_text).get("op", "INVALID") for s in ep.steps]


def _analyze(label: str, episodes: list[Episode]) -> BehaviorReport:
    n = len(episodes)
    valid = invalid = 0
    premature = defended = 0
    recovery_num = recovery_den = 0
    succ = 0
    rets = []
    seq_counter: Counter[str] = Counter()

    for ep in episodes:
        ops = _ops_of(ep)
        for op in ops:
            if op == "INVALID":
                invalid += 1
            else:
                valid += 1

        # premature VERIFY: VERIFY appears before any PUT
        verify_idx = next((i for i, op in enumerate(ops) if op == "VERIFY"), None)
        first_put = next((i for i, op in enumerate(ops) if op == "PUT"), None)
        if verify_idx is not None and (first_put is None or first_put > verify_idx):
            premature += 1

        # defended VERIFY: >= 2 PUTs before the VERIFY
        if verify_idx is not None:
            puts_before = sum(1 for op in ops[:verify_idx] if op == "PUT")
            if puts_before >= 2:
                defended += 1

        # recovery rate: at any step where the env reported a failure,
        # did the *next* action differ from the failing action?
        for i, step in enumerate(ep.steps[:-1]):
            if step.info.get("failure", "none") != "none":
                recovery_den += 1
                if ops[i] != ops[i + 1]:
                    recovery_num += 1

        if ep.succeeded:
            succ += 1
        rets.append(ep.episode_return)

        seq_counter[" -> ".join(ops)] += 1

    total_actions = max(1, valid + invalid)
    return BehaviorReport(
        label=label,
        n_episodes=n,
        valid_json_rate=valid / total_actions,
        premature_verify_rate=premature / n,
        defended_verify_rate=defended / n,
        recovery_rate=(recovery_num / recovery_den) if recovery_den else float("nan"),
        success_rate=succ / n,
        mean_return=sum(rets) / n,
        top_sequences=seq_counter.most_common(3),
    )


def _print_report(reports: list[BehaviorReport]) -> None:
    cols = ["metric", *[r.label for r in reports]]
    rows = [
        ("episodes", *[str(r.n_episodes) for r in reports]),
        ("success rate", *[f"{r.success_rate:.0%}" for r in reports]),
        ("mean return", *[f"{r.mean_return:+.3f}" for r in reports]),
        ("valid JSON %", *[f"{r.valid_json_rate:.0%}" for r in reports]),
        ("premature VERIFY %", *[f"{r.premature_verify_rate:.0%}" for r in reports]),
        ("defended VERIFY % (>=2 PUTs)", *[f"{r.defended_verify_rate:.0%}" for r in reports]),
        ("recovery rate after failure", *[
            f"{r.recovery_rate:.0%}" if r.recovery_rate == r.recovery_rate else "n/a"
            for r in reports
        ]),
    ]
    widths = [max(len(c) for c in col) for col in zip(*([cols, *rows]))]
    print(" | ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(r, widths)))

    print("\nTop action sequences per policy:")
    for r in reports:
        print(f"  {r.label}:")
        for seq, count in r.top_sequences:
            print(f"    {count:>3}x  {seq[:120]}")


def _to_markdown(reports: list[BehaviorReport]) -> str:
    headers = ["Metric", *[r.label for r in reports]]
    sep = ["---"] * len(headers)
    rows = [
        ["Episodes", *[str(r.n_episodes) for r in reports]],
        ["Success rate", *[f"{r.success_rate:.0%}" for r in reports]],
        ["Mean return", *[f"{r.mean_return:+.3f}" for r in reports]],
        ["Valid JSON %", *[f"{r.valid_json_rate:.0%}" for r in reports]],
        ["Premature VERIFY %", *[f"{r.premature_verify_rate:.0%}" for r in reports]],
        ["Defended VERIFY % (≥2 PUTs)", *[f"{r.defended_verify_rate:.0%}" for r in reports]],
        ["Recovery after failure",
         *[f"{r.recovery_rate:.0%}" if r.recovery_rate == r.recovery_rate else "n/a"
           for r in reports]],
    ]
    lines = ["# Behavioral diff — ChaosAgentGym\n",
             "How the trained agent's *behavior* changed, not just its score.\n",
             "| " + " | ".join(headers) + " |",
             "| " + " | ".join(sep) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")

    lines.append("\n## Top action sequences\n")
    for r in reports:
        lines.append(f"**{r.label}**\n")
        for seq, count in r.top_sequences:
            lines.append(f"- `{count}x`  {seq}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def _eval_llm_episodes(model_path: str, n: int, offset: int,
                       temperature: float) -> list[Episode]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from training.policies import LLMPolicy

    print(f"[load] {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model.to(device)
    model.eval()
    policy = LLMPolicy(model, tok, temperature=temperature)
    env = ChaosEnv()
    return [rollout_episode(env, policy, seed=s) for s in range(offset, offset + n)]


def main(args):
    reports: list[BehaviorReport] = []

    # baselines
    env = ChaosEnv()
    rand_eps = [rollout_episode(env, RandomPolicy(seed=s + 1), seed=s)
                for s in range(args.seed_offset, args.seed_offset + args.n_seeds)]
    reports.append(_analyze("RandomPolicy", rand_eps))
    scripted_eps = [rollout_episode(env, ScriptedPolicy(), seed=s)
                    for s in range(args.seed_offset, args.seed_offset + args.n_seeds)]
    reports.append(_analyze("ScriptedPolicy (oracle)", scripted_eps))

    if args.base:
        eps = _eval_llm_episodes(args.base, args.n_seeds, args.seed_offset, args.temperature)
        reports.append(_analyze(f"LLM base ({Path(args.base).name})", eps))
    if args.trained:
        eps = _eval_llm_episodes(args.trained, args.n_seeds, args.seed_offset, args.temperature)
        reports.append(_analyze(f"LLM trained ({Path(args.trained).name})", eps))

    print()
    _print_report(reports)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_to_markdown(reports))
    print(f"\nwrote {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=None)
    p.add_argument("--trained", default=None)
    p.add_argument("--n_seeds", type=int, default=50)
    p.add_argument("--seed_offset", type=int, default=60_000)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--out", default="logs/behavior_diff.md")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
