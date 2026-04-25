"""Held-out quantitative evaluation: base model vs trained checkpoint.

Runs N seeds (default 100) the model has never seen, reports a clean table:
success rate, mean return, mean steps, action mix, and a breakdown of how
often the agent encountered each failure type.

Usage:
    python -m eval.quantitative \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --trained logs/ckpt_final \
        --n_seeds 100 \
        --seed_offset 50000

The --seed_offset keeps eval seeds disjoint from training seeds (training uses
10000..). Random/scripted baselines are included for context — no GPU needed
for those.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from env import ChaosEnv, TaskDistribution, parse_action

from training.policies import RandomPolicy, ScriptedPolicy
from training.rollout import Episode, rollout_episode


@dataclass
class EvalRow:
    label: str
    n: int
    success_rate: float
    mean_return: float
    stdev_return: float
    mean_steps: float
    action_mix: dict[str, float]
    failures_per_episode: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "success_rate": self.success_rate,
            "mean_return": self.mean_return,
            "stdev_return": self.stdev_return,
            "mean_steps": self.mean_steps,
            "action_mix": self.action_mix,
            "failures_per_episode": self.failures_per_episode,
        }


def _summarize(label: str, episodes: list[Episode]) -> EvalRow:
    rets = [e.episode_return for e in episodes]
    lens = [e.length for e in episodes]
    succ = sum(e.succeeded for e in episodes) / len(episodes)
    action_counter: Counter[str] = Counter()
    failure_counter = 0
    total_actions = 0
    for ep in episodes:
        for s in ep.steps:
            op = parse_action(s.action_text).get("op", "INVALID")
            action_counter[op] += 1
            total_actions += 1
            if s.info.get("failure", "none") != "none":
                failure_counter += 1
    mix = {op: action_counter[op] / total_actions for op in action_counter}
    return EvalRow(
        label=label,
        n=len(episodes),
        success_rate=succ,
        mean_return=statistics.mean(rets),
        stdev_return=statistics.stdev(rets) if len(rets) > 1 else 0.0,
        mean_steps=statistics.mean(lens),
        action_mix=dict(sorted(mix.items())),
        failures_per_episode=failure_counter / len(episodes),
    )


def _seeds(n: int, offset: int) -> Iterable[int]:
    return range(offset, offset + n)


def _make_env(task_curriculum: bool, seed: int = 12345) -> ChaosEnv:
    if task_curriculum:
        return ChaosEnv(task_distribution=TaskDistribution(seed=seed))
    return ChaosEnv()


def _eval_policy_factory(make_policy, label: str, n: int, offset: int,
                         task_curriculum: bool) -> EvalRow:
    env = _make_env(task_curriculum)
    eps: list[Episode] = []
    for seed in _seeds(n, offset):
        eps.append(rollout_episode(env, make_policy(), seed=seed))
    return _summarize(label, eps)


def _eval_llm(model_path: str, label: str, n: int, offset: int,
              temperature: float, task_curriculum: bool) -> EvalRow:
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
    env = _make_env(task_curriculum)
    eps: list[Episode] = []
    for seed in _seeds(n, offset):
        eps.append(rollout_episode(env, policy, seed=seed))
    return _summarize(label, eps)


# ---------------------------------------------------------------------------
# pretty-print
# ---------------------------------------------------------------------------

def _print_table(rows: list[EvalRow]) -> None:
    cols = ["label", "n", "success", "mean_ret", "±stdev", "mean_steps",
            "fails/ep", "action mix"]
    widths = [max(28, max(len(r.label) for r in rows)), 5, 8, 9, 7, 10, 9, 40]
    header = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(header)
    print("-" * len(header))
    for r in rows:
        mix = ", ".join(f"{k}:{v:.0%}" for k, v in r.action_mix.items())
        line = " | ".join([
            r.label.ljust(widths[0]),
            str(r.n).ljust(widths[1]),
            f"{r.success_rate:.0%}".ljust(widths[2]),
            f"{r.mean_return:+.3f}".ljust(widths[3]),
            f"{r.stdev_return:.3f}".ljust(widths[4]),
            f"{r.mean_steps:.2f}".ljust(widths[5]),
            f"{r.failures_per_episode:.2f}".ljust(widths[6]),
            mix.ljust(widths[7]),
        ])
        print(line)


def _write_markdown(rows: list[EvalRow], path: Path) -> None:
    lines = ["# Quantitative evaluation\n",
             "Held-out seeds (disjoint from training).\n",
             "| Policy | N | Success | Mean return | ± stdev | Mean steps | Failures/ep | Action mix |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mix = ", ".join(f"{k}:{v:.0%}" for k, v in r.action_mix.items())
        lines.append(
            f"| {r.label} | {r.n} | {r.success_rate:.0%} | "
            f"{r.mean_return:+.3f} | {r.stdev_return:.3f} | "
            f"{r.mean_steps:.2f} | {r.failures_per_episode:.2f} | {mix} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main(args):
    rows: list[EvalRow] = []

    # Fresh seed per episode so RandomPolicy actually explores the action space
    # (without this it picks the same fixed sequence every rollout).
    _rand_counter = [0]

    def _make_random():
        _rand_counter[0] += 1
        return RandomPolicy(seed=_rand_counter[0])

    rows.append(_eval_policy_factory(
        _make_random, "RandomPolicy", args.n_seeds, args.seed_offset,
        args.task_curriculum,
    ))
    rows.append(_eval_policy_factory(
        ScriptedPolicy, "ScriptedPolicy (oracle baseline)",
        args.n_seeds, args.seed_offset, args.task_curriculum,
    ))

    if args.base:
        rows.append(_eval_llm(
            args.base, f"LLM base ({Path(args.base).name})",
            args.n_seeds, args.seed_offset, args.temperature,
            args.task_curriculum,
        ))
    if args.trained:
        rows.append(_eval_llm(
            args.trained, f"LLM trained ({Path(args.trained).name})",
            args.n_seeds, args.seed_offset, args.temperature,
            args.task_curriculum,
        ))

    print()
    _print_table(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(rows, out)
    out.with_suffix(".json").write_text(
        json.dumps([r.to_dict() for r in rows], indent=2)
    )
    print(f"\nwrote {out} and {out.with_suffix('.json')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=None,
                   help="HF model id or path; omit to skip LLM rows")
    p.add_argument("--trained", default=None,
                   help="path to trained checkpoint; omit to skip")
    p.add_argument("--n_seeds", type=int, default=100)
    p.add_argument("--seed_offset", type=int, default=50_000,
                   help="must be disjoint from training seeds (training "
                        "uses --rollout_seed_offset, default 10000)")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--out", default="logs/quantitative.md")
    p.add_argument("--task_curriculum", action="store_true",
                   help="Eval on the mixed update_email + rollback_partial "
                        "distribution (tests generalisation).")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
