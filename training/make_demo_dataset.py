"""Generate (observation, action) pairs from the scripted oracle for SFT warmup.

A 0.5B base model often can't reliably emit our JSON action format on step 0,
which leaves REINFORCE with no signal to amplify. A short SFT pass on
demonstrations from `ScriptedPolicy` fixes that — after ~1 epoch on a few
hundred examples, the model emits parseable actions and RL can take over.

We only keep transitions from *successful* episodes so we don't teach the
model the oracle's mistakes (it succeeds ~89% of the time).

Usage:
    python -m training.make_demo_dataset --n_episodes 300 --out logs/demos.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from env import ChaosEnv, TaskDistribution

from .policies import ScriptedPolicy
from .rollout import rollout_episode


def main(args):
    task_dist = TaskDistribution(seed=args.seed_offset) if args.task_curriculum else None
    env = ChaosEnv(task_distribution=task_dist)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_eps = 0
    n_kept = 0
    n_pairs = 0
    with out_path.open("w") as f:
        for seed in range(args.seed_offset, args.seed_offset + args.n_episodes):
            ep = rollout_episode(env, ScriptedPolicy(), seed=seed)
            n_eps += 1
            if not ep.succeeded:
                continue
            n_kept += 1
            for step in ep.steps:
                f.write(json.dumps({
                    "observation": step.observation,
                    "action": step.action_text,
                }) + "\n")
                n_pairs += 1

    print(f"wrote {n_pairs} (obs, action) pairs from "
          f"{n_kept}/{n_eps} successful episodes -> {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=300)
    p.add_argument("--seed_offset", type=int, default=90_000,
                   help="must be disjoint from train (10000+) and eval (50000+)")
    p.add_argument("--out", default="logs/demos.jsonl")
    p.add_argument("--task_curriculum", action="store_true",
                   help="Generate demos from both update_email and rollback_partial.")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
