"""Pipeline test: rollout glue works without any ML deps.

Runs the random and scripted policies through `rollout_episode` and prints
aggregate stats. If this passes, the trainer's environment-side code is sound;
only the LLM-specific bits remain to validate on a GPU.
"""
from __future__ import annotations

import statistics

from env import ChaosEnv

from .policies import RandomPolicy, ScriptedPolicy
from .rollout import rollout_episode


def _summarize(label: str, episodes):
    rets = [e.episode_return for e in episodes]
    succ = sum(e.succeeded for e in episodes)
    print(f"--- {label} (n={len(episodes)}) ---")
    print(f"  mean return : {statistics.mean(rets):+.3f} "
          f"(stdev {statistics.stdev(rets):.3f})")
    print(f"  success rate: {succ / len(episodes):.0%}")
    print(f"  mean length : {statistics.mean(e.length for e in episodes):.2f}")


if __name__ == "__main__":
    env = ChaosEnv()

    rand_eps = [rollout_episode(env, RandomPolicy(seed=42 + s), seed=s) for s in range(50)]
    _summarize("RandomPolicy", rand_eps)

    # ScriptedPolicy is stateful per-episode — rebuild each time.
    scripted_eps = [rollout_episode(env, ScriptedPolicy(), seed=s) for s in range(50)]
    _summarize("ScriptedPolicy", scripted_eps)

    print("\nExample scripted episode (seed=3):")
    env2 = ChaosEnv()
    ep = rollout_episode(env2, ScriptedPolicy(), seed=3)
    for i, s in enumerate(ep.steps, 1):
        print(f"  [{i}] r={s.reward:+.2f} done={s.done} action={s.action_text[:80]}")
        print(f"       failure={s.info.get('failure')}")
    print(f"  return = {ep.episode_return:+.3f}, success = {ep.succeeded}")
