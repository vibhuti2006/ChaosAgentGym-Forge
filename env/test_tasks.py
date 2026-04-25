"""Smoke test for task variety.

Runs the scripted oracle through 100 episodes with TaskDistribution sampling
both update_email and rollback_partial. Expectation: oracle still hits ~85%
success because the scripted plan (PUT target -> GET -> PUT -> GET -> VERIFY)
is task-shape-agnostic — it uses whatever target the env exposes.

Run as: python -m env.test_tasks
"""
from __future__ import annotations

import statistics
from collections import Counter

from env import (
    ChaosEnv,
    TaskDistribution,
    rollback_partial_task,
    update_email_task,
)

from training.policies import ScriptedPolicy
from training.rollout import rollout_episode


def _summarise(label, eps):
    rets = [e.episode_return for e in eps]
    succ = sum(e.succeeded for e in eps) / len(eps)
    print(f"--- {label} (n={len(eps)}) ---")
    print(f"  mean return : {statistics.mean(rets):+.3f}")
    print(f"  success rate: {succ:.0%}")
    print(f"  mean length : {statistics.mean(e.length for e in eps):.2f}")


if __name__ == "__main__":
    # Single fixed task — should match the legacy numbers.
    env = ChaosEnv(task=update_email_task())
    eps = [rollout_episode(env, ScriptedPolicy(), seed=s) for s in range(50)]
    _summarise("update_email only (fixed)", eps)

    env = ChaosEnv(task=rollback_partial_task())
    eps = [rollout_episode(env, ScriptedPolicy(), seed=s) for s in range(50)]
    _summarise("rollback_partial only (fixed)", eps)

    # Mixed task distribution — agent must generalise.
    dist = TaskDistribution(p_update=0.5, p_rollback=0.5, seed=99)
    env = ChaosEnv(task_distribution=dist)
    eps = [rollout_episode(env, ScriptedPolicy(), seed=s) for s in range(100)]
    _summarise("mixed distribution (50/50)", eps)

    # What was actually sampled?
    sampled = Counter()
    for s in range(100):
        sampled[dist.sample(s).name] += 1
    print(f"  sampled task mix: {dict(sampled)}")

    # Spot-check a rollback episode end-to-end.
    print("\n--- example rollback_partial episode (seed=7) ---")
    env = ChaosEnv(task=rollback_partial_task(), seed=7)
    ep = rollout_episode(env, ScriptedPolicy(), seed=7)
    print(f"task: {env.task.description}")
    for i, s in enumerate(ep.steps, 1):
        print(f"  [{i}] r={s.reward:+.2f} action={s.action_text[:120]}")
    print(f"  return={ep.episode_return:+.3f} success={ep.succeeded}")
