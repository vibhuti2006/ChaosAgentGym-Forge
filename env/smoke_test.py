"""Smoke test for ChaosEnv. Run as: python -m env.smoke_test

Three scripted policies exercise the env:
  1. blind-retry GET loop  -> should accumulate negative reward, never VERIFY
  2. premature VERIFY      -> should hit the false-claim penalty
  3. PUT-GET-PUT-VERIFY    -> reasonable strategy, should succeed often
"""
from __future__ import annotations

import statistics

from .chaos_env import ChaosEnv, TARGET_EMAIL, USER_ID


def _run(policy, n_episodes: int = 50, label: str = "") -> dict:
    returns, successes, lengths = [], 0, []
    for seed in range(n_episodes):
        env = ChaosEnv(seed=seed, difficulty=1.0)
        env.reset()
        episode_actions = policy()
        for action in episode_actions:
            result = env.step(action)
            if result.done:
                if "VERIFY ok" in result.observation:
                    successes += 1
                break
        returns.append(env.episode_return)
        lengths.append(env._steps)
    print(f"--- {label} (n={n_episodes}) ---")
    print(f"  mean return : {statistics.mean(returns):+.3f} "
          f"(stdev {statistics.stdev(returns):.3f})")
    print(f"  success rate: {successes / n_episodes:.0%}")
    print(f"  mean length : {statistics.mean(lengths):.2f}")
    return {"mean_return": statistics.mean(returns),
            "success_rate": successes / n_episodes}


def policy_blind_get():
    return [f'{{"op": "GET", "user": "{USER_ID}"}}'] * 10


def policy_premature_verify():
    return [
        f'{{"op": "PUT", "user": "{USER_ID}", "patch": {{"email": "{TARGET_EMAIL}"}}}}',
        f'{{"op": "VERIFY", "user": "{USER_ID}", "expect": {{"email": "{TARGET_EMAIL}"}}}}',
    ]


def policy_put_get_put_verify():
    return [
        f'{{"op": "PUT", "user": "{USER_ID}", "patch": {{"email": "{TARGET_EMAIL}"}}}}',
        f'{{"op": "GET", "user": "{USER_ID}"}}',
        f'{{"op": "PUT", "user": "{USER_ID}", "patch": {{"email": "{TARGET_EMAIL}"}}}}',
        f'{{"op": "GET", "user": "{USER_ID}"}}',
        f'{{"op": "VERIFY", "user": "{USER_ID}", "expect": {{"email": "{TARGET_EMAIL}"}}}}',
    ]


def policy_messy_llm_output():
    """Simulates a model that wraps JSON in prose."""
    return [
        f'Sure, I will start by updating the record. {{"op": "PUT", "user": "{USER_ID}", "patch": {{"email": "{TARGET_EMAIL}"}}}}',
        f'Now let me confirm. {{"op": "GET", "user": "{USER_ID}"}} that should do it',
        f'Looks good. {{"op": "VERIFY", "user": "{USER_ID}", "expect": {{"email": "{TARGET_EMAIL}"}}}}',
    ]


if __name__ == "__main__":
    _run(policy_blind_get, label="blind-GET-loop")
    _run(policy_premature_verify, label="PUT-then-immediate-VERIFY")
    _run(policy_put_get_put_verify, label="PUT-GET-PUT-GET-VERIFY")
    _run(policy_messy_llm_output, label="messy-LLM-style-output")

    print("\n--- example transcript (PUT-GET-PUT-GET-VERIFY, seed=3) ---")
    env = ChaosEnv(seed=3, difficulty=1.0)
    env.reset()
    for action in policy_put_get_put_verify():
        result = env.step(action)
        if result.done:
            break
    print(env.transcript())
