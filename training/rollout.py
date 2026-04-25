"""Episode rollout glue between a Policy and the env.

A Policy is anything with `.act(observation: str) -> ActionResult`. ActionResult
carries both the raw action text (what we send to the env) and any auxiliary
data the trainer needs (token ids + log-probs for an LLM policy, nothing for a
scripted policy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from env import ChaosEnv


@dataclass
class ActionResult:
    text: str
    aux: dict[str, Any] = field(default_factory=dict)


class Policy(Protocol):
    def act(self, observation: str) -> ActionResult: ...


@dataclass
class StepRecord:
    observation: str
    action_text: str
    reward: float
    done: bool
    info: dict[str, Any]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    seed: int
    steps: list[StepRecord]

    @property
    def episode_return(self) -> float:
        return sum(s.reward for s in self.steps)

    @property
    def length(self) -> int:
        return len(self.steps)

    @property
    def succeeded(self) -> bool:
        if not self.steps:
            return False
        last = self.steps[-1]
        return last.done and "VERIFY ok" in last.info.get("terminal_obs", "")


def rollout_episode(env: ChaosEnv, policy: Policy, seed: int,
                    difficulty: float = 1.0) -> Episode:
    env.difficulty = difficulty
    obs = env.reset(seed=seed)
    steps: list[StepRecord] = []
    while True:
        action = policy.act(obs)
        result = env.step(action.text)
        info = dict(result.info)
        if result.done:
            info["terminal_obs"] = result.observation
        steps.append(StepRecord(
            observation=obs,
            action_text=action.text,
            reward=result.reward,
            done=result.done,
            info=info,
            aux=action.aux,
        ))
        if result.done:
            break
        obs = result.observation
    return Episode(seed=seed, steps=steps)
