"""Deterministic, seeded failure injection for the mock API.

Three failure modes:
  - SERVICE_UNAVAILABLE (503): the API call is dropped, no state change.
  - STALE_READ:                a GET returns the value from before the last PUT.
  - PARTIAL_WRITE:             a PUT updates the *visible* store but not the
                               *ground truth* used to score VERIFY.

Failure rates ramp with `difficulty` in [0, 1]. We expose a single Injector
object so the env can carry it through a rollout and reproducibility is just a
matter of passing the same seed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Failure(str, Enum):
    NONE = "none"
    SERVICE_UNAVAILABLE = "service_unavailable"
    STALE_READ = "stale_read"
    PARTIAL_WRITE = "partial_write"


@dataclass
class InjectorConfig:
    p_503: float = 0.25
    p_stale: float = 0.15
    p_partial: float = 0.15

    def scaled(self, difficulty: float) -> "InjectorConfig":
        # difficulty=0 -> 40% of nominal, difficulty=1 -> 100% of nominal.
        s = 0.4 + 0.6 * max(0.0, min(1.0, difficulty))
        return InjectorConfig(self.p_503 * s, self.p_stale * s, self.p_partial * s)


class ChaosInjector:
    """One injector per episode. Seeded for reproducibility."""

    def __init__(self, seed: int, config: InjectorConfig | None = None,
                 difficulty: float = 1.0):
        self.rng = random.Random(seed)
        self.config = (config or InjectorConfig()).scaled(difficulty)

    def roll_get(self) -> Failure:
        r = self.rng.random()
        if r < self.config.p_503:
            return Failure.SERVICE_UNAVAILABLE
        if r < self.config.p_503 + self.config.p_stale:
            return Failure.STALE_READ
        return Failure.NONE

    def roll_put(self) -> Failure:
        r = self.rng.random()
        if r < self.config.p_503:
            return Failure.SERVICE_UNAVAILABLE
        if r < self.config.p_503 + self.config.p_partial:
            return Failure.PARTIAL_WRITE
        return Failure.NONE
