"""ChaosAgentGym: an OpenEnv-compatible env where API tools misbehave.

Task (v0): "Update user `u_42`'s email to `<new_email>` and confirm it stuck."

Action space (one per step, emitted as a single line of JSON):
  {"op": "GET",    "user": "u_42"}
  {"op": "PUT",    "user": "u_42", "patch": {"email": "..."}}
  {"op": "VERIFY", "user": "u_42", "expect": {"email": "..."}}     # terminal
  {"op": "RETRY"}                                                  # no-op

Reward (dense, designed to discourage blind retries and reward recovery):
  per-step  -0.05                       action cost
            -0.20  if action == prev    blind retry penalty
            +0.30  first time the agent picks a *different* action after a
                   failed step (recovery bonus, fires at most once)
            -0.50  on VERIFY when it disagrees with ground truth (false claim)
  terminal  +1.00  on VERIFY that matches ground truth
            -0.50  on VERIFY that doesn't match
             0.00  if step budget exhausted with no VERIFY
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .chaos_injector import ChaosInjector, Failure, InjectorConfig
from .mock_api import MockUserApi
from .tasks import Task, TaskDistribution, update_email_task


# Backwards-compat constants (older callers / docs reference these).
USER_ID = "u_42"
INITIAL_EMAIL = "old@example.com"
TARGET_EMAIL = "new@example.com"
MAX_STEPS = 8


SYSTEM_PROMPT_HEADER = """You are an API-using agent. Your job is described in the TASK block below.

The API is unreliable. Calls may:
  - return HTTP 503 (transient, retry the same call)
  - return a STALE value on GET (the cached pre-update value)
  - return 200 on PUT but only PARTIALLY apply the write (visible store updates, ground truth does not — re-PUT to fix)

You may emit ONE action per step. Respond with a single line of JSON, nothing else. Allowed ops:
  {"op": "GET",    "user": "<user_id>"}
  {"op": "PUT",    "user": "<user_id>", "patch": {"<field>": "<value>"}}
  {"op": "VERIFY", "user": "<user_id>", "expect": {"<field>": "<value>"}}
  {"op": "RETRY"}

VERIFY ends the episode. Only VERIFY when you are confident the change actually persisted (a single GET after a PUT is not enough — the GET could be stale or the PUT could be partial).

Strategy hint: PUT (or repair), then GET to confirm. If GET disagrees, PUT again. If you suspect a partial write (GET says yes but you've only done one PUT), PUT once more before VERIFY.
"""


# Kept for callers that imported the constant directly.
SYSTEM_PROMPT = SYSTEM_PROMPT_HEADER + "\n" + update_email_task().system_prompt_tail()


@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Transition:
    action_text: str
    parsed_op: str
    observation: str
    reward: float
    failure: Failure


# --- action parsing ----------------------------------------------------------

_JSON_LINE = re.compile(r"\{[^\n]*\}")


def parse_action(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model emission. Tolerant by design.

    Returns {"op": "INVALID", "raw": text} if nothing parseable is found.
    """
    if not text:
        return {"op": "INVALID", "raw": ""}
    candidates = _JSON_LINE.findall(text)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and "op" in obj:
                obj["op"] = str(obj["op"]).upper()
                return obj
        except json.JSONDecodeError:
            continue
    return {"op": "INVALID", "raw": text[:120]}


# --- environment -------------------------------------------------------------

class ChaosEnv:
    """OpenEnv-compatible environment. Single-agent, text-in / text-out."""

    def __init__(self, seed: int = 0, difficulty: float = 1.0,
                 injector_config: InjectorConfig | None = None,
                 max_steps: int = MAX_STEPS,
                 task: Task | None = None,
                 task_distribution: TaskDistribution | None = None):
        """If both `task` and `task_distribution` are None, the env defaults to
        the canonical update_email task — keeps old call sites working.

        If `task_distribution` is supplied, each `reset()` samples a fresh task
        (forcing the agent to generalise the recovery pattern instead of
        memorising a specific email value).
        """
        self.seed = seed
        self.difficulty = difficulty
        self.injector_config = injector_config
        self.max_steps = max_steps
        self.task_distribution = task_distribution
        self._fixed_task = task
        self._reset_state()

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> str:
        if seed is not None:
            self.seed = seed
        self._reset_state()
        return self._render_observation()

    def step(self, action_text: str) -> StepResult:
        if self._done:
            raise RuntimeError("step() called on terminated episode; call reset()")

        parsed = parse_action(action_text)
        op = parsed.get("op", "INVALID")
        prev_op = self.history[-1].parsed_op if self.history else None

        # baseline action cost
        reward = -0.05
        info: dict[str, Any] = {"parsed": parsed}

        # blind-retry penalty
        if prev_op is not None and op == prev_op and op in {"GET", "PUT", "RETRY"}:
            reward -= 0.20
            info["blind_retry"] = True

        # one-time recovery bonus: agent changed action after a failure
        if (
            not self._recovery_fired
            and self._last_failure is not Failure.NONE
            and prev_op is not None
            and op != prev_op
            and op != "INVALID"
        ):
            reward += 0.30
            self._recovery_fired = True
            info["recovery_bonus"] = True

        observation, op_failure, terminated, terminal_reward = self._dispatch(parsed)
        reward += terminal_reward

        self._last_failure = op_failure
        self.history.append(_Transition(
            action_text=action_text,
            parsed_op=op,
            observation=observation,
            reward=reward,
            failure=op_failure,
        ))
        self._steps += 1

        # budget exhaustion (no terminal reward — already 0)
        if not terminated and self._steps >= self.max_steps:
            terminated = True
            info["budget_exhausted"] = True

        self._done = terminated
        info["failure"] = op_failure.value
        info["step"] = self._steps
        info["episode_return"] = self.episode_return + reward

        self.episode_return += reward
        return StepResult(
            observation=self._render_observation() if not terminated else observation,
            reward=reward,
            done=terminated,
            info=info,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        # Pick this episode's task: explicit Task wins over distribution wins
        # over the legacy default.
        if self._fixed_task is not None:
            self.task = self._fixed_task
        elif self.task_distribution is not None:
            self.task = self.task_distribution.sample(self.seed)
        else:
            self.task = update_email_task()

        injector = ChaosInjector(seed=self.seed, config=self.injector_config,
                                 difficulty=self.difficulty)
        # Build the API with the task's *truth* state, then patch the visible
        # store separately so rollback-style tasks (visible != truth at start)
        # work correctly.
        self.api = MockUserApi.with_user(
            injector, self.task.user_id, dict(self.task.initial_truth),
        )
        if self.task.initial_visible != self.task.initial_truth:
            self.api.visible[self.task.user_id] = dict(self.task.initial_visible)
            self.api._prev_visible[self.task.user_id] = dict(self.task.initial_visible)

        self.history: list[_Transition] = []
        self._steps = 0
        self._done = False
        self._last_failure = Failure.NONE
        self._recovery_fired = False
        self.episode_return = 0.0

    def _dispatch(self, parsed: dict[str, Any]) -> tuple[str, Failure, bool, float]:
        """Returns (observation, failure_kind, terminated, terminal_reward)."""
        op = parsed.get("op", "INVALID")

        if op == "GET":
            resp = self.api.get(parsed.get("user", self.task.user_id))
            return resp.to_text(), resp.failure, False, 0.0

        if op == "PUT":
            patch = parsed.get("patch") or {}
            if not isinstance(patch, dict) or not patch:
                return "ERROR: PUT requires a non-empty 'patch' dict", Failure.NONE, False, 0.0
            resp = self.api.put(parsed.get("user", self.task.user_id), patch)
            return resp.to_text(), resp.failure, False, 0.0

        if op == "VERIFY":
            expect = parsed.get("expect") or {}
            if not isinstance(expect, dict) or not expect:
                return "ERROR: VERIFY requires an 'expect' dict", Failure.NONE, True, -0.5
            ok = self.api.verify_truth(parsed.get("user", self.task.user_id), expect)
            if ok:
                return "VERIFY ok — change confirmed against ground truth", Failure.NONE, True, 1.0
            return "VERIFY failed — ground truth does not match", Failure.NONE, True, -0.5

        if op == "RETRY":
            return "RETRY noted (no-op)", Failure.NONE, False, 0.0

        return f"ERROR: invalid action ({parsed.get('raw', op)})", Failure.NONE, False, 0.0

    def _render_observation(self) -> str:
        """Build the prompt the model sees on its next turn.

        Includes the system prompt, the task, and a truncated scratchpad of
        recent (action, observation) pairs.
        """
        recent = self.history[-5:]
        scratch_lines = []
        for i, t in enumerate(recent, start=max(1, self._steps - len(recent) + 1)):
            scratch_lines.append(f"  step {i}: action={t.action_text.strip()[:120]}")
            scratch_lines.append(f"           result={t.observation}")
        scratch = "\n".join(scratch_lines) if scratch_lines else "  (no actions yet)"

        return (
            f"{SYSTEM_PROMPT_HEADER}\n"
            f"{self.task.system_prompt_tail()}\n"
            f"=== Recent history (last {len(recent)} of {self._steps} steps) ===\n"
            f"{scratch}\n\n"
            f"Steps remaining: {self.max_steps - self._steps}\n"
            f"Emit your next action as a single line of JSON:"
        )

    # ------------------------------------------------------------------
    # debug helpers
    # ------------------------------------------------------------------

    def transcript(self) -> str:
        lines = [f"Episode seed={self.seed} return={self.episode_return:.3f}"]
        for i, t in enumerate(self.history, start=1):
            lines.append(
                f"  [{i}] op={t.parsed_op:<7} reward={t.reward:+.2f} "
                f"failure={t.failure.value:<22} obs={t.observation}"
            )
        return "\n".join(lines)
