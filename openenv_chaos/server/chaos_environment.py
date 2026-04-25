"""ChaosAgentGym environment in OpenEnv-compatible form.

Wraps the in-process ChaosEnv (env/chaos_env.py) as an MCP environment so it
can be deployed to a HuggingFace Space and consumed by any OpenEnv-compatible
client.

The four chaos actions (GET / PUT / VERIFY / RETRY) are exposed as MCP tools.
Chaos failures (503, stale read, partial write) are still injected inside the
underlying env — the wrapper just translates between MCP tool calls and the
in-process ChaosEnv.step() interface.
"""
from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

# Reuse the existing chaos env logic untouched. The package layout assumes
# this server module is imported from a context that has the parent dir on
# sys.path (Dockerfile sets PYTHONPATH=/app/env which contains both env/ and
# openenv_chaos/).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from env import ChaosEnv, TaskDistribution, USER_ID  # noqa: E402


def _make_env(seed: int, difficulty: float, task_curriculum: bool) -> ChaosEnv:
    if task_curriculum:
        return ChaosEnv(
            seed=seed,
            difficulty=difficulty,
            task_distribution=TaskDistribution(seed=seed),
        )
    return ChaosEnv(seed=seed, difficulty=difficulty)


class ChaosEnvironment(MCPEnvironment):
    """OpenEnv server for ChaosAgentGym.

    Tools exposed (all return a dict the agent inspects):
      - get_user(user_id)               -> {"observation", "reward", "done", "failure"}
      - put_user(user_id, patch)        -> same shape
      - verify_user(user_id, expect)    -> same shape (terminal)
      - retry()                         -> same shape (no-op)
      - read_task()                     -> {"description", "user_id", "target", "max_steps"}
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        mcp = FastMCP("chaos_env")
        # Capture self for the tool closures (decorator timing means we can't
        # rely on `self` directly inside the inline functions).
        env_self = self

        # Initial placeholder env so list_tools etc. don't crash before reset.
        env_self._env: Optional[ChaosEnv] = None
        env_self._task_curriculum: bool = True

        def _ensure_env():
            if env_self._env is None:
                env_self._env = _make_env(
                    seed=0, difficulty=1.0, task_curriculum=env_self._task_curriculum,
                )

        def _step_to_dict(result) -> dict:
            return {
                "observation": result.observation,
                "reward": result.reward,
                "done": result.done,
                "failure": result.info.get("failure", "none"),
                "episode_return": result.info.get("episode_return"),
                "step": result.info.get("step"),
            }

        @mcp.tool
        def read_task() -> dict:
            """Return the current task definition (description, user_id, target).

            Call this after reset() to learn what to do this episode.
            """
            _ensure_env()
            t = env_self._env.task
            return {
                "description": t.description,
                "user_id": t.user_id,
                "target": t.target,
                "max_steps": env_self._env.max_steps,
                "steps_remaining": env_self._env.max_steps - env_self._env._steps,
            }

        @mcp.tool
        def get_user(user_id: str = USER_ID) -> dict:
            """GET the user record. May return 503, stale data, or normal.

            Args:
                user_id: ID of the user to fetch.
            """
            _ensure_env()
            action = json.dumps({"op": "GET", "user": user_id})
            return _step_to_dict(env_self._env.step(action))

        @mcp.tool
        def put_user(patch: dict, user_id: str = USER_ID) -> dict:
            """PUT a patch onto the user record. May 503 or partially apply.

            Args:
                patch: Field-value pairs to update (e.g. {"email": "x@y.com"}).
                user_id: ID of the user to update.
            """
            _ensure_env()
            action = json.dumps({"op": "PUT", "user": user_id, "patch": patch})
            return _step_to_dict(env_self._env.step(action))

        @mcp.tool
        def verify_user(expect: dict, user_id: str = USER_ID) -> dict:
            """Terminal action: claim the user record matches `expect`.

            +1.0 reward if ground truth matches, -0.5 if not.

            Args:
                expect: Field-value pairs that should match ground truth.
                user_id: ID of the user to verify.
            """
            _ensure_env()
            action = json.dumps({"op": "VERIFY", "user": user_id, "expect": expect})
            return _step_to_dict(env_self._env.step(action))

        @mcp.tool
        def retry() -> dict:
            """No-op step. Costs an action; penalised if used immediately
            after another retry/identical action."""
            _ensure_env()
            action = json.dumps({"op": "RETRY"})
            return _step_to_dict(env_self._env.step(action))

        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # OpenEnv lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: Optional[float] = None,
        task_curriculum: Optional[bool] = None,
        **kwargs: Any,
    ) -> Observation:
        """Start a new episode.

        Args:
            seed:            episode seed (drives chaos + task sampling)
            episode_id:      OpenEnv episode id (auto-generated if omitted)
            difficulty:      0.0–1.0 chaos rate scale (default 1.0)
            task_curriculum: if True, sample a task per episode from the
                             update_email + rollback_partial + gdpr_anonymize
                             distribution. If False, use the canonical
                             update_email task.
        """
        if task_curriculum is not None:
            self._task_curriculum = task_curriculum
        self._env = _make_env(
            seed=seed if seed is not None else 0,
            difficulty=difficulty if difficulty is not None else 1.0,
            task_curriculum=self._task_curriculum,
        )
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "task": self._env.task.name,
                "task_description": self._env.task.description,
                "target": self._env.task.target,
                "user_id": self._env.task.user_id,
                "max_steps": self._env.max_steps,
                "instructions": (
                    "Call read_task() to see the goal, then use get_user / "
                    "put_user / verify_user / retry tools. The API is unreliable: "
                    "calls may 503, return stale data, or partially apply writes. "
                    "Defend against partial writes by repeating PUTs before VERIFY."
                ),
            },
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": (
                    f"Unknown action type: {type(action).__name__}. "
                    "Use ListToolsAction or CallToolAction."
                )
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        return self._state
