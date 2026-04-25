"""Policies for ChaosAgentGym.

  RandomPolicy   — uniform over the action template set; for pipeline tests.
  ScriptedPolicy — a hand-written sensible strategy; an "expert" baseline.
  LLMPolicy      — wraps a HuggingFace causal LM; produces (text, logprob, ids).

Only LLMPolicy needs torch/transformers — the other two are import-free so the
pipeline test can run on a machine without ML deps.

When task variety is enabled (TaskDistribution), Random/Scripted parse the
goal-state line out of the observation so the same policy works across
update_email and rollback_partial without changes.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any

from env import TARGET_EMAIL, USER_ID

from .rollout import ActionResult


# ---------------------------------------------------------------------------
# observation parsing helpers
# ---------------------------------------------------------------------------

_GOAL_RE = re.compile(r"Goal state .*?: (\{.*?\})", re.DOTALL)
_USER_RE = re.compile(r"^User: (\S+)$", re.MULTILINE)


def _parse_task_from_observation(obs: str) -> tuple[str, dict[str, Any]]:
    """Extract (user_id, target_dict) from the env's rendered observation.

    Falls back to the legacy defaults if anything is missing — this keeps
    test fixtures and unit-test prompts working.
    """
    user_match = _USER_RE.search(obs)
    user = user_match.group(1) if user_match else USER_ID

    goal_match = _GOAL_RE.search(obs)
    target: dict[str, Any] = {"email": TARGET_EMAIL}
    if goal_match:
        raw = goal_match.group(1)
        # Goal state is rendered with single quotes (from str(dict)); convert.
        try:
            target = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            pass
    return user, target


def _build_action_templates(user: str, target: dict[str, Any]) -> list[str]:
    return [
        f'{{"op": "GET", "user": "{user}"}}',
        f'{{"op": "PUT", "user": "{user}", "patch": {json.dumps(target)}}}',
        f'{{"op": "VERIFY", "user": "{user}", "expect": {json.dumps(target)}}}',
        '{"op": "RETRY"}',
    ]


# Legacy module-level templates (kept so old call sites don't break).
ACTION_TEMPLATES: list[str] = _build_action_templates(USER_ID, {"email": TARGET_EMAIL})


# ---------------------------------------------------------------------------
# scripted / random
# ---------------------------------------------------------------------------

@dataclass
class RandomPolicy:
    seed: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)

    def act(self, observation: str) -> ActionResult:
        user, target = _parse_task_from_observation(observation)
        templates = _build_action_templates(user, target)
        return ActionResult(text=self.rng.choice(templates))


@dataclass
class ScriptedPolicy:
    """Sensible PUT -> GET -> PUT -> GET -> VERIFY plan.

    Defends against partial writes by always doing a second PUT before VERIFY.
    Doesn't try to react to 503s — the env penalises blind retries anyway, and
    the second PUT in the plan covers most failure modes.

    Task-aware: re-reads the goal state from each observation so it works for
    update_email and rollback_partial (and any future Task) unchanged.
    """

    def __post_init__(self):
        self._step = 0

    def act(self, observation: str) -> ActionResult:
        user, target = _parse_task_from_observation(observation)
        templates = _build_action_templates(user, target)
        plan = [templates[1], templates[0], templates[1], templates[0], templates[2]]
        action = plan[min(self._step, len(plan) - 1)]
        self._step += 1
        return ActionResult(text=action)


# ---------------------------------------------------------------------------
# LLM policy (lazy import so non-ML callers don't need torch)
# ---------------------------------------------------------------------------

class LLMPolicy:
    """Causal-LM policy. Generates one action at a time, returning per-token
    log-probs so the trainer can compute a policy gradient.
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 64,
                 temperature: float = 0.8, device: str | None = None):
        import torch  # local import keeps env light
        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or next(model.parameters()).device

    def _format_prompt(self, observation: str) -> str:
        # Use chat template if available; otherwise fall back to raw text.
        if hasattr(self.tokenizer, "apply_chat_template") and \
                getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": observation}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return observation + "\nAction: "

    def act(self, observation: str) -> ActionResult:
        torch = self.torch
        prompt = self._format_prompt(observation)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = enc.input_ids.shape[1]
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        gen_ids = out.sequences[0, prompt_len:]
        # Per-token log-prob under the sampling distribution.
        # HF generate returns out.scores as a tuple of [batch=1, vocab]
        # tensors. Stacking gives [T, 1, vocab] — squeeze the batch dim
        # so the gather below indexes the vocab axis, not the (size-1)
        # batch axis (which would silently index out of bounds on GPU).
        scores = torch.stack(out.scores, dim=0)
        if scores.dim() == 3:
            scores = scores.squeeze(1)                        # [T, vocab]
        log_probs = torch.log_softmax(scores / self.temperature, dim=-1)
        token_logprobs = log_probs[range(len(gen_ids)), gen_ids]

        # Strip an EOS at the very end so we don't pad with a no-op token.
        eos = self.tokenizer.eos_token_id
        if eos is not None and len(gen_ids) > 0 and gen_ids[-1].item() == eos:
            gen_ids = gen_ids[:-1]
            token_logprobs = token_logprobs[:-1]

        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return ActionResult(
            text=text or '{"op": "RETRY"}',
            aux={
                "prompt_ids": enc.input_ids[0].detach().cpu(),
                "gen_ids": gen_ids.detach().cpu(),
                "token_logprobs": token_logprobs.detach().cpu(),
            },
        )
