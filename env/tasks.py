"""Task parameterization for ChaosEnv.

Originally the env had a single hard-coded task ("set email to new@example.com").
That risks the agent memorising the specific value rather than learning the
recovery pattern. This module introduces a `Task` dataclass and a
`TaskDistribution` so each episode can use a different target.

Two task shapes ship out of the box:
  - UPDATE_EMAIL:       set a clean record's email to a new value
  - ROLLBACK_PARTIAL:   a malicious PUT corrupted the visible store; restore
                        the field to the documented audit value

Both use the same action space, so the agent's policy generalises across them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Task:
    name: str                  # short id: "update_email", "rollback_partial"
    description: str           # one-line natural-language framing
    user_id: str
    initial_truth: dict        # ground-truth state at episode start
    initial_visible: dict      # what GET would return (may differ from truth)
    target: dict               # what VERIFY's expect should match against truth

    def system_prompt_tail(self) -> str:
        """Task-specific text appended to the generic system prompt."""
        return (
            f"=== TASK ===\n"
            f"{self.description}\n"
            f"User: {self.user_id}\n"
            f"Goal state (must hold in ground truth before VERIFY): {self.target}\n"
        )


# ---------------------------------------------------------------------------
# canonical tasks
# ---------------------------------------------------------------------------

_DEFAULT_PROFILE = {"name": "Ada Lovelace", "version": 1}


def update_email_task(target_email: str = "new@example.com",
                      initial_email: str = "old@example.com",
                      user_id: str = "u_42") -> Task:
    base = {"email": initial_email, **_DEFAULT_PROFILE}
    return Task(
        name="update_email",
        description=(
            f"Update user {user_id}'s email to {target_email} and confirm "
            f"the change persisted (resist 503s, stale reads, partial writes)."
        ),
        user_id=user_id,
        initial_truth=dict(base),
        initial_visible=dict(base),
        target={"email": target_email},
    )


def gdpr_anonymize_task(redacted_email: str = "deleted-u_42@redacted.example.com",
                        current_email: str = "user@example.com",
                        user_id: str = "u_42") -> Task:
    """GDPR-style compliance task: anonymise a real email to a redacted form
    and confirm the change actually persisted in ground truth (which is what
    the legal team will audit)."""
    base = {"email": current_email, **_DEFAULT_PROFILE}
    return Task(
        name="gdpr_anonymize",
        description=(
            f"GDPR REQUEST: user {user_id} has requested account "
            f"anonymisation. Their email currently shows {current_email} — "
            f"update it to {redacted_email} to satisfy the legal request. "
            f"Confirm the change actually persisted (compliance auditors will "
            f"check ground truth, not the cached value)."
        ),
        user_id=user_id,
        initial_truth=dict(base),
        initial_visible=dict(base),
        target={"email": redacted_email},
    )


def rollback_partial_task(restore_email: str = "admin@example.com",
                          corrupted_email: str = "spam@bad.example.com",
                          user_id: str = "u_42") -> Task:
    """A bad PUT corrupted the record (both visible AND truth carry the
    attacker's value). The audit log says the correct value is
    `restore_email`. Agent must overwrite the corruption and confirm.

    Mechanically this is "update from corrupted_email to restore_email" — same
    shape as update_email but different framing and target. Whether the agent
    learns one general recovery pattern or two task-specific ones is
    exactly the generalisation question the curriculum is testing.
    """
    corrupted_record = {"email": corrupted_email, **_DEFAULT_PROFILE}
    return Task(
        name="rollback_partial",
        description=(
            f"SECURITY INCIDENT: a malicious PUT changed user {user_id}'s email "
            f"to {corrupted_email}. The audit log says the correct value is "
            f"{restore_email}. Restore it and confirm the fix actually persisted."
        ),
        user_id=user_id,
        initial_truth=dict(corrupted_record),
        initial_visible=dict(corrupted_record),
        target={"email": restore_email},
    )


# A small "vocabulary" of email targets so each episode is structurally
# similar but never identical — prevents the agent from memorising one string.
_NEW_EMAILS = [
    "alice@new.example.com",
    "ops@updated.example.com",
    "ada+v2@example.com",
    "support@chaos.example.com",
    "ceo@scaled.example.com",
]
_RESTORE_EMAILS = [
    "admin@example.com",
    "root@example.com",
    "owner@example.com",
]
_CORRUPT_EMAILS = [
    "spam@bad.example.com",
    "phish@evil.example.com",
    "drop@table.example.com",
]
_REDACTED_EMAILS = [
    "deleted-u_42@redacted.example.com",
    "anon-2026q2@redacted.example.com",
    "gdpr-removed@redacted.example.com",
]
_REAL_EMAILS = [
    "user@example.com",
    "person@example.com",
    "customer@example.com",
]


@dataclass
class TaskDistribution:
    """Samples a Task per reset. Composable: tweak weights for curriculum."""
    p_update: float = 1 / 3
    p_rollback: float = 1 / 3
    p_gdpr: float = 1 / 3
    seed: int = 0
    targets_update: Sequence[str] = field(default_factory=lambda: list(_NEW_EMAILS))
    targets_restore: Sequence[str] = field(default_factory=lambda: list(_RESTORE_EMAILS))
    targets_corrupt: Sequence[str] = field(default_factory=lambda: list(_CORRUPT_EMAILS))
    targets_redacted: Sequence[str] = field(default_factory=lambda: list(_REDACTED_EMAILS))
    targets_real: Sequence[str] = field(default_factory=lambda: list(_REAL_EMAILS))

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        total = self.p_update + self.p_rollback + self.p_gdpr
        if total <= 0:
            raise ValueError("TaskDistribution probabilities must be > 0")
        self.p_update /= total
        self.p_rollback /= total
        self.p_gdpr /= total

    def sample(self, episode_seed: int) -> Task:
        # Combine the two ints into a single deterministic seed Python's
        # random module accepts. Large prime keeps adjacent episode_seeds
        # decorrelated.
        rng = random.Random(self.seed * 1_000_003 + episode_seed)
        r = rng.random()
        if r < self.p_update:
            return update_email_task(target_email=rng.choice(self.targets_update))
        if r < self.p_update + self.p_rollback:
            return rollback_partial_task(
                restore_email=rng.choice(self.targets_restore),
                corrupted_email=rng.choice(self.targets_corrupt),
            )
        return gdpr_anonymize_task(
            redacted_email=rng.choice(self.targets_redacted),
            current_email=rng.choice(self.targets_real),
        )
