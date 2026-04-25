# ChaosAgentGym (OpenEnv)

OpenEnv-compatible environment for training LLM agents to survive **failing
tools** — 503s, stale reads, and partial writes — instead of looping on
retries or confidently lying about success.

## Quick start

### Install

```bash
pip install openenv-core
```

### Run locally

```bash
# from the repo root
uvicorn openenv_chaos.server.app:app --host 0.0.0.0 --port 8000
```

### Drive it from a client

```python
from openenv_chaos import ChaosEnv

# .sync() wrapper required for synchronous use; or use `async with` directly
with ChaosEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(seed=7)

    task = env.call_tool("read_task")
    print("task:", task["description"])
    # e.g. "Update user u_42's email to new@example.com..."

    # Run the chaos-defended recipe: PUT → GET → PUT → GET → VERIFY
    env.call_tool("put_user", patch={"email": "new@example.com"})
    env.call_tool("get_user")
    env.call_tool("put_user", patch={"email": "new@example.com"})  # defend partial
    env.call_tool("get_user")
    out = env.call_tool("verify_user", expect={"email": "new@example.com"})
    print("reward:", out["reward"], "done:", out["done"])
```

### Run from a HuggingFace Space

```python
env = ChaosEnv.from_env("<your-hf-username>/chaos-env")
```

## Tools

| Tool | Args | Returns |
|---|---|---|
| `read_task` | — | `{description, user_id, target, max_steps, steps_remaining}` |
| `get_user` | `user_id` | `{observation, reward, done, failure}` |
| `put_user` | `patch, user_id` | same |
| `verify_user` | `expect, user_id` | same (terminal) |
| `retry` | — | same (no-op) |

## Failure modes (seeded, reproducible)

| Mode | Effect |
|---|---|
| `service_unavailable` | Call drops with HTTP 503 |
| `stale_read` | GET returns the pre-update value |
| `partial_write` | PUT returns 200, updates visible store but not ground truth |

## Tasks

| Task | Framing |
|---|---|
| `update_email` | Set u_42's email to *X* |
| `rollback_partial` | Restore u_42's email after a malicious PUT |
| `gdpr_anonymize` | Anonymise u_42's email per GDPR request |

Sampled from a `TaskDistribution` per episode when `task_curriculum=True`
(default in `reset()`).

## Reward (dense)

| Event | Δ |
|---|---|
| Per step | -0.05 |
| Identical to prior action | -0.20 |
| Different action after a failure (one-shot) | +0.30 |
| `verify_user` matches ground truth | +1.00 (terminal) |
| `verify_user` doesn't match | -0.50 (terminal) |

## See also

Full project (training, eval, results): see the parent repo's `README.md`.
