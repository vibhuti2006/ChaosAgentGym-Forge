<!--
ChaosAgentGym — slide deck (Marp format)

Render to PDF:
  npm install -g @marp-team/marp-cli
  marp notebooks/slides.md -o slides.pdf

Or paste this whole file into the Marp web app:
  https://web.marp.app

Or copy each slide (separated by `---`) into Google Slides / Keynote / PowerPoint
manually. Each "---" denotes a new slide.
-->

---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- _class: lead -->

# 🌀 ChaosAgentGym

### Teaching LLM agents to **survive failing tools**

503s. Stale reads. Partial writes.
The agents in production hit these every day — and break.

**Meta PyTorch / OpenEnv Hackathon 2026**

🌐 Live: huggingface.co/spaces/jaivardhandrao/chaos-env

---

## The problem

Today's tool-using agents look great in demos. They **break in production** the moment something misbehaves.

* APIs return 503s, agents loop forever on retries
* Caches return stale data, agents trust the first GET
* Writes get half-applied, agents confidently lie about success

> There is **almost no training data** for chaos-handling.
> Every fine-tuning corpus assumes the tool always returned the right answer.

**ChaosAgentGym** is a tiny RL environment that changes that.

---

## The environment

**Domain:** mock user-record API — `GET / PUT / VERIFY / RETRY`

**Three failure modes** (seeded, reproducible):

| Mode | Effect |
|---|---|
| **`503`** | Call drops. No state change. Looks like a transient. |
| **Stale read** | GET returns the *pre-update* value. Looks correct. |
| **Partial write** | PUT returns 200, updates the visible store but **not the ground truth** that VERIFY checks. |

**Three task variants** sampled per episode (curriculum):

* `update_email` — productivity
* `rollback_partial` — security
* `gdpr_anonymize` — compliance

---

## Reward function

**Dense, designed to penalise blind retries and reward intelligent recovery.**

| Event | Δ |
|---|---|
| Per step (action cost) | `-0.05` |
| Same action as previous (blind retry) | `-0.20` |
| First *different* action after a failure | `+0.30` (one-shot) |
| `VERIFY` matches ground truth | **`+1.00`** terminal |
| `VERIFY` doesn't match (false claim) | `-0.50` terminal |

The **false-claim penalty** is the keystone. Without it, the agent learns to VERIFY immediately and collect partial credit.

---

## Training pipeline

**Two-stage:** distillation → refinement.

1. **SFT (HuggingFace TRL `SFTTrainer`)** on ~1300 trajectories from a hand-coded scripted oracle. Teaches the model the recovery pattern.
2. **Reinforcement learning (REINFORCE-with-baseline)** on the live env with curriculum (chaos difficulty 0.4 → 1.0). Refines at the upper edge.

**Model:** `Qwen2.5-0.5B-Instruct` — fits a free Colab T4 with gradient checkpointing.

**Total wall-clock: ~30 minutes** on a single T4.

One-click reproducer: `notebooks/colab_train.ipynb`

---

## Results — 0% → 89%

100 held-out chaotic episodes (disjoint seeds, 3-task curriculum):

| Policy | Success | Mean return | Notes |
|---|---|---|---|
| Random | 30% | -0.20 | Coin-flip baseline |
| **Base Qwen-0.5B** | **0%** | -0.69 | Worse than random — never VERIFYs |
| **Trained model** | **89%** | **+0.82** | **Matches the scripted oracle** |
| Scripted oracle | 89% | +0.82 | Hand-coded recipe |

The base Qwen 0.5B fails harder than random: it spins on RETRYs and never issues a VERIFY. After SFT + RL, the same 0.5B model matches a hand-coded expert.

---

## Behavioral fingerprint — *how* did behavior change?

| Metric | Random | Base Qwen | **Trained** |
|---|---|---|---|
| Premature VERIFY % | 50% | 0%* | **0%** |
| Defended VERIFY (≥2 PUTs) | 16% | 0% | **100%** |
| Recovery rate after failure | 66% | 70% | **100%** |

\*Base Qwen has 0% premature VERIFY because it has 0% VERIFY *at all* — it never gets there.

**Top action sequence per policy:**

* Random: `VERIFY` (10x), `PUT → VERIFY` (6x)
* Base Qwen: `PUT → GET → RETRY → GET → GET → RETRY → PUT → PUT` (15x — never VERIFYs)
* **Trained:** `PUT → GET → PUT → GET → VERIFY` (50x — the canonical recipe)

---

## Demo: same seed, before vs after

**Seed 89** (rollback_partial task, 2× HTTP 503 on PUTs)

**Base Qwen (untrained):**
```
PUT (503) → RETRY → GET → GET → GET (503) → RETRY → PUT → PUT (503) → BUDGET HIT
return: -0.70   success: ❌
```

**Trained model:**
```
PUT (503) → GET → PUT → GET (503) → VERIFY ✓
return: +1.05   success: ✅
```

The trained model survives both 503s. Its second PUT defends against a possible partial write. VERIFY succeeds against ground truth.

---

## OpenEnv compliance — live demo

The env is **publicly hosted** on HuggingFace Spaces.

🌀 **https://huggingface.co/spaces/jaivardhandrao/chaos-env**

Any OpenEnv client can talk to it in 4 lines:

```python
from openenv.core.mcp_client import MCPToolClient

with MCPToolClient(base_url="https://jaivardhandrao-chaos-env.hf.space").sync() as env:
    env.reset(seed=7)
    print(env.call_tool("read_task"))
```

Built with: `openenv-core 0.2.3` + `fastmcp 3.2.4` + FastAPI + Docker.

---

## Why this matters

Agents are about to be deployed against APIs that are *already* flaky.
There's no public training data for "tools that misbehave."

**ChaosAgentGym is the smallest possible end-to-end demonstration** of how to fix that:
3 failure modes × 3 tasks × dense recovery reward × a curve that goes up.

The same recipe extends naturally to:
* file-system tools with eventual consistency
* search APIs with adversarial / outdated results
* multi-step workflows where one of N tools partially commits

---

<!-- _class: lead -->

## Links

🌀 **Live env**: https://huggingface.co/spaces/jaivardhandrao/chaos-env
📓 **Code + Colab notebook**: _add GitHub link_
📊 **Reward curve, transcripts, eval tables**: in the project README

Built by **Team GraderOne** — Vibhuti Bhatnagar & Jaivardhan D Rao
for the **Meta PyTorch / OpenEnv Hackathon 2026**

*Thank you.*
