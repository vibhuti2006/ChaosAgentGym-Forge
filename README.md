# ChaosAgentGym

> An OpenEnv-compatible environment that teaches LLM agents to **survive
> failing tools** — 503s, stale reads, and partial writes — instead of looping
> on retries or confidently lying about success.

Built by **Team GraderOne** — Vibhuti Bhatnagar & Jaivardhan D Rao — for the **Meta PyTorch / OpenEnv Hackathon 2026**.

## Links

| Resource | URL |
|---|---|
| 🌀 **Live OpenEnv Space (HF)** | https://huggingface.co/spaces/jaivardhandrao/chaos-env |
| 💻 **Source code (GitHub)** | https://github.com/vibhuti2006/ChaosAgentGym-Forge |
| 📓 **Colab training notebook** | [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) (one-click reproducer) |
| ✍️ **HF blog (writeup)** | https://huggingface.co/spaces/jaivardhandrao/chaos-env/blob/main/Blog.md |
| 📊 **Reward curve** | [`logs/reward_curve.png`](logs/reward_curve.png) |
| 📝 **Side-by-side transcripts** | [`logs/before_after.md`](logs/before_after.md) |

## Talk to the live env in 4 lines

```python
# pip install openenv-core fastmcp
from openenv.core.mcp_client import MCPToolClient

with MCPToolClient(base_url="https://jaivardhandrao-chaos-env.hf.space").sync() as env:
    env.reset(seed=7)
    print(env.call_tool("read_task"))
```

---

## Why this exists

Today's tool-using agents look great in demos and break in production. The
tools they call are *unreliable*: APIs flap, caches go stale, writes get
half-applied. There is almost no training data for these scenarios — every
fine-tuning corpus assumes the tool always returns the right answer.

So agents learn one strategy: trust the first response. When that response is
wrong, they either spin in retry loops or — worse — confidently report success
on a write that didn't persist.

**ChaosAgentGym** is a tiny RL environment that gives agents a controlled dose
of chaos and a reward signal for handling it intelligently.

---

## The environment

**Domain:** a single mock user-record API.
**Task:** *"Update user `u_42`'s email to `new@example.com` and confirm it stuck."*

**Action space** (one JSON action per step):

```jsonc
{"op": "GET",    "user": "u_42"}
{"op": "PUT",    "user": "u_42", "patch": {"email": "new@example.com"}}
{"op": "VERIFY", "user": "u_42", "expect": {"email": "new@example.com"}}  // terminal
{"op": "RETRY"}                                                           // no-op
```

**Three task variants** (mixed via `TaskDistribution` for curriculum training):

| Task | Framing | Why it matters |
|---|---|---|
| `update_email` | "Set u_42's email to *X*" | The canonical task |
| `rollback_partial` | "A malicious PUT corrupted u_42's email to *X*; the audit log says it should be *Y* — restore it." | Tests **generalisation** — same action space and recovery pattern, different framing and target value |
| `gdpr_anonymize` | "GDPR REQUEST: anonymise u_42's email to *X@redacted.example.com*" | Compliance framing, real-world use case (audit-driven write) |

Both variants use the same JSON action grammar. Within each variant, the
target value is sampled from a small vocabulary so the agent can't memorise
"always PUT new@example.com" — it has to learn the *recovery pattern* as a
strategy. Enable via `--task_curriculum` on training and eval scripts.

**Three failure modes**, seeded so episodes are reproducible:

| Failure | What it does | Why it's hard |
|---|---|---|
| `503` Service Unavailable | Drops the call, no state change | Agent must distinguish a *transient* failure from a real one |
| Stale read | GET returns the *pre-update* value even after a successful PUT | Agent can't trust a single GET as confirmation |
| Partial write | PUT returns 200, updates the visible store, but **not** the ground truth that `VERIFY` is judged against | Agent has to defend with a second PUT before verifying |

OpenEnv interface lives in [env/chaos_env.py](env/chaos_env.py):

```python
env = ChaosEnv(seed=0, difficulty=1.0)
obs = env.reset()
result = env.step(action_text)   # -> StepResult(observation, reward, done, info)
```

A difficulty knob (`0.0` → `1.0`) scales failure rates, enabling a curriculum
during training.

---

## Reward design

The reward is **dense**, not sparse — every step has signal. This is what makes
a 0.5B model trainable in 200 episodes.

| Event | Δreward | Purpose |
|---|---|---|
| Per-step action cost | `-0.05` | Discourages wasted calls |
| Action == previous action (GET/PUT/RETRY) | `-0.20` | Penalises blind retries |
| First time agent picks a *different* action after a failure | `+0.30` (one-shot) | Rewards adaptive recovery |
| `VERIFY` matches ground truth | `+1.00` (terminal) | Task success |
| `VERIFY` doesn't match ground truth | `-0.50` (terminal) | Punishes confidently-wrong success claims |
| Step budget exhausted (no VERIFY) | `0.00` | Soft failure, no extra penalty |

This shaping is what we found in 30 minutes of iteration:
* the false-VERIFY penalty is essential — without it, the agent learns to
  VERIFY immediately and collect partial credit;
* the recovery bonus kicks the agent off the "do nothing" attractor
  early in training.

---

## Training

Two-stage pipeline: **a tiny SFT warmup**, then **REINFORCE with a moving-average baseline**.

### Why SFT warmup first

A 0.5B base model often emits malformed JSON or refuses our action grammar on
the first turn, leaving REINFORCE with no positive trajectories to amplify.
~1 epoch over a few hundred scripted demonstrations fixes that — after warmup
the model emits parseable actions, and RL can do its job.

```bash
# generate ~1300 (obs, action) pairs from the oracle (CPU, ~5s)
python -m training.make_demo_dataset --n_episodes 300

# warmup (T4: ~5 min)
python -m training.sft_warmup \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --demos logs/demos.jsonl \
    --epochs 1 --lr 2e-5 \
    --out logs/ckpt_sft
```

The SFT loss is masked to the action tokens only — we don't waste capacity
making the model better at reproducing the system prompt.

### Two trainer paths: TRL PPO or hand-rolled REINFORCE

We ship both:

* **`training/train_trl.py`** — official TRL stack
  (`PPOTrainer` + `AutoModelForCausalLMWithValueHead`). Use this for the
  hackathon's "Reward + Training Setup" criterion.

  ```bash
  pip install 'trl>=0.8.6,<0.13'
  python -m training.train_trl --model logs/ckpt_sft --episodes 200 --batch 8
  ```

  Multi-step adapter: each env step becomes a `(query, response, reward)`
  triple; the whole batch (≈ 8 episodes × ~5 steps = 40 triples) is fed to
  `ppo_trainer.step()` per update.

* **`training/train.py`** — hand-rolled REINFORCE-with-baseline. Useful as a
  fallback when TRL versions break in Colab, and as a transparent reference
  for what TRL is doing under the hood. ~200 lines, every gradient inspectable.

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` — small enough for free Colab T4,
instruction-tuned so it can emit structured JSON on step 0.

**Loop:**

1. Roll out a batch of 8 episodes (each up to 8 steps).
2. Compute reward-to-go per step, normalise advantages across the batch.
3. Backprop `-advantage * log π(action_tokens | observation)`.
4. Update an EMA baseline on episode returns; ramp difficulty from 0.4 → 1.0.

```bash
# RL from the warmup checkpoint, NOT from the raw base
python -m training.train \
    --model logs/ckpt_sft \
    --episodes 200 --batch 8 --lr 1e-6 \
    --difficulty_ramp
```

---

## Results

> Run [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) end-to-end on
> a free Colab T4 to reproduce.

### Reward curve

`logs/reward_curve.png` (generated by `python -m eval.plot_rewards`):

* Episode return climbs from ≈ –0.2 (random) toward +0.7 (matches the scripted
  oracle).
* Success rate climbs from ~30% to ~80% as the agent learns the
  PUT → GET → PUT → GET → VERIFY pattern that defends against partial writes.

### Before vs after

`logs/before_after.md` (generated by `python -m eval.before_after`):

The base model typically does one of:
* hallucinates an unrelated action ("INVALID"), or
* PUTs once and immediately VERIFYs (gets fooled by partial writes), or
* loops on GETs after a 503.

The trained model:
* PUTs, GETs, **PUTs again**, then VERIFYs — defending against partials;
* changes action after a 503 instead of repeating verbatim.

### Held-out sanity baselines (100 seeds, disjoint from training)

From `python -m eval.quantitative`:

| Policy | Success | Mean return | Mean steps | Failures/ep |
|---|---|---|---|---|
| RandomPolicy | 30% | −0.20 | 3.4 | 0.61 |
| ScriptedPolicy (oracle) | **89%** | **+0.82** | 5.0 | 1.43 |

The 30% → 89% gap is exactly the behavior change RL has to discover.

### Behavioral fingerprint (the demo's headline)

From `python -m eval.behavior_diff` — *how* policies differ, not just whether
they succeed:

| Metric | Random | Oracle |
|---|---|---|
| Premature VERIFY % | 50% | **0%** |
| Defended VERIFY % (≥2 PUTs) | 16% | **100%** |
| Recovery after failure | 66% | **100%** |
| Top sequence | various | `PUT → GET → PUT → GET → VERIFY` (50/50) |

The trained model is judged against these structural metrics, not just the
scalar return — "defended VERIFY %" is the single number that captures whether
the model has internalised the partial-write defence.

---

## Repository layout

```
chaos_agent_gym/
├── env/
│   ├── chaos_env.py            OpenEnv-compatible env, reward function
│   ├── chaos_injector.py       Seeded failure injection (503 / stale / partial)
│   ├── mock_api.py             The user-record API (visible vs ground-truth stores)
│   ├── tasks.py                Task dataclass + TaskDistribution (curriculum)
│   ├── smoke_test.py           Four scripted policies — sanity baseline
│   └── test_tasks.py           Task-variety smoke test (curriculum sanity)
├── training/
│   ├── make_demo_dataset.py    Scripted-oracle rollouts -> SFT JSONL
│   ├── sft_warmup.py           Action-only-loss SFT pass (de-risks JSON output)
│   ├── train.py                REINFORCE-with-baseline RL loop (no TRL)
│   ├── train_trl.py            TRL PPO trainer (PPOTrainer + value head)
│   ├── policies.py             Random / Scripted / LLM policies
│   ├── rollout.py              Episode rollout glue
│   └── test_pipeline.py        Pipeline test (no GPU needed)
├── eval/
│   ├── plot_rewards.py         Reward + success-rate curve
│   ├── before_after.py         Side-by-side trained vs untrained transcripts
│   ├── quantitative.py         Held-out success-rate / action-mix table
│   └── behavior_diff.py        Structural behavior fingerprint (defended-VERIFY etc)
├── notebooks/
│   └── colab_train.ipynb       One-click reproducer
├── logs/                       rewards.csv, transcripts, checkpoints, plots
└── requirements.txt
```

---

## Why this matters

Agentic systems are about to be deployed against real APIs that are *already*
flaky — and there's no public training data for "tools that misbehave." A
reusable chaos environment fills that gap. The same recipe extends naturally
to:

* file-system tools that surface eventual consistency,
* search APIs that occasionally return adversarial / outdated results,
* multi-step workflows where one of N tool calls partially commits.

ChaosAgentGym is the smallest possible end-to-end demonstration of that
recipe: 4 actions, 3 failure modes, 1 task, a learnable reward, and a curve
that goes up.

---

## Quickstart

```bash
pip install -r requirements.txt

# --- env sanity (no GPU) -----------------------------------------------
python -m env.smoke_test
python -m training.test_pipeline
python -m eval.quantitative --n_seeds 100   # Random vs Scripted baselines

# --- training pipeline (T4 ~30 min total) ------------------------------
python -m training.make_demo_dataset --n_episodes 300 --task_curriculum
python -m training.sft_warmup       --out logs/ckpt_sft         # ~5 min
# TRL path (preferred for hackathon scoring):
python -m training.train_trl --model logs/ckpt_sft --episodes 200 --task_curriculum
# OR hand-rolled fallback (no TRL dependency):
# python -m training.train --model logs/ckpt_sft --episodes 200 --task_curriculum

# --- evaluation --------------------------------------------------------
python -m eval.plot_rewards
python -m eval.before_after  --base Qwen/Qwen2.5-0.5B-Instruct --trained logs/ckpt_final
python -m eval.quantitative  --base Qwen/Qwen2.5-0.5B-Instruct --trained logs/ckpt_final --task_curriculum
python -m eval.behavior_diff --base Qwen/Qwen2.5-0.5B-Instruct --trained logs/ckpt_final
```
