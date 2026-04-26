# ChaosAgentGym: Teaching LLM Agents to Survive Failing Tools

> *Built for the Meta PyTorch / OpenEnv Hackathon 2026.
> Author: Vibhuti Bhatnagar.*

## TL;DR

Today's LLM agents look great in demos and break the moment a tool misbehaves — APIs return 503s, caches go stale, writes get half-applied. There is **almost no training data** for agents that have to recover from this kind of failure.

I built **ChaosAgentGym**: a small, OpenEnv-compatible RL environment that simulates these failures on a mock API and a reward function that punishes blind retries and false success claims. After two-stage training (TRL SFT distillation + REINFORCE refinement) on **Qwen2.5-0.5B-Instruct**, the same model went from **0% success** (it never even issued a `VERIFY`) to **89% success** on held-out chaotic episodes — matching a hand-coded expert oracle.

The trained agent has fundamentally different *behavior*, not just a higher score: 0 → 100% defended-VERIFY rate, 70% → 100% recovery-after-failure rate, no premature VERIFYs.

🌐 **Live env:** https://huggingface.co/spaces/jaivardhandrao/chaos-env
💻 **Code:** https://github.com/vibhuti2006/ChaosAgentGym-Forge

---

## The problem

Talk to anyone who's deployed an LLM agent in production and you'll hear the same story.

The agent calls a tool. The tool returns a 503. The agent calls again. Another 503. The agent calls again. And again. Within two minutes the agent has burned its entire budget on the same retry, and the user is staring at a spinner.

Or: the agent calls a tool. The tool returns 200 with a value the agent expected to write. The agent reports success. Except the write only landed in a read replica — the source-of-truth database never received it. The user thinks the change went through. It didn't.

These failure modes are **everywhere** in real distributed systems:
- **HTTP 503** — service unavailable, transient
- **Stale reads** — cache returned the pre-update value
- **Partial writes** — the write looked successful but the ground truth wasn't updated

Every fine-tuning corpus for tool-using LLM agents assumes the tool always returned the right answer. So when these failures happen in production, the agent has no learned response. It either spins forever or it confidently lies.

## ChaosAgentGym in one paragraph

ChaosAgentGym is a tiny RL environment built on Meta's [OpenEnv framework](https://github.com/meta-pytorch/OpenEnv). It exposes a mock user-record API as four MCP tools (`get_user`, `put_user`, `verify_user`, `retry`) that the agent can call. Inside each tool, a seeded chaos injector rolls dice and decides whether to drop the call (503), return a stale value, or accept the write but corrupt the ground truth (partial write). The agent's job is to **complete a task** despite the chaos — and **VERIFY** that the change actually persisted, with a heavy penalty for falsely claiming success.

There are three task variants sampled per episode:
- **`update_email`** — set a user's email to a target value
- **`rollback_partial`** — reverse a malicious PUT that corrupted the record
- **`gdpr_anonymize`** — anonymize the email per a compliance request

All three use the same action grammar, so the agent has to learn the *recovery pattern* (a strategy that generalizes) rather than memorizing one specific action sequence.

## The reward function (the part that actually mattered)

I spent more time iterating on the reward than on any other piece of the project. The keystone is the **false-VERIFY penalty**.

```
+1.00   VERIFY against ground truth succeeds (terminal)
-0.50   VERIFY against ground truth fails (false claim, terminal)
+0.30   Agent picks a different action after a failure (one-shot recovery bonus)
-0.20   Same action as previous step (blind retry)
-0.05   Per step (action cost)
 0.00   Step budget exhausted, no VERIFY (soft fail)
```

Without the false-VERIFY penalty, the agent learned to **VERIFY immediately and collect partial credit** in the cases where chaos didn't fire. Adding the −0.5 made VERIFY genuinely costly when not earned, which forced the policy to actually inspect state before claiming success.

The recovery bonus (+0.30 for the first different action after a failure) was needed to break out of an early-training "do nothing" attractor where the agent just emitted RETRY actions to dodge the per-step cost.

## Training pipeline

Two stages, both reproducible on a free Colab T4 in about half an hour.

### Stage 1 — SFT distillation with HuggingFace TRL

I generate ~1300 (observation, action) pairs by running a hand-coded scripted oracle on the env and keeping only successful episodes. The oracle's strategy is the canonical chaos-defended recipe:

```
PUT  →  GET  →  PUT  →  GET  →  VERIFY
```

The second PUT defends against partial writes (since you can't tell from a single GET whether the first PUT made it to ground truth). The two GETs guard against a single stale read.

Then I fine-tune Qwen2.5-0.5B-Instruct with `trl.SFTTrainer`:

```bash
python -m training.train_trl_sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --demos logs/demos.jsonl \
    --epochs 1 --lr 2e-5 --batch 2
```

After 17 minutes on a T4: loss `1.235 → 0.011`, mean token accuracy `77% → 99.4%`. The model has internalized the action grammar and the oracle's strategy.

### Stage 2 — REINFORCE refinement

The SFT model is then placed in the env to play episodes for real. A REINFORCE-with-baseline loop computes per-step advantages (reward-to-go minus an EMA baseline), normalizes them across the batch, and applies the policy gradient `−advantage × log π(action | state)`.

A difficulty curriculum ramps the chaos rate from `0.4 → 1.0` over 50 updates so the SFT-warmed policy doesn't get clobbered by full chaos on update zero.

I also wrote `training/train_trl.py` using `trl.PPOTrainer` for completeness, but TRL 0.12+ broke the PPO API we depended on (`PPOConfig` no longer accepts `model_name`). The hand-rolled REINFORCE loop is what produced the curve below.

## Results

### The headline

100 held-out chaotic episodes (disjoint seeds, all three tasks, full chaos):

| Policy | Success rate | Mean return |
|---|---|---|
| Random baseline | 30% | −0.20 |
| **Base Qwen-0.5B (zero-shot)** | **0%** | **−0.69** |
| **Trained model** | **89%** | **+0.82** |
| Hand-coded scripted oracle | 89% | +0.82 |

Read that twice. The base Qwen-0.5B doesn't just fail — **it fails harder than literal random**. Looking at its action distribution, the base model emits `RETRY` 48% of the time and `VERIFY` 0% of the time. It physically never reaches a VERIFY action in 100 episodes. It just spins on retries until the step budget runs out.

After training, the same architecture matches a hand-coded expert.

### The behavioral fingerprint

Score is one thing. *How* the agent behaves is the more revealing metric.

| Metric | Random | Base Qwen | **Trained** |
|---|---|---|---|
| Premature VERIFY % | 50% | 0% (never gets there) | **0%** |
| Defended VERIFY (≥2 PUTs) | 16% | 0% | **100%** |
| Recovery rate after a failure | 66% | 70% | **100%** |

The trained model didn't just get lucky 89% of the time. It learned the **structural recovery pattern** — always defend a VERIFY with at least two PUTs, always change action after a failure. Its top action sequence (50 of 50 episodes): `PUT → GET → PUT → GET → VERIFY`. Identical to the oracle.

### Side-by-side: same seed, before and after

**Seed 89** sampled `rollback_partial`, with two HTTP 503s in the episode.

```
BEFORE (untrained Qwen-0.5B):
  step 1: PUT (503)
  step 2: RETRY
  step 3: GET
  step 4: GET
  step 5: GET (503)
  step 6: RETRY
  step 7: PUT
  step 8: PUT (503)        ← budget exhausted, never VERIFYs
  return: -0.700, FAIL
```

```
AFTER (SFT + REINFORCE):
  step 1: PUT (503)
  step 2: GET
  step 3: PUT              ← second PUT defends against partial write
  step 4: GET (503)
  step 5: VERIFY → ground truth matches ✓
  return: +1.050, SUCCESS
```

Same chaos. Same model architecture. Different policy.

## The OpenEnv layer

The whole environment is hosted as a public HuggingFace Space at https://huggingface.co/spaces/jaivardhandrao/chaos-env. Anyone can connect with the OpenEnv MCP client and drive it:

```python
from openenv.core.mcp_client import MCPToolClient

with MCPToolClient(base_url="https://jaivardhandrao-chaos-env.hf.space").sync() as env:
    env.reset(seed=7)
    print(env.call_tool("read_task"))
    env.call_tool("put_user", patch={"email": "new@example.com"})
    env.call_tool("get_user")
    env.call_tool("put_user", patch={"email": "new@example.com"})
    env.call_tool("get_user")
    out = env.call_tool("verify_user", expect={"email": "new@example.com"})
    print("reward:", out["reward"])
```

Built with `openenv-core 0.2.3` + `fastmcp 3.2.4` + FastAPI, packaged into a Docker container that HF Spaces builds and serves.

## Resources

| Resource | URL |
|---|---|
| Live OpenEnv Space | https://huggingface.co/spaces/jaivardhandrao/chaos-env |
| Source code | https://github.com/vibhuti2006/ChaosAgentGym-Forge |
| Reward curve | `logs/reward_curve.png` |
| Quantitative eval | `logs/quantitative.md` |
| Behavioral diff | `logs/behavior_diff.md` |
| Side-by-side transcripts | `logs/before_after.md` |
| TRL training script | `training/train_trl_sft.py` |
| Colab one-click reproducer | `notebooks/colab_train.ipynb` |

Built for the **Meta PyTorch / OpenEnv Hackathon 2026**.
