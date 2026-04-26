// JS port of env/chaos_env.py + chaos_injector.py + mock_api.py + tasks.py.
// Faithful to the reward function, action grammar, and failure semantics —
// numerical RNG output won't bit-match Python's `random` module, but the
// behaviour is identical (same probabilities, same task shapes).

export const Failure = Object.freeze({
  NONE: "none",
  SERVICE_UNAVAILABLE: "service_unavailable",
  STALE_READ: "stale_read",
  PARTIAL_WRITE: "partial_write",
});

// ---- seeded RNG (mulberry32) ------------------------------------------------

export function makeRng(seed) {
  let s = (seed >>> 0) || 1;
  return function () {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function rngChoice(rng, arr) {
  return arr[Math.floor(rng() * arr.length)];
}

// ---- ChaosInjector ----------------------------------------------------------

const DEFAULT_CFG = { p_503: 0.25, p_stale: 0.15, p_partial: 0.15 };

function scaledCfg(cfg, difficulty) {
  const s = 0.4 + 0.6 * Math.max(0, Math.min(1, difficulty));
  return {
    p_503: cfg.p_503 * s,
    p_stale: cfg.p_stale * s,
    p_partial: cfg.p_partial * s,
  };
}

export class ChaosInjector {
  constructor(seed, difficulty = 1.0, cfg = DEFAULT_CFG) {
    this.rng = makeRng(seed);
    this.cfg = scaledCfg(cfg, difficulty);
  }
  rollGet() {
    const r = this.rng();
    if (r < this.cfg.p_503) return Failure.SERVICE_UNAVAILABLE;
    if (r < this.cfg.p_503 + this.cfg.p_stale) return Failure.STALE_READ;
    return Failure.NONE;
  }
  rollPut() {
    const r = this.rng();
    if (r < this.cfg.p_503) return Failure.SERVICE_UNAVAILABLE;
    if (r < this.cfg.p_503 + this.cfg.p_partial) return Failure.PARTIAL_WRITE;
    return Failure.NONE;
  }
}

// ---- Mock API ---------------------------------------------------------------

export class MockUserApi {
  constructor(injector) {
    this.injector = injector;
    this.truth = {};
    this.visible = {};
    this.prevVisible = {};
    this.callCount = 0;
  }
  static withUser(injector, userId, record) {
    const api = new MockUserApi(injector);
    api.truth[userId] = { ...record };
    api.visible[userId] = { ...record };
    api.prevVisible[userId] = { ...record };
    return api;
  }
  get(userId) {
    this.callCount++;
    const fail = this.injector.rollGet();
    if (fail === Failure.SERVICE_UNAVAILABLE)
      return { status: 503, body: null, failure: fail };
    if (!(userId in this.visible))
      return { status: 404, body: null, failure: Failure.NONE };
    if (fail === Failure.STALE_READ && userId in this.prevVisible)
      return { status: 200, body: { ...this.prevVisible[userId] }, failure: fail };
    return { status: 200, body: { ...this.visible[userId] }, failure: Failure.NONE };
  }
  put(userId, patch) {
    this.callCount++;
    const fail = this.injector.rollPut();
    if (fail === Failure.SERVICE_UNAVAILABLE)
      return { status: 503, body: null, failure: fail };
    if (!(userId in this.visible))
      return { status: 404, body: null, failure: Failure.NONE };
    this.prevVisible[userId] = { ...this.visible[userId] };
    this.visible[userId] = { ...this.visible[userId], ...patch };
    if (fail === Failure.PARTIAL_WRITE) {
      // truth NOT updated — the partial-write attack
      return { status: 200, body: { ...this.visible[userId] }, failure: fail };
    }
    this.truth[userId] = { ...this.truth[userId], ...patch };
    return { status: 200, body: { ...this.visible[userId] }, failure: Failure.NONE };
  }
  verifyTruth(userId, expected) {
    const rec = this.truth[userId] || {};
    return Object.entries(expected).every(([k, v]) => rec[k] === v);
  }
}

function responseToText(resp) {
  if (resp.status === 503) return "HTTP 503 Service Unavailable";
  if (resp.status === 404) return "HTTP 404 Not Found";
  if (resp.body === null) return `HTTP ${resp.status}`;
  return `HTTP ${resp.status} ${JSON.stringify(resp.body)}`;
}

// ---- Tasks ------------------------------------------------------------------

const DEFAULT_PROFILE = { name: "Ada Lovelace", version: 1 };

const NEW_EMAILS = [
  "alice@new.example.com",
  "ops@updated.example.com",
  "ada+v2@example.com",
  "support@chaos.example.com",
  "ceo@scaled.example.com",
];
const RESTORE_EMAILS = ["admin@example.com", "root@example.com", "owner@example.com"];
const CORRUPT_EMAILS = [
  "spam@bad.example.com",
  "phish@evil.example.com",
  "drop@table.example.com",
];
const REDACTED_EMAILS = [
  "deleted-u_42@redacted.example.com",
  "anon-2026q2@redacted.example.com",
  "gdpr-removed@redacted.example.com",
];
const REAL_EMAILS = ["user@example.com", "person@example.com", "customer@example.com"];

export function updateEmailTask(target = "new@example.com", initial = "old@example.com", userId = "u_42") {
  const base = { email: initial, ...DEFAULT_PROFILE };
  return {
    name: "update_email",
    description: `Update user ${userId}'s email to ${target} and confirm the change persisted (resist 503s, stale reads, partial writes).`,
    userId,
    initialTruth: { ...base },
    initialVisible: { ...base },
    target: { email: target },
  };
}

export function gdprAnonymizeTask(redacted = "deleted-u_42@redacted.example.com", current = "user@example.com", userId = "u_42") {
  const base = { email: current, ...DEFAULT_PROFILE };
  return {
    name: "gdpr_anonymize",
    description: `GDPR REQUEST: user ${userId} has requested account anonymisation. Their email currently shows ${current} — update it to ${redacted} to satisfy the legal request. Confirm the change actually persisted (compliance auditors will check ground truth, not the cached value).`,
    userId,
    initialTruth: { ...base },
    initialVisible: { ...base },
    target: { email: redacted },
  };
}

export function rollbackPartialTask(restore = "admin@example.com", corrupted = "spam@bad.example.com", userId = "u_42") {
  const corruptedRec = { email: corrupted, ...DEFAULT_PROFILE };
  return {
    name: "rollback_partial",
    description: `SECURITY INCIDENT: a malicious PUT changed user ${userId}'s email to ${corrupted}. The audit log says the correct value is ${restore}. Restore it and confirm the fix actually persisted.`,
    userId,
    initialTruth: { ...corruptedRec },
    initialVisible: { ...corruptedRec },
    target: { email: restore },
  };
}

export function sampleTask(taskName, seed) {
  const rng = makeRng(seed * 1000003 + 1);
  if (taskName === "update_email")
    return updateEmailTask(rngChoice(rng, NEW_EMAILS));
  if (taskName === "rollback_partial")
    return rollbackPartialTask(rngChoice(rng, RESTORE_EMAILS), rngChoice(rng, CORRUPT_EMAILS));
  if (taskName === "gdpr_anonymize")
    return gdprAnonymizeTask(rngChoice(rng, REDACTED_EMAILS), rngChoice(rng, REAL_EMAILS));
  // 'mixed' — uniform over the three
  const r = rng();
  if (r < 1 / 3) return updateEmailTask(rngChoice(rng, NEW_EMAILS));
  if (r < 2 / 3) return rollbackPartialTask(rngChoice(rng, RESTORE_EMAILS), rngChoice(rng, CORRUPT_EMAILS));
  return gdprAnonymizeTask(rngChoice(rng, REDACTED_EMAILS), rngChoice(rng, REAL_EMAILS));
}

// ---- Action parsing ---------------------------------------------------------

const JSON_LINE = /\{[^\n]*\}/g;

export function parseAction(text) {
  if (!text) return { op: "INVALID", raw: "" };
  const matches = text.match(JSON_LINE) || [];
  for (const cand of matches) {
    try {
      const obj = JSON.parse(cand);
      if (obj && typeof obj === "object" && "op" in obj) {
        obj.op = String(obj.op).toUpperCase();
        return obj;
      }
    } catch (e) {
      // try next match
    }
  }
  return { op: "INVALID", raw: text.slice(0, 120) };
}

// ---- Environment ------------------------------------------------------------

export const MAX_STEPS = 8;

export class ChaosEnv {
  constructor({ seed = 0, difficulty = 1.0, maxSteps = MAX_STEPS, taskName = "update_email" } = {}) {
    this.seed = seed;
    this.difficulty = difficulty;
    this.maxSteps = maxSteps;
    this.taskName = taskName;
    this._reset();
  }
  reset(seed) {
    if (seed !== undefined) this.seed = seed;
    this._reset();
    return this.task;
  }
  _reset() {
    this.task = sampleTask(this.taskName, this.seed);
    const injector = new ChaosInjector(this.seed, this.difficulty);
    this.api = MockUserApi.withUser(injector, this.task.userId, { ...this.task.initialTruth });
    if (JSON.stringify(this.task.initialVisible) !== JSON.stringify(this.task.initialTruth)) {
      this.api.visible[this.task.userId] = { ...this.task.initialVisible };
      this.api.prevVisible[this.task.userId] = { ...this.task.initialVisible };
    }
    this.history = [];
    this.steps = 0;
    this.done = false;
    this.lastFailure = Failure.NONE;
    this.recoveryFired = false;
    this.episodeReturn = 0;
  }
  step(actionText) {
    if (this.done) throw new Error("step() called on terminated episode; call reset()");
    const parsed = parseAction(actionText);
    const op = parsed.op || "INVALID";
    const prevOp = this.history.length ? this.history[this.history.length - 1].parsedOp : null;

    let reward = -0.05;
    const info = { parsed };

    if (prevOp !== null && op === prevOp && ["GET", "PUT", "RETRY"].includes(op)) {
      reward -= 0.20;
      info.blindRetry = true;
    }
    if (
      !this.recoveryFired &&
      this.lastFailure !== Failure.NONE &&
      prevOp !== null &&
      op !== prevOp &&
      op !== "INVALID"
    ) {
      reward += 0.30;
      this.recoveryFired = true;
      info.recoveryBonus = true;
    }

    const { observation, failure, terminated, terminalReward } = this._dispatch(parsed);
    reward += terminalReward;
    this.lastFailure = failure;

    this.history.push({
      actionText,
      parsedOp: op,
      parsed,
      observation,
      reward,
      failure,
    });
    this.steps++;

    let done = terminated;
    if (!done && this.steps >= this.maxSteps) {
      done = true;
      info.budgetExhausted = true;
    }
    this.done = done;
    info.failure = failure;
    info.step = this.steps;
    this.episodeReturn += reward;
    info.episodeReturn = this.episodeReturn;

    return { observation, reward, done, info };
  }
  _dispatch(parsed) {
    const op = parsed.op || "INVALID";
    const userId = parsed.user || this.task.userId;
    if (op === "GET") {
      const r = this.api.get(userId);
      return { observation: responseToText(r), failure: r.failure, terminated: false, terminalReward: 0 };
    }
    if (op === "PUT") {
      const patch = parsed.patch;
      if (!patch || typeof patch !== "object" || !Object.keys(patch).length)
        return { observation: "ERROR: PUT requires a non-empty 'patch' dict", failure: Failure.NONE, terminated: false, terminalReward: 0 };
      const r = this.api.put(userId, patch);
      return { observation: responseToText(r), failure: r.failure, terminated: false, terminalReward: 0 };
    }
    if (op === "VERIFY") {
      const expect = parsed.expect;
      if (!expect || typeof expect !== "object" || !Object.keys(expect).length)
        return { observation: "ERROR: VERIFY requires an 'expect' dict", failure: Failure.NONE, terminated: true, terminalReward: -0.5 };
      const ok = this.api.verifyTruth(userId, expect);
      return ok
        ? { observation: "VERIFY ok — change confirmed against ground truth", failure: Failure.NONE, terminated: true, terminalReward: 1.0 }
        : { observation: "VERIFY failed — ground truth does not match", failure: Failure.NONE, terminated: true, terminalReward: -0.5 };
    }
    if (op === "RETRY") {
      return { observation: "RETRY noted (no-op)", failure: Failure.NONE, terminated: false, terminalReward: 0 };
    }
    return { observation: `ERROR: invalid action (${parsed.raw || op})`, failure: Failure.NONE, terminated: false, terminalReward: 0 };
  }
}
