// JS port of training/policies.py — RandomPolicy and ScriptedPolicy.
// LLMPolicy intentionally omitted (would need a model server).

import { makeRng } from "./env.js";

function actionTemplates(userId, target) {
  return [
    JSON.stringify({ op: "GET", user: userId }),
    JSON.stringify({ op: "PUT", user: userId, patch: target }),
    JSON.stringify({ op: "VERIFY", user: userId, expect: target }),
    JSON.stringify({ op: "RETRY" }),
  ];
}

export class RandomPolicy {
  constructor(seed = 0) {
    this.rng = makeRng(seed);
    this.name = "Random";
  }
  reset() {}
  act(env) {
    const t = actionTemplates(env.task.userId, env.task.target);
    return t[Math.floor(this.rng() * t.length)];
  }
}

export class ScriptedPolicy {
  constructor() {
    this.step = 0;
    this.name = "Scripted Oracle";
  }
  reset() {
    this.step = 0;
  }
  act(env) {
    const t = actionTemplates(env.task.userId, env.task.target);
    // PUT -> GET -> PUT -> GET -> VERIFY (defends against partial writes)
    const plan = [t[1], t[0], t[1], t[0], t[2]];
    const a = plan[Math.min(this.step, plan.length - 1)];
    this.step++;
    return a;
  }
}

export class NaivePolicy {
  // Strawman — PUT once, VERIFY immediately. Gets fooled by partial writes.
  constructor() {
    this.step = 0;
    this.name = "Naive (PUT-then-VERIFY)";
  }
  reset() {
    this.step = 0;
  }
  act(env) {
    const t = actionTemplates(env.task.userId, env.task.target);
    const plan = [t[1], t[2]];
    const a = plan[Math.min(this.step, plan.length - 1)];
    this.step++;
    return a;
  }
}
