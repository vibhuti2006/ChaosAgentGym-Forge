// ChaosAgentGym frontend — wires the env, policies, and UI together.

import { ChaosEnv, Failure, parseAction } from "./env.js";
import { RandomPolicy, ScriptedPolicy, NaivePolicy } from "./policies.js";

// ---- state ------------------------------------------------------------------

const state = {
  env: null,
  truthRevealed: false,
  rolloutTimer: null,
  evalRunning: false,
  // for the multi-episode chart
  rewardCurve: { Random: [], Scripted: [], Naive: [] },
};

// ---- DOM helpers ------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function el(tag, props = {}, children = []) {
  const e = document.createElement(tag);
  Object.entries(props).forEach(([k, v]) => {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "data") Object.entries(v).forEach(([dk, dv]) => (e.dataset[dk] = dv));
    else e.setAttribute(k, v);
  });
  (Array.isArray(children) ? children : [children]).forEach((c) => {
    if (c == null) return;
    if (typeof c === "string") e.appendChild(document.createTextNode(c));
    else e.appendChild(c);
  });
  return e;
}

function toast(msg, kind = "") {
  let t = $(".toast");
  if (!t) {
    t = el("div", { class: "toast" });
    document.body.appendChild(t);
  }
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  clearTimeout(t._timer);
  t._timer = setTimeout(() => (t.className = "toast"), 1800);
}

// ---- env construction -------------------------------------------------------

function newEnv() {
  const seed = Number($("#seed").value);
  const difficulty = Number($("#difficulty").value);
  const taskName = $("#task").value;
  state.env = new ChaosEnv({ seed, difficulty, taskName });
  state.truthRevealed = false;
  if (state.rolloutTimer) {
    clearInterval(state.rolloutTimer);
    state.rolloutTimer = null;
  }
  renderAll();
  toast(`reset · seed=${seed} · ${state.env.task.name}`);
}

// ---- rendering --------------------------------------------------------------

function renderAll() {
  renderTask();
  renderState();
  renderTranscript();
  renderMetrics();
  renderActionAvailability();
}

function renderTask() {
  const env = state.env;
  const card = $("#task-card");
  card.classList.remove("task-rollback", "task-gdpr", "task-update");
  if (env.task.name === "rollback_partial") card.classList.add("task-rollback");
  else if (env.task.name === "gdpr_anonymize") card.classList.add("task-gdpr");
  else card.classList.add("task-update");

  $("#task-desc").textContent = env.task.description;
  const targetEmail = env.task.target.email || JSON.stringify(env.task.target);
  $("#task-target").textContent = targetEmail;
  $("#task-user").textContent = env.task.userId;
  $("#task-name-badge").textContent = env.task.name;

  // pre-fill the PUT input with the target so manual play is easy
  const putIn = $("#put-email");
  if (putIn && document.activeElement !== putIn) putIn.value = targetEmail;
}

function renderState() {
  const env = state.env;
  const userId = env.task.userId;
  const visible = env.api.visible[userId] || {};
  const truth = env.api.truth[userId] || {};
  const diverged = JSON.stringify(visible) !== JSON.stringify(truth);

  const visEl = $("#store-visible");
  visEl.textContent = JSON.stringify(visible, null, 2);
  visEl.classList.remove("diverged");

  const truthEl = $("#store-truth");
  truthEl.textContent = JSON.stringify(truth, null, 2);
  truthEl.classList.toggle("hidden", !state.truthRevealed);
  truthEl.classList.toggle("diverged", diverged && state.truthRevealed);

  $("#diverge-flag").textContent = diverged ? "diverged" : "in sync";
  $("#diverge-flag").style.color = diverged ? "var(--red)" : "var(--green)";

  $("#reveal-btn").textContent = state.truthRevealed ? "Hide truth" : "Reveal truth";
}

function renderMetrics() {
  const env = state.env;
  $("#m-return").textContent = env.episodeReturn.toFixed(2);
  $("#m-return").className = "val " + (env.episodeReturn > 0 ? "green" : env.episodeReturn < 0 ? "red" : "");
  $("#m-steps").textContent = `${env.steps} / ${env.maxSteps}`;
  $("#m-failures").textContent = env.history.filter((h) => h.failure !== Failure.NONE).length;
  $("#m-status").textContent = env.done ? (env.history.length && env.history[env.history.length - 1].observation.startsWith("VERIFY ok") ? "✓ success" : "✗ done") : "running";
  $("#m-status").className = "val " + (env.done ? (env.history.length && env.history[env.history.length - 1].observation.startsWith("VERIFY ok") ? "green" : "red") : "yellow");
}

function renderTranscript() {
  const env = state.env;
  const t = $("#transcript");
  t.innerHTML = "";
  env.history.forEach((h, i) => {
    const isTerm = h.parsedOp === "VERIFY";
    const success = isTerm && h.observation.startsWith("VERIFY ok");
    const line = el("div", {
      class: `tline ${isTerm ? (success ? "terminal" : "terminal-fail") : ""}`,
    });
    line.appendChild(el("div", { class: "step-num" }, String(i + 1)));
    line.appendChild(el("div", { class: "op", data: { op: h.parsedOp } }, h.parsedOp));

    const obs = el("div", { class: "obs" });
    obs.appendChild(document.createTextNode(h.observation));
    if (h.failure && h.failure !== Failure.NONE) {
      obs.appendChild(el("span", { class: `fail-tag ${h.failure.toUpperCase()}` }, h.failure.replace(/_/g, " ")));
    }
    line.appendChild(obs);

    const r = h.reward;
    line.appendChild(el("div", { class: `reward ${r > 0 ? "pos" : r < 0 ? "neg" : ""}` }, (r >= 0 ? "+" : "") + r.toFixed(2)));
    t.appendChild(line);
  });
  // auto-scroll
  t.scrollTop = t.scrollHeight;
}

function renderActionAvailability() {
  const disabled = state.env.done;
  $$("#action-grid .action-btn").forEach((b) => (b.disabled = disabled));
  $("#put-email").disabled = disabled;
  $("#run-policy").disabled = false; // can always run a fresh rollout
}

// ---- manual actions ---------------------------------------------------------

function takeAction(actionText) {
  if (state.env.done) return;
  state.env.step(actionText);
  renderAll();
  if (state.env.done) {
    const last = state.env.history[state.env.history.length - 1];
    if (last.observation.startsWith("VERIFY ok")) toast("Episode success ✓", "success");
    else if (last.parsedOp === "VERIFY") toast("VERIFY failed ✗", "error");
    else toast("Step budget exhausted", "error");
  }
}

function bindActions() {
  $$("#action-grid .action-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const op = btn.dataset.op;
      const userId = state.env.task.userId;
      let action;
      if (op === "GET") action = JSON.stringify({ op: "GET", user: userId });
      else if (op === "PUT") {
        const email = $("#put-email").value || state.env.task.target.email;
        action = JSON.stringify({ op: "PUT", user: userId, patch: { email } });
      } else if (op === "VERIFY") {
        const email = $("#put-email").value || state.env.task.target.email;
        action = JSON.stringify({ op: "VERIFY", user: userId, expect: { email } });
      } else if (op === "RETRY") action = JSON.stringify({ op: "RETRY" });
      takeAction(action);
    });
  });
}

// ---- policy rollout (animated) ---------------------------------------------

function makePolicy(name) {
  if (name === "Random") return new RandomPolicy(Number($("#seed").value) + 1);
  if (name === "Scripted") return new ScriptedPolicy();
  if (name === "Naive") return new NaivePolicy();
  return new ScriptedPolicy();
}

function runPolicyAnimated() {
  if (state.rolloutTimer) {
    clearInterval(state.rolloutTimer);
    state.rolloutTimer = null;
    $("#run-policy").textContent = "▶ Run policy";
    return;
  }
  newEnv();
  const policy = makePolicy($("#policy-select").value);
  $("#run-policy").textContent = "■ Stop";
  state.rolloutTimer = setInterval(() => {
    if (state.env.done) {
      clearInterval(state.rolloutTimer);
      state.rolloutTimer = null;
      $("#run-policy").textContent = "▶ Run policy";
      const last = state.env.history[state.env.history.length - 1];
      const ok = last && last.observation.startsWith("VERIFY ok");
      toast(`${policy.name} → return=${state.env.episodeReturn.toFixed(2)} ${ok ? "✓" : "✗"}`, ok ? "success" : "error");
      return;
    }
    const action = policy.act(state.env);
    state.env.step(action);
    renderAll();
  }, 550);
}

// ---- multi-episode evaluation ----------------------------------------------

function runEpisode(policy, taskName, seed, difficulty) {
  const env = new ChaosEnv({ seed, difficulty, taskName });
  if (policy.reset) policy.reset();
  while (!env.done) {
    const action = policy.act(env);
    env.step(action);
  }
  const last = env.history[env.history.length - 1];
  const success = last && last.observation.startsWith("VERIFY ok");
  // behaviour fingerprint
  const ops = env.history.map((h) => h.parsedOp);
  const putCount = ops.filter((o) => o === "PUT").length;
  const verifyIdx = ops.indexOf("VERIFY");
  const prematureVerify = verifyIdx !== -1 && putCount === 0;
  const defendedVerify = verifyIdx !== -1 && putCount >= 2;
  return {
    success,
    return: env.episodeReturn,
    steps: env.steps,
    putCount,
    prematureVerify,
    defendedVerify,
    failures: env.history.filter((h) => h.failure !== Failure.NONE).length,
  };
}

async function runEval() {
  if (state.evalRunning) return;
  state.evalRunning = true;
  $("#run-eval").disabled = true;
  $("#eval-status").textContent = "running…";

  const N = Number($("#eval-n").value);
  const taskName = $("#eval-task").value;
  const difficulty = Number($("#difficulty").value);
  const policies = [
    { name: "Random", make: (s) => new RandomPolicy(s) },
    { name: "Naive", make: () => new NaivePolicy() },
    { name: "Scripted", make: () => new ScriptedPolicy() },
  ];

  const results = {};
  state.rewardCurve = {};
  for (const p of policies) {
    const rows = [];
    for (let i = 0; i < N; i++) {
      const seed = 1000 + i;
      const policy = p.make(seed);
      rows.push(runEpisode(policy, taskName, seed, difficulty));
      // yield to keep UI responsive every 25 episodes
      if (i % 25 === 0) await new Promise((r) => setTimeout(r));
    }
    results[p.name] = rows;
    state.rewardCurve[p.name] = rows.map((r) => r.return);
  }

  renderEvalTable(results, N);
  renderRewardCurve(state.rewardCurve);

  state.evalRunning = false;
  $("#run-eval").disabled = false;
  $("#eval-status").textContent = `done · ${N} eps × 3 policies`;
}

function renderEvalTable(results, N) {
  const tbody = $("#eval-tbody");
  tbody.innerHTML = "";
  Object.entries(results).forEach(([name, rows]) => {
    const succ = rows.filter((r) => r.success).length / N;
    const meanReturn = rows.reduce((a, r) => a + r.return, 0) / N;
    const meanSteps = rows.reduce((a, r) => a + r.steps, 0) / N;
    const meanFails = rows.reduce((a, r) => a + r.failures, 0) / N;
    const prem = rows.filter((r) => r.prematureVerify).length / N;
    const def = rows.filter((r) => r.defendedVerify).length / N;
    const tr = el("tr", {}, [
      el("td", {}, name),
      el("td", { html: `<span class="bar" style="width:${(succ * 60).toFixed(0)}px"></span>${(succ * 100).toFixed(0)}%` }),
      el("td", {}, (meanReturn >= 0 ? "+" : "") + meanReturn.toFixed(2)),
      el("td", {}, meanSteps.toFixed(1)),
      el("td", {}, meanFails.toFixed(2)),
      el("td", {}, (prem * 100).toFixed(0) + "%"),
      el("td", {}, (def * 100).toFixed(0) + "%"),
    ]);
    tbody.appendChild(tr);
  });
}

function renderRewardCurve(curves) {
  const svg = $("#reward-chart");
  const W = svg.viewBox.baseVal.width;
  const H = svg.viewBox.baseVal.height;
  const padL = 36, padR = 12, padT = 12, padB = 24;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  svg.innerHTML = "";

  const all = Object.values(curves).flat();
  if (!all.length) return;
  const N = Math.max(...Object.values(curves).map((c) => c.length));

  const yMin = -1.0;
  const yMax = 1.2;
  const x = (i) => padL + (i / Math.max(N - 1, 1)) * innerW;
  const y = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * innerH;

  // y gridlines
  for (let i = 0; i <= 4; i++) {
    const v = yMin + (i / 4) * (yMax - yMin);
    const yy = y(v);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", padL); line.setAttribute("x2", W - padR);
    line.setAttribute("y1", yy); line.setAttribute("y2", yy);
    line.setAttribute("class", "grid");
    svg.appendChild(line);
    const lbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    lbl.setAttribute("x", padL - 6); lbl.setAttribute("y", yy + 3);
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("class", "label");
    lbl.textContent = v.toFixed(1);
    svg.appendChild(lbl);
  }

  // axes
  const axisX = document.createElementNS("http://www.w3.org/2000/svg", "line");
  axisX.setAttribute("x1", padL); axisX.setAttribute("x2", W - padR);
  axisX.setAttribute("y1", H - padB); axisX.setAttribute("y2", H - padB);
  axisX.setAttribute("class", "axis");
  svg.appendChild(axisX);

  const xLbl = document.createElementNS("http://www.w3.org/2000/svg", "text");
  xLbl.setAttribute("x", W / 2); xLbl.setAttribute("y", H - 6);
  xLbl.setAttribute("text-anchor", "middle");
  xLbl.setAttribute("class", "label");
  xLbl.textContent = "episode";
  svg.appendChild(xLbl);

  const colors = { Random: "#5ccfe6", Naive: "#ffcc66", Scripted: "#a3e635" };
  Object.entries(curves).forEach(([name, c]) => {
    if (!c.length) return;
    const pts = c.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", pts);
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", colors[name] || "#ccc");
    poly.setAttribute("stroke-width", "1.5");
    poly.setAttribute("opacity", "0.55");
    svg.appendChild(poly);

    // moving-average line (window 10) — easier to read than per-episode
    const win = 10;
    const ma = [];
    for (let i = 0; i < c.length; i++) {
      const lo = Math.max(0, i - win + 1);
      const slice = c.slice(lo, i + 1);
      ma.push(slice.reduce((a, b) => a + b, 0) / slice.length);
    }
    const maPts = ma.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    const mLine = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    mLine.setAttribute("points", maPts);
    mLine.setAttribute("fill", "none");
    mLine.setAttribute("stroke", colors[name] || "#ccc");
    mLine.setAttribute("stroke-width", "2.5");
    svg.appendChild(mLine);
  });
}

// ---- copy button ------------------------------------------------------------

function bindCopyButtons() {
  $$(".code .copy").forEach((btn) => {
    btn.addEventListener("click", () => {
      const code = btn.parentElement.dataset.code || btn.parentElement.textContent.replace(btn.textContent, "").trim();
      navigator.clipboard.writeText(code).then(() => toast("copied to clipboard", "success"));
    });
  });
}

// ---- init -------------------------------------------------------------------

function init() {
  // bind controls
  $("#reset-btn").addEventListener("click", newEnv);
  $("#task").addEventListener("change", newEnv);
  $("#seed").addEventListener("change", newEnv);
  $("#difficulty").addEventListener("input", () => {
    $("#difficulty-val").textContent = Number($("#difficulty").value).toFixed(2);
  });
  $("#difficulty").addEventListener("change", newEnv);
  $("#new-seed").addEventListener("click", () => {
    $("#seed").value = Math.floor(Math.random() * 99999);
    newEnv();
  });
  $("#reveal-btn").addEventListener("click", () => {
    state.truthRevealed = !state.truthRevealed;
    renderState();
  });
  $("#run-policy").addEventListener("click", runPolicyAnimated);
  $("#run-eval").addEventListener("click", runEval);

  bindActions();
  bindCopyButtons();
  newEnv();
}

document.addEventListener("DOMContentLoaded", init);
