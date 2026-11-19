// G1 Control UI. Talks only to the server (/api/*); the server relays to the robot Agent.
const $ = (id) => document.getElementById(id);
let UI = null;          // /api/ui-config
let state = null;       // last robot state

const MODE_LABEL = {
  zero_torque: "Zero Torque", damping: "Damping", squat: "Squat", sit: "Sit",
  stand_up: "Stand Up", locomotion: "Locomotion", manipulation: "Manipulation",
};

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}

function log(text, cls = "") {
  const li = document.createElement("li");
  li.className = cls;
  li.textContent = `${new Date().toLocaleTimeString()}  ${text}`;
  $("log").prepend(li);
  while ($("log").children.length > 80) $("log").lastChild.remove();
}

async function cmd(action, params = {}) {
  log(`→ ${action} ${Object.keys(params).length ? JSON.stringify(params) : ""}`, "run");
  try {
    const r = await api("/api/cmd", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ action, params, wait: true }),
    });
    log(`${r.ok ? "✓" : "✗"} ${action}: ${r.ok ? JSON.stringify(r.result) : r.error}`, r.ok ? "ok" : "bad");
    return r;
  } catch (e) {
    log(`✗ ${action}: ${e.message}`, "bad");
  }
}

// ------------------------------------------------------------------ 0 connection
$("btn-discover").onclick = async () => {
  $("found").innerHTML = "<li>Scanning…</li>";
  const { found } = await api("/api/discover", { method: "POST" });
  $("found").innerHTML = found.length ? "" : "<li>No robot found</li>";
  for (const f of found) {
    const li = document.createElement("li");
    li.innerHTML = `<span><b>${f.host}</b> <span class="muted">${f.robot} · mode_machine ${f.mode_machine}</span></span>`;
    const b = document.createElement("button");
    b.textContent = "Connect";
    b.onclick = () => connect(f.host);
    li.append(b);
    $("found").append(li);
  }
};

async function connect(host) {
  try {
    await api("/api/connect", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ host }) });
    log(`connected to ${host}`, "ok");
    const v = await cmd("get_volume");
    if (v && v.ok) { $("vol").value = v.result.volume; $("vol-val").textContent = v.result.volume; }
  } catch (e) { log(`connect failed: ${e.message}`, "bad"); }
}

$("btn-terminate").onclick = async () => {
  await api("/api/disconnect", { method: "POST" });
  log("connection terminated");
};

// ------------------------------------------------------------------ 1 modes
function buildModes() {
  const box = $("modes");
  for (const name of Object.keys(UI.modes.ids)) {
    const b = document.createElement("button");
    b.id = `mode-${name}`;
    b.textContent = `${MODE_LABEL[name] || name} (${UI.modes.ids[name]})`;
    if (name === "damping") b.classList.add("damping");
    if (UI.modes.disabled.includes(name)) b.title = "currently not in priority";
    b.onclick = () => {
      if (!confirm(`Switch to ${MODE_LABEL[name]}? The robot will move.`)) return;
      cmd("set_mode", { mode: name });
    };
    box.append(b);
  }
}

function updateModes(cur) {
  const allowed = new Set([...(UI.modes.transitions[cur] || []), "damping"]);
  for (const name of Object.keys(UI.modes.ids)) {
    const b = $(`mode-${name}`);
    b.classList.toggle("current", name === cur);
    b.disabled = !state || name === cur || UI.modes.disabled.includes(name) || !allowed.has(name);
  }
}

// ------------------------------------------------------------------ 2 basic
$("hl-color").oninput = (e) => ($("hl-hex").value = e.target.value.toUpperCase());
$("hl-hex").oninput = (e) => { if (/^#[0-9a-f]{6}$/i.test(e.target.value)) $("hl-color").value = e.target.value; };
$("btn-hl").onclick = () => cmd("headlight", { hex: $("hl-hex").value });

$("vol").oninput = (e) => ($("vol-val").textContent = e.target.value);
$("btn-vol").onclick = () => cmd("set_volume", { volume: Number($("vol").value) });

$("tts").oninput = () => ($("tts-count").textContent = `${$("tts").value.length}/${UI.audio.tts_max_chars}`);
$("btn-tts").onclick = () => { const t = $("tts").value.trim(); if (t) cmd("speak", { text: t }); };

$("btn-audio").onclick = async () => {
  const f = $("audio-file").files[0];
  if (!f) return log("choose an audio file first", "bad");
  if (f.size > UI.audio.file_max_mb * 1048576) return log(`file is over ${UI.audio.file_max_mb} MB`, "bad");
  const fd = new FormData();
  fd.append("file", f);
  log(`→ play_audio ${f.name}`, "run");
  try {
    const r = await api("/api/audio", { method: "POST", body: fd });
    log(`${r.ok ? "✓ playing" : "✗"} ${f.name} ${r.error || ""}`, r.ok ? "ok" : "bad");
  } catch (e) { log(`✗ audio: ${e.message}`, "bad"); }
};
$("btn-audio-stop").onclick = () => cmd("stop_audio");

function buildJoints() {
  const box = $("joints");
  for (const [group, idxs] of Object.entries(UI.joint_groups)) {
    const div = document.createElement("div");
    div.innerHTML = `<h4>${group.replace("_", " ")}</h4><table><thead><tr><th>Joint</th><th>Now °</th>
      <th>Min</th><th>Max</th><th>°C</th></tr></thead><tbody>${idxs.map((i) =>
      `<tr id="j${i}"><td>${i}</td><td></td><td></td><td></td><td></td></tr>`).join("")}</tbody></table>`;
    box.append(div);
  }
}

function updateJoints(js) {
  for (const j of js) {
    const tds = $(`j${j.index}`).children;
    tds[0].textContent = `${j.index} ${j.name.replace(/^(left|right)_/, "")}`;
    tds[1].textContent = j.deg == null ? "—" : j.deg.toFixed(1);
    tds[2].textContent = j.min;
    tds[3].textContent = j.max;
    tds[4].textContent = j.temp ?? "—";
    tds[4].className = j.temp >= 70 ? "vhot" : j.temp >= 55 ? "hot" : "";
  }
}

function kv(table, rows) {
  table.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

let camOpen = false;
function setCamera(open) {
  camOpen = open;
  $("cam-on").hidden = !open;
  $("cam-off").hidden = open;
  $("btn-camera").textContent = open ? "Close camera live feed" : "Open camera live feed";
  const t = Date.now();  // fresh URL so the browser reconnects the stream
  for (const kind of ["rgb", "depth", "view"]) {
    if (open) $(`cam-${kind}`).src = `/video/${kind}.mjpg?${t}`;
    else $(`cam-${kind}`).removeAttribute("src");
  }
}
$("btn-camera").onclick = () => setCamera(!camOpen);

$("btn-detect").onclick = async () => {
  const text = $("det-text").value.trim();
  $("btn-detect").disabled = true;
  log(text ? `→ detect "${text}"` : "→ clear tracking", "run");
  try {
    const r = await api("/api/detect", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }) });
    const found = r.items.filter((i) => i.found).map((i) => i.label);
    const missing = r.items.filter((i) => !i.found).map((i) => i.label);
    log(r.cleared ? "✓ tracking cleared" : `✓ tracking: ${found.join(", ") || "none"}${missing.length ? ` · not found: ${missing.join(", ")}` : ""}`, "ok");
  } catch (e) { log(`✗ detect: ${e.message}`, "bad"); }
  $("btn-detect").disabled = false;
};

function updateTracked(items) {
  $("tracked").innerHTML = (items || []).map((i) => !i.found ? `<li><span>${i.label}</span><span class="muted">not found</span></li>`
    : `<li><span>${i.label} <span class="muted">(${i.score})</span></span><span>${!i.ok ? "lost" : i.dist_cm ? `${i.dist_cm.toFixed(0)} cm` : "-- cm"}</span></li>`).join("");
}

// ------------------------------------------------------------------ 3 locomotion
function setupLoco() {
  const L = UI.locomotion;
  $("loco-v").value = L.speed_default;
  $("loco-v").max = Math.max(L.speed_max, L.turn_speed_max);
  $("loco-d").value = 2;
  $("loco-d").max = L.duration_max_s;
  document.querySelectorAll("[data-walk]").forEach((b) => (b.onclick = () => {
    const action = b.dataset.walk;
    const vmax = action.startsWith("turn") ? L.turn_speed_max : L.speed_max;
    const velocity = Math.min(Number($("loco-v").value), vmax);
    const duration = Math.min(Number($("loco-d").value), L.duration_max_s);
    if (!(velocity > 0 && duration > 0)) return log("velocity and duration must be > 0", "bad");
    cmd(action, { velocity, duration });
  }));
}

// ------------------------------------------------------------------ 4 manipulation
function buildManip() {
  for (const [group, idxs] of Object.entries(UI.arms.groups)) {
    const div = document.createElement("div");
    div.innerHTML = `<h4>${group.replace("_", " ")}</h4><table><thead><tr><th>Joint</th><th>Min</th><th>Max</th>
      <th>Now °</th><th>Set °</th><th>Add ±°</th></tr></thead><tbody>${idxs.map((i) => `<tr id="m${i}">
      <td></td><td></td><td></td><td></td>
      <td><input id="mset${i}" type="number" step="1"> <button data-set="${i}">Set</button></td>
      <td><input id="madd${i}" type="number" step="1" value="5"> <button data-add="${i}">Add</button></td>
      </tr>`).join("")}</tbody></table>`;
    $("manip").append(div);
  }
  $("manip").querySelectorAll("[data-set]").forEach((b) => (b.onclick = () => {
    const i = Number(b.dataset.set), v = $(`mset${i}`).value;
    if (v !== "") moveJoint(i, Number(v));
  }));
  $("manip").querySelectorAll("[data-add]").forEach((b) => (b.onclick = () => {
    const i = Number(b.dataset.add), v = $(`madd${i}`).value;
    if (v !== "" && state) moveJoint(i, state.joints[i].deg + Number(v));
  }));
  $("btn-release").onclick = () => cmd("release_upperbody");
  $("btn-reset").onclick = resetRobot;
}

// Reset after a failure (the robot freezes: hands open where they are, head light solid red): arms down, normal light
async function resetRobot() {
  const b = $("btn-reset");
  b.disabled = true;
  $("reset-stat").textContent = "resetting: arms down…";
  try {
    const r = await post("/api/reset");
    $("reset-stat").textContent = r.ok ? `ready · ${r.done}` : `NOT reset · ${r.done}`;
  } catch (e) {
    $("reset-stat").textContent = `reset failed · ${e.message || e}`;
  } finally {
    b.disabled = false;
  }
}

function showFailed(why) {
  const b = $("btn-reset");
  b.classList.toggle("alert", !!why);
  if (why) $("reset-stat").textContent = `FAILED · ${why} · press Reset`;
}

function moveJoint(i, target) {
  if (!state) return;
  const j = state.joints[i];
  target = Math.min(Math.max(target, j.min), j.max);
  const t = Math.max(0.5, Math.abs(target - j.deg) / UI.arms.max_speed_dps);
  cmd("play_upperbody", { keyframes: [{ t: Number(t.toFixed(2)), q: { [i]: Number(target.toFixed(2)) } }] });
}

function updateManip(js) {
  for (const idxs of Object.values(UI.arms.groups)) {
    for (const i of idxs) {
      const j = js[i], tds = $(`m${i}`).children;
      tds[0].textContent = `${i} ${j.name.replace(/^(left|right)_/, "")}`;
      tds[1].textContent = j.min;
      tds[2].textContent = j.max;
      tds[3].textContent = j.deg == null ? "—" : j.deg.toFixed(1);
      $(`mset${i}`).placeholder = j.deg == null ? "" : j.deg.toFixed(0);
    }
  }
}

function gate(id, on, why) {
  const sec = $(id);
  sec.classList.toggle("off", !on);
  sec.querySelectorAll("button, input").forEach((el) => {
    if (!el.hasAttribute("data-always")) el.disabled = !on || el.hasAttribute("data-never");
  });
  $(`${id}-gate`).textContent = on ? "" : why;
}

// ------------------------------------------------------------------ 6 motion planning, 7 AI control
async function post(path, body) {
  return api(path, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body || {}) });
}
const esc = (x) => String(x).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
const short = (o) => (o && Object.keys(o).length ? esc(JSON.stringify(o)) : "");

function setupPlanning() {
  for (const [ta, cnt] of [["p6-text", "p6-count"], ["ai-text", "ai-count"]]) {
    $(ta).maxLength = UI.ui.nl_text_max;
    $(ta).oninput = () => ($(cnt).textContent = `${$(ta).value.length}/${UI.ui.nl_text_max}`);
    $(cnt).textContent = `0/${UI.ui.nl_text_max}`;
  }
  $("btn-p6-gen").onclick = async () => {
    const text = $("p6-text").value.trim();
    if (!text) return;
    $("p6-state").textContent = "asking the LLM and planning…";
    log(`→ motion planning: ${text}`, "run");
    try {
      const r = await post("/api/plan6/generate", { text });
      log(`✓ motion ready: ${r.steps.length} steps, ${r.duration_s} s`, "ok");
    } catch (e) { log(`✗ motion planning: ${e.message}`, "bad"); }
  };
  $("btn-p6-play").onclick = async () => {
    if (!confirm("Play the planned arm motion on the robot?")) return;
    try { await post("/api/plan6/play"); log("→ playing motion", "run"); } catch (e) { log(`✗ play: ${e.message}`, "bad"); }
  };
  $("btn-ai-start").onclick = async () => {
    const text = $("ai-text").value.trim();
    if (!text) return;
    const mode = document.querySelector("input[name=ai-mode]:checked").value;
    if (mode === "auto" && !confirm("Run the whole task automatically?")) return;
    try { await post("/api/ai/start", { text, mode }); log(`→ AI task (${mode}): ${text}`, "run"); } catch (e) { log(`✗ AI: ${e.message}`, "bad"); }
  };
  $("btn-ai-next").onclick = () => post("/api/ai/next");
  $("btn-ai-abort").onclick = () => post("/api/ai/abort").then(() => log("AI abort requested", "bad"));
  document.querySelectorAll("input[name=ai-mode]").forEach((r) => (r.onchange = () => post("/api/ai/mode", { text: "-", mode: r.value })));
}

function renderHands(ee) {
  // section 5 follows what the robot detects on its wrists (dummy hands or the Inspire hands)
  if (!ee) return;
  const inspire = ee.type === "inspire";
  $("ee-model").hidden = $("ee-hands").hidden = !inspire;
  if (inspire) $("ee-model").textContent = ee.model || "Inspire";
  const pending = ee.pending ? (inspire ? " The hands stopped answering: dummy hands from the next time the arms are released."
    : " Inspire hands answer: they take over the next time the arms are released.") : "";
  $("ee-hint").textContent = (inspire
    ? "Open = every finger fully open; Close = the four fingers, then the thumb over them. Manual only: motion "
      + "planning treats them as hands without fingers."
    : "No end effector detected (dummy hands). This section activates by itself when the Inspire hands answer.") + pending;
  if (!inspire || !ee.hands) return;
  for (const side of ["left", "right"]) {
    const h = ee.hands[side], el = $(`ee-${side}-stat`);
    if (!h || !h.connected) { el.textContent = "not connected"; el.className = "pill bad"; el.title = h ? h.error || "" : ""; continue; }
    el.textContent = h.moving ? `${h.moving === "close" ? "closing" : "opening"}…` : (h.pose || "—");
    el.className = `pill ${h.moving ? "warn" : "ok"}`;
    el.title = `finger angles (0 closed .. 1000 open): ${h.angles.join(", ")}`;
  }
}

function renderPlan6(p) {
  if (!p) return;
  $("p6-state").textContent = p.state === "idle" ? "" : `${p.state}${p.current ? ` · ${p.current}` : ""}${p.duration_s ? ` · ${p.duration_s} s` : ""}`;
  $("btn-p6-play").disabled = p.state !== "ready" || $("s6").classList.contains("off");
  const steps = p.steps || [];
  $("p6-plan").innerHTML = p.llm_plan.map((n) => `<li><b>${esc(n.skill)}</b> ${short(n.params)}<span class="why">${esc(n.why || "")}</span></li>`).join("")
    + (steps.length ? `<li class="muted">${steps.length} arm steps: ${steps.map((x) => esc(x.label)).join(" → ")}</li>` : "")
    + (p.error ? `<li><span class="err">${esc(p.error)}</span></li>` : "");
  // the task scene: what each name refers to, the margin check, and the straight view with the names on it
  const sc = p.scene || {}, m = p.margin;
  const objs = Object.entries(sc.objects || {}).map(([k, o]) =>
    `<b>${esc(k)}</b>: ${o.stack > 1 ? `stack of ${o.stack}` : esc(o.shape || "")}, ${o.size_m.map((v) => Math.round(v * 100)).join("×")} cm at (${o.center_m[0].toFixed(2)}, ${o.center_m[1].toFixed(2)})`).join(" · ");
  $("p6-scene").innerHTML = (objs ? `<div>${objs}${sc.views_deg ? ` <span class="muted">(views at ${sc.views_deg.join(", ")}°)</span>` : ""}</div>` : "")
    + (m ? `<div class="${m.robust ? "" : "err"}">${m.robust ? "Margin: ROBUST (still plans with every object 2 cm off or turned 3°)"
        : `Margin: MARGINAL, fails if the scene is ${m.failed.map(esc).join(", ")}: expect some runs to fail`}</div>` : "");
  const img = $("p6-img");
  if (sc.dir && img.dataset.dir !== sc.dir) { img.dataset.dir = sc.dir; img.src = `/api/plan6/scene.png?d=${encodeURIComponent(sc.dir)}`; img.hidden = false; }
}

function renderAI(a) {
  if (!a) return;
  $("ai-state").textContent = a.state === "idle" ? "" : `${a.state} · ${a.mode}`;
  $("btn-ai-next").disabled = a.state !== "waiting";
  $("btn-ai-abort").disabled = !["planning", "running", "waiting"].includes(a.state);
  const loopLine = (n) => n.skill !== "approach" ? "" : `<span class="why">⟳ LOOP: look ahead → in reach? → exit (last leg, place)`
    + ` · else walk ≤50 cm → look again${n.loop ? ` · pass ${n.loop.pass} · walked ${n.loop.walked_m.toFixed(2)} of max ${n.loop.limit_m.toFixed(2)} m`
    + ` · ${esc(n.loop.last || "")}` : ""}</span>`;
  $("ai-nodes").innerHTML = a.nodes.map((n) => `<li><b>${esc(n.skill)}</b> ${short(n.params)}<span class="st ${n.status}">${n.status}</span>
    ${loopLine(n)}
    <span class="why">${esc(n.why || "")}${n.review ? ` · review: ${esc(n.review.decision)} (${esc(n.review.reason)})` : ""}</span>
    ${n.error ? `<span class="err">${esc(n.error)}</span>` : ""}</li>`).join("") + (a.error ? `<li><span class="err">${esc(a.error)}</span></li>` : "");
  $("ai-log").innerHTML = a.log.slice().reverse().map((l) => `<li>${esc(l)}</li>`).join("");
}

// ------------------------------------------------------------------ polling
function pill(id, text, cls = "") { const p = $(id); p.textContent = text; p.className = `pill ${cls}`; }

async function poll() {
  try {
    const s = await api("/api/status");
    if (!s.connected || !s.state) {
      state = null;
      pill("pill-conn", s.connected ? `${s.host} · no response` : "offline", "bad");
      updateModes(null);
      for (const id of ["s3", "s4", "s6", "s7"]) gate(id, false, "· not connected");
      return;
    }
    state = s.state;
    pill("pill-conn", `${s.host}${state.connected_dds ? "" : " · no DDS"}`, state.connected_dds ? "ok" : "warn");
    pill("pill-mode", `${MODE_LABEL[state.mode] || state.mode} (${state.fsm_id})`);
    const b = state.battery;
    pill("pill-batt", b ? `battery ${b.soc}%` : "battery —", b && b.soc < 20 ? "bad" : b && b.soc < 35 ? "warn" : "ok");
    const busy = [state.busy.motion, state.busy.audio].filter(Boolean);
    pill("pill-busy", busy.length ? busy.map((x) => `${x.action} ${x.for_s}s`).join(", ") : "idle", busy.length ? "warn" : "");
    updateModes(state.mode);
    updateJoints(state.joints);
    updateManip(state.joints);
    gate("s3", (UI.modes.walk_modes || ["locomotion"]).includes(state.mode), "· active in Locomotion or Manipulation mode");
    gate("s4", state.mode === "manipulation", "· active only in Manipulation mode");
    gate("s6", state.mode === "manipulation", "· active only in Manipulation mode");
    gate("s7", state.mode === "manipulation", "· active only in Manipulation mode");
    renderHands(state.end_effector);
    renderPlan6(s.plan6);
    renderAI(s.ai);
    showFailed(s.failed);
    const a = state.arms;
    $("arms-stat").textContent = `arms ${a.engaged ? "held by us" : "with policy"}${a.moving ? " · moving" : ""} · ${a.loop_hz} Hz`;
    updateTracked(s.tracking);
    const c = s.camera;
    $("cam-stat").textContent = c && c.online ? `${c.size[0]}×${c.size[1]} · ${c.fps} fps` : "· camera offline";
    if (b) kv($("batt"), [["SOC", `${b.soc} %`], ["Voltage", `${b.voltage_v} V`], ["Current", `${(b.current_ma / 1000).toFixed(2)} A`],
      ["Temp", `${Math.max(...b.temps_c)} °C`], ["Cycles", b.cycle]]);
    const p = state.imu.pelvis_rpy, t = state.imu.torso_rpy, f = (a, i) => (a ? a[i].toFixed(1) : "—");
    $("imu").innerHTML = `<tr><th>IMU °</th><th>Roll</th><th>Pitch</th><th>Yaw</th></tr>
      <tr><td>Pelvis (lower)</td><td>${f(p, 0)}</td><td>${f(p, 1)}</td><td>${f(p, 2)}</td></tr>
      <tr><td>Torso (upper)</td><td>${f(t, 0)}</td><td>${f(t, 1)}</td><td>${f(t, 2)}</td></tr>`;
  } catch (e) {
    pill("pill-conn", "server error", "bad");
  } finally {
    setTimeout(poll, UI.poll_ms);
  }
}

(async function init() {
  UI = await api("/api/ui-config");
  $("tts").maxLength = UI.audio.tts_max_chars;
  $("tts-count").textContent = `0/${UI.audio.tts_max_chars}`;
  $("audio-max").textContent = UI.audio.file_max_mb;
  $("det-text").maxLength = UI.ui.detect_text_max;
  $("det-max").textContent = UI.ui.detect_text_max;
  document.querySelectorAll(".hand").forEach((card) => card.querySelectorAll("button[data-ee]").forEach((b) =>
    b.addEventListener("click", () => cmd("set_end_effector_state", { side: card.dataset.side, state: b.dataset.ee }))));
  buildModes();
  buildJoints();
  setupLoco();
  buildManip();
  setupPlanning();
  poll();
})();
