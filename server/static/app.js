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

// ------------------------------------------------------------------ polling
function pill(id, text, cls = "") { const p = $(id); p.textContent = text; p.className = `pill ${cls}`; }

async function poll() {
  try {
    const s = await api("/api/status");
    if (!s.connected) { state = null; pill("pill-conn", "offline", "bad"); updateModes(null); return; }
    if (!s.state) { state = null; pill("pill-conn", `${s.host} · no response`, "bad"); updateModes(null); return; }
    state = s.state;
    pill("pill-conn", `${s.host}${state.connected_dds ? "" : " · no DDS"}`, state.connected_dds ? "ok" : "warn");
    pill("pill-mode", `${MODE_LABEL[state.mode] || state.mode} (${state.fsm_id})`);
    const b = state.battery;
    pill("pill-batt", b ? `battery ${b.soc}%` : "battery —", b && b.soc < 20 ? "bad" : b && b.soc < 35 ? "warn" : "ok");
    const busy = [state.busy.motion, state.busy.audio].filter(Boolean);
    pill("pill-busy", busy.length ? busy.map((x) => `${x.action} ${x.for_s}s`).join(", ") : "idle", busy.length ? "warn" : "");
    updateModes(state.mode);
    updateJoints(state.joints);
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
  $("ee-hint").textContent = UI.end_effector.type === "dummy" ? "Dummy hands: no end effector detected." : "";
  buildModes();
  buildJoints();
  poll();
})();
