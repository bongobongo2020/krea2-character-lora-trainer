"use strict";

// ---- tiny helpers -------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const api = async (path, opts = {}) => {
  const res = await fetch(`/api${path}`, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.headers.get("content-type")?.includes("application/json") ? res.json() : res;
};
const fmtBytes = (n) => {
  if (n > 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n > 1e6) return (n / 1e6).toFixed(1) + " MB";
  if (n > 1e3) return (n / 1e3).toFixed(0) + " KB";
  return n + " B";
};
const fmtTime = (s) => {
  if (s == null) return "";
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return `${h ? h + "h " : ""}${m}m ${sec}s`;
};

// ---- per-engine parameter schemas ---------------------------------------
const PARAM_FIELDS_AITK = [
  { key: "resolution", label: "Resolution", type: "number", desc: "Square training size (px). 1024 recommended." },
  { key: "network_dim", label: "LoRA rank (dim)", type: "number", desc: "Capacity. 32 is a good default." },
  { key: "network_alpha", label: "LoRA alpha", type: "number", desc: "Usually equal to rank." },
  { key: "learning_rate", label: "Learning rate", type: "number", step: "any", desc: "1e-4 works well for characters." },
  { key: "steps", label: "Steps", type: "number", desc: "Total training steps. 500–4000; ~2000 for a character." },
  { key: "save_every", label: "Save every (steps)", type: "number", desc: "Checkpoint frequency." },
  { key: "sample_every", label: "Sample every (steps)", type: "number", desc: "Preview-image frequency." },
  { key: "batch_size", label: "Batch size", type: "number", desc: "1 unless you have lots of VRAM." },
  { key: "seed", label: "Seed", type: "number", desc: "For reproducibility." },
  { key: "quantize", label: "Quantize base model", type: "checkbox", desc: "Lower VRAM (recommended)." },
  { key: "low_vram", label: "Low-VRAM mode", type: "checkbox", desc: "Enable for ~24GB or less." },
];

const PARAM_FIELDS_MUSUBI = [
  { key: "resolution", label: "Resolution", type: "number", desc: "Square training size (px). 1024 recommended." },
  { key: "num_repeats", label: "Repeats per image", type: "number", desc: "How many times each image is seen per epoch." },
  { key: "network_dim", label: "LoRA rank (dim)", type: "number", desc: "Capacity. 32 is a good default." },
  { key: "network_alpha", label: "LoRA alpha", type: "number", desc: "Usually equal to rank." },
  { key: "learning_rate", label: "Learning rate", type: "number", step: "any", desc: "1e-4 works well for characters." },
  { key: "max_train_epochs", label: "Epochs", type: "number", desc: "Total passes over the dataset." },
  { key: "save_every_n_epochs", label: "Save every N epochs", type: "number", desc: "Checkpoint frequency." },
  { key: "blocks_to_swap", label: "Blocks to swap", type: "number", desc: "Higher = less VRAM, slower. 26 for 12GB." },
  { key: "discrete_flow_shift", label: "Flow shift", type: "number", step: "any", desc: "Timestep shift. 2.5 default." },
  { key: "seed", label: "Seed", type: "number", desc: "For reproducibility." },
  { key: "fp8", label: "FP8 quantization", type: "checkbox", desc: "Required to fit on 12GB GPUs." },
];

const paramFields = (engine) => engine === "musubi" ? PARAM_FIELDS_MUSUBI : PARAM_FIELDS_AITK;
const engineLabel = (engine) => engine === "musubi" ? "Krea 2 Raw · musubi-tuner" : "Krea 2 Turbo · AI Toolkit";

const SETTINGS_FIELDS = [
  { key: "hf_home", label: "Central model cache (HF_HOME) — e.g. /mnt/ai-models/huggingface" },
  { key: "caption_model", label: "Auto-caption model (Qwen-VL HF repo)" },
  { key: "caption_python", label: "Caption Python (blank = auto-pick engine venv)" },
  { key: "aitoolkit_dir", label: "AI Toolkit directory" },
  { key: "aitoolkit_python", label: "AI Toolkit Python binary" },
  { key: "base_model", label: "Krea-2-Turbo base model (HF repo or path)" },
  { key: "assistant_lora", label: "De-distill adapter (HF repo or path)" },
  { key: "hf_token", label: "Hugging Face token (optional)" },
  { key: "musubi_dir", label: "Musubi-Tuner directory" },
  { key: "accelerate_bin", label: "accelerate binary" },
  { key: "dit_model", label: "Krea2 raw model (krea2_raw.safetensors)" },
  { key: "vae_model", label: "VAE model (qwen_image_vae.safetensors)" },
];

// ---- app state ----------------------------------------------------------
const state = {
  view: "projects",
  project: null,        // currently open project object
  logOffset: 0,
  pollTimer: null,
  lastOutputsRefresh: 0,  // throttle live checkpoint refresh during training
};

// ---- view switching -----------------------------------------------------
function showView(view) {
  state.view = view;
  $$(".view").forEach((v) => v.classList.remove("active"));
  $$(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  if (view === "project") {
    $("#view-project").classList.add("active");
  } else {
    $(`#view-${view}`).classList.add("active");
  }
  if (view === "settings") loadSettings();
  if (view === "projects") { stopPolling(); loadProjects(); }
}

$$("[data-view]").forEach((el) => {
  if (el.classList.contains("tab") || el.classList.contains("back")) {
    el.addEventListener("click", () => showView(el.dataset.view));
  }
});

// ---- projects list ------------------------------------------------------
async function loadProjects() {
  const { settings, environment } = await api("/settings");
  renderEnvBanner(environment);
  const { projects } = await api("/projects");
  const list = $("#project-list");
  list.innerHTML = "";
  if (!projects.length) {
    list.innerHTML = `<p class="muted">No projects yet. Create one above to get started.</p>`;
    return;
  }
  for (const p of projects) {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h4></h4><div class="meta">${p.images} image(s) · ${p.outputs} checkpoint(s)</div>` +
      `<span class="badge ${p.engine === "aitoolkit" ? "rec" : ""}">${engineLabel(p.engine)}</span>`;
    card.querySelector("h4").textContent = p.name;
    card.addEventListener("click", () => openProject(p.id));
    list.appendChild(card);
  }
}

// Warn only if neither engine is fully installed.
function renderEnvBanner(env) {
  const engines = Object.values(env.engines || {});
  const anyReady = engines.some((e) => e.ready);
  const banner = $("#env-banner");
  if (!anyReady) {
    banner.className = "banner warn";
    banner.innerHTML = `⚠ No training engine is installed yet. Open the <b>Setup</b> tab ` +
      `and click <b>Install AI Toolkit</b> (recommended) to get started.`;
  } else {
    banner.classList.add("hidden");
  }
}

$("#create-project").addEventListener("click", async () => {
  const name = $("#new-project-name").value.trim();
  if (!name) return;
  try {
    const proj = await api("/projects", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, engine: $("#new-project-engine").value }),
    });
    $("#new-project-name").value = "";
    openProject(proj.id);
  } catch (e) { alert("Could not create project: " + e.message); }
});

// ---- project detail -----------------------------------------------------
async function openProject(id) {
  const proj = await api(`/projects/${id}`);
  state.project = proj;
  state.logOffset = 0;
  $("#project-title").textContent = proj.name;
  const badge = $("#project-engine-badge");
  badge.textContent = engineLabel(proj.engine);
  badge.className = "badge " + (proj.engine === "aitoolkit" ? "rec" : "");
  $("#trigger-word").value = proj.params.trigger_word || "";
  renderParams(proj.params);
  renderHardware();
  renderImages(proj.images);
  renderOutputs(proj.outputs);
  renderSamples();
  $("#log-output").textContent = "No logs yet. Start a training run to see live output.";
  $("#progress-wrap").classList.add("hidden");
  showView("project");
  refreshStatus();
  refreshCommandPreview();
  refreshVram();
  loadAutoFreeToggle();
}

async function loadAutoFreeToggle() {
  try {
    const { settings } = await api("/settings");
    const on = String(settings.auto_free_vram ?? "true").toLowerCase();
    $("#auto-free-toggle").checked = ["1", "true", "yes", "on"].includes(on);
  } catch (_) {}
}

$("#auto-free-toggle").addEventListener("change", async (e) => {
  try {
    await api("/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ auto_free_vram: e.target.checked ? "true" : "false" }),
    });
  } catch (_) {}
});

// Human-readable summary of what auto-free did (empty if it did nothing).
function autoFreeMsg(a) {
  if (!a || !a.attempted || !a.actions?.length) return "";
  const names = a.actions.map((x) => `${x.name} (${x.method})`).join(", ");
  return `▸ Auto-freed VRAM: ${names} → ${(a.free_mb / 1024).toFixed(1)} GB free\n\n`;
}

async function refreshVram() {
  const banner = $("#vram-banner");
  try {
    const g = await api("/gpu");
    if (!g.available || !g.gpus?.length) { banner.classList.add("hidden"); renderVramProcs([]); return; }
    const gpu = g.gpus[0];
    const freeGb = (gpu.free_mb / 1024).toFixed(1), totalGb = (gpu.total_mb / 1024).toFixed(0);
    const isTurbo = state.project.engine === "aitoolkit";
    // Krea 2 (12B) training wants most of the card; warn if little is free.
    const needGb = isTurbo ? 18 : 11;
    if (gpu.free_mb / 1024 < needGb) {
      const hogs = (g.processes || []).filter((p) => p.used_mb > 500)
        .map((p) => `${p.name} (${(p.used_mb / 1024).toFixed(1)} GB)`).join(", ");
      banner.className = "banner warn";
      banner.innerHTML = `⚠ Only <b>${freeGb} GB</b> of ${totalGb} GB VRAM free — ` +
        `this engine typically needs ~${needGb} GB. Free it to avoid OOM` +
        (hogs ? `. Currently using VRAM: ${hogs}` : "") + ".";
    } else {
      banner.className = "banner ok-banner";
      banner.innerHTML = `✓ ${freeGb} GB of ${totalGb} GB VRAM free.`;
    }
    banner.classList.remove("hidden");
    renderVramProcs(g.processes || []);
  } catch (_) { banner.classList.add("hidden"); }
}

function renderVramProcs(processes) {
  const hogs = processes.filter((p) => p.used_mb > 500);
  // Populate every VRAM-procs widget on the page (Train step + Caption step).
  for (const el of $$(".vram-procs")) {
    el.innerHTML = "";
    if (!hogs.length) continue;
    const head = document.createElement("div");
    head.className = "vram-procs-head";
    head.textContent = "Processes holding the GPU (auto-freed on start; manual override below):";
    el.appendChild(head);
    for (const p of hogs) {
      const row = document.createElement("div");
      row.className = "vram-proc";
      const isComfy = /comfy/i.test(p.name);
      const ports = (p.ports || []).length ? ` · :${p.ports.join(", :")}` : "";
      row.innerHTML = `<span class="vram-proc-name">${p.name} <span class="muted">· PID ${p.pid} · ${(p.used_mb / 1024).toFixed(1)} GB${ports}</span></span>`;
      const actions = document.createElement("span");
      actions.className = "vram-proc-actions";
      if (p.system) {
        // Core OS/compositor process — show it for context, but never offer to kill it.
        const tag = document.createElement("span");
        tag.className = "muted";
        tag.textContent = "system process";
        actions.appendChild(tag);
      } else {
        if (isComfy) {
          const unload = document.createElement("button");
          unload.className = "ghost small";
          unload.textContent = "Unload models";
          unload.title = "Calls ComfyUI's /free API — frees VRAM without closing ComfyUI";
          unload.addEventListener("click", () => freeVram(p.pid, "comfy", p.name));
          actions.appendChild(unload);
        }
        const kill = document.createElement("button");
        kill.className = "danger small";
        kill.textContent = "End process";
        kill.addEventListener("click", () => freeVram(p.pid, "kill", p.name));
        actions.appendChild(kill);
      }
      row.appendChild(actions);
      el.appendChild(row);
    }
  }
}

async function freeVram(pid, mode, name) {
  if (mode === "kill" && !confirm(`End process "${name}" (PID ${pid})? This will close the app and free its VRAM.`)) return;
  try {
    const r = await api("/gpu/free", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pid, mode }),
    });
    const freedGb = (r.freed_mb / 1024).toFixed(1);
    await refreshVram();
    alert(`Freed ${freedGb} GB via ${r.method}. Now ${(r.free_mb / 1024).toFixed(1)} GB free.`);
  } catch (e) { alert(e.message); }
}

async function renderHardware() {
  const el = $("#hw-chip");
  try {
    const { hardware } = await api("/hardware");
    if (!hardware.gpu_name && !hardware.ram_mb) { el.classList.add("hidden"); return; }
    const parts = [];
    if (hardware.gpu_name) parts.push(`${hardware.gpu_name}`);
    if (hardware.vram_mb) parts.push(`${(hardware.vram_mb / 1024).toFixed(0)} GB VRAM`);
    if (hardware.ram_mb) parts.push(`${(hardware.ram_mb / 1024).toFixed(0)} GB RAM`);
    el.innerHTML = `⚙ Auto-tuned for <b>${parts.join(" · ")}</b>`;
    el.classList.remove("hidden");
  } catch (_) { el.classList.add("hidden"); }
}

function renderParams(params) {
  const grid = $("#params-grid");
  grid.innerHTML = "";
  for (const f of paramFields(state.project.engine)) {
    const wrap = document.createElement("div");
    wrap.className = "param" + (f.type === "checkbox" ? " checkbox" : "");
    if (f.type === "checkbox") {
      wrap.innerHTML = `<input type="checkbox" id="p-${f.key}" ${params[f.key] ? "checked" : ""}/>` +
        `<label for="p-${f.key}">${f.label}</label>`;
    } else {
      wrap.innerHTML = `<label>${f.label}` +
        `<input type="number" id="p-${f.key}" value="${params[f.key]}" ${f.step ? `step="${f.step}"` : ""}/></label>` +
        `<div class="desc">${f.desc}</div>`;
    }
    grid.appendChild(wrap);
  }
  grid.querySelectorAll("input").forEach((inp) =>
    inp.addEventListener("change", saveParams));
}

async function saveParams() {
  const params = {};
  for (const f of paramFields(state.project.engine)) {
    const el = $(`#p-${f.key}`);
    params[f.key] = f.type === "checkbox" ? el.checked : Number(el.value);
  }
  params.trigger_word = $("#trigger-word").value.trim();
  const proj = await api(`/projects/${state.project.id}/params`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ params }),
  });
  state.project = proj;
  refreshCommandPreview();
}

$("#trigger-word").addEventListener("change", saveParams);

// ---- images + captions --------------------------------------------------
function renderImages(images) {
  const grid = $("#image-grid");
  grid.innerHTML = "";
  for (const img of images) {
    const card = document.createElement("div");
    card.className = "thumb";
    const src = `/api/projects/${state.project.id}/images/${encodeURIComponent(img.filename)}`;
    card.innerHTML =
      `<img src="${src}" alt="" />` +
      `<textarea class="cap" data-file="${img.filename}" placeholder="caption…"></textarea>` +
      `<button class="delimg" data-file="${img.filename}">Remove</button>`;
    card.querySelector("textarea").value = img.caption;
    card.querySelector(".delimg").addEventListener("click", () => deleteImage(img.filename));
    grid.appendChild(card);
  }
}

async function deleteImage(filename) {
  const { images } = await api(`/projects/${state.project.id}/images/${encodeURIComponent(filename)}`, { method: "DELETE" });
  renderImages(images);
}

async function uploadFiles(files) {
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  try {
    const { images } = await api(`/projects/${state.project.id}/images`, { method: "POST", body: fd });
    renderImages(images);
  } catch (e) { alert("Upload failed: " + e.message); }
}

async function uploadZip(file) {
  const fd = new FormData();
  fd.append("file", file);
  try {
    const { images, saved } = await api(`/projects/${state.project.id}/import_zip`, { method: "POST", body: fd });
    renderImages(images);
    flash("#captions-saved", `Imported ${saved.length} image(s) from zip ✓`);
  } catch (e) { alert("Zip import failed: " + e.message); }
}

// Route a dropped/selected batch: a single .zip goes to the zip importer,
// everything else is treated as image uploads.
function handleDropped(fileList) {
  const files = Array.from(fileList);
  const zips = files.filter((f) => f.name.toLowerCase().endsWith(".zip"));
  const imgs = files.filter((f) => !f.name.toLowerCase().endsWith(".zip"));
  if (zips.length) zips.forEach(uploadZip);
  if (imgs.length) uploadFiles(imgs);
}

$("#browse-btn").addEventListener("click", () => $("#file-input").click());
$("#zip-btn").addEventListener("click", () => $("#zip-input").click());
$("#file-input").addEventListener("change", (e) => uploadFiles(e.target.files));
$("#zip-input").addEventListener("change", (e) => { if (e.target.files[0]) uploadZip(e.target.files[0]); });
const dz = $("#dropzone");
["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => handleDropped(e.dataTransfer.files));

$("#save-captions").addEventListener("click", async () => {
  const captions = {};
  $$("#image-grid .cap").forEach((t) => { captions[t.dataset.file] = t.value; });
  try {
    await api(`/projects/${state.project.id}/captions`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ captions }),
    });
    flash("#captions-saved", "Saved ✓");
  } catch (e) { flash("#captions-saved", e.message, true); }
});

// ---- auto-captioning (Qwen-VL) ------------------------------------------
let capTimer = null, capOffset = 0;

$("#autocaption-btn").addEventListener("click", async () => {
  await saveParams();  // persist trigger word first
  const overwrite = $("#autocaption-overwrite").checked;
  try {
    const r = await api(`/projects/${state.project.id}/autocaption`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ overwrite }),
    });
    capOffset = 0;
    $("#autocaption-log").textContent = autoFreeMsg(r.auto_free);
    $("#autocaption-log").classList.remove("hidden");
    $("#autocaption-state").textContent = `running ${r.model}…`;
    startCaptionPolling();
  } catch (e) { alert("Could not start captioning:\n\n" + e.message); }
});

function startCaptionPolling() {
  $("#autocaption-btn").disabled = true;
  if (capTimer) clearInterval(capTimer);
  capTimer = setInterval(pollCaption, 1500);
  pollCaption();
}

async function pollCaption() {
  let st;
  try { st = await api(`/projects/${state.project.id}/autocaption/status`); } catch (_) { return; }
  try {
    const { offset, content } = await api(`/projects/${state.project.id}/autocaption/logs?offset=${capOffset}`);
    if (content) {
      const box = $("#autocaption-log");
      box.textContent += content; capOffset = offset; box.scrollTop = box.scrollHeight;
    }
  } catch (_) {}
  if (st.running) { $("#autocaption-state").textContent = "captioning…"; refreshVram(); }
  if (!st.running) {
    $("#autocaption-state").textContent = st.exit_code === 0 ? "done ✓" : `failed (exit ${st.exit_code})`;
    if (capTimer) { clearInterval(capTimer); capTimer = null; }
    $("#autocaption-btn").disabled = false;
    // reload images so the new captions show in the boxes
    const proj = await api(`/projects/${state.project.id}`);
    renderImages(proj.images);
  }
}

$("#apply-trigger").addEventListener("click", () => {
  const tw = $("#trigger-word").value.trim();
  if (!tw) return;
  $$("#image-grid .cap").forEach((t) => {
    if (!t.value.includes(tw)) t.value = (tw + ", " + t.value).replace(/,\s*$/, "");
  });
});

// ---- command preview ----------------------------------------------------
async function refreshCommandPreview() {
  try {
    const { command, config, config_label } = await api(`/projects/${state.project.id}/preview_command`);
    $("#cmd-preview").textContent = `# ${config_label}\n${config}\n\n# command\n${command}`;
  } catch (e) { $("#cmd-preview").textContent = "Could not build command: " + e.message; }
}

// ---- training -----------------------------------------------------------
$("#start-train").addEventListener("click", async () => {
  await saveParams();
  const captions = {};
  $$("#image-grid .cap").forEach((t) => { captions[t.dataset.file] = t.value; });
  try {
    await api(`/projects/${state.project.id}/captions`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ captions }),
    });
    const res = await api(`/projects/${state.project.id}/train`, { method: "POST" });
    state.logOffset = 0;
    $("#log-output").textContent = autoFreeMsg(res.auto_free);
    refreshVram();
    startPolling();
  } catch (e) { alert("Could not start training:\n\n" + e.message); }
});

$("#stop-train").addEventListener("click", async () => {
  if (!confirm("Stop the current training run?")) return;
  await api(`/projects/${state.project.id}/stop`, { method: "POST" });
});

function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(() => { refreshStatus(); pullLogs(); }, 2000);
  refreshStatus(); pullLogs();
}
function stopPolling() {
  if (state.pollTimer) { clearInterval(state.pollTimer); state.pollTimer = null; }
}

async function pullLogs() {
  if (!state.project) return;
  try {
    const { offset, content } = await api(`/projects/${state.project.id}/logs?offset=${state.logOffset}`);
    if (content) {
      const box = $("#log-output");
      const atBottom = box.scrollTop + box.clientHeight >= box.scrollHeight - 30;
      box.textContent += content;
      state.logOffset = offset;
      if (atBottom) box.scrollTop = box.scrollHeight;
    }
  } catch (_) {}
}

async function refreshStatus() {
  if (!state.project) return;
  let st;
  try { st = await api(`/projects/${state.project.id}/status`); } catch (_) { return; }
  const stateEl = $("#train-state");
  const startBtn = $("#start-train"), stopBtn = $("#stop-train");

  if (st.running) {
    stateEl.textContent = `Training… ${fmtTime(st.elapsed)}`;
    stateEl.className = "train-state running";
    startBtn.disabled = true; stopBtn.disabled = false;
    renderProgress(st);
    refreshVram();
    if (!state.pollTimer) startPolling();
    // Checkpoints/samples are written mid-run; refresh them live (throttled)
    // so they show up without waiting for training to finish.
    if (Date.now() - state.lastOutputsRefresh > 15000) {
      state.lastOutputsRefresh = Date.now();
      refreshOutputs();
    }
  } else {
    stopBtn.disabled = true; startBtn.disabled = false;
    if (st.exit_code === 0) {
      stateEl.textContent = "Completed ✓"; stateEl.className = "train-state done";
    } else if (st.exit_code != null) {
      stateEl.textContent = `Stopped (exit ${st.exit_code})`; stateEl.className = "train-state error";
    } else {
      stateEl.textContent = "Idle"; stateEl.className = "train-state";
    }
    if (state.pollTimer) {  // finished: do a last log pull, refresh outputs, then stop
      pullLogs();
      refreshOutputs();
      stopPolling();
    }
  }
}

function renderProgress(st) {
  const wrap = $("#progress-wrap");
  if (st.step && st.total_steps) {
    wrap.classList.remove("hidden");
    const pct = Math.min(100, (st.step / st.total_steps) * 100);
    $("#progress-fill").style.width = pct.toFixed(1) + "%";
    let txt = `Step ${st.step} / ${st.total_steps} (${pct.toFixed(0)}%)`;
    if (st.epoch) txt += ` · epoch ${st.epoch}${st.total_epochs ? "/" + st.total_epochs : ""}`;
    if (st.loss != null) txt += ` · loss ${st.loss}`;
    $("#progress-text").textContent = txt;
  } else {
    wrap.classList.remove("hidden");
    $("#progress-fill").style.width = "100%";
    $("#progress-fill").style.opacity = "0.4";
    $("#progress-text").textContent = "Initializing / caching latents… (this can take a few minutes)";
  }
}

// ---- outputs ------------------------------------------------------------
async function refreshOutputs() {
  try {
    const proj = await api(`/projects/${state.project.id}`);
    renderOutputs(proj.outputs);
  } catch (_) {}
  renderSamples();
}

async function renderSamples() {
  const section = $("#samples-section"), wrap = $("#samples-groups");
  try {
    const { groups } = await api(`/projects/${state.project.id}/samples`);
    if (!groups.length) { section.classList.add("hidden"); return; }
    wrap.innerHTML = "";
    // Newest steps first so the latest/final samples are at the top.
    for (const g of [...groups].reverse()) {
      const row = document.createElement("div");
      row.className = "sample-group";
      row.innerHTML = `<div class="sample-step">${g.label}</div>`;
      const strip = document.createElement("div");
      strip.className = "sample-strip";
      for (const img of g.images) {
        const url = `/api/projects/${state.project.id}/sample/${img.relpath.split("/").map(encodeURIComponent).join("/")}`;
        const a = document.createElement("a");
        a.href = url; a.target = "_blank"; a.title = `${g.label} · prompt ${img.index + 1}`;
        a.innerHTML = `<img loading="lazy" src="${url}" alt="${g.label} prompt ${img.index + 1}" />`;
        strip.appendChild(a);
      }
      row.appendChild(strip);
      wrap.appendChild(row);
    }
    section.classList.remove("hidden");
  } catch (_) { section.classList.add("hidden"); }
}
function renderOutputs(outputs) {
  const list = $("#output-list");
  if (!outputs.length) { list.innerHTML = `<p class="muted">No checkpoints yet.</p>`; return; }
  list.innerHTML = "";
  for (const o of outputs) {
    const item = document.createElement("div");
    item.className = "output-item" + (o.final ? " final" : "");
    const rel = o.relpath || o.filename;
    const label = o.label || rel;
    const badge = o.final ? `<span class="badge rec">final</span> ` : "";
    item.innerHTML = `<span>${badge}<span class="name">${label}</span> ` +
      `<span class="size">${o.filename} · ${fmtBytes(o.size)}</span></span>`;
    const a = document.createElement("a");
    a.href = `/api/projects/${state.project.id}/outputs/${rel.split("/").map(encodeURIComponent).join("/")}`;
    a.textContent = "Download";
    a.className = o.final ? "primary" : "ghost";
    a.setAttribute("download", "");
    item.appendChild(a);
    list.appendChild(item);
  }
}

// ---- settings -----------------------------------------------------------
async function loadSettings() {
  const { settings, environment } = await api("/settings");
  const form = $("#settings-form");
  form.innerHTML = "";
  for (const f of SETTINGS_FIELDS) {
    const lbl = document.createElement("label");
    lbl.innerHTML = `${f.label}<input type="text" id="s-${f.key}" value="${settings[f.key] || ""}"/>`;
    form.appendChild(lbl);
  }
  const status = $("#env-status");
  status.innerHTML = "";
  for (const eng of Object.values(environment.engines)) {
    const group = document.createElement("div");
    group.className = "env-group";
    group.innerHTML = `<div class="env-group-head">${eng.label} ` +
      `<span class="badge ${eng.ready ? "ok" : "bad"}">${eng.ready ? "ready" : "not installed"}</span></div>`;
    for (const item of eng.items) {
      const row = document.createElement("div");
      row.className = "env-row";
      const note = item.info && !item.exists ? `` : (item.info ? ` <span class="muted">(${item.info})</span>` : "");
      row.innerHTML = `<span class="env-dot ${item.exists ? "ok" : "bad"}"></span>` +
        `<span>${item.label}${note}</span><span class="path">${item.path}</span>`;
      group.appendChild(row);
    }
    status.appendChild(group);
  }
  renderModelScan();
  // Resume the live log if a setup job is already running in the background.
  try {
    const st = await api("/setup/status");
    if (st.running && !setupTimer) startSetupPolling();
  } catch (_) {}
}

async function renderModelScan() {
  const el = $("#model-scan");
  if (!el) return;
  try {
    const scan = await api("/models/scan");
    let html = `<div class="env-group-head">Model cache <span class="path">${scan.hub}</span></div>`;
    for (const m of scan.models) {
      const sz = m.size ? ` <span class="muted">(${(m.size / 1e9).toFixed(1)} GB)</span>` : "";
      html += `<div class="env-row"><span class="env-dot ${m.cached ? "ok" : "bad"}"></span>` +
        `<span>${m.label}${m.cached ? " — cached" : " — will download"}${sz}</span>` +
        `<span class="path">${m.repo}</span></div>`;
    }
    el.innerHTML = html;
  } catch (_) { el.innerHTML = ""; }
}

// ---- setup jobs (install / download) ------------------------------------
let setupTimer = null, setupOffset = 0;

async function pollSetup() {
  let st;
  try { st = await api("/setup/status"); } catch (_) { return; }
  const wrap = $("#setup-log-wrap");
  if (st.running || st.exit_code != null) wrap.classList.remove("hidden");
  $("#setup-job-name").textContent = st.running
    ? `${st.name}…` : (st.exit_code === 0 ? "Done ✓" : st.exit_code != null ? `Failed (exit ${st.exit_code})` : "");
  $("#setup-spinner").style.display = st.running ? "inline-block" : "none";

  try {
    const { offset, content } = await api(`/setup/logs?offset=${setupOffset}`);
    if (content) {
      const box = $("#setup-log");
      box.textContent += content;
      setupOffset = offset;
      box.scrollTop = box.scrollHeight;
    }
  } catch (_) {}

  if (!st.running && setupTimer) {
    clearInterval(setupTimer); setupTimer = null;
    loadSettings();           // refresh the readiness dots
    setupButtons(false);
  }
}

const SETUP_BTNS = "#install-aitoolkit,#predownload-turbo,#install-musubi,.dl-btn";
const setupButtons = (disabled) =>
  document.querySelectorAll(SETUP_BTNS).forEach((b) => (b.disabled = disabled));

function startSetupPolling() {
  setupOffset = 0;
  $("#setup-log").textContent = "";
  setupButtons(true);
  if (setupTimer) clearInterval(setupTimer);
  setupTimer = setInterval(pollSetup, 1500);
  pollSetup();
}

async function runSetup(path) {
  try {
    await api(path, { method: "POST" });
    startSetupPolling();
  } catch (e) { alert(e.message); }
}

$("#install-aitoolkit").addEventListener("click", () => runSetup("/setup/install_aitoolkit"));
$("#predownload-turbo").addEventListener("click", () => runSetup("/setup/predownload_turbo"));
$("#install-musubi").addEventListener("click", () => runSetup("/setup/install_musubi"));

let modelSuggestions = null;
async function getSuggestions() {
  if (!modelSuggestions) modelSuggestions = await api("/setup/model_suggestions");
  return modelSuggestions;
}

$$(".model-dl").forEach((block) => {
  const target = block.dataset.target;

  block.querySelector(".autofill-btn").addEventListener("click", async () => {
    try {
      const s = (await getSuggestions())[target];
      block.querySelector(".dl-file").value = s.filename;
      if (s.repos?.length && !block.querySelector(".dl-repo").value)
        block.querySelector(".dl-repo").value = s.repos[0];
      const dl = block.querySelector(`#${target}-repos`);
      dl.innerHTML = (s.repos || []).map((r) => `<option value="${r}"></option>`).join("");
      block.querySelector(".dl-note").textContent = s.note || "";
    } catch (e) { alert(e.message); }
  });

  block.querySelector(".dl-btn").addEventListener("click", async () => {
    const payload = {
      target,
      url: block.querySelector(".dl-url").value.trim(),
      hf_repo: block.querySelector(".dl-repo").value.trim(),
      hf_file: block.querySelector(".dl-file").value.trim(),
      hf_token: block.querySelector(".dl-token").value.trim(),
    };
    if (!payload.url && !(payload.hf_repo && payload.hf_file)) {
      alert("Provide a direct URL, or a Hugging Face repo + filename."); return;
    }
    try {
      await api("/setup/download_model", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      startSetupPolling();
    } catch (e) { alert(e.message); }
  });
});

$("#save-settings").addEventListener("click", async () => {
  const payload = {};
  for (const f of SETTINGS_FIELDS) payload[f.key] = $(`#s-${f.key}`).value.trim();
  try {
    await api("/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    flash("#settings-saved", "Saved ✓");
    loadSettings();
  } catch (e) { flash("#settings-saved", e.message, true); }
});

function flash(sel, msg, isErr = false) {
  const el = $(sel);
  el.textContent = msg;
  el.className = "saved-msg" + (isErr ? " err" : "");
  setTimeout(() => { el.textContent = ""; }, 3000);
}

// ---- boot ---------------------------------------------------------------
loadProjects();
