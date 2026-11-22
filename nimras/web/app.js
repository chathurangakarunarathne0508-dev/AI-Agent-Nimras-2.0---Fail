"use strict";
// Everything shown here may originate from the open web: ALWAYS escape.
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? esc(u) : "#");
const $ = (id) => document.getElementById(id);
const fmtTime = (t) => (t ? new Date(t * 1000).toLocaleString() : "–");
const ago = (t) => {
  if (!t) return "–";
  const s = Math.round(Date.now() / 1000 - t), a = Math.abs(s);
  const v = a < 60 ? `${a}s` : a < 3600 ? `${Math.round(a / 60)}m` : a < 86400 ? `${Math.round(a / 3600)}h` : `${Math.round(a / 86400)}d`;
  return s >= 0 ? `${v} ago` : `in ${v}`;
};
const badge = (s) => `<span class="badge b-${esc(s)}">${esc(s)}</span>`;
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
const bar = (x) => `<div class="bar"><i style="width:${Math.max(0, Math.min(100, (x || 0) * 100))}%"></i></div>`;
const empty = (msg) => `<div class="empty">${esc(msg)}</div>`;

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-NIMRAS": "1" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

function table(cols, rows, emptyMsg = "Nothing yet.") {
  if (!rows.length) return empty(emptyMsg);
  return `<div class="tablewrap"><table><thead><tr>${cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.h)}</th>`).join("")}</tr></thead><tbody>${rows
    .map((r) => `<tr>${cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.f ? c.f(r) : esc(r[c.k])}</td>`).join("")}</tr>`)
    .join("")}</tbody></table></div>`;
}

// ---------------------------------------------------------------- tabs
let currentTab = "voice";
try { currentTab = localStorage.getItem("nimras.tab") || "voice"; } catch (_) {}
function showTab(name) {
  currentTab = name;
  try { localStorage.setItem("nimras.tab", name); } catch (_) {}
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("on", s.id === `tab-${name}`));
  ({ library: loadLibrary, topics: loadTopics, sessions: loadSessions, memory: loadMemory, audit: loadAudit, settings: loadSettings, performance: loadModels, voicelab: () => window.loadVoiceLab && window.loadVoiceLab() }[name] || (() => {}))();
}
$("tabs").addEventListener("click", (e) => e.target.dataset.tab && showTab(e.target.dataset.tab));

// ---------------------------------------------------------------- overview
let dash = null;
async function refresh() {
  try { dash = await api("/api/dashboard"); } catch (e) { $("stateLine").textContent = `API unreachable: ${e.message}`; return; }
  const s = dash.status, c = dash.counts;
  $("pulse").className = `pulse ${s.state}`;
  $("stateLine").textContent = `${s.state.toUpperCase()}${s.detail ? " — " + s.detail : ""} · next session ${ago(s.next_session_at)}`;
  $("btnPause").textContent = s.paused ? "Resume" : "Pause";

  const issues = [];
  if (!s.ollama.reachable) issues.push(`Ollama is not reachable (${s.ollama.error}). Start Ollama to enable learning.`);
  else {
    if (!s.ollama.analysis_model_ready) issues.push(`Model <code>${esc(s.ollama.analysis_model)}</code> is not pulled: run <code>ollama pull ${esc(s.ollama.analysis_model)}</code>`);
    if (!s.ollama.embed_model_ready) issues.push(`Embedding model <code>${esc(s.ollama.embed_model)}</code> is not pulled: run <code>ollama pull ${esc(s.ollama.embed_model)}</code> (keyword search is used until then)`);
  }
  if (!s.internet_enabled) issues.push("Internet access is disabled in settings, so autonomous learning is idle.");
  $("healthBanner").hidden = !issues.length;
  $("healthBanner").innerHTML = issues.map((i) => `<div>⚠ ${i}</div>`).join("");

  const k = [
    ["Resources collected", c.resources_collected], ["Verified entries", c.entries_verified],
    ["Knowledge entries", c.entries_total], ["Verified claims", `${c.claims_verified}/${c.claims_total}`],
    ["Topics learned", c.topics_learned], ["Topics queued", c.topics_queued],
    ["Learning sessions", c.sessions], ["Archive chunks", c.archive_chunks],
    ["Open contradictions", c.open_contradictions], ["Injections blocked", c.injections_blocked],
  ];
  $("kpis").innerHTML = k.map(([l, v]) => `<div class="kpi"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("");

  const ls = dash.last_session;
  $("statusBox").innerHTML = `<div class="kv">
    <div>State</div><div>${badge(s.state)} ${esc(s.detail || "")}</div>
    <div>Current topic</div><div>${esc(s.current_topic || "–")}</div>
    <div>Next scheduled session</div><div>${fmtTime(s.next_session_at)} (${ago(s.next_session_at)})</div>
    <div>Discovery providers</div><div>${s.providers.map(esc).join(", ") || "none available"}</div>
    <div>Models</div><div>chat <code>${esc(s.ollama.chat_model || "?")}</code> · embed <code>${esc(s.ollama.embed_model || "?")}</code></div>
    <div>Ollama web search</div><div>${s.ollama.web_search_configured ? "configured" : "not configured (set OLLAMA_API_KEY)"}</div>
    <div>Sandbox</div><div>${s.sandbox.available ? "Docker isolated" : esc(s.sandbox.detail)}</div>
    <div>Last session</div><div>${ls ? `#${ls.id} ${badge(ls.status)} ${ago(ls.ended_at || ls.started_at)}` : "none yet"}</div>
    <div>Last backup</div><div>${ago(s.last_backup_at)}</div></div>`;

  $("activeTopics").innerHTML = table(
    [{ h: "Topic", f: (r) => esc(r.name) }, { h: "Origin", k: "origin" }, { h: "Status", f: (r) => badge(r.status) }, { h: "Priority", num: 1, f: (r) => r.priority.toFixed(2) }],
    dash.active_topics, "Queue is empty.");
  $("discoveries").innerHTML = table(
    [{ h: "Technology", f: (r) => `${esc(r.name)}${r.note ? `<div class="muted">${esc(r.note)}</div>` : ""}` }, { h: "Via", k: "origin" }, { h: "Status", f: (r) => badge(r.status) }, { h: "Found", f: (r) => ago(r.created_at) }],
    dash.recent_discoveries, "No discoveries yet. Emerging-tech discovery runs every few sessions.");
  $("categories").innerHTML = table(
    [{ h: "Layer", k: "layer" }, { h: "Category", k: "category" }, { h: "Entries", num: 1, k: "n" }, { h: "Verified", num: 1, k: "verified" }, { h: "", f: (r) => bar(r.verified / r.n) }],
    dash.categories);
  $("reliability").innerHTML = table(
    [{ h: "Domain", k: "domain" }, { h: "Sources", num: 1, k: "n" }, { h: "Credibility", f: (r) => `${r.avg_credibility} ${bar(r.avg_credibility)}` },
     { h: "Corrob.", num: 1, k: "supported" }, { h: "Contra.", num: 1, k: "contradicted" }, { h: "Inj.", num: 1, k: "injections" }],
    dash.source_reliability);
  $("gaps").innerHTML = table(
    [{ h: "Gap", f: (r) => esc(r.description) }, { h: "From", k: "origin" }, { h: "Status", f: (r) => badge(r.status) },
     { h: "", f: (r) => (r.status === "open" ? `<button class="small" data-dismiss="${r.id}">dismiss</button>` : "") }],
    dash.gaps, "No knowledge gaps identified.");
  $("failures").innerHTML = table(
    [{ h: "When", f: (r) => ago(r.created_at) }, { h: "Stage", k: "stage" }, { h: "Topic", k: "topic" }, { h: "Provider", k: "provider" },
     { h: "Target", f: (r) => (r.url ? `<a href="${safeUrl(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.url.slice(0, 70))}</a>` : esc(r.query || "")) },
     { h: "Error", f: (r) => `<span class="muted">${esc(r.error)}</span>` }],
    dash.failures, "No failures recorded.") +
    (dash.failed_topics.length ? `<h2 style="margin-top:14px">Topics that failed</h2>` + table(
      [{ h: "Topic", k: "name" }, { h: "Attempts", num: 1, k: "attempts" }, { h: "Status", f: (r) => badge(r.status) }, { h: "Retry", f: (r) => ago(r.next_due_at) },
       { h: "", f: (r) => `<button class="small" data-requeue="${r.id}">retry now</button>` }], dash.failed_topics) : "");
  renderPerformance();
}

document.body.addEventListener("click", async (e) => {
  const t = e.target;
  if (t.dataset.dismiss) { await api(`/api/gaps/${t.dataset.dismiss}/dismiss`, {}); refresh(); }
  if (t.dataset.requeue) { await api(`/api/topics/${t.dataset.requeue}/requeue`, {}); refresh(); loadTopics(); }
  if (t.dataset.park) { await api(`/api/topics/${t.dataset.park}/park`, {}); loadTopics(); }
  if (t.dataset.approve) { await api(`/api/memory/${t.dataset.approve}/approve`, {}); loadMemory(); }
  if (t.dataset.delmem) { if (confirm("Delete this memory?")) { await api(`/api/memory/${t.dataset.delmem}/delete`, {}); loadMemory(); } }
  if (t.dataset.entry) openEntry(t.dataset.entry);
});

$("btnRun").onclick = async () => { await api("/api/learn/run", {}); $("stateLine").textContent = "Learning session requested…"; setTimeout(refresh, 1500); };
$("btnStop").onclick = async () => { await api("/api/learn/stop", {}); refresh(); };
$("btnPause").onclick = async () => { await api(dash && dash.status.paused ? "/api/scheduler/resume" : "/api/scheduler/pause", {}); refresh(); };

// ---------------------------------------------------------------- performance
function renderPerformance() {
  if (!dash) return;
  const p = dash.performance;
  const runs = [...p.evaluations].reverse();
  const series = [["retrieval_recall_at_k", "var(--accent)", "retrieval recall@k"], ["response_score_mean", "var(--accent2)", "response score"], ["verified_entry_ratio", "var(--violet)", "verified ratio"]];
  if (runs.length >= 1) {
    const W = 600, H = 150, pad = 20, n = Math.max(runs.length - 1, 1);
    const paths = series.map(([key, color]) => {
      const pts = runs.map((r, i) => [pad + (i / n) * (W - 2 * pad), H - pad - (r.metrics?.[key] ?? NaN) * (H - 2 * pad)]).filter((p) => !isNaN(p[1]));
      return pts.length ? `<polyline fill="none" stroke="${color}" stroke-width="2" points="${pts.map((p) => p.join(",")).join(" ")}"/>${pts.map((p) => `<circle cx="${p[0]}" cy="${p[1]}" r="3" fill="${color}"/>`).join("")}` : "";
    }).join("");
    $("evalChart").innerHTML = `<div class="chart"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><line x1="${pad}" x2="${W - pad}" y1="${H - pad}" y2="${H - pad}" stroke="var(--line)"/><line x1="${pad}" x2="${W - pad}" y1="${pad}" y2="${pad}" stroke="var(--line)" stroke-dasharray="3"/>${paths}</svg></div>
      <div class="muted" style="font-size:12px">${series.map(([, c, l]) => `<span style="color:${c}">■</span> ${esc(l)}`).join(" &nbsp; ")} (0–100%)</div>`;
  } else $("evalChart").innerHTML = empty("No evaluation runs yet. They run automatically every few sessions, or click 'Run self-evaluation now'.");
  const d = (r, k) => { const v = r.deltas?.[k]; return v == null ? "" : ` <span style="color:${v >= 0 ? "var(--ok)" : "var(--bad)"}">${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}</span>`; };
  $("evalTable").innerHTML = table([
    { h: "Run", f: (r) => `#${r.id} · ${ago(r.started_at)}` },
    { h: "Responses", num: 1, f: (r) => esc(r.metrics?.responses_evaluated ?? "–") },
    { h: "Answer score", num: 1, f: (r) => pct(r.metrics?.response_score_mean) + d(r, "response_score_mean") },
    { h: "Recall@k", num: 1, f: (r) => pct(r.metrics?.retrieval_recall_at_k) + d(r, "retrieval_recall_at_k") },
    { h: "MRR", num: 1, f: (r) => esc(r.metrics?.retrieval_mrr ?? "–") },
    { h: "Verified ratio", num: 1, f: (r) => pct(r.metrics?.verified_entry_ratio) + d(r, "verified_entry_ratio") },
    { h: "Fetch success", num: 1, f: (r) => pct(r.metrics?.fetch_success_rate_7d) },
    { h: "Merged", num: 1, f: (r) => esc(r.metrics?.entries_merged ?? 0) },
    { h: "Prompt", f: (r) => esc(r.metrics?.prompt_decision || "") },
  ], p.evaluations, "");
  $("strategies").innerHTML = table([{ h: "Kind", k: "kind" }, { h: "Strategy", k: "name" }, { h: "Uses", num: 1, k: "uses" }, { h: "Success", f: (r) => `${pct(r.success_rate)} ${bar(r.success_rate)}` }], p.strategies, "No searches yet.") +
    `<div class="muted" style="margin-top:6px;font-size:12px">7-day fetch success: ${pct(p.fetch_success_7d)}. Strategies are chosen by a UCB bandit rewarded by verified knowledge yield.</div>`;
  $("mistakes").innerHTML = table([{ h: "Mistake", k: "signature" }, { h: "×", num: 1, k: "occurrences" }, { h: "Last", f: (r) => ago(r.last_seen) }], p.repeated_mistakes, "No mistakes recorded.");
  $("prompts").innerHTML = table([{ h: "Prompt", k: "name" }, { h: "v", num: 1, k: "version" }, { h: "Status", f: (r) => badge(r.status) },
    { h: "Mean score", num: 1, f: (r) => (r.mean_score == null ? "–" : r.mean_score.toFixed(3)) }, { h: "Samples", num: 1, k: "samples" },
    { h: "Rationale", f: (r) => `<span class="muted">${esc((r.rationale || "").slice(0, 160))}</span>` }], p.prompts);
}
$("runEval").onclick = async () => { await api("/api/eval/run", {}); $("evalNote").textContent = "Evaluation started in the background. Results appear here when it finishes."; };

async function loadModels() {
  renderPerformance();
  const m = await api("/api/models");
  $("models").innerHTML = (m.training_enabled ? "" : `<div class="muted" style="margin-bottom:8px">Training module is disabled (opt-in: <code>training.enabled</code>). Knowledge learning does not change model weights; it updates the knowledge base.</div>`) +
    table([{ h: "Model", k: "name" }, { h: "Status", f: (r) => badge(r.status) }, { h: "Eval", num: 1, f: (r) => esc(r.eval_score ?? "–") }, { h: "Baseline", num: 1, f: (r) => esc(r.baseline_score ?? "–") }, { h: "Notes", k: "notes" }], m.versions) +
    (m.datasets.length ? `<h2 style="margin-top:12px">Datasets</h2>` + table([{ h: "#", k: "id" }, { h: "Train", num: 1, k: "n_train" }, { h: "Hold-out", num: 1, k: "n_holdout" }, { h: "Ready", f: (r) => (r.quality?.ready_for_training ? "yes" : "not yet") }, { h: "Path", k: "path" }], m.datasets) : "");
}

// ---------------------------------------------------------------- library
let selectedEntry = null;
async function loadLibrary() {
  const q = $("libQ").value.trim();
  const params = new URLSearchParams({ q, layer: $("libLayer").value, status: $("libStatus").value });
  const r = await api(`/api/library?${params}`);
  $("libList").innerHTML = (r.entries.length ? r.entries.map((e) => `<div class="item ${e.id == selectedEntry ? "sel" : ""}" data-entry="${e.id}">
      <div class="t" data-entry="${e.id}">${esc(e.title)}</div>
      <div class="m" data-entry="${e.id}">${badge(e.status)} <span>${esc(e.layer)} / ${esc(e.category)}</span> <span>conf ${(e.confidence || 0).toFixed(2)}</span> <span>${e.n_sources} sources</span> <span>v${e.version}</span></div></div>`).join("")
    : empty(q ? "No matching entries." : "The library is empty. N.I.M.R.A.S. fills it as it learns.")) +
    (r.archive_hits.length ? `<h2 style="margin-top:10px">Internet archive matches</h2>` + r.archive_hits.map((h) => `<div class="item"><div class="t">${esc(h.title)}</div><div class="m">${badge(h.status)} <a href="${safeUrl(h.url)}" target="_blank" rel="noopener noreferrer">${esc((h.url || "").slice(0, 60))}</a></div><div class="muted" style="font-size:12px;margin-top:4px">${esc(h.text.slice(0, 260))}…</div></div>`).join("") : "");
}
$("libGo").onclick = loadLibrary;
$("libQ").addEventListener("keydown", (e) => e.key === "Enter" && loadLibrary());
["libLayer", "libStatus"].forEach((id) => ($(id).onchange = loadLibrary));

async function openEntry(id) {
  selectedEntry = id;
  document.querySelectorAll(".item").forEach((el) => el.classList.toggle("sel", el.dataset.entry == id));
  const e = await api(`/api/entries/${id}`);
  const d = e.details || {};
  const list = (arr) => (arr && arr.length ? `<ul>${arr.map((x) => `<li>${esc(typeof x === "string" ? x : `${x.with}: ${x.difference}`)}</li>`).join("")}</ul>` : empty("–"));
  const ex = d.example;
  $("libDetail").innerHTML = `<h3>${esc(e.title)}</h3>
    <div class="m">${badge(e.status)} ${esc(e.layer)} / ${esc(e.category)} · confidence ${(e.confidence || 0).toFixed(2)} · version ${e.version} · updated ${ago(e.updated_at)} · last verified ${ago(e.last_verified_at)}</div>
    <p>${esc(e.summary)}</p>
    ${d.purpose ? `<div class="sec"><h2>Purpose</h2><p>${esc(d.purpose)}</p></div>` : ""}
    ${d.advantages ? `<div class="sec"><h2>Advantages</h2>${list(d.advantages)}</div><div class="sec"><h2>Limitations</h2>${list(d.limitations)}</div><div class="sec"><h2>Compared with</h2>${list(d.comparison)}</div>` : ""}
    ${d.unverified_points?.length ? `<div class="sec"><h2>Unverified points</h2>${list(d.unverified_points)}</div>` : ""}
    ${ex && ex.code ? `<div class="sec"><h2>Example (${esc(ex.language)}) ${badge(ex.test?.status || "untested")}</h2><pre>${esc(ex.code)}</pre>${ex.test?.reason ? `<div class="muted">${esc(ex.test.reason)}</div>` : ""}${ex.test?.stdout ? `<h2>Sandbox output</h2><pre>${esc(ex.test.stdout)}</pre>` : ""}${ex.test?.stderr ? `<pre>${esc(ex.test.stderr)}</pre>` : ""}</div>` : ""}
    <div class="sec"><h2>Claims (${e.claims.length})</h2>${e.claims.map((c) => `<div class="claim">${badge(c.status)} ${esc(c.text)} <span class="muted">(${(c.confidence || 0).toFixed(2)})</span>
       <div class="srcs">${c.sources.map((s) => `${s.stance === "contradicts" ? "✗" : "✓"} <a href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.domain)}</a> (${s.credibility})`).join(" · ")}</div></div>`).join("") || empty("No claims.")}</div>
    ${e.contradictions.length ? `<div class="sec"><h2>Contradictions</h2>${e.contradictions.map((x) => `<div class="claim">${badge(x.resolution)} ${esc(x.new_statement)}<div class="srcs">${esc(x.note || "")} <a href="${safeUrl(x.url)}" target="_blank" rel="noopener noreferrer">${esc(x.url || "")}</a></div></div>`).join("")}</div>` : ""}
    <div class="sec"><h2>Sources</h2>${table([{ h: "Source", f: (s) => `<a href="${safeUrl(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title || s.url)}</a>` }, { h: "Credibility", num: 1, k: "credibility" }, { h: "Published", k: "published_at" }, { h: "Retrieved", f: (s) => fmtTime(s.retrieved_at) }], e.sources)}</div>
    <div class="sec"><h2>Version history</h2>${table([{ h: "v", num: 1, k: "version" }, { h: "Status", f: (v) => badge(v.status) }, { h: "Confidence", num: 1, k: "confidence" }, { h: "Reason", k: "reason" }, { h: "Changed", f: (v) => fmtTime(v.changed_at) }], e.versions, "First version.")}</div>`;
}

// ---------------------------------------------------------------- topics
async function loadTopics() {
  const st = $("topicFilter").value;
  const rows = await api(`/api/topics${st ? `?status=${encodeURIComponent(st)}` : ""}`);
  $("topicTable").innerHTML = table([
    { h: "Topic", f: (r) => `${esc(r.name)}${r.note ? `<div class="muted">${esc(r.note)}</div>` : ""}` }, { h: "Layer", k: "layer" }, { h: "Origin", k: "origin" },
    { h: "Status", f: (r) => badge(r.status) }, { h: "Priority", num: 1, f: (r) => r.priority.toFixed(2) }, { h: "Attempts", num: 1, k: "attempts" },
    { h: "Last researched", f: (r) => ago(r.last_researched_at) }, { h: "Next due", f: (r) => ago(r.next_due_at) },
    { h: "", f: (r) => (r.status === "parked" || r.status === "learned" || r.status === "failed" ? `<button class="small" data-requeue="${r.id}">queue</button>` : `<button class="small" data-park="${r.id}">park</button>`) },
  ], rows);
}
$("topicFilter").onchange = loadTopics;
$("addTopic").onclick = async () => {
  const name = $("newTopic").value.trim();
  if (!name) return;
  await api("/api/topics", { name, deep_study: $("newTopicDeep").checked });
  $("newTopic").value = "";
  loadTopics();
};

// ---------------------------------------------------------------- sessions
async function loadSessions() {
  const rows = await api("/api/sessions");
  $("sessionTable").innerHTML = table([
    { h: "#", k: "id" }, { h: "Trigger", k: "trigger" }, { h: "Status", f: (r) => badge(r.status) },
    { h: "Started", f: (r) => fmtTime(r.started_at) }, { h: "Duration", num: 1, f: (r) => (r.ended_at ? `${Math.round((r.ended_at - r.started_at) / 60 * 10) / 10} min` : "running") },
    { h: "Topics", f: (r) => (r.topics || []).map((t) => `${esc(t.topic)} <span class="muted">(${t.sources_used} src${t.verification ? ", " + esc(t.verification.status) : ""})</span>`).join("<br>") },
    { h: "Fetched", num: 1, f: (r) => esc(r.stats?.fetched ?? "") }, { h: "Claims +", num: 1, f: (r) => esc(r.stats?.claims_added ?? "") },
    { h: "Corrob.", num: 1, f: (r) => esc(r.stats?.claims_corroborated ?? "") }, { h: "Dupes", num: 1, f: (r) => esc(r.stats?.duplicates ?? "") },
    { h: "LLM calls", num: 1, f: (r) => esc(r.stats?.llm_calls ?? "") }, { h: "Note", f: (r) => `<span class="muted">${esc(r.error || "")}</span>` },
  ], rows, "No learning sessions yet.");
}

// ---------------------------------------------------------------- chat
let convId = null;
try { convId = sessionStorage.getItem("nimras.conv"); } catch (_) {}
function addMsg(role, text, refs) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = esc(text) + (refs && refs.length ? `<div class="refs">Knowledge used: ${refs.map((r) => `${badge(r.status)} ${r.url ? `<a href="${safeUrl(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>` : esc(r.title)}`).join(" · ")}</div>` : "");
  $("chatLog").appendChild(div);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
}
// Voice conversation lives in the Voice tab (local Voice Engine); this is the text fallback.
async function sendChat(msg) {
  addMsg("user", msg);
  const thinking = document.createElement("div");
  thinking.className = "msg assistant muted"; thinking.textContent = "Thinking…";
  $("chatLog").appendChild(thinking);
  $("chatLog").scrollTop = $("chatLog").scrollHeight;
  const started = Date.now();
  const tick = setInterval(() => { thinking.textContent = `Thinking… ${Math.round((Date.now() - started) / 1000)}s`; }, 1000);
  try {
    const r = await api("/api/chat", { message: msg, conversation_id: convId });
    convId = r.conversation_id;
    try { sessionStorage.setItem("nimras.conv", convId); } catch (_) {}
    thinking.remove();
    addMsg("assistant", r.reply, r.retrieved);
  } catch (err) { thinking.textContent = `Error: ${err.message}`; }
  finally { clearInterval(tick); }
}
$("chatForm").onsubmit = (e) => {
  e.preventDefault();
  const msg = $("chatInput").value.trim();
  if (!msg) return;
  $("chatInput").value = "";
  sendChat(msg);
};

// ---------------------------------------------------------------- memory
async function loadMemory() {
  const rows = await api("/api/memory");
  $("memTable").innerHTML = table([
    { h: "#", k: "id" }, { h: "Key", k: "key" }, { h: "Value", k: "value" }, { h: "Status", f: (r) => (r.approved ? badge("active") : badge("pending")) },
    { h: "", f: (r) => `${r.approved ? "" : `<button class="small" data-approve="${r.id}">approve</button> `}<button class="small" data-delmem="${r.id}">delete</button>` },
  ], rows, "No personal memory.");
}
$("memAdd").onclick = async () => {
  const key = $("memKey").value.trim(), value = $("memVal").value.trim();
  if (!value) return;
  await api("/api/memory", { key: key || "note", value });
  $("memKey").value = $("memVal").value = "";
  loadMemory();
};

// ---------------------------------------------------------------- audit
let auditRows = [];
async function loadAudit() {
  auditRows = await api("/api/audit?limit=400");
  renderAudit();
}
function renderAudit() {
  const f = $("auditFilter").value.toLowerCase();
  const rows = auditRows.filter((r) => !f || `${r.actor} ${r.action} ${r.target}`.toLowerCase().includes(f));
  $("auditTable").innerHTML = table([
    { h: "Time", f: (r) => fmtTime(r.ts) }, { h: "Actor", k: "actor" }, { h: "Action", k: "action" }, { h: "Target", f: (r) => esc((r.target || "").slice(0, 90)) },
    { h: "Details", f: (r) => `<span class="muted" style="font-size:12px">${esc(r.details ? JSON.stringify(r.details).slice(0, 220) : "")}</span>` },
  ], rows, "Audit log is empty.");
}
$("auditFilter").oninput = renderAudit;
$("doBackup").onclick = async () => { const r = await api("/api/backup", {}); $("backupNote").textContent = `Saved ${r.database}`; loadAudit(); };

// ---------------------------------------------------------------- settings
const SETTINGS = [
  ["internet.enabled", "Internet learning enabled", "bool"], ["scheduler.enabled", "Background scheduler enabled", "bool"],
  ["self_improvement.enabled", "Self-improvement enabled", "bool"], ["scheduler.interval_minutes", "Minutes between sessions", "int"],
  ["scheduler.max_topics_per_session", "Max topics per session", "int"], ["scheduler.max_fetches_per_session", "Max page fetches per session", "int"],
  ["scheduler.max_llm_calls_per_session", "Max LLM calls per session", "int"], ["scheduler.max_session_minutes", "Max session minutes", "int"],
  ["scheduler.defer_while_user_active_seconds", "Defer learning while user active (s)", "int"],
  ["ollama.chat_model", "Chat model", "str"], ["ollama.analysis_model", "Analysis model (blank = chat model)", "str"], ["ollama.embed_model", "Embedding model", "str"],
  ["voice.ollama_model", "Voice model (blank = chat model; Sinhala needs a multilingual model)", "str"],
  ["voice.store_transcripts", "Keep voice transcripts in history", "bool"], ["voice.tts.speed", "Default speech rate", "float"],
  ["voice.vad.endpoint_ms", "End-of-speech silence (ms)", "int"], ["voice.vad.start_ratio", "Speech start sensitivity (x noise floor)", "float"],
  ["voice.vad.min_level", "Minimum speech level (0-1)", "float"], ["voice.vad.max_utterance_s", "Max utterance (s)", "int"],
  ["voice.vad.barge_in_ratio", "Barge-in strictness while speaking", "float"], ["voice.vad.barge_in_ms", "Barge-in minimum speech (ms)", "int"],
];
let cfgCache = null;
const getPath = (o, p) => p.split(".").reduce((a, k) => (a == null ? a : a[k]), o);
async function loadSettings() {
  cfgCache = await api("/api/config");
  $("settingsForm").innerHTML = SETTINGS.map(([k, l, t]) => `<label for="s-${k}">${esc(l)}</label>` +
    (t === "bool" ? `<input type="checkbox" id="s-${k}" ${getPath(cfgCache, k) ? "checked" : ""}>` : `<input id="s-${k}" type="${t === "str" ? "text" : "number"}" ${t === "float" ? 'step="any"' : ""} value="${esc(getPath(cfgCache, k) ?? "")}">`)).join("");
}
$("saveSettings").onclick = async () => {
  const body = {};
  for (const [k, , t] of SETTINGS) {
    const el = document.getElementById(`s-${k}`);
    const v = t === "bool" ? el.checked : t === "int" ? parseInt(el.value, 10) : t === "float" ? parseFloat(el.value) : el.value.trim();
    if (v !== getPath(cfgCache, k)) body[k] = v;
  }
  try { const r = await api("/api/settings", body); $("settingsNote").textContent = `Saved ${Object.keys(r.changed).length} change(s).`; loadSettings(); }
  catch (e) { $("settingsNote").textContent = e.message; }
};

// ---------------------------------------------------------------- boot
showTab(currentTab);
refresh();
setInterval(() => { if (!document.hidden) refresh(); }, 5000);
