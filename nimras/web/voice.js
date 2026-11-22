"use strict";
// N.I.M.R.A.S. Voice Engine - voice-first UI.
// Microphone -> worklet (16 kHz) -> client VAD/endpointing -> WebSocket -> NVE
// NVE -> WAV chunks -> Web Audio playback (gain -> analyser -> speakers)
// Visuals are driven by the real mic / AI analysers, not by timers.
(() => {
  const V = {
    ws: null, connected: false, session: null, retry: 0,
    ctx: null, gain: null, aiAnalyser: null, micAnalyser: null, micStream: null, micSource: null, worklet: null,
    micOn: false, inUtt: false, above: 0, silenceMs: 0, uttMs: 0, speechMs: 0, floor: 0.006, preroll: [],
    lastVoiceAt: 0, vad: { start_ratio: 3, min_level: 0.012, start_frames: 4, endpoint_ms: 750, max_utterance_s: 30,
      preroll_ms: 300, min_speech_ms: 250, barge_in_ratio: 2.2, barge_in_ms: 280 },
    sources: new Set(), nextTime: 0, activeTurn: 0, cancelled: new Set(), decodeChain: Promise.resolve(),
    audioEndTurn: -1, state: "idle", serverStatus: null, aiLevel: 0, micLevel: 0, history: [],
    metrics: {}, ptt: false, lastFrameDraw: 0, firstAudioTurn: -1, replyText: "",
  };
  const $ = (id) => document.getElementById(id);
  const LANG = { en: "English", si: "Sinhala · සිංහල", mixed: "Sinhala-English" };

  // ------------------------------------------------------------------ state
  const LABEL = { idle: "IDLE", listening: "LISTENING", processing: "PROCESSING", speaking: "SPEAKING",
    interrupted: "INTERRUPTED", error: "ERROR", offline: "OFFLINE" };
  let interruptTimer = null;
  function setState(s, detail = "") {
    if (s === "interrupted") {
      clearTimeout(interruptTimer);
      interruptTimer = setTimeout(() => { if (V.state === "interrupted") setState(V.inUtt ? "listening" : "idle"); }, 900);
    }
    V.state = s;
    const stage = $("vStage");
    stage.dataset.state = s;
    $("vStateLabel").textContent = LABEL[s] || s.toUpperCase();
    $("vStateDetail").textContent = detail || ({ idle: V.micOn ? "Listening for your voice…" : "Press the microphone to start",
      listening: "Capturing your speech", processing: "Understanding and thinking", speaking: "Speaking - talk to interrupt",
      interrupted: "Stopped - listening to you" }[s] || "");
    $("vInterrupt").disabled = !(s === "speaking" || s === "processing");
    wakeDraw();
  }
  function showError(msg) {
    setState("error", msg);
    $("vError").textContent = msg;
    $("vError").hidden = false;
  }
  function clearError() { $("vError").hidden = true; }

  // ------------------------------------------------------------------ socket
  function connect() {
    const ws = new WebSocket(`ws://${location.host}/ws/voice`);
    ws.binaryType = "arraybuffer";
    V.ws = ws;
    $("vConn").textContent = "connecting…"; $("vConn").className = "pill warn";
    ws.onopen = () => { V.connected = true; V.retry = 0; $("vConn").textContent = "connected"; $("vConn").className = "pill ok"; if (V.state === "offline") setState("idle"); };
    ws.onclose = () => {
      V.connected = false;
      $("vConn").textContent = "disconnected"; $("vConn").className = "pill bad";
      if (V.inUtt) V.inUtt = false;
      setState("offline", "Voice engine connection lost - reconnecting");
      const delay = Math.min(15000, 1000 * 2 ** V.retry++);
      setTimeout(connect, delay);
    };
    ws.onmessage = (ev) => (typeof ev.data === "string" ? onEvent(JSON.parse(ev.data)) : onAudio(ev.data));
  }
  const send = (obj) => { if (V.connected) V.ws.send(JSON.stringify(obj)); };

  function onEvent(e) {
    switch (e.type) {
      case "hello":
        V.session = e.session; V.serverStatus = e.status;
        Object.assign(V.vad, e.vad || {});
        renderStatus(); fillVoices(); break;
      case "state":
        if (e.state === "listening" || (V.inUtt && e.state !== "error")) break;   // local VAD owns LISTENING
        if (e.state === "idle" && V.sources.size) break;                         // still playing queued audio
        if (e.state === "interrupted") { stopPlayback(); setState("interrupted"); break; }
        if (e.state === "error") { showError(e.detail || "error"); break; }
        if (e.turn) V.activeTurn = Math.max(V.activeTurn, e.turn);
        setState(e.state, e.detail);
        break;
      case "transcript": {
        V.activeTurn = Math.max(V.activeTurn, e.turn);
        clearError();
        const d = e.language || {};
        $("vHeardLang").textContent = LANG[d.spoken] || d.spoken || "?";
        $("vReplyLang").textContent = LANG[d.reply] || d.reply || "?";
        const probs = e.language_probs || {};
        $("vLangProb").textContent = Object.keys(probs).length ? Object.entries(probs).map(([k, p]) => `${k} ${Math.round(p * 100)}%`).join(" · ") : (e.typed ? "typed" : "");
        $("vYou").textContent = e.text;
        $("vFix").innerHTML = (e.changes || []).length ? "normalised: " + e.changes.map((c) => `<span class="mono">${esc(c.heard)}</span> → <span class="mono">${esc(c.normalized)}</span>`).join(", ") : "";
        V.replyText = ""; $("vReply").textContent = "";
        $("vSaveBox").hidden = true; $("vSaveBtn").disabled = !!e.typed;
        $("vSaveText").value = e.text; $("vSaveLang").value = d.spoken === "mixed" ? "mixed" : (d.spoken || "en");
        break;
      }
      case "no_speech": if (!V.sources.size) setState("idle", "Didn't catch that - no speech recognised"); break;
      case "reply_delta":
        if (e.turn < V.activeTurn || V.cancelled.has(e.turn)) break;
        V.replyText += e.text; $("vReply").textContent = V.replyText; break;
      case "reply_done":
        pushHistory($("vYou").textContent, e.text + (e.interrupted ? " …" : ""), e.language);
        break;
      case "audio_meta": break;
      case "audio_end": V.audioEndTurn = e.turn; if (!V.sources.size && V.state === "speaking") setState("idle"); break;
      case "tts_missing": $("vNote").textContent = `No local voice installed for: ${e.languages.join(", ")} - shown as text only.`; break;
      case "metrics": renderMetrics(e); break;
      case "interrupt_ack": break;
      case "error":
        if (e.where === "tts" && e.fallback_text) { $("vNote").textContent = `Speech synthesis failed (${e.message}); showing text.`; break; }
        showError({ stt: "Speech recognition failed: ", llm: "AI model (Ollama) failed: ", pipeline: "Voice pipeline error: " }[e.where] + e.message);
        break;
      case "settings": break;
    }
  }

  // ------------------------------------------------------------------ audio out
  function ensureAudio() {
    if (V.ctx) { if (V.ctx.state === "suspended") V.ctx.resume(); return V.ctx; }
    const ctx = new AudioContext({ latencyHint: "interactive" });
    V.ctx = ctx;
    V.gain = ctx.createGain();
    V.gain.gain.value = parseFloat($("vVolume").value);
    V.aiAnalyser = ctx.createAnalyser(); V.aiAnalyser.fftSize = 1024; V.aiAnalyser.smoothingTimeConstant = 0.6;
    V.gain.connect(V.aiAnalyser); V.aiAnalyser.connect(ctx.destination);
    return ctx;
  }
  function onAudio(buf) {
    const dv = new DataView(buf);
    const turn = dv.getUint32(0, true);
    if (turn < V.activeTurn || V.cancelled.has(turn)) return;           // stale / interrupted turn
    V.activeTurn = turn;
    const ctx = ensureAudio();
    const wav = buf.slice(8);
    V.decodeChain = V.decodeChain.then(() => ctx.decodeAudioData(wav)).then((audio) => {
      if (V.cancelled.has(turn)) return;
      const src = ctx.createBufferSource();
      src.buffer = audio; src.connect(V.gain);
      const start = Math.max(ctx.currentTime + 0.03, V.nextTime);
      src.start(start);
      V.nextTime = start + audio.duration;
      V.sources.add(src);
      if (V.firstAudioTurn !== turn) {
        V.firstAudioTurn = turn;
        const playAt = performance.now() + (start - ctx.currentTime) * 1000;
        if (V.metrics.speechEndAt) {
          const perceived = (playAt - V.metrics.speechEndAt) / 1000;
          $("vPerceived").textContent = `${perceived.toFixed(2)} s`;
          send({ type: "client_metrics", data: { perceived_first_audio_s: +perceived.toFixed(3), endpoint_ms: V.vad.endpoint_ms } });
          V.metrics.speechEndAt = 0;
        }
      }
      if (V.state !== "speaking" && !V.inUtt) setState("speaking");
      src.onended = () => {
        V.sources.delete(src);
        if (!V.sources.size && V.audioEndTurn >= turn && V.state === "speaking") setState("idle");
      };
    }).catch((err) => { $("vNote").textContent = `Playback failed: ${err.message}`; });
  }
  function stopPlayback() {
    const t0 = performance.now();
    V.cancelled.add(V.activeTurn);
    for (const s of V.sources) { try { s.stop(); } catch (_) {} }
    V.sources.clear();
    V.nextTime = 0;
    V.decodeChain = Promise.resolve();
    return performance.now() - t0;
  }
  function interrupt(reason) {
    const ms = stopPlayback();
    send({ type: "interrupt", reason });
    $("vInterruptMs").textContent = `${ms.toFixed(1)} ms`;
    setState("interrupted");
  }

  // ------------------------------------------------------------------ microphone
  async function micStart() {
    if (!navigator.mediaDevices || !window.isSecureContext) { showError("Microphone needs HTTPS or localhost."); return; }
    const ctx = ensureAudio();
    try {
      V.micStream = await navigator.mediaDevices.getUserMedia({ audio: {
        echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    } catch (err) {
      showError(err.name === "NotAllowedError" ? "Microphone permission denied. Allow it in the address bar to talk." :
        err.name === "NotFoundError" ? "No microphone found." : `Microphone error: ${err.message}`);
      return;
    }
    if (!V.worklet) {
      await ctx.audioWorklet.addModule("/voice-worklet.js");
    }
    V.micSource = ctx.createMediaStreamSource(V.micStream);
    V.micAnalyser = ctx.createAnalyser(); V.micAnalyser.fftSize = 1024; V.micAnalyser.smoothingTimeConstant = 0.5;
    V.worklet = new AudioWorkletNode(ctx, "nve-capture", { numberOfInputs: 1, numberOfOutputs: 0 });
    V.worklet.port.onmessage = (m) => onFrame(m.data.pcm, m.data.rms);
    V.micSource.connect(V.micAnalyser);
    V.micSource.connect(V.worklet);
    const track = V.micStream.getAudioTracks()[0];
    track.onended = () => { micStop(); showError("Microphone disconnected."); };
    V.micOn = true; V.floor = 0.006; V.above = 0;
    $("vMic").classList.add("on"); $("vMic").setAttribute("aria-pressed", "true");
    $("vMicLive").hidden = false;
    $("vMicLabel").textContent = "Microphone ON";
    clearError();
    setState("idle");
  }
  function micStop() {
    if (V.inUtt) { V.inUtt = false; send({ type: "abort_utterance" }); }
    if (V.micStream) V.micStream.getTracks().forEach((t) => t.stop());
    try { V.micSource && V.micSource.disconnect(); V.worklet && V.worklet.disconnect(); } catch (_) {}
    V.micStream = null; V.micSource = null; V.micAnalyser = null;
    if (V.worklet) { V.worklet.port.onmessage = null; }
    V.micOn = false;
    $("vMic").classList.remove("on"); $("vMic").setAttribute("aria-pressed", "false");
    $("vMicLive").hidden = true;
    $("vMicLabel").textContent = "Microphone OFF";
    if (V.state === "listening") setState("idle");
  }

  // ------------------------------------------------------------------ VAD / endpointing
  function onFrame(pcmBuf, level) {
    if (!V.micOn) return;
    V.micLevel = level;
    const vad = V.vad;
    const playing = V.sources.size > 0;
    if (playing && V.aiAnalyser) { V.aiAnalyser.getByteTimeDomainData(tdAiVad); V.aiLevel = levelOf(tdAiVad); }
    let thr = Math.max(vad.min_level, V.floor * vad.start_ratio);
    if (!V.inUtt) {
      if (playing) thr = Math.max(thr * vad.barge_in_ratio, V.aiLevel * 0.9);   // echo guard
      const need = playing ? Math.ceil(vad.barge_in_ms / 20) : vad.start_frames;
      V.preroll.push(pcmBuf);
      while (V.preroll.length > vad.preroll_ms / 20) V.preroll.shift();
      if (level > thr || V.ptt) V.above++;
      else { V.above = 0; if (!playing) V.floor = V.floor * 0.97 + Math.min(level, 0.05) * 0.03; }
      if (V.ptt || V.above >= need) startUtterance(playing);
      return;
    }
    if (V.connected) V.ws.send(pcmBuf);
    V.uttMs += 20;
    if (level > thr * 0.6 || V.ptt) { V.silenceMs = 0; V.speechMs += 20; V.lastVoiceAt = performance.now(); }
    else V.silenceMs += 20;
    if ((!V.ptt && V.silenceMs >= vad.endpoint_ms) || V.uttMs >= vad.max_utterance_s * 1000) endUtterance();
  }
  function startUtterance(bargeIn) {
    if (!V.connected) return;
    if (bargeIn || V.state === "processing" || V.state === "speaking") interrupt(bargeIn ? "barge-in" : "new speech");
    V.inUtt = true; V.silenceMs = 0; V.uttMs = 0; V.speechMs = 0; V.above = 0;
    send({ type: "start_utterance" });
    for (const f of V.preroll) V.ws.send(f);
    V.uttMs = V.preroll.length * 20;
    V.preroll = [];
    setState("listening");
  }
  function endUtterance() {
    V.inUtt = false;
    if (V.speechMs < V.vad.min_speech_ms) { send({ type: "abort_utterance" }); setState("idle", "Too short - ignored"); return; }
    V.metrics.speechEndAt = V.lastVoiceAt || performance.now();
    send({ type: "end_utterance", client: { speech_ms: V.speechMs, utterance_ms: V.uttMs, endpoint_ms: V.vad.endpoint_ms } });
    setState("processing", "Recognising speech");
  }

  // ------------------------------------------------------------------ visuals
  const tdMic = new Uint8Array(1024), tdAi = new Uint8Array(1024), tdAiVad = new Uint8Array(1024), fq = new Uint8Array(512);
  let rafId = 0, frameCount = 0;
  function wakeDraw() { if (!rafId) rafId = requestAnimationFrame(draw); }
  function levelOf(arr) { let s = 0; for (let i = 0; i < arr.length; i++) { const v = (arr[i] - 128) / 128; s += v * v; } return Math.sqrt(s / arr.length); }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function wave(canvas, an, buf, color, active) {
    const c = canvas.getContext("2d"), w = canvas.width = canvas.clientWidth * devicePixelRatio, h = canvas.height = canvas.clientHeight * devicePixelRatio;
    c.clearRect(0, 0, w, h);
    c.lineWidth = 2 * devicePixelRatio; c.strokeStyle = color; c.globalAlpha = active ? 1 : 0.35;
    c.beginPath();
    if (an && active) {
      an.getByteTimeDomainData(buf);
      for (let i = 0; i < buf.length; i++) { const x = (i / (buf.length - 1)) * w, y = (buf[i] / 255) * h; i ? c.lineTo(x, y) : c.moveTo(x, y); }
    } else { c.moveTo(0, h / 2); c.lineTo(w, h / 2); }
    c.stroke();
  }
  function draw(ts) {
    rafId = 0;
    if (document.hidden || currentTabIsNotVoice()) return;           // no work when not visible
    const playing = V.sources.size > 0, micActive = V.micOn && !!V.micAnalyser;
    const busy = playing || V.state === "processing" || V.inUtt || V.state === "interrupted";
    frameCount++;
    // idle: ~12 fps; active: full rate
    if (!busy && !micActive && frameCount % 5) { rafId = requestAnimationFrame(draw); return; }
    wave($("vMicWave"), V.micAnalyser, tdMic, css("--accent"), micActive);
    wave($("vAiWave"), V.aiAnalyser, tdAi, css("--accent2"), playing);
    V.aiLevel = playing && V.aiAnalyser ? levelOf(tdAi) : 0;
    // Orb: radius and radial spectrum follow the audio that is actually live.
    const cv = $("vOrb"), c = cv.getContext("2d");
    const W = cv.width = cv.clientWidth * devicePixelRatio, H = cv.height = cv.clientHeight * devicePixelRatio;
    const cx = W / 2, cy = H / 2, base = Math.min(W, H) * 0.24;
    const src = playing ? V.aiAnalyser : (V.inUtt || micActive) ? V.micAnalyser : null;
    const lvl = playing ? V.aiLevel : (micActive ? V.micLevel : 0);
    const col = { listening: css("--accent"), speaking: css("--accent2"), processing: css("--violet"),
      interrupted: css("--warn"), error: css("--bad"), offline: css("--muted") }[V.state] || css("--accent");
    c.clearRect(0, 0, W, H);
    const r = base * (1 + Math.min(0.45, lvl * 3.2));
    const g = c.createRadialGradient(cx, cy, r * 0.1, cx, cy, r * 1.6);
    g.addColorStop(0, col + "cc"); g.addColorStop(0.55, col + "33"); g.addColorStop(1, "transparent");
    c.fillStyle = g; c.beginPath(); c.arc(cx, cy, r * 1.6, 0, Math.PI * 2); c.fill();
    c.fillStyle = col + "55"; c.strokeStyle = col; c.lineWidth = 2 * devicePixelRatio;
    c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2); c.fill(); c.stroke();
    if (src) {
      src.getByteFrequencyData(fq);
      const bars = 96;
      c.strokeStyle = col; c.lineWidth = 3 * devicePixelRatio; c.lineCap = "round";
      for (let i = 0; i < bars; i++) {
        const v = fq[Math.floor((i / bars) * 160) + 2] / 255;
        const a = (i / bars) * Math.PI * 2 - Math.PI / 2, r1 = r * 1.08, r2 = r1 + v * base * 0.7;
        c.beginPath(); c.moveTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1); c.lineTo(cx + Math.cos(a) * r2, cy + Math.sin(a) * r2); c.stroke();
      }
    }
    if (V.state === "processing") {            // no audio while thinking: show a progress arc
      const t = (ts || 0) / 600;
      c.strokeStyle = col; c.lineWidth = 4 * devicePixelRatio;
      c.beginPath(); c.arc(cx, cy, r * 1.15, t, t + 1.3); c.stroke();
      c.beginPath(); c.arc(cx, cy, r * 1.15, t + Math.PI, t + Math.PI + 1.3); c.stroke();
    }
    $("vLevelMic").style.width = `${Math.min(100, V.micLevel * 600)}%`;
    $("vLevelAi").style.width = `${Math.min(100, V.aiLevel * 600)}%`;
    rafId = requestAnimationFrame(draw);
  }
  function currentTabIsNotVoice() { const t = document.getElementById("tab-voice"); return !t || !t.classList.contains("on"); }
  document.addEventListener("visibilitychange", wakeDraw);
  document.getElementById("tabs").addEventListener("click", () => setTimeout(wakeDraw, 0));

  // ------------------------------------------------------------------ status / settings
  function renderStatus() {
    const s = V.serverStatus; if (!s) return;
    const tts = s.tts || {};
    const vName = (lang) => { const v = (tts[lang] || []).find((x) => x.selected); return v ? v.name : "none"; };
    $("vEngine").innerHTML = [
      `<span class="pill ${s.stt.available ? "ok" : "bad"}">STT ${s.stt.available ? "ready" : "missing"}</span>`,
      `<span class="pill ${vName("en") !== "none" ? "ok" : "bad"}">EN voice: ${esc(vName("en"))}</span>`,
      `<span class="pill ${vName("si") !== "none" ? "ok" : "bad"}">SI voice: ${esc(vName("si"))}</span>`,
      `<span class="pill ${s.llm.ready ? "ok" : "bad"}">LLM ${esc(s.llm.model)}${s.llm.ready ? "" : " (not pulled)"}</span>`,
      `<span class="pill">local processing · audio not stored</span>`].join(" ");
    if (!s.stt.available) showError(s.stt.detail);
  }
  function fillVoices() {
    const tts = (V.serverStatus || {}).tts || {};
    for (const lang of ["en", "si"]) {
      const sel = $(`vVoice_${lang}`);
      sel.innerHTML = (tts[lang] || []).map((v) => `<option value="${esc(v.name)}" ${v.selected ? "selected" : ""} ${v.available ? "" : "disabled"}>${esc(v.name)}${v.available ? "" : " (missing)"}</option>`).join("");
    }
  }
  function renderMetrics(m) {
    const f = (x) => (x == null ? "–" : `${(+x).toFixed(2)} s`);
    $("vMetrics").innerHTML = `STT ${f(m.stt_latency)} · first token ${f(m.llm_first_token)} · first audio ${f(m.first_audio)} · total ${f(m.total)}` +
      (m.interrupted ? ` · <b>interrupted</b> (pipeline cancel ${f(m.interrupt_latency)})` : "") + (m.stt_model ? ` · <span class="muted">${esc(m.stt_model)}</span>` : "");
  }
  function pushHistory(you, ai, lang) {
    V.history.unshift({ you, ai, lang });
    V.history = V.history.slice(0, 4);
    $("vHistory").innerHTML = V.history.slice(1).map((h) => `<div class="vh"><div class="muted">You: ${esc(h.you)}</div><div>${esc(h.ai)}</div></div>`).join("");
  }
  function sendSettings() {
    send({ type: "settings", speed: parseFloat($("vSpeed").value), language: $("vLangPref").value,
      voices: { en: $("vVoice_en").value, si: $("vVoice_si").value } });
  }

  // ------------------------------------------------------------------ controls
  $("vMic").onclick = () => (V.micOn ? micStop() : micStart());
  $("vInterrupt").onclick = () => interrupt("stop button");
  $("vVolume").oninput = () => { if (V.gain) V.gain.gain.value = parseFloat($("vVolume").value); $("vVolumeVal").textContent = `${Math.round($("vVolume").value * 100)}%`; };
  $("vSpeed").oninput = () => { $("vSpeedVal").textContent = `${(+$("vSpeed").value).toFixed(2)}x`; };
  $("vSpeed").onchange = sendSettings;
  $("vLangPref").onchange = sendSettings;
  $("vVoice_en").onchange = sendSettings; $("vVoice_si").onchange = sendSettings;
  $("vEndpoint").oninput = () => { V.vad.endpoint_ms = parseInt($("vEndpoint").value, 10); $("vEndpointVal").textContent = `${V.vad.endpoint_ms} ms`; };
  $("vEndpoint").onchange = () => api("/api/settings", { "voice.vad.endpoint_ms": V.vad.endpoint_ms }).catch(() => {});
  $("vTypeToggle").onclick = () => { $("vTypeForm").hidden = !$("vTypeForm").hidden; if (!$("vTypeForm").hidden) $("vTypeInput").focus(); };
  $("vTypeForm").onsubmit = (e) => {
    e.preventDefault();
    const t = $("vTypeInput").value.trim(); if (!t) return;
    ensureAudio(); stopPlayback();
    send({ type: "text", text: t }); $("vTypeInput").value = ""; setState("processing");
  };
  $("vSaveBtn").onclick = () => { $("vSaveBox").hidden = !$("vSaveBox").hidden; };
  $("vSaveGo").onclick = async () => {
    if (!$("vSaveConsent").checked) { $("vSaveNote").textContent = "Tick the consent box first."; return; }
    try {
      const r = await api("/api/voice/dataset", { session: V.session, transcript: $("vSaveText").value, language: $("vSaveLang").value,
        category: $("vSaveCat").value, consent: true });
      $("vSaveNote").textContent = `Saved ${r.file} to the local test set.`; $("vSaveConsent").checked = false;
    } catch (err) { $("vSaveNote").textContent = err.message; }
  };
  // Push-to-talk: hold Space (when not typing) while the microphone is on.
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.repeat || !V.micOn || currentTabIsNotVoice() || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    e.preventDefault(); V.ptt = true;
  });
  document.addEventListener("keyup", (e) => { if (e.code === "Space" && V.ptt) { V.ptt = false; if (V.inUtt) endUtterance(); } });
  if (navigator.mediaDevices) navigator.mediaDevices.addEventListener("devicechange", async () => {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const outs = devs.filter((d) => d.kind === "audiooutput").length, ins = devs.filter((d) => d.kind === "audioinput").length;
    $("vNote").textContent = `Audio devices changed (${ins} input, ${outs} output).` + (outs === 0 ? " No speaker available - replies are shown as text." : "");
    if (V.micOn && V.micStream && V.micStream.getAudioTracks()[0].readyState === "ended") { micStop(); showError("Microphone disconnected."); }
  });

  // ================================================================== Voice Lab tab
  window.loadVoiceLab = async function loadVoiceLab() {
    const [st, hw, perf, tests, models] = await Promise.all([api("/api/voice/status"), api("/api/voice/hardware"),
      api("/api/voice/performance"), api("/api/voice/tests"), api("/api/voice/models")]);
    const r = hw.recommendation || {};
    $("labHardware").innerHTML = `<div class="kv"><div>CPU</div><div>${esc(hw.cpu)} (${esc(hw.cpu_threads)} threads)</div>
      <div>RAM</div><div>${esc(hw.ram_gb)} GB (${esc(hw.ram_free_gb ?? "?")} GB free)</div><div>GPU</div><div>${esc((hw.gpus || []).join(", ") || "none")} ${hw.nvidia ? "(NVIDIA/CUDA)" : "(no CUDA - CPU inference)"}</div>
      <div>Audio devices</div><div>${esc((hw.audio_devices || []).join("; "))}</div><div>Disk free</div><div>${esc(hw.disk_free_gb)} GB</div>
      <div>OS</div><div>${esc(hw.os)}</div></div><p class="muted">${esc(r.why || "")}</p>
      <div class="kv"><div>STT</div><div>${esc(r.stt)}</div><div>TTS</div><div>${esc(r.tts)}</div><div>LLM for Sinhala</div><div>${esc(r.llm)}</div></div>`;
    const stt = st.stt || {};
    $("labEngine").innerHTML = `<div class="kv"><div>STT</div><div>${badge(stt.available ? "ok" : "failed")} ${esc(stt.detail)}</div>
      <div>Language models</div><div>${Object.entries(stt.language_models || {}).map(([l, m]) => `${esc(l)}: <code>${esc(m.file)}</code> ${m.ready ? "✓" : "missing"}`).join("<br>")}</div>
      <div>TTS</div><div>${Object.entries(st.tts || {}).map(([l, vs]) => `${esc(l)}: ${vs.map((v) => `${esc(v.name)} ${v.available ? "✓" : "✗"}`).join(", ")}`).join("<br>")}</div>
      <div>LLM</div><div><code>${esc(st.llm.model)}</code> ${st.llm.ready ? "✓" : "(not pulled)"}</div></div>`;
    $("labPerf").innerHTML = table([{ h: "Reply lang", k: "reply_lang" }, { h: "Turns", k: "n", num: 1 }, { h: "STT s", k: "stt", num: 1 },
      { h: "1st token s", k: "llm_first", num: 1 }, { h: "1st audio s", k: "first_audio", num: 1 }, { h: "Total s", k: "total", num: 1 },
      { h: "Interrupts", k: "interrupts", num: 1 }, { h: "Cancel s", k: "interrupt_latency", num: 1 }, { h: "Errors", k: "errors", num: 1 }],
    perf.by_language || [], "No voice turns yet.");
    renderTests(tests);
    $("labModels").innerHTML = table([{ h: "#", k: "id" }, { h: "Kind", k: "kind" }, { h: "Lang", k: "language" }, { h: "Name", k: "name" },
      { h: "Status", f: (m) => badge(m.status) }, { h: "Benchmark", f: (m) => m.metrics && m.metrics.candidate ? `WER ${m.metrics.candidate.wer} vs ${m.metrics.production ? m.metrics.production.wer : "–"} ${m.metrics.passed ? "✓" : "✗"}` : "" },
      { h: "", f: (m) => m.status === "candidate" ? `<button class="small" data-veval="${m.id}">benchmark</button> <button class="small" data-vpromote="${m.id}">promote…</button>` : (m.status === "recommended" ? `<a href="${safeUrl(m.source)}" target="_blank" rel="noopener noreferrer">release notes</a>` : "") }],
    models.models || [], "No models registered.");
  };
  function renderTests(t) {
    const s = t.summary;
    $("labTestState").textContent = t.running ? "running…" : (t.run_id ? `last run ${t.run_id}` : "never run");
    $("labTestLog").textContent = (t.log || []).join("\n");
    $("labTestSummary").innerHTML = s ? `<div class="kpis">${Object.entries(s.by_category).map(([k, v]) => `<div class="kpi"><div class="v">${v.passed}/${v.n - v.skipped}</div><div class="l">${esc(k)}${v.mean_wer != null ? ` · WER ${v.mean_wer}` : ""}${v.mean_cer != null ? ` · CER ${v.mean_cer}` : ""}${v.mean_latency != null ? ` · ${v.mean_latency}s` : ""}</div></div>`).join("")}</div>` : "";
    $("labTests").innerHTML = table([{ h: "Category", k: "category" }, { h: "Test", k: "name" }, { h: "Lang", k: "language" },
      { h: "Result", f: (r) => badge(r.passed == null ? "skipped" : r.passed ? "passed" : "failed") }, { h: "WER", k: "wer", num: 1 }, { h: "CER", k: "cer", num: 1 },
      { h: "Latency s", k: "latency", num: 1 }, { h: "Details", f: (r) => `<span class="muted small">${esc(r.details && (r.details.hypothesis ? `heard: ${r.details.hypothesis}` : r.details.note || r.details.error || r.details.reply || ""))}</span>` }],
    t.results || [], "No test results yet.");
    if (t.running) setTimeout(async () => renderTests(await api("/api/voice/tests")), 3000);
  }
  $("labRunTests").onclick = async () => { try { await api("/api/voice/selftest", {}); renderTests(await api("/api/voice/tests")); } catch (e) { $("labTestState").textContent = e.message; } };
  $("labCheckUpdates").onclick = async () => { const r = await api("/api/voice/updates/check", {}); $("labUpdNote").textContent = `${r.new_releases.length} new release(s) recorded as recommendations.`; window.loadVoiceLab(); };
  $("labDeleteHistory").onclick = async () => { if (!confirm("Delete all voice transcripts and voice metrics?")) return; const r = await api("/api/voice/history/delete", { scope: "voice" }); $("labPrivNote").textContent = `Deleted ${r.deleted_messages} transcript message(s).`; };
  $("labDeleteAll").onclick = async () => { if (!confirm("Delete ALL conversation history (text and voice)?")) return; const r = await api("/api/voice/history/delete", { scope: "all" }); $("labPrivNote").textContent = `Deleted ${r.deleted_messages} message(s).`; };
  $("labDeleteSet").onclick = async () => { if (!confirm("Delete every recording in the local voice test set?")) return; const r = await api("/api/voice/dataset/delete", {}); $("labPrivNote").textContent = `Deleted ${r.deleted_files} file(s).`; };
  $("labModels").addEventListener("click", async (e) => {
    const ev = e.target.dataset.veval, pr = e.target.dataset.vpromote;
    try {
      if (ev) { e.target.textContent = "benchmarking…"; const r = await api(`/api/voice/models/${ev}/evaluate`, {}); alert(`Benchmark ${r.passed ? "PASSED" : "FAILED"}: candidate WER ${r.candidate.wer}, production WER ${r.production ? r.production.wer : "n/a"}`); }
      if (pr && confirm("Promote this speech model to production? The current model is kept for rollback.")) await api(`/api/voice/models/${pr}/promote`, { authorise: true });
    } catch (err) { alert(err.message); }
    window.loadVoiceLab();
  });

  // ------------------------------------------------------------------ boot
  $("vVolumeVal").textContent = "100%"; $("vSpeedVal").textContent = "1.00x"; $("vEndpointVal").textContent = `${V.vad.endpoint_ms} ms`;
  setState("offline", "Connecting to the voice engine…");
  connect();
  // An AudioContext may only start after a user gesture.
  document.addEventListener("pointerdown", () => ensureAudio(), { once: true });
})();
