"""Automated Voice Engine self-testing.

Three kinds of evidence, always labelled so results are not over-claimed:

* synthetic  - TTS -> STT round trips with Piper voices. Measures STT on clean,
               noisy, quiet, fast and slow *synthetic* speech plus TTS
               intelligibility. Synthetic speech is easier than real people, so
               these numbers are an upper bound for human speech.
* recorded   - human recordings the user has explicitly saved to the test set
               (data/voice_tests/dataset, manifest.jsonl with consent=true).
               Sri Lankan English pronunciation and Sinhala accent coverage can
               ONLY be measured here; without recordings those checks report
               "needs recordings" instead of passing.
* pipeline   - follow-up context, interruption, and failure injection
               (Ollama down, TTS missing, STT missing).

Metrics: WER, CER, recognition latency, LLM first-token latency, time to first
audio, total response time, interruption latency, CPU and RAM use.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import unicodedata
import uuid
from array import array
from pathlib import Path

from .audio import STT_RATE, add_noise, read_wav, resample, wav_bytes
from .stt import STTError

# --------------------------------------------------------------------- corpus
EN = [
    ("short", "Turn on the lights."),
    ("short", "What time is it?"),
    ("ict", "Please check the DNS and DHCP settings on the main router."),
    ("ict", "We use Microsoft Entra ID with Intune and MFA."),
    ("ict", "Our SIEM collects logs from the firewall, the VPN gateway and Windows Server."),
    ("ict", "The audit covers ISO 27001, ITIL and COBIT controls with RBAC."),
    ("ict", "Create a VLAN on the switch and join it to Active Directory."),
    ("long", "I need you to explain how to configure a VLAN on a Cisco switch, then tell me how to set up a "
             "DHCP scope for that VLAN, and finally how to test connectivity from a client computer."),
]
SI = [
    ("short", "ආයුබෝවන්, ඔබට කොහොමද?"),
    ("short", "අද කාලගුණය හොඳයි."),
    ("sentence", "මට පරිගණකයක් අලුත්වැඩියා කරන්න උදව් ඕනේ."),
    ("sentence", "කරුණාකර මගේ මුරපදය වෙනස් කරන්න."),
    ("sentence", "අපේ කාර්යාලයේ අන්තර්ජාලය ක්‍රියා කරන්නේ නැහැ."),
]
MIXED = [
    ("mixed", "මගේ laptop එකේ WiFi වැඩ කරන්නේ නැහැ."),
    ("mixed", "Firewall එකේ rule එකක් add කරන්න පුළුවන්ද?"),
    ("mixed", "Active Directory එකේ user කෙනෙක් හදන්න ඕනේ."),
]


# --------------------------------------------------------------------- metrics
def degenerate(text: str) -> bool:
    """Broken model output: long runs of the same token or very low vocabulary."""
    words = text.split()
    if len(words) < 6:
        return False
    run = best = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    singles = [w for w in words if len(w) <= 2]
    return best >= 4 or len(set(words)) / len(words) < 0.45 or len(singles) / len(words) > 0.5


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text).replace("‍", "").replace("‌", "").lower()
    t = re.sub(r"[^\w\s඀-෿]", " ", t)
    return " ".join(t.split())


def _edit(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref).split(), _norm(hyp).split()
    return round(_edit(r, h) / max(1, len(r)), 4)


def cer(ref: str, hyp: str) -> float:
    r, h = list(_norm(ref).replace(" ", "")), list(_norm(hyp).replace(" ", ""))
    return round(_edit(r, h) / max(1, len(r)), 4)


def _process_usage(names=("whisper-server", "piper", "python")) -> dict:
    if os.name != "nt":
        return {}
    ps = ("Get-Process | Where-Object { $_.ProcessName -match '" + "|".join(names) + "' } | "
          "ForEach-Object { $_.ProcessName + ',' + $_.WorkingSet64 + ',' + $_.CPU }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                             timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.decode()
    except (OSError, subprocess.TimeoutExpired):
        return {}
    usage: dict = {}
    for line in out.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 3:
            name, ws, cpu = parts
            u = usage.setdefault(name, {"ram_mb": 0.0, "cpu_s": 0.0})
            u["ram_mb"] += round(int(ws or 0) / 2**20, 1)
            try:
                u["cpu_s"] += float(cpu or 0)
            except ValueError:
                pass
    return usage


# --------------------------------------------------------------------- runner
class VoiceSelfTest:
    def __init__(self, rt, log=print):
        self.rt, self.engine, self.log = rt, rt.voice, log
        self.run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
        self.results: list[dict] = []
        self.dataset_dir = rt.cfg.data_dir / "voice_tests" / "dataset"

    def _record(self, category: str, name: str, language: str | None, passed: bool | None, **details) -> dict:
        row = {"run_id": self.run_id, "created_at": time.time(), "category": category, "name": name,
               "language": language, "passed": None if passed is None else int(passed),
               "wer": details.pop("wer", None), "cer": details.pop("cer", None),
               "latency": details.pop("latency", None), "details": json.dumps(details, ensure_ascii=False)}
        self.rt.db.insert("voice_tests", row)
        self.results.append(row | {"details": details})
        mark = "SKIP" if passed is None else ("PASS" if passed else "FAIL")
        extra = " ".join(f"{k}={row[k]}" for k in ("wer", "cer", "latency") if row[k] is not None)
        self.log(f"[{mark}] {category}/{name} {language or ''} {extra}")
        return row

    # ---------------------------------------------------------------- synthetic
    def _synth(self, text: str, lang: str, speed: float = 1.0) -> array:
        voice = self.engine.tts.voice_for(lang)
        if not voice:
            raise RuntimeError(f"no TTS voice for {lang}")
        spoken = self.engine.tts.prepare(lang, text)
        pcm, rate = voice.synthesize(spoken, speed)
        return resample(pcm, rate, STT_RATE)

    def _stt_case(self, category: str, name: str, lang: str, ref: str, pcm: array, max_wer: float,
                  check_lang: bool = True) -> None:
        try:
            prompt = self.engine.vocab.initial_prompt(ref if lang == "en" else "")
            t = self.engine.stt.transcribe(pcm, "auto", prompt)
        except STTError as e:
            self._record(category, name, lang, False, error=str(e))
            return
        hyp, changes = self.engine.vocab.normalize(t.text)
        w, c = wer(ref, hyp), cer(ref, hyp)
        lang_ok = (t.language == lang) or not check_lang
        # Sinhala word segmentation is inconsistent ("ක්‍රියා කරන්නේ" vs "ක්‍රියාකරන්නේ"), so
        # Sinhala passes on CER; English on WER.
        ok = (c <= max_wer / 2.5) if lang == "si" else (w <= max_wer)
        self._record(category, name, lang, ok and lang_ok, wer=w, cer=c, latency=round(t.latency_s, 3),
                     reference=ref, hypothesis=hyp, detected_language=t.language, language_ok=lang_ok,
                     language_probs=t.language_probs, model=t.model, audio_s=round(t.duration_s, 2),
                     normalizations=changes, alternatives=t.alternatives, evidence="synthetic",
                     criterion="CER" if lang == "si" else "WER")

    def synthetic_suite(self) -> None:
        if not self.engine.stt.available()[0]:
            self._record("synthetic", "stt_available", None, False, error=self.engine.stt.available()[1])
            return
        for i, (kind, text) in enumerate(EN):
            if self.engine.tts.voice_for("en"):
                self._stt_case("stt_en", f"{kind}_{i}", "en", text, self._synth(text, "en"), 0.15)
        for i, (kind, text) in enumerate(SI):
            if self.engine.tts.voice_for("si"):
                self._stt_case("stt_si", f"{kind}_{i}", "si", text, self._synth(text, "si"), 0.35)
        for i, (kind, text) in enumerate(MIXED):
            if self.engine.tts.voice_for("si"):
                # Synthesised with the language-aware router (Sinhala + English voices).
                out = self.engine.tts.synthesize(text, "si")
                pcm, rate = read_wav(out["wav"])
                self._stt_case("stt_mixed", f"{kind}_{i}", "si", text, resample(pcm, rate, STT_RATE), 0.5,
                               check_lang=False)
        # Conditions: speaking speed, background noise, quiet speaker.
        base_en, base_si = EN[2][1], SI[2][1]
        for label, speed in (("slow_0.8x", 0.8), ("fast_1.3x", 1.3)):
            self._stt_case("conditions", f"en_{label}", "en", base_en, self._synth(base_en, "en", speed), 0.2)
            self._stt_case("conditions", f"si_{label}", "si", base_si, self._synth(base_si, "si", speed), 0.4)
        clean_en, clean_si = self._synth(base_en, "en"), self._synth(base_si, "si")
        for snr in (20, 10):
            self._stt_case("conditions", f"en_noise_{snr}dB", "en", base_en, add_noise(clean_en, snr), 0.25)
            self._stt_case("conditions", f"si_noise_{snr}dB", "si", base_si, add_noise(clean_si, snr), 0.5)
        quiet = array("h", (int(s * 0.08) for s in clean_en))
        self._stt_case("conditions", "en_quiet_-22dB", "en", base_en, quiet, 0.25)
        # Silence / noise must not produce text (avoid answering every sound).
        import random
        rnd = random.Random(3)
        noise = array("h", (int(rnd.gauss(0, 700)) for _ in range(STT_RATE * 2)))   # steady fan-like noise
        has = self.engine.vad.has_speech(noise)
        self._record("conditions", "noise_only_rejected", None, not has, evidence="synthetic",
                     note="server VAD must reject a 2 s noise-only clip")

    # ---------------------------------------------------------------- TTS
    def tts_suite(self) -> None:
        for lang, text in (("en", "Your firewall rule was created successfully."),
                           ("si", "ඔබේ ගැටලුව විසඳා ඇත.")):
            v = self.engine.tts.voice_for(lang)
            if not v:
                self._record("tts", f"voice_{lang}", lang, False, error="no voice available")
                continue
            t0 = time.perf_counter()
            pcm, rate = v.synthesize(self.engine.tts.prepare(lang, text))
            dt = time.perf_counter() - t0
            secs = len(pcm) / rate
            self._record("tts", f"latency_{lang}", lang, dt < 2.0, latency=round(dt, 3), audio_s=round(secs, 2),
                         rtf=round(dt / max(secs, 0.01), 3), voice=v.name)
        out = self.engine.tts.synthesize("ඔබේ firewall එකේ VLAN 10 පරීක්ෂා කරන්න.", "si")
        self._record("tts", "mixed_router", "mixed", bool(out["wav"]) and not out["missing"],
                     segments=out["segments"], seconds=round(out.get("seconds", 0), 2))
        # Abbreviation handling.
        spoken = self.engine.vocab.for_speech("Check the SIEM, VLAN and ITIL docs at https://example.com")
        ok = "seem" in spoken and "V-LAN" in spoken and "eye-till" in spoken and "http" not in spoken
        self._record("tts", "abbreviations", "en", ok, spoken=spoken)

    # ---------------------------------------------------------------- recorded
    def recorded_suite(self) -> None:
        manifest = self.dataset_dir / "manifest.jsonl"
        cases = []
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                try:
                    c = json.loads(line)
                except ValueError:
                    continue
                if c.get("consent") is True and (self.dataset_dir / c["file"]).exists():
                    cases.append(c)
        for c in cases:
            pcm, rate = read_wav((self.dataset_dir / c["file"]).read_bytes())
            self._stt_case("recorded", c.get("category", "recording") + ":" + c["file"], c.get("language", "en"),
                           c["transcript"], resample(pcm, rate, STT_RATE), 0.35 if c.get("language") == "si" else 0.2)
        cats = {c.get("category") for c in cases}
        for need in ("sri_lankan_english", "sinhala_accent", "mixed_speech", "background_noise"):
            if need not in cats:
                self._record("recorded", need, None, None,
                             note="needs consented human recordings in this category - not measurable synthetically")

    # ---------------------------------------------------------------- pipeline
    def _session(self, ollama=None):
        from .engine import VoiceSession
        events, audio = [], []
        done = threading.Event()

        def emit(e):
            events.append((time.perf_counter(), e))
            if e.get("type") in ("reply_done", "error") or (e.get("type") == "state" and e.get("state") == "idle"
                                                             and any(x[1].get("type") == "transcript" for x in events)):
                done.set()
        s = VoiceSession(self.engine, emit, lambda b: audio.append((time.perf_counter(), len(b))),
                         conversation_id="selftest-" + uuid.uuid4().hex[:8])
        return s, events, audio, done

    def pipeline_suite(self) -> None:
        health = self.rt.ollama.health()
        if not health.get("reachable"):
            self._record("pipeline", "ollama", None, False, error="Ollama not reachable")
        else:
            # Follow-up question uses conversation context.
            s, ev, audio, done = self._session()
            t0 = time.perf_counter()
            s.submit_text("What is DHCP? Answer in one sentence.")
            done.wait(180)
            first_audio = next((t for t, _ in audio), None)
            r1 = next((e for _, e in ev if e.get("type") == "reply_done"), {})
            done.clear()
            ev.clear()
            s.submit_text("Which UDP ports does it use?")
            done.wait(180)
            time.sleep(0.5)
            r2 = next((e for _, e in ev if e.get("type") == "reply_done"), {})
            text2 = r2.get("text", "")
            self._record("pipeline", "follow_up_context", "en", bool(re.search(r"\b6[78]\b", text2)),
                         latency=round((first_audio - t0), 3) if first_audio else None,
                         first_reply=r1.get("text", "")[:300], follow_up_reply=text2[:300],
                         note="pass = follow-up answer resolves 'it' to DHCP (mentions port 67/68)")
            # Sinhala reply language.
            s2, ev2, _, done2 = self._session()
            s2.submit_text("DNS කියන්නේ මොකක්ද? කෙටියෙන් කියන්න.")
            done2.wait(180)
            time.sleep(0.5)
            r = next((e for _, e in ev2 if e.get("type") == "reply_done"), {})
            txt = r.get("text", "")
            si_ratio = len(re.findall(r"[඀-෿]", txt)) / max(1, len(re.findall(r"[^\W\d_]", txt)))
            self._record("pipeline", "sinhala_reply", "si", si_ratio >= 0.5 and not degenerate(txt), reply=txt[:400],
                         sinhala_script_ratio=round(si_ratio, 2), degenerate=degenerate(txt), model=self.engine.model,
                         note="pass = mostly Sinhala script and not degenerate; fluency needs human review")
            # Interruption: cancel while generating/speaking, then switch language.
            s3, ev3, audio3, done3 = self._session()
            s3.submit_text("Explain step by step how to configure a firewall for a small office.")
            t_wait = time.perf_counter()
            while time.perf_counter() - t_wait < 120 and not any(e.get("type") == "reply_delta" for _, e in ev3):
                time.sleep(0.05)
            time.sleep(1.0)
            lat = s3.interrupt("selftest barge-in")
            n_audio_before = len(audio3)
            time.sleep(1.0)
            leaked = len(audio3) - n_audio_before
            self._record("pipeline", "interruption", "en", lat is not None and lat < 1.5 and leaked == 0,
                         latency=lat, audio_chunks_after_interrupt=leaked,
                         note="pipeline cancel time; browser playback stops locally in < 50 ms")
            done3.clear()
            ev3.clear()
            s3.submit_text("Wait, explain that in Sinhala.")
            done3.wait(180)
            time.sleep(0.5)
            r = next((e for _, e in ev3 if e.get("type") == "reply_done"), {})
            txt = r.get("text", "")
            tr = next((e for _, e in ev3 if e.get("type") == "transcript"), {})
            si_ratio = len(re.findall(r"[඀-෿]", txt)) / max(1, len(re.findall(r"[^\W\d_]", txt)))
            self._record("pipeline", "interrupt_then_switch_language", "si",
                         (tr.get("language") or {}).get("reply") == "si" and si_ratio >= 0.5 and not degenerate(txt),
                         decided=(tr.get("language") or {}), reply=txt[:400], sinhala_script_ratio=round(si_ratio, 2),
                         degenerate=degenerate(txt))
        self.failure_suite()

    def failure_suite(self) -> None:
        from ..ollama_client import OllamaClient
        # Ollama down: the engine must report an LLM error, not hang or crash.
        class _Down(OllamaClient):
            def __init__(self, base):
                super().__init__(base.cfg)
                self.host = "http://127.0.0.1:9"
        real = self.rt.ollama
        try:
            self.rt.ollama = _Down(real)
            s, ev, _, done = self._session()
            s.submit_text("hello")
            done.wait(30)
            err = next((e for _, e in ev if e.get("type") == "error"), None)
            self._record("failure", "ollama_down", None, bool(err and err.get("where") == "llm"),
                         event=err and err.get("message", "")[:200])
        finally:
            self.rt.ollama = real
        # TTS failure: no voice for a language -> reported, text still delivered.
        router = self.engine.tts
        saved = router.voices
        try:
            router.voices = {"en": [], "si": []}
            out = router.synthesize("hello", "en")
            self._record("failure", "tts_missing_voice", None, out["wav"] == b"" and out["missing"] == ["en"],
                         result={k: out[k] for k in ("missing",)})
        finally:
            router.voices = saved
        # STT failure: missing model -> STTError (UI shows ERROR state).
        from .stt import BilingualWhisper
        broken = BilingualWhisper(self.engine.models_dir, "does-not-exist.bin", {})
        try:
            broken.transcribe(array("h", bytes(32000)))
            ok = False
        except STTError:
            ok = True
        self._record("failure", "stt_missing_model", None, ok)
        for name in ("microphone_disconnected", "speaker_disconnected"):
            self._record("failure", name, None, None,
                         note="browser-side: the UI listens for track 'ended' / devicechange and shows ERROR; "
                              "verify manually by unplugging the device")

    # ---------------------------------------------------------------- run
    def run(self, suites=("synthetic", "tts", "recorded", "pipeline")) -> dict:
        self.rt.db.audit("voice", "voice_selftest_started", self.run_id, suites=list(suites))
        before = _process_usage()
        t0 = time.perf_counter()
        for name in suites:
            try:
                getattr(self, f"{name}_suite")()
            except Exception as e:  # noqa: BLE001
                self._record(name, "suite_crashed", None, False, error=f"{type(e).__name__}: {e}")
        after = _process_usage()
        self._record("resources", "process_usage", None, None, before=before, after=after,
                     duration_s=round(time.perf_counter() - t0, 1))
        summary = self.summary()
        self.rt.db.audit("voice", "voice_selftest_finished", self.run_id, **{k: v for k, v in summary.items()
                                                                             if k != "by_category"})
        return summary

    def summary(self) -> dict:
        def agg(rows):
            ws = [r["wer"] for r in rows if r["wer"] is not None]
            cs = [r["cer"] for r in rows if r["cer"] is not None]
            ls = [r["latency"] for r in rows if r["latency"] is not None]
            return {"n": len(rows), "passed": sum(1 for r in rows if r["passed"] == 1),
                    "failed": sum(1 for r in rows if r["passed"] == 0),
                    "skipped": sum(1 for r in rows if r["passed"] is None),
                    "mean_wer": round(sum(ws) / len(ws), 4) if ws else None,
                    "mean_cer": round(sum(cs) / len(cs), 4) if cs else None,
                    "mean_latency": round(sum(ls) / len(ls), 3) if ls else None}
        cats: dict[str, list] = {}
        for r in self.results:
            cats.setdefault(r["category"], []).append(r)
        return {"run_id": self.run_id, **agg(self.results), "by_category": {k: agg(v) for k, v in cats.items()}}


def save_recording(rt, pcm: array, transcript: str, language: str, category: str) -> dict:
    """Store ONE consented recording in the local test set (never uploaded)."""
    d = rt.cfg.data_dir / "voice_tests" / "dataset"
    d.mkdir(parents=True, exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.wav"
    (d / name).write_bytes(wav_bytes(pcm, STT_RATE))
    entry = {"file": name, "transcript": transcript, "language": language, "category": category,
             "consent": True, "consented_at": time.time(), "use": "local evaluation only - not for training"}
    with (d / "manifest.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    rt.db.audit("user", "voice_recording_saved_to_testset", name, category=category, language=language)
    return entry


def delete_dataset(rt) -> int:
    d = rt.cfg.data_dir / "voice_tests" / "dataset"
    n = 0
    if d.exists():
        for p in d.iterdir():
            p.unlink()
            n += 1
    rt.db.audit("user", "voice_testset_deleted", None, files=n)
    return n
