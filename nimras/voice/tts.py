"""Text-to-speech providers + language-aware speech router (pluggable).

* PiperVoice  - Piper (MIT) neural TTS, run as a persistent local process per
                voice so the model is loaded once. Default voices:
                English  en_US-lessac-medium,  Sinhala  si_LK-dilu-medium (Dilu).
* SapiVoice   - Windows' built-in System.Speech (offline). English fallback only.
* TTSRouter   - splits mixed Sinhala-English text into script runs, sends each
                run to the right voice, trims edge silence and joins with short
                crossfades so a mixed sentence sounds continuous.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from array import array
from pathlib import Path

from .assets import find_file
from .audio import concat_with_crossfade, read_wav, resample, trim_silence_edges, wav_bytes
from .language import segments
from .vocabulary import Vocabulary


class TTSError(RuntimeError):
    pass


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Voice:
    name = "base"
    language = "en"
    rate = 22050

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[array, int]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class PiperVoice(Voice):
    def __init__(self, models_dir: Path, model_file: str, language: str, label: str = ""):
        self.models_dir = Path(models_dir).resolve()
        self.model_file = model_file
        self.language = language
        self.name = label or Path(model_file).stem
        self._proc: subprocess.Popen | None = None
        self._speed = None
        self._lock = threading.Lock()
        self._out = Path(tempfile.gettempdir()) / "nimras-nve"
        self._out.mkdir(exist_ok=True)
        cfg = self.model_path.with_suffix(".onnx.json") if self.model_path else None
        if cfg and cfg.exists():
            try:
                self.rate = json.loads(cfg.read_text(encoding="utf-8-sig"))["audio"]["sample_rate"]
            except (ValueError, KeyError):
                pass

    @property
    def model_path(self) -> Path | None:
        hit = find_file(self.models_dir, Path(self.model_file).name)
        return hit.resolve() if hit else None

    @property
    def exe(self) -> Path | None:
        for n in ("piper.exe", "piper"):
            hit = find_file(self.models_dir / "piper", n)
            if hit:
                return hit.resolve()
        return None

    def available(self) -> tuple[bool, str]:
        if not self.exe:
            return False, "Piper not installed (python -m nimras voice setup)"
        if not self.model_path:
            return False, f"voice {self.model_file} not installed"
        return True, f"piper:{self.name}"

    def _start(self, speed: float) -> None:
        self.close()
        # length_scale > 1 is slower speech.
        cmd = [str(self.exe), "-m", str(self.model_path), "--json-input", "-q",
               "--length_scale", f"{1.0 / max(0.5, min(2.0, speed)):.3f}", "--sentence_silence", "0.12"]
        self._proc = subprocess.Popen(cmd, cwd=str(self.exe.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, creationflags=_no_window())
        self._speed = speed

    def warm_up(self) -> None:
        try:
            self.synthesize("ok" if self.language == "en" else "හරි")
        except TTSError:
            pass

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[array, int]:
        ok, why = self.available()
        if not ok:
            raise TTSError(why)
        with self._lock:
            if not self._proc or self._proc.poll() is not None or self._speed != speed:
                self._start(speed)
            out = self._out / f"{uuid.uuid4().hex}.wav"
            line = json.dumps({"text": text.replace("\n", " "), "output_file": str(out)}, ensure_ascii=False)
            try:
                self._proc.stdin.write(line.encode("utf-8") + b"\n")
                self._proc.stdin.flush()
                done = self._proc.stdout.readline()       # piper prints the path when finished
            except OSError as e:
                self.close()
                raise TTSError(f"piper crashed: {e}") from e
            if not done:
                self.close()
                raise TTSError("piper exited unexpectedly")
            for _ in range(50):
                if out.exists() and out.stat().st_size > 44:
                    break
                time.sleep(0.02)
            try:
                data = out.read_bytes()
            finally:
                out.unlink(missing_ok=True)
        pcm, rate = read_wav(data)
        return pcm, rate

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except (OSError, subprocess.TimeoutExpired):
                self._proc.kill()
        self._proc = None


class SapiVoice(Voice):
    """Windows offline speech (English). Used when Piper is unavailable."""
    name = "windows-sapi"

    def available(self) -> tuple[bool, str]:
        return (os.name == "nt", "Windows SAPI" if os.name == "nt" else "not Windows")

    def synthesize(self, text: str, speed: float = 1.0) -> tuple[array, int]:
        if os.name != "nt":
            raise TTSError("SAPI only exists on Windows")
        with tempfile.TemporaryDirectory(prefix="nve-sapi-") as td:
            txt, wav = Path(td) / "t.txt", Path(td) / "o.wav"
            txt.write_text(text, encoding="utf-8")
            rate = max(-10, min(10, round((speed - 1.0) * 10)))
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  f"$s.Rate = {rate}; $s.SetOutputToWaveFile('{wav}'); "
                  f"$s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8)); $s.Dispose()")
            r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                               capture_output=True, timeout=60, creationflags=_no_window())
            if r.returncode != 0 or not wav.exists():
                raise TTSError(f"SAPI failed: {r.stderr.decode(errors='replace')[-200:]}")
            return read_wav(wav.read_bytes())


class TTSRouter:
    OUT_RATE = 22050

    def __init__(self, voices: dict[str, list[Voice]], vocab: Vocabulary):
        self.voices = voices          # language -> ordered candidates (first available wins)
        self.vocab = vocab
        self.selected: dict[str, str] = {}

    def voice_for(self, lang: str) -> Voice | None:
        cands = self.voices.get(lang, [])
        pref = self.selected.get(lang)
        if pref:
            cands = sorted(cands, key=lambda v: v.name != pref)
        for v in cands:
            if v.available()[0]:
                return v
        return None

    def status(self) -> dict:
        return {lang: [{"name": v.name, "available": v.available()[0], "detail": v.available()[1],
                        "selected": (self.voice_for(lang) is v)} for v in vs] for lang, vs in self.voices.items()}

    def prepare(self, lang: str, text: str) -> str:
        if lang == "en":
            return self.vocab.for_speech(text)
        # Sinhala: strip markdown; espeak-ng's Sinhala rules read digits correctly
        # (verified: "25" -> "visipəhə"), so numbers are left as they are.
        import re
        t = re.sub(r"```.*?```", " ", text, flags=re.S)
        t = re.sub(r"[*_`#>|]+", " ", t)
        t = re.sub(r"https?://\S+", " ", t)
        return re.sub(r"\s{2,}", " ", t).strip()

    def synthesize(self, text: str, default_lang: str = "en", speed: float = 1.0) -> dict:
        """Returns {"wav": bytes, "segments": [...], "missing": [langs without a voice]}."""
        parts: list[array] = []
        used, missing = [], []
        for lang, chunk in segments(text) or [(default_lang, text)]:
            v = self.voice_for(lang)
            if v is None:
                missing.append(lang)
                continue
            spoken = self.prepare(lang, chunk)
            if not spoken.strip(" .,!?;:-"):
                continue
            pcm, rate = v.synthesize(spoken, speed)
            pcm = resample(pcm, rate, self.OUT_RATE)
            parts.append(trim_silence_edges(pcm, self.OUT_RATE))
            used.append({"lang": lang, "voice": v.name, "text": chunk.strip()})
        if not parts:
            return {"wav": b"", "segments": used, "missing": sorted(set(missing))}
        pcm = concat_with_crossfade(parts, self.OUT_RATE, gap_ms=60 if len(parts) > 1 else 0)
        return {"wav": wav_bytes(pcm, self.OUT_RATE), "segments": used, "missing": sorted(set(missing)),
                "seconds": len(pcm) / self.OUT_RATE}

    def warm_up(self) -> None:
        def go():
            for vs in self.voices.values():
                for v in vs:
                    if isinstance(v, PiperVoice) and v.available()[0]:
                        v.warm_up()
        threading.Thread(target=go, daemon=True, name="nve-tts-warmup").start()

    def close(self) -> None:
        for vs in self.voices.values():
            for v in vs:
                v.close()


def build_tts(cfg, models_dir: Path, vocab: Vocabulary) -> TTSRouter:
    voices: dict[str, list[Voice]] = {}
    for lang, files in (cfg.get("voice.tts.voices") or {}).items():
        voices[lang] = [PiperVoice(models_dir, f, lang) for f in files]
    if cfg.get("voice.tts.sapi_fallback", True):
        voices.setdefault("en", []).append(SapiVoice())
    router = TTSRouter(voices, vocab)
    router.selected = dict(cfg.get("voice.tts.selected") or {})
    return router
