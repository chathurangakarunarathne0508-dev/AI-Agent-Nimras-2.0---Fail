"""Speech-to-text providers (pluggable).

Default provider: `bilingual-whisper` built on whisper.cpp (MIT), fully offline:

  1. Language ID with the multilingual model (whisper detect-only pass,
     probabilities restricted to the configured languages: en / si).
  2. English  -> multilingual model (ggml-small-q5_1).
     Sinhala  -> Sinhala fine-tuned model (SinhaSpeech run12, converted locally).
     Stock Whisper produces degenerate output on Sinhala (measured), hence the
     dedicated model.

Each model runs in its own `whisper-server` on loopback so it is loaded once;
`whisper-cli` is the fallback. Any class implementing `STTProvider` can be
swapped in via `voice.stt.provider`.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .assets import find_file
from .audio import STT_RATE, wav_bytes


class STTError(RuntimeError):
    pass


@dataclass
class Transcript:
    text: str
    language: str | None          # acoustic language ID ("en", "si", ...)
    duration_s: float
    latency_s: float
    engine: str
    model: str = ""
    language_probs: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)
    alternatives: list = field(default_factory=list)   # dual-decode hypotheses (ambiguous language)


class STTProvider:
    name = "base"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def transcribe(self, pcm16k, language: str = "auto", prompt: str = "") -> Transcript:
        raise NotImplementedError

    def warm_up(self) -> None:
        pass

    def close(self) -> None:
        pass


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


class WhisperModel:
    """One ggml model served by a loopback whisper-server (CLI fallback)."""

    def __init__(self, models_dir: Path, model_file: str, port: int, threads: int, beam_size: int):
        self.models_dir = Path(models_dir).resolve()
        self.model_file = model_file
        self.port, self.threads, self.beam_size = port, threads, beam_size
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._server_failed = False

    @property
    def model_path(self) -> Path | None:
        p = Path(self.model_file)
        if p.is_absolute():
            return p if p.exists() else None
        hit = find_file(self.models_dir, p.name)
        return hit.resolve() if hit else None

    def exe(self, name: str) -> Path | None:
        for n in (name + ".exe", name):
            hit = find_file(self.models_dir / "whisper.cpp", n)
            if hit:
                return hit.resolve()
        return None

    def ready(self) -> bool:
        return bool(self.model_path and (self.exe("whisper-server") or self.exe("whisper-cli")))

    # ---------------------------------------------------------------- server
    def _url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def ensure_server(self) -> bool:
        if self._server_failed:
            return False
        if self._proc and self._proc.poll() is None:
            return True
        exe = self.exe("whisper-server")
        if not exe or not self.model_path:
            return False
        cmd = [str(exe), "-m", str(self.model_path), "--host", "127.0.0.1", "--port", str(self.port),
               "-t", str(self.threads), "-bs", str(self.beam_size), "-l", "auto"]
        self._proc = subprocess.Popen(cmd, cwd=str(exe.parent), stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL, creationflags=_no_window())
        deadline = time.time() + 90
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._server_failed = True
                return False
            try:
                urllib.request.urlopen(self._url() + "/", timeout=2).read()
                return True
            except urllib.error.HTTPError:
                return True
            except (urllib.error.URLError, OSError):
                time.sleep(0.25)
        self._server_failed = True
        self.close()
        return False

    @staticmethod
    def _multipart(fields: dict, wav: bytes) -> tuple[bytes, str]:
        boundary = uuid.uuid4().hex
        parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                 for k, v in fields.items()]
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
                     f"Content-Type: audio/wav\r\n\r\n".encode() + wav + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return b"".join(parts), f"multipart/form-data; boundary={boundary}"

    def _server(self, wav: bytes, fields: dict) -> dict:
        body, ctype = self._multipart(fields, wav)
        req = urllib.request.Request(self._url() + "/inference", data=body, method="POST",
                                     headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        if isinstance(data, dict) and data.get("error"):
            raise STTError(str(data["error"]))
        return data

    def _cli(self, wav: bytes, language: str, prompt: str, detect_only: bool) -> dict:
        exe = self.exe("whisper-cli")
        if not exe:
            raise STTError("whisper-cli not found")
        with tempfile.TemporaryDirectory(prefix="nve-") as td:
            wp = Path(td) / "in.wav"
            wp.write_bytes(wav)
            out = Path(td) / "out"
            cmd = [str(exe), "-m", str(self.model_path), "-f", str(wp), "-l", language, "-t", str(self.threads),
                   "-bs", str(self.beam_size), "-oj", "-of", str(out), "-np", "-nt"]
            if detect_only:
                cmd.append("-dl")
            if prompt:
                cmd += ["--prompt", prompt]
            r = subprocess.run(cmd, cwd=str(exe.parent), capture_output=True, timeout=240,
                               creationflags=_no_window())
            jf = out.with_suffix(".json")
            if r.returncode != 0:
                raise STTError(f"whisper-cli failed ({r.returncode}): {r.stderr.decode(errors='replace')[-300:]}")
            if detect_only:
                # "auto-detected language: si (p = 0.83)" on stderr
                import re
                m = re.search(r"auto-detected language: (\w+) \(p = ([\d.]+)\)", r.stderr.decode(errors="replace"))
                return {"language": m.group(1) if m else None,
                        "language_probabilities": {m.group(1): float(m.group(2))} if m else {}, "text": ""}
            data = json.loads(jf.read_text(encoding="utf-8", errors="replace")) if jf.exists() else {}
        segs = data.get("transcription", [])
        return {"text": "".join(s.get("text", "") for s in segs),
                "language": (data.get("result") or {}).get("language"), "segments": segs}

    def run(self, wav: bytes, language: str = "auto", prompt: str = "", detect_only: bool = False,
            audio_ctx: int = 0) -> tuple[dict, str]:
        with self._lock:
            if self.ensure_server():
                fields = {"response_format": "verbose_json", "temperature": "0.0", "language": language}
                if audio_ctx:
                    fields["audio_ctx"] = str(audio_ctx)
                if detect_only:
                    fields["detect_language"] = "true"
                if prompt and not detect_only:
                    fields["prompt"] = prompt
                try:
                    d = self._server(wav, fields)
                    lang = (d.get("detected_language") or d.get("language") or "").lower()
                    return {"text": d.get("text", ""), "language": _LANG_CODES.get(lang, lang) or None,
                            "language_probabilities": d.get("language_probabilities") or {},
                            "segments": d.get("segments", [])}, "whisper-server"
                except (urllib.error.URLError, OSError, ValueError, STTError):
                    pass
            return self._cli(wav, language, prompt, detect_only), "whisper-cli"

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None


class BilingualWhisper(STTProvider):
    name = "bilingual-whisper"

    def __init__(self, models_dir: Path, general_model: str, language_models: dict[str, str],
                 allowed_languages: tuple[str, ...] = ("en", "si"), threads: int = 0, base_port: int = 8771,
                 beam_size: int = 3, adaptive_ctx: bool = True, dual_threshold: dict | None = None,
                 dual_bias: float = 0.0, lid_model: str = ""):
        threads = threads or max(2, min(12, (os.cpu_count() or 4) - 2))
        self.allowed = allowed_languages
        self.adaptive_ctx = adaptive_ctx
        self.dual_threshold = dual_threshold or {"en": 0.995, "si": 0.9}
        self.dual_bias = dual_bias
        self.general = WhisperModel(models_dir, general_model, base_port, threads, beam_size)
        # Language ID only has to choose en vs si: a small model's encoder is ~3x cheaper.
        self.lid = WhisperModel(models_dir, lid_model, base_port + 9, threads, 1) if lid_model else None
        self.lang_models: dict[str, WhisperModel] = {}
        for i, (lang, mf) in enumerate(language_models.items()):
            if mf:
                self.lang_models[lang] = WhisperModel(models_dir, mf, base_port + 1 + i, threads, beam_size)

    def available(self) -> tuple[bool, str]:
        if not self.general.ready():
            return False, (f"STT model {self.general.model_file} or whisper.cpp missing "
                           "(python -m nimras voice setup)")
        langs = ["en"] + [lang for lang, m in self.lang_models.items() if m.ready()]
        missing = [lang for lang, m in self.lang_models.items() if not m.ready()]
        note = f"languages with dedicated models: {', '.join(sorted(set(langs)))}"
        if missing:
            note += f"; MISSING dedicated model for: {', '.join(missing)} (falls back to multilingual model)"
        return True, note

    def status(self) -> dict:
        return {"general": self.general.model_file, "general_ready": self.general.ready(),
                "language_models": {k: {"file": m.model_file, "ready": m.ready()} for k, m in self.lang_models.items()},
                "allowed_languages": list(self.allowed)}

    def warm_up(self) -> None:
        def go():
            if self.lid and self.lid.ready():
                self.lid.ensure_server()
            self.general.ensure_server()
            for m in self.lang_models.values():
                if m.ready():
                    m.ensure_server()
        threading.Thread(target=go, daemon=True, name="nve-stt-warmup").start()

    @staticmethod
    def audio_ctx(seconds: float, adaptive: bool = True) -> int:
        """Whisper always encodes a 30 s window (1500 frames, 50/s). Shrinking it
        to the clip length is ~3.5x faster on CPU. Measured here: 448 frames
        minimum keeps transcripts exact; smaller windows started dropping letters."""
        if not adaptive:
            return 0
        n = int(seconds * 50) + 160
        n = max(448, -(-n // 64) * 64)
        return 0 if n >= 1500 else n

    def identify(self, pcm16k) -> tuple[str | None, dict]:
        # Language ID needs only the first few seconds.
        head = pcm16k[: STT_RATE * 8]
        ctx = self.audio_ctx(len(head) / STT_RATE, self.adaptive_ctx)
        model = self.lid if (self.lid and self.lid.ready()) else self.general
        res, _ = model.run(wav_bytes(head, STT_RATE), "auto", detect_only=True, audio_ctx=ctx)
        probs = {k: float(v) for k, v in (res.get("language_probabilities") or {}).items()}
        if probs and self.allowed:
            allowed = {k: probs.get(k, 0.0) for k in self.allowed}
            if any(allowed.values()):
                total = sum(allowed.values())
                allowed = {k: round(v / total, 3) for k, v in allowed.items()}
                return max(allowed, key=allowed.get), allowed
        lang = (res.get("language") or "").lower() or None
        if lang and self.allowed and lang not in self.allowed:
            # Sri Lankan speech mis-labelled as e.g. Hindi/Tamil: non-English -> Sinhala.
            lang = "si" if "si" in self.allowed else self.allowed[0]
        return lang, probs

    def transcribe(self, pcm16k, language: str = "auto", prompt: str = "") -> Transcript:
        ok, why = self.available()
        if not ok:
            raise STTError(why)
        wav = wav_bytes(pcm16k, STT_RATE)
        t0 = time.perf_counter()
        probs: dict = {}
        if language == "auto":
            language, probs = self.identify(pcm16k)
            language = language or "en"
        ctx = self.audio_ctx(len(pcm16k) / STT_RATE, self.adaptive_ctx)
        alternatives = []
        if probs and self._ambiguous(probs):
            # Code-switched speech (e.g. starting with an English term) often gets an
            # English language ID. Decode with both models and keep the hypothesis the
            # models are more confident about (mean token log-probability).
            cands = []
            for lang, model in [("en", self.general)] + [(k, m) for k, m in self.lang_models.items() if m.ready()]:
                res, engine = model.run(wav, lang, prompt if model is self.general else "", audio_ctx=ctx)
                score = _confidence(res)
                cands.append((score + (self.dual_bias if lang != "en" else 0.0), lang, model, res, engine))
                alternatives.append({"language": lang, "text": _clean(res.get("text", "")), "avg_logprob": score})
            _, language, model, res, engine = max(cands, key=lambda c: c[0])
        else:
            model = self.lang_models.get(language)
            if not (model and model.ready()):
                model = self.general
            # The ICT prompt helps the multilingual model; the Sinhala model was
            # fine-tuned without prompts, so it gets none (prompts can derail it).
            use_prompt = prompt if model is self.general else ""
            res, engine = model.run(wav, language, use_prompt, audio_ctx=ctx)
        return Transcript(text=_clean(res.get("text", "")), language=language,
                          duration_s=len(pcm16k) / STT_RATE, latency_s=time.perf_counter() - t0, engine=engine,
                          model=Path(model.model_file).name, language_probs=probs,
                          segments=res.get("segments", [])[:50], alternatives=alternatives)

    def _ambiguous(self, probs: dict) -> bool:
        if not any(m.ready() for m in self.lang_models.values()):
            return False
        sure_en = probs.get("en", 0.0) >= self.dual_threshold.get("en", 0.995)
        sure_other = any(probs.get(k, 0.0) >= self.dual_threshold.get(k, 0.9) for k in self.lang_models)
        return not (sure_en or sure_other)

    def close(self) -> None:
        self.general.close()
        if self.lid:
            self.lid.close()
        for m in self.lang_models.values():
            m.close()


def _confidence(res: dict) -> float:
    """Duration-weighted mean token log-probability (higher = more confident).
    Temperature fallback (decoder struggled) is penalised. CLI results have no
    scores and rank last."""
    segs = res.get("segments") or []
    num = den = 0.0
    for s in segs:
        lp = s.get("avg_logprob")
        if lp is None:
            continue
        dur = max(0.1, float(s.get("end", 0)) - float(s.get("start", 0)))
        num += (float(lp) - 0.5 * float(s.get("temperature") or 0.0)) * dur
        den += dur
    return round(num / den, 4) if den else -9.0


_HALLUCINATIONS = {"thank you.", "thanks for watching!", "thank you for watching.", "you", "[blank_audio]",
                   "(silence)", "[music]", "[ silence ]", "subtitles by the amara.org community", "."}


def _clean(text: str) -> str:
    t = " ".join(text.replace("[BLANK_AUDIO]", " ").split()).strip()
    if t.lower() in _HALLUCINATIONS:
        return ""
    # Collapse runaway character repetition (a known Whisper failure mode).
    import re
    t = re.sub(r"(.{1,4}?)\1{6,}", r"\1", t)
    return t


def build_stt(cfg, models_dir: Path) -> STTProvider:
    provider = cfg.get("voice.stt.provider", "bilingual-whisper")
    if provider in ("bilingual-whisper", "whisper.cpp"):
        return BilingualWhisper(
            models_dir, cfg.get("voice.stt.model", "ggml-small-q5_1.bin"),
            cfg.get("voice.stt.language_models", {"si": "ggml-sinhala-small-q5_1.bin"}),
            allowed_languages=tuple(cfg.get("voice.stt.allowed_languages", ["en", "si"])),
            threads=cfg.get("voice.stt.threads", 0), base_port=cfg.get("voice.stt.port", 8771),
            beam_size=cfg.get("voice.stt.beam_size", 3), adaptive_ctx=cfg.get("voice.stt.adaptive_audio_ctx", True),
            dual_threshold=cfg.get("voice.stt.dual_decode_threshold"), dual_bias=cfg.get("voice.stt.dual_decode_bias", 0.0),
            lid_model=cfg.get("voice.stt.language_id_model", "ggml-base-q5_1.bin"))
    raise STTError(f"unknown STT provider {provider!r}")


_LANG_CODES = {"english": "en", "sinhala": "si", "tamil": "ta", "hindi": "hi"}
