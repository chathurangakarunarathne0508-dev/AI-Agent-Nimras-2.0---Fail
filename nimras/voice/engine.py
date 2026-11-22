"""N.I.M.R.A.S. Voice Engine (NVE): the transport-independent core.

    mic audio (16 kHz PCM, already endpointed by the client VAD)
      -> preprocessing (DC/high-pass/normalise) -> server VAD check/trim
      -> STT (language ID -> English or Sinhala model)
      -> ICT vocabulary normalisation (reported, never silent)
      -> language decision (spoken / explicit request / sticky preference)
      -> local intents (stop, repeat)
      -> conversation context + RAG + Ollama (streaming)
      -> sentence splitting -> language-aware TTS -> audio chunks
    with cancellation at every stage for barge-in.

`VoiceEngine` owns the shared, expensive components (models). `VoiceSession`
is one conversation (one browser connection). Events are delivered through
callbacks, so the engine is reusable from WebSocket, CLI or tests.
"""
from __future__ import annotations

import json
import queue
import re
import struct
import threading
import time
import traceback
import uuid
from array import array
from pathlib import Path
from typing import Callable

from .audio import STT_RATE, EnergyVAD, pcm_from_bytes, preprocess, rms
from .language import LANG_NAMES, decide_language, split_sentences
from .stt import STTError, build_stt
from .tts import TTSError, build_tts
from .vocabulary import Vocabulary

VOICE_STYLE = """VOICE CONVERSATION MODE - your words are converted to speech and played aloud.
- No markdown, bullet symbols, tables, emojis, code blocks or URLs. Short natural spoken sentences.
- Simple questions: answer in one to three sentences. Give detail only when the user asks for it.
- Do not greet or introduce yourself again; just continue the conversation.
- If the request is unclear or incomplete, ask one short clarifying question instead of guessing.
- If the user corrects themselves or you, follow the correction.
- The user's words come from speech recognition and may contain recognition mistakes; infer the intended meaning from context.
{lang_rule}{interrupted}"""

LANG_RULES = {
    "en": "- Reply in English.",
    "si": ("- Reply in Sinhala, written in Sinhala script (සිංහල අකුරු), never romanised Sinhala. Keep technical "
           "terms and product names in English (for example firewall, VLAN, DHCP, Active Directory, Intune)."),
}

_STOP = re.compile(r"^\s*(stop|cancel|be quiet|shut up|never ?mind|නවත්තන්න|නවත්වන්න|එපා|nawaththanna)[.!]?\s*$", re.I)
_REPEAT = re.compile(r"^\s*(repeat( that)?|say (that|it) again|again please|ආයෙ(ත්)? කියන්න|aye kiyanna)[.!?]?\s*$", re.I)


class Turn:
    def __init__(self, turn_id: int):
        self.id = turn_id
        self.cancel = threading.Event()
        self.t_speech_end = time.perf_counter()
        self.marks: dict[str, float] = {}
        self.reply = ""
        self.interrupted = False
        self.interrupt_requested_at: float | None = None
        self.done = threading.Event()

    def mark(self, key: str) -> None:
        self.marks.setdefault(key, round(time.perf_counter() - self.t_speech_end, 3))


class VoiceEngine:
    def __init__(self, rt):
        self.rt = rt
        self.cfg, self.db = rt.cfg, rt.db
        root = Path(self.cfg.get("voice.models_dir", "voice_models"))
        if not root.is_absolute():
            root = (self.cfg.path.parent if self.cfg.path else Path.cwd()) / root
        self.models_dir = root.resolve()
        self.vocab = Vocabulary(self.cfg.data_dir / "voice_vocabulary.json")
        self.stt = build_stt(self.cfg, self.models_dir)
        self.tts = build_tts(self.cfg, self.models_dir, self.vocab)
        self.vad = EnergyVAD(min_speech_ms=self.cfg.get("voice.vad.min_speech_ms", 250))
        self._warm = False

    def warm_up(self) -> None:
        if not self._warm:
            self._warm = True
            self.stt.warm_up()
            self.tts.warm_up()

    def status(self) -> dict:
        stt_ok, stt_detail = self.stt.available()
        health = self.rt.ollama.health()
        model = self.model
        return {
            "stt": {"available": stt_ok, "detail": stt_detail, **getattr(self.stt, "status", lambda: {})()},
            "tts": self.tts.status(),
            "llm": {"model": model, "reachable": health.get("reachable"),
                    "ready": model in (health.get("models") or []) or f"{model}:latest" in (health.get("models") or [])},
            "vad": self.client_vad_config(),
            "privacy": {"store_transcripts": self.cfg.get("voice.store_transcripts", True),
                        "store_audio": False, "processing": "local"},
        }

    @property
    def model(self) -> str:
        return self.cfg.get("voice.ollama_model") or self.rt.ollama.chat_model

    def client_vad_config(self) -> dict:
        g = lambda k, d: self.cfg.get(f"voice.vad.{k}", d)  # noqa: E731
        return {"start_ratio": g("start_ratio", 3.0), "min_level": g("min_level", 0.012),
                "start_frames": g("start_frames", 4), "endpoint_ms": g("endpoint_ms", 750),
                "max_utterance_s": g("max_utterance_s", 30), "preroll_ms": g("preroll_ms", 300),
                "min_speech_ms": g("min_speech_ms", 250), "barge_in_ratio": g("barge_in_ratio", 2.2),
                "barge_in_ms": g("barge_in_ms", 280)}

    def close(self) -> None:
        self.stt.close()
        self.tts.close()


class VoiceSession:
    """One live conversation. Thread-safe; at most one turn is active."""

    def __init__(self, engine: VoiceEngine, emit: Callable[[dict], None], emit_audio: Callable[[bytes], None],
                 conversation_id: str | None = None):
        self.engine, self.rt = engine, engine.rt
        self.emit, self.emit_audio = emit, emit_audio
        self.conversation_id = conversation_id or "voice-" + uuid.uuid4().hex[:10]
        self.preferred_lang: str | None = None     # set only by an explicit request
        self.speed = float(engine.cfg.get("voice.tts.speed", 1.0))
        self._audio = bytearray()
        self._receiving = False
        self._turn: Turn | None = None
        self._turn_seq = 0
        self._lock = threading.Lock()
        self.last_reply: tuple[str, str] | None = None   # (text, lang)
        self.last_utterance: array | None = None          # RAM only; saved to the test set only on request

    # ------------------------------------------------------------ audio input
    def start_utterance(self) -> None:
        with self._lock:
            self._audio = bytearray()
            self._receiving = True
        self.emit({"type": "state", "state": "listening"})

    def add_audio(self, chunk: bytes) -> None:
        if self._receiving and len(self._audio) < STT_RATE * 2 * self.engine.cfg.get("voice.vad.max_utterance_s", 30):
            self._audio.extend(chunk)

    def abort_utterance(self) -> None:
        with self._lock:
            self._receiving = False
            self._audio = bytearray()
        if not (self._turn and not self._turn.done.is_set()):
            self.emit({"type": "state", "state": "idle"})

    def end_utterance(self, client: dict | None = None) -> None:
        with self._lock:
            self._receiving = False
            pcm = pcm_from_bytes(bytes(self._audio))
            self._audio = bytearray()
        self.interrupt(reason="new utterance")
        turn = self._new_turn()
        threading.Thread(target=self._run_turn, args=(turn, pcm, None, client or {}), daemon=True,
                         name=f"nve-turn-{turn.id}").start()

    def submit_text(self, text: str) -> None:
        """Typed fallback (e.g. microphone unavailable). Same pipeline minus STT."""
        self.interrupt(reason="new text")
        turn = self._new_turn()
        threading.Thread(target=self._run_turn, args=(turn, None, text, {}), daemon=True).start()

    def _new_turn(self) -> Turn:
        with self._lock:
            self._turn_seq += 1
            self._turn = Turn(self._turn_seq)
            return self._turn

    # ------------------------------------------------------------ interruption
    def interrupt(self, reason: str = "user") -> float | None:
        t = self._turn
        if not t or t.done.is_set():
            return None
        t.interrupt_requested_at = time.perf_counter()
        t.interrupted = True
        t.cancel.set()
        self.emit({"type": "state", "state": "interrupted", "turn": t.id, "detail": reason})
        t.done.wait(timeout=3)
        return round(time.perf_counter() - t.interrupt_requested_at, 3)

    # ------------------------------------------------------------ pipeline
    def _run_turn(self, turn: Turn, pcm: array | None, typed: str | None, client: dict) -> None:
        hold = getattr(self.rt.ollama, "interactive", None)
        if hold is None:
            return self._run_turn_inner(turn, pcm, typed, client)
        with hold():
            return self._run_turn_inner(turn, pcm, typed, client)

    def _run_turn_inner(self, turn: Turn, pcm: array | None, typed: str | None, client: dict) -> None:
        eng = self.engine
        rec = {"conversation_id": self.conversation_id, "created_at": time.time()}
        try:
            if pcm is not None:
                self.emit({"type": "state", "state": "processing", "turn": turn.id, "detail": "recognising speech"})
                # VAD on the raw signal: normalisation would amplify pure noise.
                if not eng.vad.has_speech(pcm):
                    self.emit({"type": "no_speech", "turn": turn.id})
                    self.emit({"type": "state", "state": "idle"})
                    return
                clean = preprocess(eng.vad.trim(pcm), STT_RATE)
                self.last_utterance = clean
                recent = " ".join(r["content"] for r in self.rt.db.all(
                    "SELECT content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 4",
                    (self.conversation_id,)))
                tr = eng.stt.transcribe(clean, "auto", eng.vocab.initial_prompt(recent))
                turn.mark("stt_done")
                rec.update(stt_engine=tr.engine, stt_model=tr.model, audio_seconds=round(tr.duration_s, 2),
                           stt_latency=round(tr.latency_s, 3))
                heard, whisper_lang = tr.text, tr.language
                lang_probs, alternatives = tr.language_probs, tr.alternatives
            else:
                heard, whisper_lang, lang_probs, alternatives = typed or "", None, {}, []
            if turn.cancel.is_set():
                return
            text, changes = eng.vocab.normalize(heard)
            if not text.strip():
                self.emit({"type": "no_speech", "turn": turn.id, "detail": "nothing recognisable"})
                self.emit({"type": "state", "state": "idle"})
                return
            decision = decide_language(text, whisper_lang, self.preferred_lang)
            if decision["explicit"]:
                self.preferred_lang = decision["explicit"]     # sticky until changed again
            reply_lang = decision["reply"]
            rec.update(spoken_lang=decision["spoken"], reply_lang=reply_lang)
            self.emit({"type": "transcript", "turn": turn.id, "text": text, "heard": heard, "changes": changes,
                       "language": decision, "language_probs": lang_probs, "typed": pcm is None,
                       "alternatives": alternatives})

            if _STOP.match(text):
                self.emit({"type": "state", "state": "idle", "detail": "stopped"})
                return
            if _REPEAT.match(text) and self.last_reply:
                self._speak_all(turn, self.last_reply[0], self.last_reply[1])
                return
            self._respond(turn, text, reply_lang, decision, rec, alternatives)
        except STTError as e:
            rec["error"] = f"stt: {e}"
            self.emit({"type": "error", "where": "stt", "message": str(e)})
        except Exception as e:  # noqa: BLE001 - report every failure to the UI
            rec["error"] = f"{type(e).__name__}: {e}"
            where = "llm" if "Ollama" in str(e) or "ollama" in type(e).__name__.lower() else "pipeline"
            self.emit({"type": "error", "where": where, "message": str(e),
                       "trace": traceback.format_exc()[-600:]})
        finally:
            if turn.interrupted and turn.interrupt_requested_at:
                rec["interrupt_latency"] = round(time.perf_counter() - turn.interrupt_requested_at, 3)
            rec.update(interrupted=int(turn.interrupted), llm_first_token=turn.marks.get("llm_first_token"),
                       first_audio=turn.marks.get("first_audio"), total=turn.marks.get("done"),
                       client_metrics=json.dumps(client) if client else None)
            turn.done.set()
            if rec.get("spoken_lang") or rec.get("error"):
                try:
                    self.rt.db.insert("voice_turns", rec)
                except Exception:
                    pass
                self.emit({"type": "metrics", "turn": turn.id, **{k: rec.get(k) for k in (
                    "stt_latency", "llm_first_token", "first_audio", "total", "interrupted", "interrupt_latency",
                    "audio_seconds", "stt_model")}, "marks": turn.marks})
            if not turn.interrupted and not rec.get("error"):
                self.emit({"type": "state", "state": "idle"})
            elif rec.get("error"):
                self.emit({"type": "state", "state": "error", "detail": rec["error"]})

    def _system_extra(self, reply_lang: str, decision: dict) -> str:
        interrupted = ""
        last = self.rt.db.one("SELECT content FROM messages WHERE conversation_id=? AND role='assistant' "
                              "ORDER BY id DESC LIMIT 1", (self.conversation_id,))
        if last and last["content"].endswith("[interrupted by user]"):
            interrupted = ("\n- Your previous answer was cut off because the user interrupted you. The user's new "
                           "words refer to it; adjust (e.g. switch language, shorten, clarify) rather than restarting "
                           "from scratch.")
        rule = LANG_RULES.get(reply_lang, LANG_RULES["en"])
        if decision["spoken"] == "mixed":
            rule += "\n- The user mixes Sinhala and English; that is normal - understand both."
        return VOICE_STYLE.format(lang_rule=rule, interrupted=interrupted)

    def _respond(self, turn: Turn, text: str, reply_lang: str, decision: dict, rec: dict,
                 alternatives: list | None = None) -> None:
        rt, eng = self.rt, self.engine
        store = eng.cfg.get("voice.store_transcripts", True)
        if store:
            rt.db.insert("messages", {"conversation_id": self.conversation_id, "role": "user", "content": text,
                                      "created_at": time.time()})
        rt.db.kv_set("user_last_active", time.time())
        extra = self._system_extra(reply_lang, decision)
        others = [a for a in (alternatives or []) if a.get("text") and a.get("text") != text]
        if others:
            # Code-switched speech: each recogniser hears part of it well (the Sinhala model the
            # Sinhala words, the English model the technical terms). Let the model combine them.
            extra += ("\n- Speech recognition was unsure which language was spoken. Other transcript(s) of the same "
                      "audio: " + " | ".join(f"[{a['language']}] {a['text']}" for a in others) +
                      " . Combine them to infer what the user meant (English technical terms are often clearer in "
                      "the English transcript). If still unclear, ask a short clarifying question.")
        ctx = self._voice_context(text, extra)
        self.emit({"type": "state", "state": "processing", "turn": turn.id, "detail": "thinking"})

        sentences: queue.Queue = queue.Queue()
        tts_thread = threading.Thread(target=self._tts_worker, args=(turn, sentences, reply_lang), daemon=True,
                                      name=f"nve-tts-{turn.id}")
        tts_thread.start()
        buf = ""
        first_chunk = True
        try:
            for delta in rt.ollama.chat_stream(ctx["messages"], model=eng.model, temperature=0.5,
                                               num_ctx=eng.cfg.get("voice.num_ctx") or 8192, cancel=turn.cancel):
                turn.mark("llm_first_token")
                turn.reply += delta
                buf += delta
                self.emit({"type": "reply_delta", "turn": turn.id, "text": delta})
                done, buf = split_sentences(buf)
                # Speak the first clause early (lower time-to-first-audio).
                if first_chunk and not done and len(buf) > 80:
                    cut = max(buf.rfind(", "), buf.rfind("، "), buf.rfind("; "))
                    if cut > 30:
                        done, buf = [buf[:cut + 1].strip()], buf[cut + 1:]
                for s in done:
                    first_chunk = False
                    sentences.put(s)
            if buf.strip() and not turn.cancel.is_set():
                sentences.put(buf.strip())
        finally:
            sentences.put(None)
            tts_thread.join(timeout=120)
        turn.mark("done")
        reply = turn.reply.strip()
        if turn.cancel.is_set():
            reply = (reply + " … [interrupted by user]").strip()
        if reply:
            self.last_reply = (turn.reply.strip(), reply_lang)
            if store:
                threading.Thread(target=rt.chat.finish, args=(text, self.conversation_id, reply, ctx, eng.model),
                                 daemon=True).start()
        self.emit({"type": "reply_done", "turn": turn.id, "text": turn.reply.strip(), "interrupted": turn.cancel.is_set(),
                   "language": reply_lang, "language_name": LANG_NAMES.get(reply_lang)})

    def _voice_context(self, text: str, extra: str) -> dict:
        """Compact, cache-friendly context for low first-token latency on CPU.

        Measured: llama3.2:1b evaluates ~110 prompt tokens/s here, so the text-chat
        context (6 x 1200-char knowledge chunks, ~1500 tokens) cost 13 s before the
        first word. Voice uses a few short, relevant chunks, and keeps the system
        prompt + history as a stable prefix (Ollama reuses its KV cache), with the
        per-question knowledge attached to the *current* user message only.
        """
        rt, cfg = self.rt, self.engine.cfg
        k = cfg.get("voice.rag_top_k", 3)
        min_score = cfg.get("voice.rag_min_score", 0.45)
        chars = cfg.get("voice.rag_chars", 450)
        retrieved = [r for r in rt.store.retrieve(text, k=k) if r.get("score", 0) >= min_score][:k] if k else []
        prompt = rt.prompts.active("chat_system")
        system = prompt["template"]
        personal = rt.chat.personal_context()
        if personal:
            system += f"\n\nAPPROVED USER PREFERENCES / FACTS:\n{personal}"
        system += "\n\n" + extra
        history = rt.db.all("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                            (self.conversation_id, cfg.get("voice.history_messages", 8)))
        msgs = [{"role": "system", "content": system}] + [
            {"role": h["role"], "content": h["content"]} for h in reversed(history)]
        if not msgs[-1:] or msgs[-1]["role"] != "user" or msgs[-1]["content"] != text:
            msgs.append({"role": "user", "content": text})
        if retrieved:
            knowledge = "\n".join(f"[{i + 1}] ({r['status']}) {r['title']}: {' '.join(r['text'].split())[:chars]}"
                                  for i, r in enumerate(retrieved))
            msgs[-1] = {"role": "user", "content": f"(Knowledge you may use, cite status if relevant:\n{knowledge})\n\n"
                                                    f"{text}"}
        slim = [{k2: r.get(k2) for k2 in ("type", "id", "entry_id", "title", "status", "confidence", "url", "score")}
                | {"text": r["text"][:300]} for r in retrieved]
        return {"messages": msgs, "retrieved": retrieved, "slim": slim, "prompt": prompt}

    def _tts_worker(self, turn: Turn, sentences: queue.Queue, reply_lang: str) -> None:
        seq = 0
        while True:
            s = sentences.get()
            if s is None:
                break
            if turn.cancel.is_set():
                continue                   # drain: pending speech chunks are cancelled
            try:
                out = self.engine.tts.synthesize(s, reply_lang, self.speed)
            except TTSError as e:
                self.emit({"type": "error", "where": "tts", "message": str(e), "turn": turn.id,
                           "fallback_text": s, "language": reply_lang})
                continue
            if out["missing"]:
                self.emit({"type": "tts_missing", "turn": turn.id, "languages": out["missing"], "text": s})
            if not out["wav"] or turn.cancel.is_set():
                continue
            if seq == 0:
                turn.mark("first_audio")
                self.emit({"type": "state", "state": "speaking", "turn": turn.id})
            self.emit({"type": "audio_meta", "turn": turn.id, "seq": seq, "text": s, "segments": out["segments"],
                       "seconds": round(out.get("seconds", 0), 2)})
            self.emit_audio(struct.pack("<II", turn.id, seq) + out["wav"])
            seq += 1
        self.emit({"type": "audio_end", "turn": turn.id, "chunks": seq})

    def _speak_all(self, turn: Turn, text: str, lang: str) -> None:
        q: queue.Queue = queue.Queue()
        for s in split_sentences(text + "\n")[0] or [text]:
            q.put(s)
        q.put(None)
        self._tts_worker(turn, q, lang)
        turn.mark("done")

    # ------------------------------------------------------------ settings
    def apply_settings(self, data: dict) -> dict:
        if "speed" in data:
            self.speed = max(0.6, min(1.6, float(data["speed"])))
        if "language" in data:
            lang = data["language"]
            self.preferred_lang = lang if lang in ("en", "si") else None
        voices = data.get("voices") or {}
        for lang, name in voices.items():
            if lang in ("en", "si") and isinstance(name, str):
                self.engine.tts.selected[lang] = name
        return {"speed": self.speed, "language": self.preferred_lang or "auto",
                "voices": dict(self.engine.tts.selected)}

    def rms_of_last(self) -> float:
        return rms(self.last_utterance) if self.last_utterance else 0.0
