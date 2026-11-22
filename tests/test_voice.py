"""Offline tests for the N.I.M.R.A.S. Voice Engine (fake STT/TTS/LLM, real pipeline).

The real-model checks (Whisper, Piper, Ollama) are in nimras/voice/selftest.py
(`python -m nimras voice test`), because they need the downloaded models.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import tempfile
import threading
import time
import unittest
from array import array
from http.server import ThreadingHTTPServer
from pathlib import Path

from nimras.voice.audio import STT_RATE, EnergyVAD, read_wav, resample, tone, wav_bytes
from nimras.voice.convert_whisper import _map_name, bytes_to_unicode
from nimras.voice.language import analyze, decide_language, requested_language, segments, split_sentences
from nimras.voice.selftest import cer, wer
from nimras.voice.stt import BilingualWhisper, STTError, Transcript
from nimras.voice.vocabulary import Vocabulary
from nimras.voice.ws import origin_allowed

from tests.test_nimras import FakeOllama, make_runtime

SI_REPLY = "DHCP යනු IP ලිපින ස්වයංක්‍රීයව ලබා දෙන සේවාවකි. එය ජාලයට පහසුවක්."
EN_REPLY = "DHCP automatically assigns IP addresses to devices. It uses UDP ports 67 and 68."


class VoiceOllama(FakeOllama):
    """Adds streaming; replies in the language the voice system prompt asks for."""
    delay = 0.0

    def chat_stream(self, messages, model=None, temperature=0.4, num_ctx=None, cancel=None, keep_alive="30m"):
        with self._slot():
            system = messages[0]["content"]
            reply = SI_REPLY if "Reply in Sinhala" in system else EN_REPLY
            if "LONG" in messages[-1]["content"]:
                reply = " ".join([EN_REPLY] * 12)
            self.last_system = system
            for word in reply.split(" "):
                if cancel is not None and cancel.is_set():
                    return
                if self.delay:
                    time.sleep(self.delay)
                yield word + " "


class DownOllama(VoiceOllama):
    def chat_stream(self, *a, **k):
        from nimras.ollama_client import OllamaError
        raise OllamaError("Ollama unreachable: connection refused")
        yield  # pragma: no cover


class FakeSTT:
    def __init__(self):
        self.next = ("what is dhcp", "en")
        self.calls = 0

    def available(self):
        return True, "fake"

    def status(self):
        return {"fake": True}

    def transcribe(self, pcm, language="auto", prompt=""):
        self.calls += 1
        text, lang = self.next
        return Transcript(text=text, language=lang, duration_s=len(pcm) / STT_RATE, latency_s=0.01, engine="fake",
                          model="fake.bin", language_probs={lang: 0.97})

    def warm_up(self):
        pass

    def close(self):
        pass


class FakeTTS:
    def __init__(self):
        self.calls = []
        self.selected = {}

    def synthesize(self, text, default_lang="en", speed=1.0):
        self.calls.append((default_lang, text))
        return {"wav": wav_bytes(tone(220, 120, 22050), 22050), "segments": [{"lang": default_lang, "text": text}],
                "missing": [], "seconds": 0.12}

    def status(self):
        return {"en": [{"name": "fake-en", "available": True, "selected": True}],
                "si": [{"name": "fake-si", "available": True, "selected": True}]}

    def voice_for(self, lang):
        return None

    def warm_up(self):
        pass

    def close(self):
        pass


def speechlike(seconds=1.0) -> array:
    """Amplitude-modulated tone: passes the energy VAD like a voiced utterance."""
    t = tone(180, int(seconds * 1000), STT_RATE, amp=0.35)
    return array("h", (int(s * (0.6 + 0.4 * ((i // 1600) % 2))) for i, s in enumerate(t)))


class Collector:
    def __init__(self):
        self.events, self.audio = [], []
        self.lock = threading.Lock()

    def emit(self, e):
        with self.lock:
            self.events.append((time.perf_counter(), e))

    def emit_audio(self, b):
        with self.lock:
            self.audio.append((time.perf_counter(), b))

    def of(self, kind):
        return [e for _, e in self.events if e.get("type") == kind]

    def wait(self, kind, n=1, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            if len(self.of(kind)) >= n:
                return True
            time.sleep(0.01)
        return False


# ---------------------------------------------------------------------- units
class LanguageTest(unittest.TestCase):
    def test_detection(self):
        self.assertEqual(analyze("How do I configure a VLAN?").code, "en")
        self.assertEqual(analyze("මට උදව් ඕනේ").code, "si")
        self.assertEqual(analyze("මගේ laptop එකේ WiFi වැඩ කරන්නේ නැහැ").code, "mixed")
        self.assertEqual(analyze("mata help ekak denna puluwanda").code, "mixed")   # Singlish

    def test_explicit_requests_and_stickiness(self):
        self.assertEqual(requested_language("Wait, explain that in Sinhala."), "si")
        self.assertEqual(requested_language("සිංහලෙන් කියන්න"), "si")
        self.assertEqual(requested_language("now answer in English please"), "en")
        self.assertIsNone(requested_language("what is sinhala new year"))
        d = decide_language("Wait, explain that in Sinhala.", "en")
        self.assertEqual((d["spoken"], d["reply"], d["explicit"]), ("en", "si", "si"))
        self.assertEqual(decide_language("what about DNS?", "en", preferred="si")["reply"], "si")
        self.assertEqual(decide_language("මට උදව් ඕනේ", "si")["reply"], "si")

    def test_segments_for_tts_router(self):
        segs = segments("ඔබේ firewall එකේ VLAN 10 පරීක්ෂා කරන්න.")
        self.assertEqual([lang for lang, _ in segs], ["si", "en", "si", "en", "si"])
        self.assertIn("10", segs[3][1])               # digits stay with their run
        self.assertEqual(segments("Hello there."), [("en", "Hello there.")])

    def test_sentence_splitting(self):
        done, rest = split_sentences("First one. Second one! Third")
        self.assertEqual(done, ["First one.", "Second one!"])
        self.assertEqual(rest, "Third")


class VocabularyTest(unittest.TestCase):
    def setUp(self):
        self.v = Vocabulary()

    def test_normalisation_only_changes_case_and_spacing(self):
        text, changes = self.v.normalize("set the v lan and d h c p on hyper v, then check dns")
        self.assertEqual(text, "set the VLAN and DHCP on Hyper-V, then check DNS")
        for c in changes:
            strip = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731
            self.assertEqual(strip(c["heard"]), strip(c["normalized"]))  # letters never change

    def test_no_meaning_changes(self):
        for s in ("it is in tune with the music", "I am here", "the soar of the eagle", "is our ram ok"):
            self.assertEqual(self.v.normalize(s)[0], s)

    def test_prompt_and_pronunciation(self):
        p = self.v.initial_prompt("we discussed Intune yesterday")
        self.assertIn("Intune", p[:120])
        self.assertLessEqual(len(p), 600)
        spoken = self.v.for_speech("Check the **SIEM** and VLAN at https://x.y/z")
        self.assertIn("seem", spoken)
        self.assertIn("V-LAN", spoken)
        self.assertNotIn("*", spoken)
        self.assertNotIn("https", spoken)


class AudioTest(unittest.TestCase):
    def test_wav_roundtrip_and_resample(self):
        pcm = tone(440, 500, 22050)
        back, rate = read_wav(wav_bytes(pcm, 22050))
        self.assertEqual((len(back), rate), (len(pcm), 22050))
        self.assertAlmostEqual(len(resample(pcm, 22050, 16000)), 8000, delta=2)

    def test_vad(self):
        vad = EnergyVAD()
        silence = array("h", bytes(2 * STT_RATE))
        self.assertFalse(vad.has_speech(silence))
        clip = array("h", bytes(2 * 8000)) + speechlike(1.0) + array("h", bytes(2 * 8000))
        self.assertTrue(vad.has_speech(clip))
        trimmed = vad.trim(clip)
        self.assertLess(len(trimmed), len(clip))
        self.assertGreater(len(trimmed), STT_RATE * 0.9)

    def test_metrics(self):
        self.assertEqual(wer("turn on the lights", "turn on the lights"), 0)
        self.assertAlmostEqual(wer("turn on the lights", "turn the light"), 0.5)
        self.assertEqual(cer("ක්‍රියා කරන්නේ", "ක්‍රියාකරන්නේ"), 0)   # spacing-only difference


class ConverterTest(unittest.TestCase):
    def test_tensor_name_mapping(self):
        self.assertEqual(_map_name("model.encoder.layers.3.self_attn.k_proj.weight"), "encoder.blocks.3.attn.key.weight")
        self.assertEqual(_map_name("model.decoder.layers.0.encoder_attn.k_proj.weight"),
                         "decoder.blocks.0.cross_attn.key.weight")
        self.assertEqual(_map_name("model.decoder.layers.1.fc2.bias"), "decoder.blocks.1.mlp.2.bias")
        self.assertEqual(_map_name("model.decoder.embed_tokens.weight"), "decoder.token_embedding.weight")
        self.assertIsNone(_map_name("proj_out.weight"))
        self.assertEqual(len(bytes_to_unicode()), 256)

    def test_audio_ctx(self):
        self.assertEqual(BilingualWhisper.audio_ctx(3.0), 448)
        self.assertEqual(BilingualWhisper.audio_ctx(29.0), 0)       # full window
        self.assertEqual(BilingualWhisper.audio_ctx(5, adaptive=False), 0)

    def test_missing_model_raises(self):
        b = BilingualWhisper(Path(tempfile.gettempdir()) / "nope", "missing.bin", {})
        with self.assertRaises(STTError):
            b.transcribe(array("h", bytes(3200)))


# ---------------------------------------------------------------------- engine
class EngineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="nimras_voice_"))
        self.rt = make_runtime(self.tmp)
        self.rt.ollama = VoiceOllama(self.rt.cfg)
        self.rt.chat.ollama = self.rt.ollama
        self.eng = self.rt.voice
        self.eng.stt, self.eng.tts = FakeSTT(), FakeTTS()
        from nimras.voice.engine import VoiceSession
        self.c = Collector()
        self.s = VoiceSession(self.eng, self.c.emit, self.c.emit_audio)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def speak(self, seconds=1.0):
        self.s.start_utterance()
        self.s.add_audio(speechlike(seconds).tobytes())
        self.s.end_utterance({"speech_ms": int(seconds * 1000)})

    def test_english_turn_end_to_end(self):
        self.speak()
        self.assertTrue(self.c.wait("reply_done"))
        self.assertTrue(self.c.wait("audio_end"))
        tr = self.c.of("transcript")[0]
        self.assertEqual(tr["text"], "what is DHCP")                       # vocabulary normalisation
        self.assertEqual(tr["changes"], [{"heard": "dhcp", "normalized": "DHCP"}])
        self.assertEqual(tr["language"]["reply"], "en")
        self.assertIn("67", self.c.of("reply_done")[0]["text"])
        turn, seq = struct.unpack("<II", self.c.audio[0][1][:8])
        self.assertEqual((turn, seq), (tr["turn"], 0))
        self.assertTrue(self.c.audio[0][1][8:12] == b"RIFF")
        time.sleep(0.2)
        row = self.rt.db.one("SELECT * FROM voice_turns ORDER BY id DESC LIMIT 1")
        self.assertEqual((row["spoken_lang"], row["reply_lang"], row["interrupted"]), ("en", "en", 0))
        self.assertIsNotNone(row["first_audio"])
        self.assertIn("VOICE CONVERSATION MODE", self.rt.ollama.last_system)

    def test_sinhala_speech_gets_sinhala_reply(self):
        self.eng.stt.next = ("DHCP කියන්නේ මොකක්ද?", "si")
        self.speak()
        self.assertTrue(self.c.wait("reply_done"))
        self.assertIn("Reply in Sinhala", self.rt.ollama.last_system)
        self.assertEqual(self.c.of("reply_done")[0]["language"], "si")
        self.assertTrue(self.c.wait("audio_end"))
        self.assertTrue(all(lang == "si" for lang, _ in self.eng.tts.calls))

    def test_interrupt_then_switch_to_sinhala(self):
        self.rt.ollama.delay = 0.02
        self.s.submit_text("Explain firewalls LONG")
        self.assertTrue(self.c.wait("audio_meta"))
        lat = self.s.interrupt("barge-in")
        self.assertIsNotNone(lat)
        self.assertLess(lat, 1.0)
        n_audio = len(self.c.audio)
        time.sleep(0.4)
        self.assertEqual(len(self.c.audio), n_audio, "no audio may be sent after an interrupt")
        done = self.c.of("reply_done")[0]
        self.assertTrue(done["interrupted"])
        time.sleep(0.2)
        last = self.rt.db.one("SELECT content FROM messages WHERE role='assistant' ORDER BY id DESC LIMIT 1")
        self.assertTrue(last["content"].endswith("[interrupted by user]"))
        # User interrupts in English asking for Sinhala.
        self.rt.ollama.delay = 0
        self.s.submit_text("Wait, explain that in Sinhala.")
        self.assertTrue(self.c.wait("reply_done", 2))
        tr = self.c.of("transcript")[-1]
        self.assertEqual((tr["language"]["spoken"], tr["language"]["reply"]), ("en", "si"))
        self.assertIn("previous answer was cut off", self.rt.ollama.last_system)
        self.assertEqual(self.c.of("reply_done")[-1]["language"], "si")
        self.assertEqual(self.s.preferred_lang, "si")                   # sticky until changed

    def test_silence_is_not_answered(self):
        self.s.start_utterance()
        self.s.add_audio(bytes(2 * STT_RATE))
        self.s.end_utterance()
        self.assertTrue(self.c.wait("no_speech"))
        self.assertEqual(self.eng.stt.calls, 0)
        self.assertEqual(self.c.of("reply_done"), [])

    def test_stop_command_is_local(self):
        self.eng.stt.next = ("stop", "en")
        self.speak(0.6)
        self.assertTrue(self.c.wait("transcript"))
        time.sleep(0.2)
        self.assertEqual(self.c.of("reply_done"), [])

    def test_ollama_failure_is_reported(self):
        self.rt.ollama = DownOllama(self.rt.cfg)
        self.s.submit_text("hello there")
        self.assertTrue(self.c.wait("error"))
        self.assertEqual(self.c.of("error")[0]["where"], "llm")
        self.assertTrue(any(e.get("state") == "error" for e in self.c.of("state")))

    def test_learning_waits_during_voice_turn(self):
        from nimras.ollama_client import LLMBudget
        self.rt.ollama.delay = 0.03
        self.s.submit_text("Explain firewalls LONG")
        self.assertTrue(self.c.wait("reply_delta"))
        got = []

        def background():
            self.rt.ollama.set_budget(LLMBudget(5))      # marks this thread as background learning
            with self.rt.ollama._slot():
                got.append(time.perf_counter())
        t = threading.Thread(target=background)
        t.start()
        time.sleep(0.2)
        self.assertEqual(got, [], "background model calls must wait while a voice turn is active")
        self.s.interrupt()
        t.join(5)
        self.assertEqual(len(got), 1)


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="nimras_voice_srv_"))
        cls.rt = make_runtime(cls.tmp)
        cls.rt.ollama = VoiceOllama(cls.rt.cfg)
        cls.rt.chat.ollama = cls.rt.ollama
        cls.rt.voice.stt, cls.rt.voice.tts = FakeSTT(), FakeTTS()
        from nimras.server import make_handler
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.rt))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def handshake(self, origin):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET /ws/voice HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\nUpgrade: websocket\r\n"
                   f"Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
                   f"Origin: {origin}\r\n\r\n").encode())
        return s, s.recv(4096)

    @staticmethod
    def frame(payload: bytes, opcode=1) -> bytes:
        mask = os.urandom(4)
        n = len(payload)
        head = struct.pack(">BB", 0x80 | opcode, 0x80 | n) if n < 126 else struct.pack(">BBH", 0x80 | opcode, 0xFE, n)
        return head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    @staticmethod
    def read_frame(s) -> tuple[int, bytes]:
        def exact(n):
            buf = b""
            while len(buf) < n:
                buf += s.recv(n - len(buf))
            return buf
        b1, b2 = exact(2)
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", exact(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", exact(8))[0]
        return b1 & 0x0F, exact(n)

    def test_foreign_origin_rejected(self):
        self.assertTrue(origin_allowed(f"http://127.0.0.1:{self.port}", self.port))
        self.assertFalse(origin_allowed("https://evil.example", self.port))
        s, resp = self.handshake("https://evil.example")
        self.assertIn(b" 403 ", resp.split(b"\r\n")[0])
        s.close()

    def test_websocket_conversation(self):
        s, resp = self.handshake(f"http://127.0.0.1:{self.port}")
        self.assertIn(b" 101 ", resp.split(b"\r\n")[0])
        _, hello = self.read_frame(s)
        self.assertEqual(json.loads(hello)["type"], "hello")
        s.sendall(self.frame(json.dumps({"type": "text", "text": "what is DHCP?"}).encode()))
        got_audio, got_done = False, False
        end = time.time() + 15
        while time.time() < end and not (got_audio and got_done):
            op, data = self.read_frame(s)
            if op == 2:
                got_audio = data[8:12] == b"RIFF"
            elif op == 1 and json.loads(data).get("type") == "reply_done":
                got_done = True
        self.assertTrue(got_audio and got_done)
        s.close()

    def test_privacy_endpoints(self):
        from nimras.voice.api import voice_post
        body, code = voice_post(self.rt, "/api/voice/dataset", {"transcript": "x"})
        self.assertEqual(code, 400)                      # consent is mandatory
        from nimras.voice.registry import VoiceModelRegistry
        reg = VoiceModelRegistry(self.rt)
        mid = reg.register("stt_lang", "candidate.bin", "si")
        with self.assertRaises(PermissionError):
            reg.promote(mid, authorised_by_user=False)
        with self.assertRaises(PermissionError):         # authorised but never benchmarked
            reg.promote(mid, authorised_by_user=True)
        res, = voice_post(self.rt, "/api/voice/history/delete", {"scope": "voice"})
        self.assertTrue(res["ok"])


if __name__ == "__main__":
    unittest.main()
