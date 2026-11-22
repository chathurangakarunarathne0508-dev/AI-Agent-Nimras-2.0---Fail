"""WebSocket transport between the voice-first UI and the Voice Engine.

Client -> server
  binary frame                 16-bit 16 kHz mono PCM, only while an utterance is open
  {"type":"start_utterance"}   client VAD detected speech (pre-roll included in audio)
  {"type":"end_utterance", "client": {...}}   endpoint reached
  {"type":"abort_utterance"}   too short / noise
  {"type":"interrupt"}         barge-in or the on-screen stop control
  {"type":"text","text":...}   typed fallback when the microphone is unavailable
  {"type":"settings", ...}     speed / language preference / voice selection
  {"type":"client_metrics", ...}

Server -> client: JSON events (state, transcript, reply_delta, audio_meta,
audio_end, reply_done, metrics, error) and binary audio frames:
  [u32 turn][u32 seq][WAV bytes]
"""
from __future__ import annotations

import json
import threading
import uuid

from .engine import VoiceSession
from .ws import WebSocket, WSClosed

SESSIONS: dict[str, VoiceSession] = {}
_LOCK = threading.Lock()


def handle_voice_socket(handler, rt) -> None:
    port = handler.server.server_address[1]
    ws = WebSocket.handshake(handler, port)
    if ws is None:
        return
    engine = rt.voice
    engine.warm_up()
    sid = uuid.uuid4().hex[:12]

    def emit(event: dict) -> None:
        try:
            ws.send_json(event)
        except WSClosed:
            pass

    def emit_audio(data: bytes) -> None:
        try:
            ws.send_bytes(data)
        except WSClosed:
            pass

    session = VoiceSession(engine, emit, emit_audio)
    with _LOCK:
        SESSIONS[sid] = session
    rt.db.audit("user", "voice_session_started", f"voice:{sid}")
    emit({"type": "hello", "session": sid, "conversation_id": session.conversation_id,
          "status": engine.status(), "vad": engine.client_vad_config()})
    try:
        while True:
            kind, data = ws.recv()
            if kind == "binary":
                session.add_audio(data)
                continue
            try:
                msg = json.loads(data)
            except ValueError:
                continue
            t = msg.get("type")
            if t == "start_utterance":
                session.start_utterance()
            elif t == "end_utterance":
                session.end_utterance(msg.get("client") or {})
            elif t == "abort_utterance":
                session.abort_utterance()
            elif t == "interrupt":
                latency = session.interrupt(reason=str(msg.get("reason", "user"))[:40])
                emit({"type": "interrupt_ack", "pipeline_cancel_s": latency})
            elif t == "text":
                text = str(msg.get("text", "")).strip()[:2000]
                if text:
                    session.submit_text(text)
            elif t == "settings":
                emit({"type": "settings", **session.apply_settings(msg)})
            elif t == "client_metrics":
                rt.db.execute("UPDATE voice_turns SET client_metrics=? WHERE id=(SELECT MAX(id) FROM voice_turns "
                              "WHERE conversation_id=?)", (json.dumps(msg.get("data") or {})[:4000],
                                                           session.conversation_id))
            elif t == "ping":
                emit({"type": "pong"})
    except WSClosed:
        pass
    finally:
        session.interrupt(reason="disconnected")
        with _LOCK:
            SESSIONS.pop(sid, None)
        rt.db.audit("user", "voice_session_ended", f"voice:{sid}")
        ws.close()
