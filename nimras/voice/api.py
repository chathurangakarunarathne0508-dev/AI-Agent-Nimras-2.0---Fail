"""HTTP routes for the voice engine (served by nimras/server.py)."""
from __future__ import annotations

import json
import re
import threading

from ..db import row_to_dict
from .registry import VoiceModelRegistry

_selftest_lock = threading.Lock()
_selftest_state: dict = {"running": False, "summary": None, "log": []}


def voice_get(rt, path: str, qs: dict) -> tuple:
    if path == "/api/voice/status":
        return (rt.voice.status(),)
    if path == "/api/voice/hardware":
        from .hardware import inspect
        return (inspect(rt.voice.models_dir),)
    if path == "/api/voice/turns":
        rows = rt.db.all("SELECT * FROM voice_turns ORDER BY id DESC LIMIT ?", (int(qs.get("limit", 50)),))
        return ([row_to_dict(r, ("client_metrics",)) for r in rows],)
    if path == "/api/voice/performance":
        q = ("SELECT reply_lang, COUNT(*) n, ROUND(AVG(stt_latency),3) stt, ROUND(AVG(llm_first_token),3) llm_first, "
             "ROUND(AVG(first_audio),3) first_audio, ROUND(AVG(total),3) total, SUM(interrupted) interrupts, "
             "ROUND(AVG(interrupt_latency),3) interrupt_latency, SUM(error IS NOT NULL) errors "
             "FROM voice_turns GROUP BY reply_lang")
        return ({"by_language": [dict(r) for r in rt.db.all(q)]},)
    if path == "/api/voice/tests":
        run = rt.db.one("SELECT run_id FROM voice_tests ORDER BY id DESC LIMIT 1")
        rows = rt.db.all("SELECT * FROM voice_tests WHERE run_id=? ORDER BY id", (run["run_id"],)) if run else []
        return ({"running": _selftest_state["running"], "summary": _selftest_state["summary"],
                 "log": _selftest_state["log"][-40:], "run_id": run and run["run_id"],
                 "results": [row_to_dict(r, ("details",)) for r in rows]},)
    if path == "/api/voice/models":
        return ({"models": VoiceModelRegistry(rt).list()},)
    return ({"error": "not found"}, 404)


def voice_post(rt, path: str, body: dict) -> tuple:
    if path == "/api/voice/selftest":
        if not _selftest_lock.acquire(blocking=False):
            return ({"error": "a self-test is already running"}, 409)
        suites = [s for s in (body.get("suites") or ["synthetic", "tts", "recorded", "pipeline"])
                  if s in ("synthetic", "tts", "recorded", "pipeline")]

        def go():
            from .selftest import VoiceSelfTest
            _selftest_state.update(running=True, summary=None, log=[])
            try:
                st = VoiceSelfTest(rt, log=_selftest_state["log"].append)
                _selftest_state["summary"] = st.run(tuple(suites))
            except Exception as e:  # noqa: BLE001
                _selftest_state["log"].append(f"self-test crashed: {e}")
            finally:
                _selftest_state["running"] = False
                _selftest_lock.release()
        threading.Thread(target=go, daemon=True, name="nve-selftest").start()
        return ({"ok": True, "started": True, "suites": suites},)

    if path == "/api/voice/history/delete":
        scope = body.get("scope", "voice")
        if scope == "all":
            n = rt.db.execute("DELETE FROM messages").rowcount
        else:
            n = rt.db.execute("DELETE FROM messages WHERE conversation_id LIKE 'voice-%'").rowcount
        rt.db.execute("DELETE FROM voice_turns")
        rt.db.audit("user", "transcripts_deleted", None, scope=scope, messages=n)
        return ({"ok": True, "deleted_messages": n},)

    if path == "/api/voice/dataset":
        if body.get("consent") is not True:
            return ({"error": "explicit consent is required to keep a recording"}, 400)
        from .selftest import save_recording
        from .transport import SESSIONS
        s = SESSIONS.get(str(body.get("session", "")))
        if not s or s.last_utterance is None:
            return ({"error": "no recent utterance in this session"}, 404)
        cat = str(body.get("category", "recording"))
        if not re.fullmatch(r"[a-z_]{3,40}", cat):
            return ({"error": "bad category"}, 400)
        lang = body.get("language") if body.get("language") in ("en", "si", "mixed") else "en"
        transcript = str(body.get("transcript", "")).strip()[:2000]
        if not transcript:
            return ({"error": "the correct transcript is required"}, 400)
        return (save_recording(rt, s.last_utterance, transcript, lang, cat),)

    if path == "/api/voice/dataset/delete":
        from .selftest import delete_dataset
        return ({"ok": True, "deleted_files": delete_dataset(rt)},)

    if path == "/api/voice/updates/check":
        return (VoiceModelRegistry(rt).check_updates(),)

    m = re.fullmatch(r"/api/voice/models/(\d+)/(evaluate|promote)", path)
    if m:
        reg, mid = VoiceModelRegistry(rt), int(m.group(1))
        if m.group(2) == "evaluate":
            return (reg.evaluate(mid),)
        # The UI sends authorise=true only from an explicit confirmation dialog.
        return (reg.promote(mid, authorised_by_user=body.get("authorise") is True),)
    if path == "/api/voice/models/rollback":
        return (VoiceModelRegistry(rt).rollback(str(body.get("kind", "stt_lang")), body.get("language")),)
    return ({"error": "not found"}, 404)


__all__ = ["voice_get", "voice_post", "json"]
