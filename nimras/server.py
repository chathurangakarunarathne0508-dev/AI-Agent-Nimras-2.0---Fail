"""Local HTTP API + N.I.M.R.A.S. LEARNING CENTRE web UI (stdlib only).

Hardening: binds to localhost; rejects foreign Host headers (DNS rebinding);
state-changing requests need a custom header, which cross-site pages cannot send
without a CORS preflight we never approve (CSRF).
"""
from __future__ import annotations

import json
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .db import row_to_dict

WEB_DIR = Path(__file__).parent / "web"
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def make_handler(rt):
    class Handler(BaseHTTPRequestHandler):
        server_version = "NIMRAS/2.0"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        # ------------------------------------------------ plumbing
        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].lower()
            return host in _ALLOWED_HOSTS

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            port = self.server.server_address[1]
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; "
                             "img-src 'self' data:; media-src 'self' blob:; worker-src 'self'; "
                             f"connect-src 'self' ws://127.0.0.1:{port} ws://localhost:{port}; frame-ancestors 'none'")
            # Only this page may use the microphone; never camera/geolocation.
            self.send_header("Permissions-Policy", "microphone=(self), camera=(), geolocation=()")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code: int = 200) -> None:
            self._send(code, json.dumps(data, default=str).encode(), "application/json")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1_000_000:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        # ------------------------------------------------ GET
        def do_GET(self):
            if not self._host_ok():
                return self._send(403, b"forbidden host", "text/plain")
            u = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(u.query).items()}
            path = u.path
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
                if path in ("/app.js", "/style.css", "/voice.js", "/voice-worklet.js", "/voice.css"):
                    ctype = "application/javascript" if path.endswith(".js") else "text/css"
                    return self._send(200, (WEB_DIR / path.lstrip("/")).read_bytes(), ctype + "; charset=utf-8")
                if path == "/ws/voice":
                    from .voice.transport import handle_voice_socket
                    return handle_voice_socket(self, rt)
                if path.startswith("/api/voice/"):
                    from .voice.api import voice_get
                    return self._json(*voice_get(rt, path, qs))
                if path == "/api/dashboard":
                    return self._json(rt.dashboard())
                if path == "/api/library":
                    return self._json(rt.library(qs.get("q", ""), qs.get("layer", ""), qs.get("category", ""),
                                                 qs.get("status", ""), int(qs.get("limit", 50)),
                                                 int(qs.get("offset", 0))))
                m = re.fullmatch(r"/api/entries/(\d+)", path)
                if m:
                    d = rt.entry_detail(int(m.group(1)))
                    return self._json(d) if d else self._json({"error": "not found"}, 404)
                if path == "/api/topics":
                    st = qs.get("status")
                    rows = rt.db.all("SELECT * FROM topics" + (" WHERE status=?" if st else "") +
                                     " ORDER BY priority DESC, id DESC LIMIT 300", (st,) if st else ())
                    return self._json([dict(r) for r in rows])
                if path == "/api/sources":
                    rows = rt.db.all("SELECT id,url,domain,title,kind,published_at,retrieved_at,credibility,status,"
                                     "injection_flags,error FROM sources ORDER BY id DESC LIMIT 200")
                    return self._json([dict(r) for r in rows])
                if path == "/api/memory":
                    return self._json([dict(r) for r in rt.db.all("SELECT * FROM personal_memory ORDER BY id DESC")])
                if path == "/api/audit":
                    rows = rt.db.all("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (int(qs.get("limit", 200)),))
                    return self._json([row_to_dict(r, ("details",)) for r in rows])
                if path == "/api/sessions":
                    rows = rt.db.all("SELECT * FROM learning_sessions ORDER BY id DESC LIMIT 50")
                    return self._json([row_to_dict(r, ("stats", "topics")) for r in rows])
                if path == "/api/config":
                    return self._json(rt.cfg.public_view())
                if path == "/api/models":
                    rt.training.ensure_production_registered()
                    return self._json({"versions": [dict(r) for r in rt.db.all(
                        "SELECT * FROM model_versions ORDER BY id DESC")],
                        "datasets": [row_to_dict(r, ("quality",)) for r in rt.db.all(
                            "SELECT * FROM training_datasets ORDER BY id DESC")],
                        "training_enabled": rt.cfg.get("training.enabled")})
                if path == "/api/conversation":
                    cid = qs.get("id", "")
                    rows = rt.db.all("SELECT role, content, retrieved, created_at FROM messages "
                                     "WHERE conversation_id=? ORDER BY id", (cid,))
                    return self._json([row_to_dict(r, ("retrieved",)) for r in rows])
                return self._json({"error": "not found"}, 404)
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}, 500)

        # ------------------------------------------------ POST
        def do_POST(self):
            if not self._host_ok() or self.headers.get("X-NIMRAS") != "1":
                return self._send(403, b"forbidden", "text/plain")
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/chat":
                    msg = str(body.get("message", "")).strip()
                    if not msg:
                        return self._json({"error": "empty message"}, 400)
                    return self._json(rt.chat.answer(msg, body.get("conversation_id")))
                if path == "/api/learn/run":
                    topics = [t for t in (body.get("topics") or []) if isinstance(t, str) and t.strip()]
                    if topics:
                        for t in topics:
                            rt.pipeline.enqueue_topic(t, origin="user", priority=1.0)
                    rt.db.audit("user", "learning_session_requested", None, topics=topics)
                    rt.scheduler.run_now()
                    return self._json({"ok": True, "queued_topics": topics})
                if path == "/api/learn/stop":
                    rt.scheduler.session_stop.set()
                    return self._json({"ok": True})
                if path == "/api/scheduler/pause":
                    rt.scheduler.pause()
                    return self._json({"ok": True})
                if path == "/api/scheduler/resume":
                    rt.scheduler.resume()
                    return self._json({"ok": True})
                if path == "/api/topics":
                    added = rt.pipeline.enqueue_topic(str(body.get("name", "")), origin="user",
                                                      layer=body.get("layer", "technical"),
                                                      priority=float(body.get("priority", 0.9)),
                                                      deep_study=bool(body.get("deep_study")))
                    rt.db.audit("user", "topic_added", body.get("name"))
                    return self._json({"ok": True, "added": added})
                m = re.fullmatch(r"/api/topics/(\d+)/(park|requeue)", path)
                if m:
                    st = "parked" if m.group(2) == "park" else "queued"
                    rt.db.execute("UPDATE topics SET status=?, next_due_at=NULL WHERE id=?", (st, int(m.group(1))))
                    rt.db.audit("user", f"topic_{m.group(2)}", f"topic:{m.group(1)}")
                    return self._json({"ok": True})
                if path == "/api/memory":
                    mid = rt.chat.propose_memory(str(body.get("key", "note")), str(body.get("value", "")))
                    return self._json({"ok": True, "id": mid})
                m = re.fullmatch(r"/api/memory/(\d+)/(approve|delete)", path)
                if m:
                    mid = int(m.group(1))
                    if m.group(2) == "approve":
                        rt.db.execute("UPDATE personal_memory SET approved=1 WHERE id=?", (mid,))
                    else:
                        rt.db.execute("DELETE FROM personal_memory WHERE id=?", (mid,))
                    rt.db.audit("user", f"personal_memory_{m.group(2)}", f"memory:{mid}")
                    return self._json({"ok": True})
                m = re.fullmatch(r"/api/gaps/(\d+)/dismiss", path)
                if m:
                    rt.db.execute("UPDATE knowledge_gaps SET status='dismissed' WHERE id=?", (int(m.group(1)),))
                    return self._json({"ok": True})
                if path == "/api/eval/run":
                    threading.Thread(target=rt.evaluator.run, daemon=True, name="nimras-eval").start()
                    rt.db.audit("user", "evaluation_requested")
                    return self._json({"ok": True, "started": True})
                if path == "/api/backup":
                    return self._json(rt.backup())
                if path == "/api/settings":
                    return self._json(apply_settings(rt, body))
                if path == "/api/training/dataset":
                    return self._json(rt.training.build_dataset())
                if path.startswith("/api/voice/"):
                    from .voice.api import voice_post
                    return self._json(*voice_post(rt, path, body))
                return self._json({"error": "not found"}, 404)
            except PermissionError as e:
                return self._json({"error": str(e)}, 403)
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}, 500)

    return Handler


# Only these settings are editable from the UI. Security-relevant ones
# (sandbox mode, allow/deny lists, training auto-promotion) stay file-only.
EDITABLE_SETTINGS = {
    "internet.enabled": bool, "scheduler.enabled": bool, "scheduler.interval_minutes": int,
    "scheduler.max_topics_per_session": int, "scheduler.max_fetches_per_session": int,
    "scheduler.max_llm_calls_per_session": int, "scheduler.max_session_minutes": int,
    "scheduler.defer_while_user_active_seconds": int, "self_improvement.enabled": bool,
    "ollama.chat_model": str, "ollama.analysis_model": str, "ollama.embed_model": str,
    "voice.ollama_model": str, "voice.store_transcripts": bool, "voice.vad.endpoint_ms": int,
    "voice.vad.start_ratio": float, "voice.vad.min_level": float, "voice.vad.max_utterance_s": int,
    "voice.vad.barge_in_ratio": float, "voice.vad.barge_in_ms": int, "voice.tts.speed": float,
}


def apply_settings(rt, body: dict) -> dict:
    changed = {}
    for key, val in body.items():
        typ = EDITABLE_SETTINGS.get(key)
        if typ is None:
            raise PermissionError(f"setting {key!r} cannot be changed from the UI (edit nimras.config.json)")
        val = typ(val) if typ is not bool else bool(val)
        if typ is float and not (0.0 <= val <= 100.0):
            raise ValueError(f"{key} out of range")
        if typ is int and not (0 <= val <= 100_000):
            raise ValueError(f"{key} out of range")
        old = rt.cfg.get(key)
        rt.cfg.set(key, val)
        changed[key] = {"from": old, "to": val}
    if changed:
        rt.cfg.save()
        rt.db.audit("user", "settings_changed", None, **{k.replace(".", "_"): v for k, v in changed.items()})
    return {"ok": True, "changed": changed}


def serve(rt, host: str | None = None, port: int | None = None) -> None:
    host = host or rt.cfg.get("server.host", "127.0.0.1")
    port = port or rt.cfg.get("server.port", 8765)
    httpd = ThreadingHTTPServer((host, port), make_handler(rt))
    print(f"N.I.M.R.A.S. Learning Centre: http://{'127.0.0.1' if host in ('0.0.0.0', '') else host}:{port}")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
