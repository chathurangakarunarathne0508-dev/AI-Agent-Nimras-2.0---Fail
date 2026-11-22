"""Continuous learning scheduler: runs learning sessions in the background,
within configured time, resource, and courtesy limits (quiet hours, user activity)."""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime

from .ollama_client import LLMBudget


class LearningScheduler:
    def __init__(self, runtime):
        self.rt = runtime
        self.cfg, self.db = runtime.cfg, runtime.db
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self.session_stop = threading.Event()
        self._run_lock = threading.Lock()
        self.state = "idle"          # idle|learning|evaluating|paused|waiting|stopped
        self.state_detail = ""
        self.current_session_id: int | None = None
        self.current_topic: str | None = None

    # ------------------------------------------------------------ control
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._shutdown.clear()
        if self.db.kv_get("next_session_at") is None:
            self.db.kv_set("next_session_at", time.time() + self.cfg.get("scheduler.initial_delay_seconds", 30))
        self._thread = threading.Thread(target=self._loop, name="nimras-learning", daemon=True)
        self._thread.start()
        self.db.audit("scheduler", "scheduler_started")

    def shutdown(self) -> None:
        self._shutdown.set()
        self.session_stop.set()
        self._wake.set()

    @property
    def paused(self) -> bool:
        return bool(self.db.kv_get("scheduler_paused", False))

    def pause(self) -> None:
        self.db.kv_set("scheduler_paused", True)
        self.session_stop.set()
        self.db.audit("user", "scheduler_paused")

    def resume(self) -> None:
        self.db.kv_set("scheduler_paused", False)
        self._wake.set()
        self.db.audit("user", "scheduler_resumed")

    def run_now(self) -> None:
        self.db.kv_set("next_session_at", time.time())
        self.db.kv_set("force_next_session", True)
        self._wake.set()

    def next_session_at(self) -> float | None:
        return self.db.kv_get("next_session_at")

    # ------------------------------------------------------------ gating
    def _blocked_reason(self, forced: bool) -> str | None:
        if not self.cfg.get("scheduler.enabled", True) and not forced:
            return "scheduler disabled in settings"
        if self.paused:
            return "paused by user"
        if not self.cfg.get("internet.enabled", True):
            return "internet access disabled in settings"
        # Model readiness applies even to forced runs: a forced request waits for the model.
        health = self.rt.ollama.health()
        if not health["reachable"]:
            return "Ollama is not reachable"
        if not health.get("analysis_model_ready"):
            return f"waiting for model '{health.get('analysis_model')}' (ollama pull {health.get('analysis_model')})"
        if forced:
            return None
        hour = datetime.now().hour
        for start, end in self.cfg.get("scheduler.quiet_hours", []):
            if (start <= hour < end) if start <= end else (hour >= start or hour < end):
                return f"quiet hours ({start}:00-{end}:00)"
        idle_need = self.cfg.get("scheduler.defer_while_user_active_seconds", 120)
        last = self.db.kv_get("user_last_active", 0) or 0
        if idle_need and time.time() - last < idle_need:
            return "user is active - deferring to keep the machine responsive"
        return None

    def _loop(self) -> None:
        while not self._shutdown.is_set():
            due = self.next_session_at() or 0
            wait = due - time.time()
            if wait > 0:
                self.state, self.state_detail = ("paused", "paused by user") if self.paused else ("idle", "")
                self._wake.wait(timeout=min(wait, 60))
                self._wake.clear()
                continue
            forced = bool(self.db.kv_get("force_next_session", False))
            reason = self._blocked_reason(forced)
            if reason:
                self.state, self.state_detail = ("paused" if self.paused else "waiting"), reason
                # Re-check soon while only waiting for a model download to finish.
                self._wake.wait(timeout=15 if "model" in reason else 60)
                self._wake.clear()
                continue
            self.db.kv_set("force_next_session", False)
            try:
                self.run_session(trigger="manual" if forced else "schedule")
            except Exception:
                self.db.audit("scheduler", "session_crashed", None, error=traceback.format_exc()[-1500:])
            interval = self.cfg.get("scheduler.interval_minutes", 60) * 60
            self.db.kv_set("next_session_at", time.time() + interval)

    # ------------------------------------------------------------ session
    def run_session(self, trigger: str = "manual", topic_names: list[str] | None = None) -> dict:
        from .learning.pipeline import LearningContext, SessionLimit  # local import avoids a cycle

        # Preflight: never start (and burn topic retry attempts) without a usable model.
        health = self.rt.ollama.health()
        if not health["reachable"] or not health.get("analysis_model_ready"):
            reason = health.get("error") or (f"analysis model '{health.get('analysis_model')}' is not pulled - run: "
                                             f"ollama pull {health.get('analysis_model')}")
            self.db.audit("scheduler", "session_not_started", None, reason=reason)
            return {"status": "not_started", "error": reason}
        if not self._run_lock.acquire(blocking=False):
            return {"error": "a learning session is already running"}
        try:
            self.session_stop.clear()
            p = self.rt.pipeline
            sid = self.db.insert("learning_sessions", {"trigger": trigger, "status": "running",
                                                       "started_at": time.time()})
            self.current_session_id = sid
            self.state, self.state_detail = "learning", "starting"
            self.db.audit("scheduler", "learning_session_started", f"session:{sid}", trigger=trigger)
            ctx = LearningContext(
                db=self.db, session_id=sid,
                max_fetches=self.cfg.get("scheduler.max_fetches_per_session", 24),
                deadline=time.time() + self.cfg.get("scheduler.max_session_minutes", 25) * 60,
                stop=self.session_stop,
                llm_budget=LLMBudget(self.cfg.get("scheduler.max_llm_calls_per_session", 80)),
            )
            self.rt.ollama.set_budget(ctx.llm_budget)
            outcomes, status, error, queue_info = [], "completed", None, {}
            n_sessions = self.db.scalar("SELECT COUNT(*) FROM learning_sessions")
            try:
                if topic_names:
                    for name in topic_names:
                        p.enqueue_topic(name, origin="user", priority=1.0)
                queue_info = p.refresh_queue()
                every = self.cfg.get("scheduler.discovery_every_n_sessions", 2)
                if every and n_sessions % every == 1 and not topic_names:
                    self.state_detail = "discovering emerging technologies"
                    p.discover_emerging(ctx)
                topics = p.next_topics(self.cfg.get("scheduler.max_topics_per_session", 3))
                if topic_names:
                    wanted = {t.lower() for t in topic_names}
                    topics = [dict(r) for r in self.db.all("SELECT * FROM topics WHERE status='queued'")
                              if r["name"].lower() in wanted] or topics
                for t in topics:
                    self.current_topic = t["name"]
                    self.state_detail = f"researching: {t['name']}"
                    outcomes.append(p.learn_topic(t, ctx))
            except SessionLimit as e:
                status, error = ("aborted" if str(e) == "stopped" else "completed"), str(e)
            except Exception as e:
                status, error = "failed", f"{type(e).__name__}: {e}"
                self.db.audit("scheduler", "session_error", f"session:{sid}", error=traceback.format_exc()[-1500:])
            finally:
                self.rt.ollama.set_budget(None)
                self.current_topic = None
            stats = ctx.stats | {"llm_calls": ctx.llm_budget.used, "fetches": ctx.fetches}
            self.db.update("learning_sessions", sid, {
                "status": status, "ended_at": time.time(), "error": error,
                "topics": json.dumps(outcomes), "stats": json.dumps(stats | {"queue": queue_info}),
            })
            self.db.audit("scheduler", "learning_session_finished", f"session:{sid}", status=status, **stats)

            # Periodic self-evaluation and backups ride on the session cadence.
            if self.cfg.get("self_improvement.enabled") and status != "aborted":
                every = self.cfg.get("scheduler.evaluation_every_n_sessions", 3)
                if every and n_sessions % every == 0:
                    self.state, self.state_detail = "evaluating", "self-evaluation"
                    try:
                        self.rt.evaluator.run()
                    except Exception:
                        self.db.audit("evaluator", "evaluation_failed", None, error=traceback.format_exc()[-1500:])
            last_bk = self.db.kv_get("last_backup_at", 0) or 0
            if time.time() - last_bk > self.cfg.get("scheduler.backup_every_hours", 24) * 3600:
                try:
                    self.rt.backup()
                except Exception:
                    self.db.audit("system", "backup_failed", None, error=traceback.format_exc()[-800:])
            # Voice engine: watch speech-model release feeds daily (recommend only, never install).
            if self.cfg.get("voice.enabled", True) and status != "aborted" and \
                    time.time() - (self.db.kv_get("voice_updates_checked_at", 0) or 0) > 86400:
                try:
                    from .voice.registry import VoiceModelRegistry
                    VoiceModelRegistry(self.rt).check_updates()
                except Exception:
                    self.db.audit("voice", "voice_update_check_failed", None, error=traceback.format_exc()[-600:])
            self.state, self.state_detail = "idle", ""
            self.current_session_id = None
            return {"session_id": sid, "status": status, "error": error, "stats": stats, "topics": outcomes}
        finally:
            self._run_lock.release()
