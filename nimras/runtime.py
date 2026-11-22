"""Wires every subsystem together and exposes dashboard/library queries."""
from __future__ import annotations

import json
import time

from .backup import backup_all
from .chat import ChatService
from .config import Config
from .db import Database, row_to_dict
from .evaluation import SelfImprovementEngine
from .knowledge import KnowledgeStore
from .learning.agent import ResearchAgent
from .learning.pipeline import LearningPipeline
from .ollama_client import OllamaClient
from .prompts import PromptRegistry
from .sandbox import Sandbox
from .scheduler import LearningScheduler
from .training import TrainingManager


class Runtime:
    def __init__(self, cfg: Config | None = None, ollama: OllamaClient | None = None, fetcher=None,
                 discoverer=None):
        self.cfg = cfg or Config.load()
        self.db = Database(self.cfg.data_dir / "nimras.db")
        self.ollama = ollama or OllamaClient(self.cfg)
        self.prompts = PromptRegistry(self.db)
        self.store = KnowledgeStore(self.cfg, self.db, self.ollama)
        self.pipeline = LearningPipeline(self.cfg, self.db, self.ollama, self.store, self.prompts,
                                         fetcher=fetcher, discoverer=discoverer)
        self.sandbox = Sandbox(self.cfg)
        self.pipeline.agent = ResearchAgent(self.pipeline, self.sandbox)
        self.evaluator = SelfImprovementEngine(self.cfg, self.db, self.ollama, self.store, self.prompts)
        self.chat = ChatService(self.cfg, self.db, self.ollama, self.store, self.prompts,
                                gap_sink=self.evaluator.add_gap)
        self.training = TrainingManager(self.cfg, self.db, self.ollama, self.prompts)
        self.scheduler = LearningScheduler(self)
        if not self.db.kv_get("seeded"):
            n = self.pipeline.seed_topics()
            self.db.kv_set("seeded", True)
            self.db.audit("system", "seed_topics_loaded", None, count=n)
        # Sessions that were running when the process died are not "running" any more.
        self.db.execute("UPDATE learning_sessions SET status='aborted', ended_at=?, error='process restarted' "
                        "WHERE status='running'", (time.time(),))
        self.db.execute("UPDATE topics SET status='queued' WHERE status='active'")

    @property
    def voice(self):
        """Voice Engine, created on first use (it starts local speech servers)."""
        if getattr(self, "_voice", None) is None:
            from .voice.engine import VoiceEngine
            self._voice = VoiceEngine(self)
        return self._voice

    def backup(self) -> dict:
        return backup_all(self.cfg, self.db)

    # ================================================================ dashboard
    def dashboard(self) -> dict:
        db = self.db
        q = db.scalar
        sch = self.scheduler
        last_session = row_to_dict(db.one("SELECT * FROM learning_sessions ORDER BY id DESC LIMIT 1"),
                                   ("stats", "topics"))
        eval_rows = db.all("SELECT id, started_at, metrics, deltas FROM eval_runs WHERE kind='self_improvement' "
                           "AND ended_at IS NOT NULL ORDER BY id DESC LIMIT 12")
        evals = [row_to_dict(r, ("metrics", "deltas")) for r in eval_rows]
        week = time.time() - 7 * 86400
        return {
            "status": {
                "state": sch.state, "detail": sch.state_detail, "paused": sch.paused,
                "current_topic": sch.current_topic, "current_session_id": sch.current_session_id,
                "next_session_at": sch.next_session_at(),
                "internet_enabled": self.cfg.get("internet.enabled"),
                "scheduler_enabled": self.cfg.get("scheduler.enabled"),
                "ollama": self.ollama.health(),
                "sandbox": dict(zip(("available", "detail"), self.sandbox.available())),
                "providers": self.pipeline.discoverer.active_providers(),
                "last_backup_at": db.kv_get("last_backup_at"),
            },
            "counts": {
                "resources_collected": q("SELECT COUNT(*) FROM sources WHERE status IN ('analyzed','fetched')"),
                "sources_rejected": q("SELECT COUNT(*) FROM sources WHERE status='rejected'"),
                "archive_chunks": q("SELECT COUNT(*) FROM chunks WHERE layer='archive'"),
                "entries_total": q("SELECT COUNT(*) FROM knowledge_entries WHERE status!='merged'"),
                "entries_verified": q("SELECT COUNT(*) FROM knowledge_entries WHERE status='verified'"),
                "entries_provisional": q("SELECT COUNT(*) FROM knowledge_entries WHERE status='provisional'"),
                "entries_contested": q("SELECT COUNT(*) FROM knowledge_entries WHERE status='contested'"),
                "claims_verified": q("SELECT COUNT(*) FROM claims WHERE status='verified'"),
                "claims_total": q("SELECT COUNT(*) FROM claims WHERE status!='rejected'"),
                "topics_queued": q("SELECT COUNT(*) FROM topics WHERE status='queued'"),
                "topics_learned": q("SELECT COUNT(*) FROM topics WHERE status='learned'"),
                "sessions": q("SELECT COUNT(*) FROM learning_sessions"),
                "injections_blocked": q("SELECT COUNT(*) FROM audit_log WHERE action='prompt_injection_neutralized'"),
                "open_contradictions": q("SELECT COUNT(*) FROM contradictions WHERE resolution='open'"),
                "personal_pending": q("SELECT COUNT(*) FROM personal_memory WHERE approved=0"),
            },
            "active_topics": [dict(r) for r in db.all(
                "SELECT id, name, layer, category, priority, origin, status, attempts FROM topics "
                "WHERE status IN ('active','queued') ORDER BY status='active' DESC, priority DESC LIMIT 15")],
            "recent_discoveries": [dict(r) for r in db.all(
                "SELECT id, name, category, origin, priority, status, note, created_at FROM topics "
                "WHERE origin IN ('emerging','discovered') ORDER BY id DESC LIMIT 12")],
            "categories": [dict(r) for r in db.all(
                "SELECT layer, category, COUNT(*) AS n, SUM(status='verified') AS verified FROM knowledge_entries "
                "WHERE status!='merged' GROUP BY layer, category ORDER BY n DESC")],
            "source_reliability": [dict(r) for r in db.all(
                "SELECT s.domain, COUNT(*) AS n, ROUND(AVG(s.credibility),3) AS avg_credibility, "
                "COALESCE(d.supported,0) AS supported, COALESCE(d.contradicted,0) AS contradicted, "
                "COALESCE(d.injections,0) AS injections FROM sources s LEFT JOIN domain_reputation d "
                "ON d.domain=s.domain WHERE s.status='analyzed' GROUP BY s.domain ORDER BY n DESC LIMIT 20")],
            "sessions": [row_to_dict(r, ("stats",)) for r in db.all(
                "SELECT id, trigger, status, started_at, ended_at, stats, error FROM learning_sessions "
                "ORDER BY id DESC LIMIT 15")],
            "last_session": last_session,
            "failures": [dict(r) for r in db.all(
                "SELECT a.id, a.stage, a.provider, a.query, a.url, a.error, a.created_at, t.name AS topic "
                "FROM research_attempts a LEFT JOIN topics t ON t.id=a.topic_id WHERE a.ok=0 "
                "ORDER BY a.id DESC LIMIT 25")],
            "failed_topics": [dict(r) for r in db.all(
                "SELECT id, name, attempts, status, next_due_at FROM topics WHERE status IN ('failed','parked') "
                "ORDER BY last_researched_at DESC LIMIT 10")],
            "gaps": [dict(r) for r in db.all(
                "SELECT id, description, origin, status, created_at FROM knowledge_gaps "
                "ORDER BY status='filled', id DESC LIMIT 20")],
            "performance": {
                "evaluations": evals,
                "strategies": self.pipeline.discoverer.strategy.report(),
                "fetch_success_7d": q("SELECT ROUND(AVG(ok),3) FROM research_attempts WHERE stage='fetch' "
                                      "AND created_at>?", (week,), None),
                "repeated_mistakes": [dict(r) for r in db.all(
                    "SELECT signature, occurrences, last_seen FROM mistakes ORDER BY occurrences DESC LIMIT 8")],
                "prompts": [dict(r) for r in db.all(
                    "SELECT id, name, version, status, mean_score, samples, rationale FROM prompts "
                    "ORDER BY name, version DESC")],
            },
        }

    # ================================================================ library
    def library(self, query: str = "", layer: str = "", category: str = "", status: str = "",
                limit: int = 50, offset: int = 0) -> dict:
        if query.strip():
            hits = self.store.retrieve(query, k=limit)
            ids = []
            for h in hits:
                if h.get("entry_id") and h["entry_id"] not in ids:
                    ids.append(h["entry_id"])
            rows = [self.db.one("SELECT * FROM knowledge_entries WHERE id=?", (i,)) for i in ids]
            rows = [r for r in rows if r]
            archive = [h for h in hits if not h.get("entry_id")][:10]
        else:
            where, params = ["status!='merged'"], []
            for col, val in (("layer", layer), ("category", category), ("status", status)):
                if val:
                    where.append(f"{col}=?")
                    params.append(val)
            rows = self.db.all(f"SELECT * FROM knowledge_entries WHERE {' AND '.join(where)} "
                               "ORDER BY updated_at DESC LIMIT ? OFFSET ?", (*params, limit, offset))
            archive = []
        entries = []
        for r in rows:
            d = row_to_dict(r, ("details",))
            if (layer and d["layer"] != layer) or (category and d["category"] != category) or \
                    (status and d["status"] != status):
                continue
            d["n_sources"] = self.db.scalar(
                "SELECT COUNT(DISTINCT cs.source_id) FROM claims c JOIN claim_sources cs ON cs.claim_id=c.id "
                "WHERE c.entry_id=?", (d["id"],))
            entries.append(d)
        return {"entries": entries, "archive_hits": archive}

    def entry_detail(self, entry_id: int) -> dict | None:
        e = row_to_dict(self.db.one("SELECT * FROM knowledge_entries WHERE id=?", (entry_id,)), ("details",))
        if not e:
            return None
        claims = []
        for c in self.store.entry_claims(entry_id):
            c["sources"] = [dict(r) for r in self.db.all(
                "SELECT s.id, s.url, s.domain, s.credibility, cs.stance FROM claim_sources cs JOIN sources s "
                "ON s.id=cs.source_id WHERE cs.claim_id=?", (c["id"],))]
            claims.append(c)
        e["claims"] = claims
        e["sources"] = self.store.entry_sources(entry_id)
        e["versions"] = [dict(r) for r in self.db.all(
            "SELECT version, status, confidence, reason, changed_at FROM entry_versions WHERE entry_id=? "
            "ORDER BY version DESC", (entry_id,))]
        e["contradictions"] = [dict(r) for r in self.db.all(
            "SELECT x.*, s.url FROM contradictions x JOIN claims c ON c.id=x.claim_id "
            "LEFT JOIN sources s ON s.id=x.source_id WHERE c.entry_id=? ORDER BY x.id DESC", (entry_id,))]
        return e
