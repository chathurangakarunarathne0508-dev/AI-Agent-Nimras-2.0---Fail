"""The autonomous learning pipeline.

    INTERNET DISCOVERY -> RESEARCH -> INFORMATION EXTRACTION -> ANALYSIS
    -> SOURCE VERIFICATION -> KNOWLEDGE ACQUISITION -> LONG-TERM STORAGE
    (-> PERFORMANCE EVALUATION -> CONTINUOUS IMPROVEMENT in evaluation.py)

Each run is bounded by a LearningContext (fetch budget, LLM-call budget,
wall-clock deadline, stop flag), and every step leaves a research_attempts row
so failures and strategy effectiveness are measurable.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ..config import Config
from ..db import Database, row_to_dict
from ..knowledge import GENERAL_CATEGORIES, TECHNICAL_CATEGORIES, KnowledgeStore, cosine
from ..ollama_client import BudgetExceeded, LLMBudget, OllamaClient, OllamaError
from ..prompts import PromptRegistry
from ..db import unpack_vec
from ..security import UNTRUSTED_POLICY, fence_untrusted, neutralize_injection, scan_for_injection
from .credibility import prior_credibility, registrable
from .discovery import Discoverer, SearchResult
from .fetcher import Fetcher, FetchError


class SessionLimit(RuntimeError):
    pass


@dataclass
class LearningContext:
    db: Database
    session_id: int | None
    max_fetches: int
    deadline: float
    stop: threading.Event
    llm_budget: LLMBudget
    fetches: int = 0
    stats: dict = field(default_factory=lambda: {
        "searches": 0, "search_failures": 0, "fetched": 0, "fetch_failures": 0, "duplicates": 0,
        "irrelevant": 0, "injections": 0, "sources_stored": 0, "claims_added": 0, "claims_corroborated": 0,
        "contradictions": 0, "entries_created": 0, "entries_updated": 0, "topics_discovered": 0,
        "deep_studies": 0, "examples_tested": 0,
    })
    current_topic_id: int | None = None

    def check(self) -> None:
        if self.stop.is_set():
            raise SessionLimit("stopped")
        if time.time() > self.deadline:
            raise SessionLimit("session time limit reached")

    def take_fetch(self) -> None:
        self.check()
        if self.fetches >= self.max_fetches:
            raise SessionLimit("fetch budget exhausted")
        self.fetches += 1

    def log(self, stage: str, ok: bool, provider: str | None = None, strategy: str | None = None,
            query: str | None = None, url: str | None = None, yield_count: int = 0,
            error: str | None = None) -> None:
        if stage == "search":
            self.stats["searches"] += 1
            if not ok:
                self.stats["search_failures"] += 1
        self.db.insert("research_attempts", {
            "session_id": self.session_id, "topic_id": self.current_topic_id, "stage": stage,
            "provider": provider, "strategy": strategy, "query": query, "url": url, "ok": int(ok),
            "yield_count": yield_count, "error": error, "created_at": time.time(),
        })


def _valid_category(layer: str, cat: str | None) -> str:
    cat = (cat or "").strip().lower().replace(" ", "_").replace("/", "_")
    allowed = TECHNICAL_CATEGORIES if layer == "technical" else GENERAL_CATEGORIES
    return cat if cat in allowed else ("software_development" if layer == "technical" else "other")


class LearningPipeline:
    def __init__(self, cfg: Config, db: Database, ollama: OllamaClient, store: KnowledgeStore,
                 prompts: PromptRegistry, fetcher: Fetcher | None = None, discoverer: Discoverer | None = None):
        self.cfg, self.db, self.ollama, self.store, self.prompts = cfg, db, ollama, store, prompts
        self.fetcher = fetcher or Fetcher(cfg, ollama)
        self.discoverer = discoverer or Discoverer(cfg, db, self.fetcher, ollama)
        self.agent = None  # set by runtime (ResearchAgent) to enable deep studies

    # ================================================================ topics
    def seed_topics(self) -> int:
        n = 0
        for layer, names in self.cfg.get("seed_topics", {}).items():
            for name in names:
                n += self.enqueue_topic(name, origin="seed", layer=layer, priority=0.5)
        return n

    def enqueue_topic(self, name: str, origin: str, layer: str = "technical", category: str = "uncategorized",
                      priority: float = 0.5, deep_study: bool = False, note: str | None = None) -> int:
        """Insert a topic, or bump the priority of an existing one (repeated mentions = more useful)."""
        name = " ".join(name.split())[:200]
        if len(name) < 3:
            return 0
        existing = self.db.one("SELECT id, status, priority FROM topics WHERE name=?", (name,))
        if existing:
            bump = 0.05 if origin in ("discovered", "emerging") else 0.15
            self.db.execute("UPDATE topics SET priority=MIN(1.0, priority+?) WHERE id=?", (bump, existing["id"]))
            if origin in ("gap", "user") and existing["status"] in ("learned", "failed", "parked"):
                self.db.execute("UPDATE topics SET status='queued', next_due_at=NULL, origin=? WHERE id=?",
                                (origin, existing["id"]))
            return 0
        cap = self.cfg.get("learning.max_queued_topics", 400)
        if origin in ("discovered", "emerging") and \
                self.db.scalar("SELECT COUNT(*) FROM topics WHERE status='queued'") >= cap:
            return 0
        self.db.insert("topics", {"name": name, "layer": layer, "category": category, "priority": priority,
                                  "origin": origin, "status": "queued", "deep_study": int(deep_study),
                                  "created_at": time.time(), "note": note})
        return 1

    def refresh_queue(self) -> dict:
        """Gap detection + revisits: turn open knowledge gaps and stale knowledge into queued topics."""
        now = time.time()
        promoted_gaps = 0
        for g in self.db.all("SELECT * FROM knowledge_gaps WHERE status='open' ORDER BY id LIMIT 20"):
            self.enqueue_topic(g["description"], origin="gap", priority=0.9, note=f"gap:{g['id']}")
            t = self.db.one("SELECT id FROM topics WHERE name=?", (" ".join(g["description"].split())[:200],))
            self.db.update("knowledge_gaps", g["id"], {"status": "queued", "topic_id": t["id"] if t else None})
            promoted_gaps += 1
        revisit_s = self.cfg.get("scheduler.revisit_after_days", 30) * 86400
        due = self.db.all("SELECT id FROM topics WHERE status='learned' AND next_due_at IS NOT NULL AND next_due_at<?",
                          (now,))
        for t in due:
            self.db.execute("UPDATE topics SET status='queued', origin='revisit', priority=MAX(priority,0.6) "
                            "WHERE id=?", (t["id"],))
        # Entries that stayed provisional for a long time are themselves gaps.
        stale = self.db.all("SELECT topic FROM knowledge_entries WHERE status IN ('provisional','contested') "
                            "AND updated_at<? LIMIT 10", (now - revisit_s / 3,))
        for e in stale:
            self.db.execute("UPDATE topics SET status='queued', priority=MAX(priority,0.55) "
                            "WHERE name=? AND status IN ('learned','failed')", (e["topic"],))
        # Failed topics with backoff elapsed get retried (max 3 attempts).
        self.db.execute("UPDATE topics SET status='queued' WHERE status='failed' AND attempts<3 AND next_due_at<?",
                        (now,))
        return {"gaps_promoted": promoted_gaps, "revisits": len(due)}

    def next_topics(self, n: int) -> list[dict]:
        rows = self.db.all(
            "SELECT * FROM topics WHERE status='queued' AND (next_due_at IS NULL OR next_due_at<=?) "
            "ORDER BY priority DESC, CASE origin WHEN 'gap' THEN 0 WHEN 'user' THEN 1 ELSE 2 END, created_at "
            "LIMIT ?", (time.time(), n))
        return [dict(r) for r in rows]

    # ================================================================ discovery
    def discover_emerging(self, ctx: LearningContext) -> int:
        """Read feeds, ask the model which specific new technologies are worth learning."""
        items = self.discoverer.read_feeds(log=ctx.log)
        seen: list[str] = self.db.kv_get("seen_feed_items", [])
        seen_set = set(seen)
        fresh = []
        for it in items:
            h = hashlib.sha1(it.url.encode()).hexdigest()[:16]
            if h not in seen_set:
                fresh.append(it)
                seen.append(h)
                seen_set.add(h)
        self.db.kv_set("seen_feed_items", seen[-5000:])
        if not fresh:
            return 0
        headlines = "\n".join(f"- {it.title} :: {it.snippet[:160]}" for it in fresh[:60])
        known = [r["name"] for r in self.db.all("SELECT name FROM topics ORDER BY id DESC LIMIT 150")]
        try:
            ctx.check()
            data = self.ollama.chat_json(
                UNTRUSTED_POLICY + "\n\n"
                "From these news/paper headlines, pick up to 8 SPECIFIC emerging technologies, tools, frameworks, "
                "models, vulnerabilities classes or research ideas worth adding to a technical knowledge base. "
                "Skip vague or marketing items and anything in the KNOWN list. Return JSON: "
                '{"topics": [{"name": str, "layer": "technical|general", "category": str, '
                '"is_software": bool, "priority": 0-1, "why": str}]}',
                f"KNOWN: {json.dumps(known)}\n\n" + fence_untrusted(headlines, "rss-feeds"),
            )
        except (OllamaError, BudgetExceeded) as e:
            ctx.log(stage="discover", ok=False, provider="ollama", error=str(e)[:300])
            return 0
        added = 0
        for t in (data.get("topics") or [])[:8]:
            if not isinstance(t, dict) or not t.get("name"):
                continue
            layer = t.get("layer") if t.get("layer") in ("technical", "general") else "technical"
            pr = float(t.get("priority") or 0.6)
            added += self.enqueue_topic(t["name"], origin="emerging", layer=layer,
                                        category=_valid_category(layer, t.get("category")),
                                        priority=min(0.85, max(0.4, pr)), deep_study=bool(t.get("is_software")),
                                        note=str(t.get("why", ""))[:300])
        ctx.stats["topics_discovered"] += added
        ctx.log(stage="discover", ok=True, provider="rss+ollama", yield_count=added)
        self.db.audit("pipeline", "emerging_topics_discovered", None, count=added)
        return added

    # ================================================================ per-topic
    def learn_topic(self, topic: dict, ctx: LearningContext) -> dict:
        ctx.current_topic_id = topic["id"]
        self.db.execute("UPDATE topics SET status='active', attempts=attempts+1 WHERE id=?", (topic["id"],))
        self.db.audit("pipeline", "topic_research_started", topic["name"], origin=topic["origin"])
        outcome = {"topic": topic["name"], "sources_used": 0, "entry_id": None, "error": None}
        try:
            ctx.check()
            candidates = self.discoverer.search_topic(topic["name"], log=ctx.log)
            picked = self._pick_candidates(candidates, per_topic=5)
            results = []
            for sr, template in picked:
                try:
                    res = self.process_source(sr, topic, ctx)
                except SessionLimit:
                    raise
                except BudgetExceeded as e:
                    raise SessionLimit(str(e))
                except Exception as e:  # one bad source never aborts a topic
                    ctx.log(stage="analyze", ok=False, url=sr.url, error=f"{type(e).__name__}: {e}"[:300])
                    res = {"relevant": False, "claims_linked": 0}
                results.append(res)
                reward = 1.0 if res.get("claims_linked") else (0.3 if res.get("relevant") else 0.0)
                self.discoverer.strategy.record("provider", sr.provider, reward)
                self.discoverer.strategy.record("query_template", template, reward)
            used = [r for r in results if r.get("claims_linked")]
            outcome["sources_used"] = len(used)
            entry = self.store.find_entry_for_topic(topic["name"]) or (
                {"id": used[0]["entry_id"]} if used and used[0].get("entry_id") else None)
            if entry:
                outcome["entry_id"] = entry["id"]
                outcome["verification"] = self.store.recompute_entry(entry["id"], "new evidence")
                self._synthesize(entry["id"])
                self.store.index_entry_text(entry["id"])
            # Queue what we learned about next.
            for r in used:
                for rel in (r.get("related_topics") or [])[: self.cfg.get("learning.max_new_topics_per_source", 3)]:
                    ctx.stats["topics_discovered"] += self.enqueue_topic(
                        rel, origin="discovered", layer=r.get("layer", "technical"), priority=0.45)
            # Deep study for software: docs, comparison, example, sandbox test.
            is_soft = topic.get("deep_study") or any(r.get("is_software") for r in used)
            if is_soft and entry and self.agent:
                ctx.check()
                self.agent.deep_study(entry["id"], topic["name"], ctx)
            self._finish_topic(topic, success=bool(used))
        except (SessionLimit, BudgetExceeded) as e:
            outcome["error"] = str(e)
            # Out of budget mid-topic: requeue it, keep everything learned so far.
            self.db.execute("UPDATE topics SET status='queued' WHERE id=? AND status='active'", (topic["id"],))
            raise SessionLimit(str(e)) from e
        finally:
            ctx.current_topic_id = None
        return outcome

    def _finish_topic(self, topic: dict, success: bool) -> None:
        now = time.time()
        if success:
            due = now + self.cfg.get("scheduler.revisit_after_days", 30) * 86400
            self.db.execute("UPDATE topics SET status='learned', last_researched_at=?, next_due_at=? WHERE id=?",
                            (now, due, topic["id"]))
            self.db.execute("UPDATE knowledge_gaps SET status='filled', resolved_at=? WHERE topic_id=? "
                            "AND status='queued'", (now, topic["id"]))
        else:
            attempts = self.db.scalar("SELECT attempts FROM topics WHERE id=?", (topic["id"],))
            backoff = 6 * 3600 * (2 ** max(0, attempts - 1))
            status = "failed" if attempts < 3 else "parked"
            self.db.execute("UPDATE topics SET status=?, last_researched_at=?, next_due_at=? WHERE id=?",
                            (status, now, now + backoff, topic["id"]))
            self.db.audit("pipeline", "topic_research_failed", topic["name"], attempts=attempts, status=status)

    def _pick_candidates(self, candidates: list[tuple[SearchResult, str]], per_topic: int):
        """Prefer credible sources and spread across independent domains (needed for verification)."""
        scored = sorted(candidates, key=lambda c: -prior_credibility(c[0].url, c[0].kind))
        picked, domains = [], {}
        for sr, t in scored:
            reg = registrable((urlparse(sr.url).hostname or "").lower())
            if domains.get(reg, 0) >= 2:
                continue
            ok, _ = self.fetcher.url_ok(sr.url) if hasattr(self.fetcher, "url_ok") else (True, "")
            if not ok:
                continue
            domains[reg] = domains.get(reg, 0) + 1
            picked.append((sr, t))
            if len(picked) >= per_topic:
                break
        return picked

    # ================================================================ per-source
    def process_source(self, sr: SearchResult, topic: dict, ctx: LearningContext) -> dict:
        # ---- RESEARCH: retrieve
        recent = self.db.one("SELECT id, retrieved_at, status FROM sources WHERE url=?", (sr.url,))
        if recent and recent["status"] == "analyzed" and \
                time.time() - recent["retrieved_at"] < self.cfg.get("scheduler.revisit_after_days", 30) * 86400:
            return {"relevant": True, "claims_linked": 0, "skipped": "recently analysed"}
        ctx.take_fetch()
        try:
            page = self.fetcher.fetch(sr.url)
        except FetchError as e:
            ctx.stats["fetch_failures"] += 1
            ctx.log(stage="fetch", ok=False, provider=sr.provider, url=sr.url, error=str(e)[:300])
            return {"relevant": False, "claims_linked": 0}
        ctx.stats["fetched"] += 1
        text = page.text.strip()
        if sr.kind == "paper" and len(text) < 400 and sr.snippet:
            text = f"{sr.title}\n\n{sr.snippet}\n\n{text}"
        if len(text) < 300:
            ctx.log(stage="fetch", ok=False, provider=sr.provider, url=sr.url, error="too little readable text")
            return {"relevant": False, "claims_linked": 0}
        ctx.log(stage="fetch", ok=True, provider=sr.provider, url=sr.url, yield_count=len(text))

        # ---- EXTRACTION: dedup + security scan
        dup = self.store.find_duplicate_source(text)
        if dup and dup["url"] != sr.url:
            ctx.stats["duplicates"] += 1
            ctx.log(stage="analyze", ok=True, url=sr.url, error=f"{dup['reason']} of {dup['url']}")
            return {"relevant": False, "claims_linked": 0, "duplicate_of": dup["url"]}
        inj = scan_for_injection(text)
        if inj:
            ctx.stats["injections"] += 1
            text = neutralize_injection(text)
            self.store.cred.record(sr.url, injections=1)
            self.db.audit("security", "prompt_injection_neutralized", sr.url, patterns=inj[:5])
        kind = sr.kind
        host = (urlparse(page.final_url).hostname or "")
        if kind == "web" and host.startswith(("docs.", "developer.", "learn.")):
            kind = "docs"
        published = page.published or sr.published
        credibility = self.store.cred.score(sr.url, kind, injection_hits=len(inj), has_date=bool(published))
        source_id = self.store.add_source(sr.url, page.title or sr.title, text, kind, published, credibility,
                                          inj, topic["id"])
        ctx.stats["sources_stored"] += 1

        # ---- ANALYSIS (Ollama)
        analysis = self._analyze(text, topic, sr.url)
        if analysis.get("injection_attempt") and not inj:
            credibility = max(0.05, credibility - 0.2)
            self.db.update("sources", source_id, {"credibility": credibility})
            self.db.audit("security", "model_flagged_injection", sr.url)
        if not analysis.get("relevant"):
            ctx.stats["irrelevant"] += 1
            self.db.update("sources", source_id, {"status": "rejected", "error": "not relevant to topic"})
            ctx.log(stage="analyze", ok=True, url=sr.url, error="irrelevant")
            return {"relevant": False, "claims_linked": 0}
        quality = float(analysis.get("quality") or 0.5)
        if quality < 0.5:  # thin / low-effort content lowers trust in this copy
            credibility = round(max(0.05, credibility - 0.1 * (0.5 - quality) * 2), 3)
        self.db.update("sources", source_id, {"status": "analyzed", "credibility": credibility,
                                              "published_at": published or analysis.get("published")})
        self.store.archive_source(source_id, text)

        # ---- KNOWLEDGE ACQUISITION: find or create the entry
        layer = analysis.get("layer") if analysis.get("layer") in ("technical", "general") else topic["layer"]
        category = _valid_category(layer, analysis.get("category") or topic.get("category"))
        entry = self.store.find_entry_for_topic(topic["name"])
        if not entry:
            related = self.store.find_related_entries(f"{analysis.get('title', '')}\n{analysis.get('summary', '')}", k=1)
            if related and related[0][1] >= self.cfg.get("learning.same_entry_similarity", 0.8):
                entry = related[0][0]  # same concept under another name: enrich, don't duplicate
        if entry:
            entry_id = entry["id"]
            ctx.stats["entries_updated"] += 1
        else:
            entry_id = self.store.create_entry(layer, category, topic["name"],
                                               str(analysis.get("title") or topic["name"])[:200],
                                               str(analysis.get("summary") or "")[:2000])
            ctx.stats["entries_created"] += 1

        # ---- COMPARE with existing knowledge + SOURCE VERIFICATION
        linked = self._integrate_claims(entry_id, source_id, analysis.get("claims") or [], ctx)
        ctx.log(stage="verify", ok=True, url=sr.url, yield_count=linked)
        return {"relevant": True, "claims_linked": linked, "entry_id": entry_id, "layer": layer,
                "related_topics": analysis.get("related_topics") or [],
                "is_software": bool(analysis.get("is_software"))}

    def _analyze(self, text: str, topic: dict, url: str) -> dict:
        cats = TECHNICAL_CATEGORIES + GENERAL_CATEGORIES
        limit = self.cfg.get("learning.max_chars_analyzed_per_source", 9000)
        data = self.ollama.chat_json(
            self.prompts.get("analyze_source", categories=", ".join(cats)),
            f"RESEARCH TOPIC: {topic['name']}\nSOURCE URL: {url}\n\n" + fence_untrusted(text[:limit], url),
        )
        if not isinstance(data, dict):
            raise OllamaError("analysis was not a JSON object")
        claims = [str(c).strip() for c in (data.get("claims") or []) if isinstance(c, (str, int, float))]
        data["claims"] = [c[:500] for c in claims if 15 <= len(c)][:8]
        data["related_topics"] = [str(t)[:120] for t in (data.get("related_topics") or []) if t][:5]
        return data

    def _integrate_claims(self, entry_id: int, source_id: int, new_claims: list[str], ctx: LearningContext) -> int:
        if not new_claims:
            return 0
        existing = self.store.entry_claims(entry_id)
        relations: dict[int, tuple[str, int | None, str]] = {}
        if existing:
            relations = self._compare(new_claims, existing)
        linked = 0
        for i, text in enumerate(new_claims):
            rel, j, note = relations.get(i, ("new", None, ""))
            if rel == "same" and j is not None and 0 <= j < len(existing):
                self.store.link_claim(existing[j]["id"], source_id, "supports")
                ctx.stats["claims_corroborated"] += 1
            elif rel == "contradicts" and j is not None and 0 <= j < len(existing):
                cid = self.store.add_claim(entry_id, text)
                self.store.link_claim(cid, source_id, "supports")
                self.store.link_claim(existing[j]["id"], source_id, "contradicts")
                self.store.record_contradiction(existing[j]["id"], source_id, text, note)
                ctx.stats["contradictions"] += 1
                ctx.stats["claims_added"] += 1
            else:
                cid = self.store.add_claim(entry_id, text)
                self.store.link_claim(cid, source_id, "supports")
                ctx.stats["claims_added"] += 1
            linked += 1
        return linked

    def _compare(self, new_claims: list[str], existing: list[dict]) -> dict[int, tuple[str, int | None, str]]:
        """LLM comparison; embedding similarity as a fallback for 'same'."""
        ex = existing[-40:]
        offset = len(existing) - len(ex)
        try:
            data = self.ollama.chat_json(
                self.prompts.get("compare_claims"),
                "EXISTING:\n" + "\n".join(f"{j}. {c['text']}" for j, c in enumerate(ex)) + "\n\n" +
                fence_untrusted("\n".join(f"{i}. {c}" for i, c in enumerate(new_claims)), "new-source-claims"),
            )
            out = {}
            for r in data.get("results", []):
                try:
                    i = int(r["i"])
                    j = r.get("j")
                    j = int(j) + offset if j is not None else None
                    out[i] = (str(r.get("relation", "new")), j, str(r.get("note", ""))[:300])
                except (KeyError, TypeError, ValueError):
                    continue
            return out
        except BudgetExceeded:
            raise
        except OllamaError:
            vecs = self.store.embed(new_claims)
            out = {}
            ex_vecs = [unpack_vec(self.db.scalar("SELECT embedding FROM claims WHERE id=?", (c["id"],), None))
                       for c in existing]
            for i, v in enumerate(vecs):
                if v is None:
                    continue
                best = max(((cosine(v, ev), j) for j, ev in enumerate(ex_vecs) if ev), default=(0, None))
                if best[0] >= 0.9:
                    out[i] = ("same", best[1], "embedding match")
            return out

    def _synthesize(self, entry_id: int) -> None:
        claims = self.store.entry_claims(entry_id)
        if len(claims) < 2:
            return
        e = self.db.one("SELECT title, summary FROM knowledge_entries WHERE id=?", (entry_id,))
        listing = "\n".join(f"- [{c['status'].upper()}] {c['text']}" for c in claims[:40])
        try:
            data = self.ollama.chat_json(self.prompts.get("synthesize_entry"), f"TOPIC: {e['title']}\nCLAIMS:\n{listing}")
            summary = str(data.get("summary") or "").strip()
        except (OllamaError, AttributeError, TypeError):
            return  # keep the previous summary; BudgetExceeded propagates to end the session
        if summary and summary != e["summary"]:
            self.store.update_entry(entry_id, {"summary": summary[:3000]}, "re-synthesised from claims")
