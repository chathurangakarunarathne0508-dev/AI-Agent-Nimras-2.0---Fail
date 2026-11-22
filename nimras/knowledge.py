"""Long-term knowledge: layered entries, claims with provenance, versions, RAG retrieval.

Layers:  personal (approved user memory, separate table)
         general   - science, business, education, engineering...
         technical - programming, OS, networking, security, AI/ML...
         archive   - raw text chunks of every retained internet source

Truth model: nothing from the internet is "fact" by default. A claim becomes
`verified` only when backed by enough independent, credible sources (or one
authoritative primary source) and is not credibly contradicted.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import time
from typing import Any

from .config import Config
from .db import Database, pack_vec, row_to_dict, unpack_vec
from .learning.credibility import CredibilityModel, domain_of, registrable

try:
    import numpy as np
except ImportError:  # numpy is optional; pure-python fallback below
    np = None

GENERAL_CATEGORIES = ["science", "technology", "education", "business", "engineering", "computing", "other"]
TECHNICAL_CATEGORIES = ["programming", "operating_systems", "networking", "cybersecurity",
                        "ai_ml", "software_development", "hardware", "tools_apis", "data"]


# ------------------------------------------------------------------ utilities
def content_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha256(norm.encode()).hexdigest()


def simhash(text: str, bits: int = 64) -> str:
    words = re.findall(r"\w+", text.lower())
    shingles = [" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))]
    v = [0] * bits
    for sh in shingles:
        h = int.from_bytes(hashlib.blake2b(sh.encode(), digest_size=8).digest(), "big")
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= 1 << i
    return f"{out:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=## )", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 1 <= size:
            cur = f"{cur}\n{p}" if cur else p
            continue
        if cur:
            chunks.append(cur)
        while len(p) > size:
            chunks.append(p[:size])
            p = p[size - overlap:]
        cur = (chunks[-1][-overlap:] + "\n" + p) if chunks and overlap else p
    if cur:
        chunks.append(cur)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorIndex:
    """In-memory cosine index over one table's embedding column, lazily rebuilt."""

    def __init__(self, db: Database, table: str, where: str = "1=1"):
        self.db, self.table, self.where = db, table, where
        self._ids: list[int] = []
        self._mat: Any = None
        self._rows: list[list[float]] = []
        self._dirty = True
        self._lock = threading.Lock()

    def invalidate(self) -> None:
        self._dirty = True

    def _load(self) -> None:
        rows = self.db.all(f"SELECT id, embedding FROM {self.table} WHERE embedding IS NOT NULL AND {self.where}")
        ids, vecs = [], []
        dim = None
        for r in rows:
            v = unpack_vec(r["embedding"])
            if dim is None:
                dim = len(v)
            if len(v) == dim:  # skip rows from a previous embedding model
                ids.append(r["id"])
                vecs.append(v)
        self._ids = ids
        if np is not None and vecs:
            m = np.asarray(vecs, dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._mat = m / norms
        else:
            self._mat, self._rows = None, vecs
        self._dirty = False

    def search(self, qvec: list[float], k: int) -> list[tuple[int, float]]:
        with self._lock:
            if self._dirty:
                self._load()
            if not self._ids:
                return []
            if self._mat is not None:
                q = np.asarray(qvec, dtype=np.float32)
                if q.shape[0] != self._mat.shape[1]:
                    return []
                q = q / (np.linalg.norm(q) or 1)
                sims = self._mat @ q
                top = np.argsort(-sims)[:k]
                return [(self._ids[i], float(sims[i])) for i in top]
            scored = [(i, cosine(qvec, v)) for i, v in zip(self._ids, self._rows)]
            return sorted(scored, key=lambda t: -t[1])[:k]


# ------------------------------------------------------------------ store
class KnowledgeStore:
    def __init__(self, cfg: Config, db: Database, ollama):
        self.cfg, self.db, self.ollama = cfg, db, ollama
        self.cred = CredibilityModel(db)
        self.entry_index = VectorIndex(db, "knowledge_entries", "status NOT IN ('merged')")
        self.chunk_index = VectorIndex(db, "chunks")
        self.claim_index = VectorIndex(db, "claims", "status != 'rejected'")

    # ------------------------------------------------ embeddings (fail-soft)
    def embed(self, texts: list[str]) -> list[list[float] | None]:
        try:
            return self.ollama.embed(texts)
        except Exception as e:
            self.db.kv_set("last_embed_error", str(e)[:300])
            return [None] * len(texts)

    # ------------------------------------------------ sources
    def find_duplicate_source(self, text: str) -> dict | None:
        h = content_hash(text)
        r = self.db.one("SELECT id, url FROM sources WHERE content_hash=? AND status!='failed'", (h,))
        if r:
            return dict(r) | {"reason": "exact"}
        sh = simhash(text)
        for row in self.db.all("SELECT id, url, simhash FROM sources WHERE simhash IS NOT NULL "
                               "AND status!='failed' ORDER BY id DESC LIMIT 5000"):
            if hamming(sh, row["simhash"]) <= 3:
                return {"id": row["id"], "url": row["url"], "reason": "near-duplicate"}
        return None

    def add_source(self, url: str, title: str, text: str, kind: str, published: str | None,
                   credibility: float, injection_flags: list[str], topic_id: int | None,
                   status: str = "fetched") -> int:
        existing = self.db.one("SELECT id FROM sources WHERE url=?", (url,))
        row = {
            "domain": domain_of(url), "title": title[:500], "kind": kind, "published_at": published,
            "retrieved_at": time.time(), "content": text, "content_hash": content_hash(text),
            "simhash": simhash(text), "credibility": credibility,
            "injection_flags": json.dumps(injection_flags) if injection_flags else None,
            "status": status, "topic_id": topic_id, "error": None,
        }
        if existing:
            self.db.update("sources", existing["id"], row)
            return existing["id"]
        return self.db.insert("sources", {"url": url, **row})

    def archive_source(self, source_id: int, text: str) -> int:
        """Store a source's text as retrievable chunks in the Internet Learning Archive."""
        self.db.execute("DELETE FROM chunks WHERE source_id=? AND entry_id IS NULL", (source_id,))
        chunks = chunk_text(text, self.cfg.get("learning.chunk_chars"), self.cfg.get("learning.chunk_overlap"))
        chunks = [c for c in chunks if len(c) > 80][:60]
        vecs = self.embed(chunks) if chunks else []
        for c, v in zip(chunks, vecs):
            self.db.insert("chunks", {"source_id": source_id, "layer": "archive", "text": c, "embedding": pack_vec(v)})
        self.chunk_index.invalidate()
        return len(chunks)

    # ------------------------------------------------ entries
    def find_related_entries(self, text: str, k: int = 5, vec: list[float] | None = None) -> list[tuple[dict, float]]:
        vec = vec or self.embed([text])[0]
        out = []
        if vec is not None:
            for eid, sim in self.entry_index.search(vec, k):
                row = self.db.one("SELECT * FROM knowledge_entries WHERE id=?", (eid,))
                if row:
                    out.append((row_to_dict(row, ("details",)), sim))
        return out

    def find_entry_for_topic(self, topic: str) -> dict | None:
        r = self.db.one("SELECT * FROM knowledge_entries WHERE topic=? COLLATE NOCASE AND status!='merged' "
                        "ORDER BY updated_at DESC", (topic,))
        return row_to_dict(r, ("details",))

    def create_entry(self, layer: str, category: str, topic: str, title: str, summary: str,
                     details: dict | None = None) -> int:
        now = time.time()
        vec = self.embed([f"{title}\n{summary}"])[0]
        eid = self.db.insert("knowledge_entries", {
            "layer": layer, "category": category, "topic": topic, "title": title, "summary": summary,
            "details": json.dumps(details or {}), "status": "provisional", "confidence": 0.0,
            "version": 1, "embedding": pack_vec(vec), "created_at": now, "updated_at": now,
        })
        self.entry_index.invalidate()
        self.db.audit("pipeline", "entry_created", f"entry:{eid}", topic=topic, layer=layer, category=category)
        return eid

    def update_entry(self, entry_id: int, fields: dict, reason: str, actor: str = "pipeline") -> None:
        """Every change snapshots the previous version first - nothing is silently overwritten."""
        cur = self.db.one("SELECT * FROM knowledge_entries WHERE id=?", (entry_id,))
        if not cur:
            return
        self.db.insert("entry_versions", {
            "entry_id": entry_id, "version": cur["version"], "summary": cur["summary"],
            "details": cur["details"], "status": cur["status"], "confidence": cur["confidence"],
            "reason": reason, "changed_at": time.time(),
        })
        fields = dict(fields)
        if "details" in fields and not isinstance(fields["details"], str):
            fields["details"] = json.dumps(fields["details"])
        if "summary" in fields or "title" in fields:
            title = fields.get("title", cur["title"])
            vec = self.embed([f"{title}\n{fields.get('summary', cur['summary'])}"])[0]
            if vec is not None:
                fields["embedding"] = pack_vec(vec)
            self.entry_index.invalidate()
        fields["version"] = cur["version"] + 1
        fields["updated_at"] = time.time()
        self.db.update("knowledge_entries", entry_id, fields)
        self.db.audit(actor, "entry_updated", f"entry:{entry_id}", reason=reason, version=fields["version"])

    def index_entry_text(self, entry_id: int) -> None:
        e = self.db.one("SELECT * FROM knowledge_entries WHERE id=?", (entry_id,))
        if not e:
            return
        self.db.execute("DELETE FROM chunks WHERE entry_id=?", (entry_id,))
        text = f"{e['title']}\n{e['summary']}"
        details = json.loads(e["details"] or "{}")
        for key in ("purpose", "advantages", "limitations", "comparison", "example"):
            if details.get(key):
                val = details[key]
                text += f"\n{key}: " + (val if isinstance(val, str) else json.dumps(val))
        for c, v in zip([text[:4000]], self.embed([text[:4000]])):
            self.db.insert("chunks", {"entry_id": entry_id, "layer": e["layer"], "text": c, "embedding": pack_vec(v)})
        self.chunk_index.invalidate()

    # ------------------------------------------------ claims & verification
    def entry_claims(self, entry_id: int) -> list[dict]:
        return [row_to_dict(r) for r in self.db.all(
            "SELECT * FROM claims WHERE entry_id=? AND status!='rejected' ORDER BY id", (entry_id,))]

    def add_claim(self, entry_id: int, text: str) -> int:
        now = time.time()
        vec = self.embed([text])[0]
        cid = self.db.insert("claims", {"entry_id": entry_id, "text": text, "status": "unverified",
                                        "confidence": 0.0, "embedding": pack_vec(vec),
                                        "created_at": now, "updated_at": now})
        self.claim_index.invalidate()
        return cid

    def link_claim(self, claim_id: int, source_id: int, stance: str) -> None:
        src = self.db.one("SELECT url FROM sources WHERE id=?", (source_id,))
        already = self.db.one("SELECT 1 FROM claim_sources WHERE claim_id=? AND source_id=? AND stance=?",
                              (claim_id, source_id, stance))
        if already:
            return
        others = self.db.all(
            "SELECT s.url, s.domain FROM claim_sources cs JOIN sources s ON s.id=cs.source_id "
            "WHERE cs.claim_id=? AND cs.stance='supports'", (claim_id,))
        self.db.execute("INSERT OR IGNORE INTO claim_sources(claim_id,source_id,stance) VALUES(?,?,?)",
                        (claim_id, source_id, stance))
        # Track record: independent corroboration raises a domain's reputation;
        # being contradicted by the existing body of support lowers it.
        if src:
            new_reg = registrable(domain_of(src["url"]))
            independent = [o for o in others if registrable(o["domain"]) != new_reg]
            if stance == "supports" and independent:
                self.cred.record(src["url"], supported=1)
                for o in independent:
                    self.cred.record(o["url"], supported=1)
            elif stance == "contradicts" and len(independent) >= 2:
                self.cred.record(src["url"], contradicted=1)

    def recompute_claim(self, claim_id: int) -> str:
        rows = self.db.all(
            "SELECT cs.stance, s.url, s.domain, s.credibility FROM claim_sources cs "
            "JOIN sources s ON s.id=cs.source_id WHERE cs.claim_id=?", (claim_id,))

        def by_domain(stance: str) -> dict[str, float]:
            best: dict[str, float] = {}
            for r in rows:
                if r["stance"] == stance:
                    reg = registrable(r["domain"])
                    best[reg] = max(best.get(reg, 0.0), r["credibility"])
            return best

        sup, con = by_domain("supports"), by_domain("contradicts")
        min_dom = self.cfg.get("learning.verify_min_independent_domains", 2)
        min_cred = self.cfg.get("learning.verify_min_credibility", 0.6)
        auth = self.cfg.get("learning.authoritative_credibility", 0.9)

        p_wrong = 1.0
        for c in sup.values():
            p_wrong *= 1 - 0.85 * c
        conf = 1 - p_wrong
        con_weight = 1 - math.prod(1 - 0.85 * c for c in con.values()) if con else 0.0
        conf = max(0.0, conf - con_weight)

        credible_sup = [c for c in sup.values() if c >= min_cred]
        max_con = max(con.values(), default=0.0)
        if con and max_con >= min_cred and con_weight >= 0.5 * (1 - p_wrong):
            status = "contested"
        elif any(c >= auth for c in sup.values()) and max_con < auth:
            status = "verified"
        elif len(credible_sup) >= min_dom:
            status = "verified"
        else:
            status = "unverified"
        self.db.update("claims", claim_id, {"status": status, "confidence": round(conf, 3),
                                            "updated_at": time.time()})
        return status

    def recompute_entry(self, entry_id: int, reason: str = "re-verification") -> dict:
        claims = self.entry_claims(entry_id)
        statuses = [self.recompute_claim(c["id"]) for c in claims]
        # Only claims backed by at least one credible source count toward the entry's
        # status; otherwise any weak page could drag well-corroborated knowledge down.
        # Weak claims stay individually labelled 'unverified'.
        min_cred = self.cfg.get("learning.verify_min_credibility", 0.6)
        credible = []
        for c, st in zip(claims, statuses):
            best = self.db.scalar("SELECT MAX(s.credibility) FROM claim_sources cs JOIN sources s ON s.id=cs.source_id "
                                  "WHERE cs.claim_id=? AND cs.stance='supports'", (c["id"],), 0.0)
            if best >= min_cred or st in ("verified", "contested"):
                credible.append(st)
        n = len(credible) or 1
        verified = credible.count("verified")
        contested = credible.count("contested")
        confs = [r["confidence"] for r in self.db.all(
            "SELECT confidence FROM claims WHERE entry_id=? AND status!='rejected'", (entry_id,))]
        confidence = round(sum(confs) / len(confs), 3) if confs else 0.0
        if contested and contested / n > 0.25:
            status = "contested"
        elif verified and verified / n >= 0.5:
            status = "verified"
        else:
            status = "provisional"
        cur = self.db.one("SELECT status, confidence FROM knowledge_entries WHERE id=?", (entry_id,))
        fields: dict[str, Any] = {}
        if cur and (cur["status"] != status or abs(cur["confidence"] - confidence) > 0.01):
            fields = {"status": status, "confidence": confidence}
        if status == "verified":
            fields["last_verified_at"] = time.time()
        if fields:
            if "status" in fields:
                self.update_entry(entry_id, fields, reason)
            else:
                self.db.update("knowledge_entries", entry_id, fields)
        return {"status": status, "confidence": confidence, "claims": len(claims),
                "verified_claims": verified, "contested_claims": contested}

    def record_contradiction(self, claim_id: int, source_id: int, statement: str, note: str = "") -> None:
        self.db.insert("contradictions", {"claim_id": claim_id, "source_id": source_id,
                                          "new_statement": statement[:1000], "note": note[:500],
                                          "created_at": time.time()})
        self.db.audit("pipeline", "contradiction_detected", f"claim:{claim_id}", source_id=source_id)

    # ------------------------------------------------ retrieval (RAG)
    def retrieve(self, query: str, k: int | None = None, layers: list[str] | None = None) -> list[dict]:
        k = k or self.cfg.get("retrieval.top_k", 6)
        min_score = self.cfg.get("retrieval.min_score", 0.3)
        include_prov = self.cfg.get("retrieval.include_provisional", True)
        qvec = self.embed([query])[0]
        hits: dict[tuple[str, int], dict] = {}

        if qvec is not None:
            for eid, sim in self.entry_index.search(qvec, k * 2):
                hits[("entry", eid)] = {"score": sim}
            for cid, sim in self.chunk_index.search(qvec, k * 3):
                hits[("chunk", cid)] = {"score": sim}
        # Lexical recall (also the full fallback when embeddings are unavailable).
        fts_q = " OR ".join(w for w in re.findall(r"\w{3,}", query.lower())[:12])
        if fts_q:
            try:
                for r in self.db.all("SELECT rowid, bm25(chunks_fts) AS b FROM chunks_fts WHERE chunks_fts MATCH ? "
                                     "ORDER BY b LIMIT ?", (fts_q, k * 3)):
                    # bm25() is negative (more negative = better); squash to 0..1.
                    lex = (1 / (1 + math.exp(r["b"] / 4)) - 0.5) * 2
                    key = ("chunk", r["rowid"])
                    prev = hits.get(key, {"score": 0.0})["score"]
                    if qvec is None:   # lexical-only mode
                        hits[key] = {"score": max(prev, 0.3 + 0.5 * lex)}
                    elif key in hits:  # hybrid: lexical match nudges the vector score
                        hits[key] = {"score": prev + 0.1 * lex}
                    else:              # exact-term hit the vectors missed (names, versions)
                        hits[key] = {"score": 0.2 + 0.35 * lex}
            except Exception:
                pass

        results = []
        for (kind, rid), h in hits.items():
            if kind == "entry":
                e = self.db.one("SELECT * FROM knowledge_entries WHERE id=?", (rid,))
                if not e or e["status"] == "merged":
                    continue
                item = {"type": "entry", "id": rid, "entry_id": rid, "layer": e["layer"], "title": e["title"],
                        "text": e["summary"], "status": e["status"], "confidence": e["confidence"]}
            else:
                c = self.db.one(
                    "SELECT c.*, s.url, s.title AS stitle, s.credibility, e.status AS estatus, e.title AS etitle "
                    "FROM chunks c LEFT JOIN sources s ON s.id=c.source_id "
                    "LEFT JOIN knowledge_entries e ON e.id=c.entry_id WHERE c.id=?", (rid,))
                if not c:
                    continue
                item = {"type": "chunk", "id": rid, "entry_id": c["entry_id"], "layer": c["layer"],
                        "title": c["etitle"] or c["stitle"] or "archive", "text": c["text"],
                        "status": c["estatus"] or "unverified-source", "url": c["url"],
                        "confidence": c["credibility"] if c["credibility"] is not None else 0.5}
            if layers and item["layer"] not in layers:
                continue
            if not include_prov and item["status"] not in ("verified",):
                continue
            # Verified knowledge outranks equally similar unverified material.
            bonus = {"verified": 0.08, "provisional": 0.0, "contested": -0.05,
                     "outdated": -0.08}.get(item["status"], -0.03)
            item["score"] = round(h["score"] + bonus, 4)
            if item["score"] >= min_score:
                results.append(item)
        results.sort(key=lambda x: -x["score"])
        seen_entries, final = set(), []
        for r in results:  # one representative per entry keeps context diverse
            key = r["entry_id"] or ("c", r["id"])
            if key in seen_entries:
                continue
            seen_entries.add(key)
            final.append(r)
            if len(final) >= k:
                break
        return final

    def entry_sources(self, entry_id: int) -> list[dict]:
        return [dict(r) for r in self.db.all(
            "SELECT DISTINCT s.id, s.url, s.title, s.domain, s.credibility, s.published_at, s.retrieved_at "
            "FROM claims c JOIN claim_sources cs ON cs.claim_id=c.id JOIN sources s ON s.id=cs.source_id "
            "WHERE c.entry_id=? ORDER BY s.credibility DESC", (entry_id,))]
