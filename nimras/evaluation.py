"""Self-improvement engine: measure, diagnose, change, and re-measure.

Each run produces an eval_runs row with metrics and deltas against the previous
run, so "improvement" is a number, not a feeling:
  * response quality   - LLM-judge scores of past answers (correct/grounded/helpful)
  * repeated mistakes  - clustered error signatures with occurrence counts
  * retrieval accuracy - recall@k and MRR on self-generated probe questions
  * knowledge gaps     - missing knowledge found in failures -> learning queue
  * prompt experiments - candidate prompts A/B-tested, promoted only on a margin
  * knowledge hygiene  - near-duplicate entries merged with version history
  * search strategy    - provider / query-template success rates
"""
from __future__ import annotations

import json
import random
import re
import time

from .config import Config
from .db import Database, pack_vec, unpack_vec
from .knowledge import KnowledgeStore, cosine
from .ollama_client import OllamaClient, OllamaError
from .prompts import PromptRegistry

JUDGE_PROMPT = (
    "You are a strict evaluator of an AI assistant's answer. You are given the user question, the knowledge "
    "context the assistant had (with verification labels), and the answer. Judge:\n"
    "- correct: 0-1, factual accuracy (penalise invented versions, dates, APIs, sources)\n"
    "- grounded: 0-1, consistency with the provided context and honest labelling of unverified info\n"
    "- helpful: 0-1, does it actually address the question\n"
    "Return JSON: {\"correct\": float, \"grounded\": float, \"helpful\": float, \"errors\": [short str], "
    "\"missing_knowledge\": str or null (a specific topic the assistant needed but lacked)}"
)


class SelfImprovementEngine:
    def __init__(self, cfg: Config, db: Database, ollama: OllamaClient, store: KnowledgeStore,
                 prompts: PromptRegistry):
        self.cfg, self.db, self.ollama, self.store, self.prompts = cfg, db, ollama, store, prompts

    # ------------------------------------------------------------ orchestration
    def run(self) -> dict:
        prev = self.db.one("SELECT id, metrics FROM eval_runs WHERE kind='self_improvement' AND ended_at IS NOT NULL "
                           "ORDER BY id DESC LIMIT 1")
        run_id = self.db.insert("eval_runs", {"kind": "self_improvement", "started_at": time.time(),
                                              "previous_run_id": prev["id"] if prev else None})
        self.db.audit("evaluator", "evaluation_started", f"run:{run_id}")
        metrics: dict = {}
        for name, step in (("responses", self.evaluate_responses), ("retrieval", self.retrieval_probes),
                           ("prompts", self.manage_prompt_experiments), ("organization", self.organize_knowledge)):
            try:
                metrics.update(step(run_id))
            except OllamaError as e:
                metrics[f"{name}_error"] = str(e)[:200]
        metrics.update(self.knowledge_metrics())
        deltas = {}
        if prev and prev["metrics"]:
            old = json.loads(prev["metrics"])
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and isinstance(old.get(k), (int, float)):
                    deltas[k] = round(v - old[k], 4)
        self.db.update("eval_runs", run_id, {"ended_at": time.time(), "metrics": json.dumps(metrics),
                                             "deltas": json.dumps(deltas)})
        self.db.audit("evaluator", "evaluation_completed", f"run:{run_id}", metrics=metrics, deltas=deltas)
        return {"run_id": run_id, "metrics": metrics, "deltas": deltas}

    # ------------------------------------------------------------ responses
    def evaluate_responses(self, run_id: int) -> dict:
        n = self.cfg.get("self_improvement.responses_per_run", 20)
        low = self.cfg.get("self_improvement.low_score_threshold", 0.55)
        rows = self.db.all(
            "SELECT m.* FROM messages m WHERE m.role='assistant' AND NOT EXISTS "
            "(SELECT 1 FROM evaluations e WHERE e.kind='response' AND e.subject_id=m.id) ORDER BY m.id DESC LIMIT ?",
            (n,))
        scores, low_count, gaps = [], 0, 0
        for m in rows:
            q = self.db.one("SELECT content FROM messages WHERE conversation_id=? AND role='user' AND id<? "
                            "ORDER BY id DESC LIMIT 1", (m["conversation_id"], m["id"]))
            if not q:
                continue
            context = self._context_text(json.loads(m["retrieved"] or "[]"))
            verdict = self.ollama.chat_json(
                JUDGE_PROMPT, f"QUESTION:\n{q['content']}\n\nCONTEXT:\n{context or '(none)'}\n\nANSWER:\n{m['content']}")
            parts = [float(verdict.get(k, 0) or 0) for k in ("correct", "grounded", "helpful")]
            score = round(sum(max(0.0, min(1.0, p)) for p in parts) / 3, 3)
            scores.append(score)
            self.db.insert("evaluations", {"run_id": run_id, "kind": "response", "subject_id": m["id"],
                                           "score": score, "details": json.dumps(verdict)[:4000],
                                           "created_at": time.time()})
            if m["prompt_version"]:
                self.db.execute("UPDATE prompts SET mean_score=(COALESCE(mean_score,0)*samples+?)/(samples+1), "
                                "samples=samples+1 WHERE id=?", (score, m["prompt_version"]))
            if score < low:
                low_count += 1
                for err in (verdict.get("errors") or [])[:3]:
                    self._record_mistake(str(err), m["id"])
            missing = verdict.get("missing_knowledge")
            if missing and isinstance(missing, str) and len(missing) > 3:
                gaps += self.add_gap(missing, "evaluation")
        return {"responses_evaluated": len(scores),
                "response_score_mean": round(sum(scores) / len(scores), 3) if scores else None,
                "low_score_responses": low_count, "gaps_from_responses": gaps,
                "repeated_mistakes": self.db.scalar("SELECT COUNT(*) FROM mistakes WHERE occurrences>=2")}

    def _context_text(self, retrieved: list) -> str:
        out = []
        for r in retrieved[:8]:
            if isinstance(r, dict):
                out.append(f"[{r.get('status')}] {r.get('title')}: {str(r.get('text'))[:400]}")
        return "\n".join(out)

    def _record_mistake(self, error: str, message_id: int) -> None:
        sig = re.sub(r"\s+", " ", error.lower()).strip()[:160]
        if len(sig) < 5:
            return
        vec = self.store.embed([sig])[0]
        now = time.time()
        if vec is not None:
            for r in self.db.all("SELECT id, embedding FROM mistakes WHERE embedding IS NOT NULL"):
                if cosine(vec, unpack_vec(r["embedding"])) >= 0.85:
                    self.db.execute("UPDATE mistakes SET occurrences=occurrences+1, last_seen=? WHERE id=?",
                                    (now, r["id"]))
                    return
        self.db.insert("mistakes", {"signature": sig, "example_message_id": message_id, "occurrences": 1,
                                    "embedding": pack_vec(vec), "first_seen": now, "last_seen": now})

    def add_gap(self, description: str, origin: str) -> int:
        description = " ".join(description.split())[:200]
        if self.db.one("SELECT 1 FROM knowledge_gaps WHERE description=? COLLATE NOCASE AND status IN ('open','queued')",
                       (description,)):
            return 0
        self.db.insert("knowledge_gaps", {"description": description, "origin": origin, "status": "open",
                                          "created_at": time.time()})
        self.db.audit("evaluator", "knowledge_gap_identified", description, origin=origin)
        return 1

    # ------------------------------------------------------------ retrieval
    def retrieval_probes(self, run_id: int) -> dict:
        n = self.cfg.get("self_improvement.retrieval_probes_per_run", 15)
        claims = self.db.all("SELECT c.id, c.text, c.entry_id FROM claims c JOIN knowledge_entries e "
                             "ON e.id=c.entry_id WHERE c.status='verified' AND e.status!='merged'")
        if not claims:
            return {"retrieval_probes": 0}
        sample = random.sample(list(claims), min(n, len(claims)))
        k = self.cfg.get("retrieval.top_k", 6)
        hits, rr = 0, 0.0
        for c in sample:
            try:
                q = self.ollama.chat_json(
                    "Write one natural question a user might ask whose answer is the given fact. Paraphrase; do "
                    "not copy the fact's wording. Return JSON {\"question\": str}", c["text"]).get("question")
            except OllamaError:
                q = None
            q = q or c["text"]
            results = self.store.retrieve(q, k=k)
            rank = next((i + 1 for i, r in enumerate(results) if r.get("entry_id") == c["entry_id"]), None)
            if rank:
                hits += 1
                rr += 1 / rank
            self.db.insert("evaluations", {"run_id": run_id, "kind": "retrieval_probe", "subject_id": c["id"],
                                           "score": 1.0 / rank if rank else 0.0,
                                           "details": json.dumps({"question": q, "rank": rank}),
                                           "created_at": time.time()})
        return {"retrieval_probes": len(sample), "retrieval_recall_at_k": round(hits / len(sample), 3),
                "retrieval_mrr": round(rr / len(sample), 3)}

    # ------------------------------------------------------------ prompts
    def manage_prompt_experiments(self, run_id: int) -> dict:
        name = "chat_system"
        min_n = self.cfg.get("self_improvement.prompt_experiment_min_samples", 12)
        margin = self.cfg.get("self_improvement.prompt_promotion_margin", 0.05)
        active, cand = self.prompts.active(name), self.prompts.candidate(name)
        out = {"prompt_active_version": active["version"], "prompt_active_score": active["mean_score"]}
        if cand:
            out["prompt_candidate_version"] = cand["version"]
            if cand["samples"] >= min_n and active["samples"] >= min_n:
                if (cand["mean_score"] or 0) >= (active["mean_score"] or 0) + margin:
                    self.prompts.promote(cand["id"], f"{cand['mean_score']:.3f} vs {active['mean_score']:.3f}")
                    out["prompt_decision"] = "promoted"
                else:
                    self.prompts.retire(cand["id"], f"no significant gain ({cand['mean_score']} vs {active['mean_score']})")
                    out["prompt_decision"] = "retired"
            return out
        low = self.cfg.get("self_improvement.low_score_threshold", 0.55)
        if active["samples"] >= min_n and (active["mean_score"] or 1) < low + 0.25:
            mistakes = self.db.all("SELECT signature, occurrences FROM mistakes ORDER BY occurrences DESC LIMIT 8")
            if mistakes:
                data = self.ollama.chat_json(
                    "You improve system prompts. Given the current prompt and the assistant's most frequent "
                    "mistakes, write an improved prompt that keeps every safety/honesty rule and fixes the "
                    "mistakes. Return JSON {\"prompt\": str, \"rationale\": str}",
                    f"CURRENT PROMPT:\n{active['template']}\n\nFREQUENT MISTAKES:\n" +
                    "\n".join(f"- ({m['occurrences']}x) {m['signature']}" for m in mistakes))
                new = str(data.get("prompt") or "").strip()
                if len(new) > 200:
                    self.prompts.propose(name, new, str(data.get("rationale", "")))
                    out["prompt_decision"] = "candidate_proposed"
        return out

    # ------------------------------------------------------------ organization
    def organize_knowledge(self, run_id: int) -> dict:
        thr = self.cfg.get("self_improvement.merge_duplicate_similarity", 0.93)
        rows = self.db.all("SELECT id, layer, topic, embedding, (SELECT COUNT(*) FROM claims WHERE entry_id=k.id) AS n "
                           "FROM knowledge_entries k WHERE status!='merged' AND embedding IS NOT NULL")
        vecs = [(r, unpack_vec(r["embedding"])) for r in rows]
        merged = 0
        gone: set[int] = set()
        for i, (a, va) in enumerate(vecs):
            if a["id"] in gone or merged >= 5:
                continue
            for b, vb in vecs[i + 1:]:
                if b["id"] in gone or a["layer"] != b["layer"] or len(va) != len(vb):
                    continue
                if cosine(va, vb) >= thr:
                    keep, drop = (a, b) if a["n"] >= b["n"] else (b, a)
                    self.db.execute("UPDATE claims SET entry_id=? WHERE entry_id=?", (keep["id"], drop["id"]))
                    self.db.execute("UPDATE chunks SET entry_id=? WHERE entry_id=?", (keep["id"], drop["id"]))
                    self.store.update_entry(drop["id"], {"status": "merged", "merged_into": keep["id"]},
                                            f"merged into entry {keep['id']} (duplicate)", actor="evaluator")
                    self.store.recompute_entry(keep["id"], "absorbed duplicate entry")
                    gone.add(drop["id"])
                    merged += 1
                    if drop["id"] == a["id"]:
                        break
        if merged:
            self.store.entry_index.invalidate()
        return {"entries_merged": merged}

    # ------------------------------------------------------------ learning metrics
    def knowledge_metrics(self) -> dict:
        week = time.time() - 7 * 86400
        total = self.db.scalar("SELECT COUNT(*) FROM knowledge_entries WHERE status!='merged'")
        verified = self.db.scalar("SELECT COUNT(*) FROM knowledge_entries WHERE status='verified'")
        fetch_ok = self.db.scalar("SELECT COUNT(*) FROM research_attempts WHERE stage='fetch' AND ok=1 AND created_at>?", (week,))
        fetch_all = self.db.scalar("SELECT COUNT(*) FROM research_attempts WHERE stage='fetch' AND created_at>?", (week,))
        claims = self.db.scalar("SELECT COUNT(*) FROM claims WHERE status!='rejected'")
        vclaims = self.db.scalar("SELECT COUNT(*) FROM claims WHERE status='verified'")
        return {
            "entries_total": total, "entries_verified": verified,
            "verified_entry_ratio": round(verified / total, 3) if total else 0.0,
            "claims_total": claims, "claim_verification_rate": round(vclaims / claims, 3) if claims else 0.0,
            "fetch_success_rate_7d": round(fetch_ok / fetch_all, 3) if fetch_all else None,
            "open_contradictions": self.db.scalar("SELECT COUNT(*) FROM contradictions WHERE resolution='open'"),
            "open_gaps": self.db.scalar("SELECT COUNT(*) FROM knowledge_gaps WHERE status IN ('open','queued')"),
        }
