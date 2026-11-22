"""Research agent: deep study of a discovered piece of software.

  1. discover (done by the pipeline)      6. create internal knowledge entry
  2. research the official documentation  7. generate a practical example
  3. understand its purpose               8. test the example in an isolated sandbox
  4. compare with known alternatives      9. store the verified result
  5. identify advantages / limitations   10. available to RAG for future chats
"""
from __future__ import annotations

import ast
import json
import re
import sys
import time

from ..ollama_client import BudgetExceeded, OllamaError
from ..sandbox import Sandbox
from .credibility import prior_credibility
from .discovery import SearchResult


def third_party_imports(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ["<syntax error>"]
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module.split(".")[0])
    return sorted(m for m in mods if m not in sys.stdlib_module_names)


class ResearchAgent:
    def __init__(self, pipeline, sandbox: Sandbox):
        self.p = pipeline
        self.db, self.store, self.ollama, self.cfg = pipeline.db, pipeline.store, pipeline.ollama, pipeline.cfg
        self.sandbox = sandbox

    def deep_study(self, entry_id: int, topic: str, ctx) -> dict | None:
        entry = self.db.one("SELECT * FROM knowledge_entries WHERE id=?", (entry_id,))
        details = json.loads(entry["details"] or "{}")
        recent = details.get("deep_study_at", 0) > time.time() - self.cfg.get("scheduler.revisit_after_days", 30) * 86400
        if recent:
            return None
        ctx.stats["deep_studies"] += 1
        self.db.audit("agent", "deep_study_started", topic, entry_id=entry_id)

        # ---- 2. official documentation
        docs_url = self._research_official_docs(topic, ctx)

        # ---- 3-5, 7. purpose, comparison, pros/cons, example
        claims = self.store.entry_claims(entry_id)
        related = [e for e, sim in self.store.find_related_entries(f"{entry['title']}\n{entry['summary']}", k=6)
                   if e["id"] != entry_id and sim > 0.45][:4]
        facts = "\n".join(f"- [{c['status']}] {c['text']}" for c in claims[:30])
        alternatives = "\n".join(f"- {e['title']}: {e['summary'][:300]}" for e in related) or "(none known yet)"
        try:
            ctx.check()
            plan = self.ollama.chat_json(
                "You are a senior engineer writing an internal knowledge-base study. Use the FACTS (with their "
                "verification status) and general knowledge, but flag anything not supported by the facts as "
                "'unverified'. Return JSON: {\"purpose\": str, \"advantages\": [str], \"limitations\": [str], "
                "\"comparison\": [{\"with\": str, \"difference\": str}], \"example_language\": str, "
                "\"example_code\": str, \"example_expected_output\": str, \"unverified_points\": [str]}.\n"
                "The example must be short, self-contained, deterministic, print its result, and must not access "
                "the network, the filesystem outside /tmp, or environment secrets.",
                f"SUBJECT: {entry['title']}\nSUMMARY: {entry['summary']}\nFACTS:\n{facts}\n\n"
                f"KNOWN ALTERNATIVES IN KNOWLEDGE BASE:\n{alternatives}",
            )
        except (OllamaError, BudgetExceeded) as e:
            ctx.log(stage="analyze", ok=False, provider="agent", query=topic, error=str(e)[:300])
            if isinstance(e, BudgetExceeded):
                raise
            return None

        # ---- 8. isolated test
        code = str(plan.get("example_code") or "")
        lang = str(plan.get("example_language") or "").lower()
        test = self._test_example(code, lang, str(plan.get("example_expected_output") or ""), ctx)

        # ---- 6, 9. store
        details.update({
            "purpose": plan.get("purpose"),
            "advantages": plan.get("advantages") or [],
            "limitations": plan.get("limitations") or [],
            "comparison": plan.get("comparison") or [],
            "unverified_points": plan.get("unverified_points") or [],
            "example": {"language": lang, "code": code[:6000], "test": test},
            "official_docs": docs_url,
            "deep_study_at": time.time(),
        })
        self.store.update_entry(entry_id, {"details": details}, "deep study (agent)", actor="agent")
        self.store.index_entry_text(entry_id)
        self.db.audit("agent", "deep_study_completed", topic, entry_id=entry_id, example_status=test["status"])
        return details

    def _research_official_docs(self, topic: str, ctx) -> str | None:
        slug = re.sub(r"[^a-z0-9]", "", topic.lower().split(" ")[0])
        results: list[SearchResult] = []
        for name in self.p.discoverer.strategy.rank("provider", self.p.discoverer.active_providers())[:2]:
            q = self.p.discoverer.safe_query(f"{topic} official documentation")
            if not q:
                break
            try:
                results += self.p.discoverer.providers[name].search(q, 6)
                ctx.log(stage="search", ok=True, provider=name, strategy="official_docs", query=q,
                        yield_count=len(results))
            except Exception as e:
                ctx.log(stage="search", ok=False, provider=name, strategy="official_docs", query=q, error=str(e)[:200])

        def officialness(r: SearchResult) -> float:
            host = r.url.lower().split("/")[2] if "//" in r.url else ""
            score = prior_credibility(r.url, "docs")
            if slug and slug in host.replace("-", "").replace(".", ""):
                score += 0.2  # e.g. fastapi.tiangolo.com for "FastAPI"
            if host.startswith(("docs.", "doc.")) or "/docs" in r.url.lower():
                score += 0.1
            return score

        for r in sorted(results, key=officialness, reverse=True)[:1]:
            r.kind = "docs"
            topic_row = self.db.one("SELECT * FROM topics WHERE name=?", (topic,))
            try:
                self.p.process_source(r, dict(topic_row) if topic_row else {"id": None, "name": topic,
                                                                             "layer": "technical"}, ctx)
                return r.url
            except Exception as e:
                if type(e).__name__ in ("SessionLimit", "BudgetExceeded"):
                    raise
                ctx.log(stage="fetch", ok=False, url=r.url, error=str(e)[:200])
        return None

    def _test_example(self, code: str, lang: str, expected: str, ctx) -> dict:
        if not code.strip():
            return {"status": "no_example"}
        if "python" not in lang:
            return {"status": "not_executed", "reason": f"sandbox runs Python only (example is {lang or 'unknown'})"}
        deps = third_party_imports(code)
        if deps:
            return {"status": "not_executed",
                    "reason": f"needs third-party packages {deps}; installing downloaded packages is disabled"}
        res = self.sandbox.run_python(code)
        ctx.stats["examples_tested"] += int(res.ran)
        ctx.log(stage="sandbox", ok=res.ok, query=lang, error=None if res.ok else (res.reason or res.stderr[-200:]))
        if not res.ran:
            return {"status": "not_executed", "reason": res.reason}
        status = "passed" if res.ok else "failed"
        if res.ok and expected.strip() and expected.strip()[:40] not in res.stdout:
            status = "ran_output_differs"
        return {"status": status, "exit_code": res.exit_code, "stdout": res.stdout[-1500:],
                "stderr": res.stderr[-1500:], "reason": res.reason, "tested_at": time.time()}
