"""Conversation with RAG over the knowledge layers and approved personal memory.

Chat also feeds learning: questions the knowledge base can't answer become
knowledge gaps, which the scheduler turns into research topics.
"""
from __future__ import annotations

import json
import re
import time
import uuid

from .config import Config
from .db import Database
from .knowledge import KnowledgeStore
from .ollama_client import OllamaClient
from .prompts import PromptRegistry

_REMEMBER = re.compile(r"^\s*(?:please\s+)?remember(?: that)?\s+(.+)$", re.I | re.S)
_DONT_KNOW = re.compile(r"\b(i don'?t know yet|i do not know|not in my knowledge|no information (about|on))\b", re.I)


class ChatService:
    def __init__(self, cfg: Config, db: Database, ollama: OllamaClient, store: KnowledgeStore,
                 prompts: PromptRegistry, gap_sink=None):
        self.cfg, self.db, self.ollama, self.store, self.prompts = cfg, db, ollama, store, prompts
        self.gap_sink = gap_sink  # callable(description, origin)

    def personal_context(self) -> str:
        rows = self.db.all("SELECT key, value FROM personal_memory WHERE approved=1 ORDER BY id")
        return "\n".join(f"- {r['key']}: {r['value']}" for r in rows)

    def propose_memory(self, key: str, value: str) -> int:
        """Personal info is stored only as a PENDING item until the user approves it."""
        mid = self.db.insert("personal_memory", {"key": key[:100], "value": value[:1000], "approved": 0,
                                                 "sensitive": 1, "created_at": time.time()})
        self.db.audit("user", "personal_memory_proposed", f"memory:{mid}")
        return mid

    def answer(self, message: str, conversation_id: str | None = None) -> dict:
        conversation_id = conversation_id or uuid.uuid4().hex[:12]
        self.db.kv_set("user_last_active", time.time())
        self.db.insert("messages", {"conversation_id": conversation_id, "role": "user", "content": message,
                                    "created_at": time.time()})

        m = _REMEMBER.match(message)
        if m:
            mid = self.propose_memory("note", m.group(1).strip())
            reply = ("Noted. I've saved that as a pending personal memory. Approve it in the Learning Centre "
                     f"(Memory tab, item #{mid}) and I'll use it from then on. It is never sent to the internet.")
            self.db.insert("messages", {"conversation_id": conversation_id, "role": "assistant", "content": reply,
                                        "created_at": time.time()})
            return {"conversation_id": conversation_id, "reply": reply, "retrieved": []}

        ctx = self.build_context(message, conversation_id)
        reply = self.ollama.chat(ctx["messages"], temperature=0.4)
        self.finish(message, conversation_id, reply, ctx)
        return {"conversation_id": conversation_id, "reply": reply, "retrieved": ctx["slim"]}

    def build_context(self, message: str, conversation_id: str, extra_system: str = "",
                      history_limit: int = 9) -> dict:
        """System prompt (+ RAG knowledge + approved personal memory) and recent
        history. Shared by text chat and the voice engine."""
        retrieved = self.store.retrieve(message)
        traffic = self.cfg.get("self_improvement.prompt_experiment_traffic", 0.25)
        prompt = self.prompts.choose("chat_system", traffic)
        knowledge = "\n\n".join(
            f"[{i + 1}] ({r['status'].upper()}, confidence {r['confidence']:.2f}) {r['title']}"
            f"{' <' + r['url'] + '>' if r.get('url') else ''}\n{r['text'][:1200]}"
            for i, r in enumerate(retrieved))
        system = prompt["template"]
        personal = self.personal_context()
        if personal:
            system += f"\n\nAPPROVED USER PREFERENCES / FACTS:\n{personal}"
        system += f"\n\nKNOWLEDGE:\n{knowledge or '(nothing relevant in the knowledge base yet)'}"
        if extra_system:
            system += "\n\n" + extra_system
        history = self.db.all("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                              (conversation_id, history_limit))
        msgs = [{"role": "system", "content": system}] + [
            {"role": h["role"], "content": h["content"]} for h in reversed(history)]
        slim = [{k: r.get(k) for k in ("type", "id", "entry_id", "title", "status", "confidence", "url", "score")}
                | {"text": r["text"][:500]} for r in retrieved]
        return {"messages": msgs, "retrieved": retrieved, "slim": slim, "prompt": prompt}

    def finish(self, message: str, conversation_id: str, reply: str, ctx: dict, model: str | None = None) -> None:
        self.db.insert("messages", {"conversation_id": conversation_id, "role": "assistant", "content": reply,
                                    "retrieved": json.dumps(ctx["slim"]), "prompt_version": ctx["prompt"]["id"],
                                    "model": model or self.ollama.chat_model, "created_at": time.time()})
        # Missing knowledge -> learning queue.
        best = max((r["score"] for r in ctx["retrieved"]), default=0.0)
        if self.gap_sink and (_DONT_KNOW.search(reply) or best < 0.45) and len(message.split()) >= 3:
            topic = self._gap_topic(message)
            if topic:
                self.gap_sink(topic, "chat")

    def _gap_topic(self, question: str) -> str | None:
        """Turn a question into a researchable, PII-free topic name."""
        try:
            data = self.ollama.chat_json(
                "Extract the single specific technical or general-knowledge topic a researcher would need to study "
                "to answer this question. Never include personal details about the user. Return JSON "
                "{\"topic\": str or null} (null if the question is personal or chit-chat).", question)
            t = data.get("topic")
            return str(t)[:150] if t else None
        except Exception:
            return None
