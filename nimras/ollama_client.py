"""Ollama gateway: local chat + embeddings, and Ollama's hosted web search / fetch.

All model inference in N.I.M.R.A.S. goes through this class.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any

from .config import Config


class OllamaError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class LLMBudget:
    """Per-session cap on model calls so background learning stays bounded."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.used >= self.max_calls:
                raise BudgetExceeded(f"LLM call budget of {self.max_calls} exhausted")
            self.used += 1


_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)


def parse_json_loose(text: str) -> Any:
    """Models sometimes wrap JSON in prose or code fences; recover the payload."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        return json.loads(text)
    except ValueError:
        m = _JSON_BLOCK.search(text)
        if m:
            return json.loads(m.group(0))
        raise


class OllamaClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.host = cfg.get("ollama.host").rstrip("/")
        self.timeout = cfg.get("ollama.request_timeout", 180)
        self._local = threading.local()
        # Interactive (chat) calls take priority over background learning:
        # while any are in flight, background calls wait before hitting Ollama.
        self._interactive = 0
        self._gate = threading.Condition()

    @contextlib.contextmanager
    def interactive(self):
        """Mark a whole user-facing operation (e.g. a voice turn: STT + LLM +
        TTS) as interactive, so background learning holds its model calls."""
        with self._gate:
            self._interactive += 1
        try:
            yield
        finally:
            with self._gate:
                self._interactive -= 1
                self._gate.notify_all()

    @contextlib.contextmanager
    def _slot(self):
        if self.budget is not None:          # background learning thread
            with self._gate:
                self._gate.wait_for(lambda: self._interactive == 0, timeout=300)
            yield
            return
        with self._gate:
            self._interactive += 1
        try:
            yield
        finally:
            with self._gate:
                self._interactive -= 1
                self._gate.notify_all()

    # ----------------------------------------------------------- budget scope
    @property
    def budget(self) -> LLMBudget | None:
        return getattr(self._local, "budget", None)

    def set_budget(self, budget: LLMBudget | None) -> None:
        """Budget is thread-local: a background session never starves chat."""
        self._local.budget = budget

    # ----------------------------------------------------------- models
    @property
    def chat_model(self) -> str:
        return self.cfg.get("ollama.chat_model")

    @property
    def analysis_model(self) -> str:
        return self.cfg.get("ollama.analysis_model") or self.chat_model

    @property
    def embed_model(self) -> str:
        return self.cfg.get("ollama.embed_model")

    def _post(self, url: str, payload: dict, headers: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            raise OllamaError(f"{url} -> HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise OllamaError(f"{url} unreachable: {e}") from e

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.host + path, timeout=10) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError(f"Ollama unreachable at {self.host}: {e}") from e

    def available_models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def health(self) -> dict:
        try:
            models = self.available_models()
        except OllamaError as e:
            return {"reachable": False, "error": str(e), "models": []}

        def has(name: str) -> bool:
            # "llama3.2:1b" must match exactly; an untagged name means ":latest".
            wanted = name if ":" in name else f"{name}:latest"
            return any(m == wanted or m == name for m in models)

        return {
            "reachable": True,
            "models": models,
            "chat_model": self.chat_model,
            "chat_model_ready": has(self.chat_model),
            "analysis_model": self.analysis_model,
            "analysis_model_ready": has(self.analysis_model),
            "embed_model": self.embed_model,
            "embed_model_ready": has(self.embed_model),
            "web_search_configured": bool(self.web_api_key),
        }

    # ----------------------------------------------------------- inference
    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
        num_ctx: int | None = 8192,
    ) -> str:
        if self.budget:
            self.budget.take()
        payload: dict[str, Any] = {
            "model": model or self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx
        if json_mode:
            payload["format"] = "json"
        with self._slot():
            resp = self._post(self.host + "/api/chat", payload)
        return resp.get("message", {}).get("content", "")

    def chat_stream(self, messages: list[dict], model: str | None = None, temperature: float = 0.4,
                    num_ctx: int | None = 8192, cancel: threading.Event | None = None, keep_alive: str = "30m"):
        """Yield text deltas as Ollama generates them. Stops promptly when
        `cancel` is set (closing the HTTP stream makes Ollama abort generation)."""
        if self.budget:
            self.budget.take()
        payload: dict[str, Any] = {"model": model or self.chat_model, "messages": messages, "stream": True,
                                   "options": {"temperature": temperature}, "keep_alive": keep_alive}
        if num_ctx:
            payload["options"]["num_ctx"] = num_ctx
        req = urllib.request.Request(self.host + "/api/chat", data=json.dumps(payload).encode(), method="POST")
        req.add_header("Content-Type", "application/json")
        with self._slot():
            try:
                resp = urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                raise OllamaError(f"/api/chat -> HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise OllamaError(f"Ollama unreachable: {e}") from e
            with resp:
                for raw in resp:
                    if cancel is not None and cancel.is_set():
                        return
                    if not raw.strip():
                        continue
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    if msg.get("error"):
                        raise OllamaError(str(msg["error"]))
                    delta = (msg.get("message") or {}).get("content", "")
                    if delta:
                        yield delta
                    if msg.get("done"):
                        return

    def chat_json(self, system: str, user: str, model: str | None = None, retries: int = 1) -> Any:
        """Structured output. Retries once with a corrective nudge on malformed JSON."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_err: Exception | None = None
        for _ in range(retries + 1):
            raw = self.chat(messages, model=model or self.analysis_model, json_mode=True, temperature=0.1)
            try:
                return parse_json_loose(raw)
            except ValueError as e:
                last_err = e
                messages += [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "That was not valid JSON. Reply with only the JSON object."},
                ]
        raise OllamaError(f"model did not return valid JSON: {last_err}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._slot():
            resp = self._post(
                self.host + "/api/embed",
                {"model": self.embed_model, "input": [t[:8000] for t in texts]},
            )
        embs = resp.get("embeddings")
        if not embs or len(embs) != len(texts):
            raise OllamaError("embedding response malformed")
        return embs

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    # ----------------------------------------------------------- web (hosted)
    @property
    def web_api_key(self) -> str:
        return os.environ.get(self.cfg.get("ollama.api_key_env", "OLLAMA_API_KEY"), "")

    def web_search(self, query: str, max_results: int = 5) -> list[dict]:
        if not self.web_api_key:
            raise OllamaError("Ollama web search needs OLLAMA_API_KEY")
        base = self.cfg.get("ollama.web_api_base").rstrip("/")
        resp = self._post(
            base + "/web_search",
            {"query": query, "max_results": max_results},
            headers={"Authorization": f"Bearer {self.web_api_key}"},
            timeout=30,
        )
        return resp.get("results", [])

    def web_fetch(self, url: str) -> dict:
        if not self.web_api_key:
            raise OllamaError("Ollama web fetch needs OLLAMA_API_KEY")
        base = self.cfg.get("ollama.web_api_base").rstrip("/")
        return self._post(
            base + "/web_fetch",
            {"url": url},
            headers={"Authorization": f"Bearer {self.web_api_key}"},
            timeout=45,
        )
