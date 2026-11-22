"""Configuration: a single JSON file, merged over safe defaults.

Every autonomous behaviour has a switch and a resource limit here.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "nimras.config.json"

DEFAULTS: dict[str, Any] = {
    "data_dir": "data",
    "ollama": {
        "host": "http://127.0.0.1:11434",
        "chat_model": "llama3.2",
        "analysis_model": "",          # empty = same as chat_model
        "embed_model": "nomic-embed-text",
        "request_timeout": 180,
        # Ollama hosted web search / fetch (https://ollama.com/api/web_search).
        "web_api_base": "https://ollama.com/api",
        "api_key_env": "OLLAMA_API_KEY",
    },
    "internet": {
        "enabled": True,
        # Discovery providers, tried in learned order of usefulness.
        "providers": ["ollama", "searxng", "wikipedia", "arxiv", "duckduckgo"],
        "searxng_url": "",             # e.g. http://127.0.0.1:8888
        "duckduckgo_enabled": False,   # HTML scraping; off unless you opt in
        "user_agent": "NIMRAS-Learner/2.0 (+local research agent)",
        "respect_robots_txt": True,
        "request_timeout": 20,
        "max_bytes_per_page": 2_000_000,
        "min_seconds_between_requests_per_domain": 2.0,
        # Empty allow-list = any public domain not on the deny-list.
        "allowed_domains": [],
        "denied_domains": [
            "facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
            "pinterest.com", "linkedin.com",
        ],
        "crawl_max_depth": 1,
        "crawl_max_pages_per_domain": 4,
        "use_ollama_web_fetch": True,
    },
    "rss_feeds": [
        "https://rss.arxiv.org/rss/cs.AI",
        "https://rss.arxiv.org/rss/cs.CR",
        "https://peps.python.org/peps.rss",
        "https://www.cisa.gov/cybersecurity-advisories/all.xml",
        "https://github.blog/feed/",
        "https://ollama.com/blog/rss.xml",
    ],
    "seed_topics": {
        "technical": [
            "Python programming language",
            "Rust programming language",
            "Linux kernel",
            "TCP/IP networking",
            "Zero trust security architecture",
            "Transformer neural networks",
            "Retrieval-augmented generation",
            "Ollama local LLM runtime",
            "Docker containers",
            "WebAssembly",
        ],
        "general": [
            "Semiconductor manufacturing",
            "Renewable energy storage",
            "Project management methodologies",
            "Scientific method",
        ],
    },
    "scheduler": {
        "enabled": True,
        "interval_minutes": 60,
        "initial_delay_seconds": 30,
        "max_topics_per_session": 3,
        "max_fetches_per_session": 24,
        "max_llm_calls_per_session": 80,
        "max_session_minutes": 25,
        "defer_while_user_active_seconds": 120,
        "quiet_hours": [],             # e.g. [[0, 6]] = no learning 00:00-06:00
        "revisit_after_days": 30,
        "discovery_every_n_sessions": 2,
        "evaluation_every_n_sessions": 3,
        "backup_every_hours": 24,
    },
    "learning": {
        "chunk_chars": 1200,
        "chunk_overlap": 150,
        "max_chars_analyzed_per_source": 9000,
        "near_duplicate_similarity": 0.95,
        "same_entry_similarity": 0.80,
        "verify_min_independent_domains": 2,
        "verify_min_credibility": 0.60,
        "authoritative_credibility": 0.90,  # one official source can verify on its own
        "max_new_topics_per_source": 3,
        "max_queued_topics": 400,
    },
    "retrieval": {
        "top_k": 6,
        "min_score": 0.30,
        "include_provisional": True,
    },
    "self_improvement": {
        "enabled": True,
        "responses_per_run": 20,
        "retrieval_probes_per_run": 15,
        "low_score_threshold": 0.55,
        "prompt_experiment_traffic": 0.25,
        "prompt_experiment_min_samples": 12,
        "prompt_promotion_margin": 0.05,
        "merge_duplicate_similarity": 0.93,
    },
    "sandbox": {
        # "docker" = isolated container, no network. "off" = never execute.
        # There is deliberately no unsandboxed mode.
        "mode": "docker",
        "image": "python:3.12-slim",
        "timeout_seconds": 30,
        "memory": "256m",
        "cpus": "0.5",
    },
    "training": {
        "enabled": False,              # experimental, opt-in
        "min_example_score": 0.8,
        "holdout_fraction": 0.15,
        "auto_promote": False,         # production model never swapped without evaluation + approval
        "promotion_margin": 0.03,
    },
    "privacy": {
        # Terms that must never appear in outgoing search queries. Approved
        # personal memory values are added automatically at runtime.
        "blocked_query_terms": [],
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "voice": {
        "enabled": True,
        "models_dir": "voice_models",
        "ollama_model": "",              # empty = ollama.chat_model; Sinhala needs a multilingual model
        "num_ctx": 0,                    # 0 = same as chat (8192); a different value forces Ollama to reload
        "rag_top_k": 3,                  # compact knowledge context: prompt eval dominates CPU latency
        "rag_chars": 450,
        "rag_min_score": 0.45,
        "history_messages": 8,
        "store_transcripts": True,       # recognised text kept in conversation history (deletable)
        # Raw audio is never stored. A recording is only saved to the self-test
        # dataset when the user explicitly presses "save to test set".
        "stt": {
            "provider": "bilingual-whisper",
            "model": "ggml-small-q5_1.bin",                       # multilingual: English (+ fallback)
            "language_id_model": "ggml-base-q5_1.bin",            # fast en/si language identification
            "language_models": {"si": "ggml-sinhala-small-q5_1.bin"},
            "allowed_languages": ["en", "si"],
            "threads": 0,                # 0 = auto (CPU count - 2, max 12)
            "beam_size": 3,
            "adaptive_audio_ctx": True,  # ~3.5x faster on CPU for short utterances
            "port": 8771,                # loopback whisper-server ports start here
        },
        "tts": {
            "voices": {"en": ["en_US-lessac-medium.onnx"], "si": ["si_LK-dilu-medium.onnx"]},
            "selected": {},
            "sapi_fallback": True,
            "speed": 1.0,
        },
        "vad": {
            "start_ratio": 3.0,          # speech = level > noise floor x ratio
            "min_level": 0.012,
            "start_frames": 4,           # 4 x 20 ms above threshold to start
            "endpoint_ms": 750,          # trailing silence that ends an utterance
            "max_utterance_s": 30,
            "preroll_ms": 300,
            "min_speech_ms": 250,
            "barge_in_ratio": 2.2,       # stricter while N.I.M.R.A.S. is speaking (echo safety)
            "barge_in_ms": 280,
        },
        "improvement": {
            "watch_feeds": [
                "https://github.com/ggml-org/whisper.cpp/releases.atom",
                "https://github.com/OHF-Voice/piper1-gpl/releases.atom",
            ],
            "auto_promote": False,       # never: promotion needs a passing benchmark AND the user
            "max_wer_regression": 0.0,
        },
    },
    "backups": {
        "keep": 10,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    """Dict-backed config with dotted lookups: cfg.get("scheduler.interval_minutes")."""

    def __init__(self, data: dict, path: Path | None = None):
        self.data = data
        self.path = path

    @classmethod
    def load(cls, path: str | os.PathLike | None = None, create: bool = True) -> "Config":
        p = Path(path) if path else DEFAULT_CONFIG_PATH
        user: dict = {}
        if p.exists():
            user = json.loads(p.read_text(encoding="utf-8"))
        elif create:
            p.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
        return cls(_deep_merge(DEFAULTS, user), p)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node = self.data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def save(self) -> None:
        if self.path:
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    @property
    def data_dir(self) -> Path:
        d = Path(self.get("data_dir", "data"))
        if not d.is_absolute():
            d = (self.path.parent if self.path else ROOT) / d
        d.mkdir(parents=True, exist_ok=True)
        return d

    def public_view(self) -> dict:
        """Config safe to show in the UI (no secrets are stored here, but be explicit)."""
        return copy.deepcopy(self.data)
