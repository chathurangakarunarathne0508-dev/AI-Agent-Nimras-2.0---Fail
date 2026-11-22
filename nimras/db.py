"""SQLite persistence for every layer of N.I.M.R.A.S. memory and learning history.

One file, WAL mode, a connection per thread. Embeddings are float32 BLOBs.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from array import array
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ---------------------------------------------------------------- sources
CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY,
    url           TEXT UNIQUE NOT NULL,
    domain        TEXT NOT NULL,
    title         TEXT,
    kind          TEXT NOT NULL DEFAULT 'web',      -- web|docs|paper|rss|encyclopedia
    published_at  TEXT,                             -- as stated by the source, if any
    retrieved_at  REAL NOT NULL,
    content       TEXT,
    content_hash  TEXT,
    simhash       TEXT,
    credibility   REAL NOT NULL DEFAULT 0.5,
    injection_flags TEXT,                           -- JSON list of matched patterns
    status        TEXT NOT NULL DEFAULT 'fetched',  -- fetched|analyzed|duplicate|rejected|failed
    topic_id      INTEGER,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS ix_sources_hash ON sources(content_hash);
CREATE INDEX IF NOT EXISTS ix_sources_domain ON sources(domain);

CREATE TABLE IF NOT EXISTS domain_reputation (
    domain      TEXT PRIMARY KEY,
    supported   INTEGER NOT NULL DEFAULT 0,   -- claims later corroborated by others
    contradicted INTEGER NOT NULL DEFAULT 0,  -- claims contradicted by more credible sources
    injections  INTEGER NOT NULL DEFAULT 0,
    updated_at  REAL
);

-- ------------------------------------------------------- knowledge layers
-- layer: personal | general | technical | archive
CREATE TABLE IF NOT EXISTS knowledge_entries (
    id             INTEGER PRIMARY KEY,
    layer          TEXT NOT NULL,
    category       TEXT NOT NULL,
    topic          TEXT NOT NULL,
    title          TEXT NOT NULL,
    summary        TEXT NOT NULL,
    details        TEXT,                        -- JSON: advantages, limitations, example, test...
    status         TEXT NOT NULL DEFAULT 'provisional', -- verified|provisional|contested|outdated|merged
    confidence     REAL NOT NULL DEFAULT 0.0,
    version        INTEGER NOT NULL DEFAULT 1,
    embedding      BLOB,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    last_verified_at REAL,
    merged_into    INTEGER
);
CREATE INDEX IF NOT EXISTS ix_entries_topic ON knowledge_entries(topic);
CREATE INDEX IF NOT EXISTS ix_entries_cat ON knowledge_entries(layer, category);

CREATE TABLE IF NOT EXISTS entry_versions (
    id          INTEGER PRIMARY KEY,
    entry_id    INTEGER NOT NULL REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    version     INTEGER NOT NULL,
    summary     TEXT,
    details     TEXT,
    status      TEXT,
    confidence  REAL,
    reason      TEXT,
    changed_at  REAL NOT NULL
);

-- Atomic, individually verifiable statements backing an entry.
CREATE TABLE IF NOT EXISTS claims (
    id          INTEGER PRIMARY KEY,
    entry_id    INTEGER NOT NULL REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'unverified', -- verified|unverified|contested|rejected
    confidence  REAL NOT NULL DEFAULT 0.0,
    embedding   BLOB,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS claim_sources (
    claim_id    INTEGER NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    source_id   INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    stance      TEXT NOT NULL,                 -- supports|contradicts
    PRIMARY KEY (claim_id, source_id, stance)
);
CREATE TABLE IF NOT EXISTS contradictions (
    id           INTEGER PRIMARY KEY,
    claim_id     INTEGER REFERENCES claims(id) ON DELETE CASCADE,
    source_id    INTEGER REFERENCES sources(id) ON DELETE SET NULL,
    new_statement TEXT NOT NULL,
    resolution   TEXT NOT NULL DEFAULT 'open',  -- open|kept_existing|replaced|both_uncertain
    note         TEXT,
    created_at   REAL NOT NULL
);

-- Raw-text retrieval units (Internet Learning Archive + entry text).
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    source_id   INTEGER REFERENCES sources(id) ON DELETE CASCADE,
    entry_id    INTEGER REFERENCES knowledge_entries(id) ON DELETE CASCADE,
    layer       TEXT NOT NULL,
    text        TEXT NOT NULL,
    embedding   BLOB
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

-- --------------------------------------------------------- personal memory
CREATE TABLE IF NOT EXISTS personal_memory (
    id          INTEGER PRIMARY KEY,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    approved    INTEGER NOT NULL DEFAULT 0,    -- only approved rows are ever used
    sensitive   INTEGER NOT NULL DEFAULT 1,    -- never sent to the internet
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    retrieved       TEXT,                    -- JSON ids used for the answer
    prompt_version  INTEGER,
    model           TEXT,
    created_at      REAL NOT NULL
);

-- ---------------------------------------------------------- learning flow
CREATE TABLE IF NOT EXISTS topics (
    id               INTEGER PRIMARY KEY,
    name             TEXT UNIQUE NOT NULL COLLATE NOCASE,
    layer            TEXT NOT NULL DEFAULT 'technical',
    category         TEXT NOT NULL DEFAULT 'uncategorized',
    priority         REAL NOT NULL DEFAULT 0.5,
    origin           TEXT NOT NULL,            -- seed|discovered|gap|revisit|user|emerging
    status           TEXT NOT NULL DEFAULT 'queued', -- queued|active|learned|failed|parked
    attempts         INTEGER NOT NULL DEFAULT 0,
    deep_study       INTEGER NOT NULL DEFAULT 0,
    last_researched_at REAL,
    next_due_at      REAL,
    created_at       REAL NOT NULL,
    note             TEXT
);
CREATE TABLE IF NOT EXISTS learning_sessions (
    id          INTEGER PRIMARY KEY,
    trigger     TEXT NOT NULL,                 -- schedule|manual|startup
    status      TEXT NOT NULL DEFAULT 'running', -- running|completed|failed|aborted
    started_at  REAL NOT NULL,
    ended_at    REAL,
    topics      TEXT,                          -- JSON
    stats       TEXT,                          -- JSON
    error       TEXT
);
CREATE TABLE IF NOT EXISTS research_attempts (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER,
    topic_id    INTEGER,
    stage       TEXT NOT NULL,                 -- search|fetch|analyze|verify|sandbox|discover
    provider    TEXT,
    strategy    TEXT,
    query       TEXT,
    url         TEXT,
    ok          INTEGER NOT NULL,
    yield_count INTEGER NOT NULL DEFAULT 0,    -- useful items produced
    error       TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_attempts_stage ON research_attempts(stage, ok);
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id          INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    origin      TEXT NOT NULL,                 -- chat|evaluation|analysis|probe
    status      TEXT NOT NULL DEFAULT 'open',  -- open|queued|filled|dismissed
    topic_id    INTEGER,
    created_at  REAL NOT NULL,
    resolved_at REAL
);

-- ------------------------------------------------------ self-improvement
CREATE TABLE IF NOT EXISTS eval_runs (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,                 -- self_improvement|model_comparison
    started_at  REAL NOT NULL,
    ended_at    REAL,
    metrics     TEXT,                          -- JSON
    previous_run_id INTEGER,
    deltas      TEXT                           -- JSON metric -> change vs previous
);
CREATE TABLE IF NOT EXISTS evaluations (
    id          INTEGER PRIMARY KEY,
    run_id      INTEGER,
    kind        TEXT NOT NULL,                 -- response|retrieval_probe|model_case
    subject_id  INTEGER,
    score       REAL,
    details     TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS mistakes (
    id          INTEGER PRIMARY KEY,
    signature   TEXT NOT NULL,                 -- short normalised description
    example_message_id INTEGER,
    occurrences INTEGER NOT NULL DEFAULT 1,
    embedding   BLOB,
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS prompts (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    version     INTEGER NOT NULL,
    template    TEXT NOT NULL,
    status      TEXT NOT NULL,                 -- active|candidate|retired
    rationale   TEXT,
    mean_score  REAL,
    samples     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    UNIQUE(name, version)
);
CREATE TABLE IF NOT EXISTS strategy_stats (
    kind        TEXT NOT NULL,                 -- provider|query_template
    name        TEXT NOT NULL,
    uses        INTEGER NOT NULL DEFAULT 0,
    successes   REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, name)
);

-- -------------------------------------------------- experimental training
CREATE TABLE IF NOT EXISTS training_datasets (
    id          INTEGER PRIMARY KEY,
    path        TEXT NOT NULL,
    holdout_path TEXT,
    n_train     INTEGER, n_holdout INTEGER,
    quality     TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_versions (
    id          INTEGER PRIMARY KEY,
    name        TEXT UNIQUE NOT NULL,          -- Ollama model tag
    base_model  TEXT,
    dataset_id  INTEGER,
    status      TEXT NOT NULL,                 -- production|candidate|rejected|retired
    eval_score  REAL,
    baseline_score REAL,
    notes       TEXT,
    created_at  REAL NOT NULL,
    promoted_at REAL
);

-- -------------------------------------------------------------- system
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY,
    ts          REAL NOT NULL,
    actor       TEXT NOT NULL,                 -- scheduler|pipeline|agent|evaluator|user|api
    action      TEXT NOT NULL,
    target      TEXT,
    details     TEXT
);
CREATE INDEX IF NOT EXISTS ix_audit_ts ON audit_log(ts);
CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ---------------------------------------------------------------- voice engine
CREATE TABLE IF NOT EXISTS voice_turns (
    id              INTEGER PRIMARY KEY,
    conversation_id TEXT,
    created_at      REAL NOT NULL,
    spoken_lang     TEXT,
    reply_lang      TEXT,
    stt_engine      TEXT,
    stt_model       TEXT,
    audio_seconds   REAL,
    stt_latency     REAL,
    llm_first_token REAL,          -- seconds from end of speech
    first_audio     REAL,          -- seconds from end of speech to first audio chunk sent
    total           REAL,
    interrupted     INTEGER DEFAULT 0,
    interrupt_latency REAL,        -- seconds from interrupt request to pipeline cancelled
    client_metrics  TEXT,          -- JSON reported by the browser (perceived latency etc.)
    error           TEXT
);
CREATE TABLE IF NOT EXISTS voice_tests (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    category    TEXT NOT NULL,
    name        TEXT NOT NULL,
    language    TEXT,
    passed      INTEGER,
    wer         REAL,
    cer         REAL,
    latency     REAL,
    details     TEXT
);
CREATE TABLE IF NOT EXISTS voice_models (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,     -- stt|stt_lang|tts
    language    TEXT,
    name        TEXT NOT NULL,     -- file name
    source      TEXT,
    status      TEXT NOT NULL,     -- candidate|production|retired|rejected
    metrics     TEXT,
    created_at  REAL NOT NULL,
    promoted_at REAL,
    notes       TEXT
);
"""


def pack_vec(vec: Iterable[float] | None) -> bytes | None:
    if vec is None:
        return None
    return array("f", vec).tobytes()


def unpack_vec(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    a = array("f")
    a.frombytes(blob)
    return a.tolist()


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        with self._write_lock:
            self.conn.executescript(SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=30000")
            self._local.conn = c
        return c

    # ---------------------------------------------------------- helpers
    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._write_lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def insert(self, table: str, row: dict) -> int:
        cols = ",".join(row)
        qs = ",".join("?" for _ in row)
        cur = self.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", tuple(row.values()))
        return int(cur.lastrowid)

    def update(self, table: str, row_id: int, fields: dict) -> None:
        if not fields:
            return
        sets = ",".join(f"{k}=?" for k in fields)
        self.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*fields.values(), row_id))

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def scalar(self, sql: str, params: tuple = (), default: Any = 0) -> Any:
        r = self.conn.execute(sql, params).fetchone()
        return default if r is None or r[0] is None else r[0]

    # ---------------------------------------------------------- kv
    def kv_get(self, key: str, default: Any = None) -> Any:
        r = self.one("SELECT value FROM kv WHERE key=?", (key,))
        return default if r is None else json.loads(r["value"])

    def kv_set(self, key: str, value: Any) -> None:
        self.execute(
            "INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    # ---------------------------------------------------------- audit
    def audit(self, actor: str, action: str, target: str | None = None, **details: Any) -> None:
        self.insert(
            "audit_log",
            {
                "ts": time.time(),
                "actor": actor,
                "action": action,
                "target": target,
                "details": json.dumps(details, default=str) if details else None,
            },
        )

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None


def row_to_dict(row: sqlite3.Row | None, json_fields: tuple[str, ...] = ()) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    d.pop("embedding", None)
    for f in json_fields:
        if d.get(f):
            try:
                d[f] = json.loads(d[f])
            except (TypeError, ValueError):
                pass
    return d
