"""Backups of the knowledge database and configuration (online SQLite backup API)."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .config import Config
from .db import Database
from .training import backup_config


def backup_all(cfg: Config, db: Database) -> dict:
    d = cfg.data_dir / "backups"
    d.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dst = d / f"knowledge_{stamp}.db"
    target = sqlite3.connect(dst)
    try:
        db.conn.backup(target)  # consistent snapshot while learning continues
    finally:
        target.close()
    cfg_copy = backup_config(cfg)
    keep = cfg.get("backups.keep", 10)
    for pattern in ("knowledge_*.db", "config_*.json"):
        files = sorted(d.glob(pattern))
        for old in files[:-keep]:
            old.unlink(missing_ok=True)
    db.kv_set("last_backup_at", time.time())
    db.audit("system", "backup_created", str(dst), config=str(cfg_copy) if cfg_copy else None)
    return {"database": str(dst), "config": str(cfg_copy) if cfg_copy else None,
            "size_bytes": Path(dst).stat().st_size}
