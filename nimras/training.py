"""EXPERIMENTAL model-improvement module (opt-in: training.enabled).

Important distinction:
  * The learning pipeline updates the KNOWLEDGE DATABASE (retrieved at answer time).
  * This module prepares data to change MODEL PARAMETERS (fine-tuning), which happens
    in an external trainer (e.g. Unsloth, Axolotl, LLaMA-Factory), not inside Ollama.

Flow: curate dataset -> quality report -> (external fine-tune -> GGUF) ->
      register candidate in Ollama -> evaluate vs production on held-out set ->
      promote only if better by a margin (and approved) -> rollback any time.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

from .config import Config
from .db import Database
from .evaluation import JUDGE_PROMPT
from .ollama_client import OllamaClient
from .prompts import PromptRegistry
from .security import sanitize_query, scan_for_injection

TRAINER_README = """# N.I.M.R.A.S. fine-tuning dataset

Files: train.jsonl / holdout.jsonl - chat format ({"messages": [...]}), one example per line.
quality.json - automated quality report.

This dataset is NOT applied automatically. To fine-tune:
1. Train a LoRA on the base model with an external trainer (Unsloth, Axolotl, LLaMA-Factory)
   using train.jsonl. Never train on holdout.jsonl.
2. Merge + export to GGUF (llama.cpp convert / Unsloth save_pretrained_gguf).
3. Register:  python -m nimras model register --name nimras-ft-v1 --gguf path/to/model.gguf --dataset <id>
4. Evaluate:  python -m nimras model evaluate --id <model id>
5. Promote:   python -m nimras model promote --id <model id>   (refused unless it beats production)
6. Rollback:  python -m nimras model rollback
"""


class TrainingManager:
    def __init__(self, cfg: Config, db: Database, ollama: OllamaClient, prompts: PromptRegistry):
        self.cfg, self.db, self.ollama, self.prompts = cfg, db, ollama, prompts
        self.dir = cfg.data_dir / "training"
        self.dir.mkdir(exist_ok=True)

    def _require_enabled(self) -> None:
        if not self.cfg.get("training.enabled"):
            raise PermissionError("training module is disabled (set training.enabled=true in nimras.config.json)")

    def ensure_production_registered(self) -> None:
        if not self.db.one("SELECT 1 FROM model_versions WHERE status='production'"):
            name = self.cfg.get("ollama.chat_model")
            self.db.execute("INSERT OR IGNORE INTO model_versions(name, base_model, status, created_at, promoted_at, "
                            "notes) VALUES(?,?,?,?,?,?)", (name, name, "production", time.time(), time.time(),
                                                           "initial production model"))

    # ------------------------------------------------------------ dataset
    def build_dataset(self) -> dict:
        self._require_enabled()
        min_score = self.cfg.get("training.min_example_score", 0.8)
        system = self.prompts.active("chat_system")["template"]
        examples, rejected = [], {"pii": 0, "injection": 0, "length": 0, "duplicate": 0}
        seen: set[str] = set()

        def consider(user: str, assistant: str, source: str, score: float) -> None:
            if not (10 <= len(user) <= 4000 and 20 <= len(assistant) <= 6000):
                rejected["length"] += 1
                return
            if sanitize_query(user + " " + assistant, [])[1]:
                rejected["pii"] += 1  # personal data never goes into weights
                return
            if scan_for_injection(user + " " + assistant):
                rejected["injection"] += 1
                return
            h = hashlib.sha1((user.lower().strip() + "|" + assistant[:200]).encode()).hexdigest()
            if h in seen:
                rejected["duplicate"] += 1
                return
            seen.add(h)
            examples.append({"messages": [{"role": "system", "content": system},
                                          {"role": "user", "content": user},
                                          {"role": "assistant", "content": assistant}],
                             "meta": {"source": source, "score": score}})

        # 1. Highly rated real answers.
        for r in self.db.all("SELECT e.score, m.id, m.content, m.conversation_id FROM evaluations e "
                             "JOIN messages m ON m.id=e.subject_id WHERE e.kind='response' AND e.score>=?", (min_score,)):
            q = self.db.one("SELECT content FROM messages WHERE conversation_id=? AND role='user' AND id<? "
                            "ORDER BY id DESC LIMIT 1", (r["conversation_id"], r["id"]))
            if q:
                consider(q["content"], r["content"], "rated_response", r["score"])
        # 2. Verified knowledge as grounded Q&A (probe questions -> verified claim + source).
        for r in self.db.all(
                "SELECT ev.details, c.text, c.confidence, (SELECT s.url FROM claim_sources cs JOIN sources s "
                "ON s.id=cs.source_id WHERE cs.claim_id=c.id AND cs.stance='supports' ORDER BY s.credibility DESC "
                "LIMIT 1) AS url FROM evaluations ev JOIN claims c ON c.id=ev.subject_id "
                "WHERE ev.kind='retrieval_probe' AND c.status='verified'"):
            q = json.loads(r["details"] or "{}").get("question")
            if q and q != r["text"]:
                consider(q, f"{r['text']}" + (f" (Source: {r['url']})" if r["url"] else ""), "verified_claim",
                         r["confidence"])

        random.Random(42).shuffle(examples)
        n_hold = max(1, int(len(examples) * self.cfg.get("training.holdout_fraction", 0.15))) if examples else 0
        hold, train = examples[:n_hold], examples[n_hold:]
        out = self.dir / f"dataset_{time.strftime('%Y%m%d_%H%M%S')}"
        out.mkdir()
        for name, rows in (("train.jsonl", train), ("holdout.jsonl", hold)):
            with open(out / name, "w", encoding="utf-8") as f:
                for ex in rows:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        scores = [e["meta"]["score"] for e in examples]
        quality = {
            "examples": len(examples), "train": len(train), "holdout": len(hold), "rejected": rejected,
            "by_source": {s: sum(1 for e in examples if e["meta"]["source"] == s)
                          for s in ("rated_response", "verified_claim")},
            "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
            "ready_for_training": len(train) >= 200,
            "note": "Below ~200 examples fine-tuning is unlikely to help; keep collecting." if len(train) < 200 else "",
        }
        (out / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
        (out / "README.md").write_text(TRAINER_README, encoding="utf-8")
        ds_id = self.db.insert("training_datasets", {"path": str(out / "train.jsonl"),
                                                     "holdout_path": str(out / "holdout.jsonl"),
                                                     "n_train": len(train), "n_holdout": len(hold),
                                                     "quality": json.dumps(quality), "created_at": time.time()})
        self.db.audit("trainer", "dataset_built", f"dataset:{ds_id}", **quality)
        return {"dataset_id": ds_id, "dir": str(out), **quality}

    # ------------------------------------------------------------ models
    def register_candidate(self, name: str, dataset_id: int | None, gguf: str | None = None,
                           base_model: str | None = None) -> int:
        self._require_enabled()
        self.ensure_production_registered()
        if gguf:
            gg = Path(gguf).resolve()
            if gg.suffix.lower() != ".gguf" or not gg.exists():
                raise ValueError("gguf must be an existing .gguf file")
            if not shutil.which("ollama"):
                raise RuntimeError("ollama CLI not found")
            mf = self.dir / f"Modelfile.{name}"
            mf.write_text(f"FROM {gg}\n", encoding="utf-8")
            r = subprocess.run(["ollama", "create", name, "-f", str(mf)], capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"ollama create failed: {r.stderr[-500:]}")
        mid = self.db.insert("model_versions", {"name": name, "base_model": base_model or self.cfg.get("ollama.chat_model"),
                                                "dataset_id": dataset_id, "status": "candidate",
                                                "created_at": time.time()})
        self.db.audit("trainer", "model_candidate_registered", name, model_id=mid, gguf=gguf)
        return mid

    def evaluate_candidate(self, model_id: int, max_cases: int = 40) -> dict:
        self._require_enabled()
        self.ensure_production_registered()
        cand = self.db.one("SELECT * FROM model_versions WHERE id=?", (model_id,))
        prod = self.db.one("SELECT * FROM model_versions WHERE status='production'")
        ds = self.db.one("SELECT holdout_path FROM training_datasets WHERE id=?", (cand["dataset_id"],)) \
            if cand["dataset_id"] else self.db.one("SELECT holdout_path FROM training_datasets ORDER BY id DESC LIMIT 1")
        if not ds or not Path(ds["holdout_path"]).exists():
            raise RuntimeError("no held-out dataset available - build a dataset first")
        cases = [json.loads(l) for l in Path(ds["holdout_path"]).read_text(encoding="utf-8").splitlines() if l.strip()]
        cases = cases[:max_cases]
        run_id = self.db.insert("eval_runs", {"kind": "model_comparison", "started_at": time.time()})
        totals = {"candidate": [], "production": []}
        for case in cases:
            msgs = case["messages"][:-1]
            reference = case["messages"][-1]["content"]
            question = msgs[-1]["content"]
            for label, model in (("candidate", cand["name"]), ("production", prod["name"])):
                answer = self.ollama.chat(msgs, model=model, temperature=0.2)
                v = self.ollama.chat_json(JUDGE_PROMPT, f"QUESTION:\n{question}\n\nCONTEXT (reference answer):\n"
                                                        f"{reference}\n\nANSWER:\n{answer}")
                score = sum(float(v.get(k, 0) or 0) for k in ("correct", "grounded", "helpful")) / 3
                totals[label].append(score)
                self.db.insert("evaluations", {"run_id": run_id, "kind": "model_case", "subject_id": model_id,
                                               "score": score, "details": json.dumps({"model": model, "label": label}),
                                               "created_at": time.time()})
        c = sum(totals["candidate"]) / len(cases) if cases else 0
        p = sum(totals["production"]) / len(cases) if cases else 0
        margin = self.cfg.get("training.promotion_margin", 0.03)
        verdict = "better" if c >= p + margin else "not_better"
        self.db.update("model_versions", model_id, {"eval_score": round(c, 4), "baseline_score": round(p, 4),
                                                    "status": "candidate" if verdict == "better" else "rejected",
                                                    "notes": f"{len(cases)} held-out cases"})
        metrics = {"candidate": round(c, 4), "production": round(p, 4), "cases": len(cases), "verdict": verdict}
        self.db.update("eval_runs", run_id, {"ended_at": time.time(), "metrics": json.dumps(metrics)})
        self.db.audit("trainer", "model_evaluated", cand["name"], **metrics)
        if verdict == "better" and self.cfg.get("training.auto_promote"):
            self.promote(model_id, approved_by="auto_promote setting")
        return metrics

    def promote(self, model_id: int, approved_by: str = "user") -> dict:
        self._require_enabled()
        cand = self.db.one("SELECT * FROM model_versions WHERE id=?", (model_id,))
        margin = self.cfg.get("training.promotion_margin", 0.03)
        if cand["eval_score"] is None or cand["baseline_score"] is None:
            raise PermissionError("refused: candidate has not been evaluated")
        if cand["eval_score"] < cand["baseline_score"] + margin:
            raise PermissionError("refused: candidate did not beat production by the required margin")
        backup_config(self.cfg)
        self.db.execute("UPDATE model_versions SET status='retired' WHERE status='production'")
        self.db.update("model_versions", model_id, {"status": "production", "promoted_at": time.time()})
        self.cfg.set("ollama.chat_model", cand["name"])
        self.cfg.save()
        self.db.audit("trainer", "model_promoted", cand["name"], approved_by=approved_by,
                      score=cand["eval_score"], baseline=cand["baseline_score"])
        return {"production": cand["name"]}

    def rollback(self) -> dict:
        prev = self.db.one("SELECT * FROM model_versions WHERE status='retired' ORDER BY promoted_at DESC LIMIT 1")
        cur = self.db.one("SELECT * FROM model_versions WHERE status='production'")
        if not prev:
            raise RuntimeError("no previous production model to roll back to")
        backup_config(self.cfg)
        if cur:
            self.db.update("model_versions", cur["id"], {"status": "rejected", "notes": "rolled back"})
        self.db.update("model_versions", prev["id"], {"status": "production", "promoted_at": time.time()})
        self.cfg.set("ollama.chat_model", prev["name"])
        self.cfg.save()
        self.db.audit("trainer", "model_rolled_back", prev["name"], from_model=cur["name"] if cur else None)
        return {"production": prev["name"]}


def backup_config(cfg: Config) -> Path | None:
    if not cfg.path or not cfg.path.exists():
        return None
    d = cfg.data_dir / "backups"
    d.mkdir(exist_ok=True)
    dst = d / f"config_{time.strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(cfg.path, dst)
    return dst
