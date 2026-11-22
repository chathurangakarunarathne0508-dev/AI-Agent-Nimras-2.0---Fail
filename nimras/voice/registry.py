"""Controlled voice-engine improvement: model registry, benchmark gate,
promotion with user authorisation, rollback, and update watching.

Rules (from the security requirements):
* The learner may DISCOVER and RECOMMEND models (release feeds, research
  topics). It never downloads or executes them.
* A candidate the user installed is benchmarked against production on the same
  self-test cases. Promotion requires (a) a passing benchmark and (b) an explicit
  user command. `voice.improvement.auto_promote` is honoured only as False.
* The previous production model is kept as `retired` so rollback is one command.
"""
from __future__ import annotations

import json
import time
from array import array

from ..backup import backup_all
from .audio import STT_RATE, resample
from .selftest import EN, SI, cer, wer
from .stt import BilingualWhisper

VOICE_TOPICS = [
    "Sinhala automatic speech recognition models",
    "Whisper fine-tuning for Sinhala",
    "Piper TTS Sinhala voices",
    "whisper.cpp releases and performance",
    "Open-source text-to-speech for low-resource languages",
]


class VoiceModelRegistry:
    def __init__(self, rt):
        self.rt, self.db, self.cfg = rt, rt.db, rt.cfg

    def ensure_production(self) -> None:
        cur = {("stt", None): self.cfg.get("voice.stt.model")}
        for lang, f in (self.cfg.get("voice.stt.language_models") or {}).items():
            cur[("stt_lang", lang)] = f
        for (kind, lang), name in cur.items():
            if name and not self.db.one("SELECT id FROM voice_models WHERE kind=? AND IFNULL(language,'')=? AND "
                                        "status='production'", (kind, lang or "")):
                self.db.insert("voice_models", {"kind": kind, "language": lang, "name": name, "status": "production",
                                                "source": "config", "created_at": time.time(),
                                                "promoted_at": time.time()})

    def list(self) -> list[dict]:
        self.ensure_production()
        return [dict(r) | {"metrics": json.loads(r["metrics"]) if r["metrics"] else None}
                for r in self.db.all("SELECT * FROM voice_models ORDER BY id DESC")]

    def register(self, kind: str, name: str, language: str | None = None, source: str = "user",
                 notes: str = "") -> int:
        mid = self.db.insert("voice_models", {"kind": kind, "language": language, "name": name, "status": "candidate",
                                              "source": source, "created_at": time.time(), "notes": notes})
        self.db.audit("user", "voice_model_registered", f"voice_model:{mid}", name=name, kind=kind)
        return mid

    # ------------------------------------------------------------- benchmark
    def _benchmark(self, model_file: str, lang: str) -> dict:
        eng = self.rt.voice
        voice = eng.tts.voice_for(lang)
        if not voice:
            raise RuntimeError(f"no TTS voice for {lang}: cannot build the benchmark set")
        stt = BilingualWhisper(eng.models_dir, model_file if lang == "en" else self.cfg.get("voice.stt.model"),
                               {} if lang == "en" else {lang: model_file},
                               base_port=self.cfg.get("voice.stt.port", 8771) + 20)
        if not stt.available()[0]:
            raise RuntimeError(stt.available()[1])
        cases = [t for _, t in (EN if lang == "en" else SI)]
        ws, cs, ls = [], [], []
        try:
            for text in cases:
                pcm, rate = voice.synthesize(eng.tts.prepare(lang, text))
                t = stt.transcribe(resample(pcm, rate, STT_RATE), lang)
                hyp = eng.vocab.normalize(t.text)[0]
                ws.append(wer(text, hyp))
                cs.append(cer(text, hyp))
                ls.append(t.latency_s)
        finally:
            stt.close()
        n = len(cases)
        return {"language": lang, "cases": n, "wer": round(sum(ws) / n, 4), "cer": round(sum(cs) / n, 4),
                "latency": round(sum(ls) / n, 3), "evidence": "synthetic round-trip", "at": time.time()}

    def evaluate(self, model_id: int) -> dict:
        cand = self.db.one("SELECT * FROM voice_models WHERE id=?", (model_id,))
        if not cand:
            raise ValueError("unknown model")
        lang = cand["language"] or "en"
        prod = self.db.one("SELECT * FROM voice_models WHERE kind=? AND IFNULL(language,'')=IFNULL(?,'') AND "
                           "status='production'", (cand["kind"], cand["language"]))
        cm = self._benchmark(cand["name"], lang)
        pm = self._benchmark(prod["name"], lang) if prod else None
        margin = self.cfg.get("voice.improvement.max_wer_regression", 0.0)
        passed = pm is None or (cm["wer"] <= pm["wer"] + margin and cm["latency"] <= max(pm["latency"] * 1.5, 3.0))
        result = {"candidate": cm, "production": pm, "passed": passed,
                  "rule": f"candidate WER <= production WER + {margin} and latency <= 1.5x production"}
        self.db.execute("UPDATE voice_models SET metrics=? WHERE id=?", (json.dumps(result), model_id))
        if prod and pm:
            self.db.execute("UPDATE voice_models SET metrics=? WHERE id=?", (json.dumps({"self": pm}), prod["id"]))
        self.db.audit("voice", "voice_model_evaluated", f"voice_model:{model_id}", passed=passed,
                      cand_wer=cm["wer"], prod_wer=pm and pm["wer"])
        return result

    # ------------------------------------------------------------- promotion
    def promote(self, model_id: int, authorised_by_user: bool) -> dict:
        if not authorised_by_user:
            raise PermissionError("promotion of a speech model requires explicit user authorisation")
        cand = self.db.one("SELECT * FROM voice_models WHERE id=?", (model_id,))
        metrics = json.loads(cand["metrics"]) if cand and cand["metrics"] else None
        if not metrics or not metrics.get("passed"):
            raise PermissionError("candidate has not passed the benchmark (run evaluate first)")
        backup_all(self.cfg, self.db)                 # config + db snapshot before the swap
        prod = self.db.one("SELECT * FROM voice_models WHERE kind=? AND IFNULL(language,'')=IFNULL(?,'') AND "
                           "status='production'", (cand["kind"], cand["language"]))
        if prod:
            self.db.execute("UPDATE voice_models SET status='retired' WHERE id=?", (prod["id"],))
        self.db.execute("UPDATE voice_models SET status='production', promoted_at=? WHERE id=?",
                        (time.time(), model_id))
        self._apply(cand)
        self.db.audit("user", "voice_model_promoted", f"voice_model:{model_id}", name=cand["name"],
                      previous=prod and prod["name"])
        return {"ok": True, "production": cand["name"], "previous": prod and prod["name"]}

    def rollback(self, kind: str, language: str | None = None) -> dict:
        prod = self.db.one("SELECT * FROM voice_models WHERE kind=? AND IFNULL(language,'')=IFNULL(?,'') AND "
                           "status='production'", (kind, language))
        prev = self.db.one("SELECT * FROM voice_models WHERE kind=? AND IFNULL(language,'')=IFNULL(?,'') AND "
                           "status='retired' ORDER BY promoted_at DESC, id DESC LIMIT 1", (kind, language))
        if not prev:
            raise ValueError("no previous model to roll back to")
        if prod:
            self.db.execute("UPDATE voice_models SET status='rejected', notes=COALESCE(notes,'')||' rolled back' "
                            "WHERE id=?", (prod["id"],))
        self.db.execute("UPDATE voice_models SET status='production', promoted_at=? WHERE id=?",
                        (time.time(), prev["id"]))
        self._apply(prev)
        self.db.audit("user", "voice_model_rolled_back", f"voice_model:{prev['id']}", name=prev["name"])
        return {"ok": True, "production": prev["name"]}

    def _apply(self, row) -> None:
        if row["kind"] == "stt":
            self.cfg.set("voice.stt.model", row["name"])
        elif row["kind"] == "stt_lang":
            lm = dict(self.cfg.get("voice.stt.language_models") or {})
            lm[row["language"]] = row["name"]
            self.cfg.set("voice.stt.language_models", lm)
        self.cfg.save()
        # Rebuild the engine on next use so the new model is loaded.
        old = getattr(self.rt, "_voice", None)
        if old:
            old.close()
            self.rt._voice = None

    # ------------------------------------------------------------- discovery
    def check_updates(self) -> dict:
        """Read release feeds through the policy-enforcing fetcher and record
        recommendations. Nothing is downloaded or installed."""
        from ..learning.discovery import parse_feed
        fetcher = self.rt.pipeline.discoverer.fetcher
        found = []
        for feed in self.cfg.get("voice.improvement.watch_feeds", []):
            try:
                items = parse_feed(fetcher.get_text(feed))[:3]
            except Exception as e:  # noqa: BLE001
                self.db.audit("voice", "voice_feed_failed", feed, error=str(e)[:200])
                continue
            for it in items:
                key = f"voice_release_seen:{it.url}"
                if self.db.kv_get(key):
                    continue
                self.db.kv_set(key, time.time())
                found.append({"title": it.title, "url": it.url, "published": it.published})
                self.db.insert("voice_models", {"kind": "recommendation", "name": it.title[:200], "source": it.url,
                                                "status": "recommended", "created_at": time.time(),
                                                "notes": "release seen in feed - review and install manually; "
                                                         "it will be benchmarked before any promotion"})
        if not self.db.kv_get("voice_topics_seeded"):
            for t in VOICE_TOPICS:
                self.rt.pipeline.enqueue_topic(t, origin="voice", priority=0.6)
            self.db.kv_set("voice_topics_seeded", True)
        self.db.kv_set("voice_updates_checked_at", time.time())
        if found:
            self.db.audit("voice", "voice_updates_found", None, count=len(found))
        return {"new_releases": found}


__all__ = ["VoiceModelRegistry", "VOICE_TOPICS", "array"]
