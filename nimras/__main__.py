"""N.I.M.R.A.S. command line.

  python -m nimras serve                 Learning Centre UI + background learning
  python -m nimras learn [--topic T]...  run one learning session now (foreground)
  python -m nimras status                health + counts
  python -m nimras ask "question"        chat once from the terminal
  python -m nimras evaluate              run the self-improvement engine once
  python -m nimras backup                snapshot knowledge DB + config
  python -m nimras dataset               build a fine-tuning dataset (training.enabled)
  python -m nimras model register|evaluate|promote|rollback ...
  python -m nimras voice setup [--with NAME]   download speech engines/models (user-initiated)
  python -m nimras voice status|hardware|test [--suite S]...
  python -m nimras voice say "text" [--out f.wav]        local TTS (language-aware)
  python -m nimras voice transcribe file.wav             local STT
  python -m nimras voice models | register | evaluate | promote | rollback
"""
from __future__ import annotations

import argparse
import json
import signal
import sys

from .config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nimras", description="N.I.M.R.A.S. autonomous learning intelligence")
    ap.add_argument("--config", help="path to nimras.config.json")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--no-scheduler", action="store_true", help="UI only; no background learning")
    l = sub.add_parser("learn")
    l.add_argument("--topic", action="append", help="topic to research (repeatable)")
    sub.add_parser("status")
    a = sub.add_parser("ask")
    a.add_argument("question")
    sub.add_parser("evaluate")
    sub.add_parser("backup")
    sub.add_parser("dataset")
    m = sub.add_parser("model")
    m.add_argument("action", choices=["register", "evaluate", "promote", "rollback", "list"])
    m.add_argument("--name")
    m.add_argument("--gguf")
    m.add_argument("--dataset", type=int)
    m.add_argument("--id", type=int)
    v = sub.add_parser("voice", help="N.I.M.R.A.S. Voice Engine")
    v.add_argument("action", choices=["setup", "status", "hardware", "test", "say", "transcribe", "models", "register",
                                      "evaluate", "promote", "rollback", "updates"])
    v.add_argument("arg", nargs="?", help="text for 'say', wav path for 'transcribe'")
    v.add_argument("--with", dest="extra", action="append", help="optional asset name for setup")
    v.add_argument("--suite", action="append", choices=["synthetic", "tts", "recorded", "pipeline"])
    v.add_argument("--out", help="output wav for 'say'")
    v.add_argument("--kind", choices=["stt", "stt_lang"], default="stt_lang")
    v.add_argument("--language")
    v.add_argument("--name")
    v.add_argument("--id", type=int)
    v.add_argument("--yes", action="store_true", help="authorise a promotion")
    args = ap.parse_args(argv)

    if args.cmd == "voice" and args.action == "setup":
        # Explicit user action: the only path that downloads speech software/models.
        from pathlib import Path
        from .voice import assets
        cfg = Config.load(args.config)
        root = Path(cfg.get("voice.models_dir", "voice_models"))
        root = root if root.is_absolute() else cfg.path.parent / root
        names = list(assets.ASSETS) + (args.extra or [])
        assets.install(root, names)
        print("Voice engine assets installed in", root)
        return 0

    from .runtime import Runtime  # deferred so --help is instant
    rt = Runtime(Config.load(args.config))
    out = lambda d: print(json.dumps(d, indent=2, default=str))  # noqa: E731

    if args.cmd == "voice":
        return _voice(rt, args, out)
    if args.cmd == "serve":
        from .server import serve
        if not args.no_scheduler:
            rt.scheduler.start()
        signal.signal(signal.SIGINT, lambda *_: (rt.scheduler.shutdown(), sys.exit(0)))
        serve(rt, args.host, args.port)
    elif args.cmd == "learn":
        res = rt.scheduler.run_session(trigger="manual", topic_names=args.topic)
        out(res)
    elif args.cmd == "status":
        d = rt.dashboard()
        out({"status": d["status"], "counts": d["counts"]})
    elif args.cmd == "ask":
        print(rt.chat.answer(args.question)["reply"])
    elif args.cmd == "evaluate":
        out(rt.evaluator.run())
    elif args.cmd == "backup":
        out(rt.backup())
    elif args.cmd == "dataset":
        out(rt.training.build_dataset())
    elif args.cmd == "model":
        t = rt.training
        if args.action == "list":
            t.ensure_production_registered()
            out([dict(r) for r in rt.db.all("SELECT * FROM model_versions ORDER BY id")])
        elif args.action == "register":
            out({"model_id": t.register_candidate(args.name, args.dataset, args.gguf)})
        elif args.action == "evaluate":
            out(t.evaluate_candidate(args.id))
        elif args.action == "promote":
            out(t.promote(args.id))
        elif args.action == "rollback":
            out(t.rollback())
    return 0




def _voice(rt, args, out) -> int:
    from pathlib import Path
    from .voice.registry import VoiceModelRegistry
    eng = rt.voice
    try:
        if args.action == "status":
            out(eng.status())
        elif args.action == "hardware":
            from .voice.hardware import inspect
            out(inspect(eng.models_dir))
        elif args.action == "test":
            from .voice.selftest import VoiceSelfTest
            res = VoiceSelfTest(rt).run(tuple(args.suite or ["synthetic", "tts", "recorded", "pipeline"]))
            out(res)
        elif args.action == "say":
            from .voice.language import analyze
            text = args.arg or "Hello, I am N.I.M.R.A.S."
            lang = "si" if analyze(text).code in ("si", "mixed") else "en"
            r = eng.tts.synthesize(text, lang)
            dest = Path(args.out or "nimras-say.wav")
            dest.write_bytes(r["wav"])
            out({"file": str(dest), "segments": r["segments"], "missing": r["missing"], "seconds": r.get("seconds")})
        elif args.action == "transcribe":
            from .voice.audio import STT_RATE, preprocess, read_wav, resample
            pcm, rate = read_wav(Path(args.arg).read_bytes())
            t = eng.stt.transcribe(preprocess(resample(pcm, rate, STT_RATE), STT_RATE), args.language or "auto",
                                   eng.vocab.initial_prompt())
            text, changes = eng.vocab.normalize(t.text)
            out({"text": text, "raw": t.text, "normalizations": changes, "language": t.language,
                 "language_probs": t.language_probs, "latency_s": round(t.latency_s, 3), "model": t.model})
        elif args.action == "models":
            out(VoiceModelRegistry(rt).list())
        elif args.action == "register":
            out({"id": VoiceModelRegistry(rt).register(args.kind, args.name, args.language)})
        elif args.action == "evaluate":
            out(VoiceModelRegistry(rt).evaluate(args.id))
        elif args.action == "promote":
            out(VoiceModelRegistry(rt).promote(args.id, authorised_by_user=args.yes))
        elif args.action == "rollback":
            out(VoiceModelRegistry(rt).rollback(args.kind, args.language))
        elif args.action == "updates":
            out(VoiceModelRegistry(rt).check_updates())
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
