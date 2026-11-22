"""Speech-engine assets: pinned, user-initiated downloads of free/open-source
speech binaries and models.

Security: these files are only ever downloaded and unpacked when the *user* runs
`python -m nimras voice setup`. The autonomous learner never downloads or runs
software; it can only *recommend* model changes (see voice/registry.py).
Every file's SHA-256 is recorded in manifest.lock.json (and checked against
Hugging Face's published LFS hash when the server provides one).
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# name -> (url, relative destination, kind, approx MB, purpose)
ASSETS: dict[str, dict] = {
    "whisper_cpp": {
        "url": "https://github.com/ggml-org/whisper.cpp/releases/download/b5454/whisper-bin-x64.zip",
        "dest": "whisper.cpp/whisper-bin-x64.zip", "kind": "zip", "mb": 8.5,
        "purpose": "whisper.cpp CPU build (whisper-server / whisper-cli), MIT",
    },
    "piper": {
        "url": "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip",
        "dest": "piper/piper_windows_amd64.zip", "kind": "zip", "mb": 21.4,
        "purpose": "Piper TTS engine with espeak-ng, MIT",
    },
    "stt_whisper_small_q5": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
        "dest": "stt/ggml-small-q5_1.bin", "kind": "file", "mb": 182,
        "purpose": "Whisper small (multilingual, 5-bit quantised) - default on CPU",
    },
    "stt_whisper_base_q5": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
        "dest": "stt/ggml-base-q5_1.bin", "kind": "file", "mb": 57,
        "purpose": "Whisper base - fast language identification (en/si)",
    },
    "tts_en_lessac": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "dest": "tts/en_US-lessac-medium.onnx", "kind": "file", "mb": 61,
        "purpose": "English Piper voice",
    },
    "tts_en_lessac_cfg": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
        "dest": "tts/en_US-lessac-medium.onnx.json", "kind": "file", "mb": 0.01, "purpose": "voice config",
    },
    "tts_si_dilu": {
        "url": "https://huggingface.co/chaturadilan/dilu-tts/resolve/main/onnx/si_LK-dilu-medium.onnx",
        "dest": "tts/si_LK-dilu-medium.onnx", "kind": "file", "mb": 61,
        "purpose": "Dilu Sinhala Piper voice (si_LK)",
    },
    "tts_si_dilu_cfg": {
        "url": "https://huggingface.co/chaturadilan/dilu-tts/resolve/main/onnx/si_LK-dilu-medium.onnx.json",
        "dest": "tts/si_LK-dilu-medium.onnx.json", "kind": "file", "mb": 0.01, "purpose": "voice config",
    },
    # Stock Whisper cannot transcribe Sinhala usefully (measured: degenerate
    # repetition). This full fine-tune reports 15.52% WER / 4.36% CER on a
    # speaker-disjoint test set (Apache-2.0). It is converted locally to
    # whisper.cpp format and quantised; see voice/convert_whisper.py.
    "stt_si_sinhaspeech": {
        "repo": "SinhaSpeech/whisper-small-sinhala-run12-best",
        "revision": "main",
        "files": ["config.json", "vocab.json", "added_tokens.json", "model.safetensors"],
        "dest": "stt/src/sinhaspeech-small-run12", "kind": "hf_whisper",
        "output": "stt/ggml-sinhala-small-q5_1.bin", "quant": "q5_1", "mb": 922,
        "purpose": "Sinhala fine-tuned Whisper small (SinhaSpeech run12), converted + quantised",
    },
}

# Optional, larger models the user may choose (not downloaded by default).
OPTIONAL_ASSETS: dict[str, dict] = {
    "stt_whisper_medium_q5": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_0.bin",
        "dest": "stt/ggml-medium-q5_0.bin", "kind": "file", "mb": 515,
        "purpose": "Whisper medium - better Sinhala, ~3x slower than small on CPU",
    },
    "stt_whisper_large_v3_turbo_q5": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
        "dest": "stt/ggml-large-v3-turbo-q5_0.bin", "kind": "file", "mb": 547,
        "purpose": "Whisper large-v3-turbo - best multilingual accuracy here; needs a fast CPU or GPU",
    },
}


def _lock_path(root: Path) -> Path:
    return root / "manifest.lock.json"


def load_lock(root: Path) -> dict:
    p = _lock_path(root)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, log=print, retries: int = 30) -> str | None:
    """Resumable download (the connection here drops). Returns the server's
    published SHA-256 if it gave one (Hugging Face LFS x-linked-etag)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    published = None
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "nimras-voice-setup/2.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                etag = (r.headers.get("x-linked-etag") or "").strip('"')
                if len(etag) == 64:
                    published = etag
                total = r.headers.get("Content-Length")
                if have and r.status != 206:      # server ignored Range: restart
                    have = 0
                    part.unlink(missing_ok=True)
                total_n = (int(total) + have) if total else None
                mode = "ab" if have else "wb"
                done, last = have, time.time()
                with part.open(mode) as f:
                    while True:
                        chunk = r.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last > 10:
                            pct = f"{done * 100 / total_n:.0f}%" if total_n else f"{done >> 20} MB"
                            log(f"  {dest.name}: {pct}")
                            last = time.time()
                if total_n and done < total_n:
                    raise OSError(f"short read {done}/{total_n}")
            part.replace(dest)
            return published
        except urllib.error.HTTPError as e:
            if e.code == 416 and part.exists():   # already complete
                part.replace(dest)
                return published
            log(f"  {dest.name}: HTTP {e.code} (attempt {attempt})")
            if e.code in (401, 403, 404):
                raise
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            log(f"  {dest.name}: {e} (attempt {attempt}, resuming)")
        time.sleep(min(30, 2 * attempt))
    raise OSError(f"giving up on {url}")


def _safe_extract(zip_path: Path, out_dir: Path) -> None:
    out_dir = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as z:
        for m in z.infolist():
            target = (out_dir / m.filename).resolve()
            if not str(target).startswith(str(out_dir)):
                raise ValueError(f"unsafe path in archive: {m.filename}")
        z.extractall(out_dir)


def install(root: Path, names: list[str] | None = None, log=print) -> dict:
    """Download (or verify) assets. Idempotent: present files are re-hashed, not re-fetched."""
    catalogue = ASSETS | OPTIONAL_ASSETS
    names = names or list(ASSETS)
    lock = load_lock(root)
    for name in names:
        a = catalogue[name]
        if a["kind"] == "hf_whisper":
            _install_hf_whisper(root, name, a, lock, log)
            continue
        dest = root / a["dest"]
        if dest.exists() and name in lock and lock[name].get("sha256") == _sha256(dest):
            log(f"ok    {name}")
            continue
        log(f"fetch {name} (~{a['mb']} MB): {a['purpose']}")
        published = download(a["url"], dest, log)
        digest = _sha256(dest)
        if published and published != digest:
            dest.unlink(missing_ok=True)
            raise ValueError(f"{name}: SHA-256 mismatch (published {published}, got {digest})")
        if a["kind"] == "zip":
            _safe_extract(dest, dest.parent)
        lock[name] = {"url": a["url"], "sha256": digest, "verified_against_publisher": bool(published),
                      "installed_at": time.time(), "bytes": dest.stat().st_size}
        _lock_path(root).write_text(json.dumps(lock, indent=2), encoding="utf-8")
        log(f"done  {name}  sha256={digest[:16]}…")
    return lock


def _install_hf_whisper(root: Path, name: str, a: dict, lock: dict, log) -> None:
    out = root / a["output"]
    if out.exists() and name in lock and lock[name].get("sha256") == _sha256(out):
        log(f"ok    {name}")
        return
    src = root / a["dest"]
    hashes = {}
    for fn in a["files"]:
        dest = src / fn
        url = f"https://huggingface.co/{a['repo']}/resolve/{a['revision']}/{fn}"
        if not dest.exists():
            log(f"fetch {name}/{fn}")
            published = download(url, dest, log)
            digest = _sha256(dest)
            if published and published != digest:
                dest.unlink(missing_ok=True)
                raise ValueError(f"{name}/{fn}: SHA-256 mismatch")
        hashes[fn] = _sha256(dest)
    from .convert_whisper import convert_hf_to_ggml, quantize
    reference = find_file(root / "stt", "ggml-small-q5_1.bin")   # supplies the mel filterbank
    f16 = src / "ggml-model-f16.bin"
    log(f"convert {name} -> whisper.cpp format")
    convert_hf_to_ggml(src, f16, reference)
    if a.get("quant"):
        log(f"quantise {name} ({a['quant']})")
        quantize(root, f16, out, a["quant"])
        f16.unlink(missing_ok=True)
    else:
        f16.replace(out)
    lock[name] = {"repo": a["repo"], "revision": a["revision"], "source_sha256": hashes,
                  "sha256": _sha256(out), "installed_at": time.time(), "bytes": out.stat().st_size}
    _lock_path(root).write_text(json.dumps(lock, indent=2), encoding="utf-8")
    log(f"done  {name} -> {out.name}")


def find_file(root: Path, filename: str) -> Path | None:
    hits = sorted(root.rglob(filename)) if root.exists() else []
    return hits[0] if hits else None
