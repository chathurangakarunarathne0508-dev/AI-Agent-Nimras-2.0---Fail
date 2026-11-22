"""Hardware inspection and speech-model recommendation (CPU-first)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess


def _ps(cmd: str) -> str:
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True,
                           timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        return r.stdout.decode(errors="replace").strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def inspect(models_dir=None) -> dict:
    info: dict = {"os": f"{platform.system()} {platform.release()} ({platform.version()})",
                  "python": platform.python_version(), "cpu_threads": os.cpu_count()}
    if os.name == "nt":
        info["cpu"] = _ps("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        ram = _ps("(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory")
        info["ram_gb"] = round(int(ram) / 2**30, 1) if ram.isdigit() else None
        free = _ps("(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory")
        info["ram_free_gb"] = round(int(free) / 2**20, 1) if free.isdigit() else None
        gpus = _ps("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }")
        info["gpus"] = [g for g in gpus.splitlines() if g.strip()]
        audio = _ps("Get-CimInstance Win32_SoundDevice | ForEach-Object { $_.Name + ' [' + $_.Status + ']' }")
        info["audio_devices"] = [a for a in audio.splitlines() if a.strip()]
    else:
        info["cpu"] = platform.processor()
        try:
            pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            info["ram_gb"] = round(pages * size / 2**30, 1)
        except (ValueError, OSError, AttributeError):
            info["ram_gb"] = None
        info["gpus"], info["audio_devices"] = [], []
    info["nvidia"] = bool(shutil.which("nvidia-smi"))
    if info["nvidia"]:
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                               capture_output=True, timeout=10)
            info["nvidia_gpus"] = r.stdout.decode().strip().splitlines()
        except (OSError, subprocess.TimeoutExpired):
            pass
    if models_dir:
        du = shutil.disk_usage(models_dir)
        info["disk_free_gb"] = round(du.free / 2**30, 1)
    info["recommendation"] = recommend(info)
    return info


def recommend(info: dict) -> dict:
    """Pick model sizes. Does not assume a GPU."""
    ram = info.get("ram_gb") or 8
    threads = info.get("cpu_threads") or 4
    if info.get("nvidia"):
        stt = "ggml-large-v3-turbo-q5_0.bin (with the CUDA build of whisper.cpp)"
        why = "NVIDIA GPU detected: the large-v3-turbo model is practical."
    elif ram >= 12 and threads >= 8:
        stt = "ggml-small-q5_1.bin + ggml-sinhala-small-q5_1.bin"
        why = ("CPU-only: Whisper small (quantised) with an adaptive audio window is roughly 1.5-2 s per short "
               "utterance on this class of CPU. Medium/large would be 3-10x slower without a GPU.")
    else:
        stt = "ggml-base-q5_1.bin (+ Sinhala small if RAM allows)"
        why = "Limited CPU/RAM: the base model keeps latency acceptable at some accuracy cost."
    llm = ("gemma3:4b or qwen2.5:7b for usable Sinhala; llama3.2:1b/3b are fast but weak in Sinhala"
           if ram >= 12 else "llama3.2:3b (Sinhala quality will be limited)")
    return {"stt": stt, "tts": "Piper medium voices (real-time factor ~0.1 on CPU)", "llm": llm,
            "threads": max(2, min(8, threads - 2)), "why": why}
