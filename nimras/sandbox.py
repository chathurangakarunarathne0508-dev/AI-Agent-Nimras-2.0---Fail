"""Isolated execution of N.I.M.R.A.S.-generated example code.

Only code WRITTEN BY THE MODEL is ever run - never anything downloaded - and only
inside a locked-down container: no network, read-only root, no capabilities,
memory/CPU/PID limits, hard timeout. If Docker is unavailable, code is not run.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass
class SandboxResult:
    ran: bool
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    reason: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Sandbox:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._cache: tuple[float, tuple[bool, str]] | None = None

    def available(self) -> tuple[bool, str]:
        if self.cfg.get("sandbox.mode") != "docker":
            return False, "sandbox disabled in settings"
        if self._cache and time.time() - self._cache[0] < 60:
            return self._cache[1]
        if not shutil.which("docker"):
            res = (False, "docker not installed")
        else:
            try:
                r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                   capture_output=True, text=True, timeout=15)
                res = (r.returncode == 0, "ok" if r.returncode == 0 else "docker daemon not running")
            except (OSError, subprocess.TimeoutExpired):
                res = (False, "docker not responding")
        self._cache = (time.time(), res)
        return res

    def run_python(self, code: str) -> SandboxResult:
        ok, why = self.available()
        if not ok:
            return SandboxResult(False, False, None, "", "", why)
        timeout = int(self.cfg.get("sandbox.timeout_seconds", 30))
        name = f"nimras-sbx-{uuid.uuid4().hex[:10]}"
        with tempfile.TemporaryDirectory(prefix="nimras_sbx_") as tmp:
            Path(tmp, "main.py").write_text(code, encoding="utf-8")
            cmd = [
                "docker", "run", "--rm", "--name", name,
                "--network", "none",
                "--read-only", "--tmpfs", "/tmp:rw,size=32m",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "64",
                "--memory", self.cfg.get("sandbox.memory", "256m"),
                "--cpus", str(self.cfg.get("sandbox.cpus", "0.5")),
                "--user", "65534:65534",
                "-v", f"{tmp}:/work:ro", "-w", "/work",
                self.cfg.get("sandbox.image", "python:3.12-slim"),
                # In-container timeout bounds the code; the outer one also covers an image pull.
                "timeout", "-s", "KILL", str(timeout), "python", "-I", "-B", "/work/main.py",
            ]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 60)
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "kill", name], capture_output=True)
                return SandboxResult(True, False, None, "", "", f"timed out after {timeout}s")
            if r.returncode == 137:
                return SandboxResult(True, False, 137, r.stdout[-4000:], r.stderr[-4000:],
                                     f"killed (timeout {timeout}s or memory limit)")
            return SandboxResult(True, r.returncode == 0, r.returncode, r.stdout[-4000:], r.stderr[-4000:])
