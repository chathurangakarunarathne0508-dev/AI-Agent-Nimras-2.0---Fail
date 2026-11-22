"""Minimal RFC 6455 WebSocket server side for the stdlib HTTP server.

Security: browsers do not apply CORS to WebSockets, so any website could try
to open ws://127.0.0.1:8765/ws/voice. The handshake therefore requires an
Origin that is this server itself (plus the existing Host check), and frames
from the client must be masked and size-limited.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import threading

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_FRAME = 2 * 1024 * 1024


class WSClosed(Exception):
    pass


def origin_allowed(origin: str | None, port: int) -> bool:
    if not origin:
        return False
    return origin.lower() in {f"http://127.0.0.1:{port}", f"http://localhost:{port}", f"http://[::1]:{port}"}


class WebSocket:
    def __init__(self, rfile, wfile):
        self.rfile, self.wfile = rfile, wfile
        self._wlock = threading.Lock()
        self.closed = False

    @staticmethod
    def handshake(handler, port: int) -> "WebSocket | None":
        h = handler.headers
        if (h.get("Upgrade", "").lower() != "websocket" or "upgrade" not in h.get("Connection", "").lower()
                or h.get("Sec-WebSocket-Version") != "13" or not h.get("Sec-WebSocket-Key")):
            handler.send_error(400, "WebSocket upgrade required")
            return None
        if not origin_allowed(h.get("Origin"), port):
            handler.send_error(403, "origin not allowed")
            return None
        accept = base64.b64encode(hashlib.sha1((h["Sec-WebSocket-Key"] + _GUID).encode()).digest()).decode()
        handler.send_response(101, "Switching Protocols")
        handler.send_header("Upgrade", "websocket")
        handler.send_header("Connection", "Upgrade")
        handler.send_header("Sec-WebSocket-Accept", accept)
        handler.end_headers()
        handler.wfile.flush()
        handler.close_connection = True
        return WebSocket(handler.rfile, handler.wfile)

    # ------------------------------------------------------------------ read
    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.rfile.read(n - len(buf))
            if not chunk:
                raise WSClosed()
            buf += chunk
        return buf

    def recv(self) -> tuple[str, bytes | str]:
        """Returns ("text", str) or ("binary", bytes). Handles ping/close and fragmentation."""
        message, kind = b"", None
        while True:
            b1, b2 = self._read_exact(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            masked, length = b2 & 0x80, b2 & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read_exact(8))[0]
            if not masked:
                self.close(1002)
                raise WSClosed()
            if length > MAX_FRAME or len(message) + length > MAX_FRAME:
                self.close(1009)
                raise WSClosed()
            mask = self._read_exact(4)
            raw = self._read_exact(length)
            if length:
                key = (mask * (length // 4 + 1))[:length]
                payload = (int.from_bytes(raw, "big") ^ int.from_bytes(key, "big")).to_bytes(length, "big")
            else:
                payload = b""
            if opcode == 0x8:
                self.close()
                raise WSClosed()
            if opcode == 0x9:
                self._send(0xA, bytes(payload))
                continue
            if opcode == 0xA:
                continue
            if opcode in (0x1, 0x2):
                kind = "text" if opcode == 0x1 else "binary"
                message = bytes(payload)
            elif opcode == 0x0:
                message += bytes(payload)
            if fin and kind:
                return (kind, message.decode("utf-8")) if kind == "text" else (kind, message)

    # ------------------------------------------------------------------ write
    def _send(self, opcode: int, payload: bytes) -> None:
        if self.closed and opcode != 0x8:
            raise WSClosed()
        n = len(payload)
        if n < 126:
            header = struct.pack(">BB", 0x80 | opcode, n)
        elif n < 65536:
            header = struct.pack(">BBH", 0x80 | opcode, 126, n)
        else:
            header = struct.pack(">BBQ", 0x80 | opcode, 127, n)
        with self._wlock:
            try:
                self.wfile.write(header + payload)
                self.wfile.flush()
            except OSError as e:
                self.closed = True
                raise WSClosed() from e

    def send_json(self, obj: dict) -> None:
        self._send(0x1, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def send_bytes(self, data: bytes) -> None:
        self._send(0x2, data)

    def close(self, code: int = 1000) -> None:
        if self.closed:
            return
        try:
            self._send(0x8, struct.pack(">H", code))
        except WSClosed:
            pass
        self.closed = True
