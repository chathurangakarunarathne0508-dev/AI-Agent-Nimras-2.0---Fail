"""Audio utilities (stdlib only): WAV I/O, resampling, preprocessing and
energy-based voice activity detection.

Audio is 16-bit signed little-endian mono PCM throughout. numpy is used when
available for speed; the pure-Python paths are exact fallbacks.
"""
from __future__ import annotations

import io
import math
import struct
import wave
from array import array

try:  # optional acceleration
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

STT_RATE = 16_000


def pcm_from_bytes(b: bytes) -> array:
    a = array("h")
    a.frombytes(b[: len(b) - len(b) % 2])
    return a


def wav_bytes(pcm: array | bytes, rate: int) -> bytes:
    if isinstance(pcm, array):
        pcm = pcm.tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def read_wav(data: bytes) -> tuple[array, int]:
    """Decode a PCM WAV (any channel count; downmixed to mono)."""
    with wave.open(io.BytesIO(data), "rb") as w:
        ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"only 16-bit WAV is supported (got {width * 8}-bit)")
    pcm = pcm_from_bytes(raw)
    if ch > 1:
        pcm = array("h", (sum(pcm[i:i + ch]) // ch for i in range(0, len(pcm), ch)))
    return pcm, rate


def rms(pcm: array) -> float:
    if not pcm:
        return 0.0
    if _np is not None:
        x = _np.frombuffer(pcm.tobytes(), dtype=_np.int16).astype(_np.float32)
        return float(_np.sqrt(_np.mean(x * x)) / 32768.0)
    return math.sqrt(sum(s * s for s in pcm) / len(pcm)) / 32768.0


def resample(pcm: array, src: int, dst: int) -> array:
    """Linear-interpolation resampler (adequate for speech at these rates)."""
    if src == dst or not pcm:
        return array("h", pcm)
    n_out = int(len(pcm) * dst / src)
    if _np is not None:
        x = _np.frombuffer(pcm.tobytes(), dtype=_np.int16).astype(_np.float32)
        t = _np.linspace(0, len(x) - 1, n_out)
        return array("h", _np.interp(t, _np.arange(len(x)), x).astype(_np.int16).tobytes())
    out = array("h", bytes(2 * n_out))
    step = (len(pcm) - 1) / max(1, n_out - 1)
    for i in range(n_out):
        p = i * step
        j = int(p)
        f = p - j
        b = pcm[j + 1] if j + 1 < len(pcm) else pcm[j]
        out[i] = int(pcm[j] + (b - pcm[j]) * f)
    return out


def preprocess(pcm: array, rate: int) -> array:
    """Light server-side clean-up on top of the browser's WebRTC noise
    suppression / echo cancellation: DC removal, 80 Hz high-pass (rumble, desk
    thumps) and peak normalisation to about -3 dBFS so quiet speakers are not lost."""
    if not pcm:
        return pcm
    if _np is not None:
        x = _np.frombuffer(pcm.tobytes(), dtype=_np.int16).astype(_np.float32)
        x -= x.mean()
        a = math.exp(-2 * math.pi * 80 / rate)
        ys, prev_x, prev_y = [], 0.0, 0.0
        # one-pole high-pass; vectorising an IIR is not worth it at these sizes
        for xi in x.tolist():
            prev_y = a * (prev_y + xi - prev_x)
            prev_x = xi
            ys.append(prev_y)
        y = _np.asarray(ys, dtype=_np.float32)
        peak = float(_np.abs(y).max()) or 1.0
        gain = min(8.0, 0.7 * 32767 / peak)
        return array("h", _np.clip(y * gain, -32768, 32767).astype(_np.int16).tobytes())
    mean = sum(pcm) / len(pcm)
    a = math.exp(-2 * math.pi * 80 / rate)
    out, prev_x, prev_y = [], 0.0, 0.0
    for s in pcm:
        xs = s - mean
        prev_y = a * (prev_y + xs - prev_x)
        prev_x = xs
        out.append(prev_y)
    peak = max(abs(v) for v in out) or 1.0
    gain = min(8.0, 0.7 * 32767 / peak)
    return array("h", (max(-32768, min(32767, int(v * gain))) for v in out))


class EnergyVAD:
    """Frame-level VAD with an adaptive noise floor.

    The browser runs the same algorithm for live endpointing; the server copy
    is used to trim uploads, reject utterances that contain no speech, and by
    the self-test module.
    """

    def __init__(self, rate: int = STT_RATE, frame_ms: int = 20, start_ratio: float = 3.0,
                 min_level: float = 0.012, min_speech_ms: int = 200, max_threshold: float = 0.05):
        self.max_threshold = max_threshold
        self.rate, self.frame = rate, rate * frame_ms // 1000
        self.frame_ms = frame_ms
        self.start_ratio, self.min_level, self.min_speech_ms = start_ratio, min_level, min_speech_ms

    def frames(self, pcm: array):
        for i in range(0, len(pcm) - self.frame + 1, self.frame):
            yield pcm[i:i + self.frame]

    def speech_mask(self, pcm: array) -> list[bool]:
        levels = [rms(f) for f in self.frames(pcm)]
        if not levels:
            return []
        floor = sorted(levels)[max(0, len(levels) // 10)]  # 10th percentile = noise floor
        # Cap the threshold: a clip that is speech end-to-end has no quiet frames,
        # so its "floor" is the speech itself.
        thr = max(self.min_level, min(floor * self.start_ratio, self.max_threshold))
        return [lv > thr for lv in levels]

    def speech_ms(self, pcm: array) -> int:
        return sum(self.speech_mask(pcm)) * self.frame_ms

    def has_speech(self, pcm: array) -> bool:
        return self.speech_ms(pcm) >= self.min_speech_ms

    def trim(self, pcm: array, pad_ms: int = 250) -> array:
        mask = self.speech_mask(pcm)
        if not any(mask):
            return array("h")
        first = mask.index(True)
        last = len(mask) - 1 - mask[::-1].index(True)
        pad = pad_ms // self.frame_ms
        a = max(0, first - pad) * self.frame
        b = min(len(pcm), (last + 1 + pad) * self.frame)
        return pcm[a:b]


def trim_silence_edges(pcm: array, rate: int, level: float = 0.01, keep_ms: int = 40) -> array:
    """Trim leading/trailing near-silence from synthesized speech so joined
    segments don't have unnatural gaps."""
    frame = max(1, rate // 100)
    n = len(pcm)
    start, end = 0, n
    while start + frame < n and rms(pcm[start:start + frame]) < level:
        start += frame
    while end - frame > start and rms(pcm[end - frame:end]) < level:
        end -= frame
    keep = rate * keep_ms // 1000
    return pcm[max(0, start - keep):min(n, end + keep)]


def concat_with_crossfade(parts: list[array], rate: int, gap_ms: int = 90, fade_ms: int = 12) -> array:
    """Join speech segments with a short natural breath gap and tiny fades
    (prevents clicks at boundaries)."""
    out = array("h")
    fade = rate * fade_ms // 1000
    gap = array("h", bytes(2 * (rate * gap_ms // 1000)))
    for k, p in enumerate(parts):
        p = array("h", p)
        for i in range(min(fade, len(p) // 2)):
            g = i / fade
            p[i] = int(p[i] * g)
            p[-1 - i] = int(p[-1 - i] * g)
        if k:
            out.extend(gap)
        out.extend(p)
    return out


def tone(freq: float, ms: int, rate: int = STT_RATE, amp: float = 0.3) -> array:
    n = rate * ms // 1000
    return array("h", (int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)))


def add_noise(pcm: array, snr_db: float, seed: int = 7) -> array:
    """White noise at a given SNR (self-test: 'background noise' condition)."""
    import random
    rnd = random.Random(seed)
    sig = rms(pcm) * 32768 or 1.0
    noise_amp = sig / (10 ** (snr_db / 20))
    return array("h", (max(-32768, min(32767, int(s + rnd.gauss(0, noise_amp)))) for s in pcm))


def change_speed(pcm: array, factor: float) -> array:
    """Naive speed change by resampling (pitch shifts too) - good enough to
    stress-test recognition at different speaking rates."""
    return resample(pcm, int(STT_RATE * factor), STT_RATE)


__all__ = ["STT_RATE", "wav_bytes", "read_wav", "rms", "resample", "preprocess", "EnergyVAD",
           "trim_silence_edges", "concat_with_crossfade", "pcm_from_bytes", "struct"]
