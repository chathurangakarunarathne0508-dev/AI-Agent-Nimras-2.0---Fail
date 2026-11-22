"""Convert a Hugging Face Whisper checkpoint (safetensors) to whisper.cpp's
ggml format, using numpy only (no torch / transformers needed, which matters
on Python 3.15 where those wheels are unavailable).

Port of whisper.cpp's models/convert-h5-to-ggml.py (MIT). The mel filterbank is
copied from an existing ggml model of the same family, so no extra download is
needed.
"""
from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path

import numpy as np

CONV_MAP = {
    "self_attn.k_proj": "attn.key", "self_attn.q_proj": "attn.query", "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out", "self_attn_layer_norm": "attn_ln",
    "encoder_attn.q_proj": "cross_attn.query", "encoder_attn.v_proj": "cross_attn.value",
    "encoder_attn.out_proj": "cross_attn.out", "encoder_attn_layer_norm": "cross_attn_ln",
    "fc1": "mlp.0", "fc2": "mlp.2", "final_layer_norm": "mlp_ln",
    "encoder.layer_norm.bias": "encoder.ln_post.bias", "encoder.layer_norm.weight": "encoder.ln_post.weight",
    "encoder.embed_positions.weight": "encoder.positional_embedding",
    "decoder.layer_norm.bias": "decoder.ln.bias", "decoder.layer_norm.weight": "decoder.ln.weight",
    "decoder.embed_positions.weight": "decoder.positional_embedding",
    "decoder.embed_tokens.weight": "decoder.token_embedding.weight",
    "encoder.conv1.weight": "encoder.conv1.weight", "encoder.conv1.bias": "encoder.conv1.bias",
    "encoder.conv2.weight": "encoder.conv2.weight", "encoder.conv2.bias": "encoder.conv2.bias",
}


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte<->unicode table used by Whisper's BPE vocabulary."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def read_safetensors(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        raw = mm[base + a: base + b]
        dt = meta["dtype"]
        if dt == "F32":
            arr = raw.view(np.float32)
        elif dt == "F16":
            arr = raw.view(np.float16)
        elif dt == "BF16":
            arr = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        out[name] = arr.reshape(meta["shape"])
    return out


def read_mel_filters(ggml_path: Path) -> np.ndarray:
    with ggml_path.open("rb") as f:
        magic = struct.unpack("<i", f.read(4))[0]
        if magic != 0x67676D6C:
            raise ValueError(f"{ggml_path} is not a ggml whisper model")
        f.read(4 * 11)                      # hparams
        n_mel, n_fft = struct.unpack("<ii", f.read(8))
        return np.frombuffer(f.read(4 * n_mel * n_fft), dtype=np.float32).reshape(n_mel, n_fft)


def _map_name(name: str) -> str | None:
    if name == "proj_out.weight":
        return None                          # tied to the token embedding
    nn = name.split(".")[1:]                 # drop leading "model."
    if len(nn) > 1 and nn[1] == "layers":
        nn[1] = "blocks"
        mid = ".".join(nn[3:-1])
        if mid == "encoder_attn.k_proj":
            mapped = "attn.key" if nn[0] == "encoder" else "cross_attn.key"
        else:
            mapped = CONV_MAP[mid]
        return ".".join(nn[:3] + [mapped] + nn[-1:])
    joined = ".".join(nn)
    return CONV_MAP.get(joined, joined)


def convert_hf_to_ggml(model_dir: Path, out_path: Path, mel_reference: Path | None) -> Path:
    model_dir = Path(model_dir)
    hp = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    vocab = json.loads((model_dir / "vocab.json").read_text(encoding="utf-8"))
    if not mel_reference:
        raise FileNotFoundError("a reference ggml whisper model is needed for the mel filterbank")
    filters = read_mel_filters(Path(mel_reference))
    if filters.shape[0] != hp["num_mel_bins"]:
        raise ValueError(f"mel bins differ: reference {filters.shape[0]} vs model {hp['num_mel_bins']}")
    tensors = read_safetensors(model_dir / "model.safetensors")
    byte_decoder = {v: k for k, v in bytes_to_unicode().items()}

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fout:
        w = lambda fmt, *v: fout.write(struct.pack(fmt, *v))  # noqa: E731
        w("<i", 0x67676D6C)
        for v in (hp["vocab_size"], hp["max_source_positions"], hp["d_model"], hp["encoder_attention_heads"],
                  hp["encoder_layers"], hp.get("max_target_positions", hp.get("max_length", 448)), hp["d_model"],
                  hp["decoder_attention_heads"], hp["decoder_layers"], hp["num_mel_bins"], 1):
            w("<i", v)
        w("<ii", *filters.shape)
        fout.write(filters.astype("<f4").tobytes())
        toks = sorted(vocab.items(), key=lambda kv: kv[1])
        w("<i", len(toks))
        for text, _ in toks:
            b = bytes(byte_decoder[c] for c in text)
            w("<i", len(b))
            fout.write(b)
        for name, arr in tensors.items():
            mapped = _map_name(name)
            if mapped is None:
                continue
            data = np.asarray(arr, dtype=np.float32).squeeze()
            if mapped in ("encoder.conv1.bias", "encoder.conv2.bias"):
                data = data.reshape(data.shape[0], 1)
            n_dims = data.ndim
            keep_f32 = n_dims < 2 or mapped in ("encoder.conv1.bias", "encoder.conv2.bias",
                                                "encoder.positional_embedding", "decoder.positional_embedding")
            data = data.astype("<f4" if keep_f32 else "<f2")
            nb = mapped.encode()
            w("<iii", n_dims, len(nb), 0 if keep_f32 else 1)
            for i in range(n_dims):
                w("<i", data.shape[n_dims - 1 - i])
            fout.write(nb)
            fout.write(np.ascontiguousarray(data).tobytes())
    return out_path


def quantize(models_root: Path, src: Path, dst: Path, qtype: str = "q5_1") -> Path:
    from .assets import find_file
    exe = find_file(Path(models_root) / "whisper.cpp", "whisper-quantize.exe") or \
        find_file(Path(models_root) / "whisper.cpp", "whisper-quantize")
    if not exe:
        raise FileNotFoundError("whisper-quantize not found")
    r = subprocess.run([str(exe.resolve()), str(Path(src).resolve()), str(Path(dst).resolve()), qtype],
                       capture_output=True, cwd=str(exe.parent.resolve()),
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0 or not Path(dst).exists():
        raise RuntimeError(f"quantize failed: {r.stderr.decode(errors='replace')[-400:]}")
    return Path(dst)
