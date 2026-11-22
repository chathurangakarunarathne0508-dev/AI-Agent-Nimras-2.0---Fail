"""Language identification and Sinhala-English code-switching helpers.

Text-level detection uses Unicode script (Sinhala block U+0D80-U+0DFF) plus a
Singlish lexicon (Sinhala written in Latin letters, which Whisper often emits for
Sinhala speech). It is combined with Whisper's acoustic language ID in
`decide_language`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SINHALA = re.compile(r"[඀-෿]")
LATIN = re.compile(r"[A-Za-z]")
_ZWJ = "‍"

# Frequent romanised Sinhala ("Singlish") words. Kept to words that are not
# also common English words, so English text is not misclassified.
SINGLISH = {
    "mata", "mage", "oya", "oyaa", "oyage", "api", "apita", "eya", "puluwan", "puluwanda", "karanna",
    "karanne", "kiyanna", "kiyala", "kiyanne", "mokakda", "mokadda", "monawada", "kohomada", "kohoma",
    "nehe", "naha", "ne", "ow", "owu", "eka", "ekak", "ekata", "denna", "dennako", "thiyenawa", "tiyenawa",
    "thiyenne", "nathi", "netha", "hari", "honda", "hondai", "machan", "aiyo", "ayyo", "ammo", "kawda",
    "koheda", "kiyada", "wenne", "wenawa", "una", "giya", "yanna", "enna", "balanna", "danna", "dannawa",
    "dannawada", "therenne", "therum", "sinhalen", "ingrisiyen", "ganna", "gana", "gena", "wage", "nisa",
    "habai", "hebei", "eth", "ehema", "mehema", "dan", "tikak", "godak", "poddak", "issara", "passe",
    "kalin", "eke", "ekeng", "walata", "wala", "kiyannako", "pennanna", "hadanna",
}

_REQ_SI = re.compile(r"\b(in|into|to|use|speak|answer in|reply in|explain (?:it |that |this )?in)\s+sinhala\b|"
                     r"\bsinhala(?:en|wen|yen)\b|සිංහලෙන්|සිංහලයෙන්", re.I)
_REQ_EN = re.compile(r"\b(in|into|to|use|speak|answer in|reply in|explain (?:it |that |this )?in)\s+english\b|"
                     r"\bingrisi(?:yen|en)\b|ඉංග්‍රීසියෙන්|ඉංග්රීසියෙන්", re.I)


@dataclass
class LangInfo:
    code: str           # "en" | "si" | "mixed"
    sinhala_ratio: float
    singlish_hits: int


def analyze(text: str) -> LangInfo:
    si = len(SINHALA.findall(text))
    la = len(LATIN.findall(text))
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for w in words if w in SINGLISH)
    letters = si + la
    ratio = si / letters if letters else 0.0
    if letters == 0:
        code = "en"
    elif ratio >= 0.75:
        code = "si"
    elif ratio >= 0.10:
        code = "mixed"
    elif words and hits >= 2 and hits / len(words) >= 0.2:
        code = "mixed"          # Singlish: Sinhala grammar in Latin letters
    else:
        code = "en"
    return LangInfo(code, round(ratio, 3), hits)


def requested_language(text: str) -> str | None:
    """Explicit instruction such as 'explain that in Sinhala' / 'සිංහලෙන් කියන්න'."""
    si, en = _REQ_SI.search(text), _REQ_EN.search(text)
    if si and en:
        return "si" if si.start() > en.start() else "en"   # the later instruction wins
    return "si" if si else "en" if en else None


def decide_language(text: str, whisper_lang: str | None, preferred: str | None = None) -> dict:
    """Choose the reply language.

    Priority: explicit request in this utterance > sticky user preference set
    by an earlier explicit request > what the user actually spoke.
    """
    info = analyze(text)
    spoken = info.code
    if whisper_lang == "si" and spoken == "en" and info.singlish_hits:
        spoken = "mixed"
    elif whisper_lang == "si" and spoken == "en" and not text.strip():
        spoken = "si"
    explicit = requested_language(text)
    if explicit:
        reply = explicit
    elif preferred in ("si", "en"):
        reply = preferred
    else:
        reply = "si" if spoken in ("si", "mixed") else "en"
    return {"spoken": spoken, "reply": reply, "explicit": explicit, "whisper": whisper_lang,
            "sinhala_ratio": info.sinhala_ratio, "singlish_hits": info.singlish_hits}


def segments(text: str) -> list[tuple[str, str]]:
    """Split text into (lang, chunk) runs for the TTS router.

    Digits, punctuation and spaces attach to the surrounding run, so a number
    inside a Sinhala sentence is read by the Sinhala voice.
    """
    runs: list[list] = []
    cur = None
    for ch in text:
        if SINHALA.match(ch) or ch == _ZWJ:
            lang = "si"
        elif LATIN.match(ch):
            lang = "en"
        else:
            lang = None
        if lang and lang != cur:
            if runs and cur is None:
                runs[-1][0] = lang
            if runs and runs[-1][0] == lang:
                runs[-1][1] += ch
            else:
                runs.append([lang, ch])
            cur = lang
        else:
            if runs:
                runs[-1][1] += ch
            else:
                runs.append([None, ch])
    out = []
    for lang, chunk in runs:
        lang = lang or (out[-1][0] if out else "en")
        if out and out[-1][0] == lang:
            out[-1] = (lang, out[-1][1] + chunk)
        else:
            out.append((lang, chunk))
    # Tiny Latin fragments wedged inside Sinhala (e.g. a single letter) are
    # merged into the Sinhala run unless they are known technical terms.
    merged: list[tuple[str, str]] = []
    for lang, chunk in out:
        if merged and lang == "en" and len(chunk.strip()) <= 1 and merged[-1][0] == "si":
            merged[-1] = ("si", merged[-1][1] + chunk)
        elif merged and merged[-1][0] == lang:
            merged[-1] = (lang, merged[-1][1] + chunk)
        else:
            merged.append((lang, chunk))
    return [(lang, c) for lang, c in merged if c.strip()]


_SENT_END = re.compile(r"(?<=[.!?។෴])\s+|(?<=[\n])")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming text buffer into complete sentences + remainder."""
    parts = _SENT_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    done = [p.strip() for p in parts[:-1] if p.strip()]
    return done, parts[-1]


LANG_NAMES = {"en": "English", "si": "Sinhala", "mixed": "Sinhala-English"}
