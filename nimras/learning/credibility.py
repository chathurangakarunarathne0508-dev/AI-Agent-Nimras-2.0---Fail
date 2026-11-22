"""Source credibility: prior by domain type, adjusted by learned track record."""
from __future__ import annotations

import time
from urllib.parse import urlparse

from ..db import Database

# Official standards bodies, vendors' own docs, primary research archives.
_AUTHORITATIVE = {
    "python.org": 0.95, "docs.python.org": 0.97, "peps.python.org": 0.97,
    "rust-lang.org": 0.95, "doc.rust-lang.org": 0.97, "go.dev": 0.95, "nodejs.org": 0.95,
    "kernel.org": 0.95, "ietf.org": 0.96, "rfc-editor.org": 0.97, "w3.org": 0.95,
    "whatwg.org": 0.95, "developer.mozilla.org": 0.93, "learn.microsoft.com": 0.92,
    "docs.github.com": 0.92, "docs.docker.com": 0.93, "kubernetes.io": 0.93,
    "nist.gov": 0.96, "csrc.nist.gov": 0.97, "nvd.nist.gov": 0.96, "cisa.gov": 0.95,
    "owasp.org": 0.92, "mitre.org": 0.93, "attack.mitre.org": 0.94, "cve.org": 0.95,
    "ollama.com": 0.93, "pytorch.org": 0.93, "tensorflow.org": 0.92, "huggingface.co": 0.85,
    "arxiv.org": 0.78, "export.arxiv.org": 0.78, "acm.org": 0.92, "ieee.org": 0.92,
    "nature.com": 0.92, "science.org": 0.92, "nih.gov": 0.94, "who.int": 0.92,
    "postgresql.org": 0.95, "sqlite.org": 0.96, "apache.org": 0.92, "gnu.org": 0.93,
    "intel.com": 0.88, "amd.com": 0.88, "nvidia.com": 0.88, "arm.com": 0.88,
}
_REPUTABLE = {
    "wikipedia.org": 0.72, "github.com": 0.70, "stackoverflow.com": 0.62,
    "readthedocs.io": 0.80, "readthedocs.org": 0.80, "lwn.net": 0.82,
    "arstechnica.com": 0.72, "theregister.com": 0.68, "zdnet.com": 0.62,
    "realpython.com": 0.75, "freecodecamp.org": 0.68, "geeksforgeeks.org": 0.55,
    "techcrunch.com": 0.62, "wired.com": 0.66, "reuters.com": 0.80, "bbc.co.uk": 0.78,
    "github.blog": 0.85, "blog.python.org": 0.92, "openai.com": 0.80, "anthropic.com": 0.85,
    "pypi.org": 0.80, "npmjs.com": 0.72, "crates.io": 0.80,
}
_LOW = {
    "medium.com": 0.45, "dev.to": 0.48, "quora.com": 0.30, "reddit.com": 0.38,
    "blogspot.com": 0.40, "wordpress.com": 0.40, "substack.com": 0.45, "hashnode.dev": 0.45,
}
_KIND_PRIOR = {"paper": 0.78, "docs": 0.85, "encyclopedia": 0.72, "rss": 0.6, "web": 0.5}


def domain_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _lookup(host: str, table: dict[str, float]) -> float | None:
    parts = host.split(".")
    for i in range(len(parts) - 1):
        cand = ".".join(parts[i:])
        if cand in table:
            return table[cand]
    return None


def registrable(host: str) -> str:
    """Approximate eTLD+1 so docs.x.org and blog.x.org count as ONE independent source."""
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in ("co", "ac", "gov", "org", "com", "edu") and len(parts[-1]) == 2:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def prior_credibility(url: str, kind: str = "web") -> float:
    host = domain_of(url)
    for table in (_AUTHORITATIVE, _REPUTABLE, _LOW):
        v = _lookup(host, table)
        if v is not None:
            return v
    if host.endswith((".gov", ".mil")) or ".gov." in host:
        return 0.90
    if host.endswith(".edu") or ".ac." in host:
        return 0.80
    if host.startswith(("docs.", "developer.", "developers.", "learn.")):
        return max(0.82, _KIND_PRIOR.get(kind, 0.5))
    return _KIND_PRIOR.get(kind, 0.5)


class CredibilityModel:
    def __init__(self, db: Database):
        self.db = db

    def score(self, url: str, kind: str = "web", injection_hits: int = 0,
              has_date: bool = False, https: bool | None = None) -> float:
        s = prior_credibility(url, kind)
        rep = self.db.one("SELECT * FROM domain_reputation WHERE domain=?", (domain_of(url),))
        if rep:
            n = rep["supported"] + rep["contradicted"]
            if n:
                # Bayesian-ish shrinkage: track record matters more as evidence grows.
                track = rep["supported"] / n
                w = min(0.5, n / 40)
                s = (1 - w) * s + w * track
            if rep["injections"]:
                s -= min(0.3, 0.1 * rep["injections"])
        if injection_hits:
            s -= 0.25
        if has_date:
            s += 0.03
        if https is None:
            https = url.startswith("https://")
        if not https:
            s -= 0.05
        return round(max(0.05, min(0.99, s)), 3)

    def record(self, url: str, supported: int = 0, contradicted: int = 0, injections: int = 0) -> None:
        self.db.execute(
            "INSERT INTO domain_reputation(domain,supported,contradicted,injections,updated_at) VALUES(?,?,?,?,?) "
            "ON CONFLICT(domain) DO UPDATE SET supported=supported+excluded.supported, "
            "contradicted=contradicted+excluded.contradicted, injections=injections+excluded.injections, "
            "updated_at=excluded.updated_at",
            (domain_of(url), supported, contradicted, injections, time.time()),
        )
