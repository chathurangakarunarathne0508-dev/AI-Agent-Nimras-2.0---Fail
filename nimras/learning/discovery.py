"""Internet discovery: search providers, feeds, and learned search strategy.

Search strategy improves itself: every provider and query template keeps a
success record (did it lead to stored, verified knowledge?) and selection uses
a UCB bandit, so useful strategies get used more while others are still probed.
"""
from __future__ import annotations

import html
import json
import math
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

from ..config import Config
from ..db import Database
from ..security import sanitize_query
from .fetcher import Fetcher, FetchError


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str
    published: str | None = None
    kind: str = "web"


QUERY_TEMPLATES = {
    "plain": "{topic}",
    "official_docs": "{topic} official documentation",
    "overview": "what is {topic} overview",
    "latest": "{topic} latest release news",
    "paper": "{topic} research paper",
    "compare": "{topic} compared to alternatives advantages limitations",
}


# ------------------------------------------------------------------ providers
class Provider:
    name = "base"

    def __init__(self, cfg: Config, fetcher: Fetcher, ollama=None):
        self.cfg, self.fetcher, self.ollama = cfg, fetcher, ollama

    def available(self) -> bool:
        return True

    def search(self, query: str, n: int) -> list[SearchResult]:
        raise NotImplementedError


class OllamaSearch(Provider):
    name = "ollama"

    def available(self) -> bool:
        return bool(self.ollama and self.ollama.web_api_key)

    def search(self, query, n):
        out = []
        for r in self.ollama.web_search(query, max_results=n):
            out.append(SearchResult(r.get("title", ""), r.get("url", ""), r.get("content", "")[:400], self.name))
        return out


class SearxngSearch(Provider):
    name = "searxng"

    def available(self) -> bool:
        return bool(self.cfg.get("internet.searxng_url"))

    def search(self, query, n):
        base = self.cfg.get("internet.searxng_url").rstrip("/")
        url = f"{base}/search?format=json&q={urllib.parse.quote(query)}"
        # Self-hosted SearXNG usually lives on the LAN; it is an explicit, trusted
        # endpoint, so it bypasses the public-only fetch policy deliberately.
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": self.cfg.get("internet.user_agent")})
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
        return [SearchResult(x.get("title", ""), x.get("url", ""), x.get("content", ""), self.name,
                             x.get("publishedDate")) for x in data.get("results", [])[:n]]


class WikipediaSearch(Provider):
    name = "wikipedia"

    def search(self, query, n):
        url = ("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit="
               f"{n}&srsearch={urllib.parse.quote(query)}")
        data = json.loads(self.fetcher.get_text(url))
        out = []
        for r in data.get("query", {}).get("search", []):
            title = r["title"]
            page = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
            snippet = re.sub(r"<[^>]+>", "", html.unescape(r.get("snippet", "")))
            out.append(SearchResult(title, page, snippet, self.name, kind="encyclopedia"))
        return out


class ArxivSearch(Provider):
    name = "arxiv"
    _NS = {"a": "http://www.w3.org/2005/Atom"}

    def search(self, query, n):
        q = urllib.parse.quote(f"all:{query}")
        url = f"https://export.arxiv.org/api/query?search_query={q}&max_results={n}&sortBy=relevance"
        root = ET.fromstring(self.fetcher.get_text(url))
        out = []
        for e in root.findall("a:entry", self._NS):
            title = " ".join((e.findtext("a:title", "", self._NS) or "").split())
            link = (e.findtext("a:id", "", self._NS) or "").replace("http://", "https://")
            summary = " ".join((e.findtext("a:summary", "", self._NS) or "").split())
            out.append(SearchResult(title, link, summary[:600], self.name,
                                    e.findtext("a:published", None, self._NS), kind="paper"))
        return out


class DuckDuckGoSearch(Provider):
    name = "duckduckgo"
    _RESULT = re.compile(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)

    def available(self) -> bool:
        return bool(self.cfg.get("internet.duckduckgo_enabled"))

    def search(self, query, n):
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)
        body = self.fetcher.get_text(url)
        out = []
        for href, title in self._RESULT.findall(body)[:n]:
            href = html.unescape(href)
            if "uddg=" in href:
                href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["uddg"][0])
            out.append(SearchResult(re.sub(r"<[^>]+>", "", html.unescape(title)), href, "", self.name))
        return out


PROVIDERS = {p.name: p for p in (OllamaSearch, SearxngSearch, WikipediaSearch, ArxivSearch, DuckDuckGoSearch)}


# ------------------------------------------------------------------ feeds
def parse_feed(xml_text: str) -> list[SearchResult]:
    """RSS 2.0 / Atom / RDF. Stdlib expat does not resolve external entities."""
    if "<!ENTITY" in xml_text[:5000]:
        return []  # refuse entity declarations outright (billion-laughs style payloads)
    root = ET.fromstring(xml_text)
    items: list[SearchResult] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag not in ("item", "entry"):
            continue
        fields: dict[str, str] = {}
        link = ""
        for c in el:
            ct = c.tag.split("}")[-1]
            if ct == "link":
                link = c.get("href") or (c.text or "").strip() or link
            elif c.text:
                fields.setdefault(ct, c.text.strip())
        title = " ".join(fields.get("title", "").split())
        desc = re.sub(r"<[^>]+>", " ", html.unescape(fields.get("description") or fields.get("summary") or ""))
        pub = fields.get("pubDate") or fields.get("published") or fields.get("updated") or fields.get("date")
        if title and link:
            items.append(SearchResult(title, link, " ".join(desc.split())[:500], "rss", pub, kind="rss"))
    return items


# ------------------------------------------------------------------ strategy
class StrategyBook:
    """UCB1 over providers and query templates, rewarded by downstream yield."""

    def __init__(self, db: Database):
        self.db = db

    def _stats(self, kind: str) -> dict[str, tuple[int, float]]:
        return {r["name"]: (r["uses"], r["successes"])
                for r in self.db.all("SELECT name, uses, successes FROM strategy_stats WHERE kind=?", (kind,))}

    def rank(self, kind: str, names: list[str]) -> list[str]:
        stats = self._stats(kind)
        total = sum(u for u, _ in stats.values()) + 1

        def ucb(name: str) -> float:
            uses, succ = stats.get(name, (0, 0.0))
            if uses == 0:
                return float("inf")  # untried strategies get explored first
            return succ / uses + math.sqrt(2 * math.log(total) / uses)

        return sorted(names, key=ucb, reverse=True)

    def record(self, kind: str, name: str, reward: float) -> None:
        self.db.execute(
            "INSERT INTO strategy_stats(kind,name,uses,successes) VALUES(?,?,1,?) "
            "ON CONFLICT(kind,name) DO UPDATE SET uses=uses+1, successes=successes+excluded.successes",
            (kind, name, max(0.0, min(1.0, reward))),
        )

    def report(self) -> list[dict]:
        return [dict(r) | {"success_rate": round(r["successes"] / r["uses"], 3) if r["uses"] else None}
                for r in self.db.all("SELECT * FROM strategy_stats ORDER BY kind, successes*1.0/MAX(uses,1) DESC")]


# ------------------------------------------------------------------ discoverer
class Discoverer:
    def __init__(self, cfg: Config, db: Database, fetcher: Fetcher, ollama=None):
        self.cfg, self.db, self.fetcher, self.ollama = cfg, db, fetcher, ollama
        self.strategy = StrategyBook(db)
        self.providers = {name: cls(cfg, fetcher, ollama) for name, cls in PROVIDERS.items()}

    def blocked_terms(self) -> list[str]:
        terms = list(self.cfg.get("privacy.blocked_query_terms", []))
        for r in self.db.all("SELECT value FROM personal_memory WHERE sensitive=1"):
            terms.extend(v for v in re.split(r"[,;\n]", r["value"]) if len(v.strip()) >= 3)
        return terms

    def safe_query(self, query: str) -> str | None:
        clean, reasons = sanitize_query(query, self.blocked_terms())
        if reasons:
            self.db.audit("discovery", "query_sanitized", clean, removed=reasons)
        if len(clean) < 3 or len(clean) < len(query) * 0.5:
            return None  # too much personal content - do not send at all
        return clean

    def active_providers(self) -> list[str]:
        configured = self.cfg.get("internet.providers", list(PROVIDERS))
        return [n for n in configured if n in self.providers and self.providers[n].available()]

    def search_topic(self, topic: str, per_query: int = 5, max_queries: int = 3,
                     log=None) -> list[tuple[SearchResult, str]]:
        """Return (result, template_name) pairs for a topic using the best strategies."""
        templates = self.strategy.rank("query_template", list(QUERY_TEMPLATES))[:max_queries]
        providers = self.strategy.rank("provider", self.active_providers())
        results: list[tuple[SearchResult, str]] = []
        seen: set[str] = set()
        for tname in templates:
            query = self.safe_query(QUERY_TEMPLATES[tname].format(topic=topic))
            if not query:
                continue
            for pname in providers[:2]:  # top-2 providers per query keeps load bounded
                try:
                    found = self.providers[pname].search(query, per_query)
                    ok, err = True, None
                except Exception as e:  # provider failure is data, not a crash
                    found, ok, err = [], False, str(e)[:300]
                if log:
                    log(stage="search", provider=pname, strategy=tname, query=query, ok=ok,
                        yield_count=len(found), error=err)
                for r in found:
                    if r.url and r.url not in seen:
                        seen.add(r.url)
                        results.append((r, tname))
        return results

    def read_feeds(self, log=None) -> list[SearchResult]:
        items: list[SearchResult] = []
        for feed in self.cfg.get("rss_feeds", []):
            try:
                found = parse_feed(self.fetcher.get_text(feed))
                ok, err = True, None
            except (FetchError, ET.ParseError, ValueError) as e:
                found, ok, err = [], False, str(e)[:300]
            if log:
                log(stage="discover", provider="rss", query=feed, ok=ok, yield_count=len(found), error=err)
            items.extend(found[:25])
        return items
