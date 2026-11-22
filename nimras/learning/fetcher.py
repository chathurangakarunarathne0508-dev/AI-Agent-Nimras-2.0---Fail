"""Polite, policy-enforcing web retrieval and content extraction.

Every network read in the learning pipeline goes through `Fetcher.fetch`, which
applies the URL policy, robots.txt, per-domain rate limits, a size cap and the
text-only content-type rule.
"""
from __future__ import annotations

import gzip
import re
import threading
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

from ..config import Config
from ..security import check_url, content_type_allowed


class FetchError(RuntimeError):
    pass


@dataclass
class Page:
    url: str
    final_url: str
    title: str
    text: str
    published: str | None = None
    links: list[str] = field(default_factory=list)
    content_type: str = "text/html"


class _NoRedirectToPrivate(urllib.request.HTTPRedirectHandler):
    """Re-check URL policy on every redirect hop (blocks redirect-based SSRF)."""

    def __init__(self, fetcher: "Fetcher"):
        self.fetcher = fetcher

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        ok, why = self.fetcher.url_ok(newurl)
        if not ok:
            raise FetchError(f"redirect to {newurl} blocked: {why}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


# ------------------------------------------------------------------ extraction
_SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form",
              "svg", "button", "iframe", "template", "select"}
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "h5", "h6",
               "pre", "br", "tr", "table", "blockquote", "dd", "dt", "main"}
_DATE_META = {"article:published_time", "og:published_time", "datepublished", "date",
              "dc.date", "dc.date.issued", "citation_publication_date", "pubdate",
              "article:modified_time", "last-modified"}


class _Extractor(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.parts: list[str] = []
        self.title = ""
        self.published: str | None = None
        self.links: list[str] = []
        self._skip = 0
        self._in_title = False
        self._in_ld = False
        self._ld = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in _SKIP_TAGS:
            if tag == "script" and (a.get("type") or "").lower() == "application/ld+json":
                self._in_ld = True
            self._skip += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key in _DATE_META and a.get("content") and not self.published:
                self.published = a["content"].strip()
        elif tag == "time" and a.get("datetime") and not self.published:
            self.published = a["datetime"].strip()
        elif tag == "a" and a.get("href"):
            href = urljoin(self.base, a["href"].split("#")[0])
            if href.startswith("http"):
                self.links.append(href)
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.parts.append("## ")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            if tag == "script" and self._in_ld:
                self._in_ld = False
                m = re.search(r'"datePublished"\s*:\s*"([^"]+)"', self._ld)
                if m and not self.published:
                    self.published = m.group(1)
                self._ld = ""
            return
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_ld:
            self._ld += data
            return
        if self._in_title:
            self.title += data
            return
        if self._skip:
            return
        self.parts.append(data)


def extract_html(html: str, base_url: str) -> tuple[str, str, str | None, list[str]]:
    ex = _Extractor(base_url)
    try:
        ex.feed(html)
        ex.close()
    except Exception:  # malformed HTML must never break learning
        pass
    text = "".join(ex.parts)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop boilerplate-ish short fragments, keep headings and real sentences.
    kept = [ln for ln in lines if len(ln) > 40 or ln.startswith("## ") or ln.endswith((".", ":"))]
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    seen, links = set(), []
    for link in ex.links:
        if link not in seen:
            seen.add(link)
            links.append(link)
    return ex.title.strip(), text, ex.published, links


# ------------------------------------------------------------------ fetcher
class Fetcher:
    def __init__(self, cfg: Config, ollama=None):
        self.cfg = cfg
        self.ollama = ollama
        self.ua = cfg.get("internet.user_agent")
        self.timeout = cfg.get("internet.request_timeout", 20)
        self.max_bytes = cfg.get("internet.max_bytes_per_page", 2_000_000)
        self.min_gap = cfg.get("internet.min_seconds_between_requests_per_domain", 2.0)
        self._last_hit: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()
        self._opener = urllib.request.build_opener(_NoRedirectToPrivate(self))

    def url_ok(self, url: str) -> tuple[bool, str]:
        if not self.cfg.get("internet.enabled", True):
            return False, "internet access disabled in settings"
        return check_url(
            url,
            self.cfg.get("internet.allowed_domains", []),
            self.cfg.get("internet.denied_domains", []),
        )

    def _throttle(self, host: str) -> None:
        with self._lock:
            wait = self._last_hit.get(host, 0) + self.min_gap - time.time()
            self._last_hit[host] = time.time() + max(0.0, wait)
        if wait > 0:
            time.sleep(wait)

    def robots_allows(self, url: str) -> bool:
        if not self.cfg.get("internet.respect_robots_txt", True):
            return True
        p = urlparse(url)
        root = f"{p.scheme}://{p.netloc}"
        if root not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                raw = self._raw_get(root + "/robots.txt", check_robots=False, accept_any_text=True)[0]
                rp.parse(raw.splitlines())
            except Exception:
                rp = None  # unreachable robots.txt -> treat as allowed (standard practice)
            self._robots[root] = rp
        rp = self._robots[root]
        return True if rp is None else rp.can_fetch(self.ua, url)

    def _raw_get(self, url: str, check_robots: bool = True, accept_any_text: bool = False) -> tuple[str, str, str]:
        ok, why = self.url_ok(url)
        if not ok:
            raise FetchError(f"blocked: {why}")
        if check_robots and not self.robots_allows(url):
            raise FetchError("disallowed by robots.txt")
        host = urlparse(url).hostname or ""
        self._throttle(host)
        req = urllib.request.Request(url, headers={
            "User-Agent": self.ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8",
            "Accept-Encoding": "gzip",
        })
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                ctype = r.headers.get("Content-Type", "")
                if not accept_any_text and not content_type_allowed(ctype):
                    raise FetchError(f"content type {ctype!r} refused (text only; downloads are never executed)")
                body = r.read(self.max_bytes + 1)
                if len(body) > self.max_bytes:
                    body = body[: self.max_bytes]
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                charset = r.headers.get_content_charset() or "utf-8"
                return body.decode(charset, errors="replace"), r.geturl(), ctype
        except urllib.error.HTTPError as e:
            raise FetchError(f"HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise FetchError(str(e)) from e

    def get_text(self, url: str) -> str:
        """Fetch from a documented machine endpoint (search API, RSS/Atom feed).

        robots.txt governs crawling; published APIs and feeds exist for programmatic
        clients (e.g. Wikipedia disallows /w/ for crawlers but documents /w/api.php for
        bots). URL/SSRF policy, rate limits, size caps and content-type rules still apply.
        """
        return self._raw_get(url, check_robots=False)[0]

    def fetch(self, url: str) -> Page:
        """Fetch and extract a readable page."""
        # Prefer local fetching (full policy control). Fall back to Ollama's hosted
        # web_fetch if the site blocks simple clients and the API is configured.
        try:
            body, final, ctype = self._raw_get(url)
        except FetchError as local_err:
            if (self.ollama and self.ollama.web_api_key and self.cfg.get("internet.use_ollama_web_fetch")
                    and "robots" not in str(local_err) and "blocked" not in str(local_err)):
                return self._fetch_via_ollama(url)
            raise
        if "html" in ctype or body.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            title, text, published, links = extract_html(body, final)
        else:
            title, text, published, links = "", body, None, []
        return Page(url=url, final_url=final, title=title or url, text=text,
                    published=published, links=links, content_type=ctype)

    def _fetch_via_ollama(self, url: str) -> Page:
        ok, why = self.url_ok(url)
        if not ok:
            raise FetchError(f"blocked: {why}")
        try:
            r = self.ollama.web_fetch(url)
        except Exception as e:
            raise FetchError(f"ollama web_fetch failed: {e}") from e
        return Page(url=url, final_url=url, title=r.get("title") or url,
                    text=r.get("content") or "", links=r.get("links") or [])
