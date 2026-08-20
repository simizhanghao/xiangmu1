"""Zero-shot Web search adapter; policy action remains <search>query</search>.

Providers are replaceable. The adapter normalizes search results, fetches pages,
extracts visible text, ranks query-relevant chunks, deduplicates, and returns the
same document/observation shape used by the Controlled Candidate-BM25 tool.
"""

from __future__ import annotations

import hashlib
import html
import ipaddress
import json
import os
import re
import socket
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip += 1
        elif tag in {"p", "div", "article", "section", "li", "h1", "h2", "h3", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self.skip:
            self.skip -= 1

    def handle_data(self, data: str) -> None:
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape(" ".join(self.parts))
        return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", value)).strip()


def _public_http_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        for item in socket.getaddrinfo(parsed.hostname, port):
            ip = ipaddress.ip_address(item[4][0])
            if not ip.is_global:
                return False
    except (OSError, ValueError):
        return False
    return True


def _tokens(text: str) -> set[str]:
    return {x for x in re.findall(r"[a-z0-9]{3,}", text.lower())}


class WebAdapter:
    def __init__(
        self,
        *,
        provider: str = "duckduckgo",
        cache_dir: str | Path = "results/web_cache",
        timeout_s: float = 45.0,
        retries: int = 3,
        max_page_bytes: int = 2_000_000,
        chunk_chars: int = 1400,
    ) -> None:
        self.provider = provider
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.retries = retries
        self.max_page_bytes = max_page_bytes
        self.chunk_chars = chunk_chars
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "EvidenceResearchAgent/1.0"
        retry = Retry(
            total=retries,
            connect=retries,
            read=retries,
            status=retries,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "POST"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    @property
    def request_timeout(self) -> tuple[float, float]:
        """Separate connect/read deadlines; public Web reads need a wider budget."""
        return (min(self.timeout_s, 10.0), self.timeout_s)

    def _search(self, query: str, count: int) -> list[dict[str, str]]:
        if self.provider == "brave":
            key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
            if not key:
                raise RuntimeError("BRAVE_SEARCH_API_KEY is required")
            response = self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": count},
                headers={"X-Subscription-Token": key, "Accept": "application/json"},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return [
                {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("description", "")}
                for x in response.json().get("web", {}).get("results", [])
            ]
        if self.provider == "searxng":
            endpoint = os.environ.get("SEARXNG_URL", "").rstrip("/")
            if not endpoint:
                raise RuntimeError("SEARXNG_URL is required")
            response = self.session.get(
                f"{endpoint}/search",
                params={"q": query, "format": "json", "language": "en"},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            return [
                {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": x.get("content", "")}
                for x in response.json().get("results", [])[:count]
            ]
        if self.provider != "duckduckgo":
            raise ValueError(f"unsupported provider: {self.provider}")
        response = self.session.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        links = re.findall(
            r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.I | re.S
        )
        out = []
        for raw_url, raw_title in links[:count]:
            parsed = urlparse(html.unescape(raw_url))
            target = unquote(parse_qs(parsed.query).get("uddg", [raw_url])[0])
            title = re.sub(r"<[^>]+>", "", html.unescape(raw_title)).strip()
            out.append({"title": title, "url": target, "snippet": ""})
        return out

    def _fetch_text(self, url: str) -> str:
        if not _public_http_url(url):
            raise ValueError("URL is not public HTTP(S)")
        key = hashlib.sha256(url.encode()).hexdigest()
        cached = self.cache / f"{key}.json"
        if cached.is_file():
            return str(json.loads(cached.read_text()).get("text", ""))
        current = url
        for _ in range(5):
            response = self.session.get(
                current, timeout=self.request_timeout, allow_redirects=False, stream=True
            )
            if response.is_redirect or response.is_permanent_redirect:
                target = requests.compat.urljoin(current, response.headers.get("location", ""))
                response.close()
                if not _public_http_url(target):
                    raise ValueError("redirected URL is not public HTTP(S)")
                current = target
                continue
            break
        else:
            raise ValueError("too many redirects")
        response.raise_for_status()
        data = bytearray()
        for block in response.iter_content(65536):
            data.extend(block)
            if len(data) >= self.max_page_bytes:
                break
        parser = _TextExtractor()
        parser.feed(bytes(data).decode(response.encoding or "utf-8", errors="replace"))
        text = parser.text()
        cached.write_text(json.dumps({"url": response.url, "text": text}, ensure_ascii=False))
        return text

    def retrieve(self, sample: dict[str, Any], query: str, top_k: int = 5) -> dict[str, Any]:
        del sample
        try:
            results = self._search(query, max(top_k * 2, top_k))
        except requests.RequestException as exc:
            return {
                "query": query,
                "retriever": {
                    "name": self.provider,
                    "scope": "live_web",
                    "top_k": top_k,
                    "search_ok": False,
                },
                "documents": [],
                "errors": [{"stage": "search", "error": str(exc)[:300]}],
            }
        qtok = _tokens(query)
        candidates: list[tuple[int, str, dict[str, str]]] = []
        errors: list[dict[str, str]] = []
        for result in results:
            try:
                text = self._fetch_text(result["url"])
            except Exception as exc:  # noqa: BLE001
                errors.append({"url": result.get("url", ""), "error": str(exc)[:200]})
                continue
            blocks = [x.strip() for x in re.split(r"\n\s*\n", text) if len(x.strip()) >= 40]
            for block in blocks:
                score = len(qtok & _tokens(block))
                candidates.append((score, block[: self.chunk_chars], result))
        candidates.sort(key=lambda x: (-x[0], x[2].get("url", "")))
        seen: set[str] = set()
        docs = []
        for score, text, result in candidates:
            digest = hashlib.sha256(re.sub(r"\s+", " ", text.lower()).encode()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            docs.append(
                {
                    "document_id": f"web_{digest[:12]}",
                    "title": result.get("title") or result.get("url", ""),
                    "text": text,
                    "rank": len(docs) + 1,
                    "score": float(score),
                    "metadata": {"url": result.get("url", ""), "source": self.provider},
                }
            )
            if len(docs) >= top_k:
                break
        return {
            "query": query,
            "retriever": {
                "name": self.provider,
                "scope": "live_web",
                "top_k": top_k,
                "search_ok": True,
            },
            "documents": docs,
            "errors": errors,
        }
