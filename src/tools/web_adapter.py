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
import time
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


def _clean_context_snippet(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _is_benchmark_leak_url(url: str) -> bool:
    """Reject public benchmark mirrors without consulting sample labels or gold data."""
    parsed = urlparse(str(url or ""))
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    joined = f"{host}{path}"
    if host in {"huggingface.co", "www.huggingface.co", "datasets-server.huggingface.co"}:
        return path.startswith("/datasets/") or host.startswith("datasets-server")
    if host in {"kaggle.com", "www.kaggle.com"} and path.startswith("/datasets/"):
        return True
    if host in {"github.com", "www.github.com", "raw.githubusercontent.com"}:
        return any(marker in path for marker in ("hotpotqa", "hotpot_qa", "hotpot-qa"))
    return any(marker in joined for marker in ("rag-rl-hotpotqa", "hotpotqa-eval"))


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
        llm_context_tokens: int = 4096,
        llm_context_threshold: str = "balanced",
    ) -> None:
        self.provider = provider
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.retries = retries
        self.max_page_bytes = max_page_bytes
        self.chunk_chars = chunk_chars
        self.llm_context_tokens = llm_context_tokens
        self.llm_context_threshold = llm_context_threshold
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

    def _brave_llm_context(self, query: str, top_k: int) -> list[dict[str, Any]]:
        key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if not key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY is required")
        api_query = " ".join(str(query).split()[:50])[:400]
        response = self.session.post(
            "https://api.search.brave.com/res/v1/llm/context",
            json={
                "q": api_query,
                "country": "US",
                "search_lang": "en",
                "count": max(5, top_k),
                "maximum_number_of_urls": top_k,
                "maximum_number_of_tokens": self.llm_context_tokens,
                "maximum_number_of_snippets": max(10, top_k * 4),
                "maximum_number_of_tokens_per_url": 1024,
                "maximum_number_of_snippets_per_url": 8,
                "context_threshold_mode": self.llm_context_threshold,
                "enable_source_metadata": True,
            },
            headers={
                "X-Subscription-Token": key,
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "Content-Type": "application/json",
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        generic = (payload.get("grounding") or {}).get("generic") or []
        sources = payload.get("sources") or {}
        out = []
        for item in generic[:top_k]:
            url = str(item.get("url") or "")
            source = sources.get(url) or {}
            snippets = [_clean_context_snippet(x) for x in (item.get("snippets") or [])]
            snippets = [x for x in snippets if x]
            if not snippets:
                continue
            out.append(
                {
                    "title": str(item.get("title") or source.get("title") or url),
                    "url": url,
                    "snippets": snippets,
                    "source_metadata": source,
                }
            )
        return out

    def _bocha_context(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return Bocha Web Search summaries in the context-provider shape."""
        key = os.environ.get("BOCHA_API_KEY", "")
        if not key:
            raise RuntimeError("BOCHA_API_KEY is required")
        response = self.session.post(
            "https://api.bochaai.com/v1/web-search",
            json={
                "query": " ".join(str(query).split()[:50])[:400],
                "summary": True,
                "count": max(5, top_k),
            },
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        # Current responses expose webPages directly; tolerate an optional data
        # envelope used by some Bocha SDK examples.
        body = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        values = ((body.get("webPages") or {}).get("value") or [])
        out = []
        for item in values[:top_k]:
            url = str(item.get("url") or "")
            snippets = []
            for value in (item.get("summary"), item.get("snippet")):
                cleaned = _clean_context_snippet(value)
                if cleaned and cleaned not in snippets:
                    snippets.append(cleaned)
            if not url or not snippets:
                continue
            out.append(
                {
                    "title": str(item.get("name") or item.get("title") or url),
                    "url": url,
                    "snippets": snippets,
                    "source_metadata": {
                        "id": item.get("id"),
                        "site_name": item.get("siteName"),
                        "date_published": item.get("datePublished"),
                    },
                }
            )
        return out

    def _fetch_text(self, url: str) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        if not _public_http_url(url):
            raise ValueError("URL is not public HTTP(S)")
        key = hashlib.sha256(url.encode()).hexdigest()
        cached = self.cache / f"{key}.json"
        if cached.is_file():
            return str(json.loads(cached.read_text()).get("text", "")), {
                "fetch_ms": 0.0,
                "extract_ms": 0.0,
                "cache_hit": True,
            }
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
        fetch_ms = (time.perf_counter() - started) * 1000.0
        extract_started = time.perf_counter()
        parser = _TextExtractor()
        parser.feed(bytes(data).decode(response.encoding or "utf-8", errors="replace"))
        text = parser.text()
        cached.write_text(json.dumps({"url": response.url, "text": text}, ensure_ascii=False))
        return text, {
            "fetch_ms": round(fetch_ms, 2),
            "extract_ms": round((time.perf_counter() - extract_started) * 1000.0, 2),
            "cache_hit": False,
            "bytes": len(data),
        }

    def retrieve(self, sample: dict[str, Any], query: str, top_k: int = 5) -> dict[str, Any]:
        del sample
        tool_started = time.perf_counter()
        search_started = time.perf_counter()
        if self.provider in {"brave_llm_context", "bocha"}:
            try:
                raw_contexts = (
                    self._brave_llm_context(query, top_k)
                    if self.provider == "brave_llm_context"
                    else self._bocha_context(query, top_k)
                )
            except requests.RequestException as exc:
                search_ms = (time.perf_counter() - search_started) * 1000.0
                return self._failed_search(query, top_k, exc, search_ms)
            search_ms = (time.perf_counter() - search_started) * 1000.0
            contexts = [x for x in raw_contexts if not _is_benchmark_leak_url(x.get("url", ""))]
            filtered_urls = len(raw_contexts) - len(contexts)
            qtok = _tokens(query)
            docs = []
            for item in contexts:
                text = "\n\n".join(item["snippets"])[: self.chunk_chars]
                digest = hashlib.sha256(f"{item['url']}\n{text}".encode()).hexdigest()
                docs.append(
                    {
                        "document_id": f"web_{digest[:12]}",
                        "title": item["title"],
                        "text": text,
                        "rank": len(docs) + 1,
                        "score": float(len(qtok & _tokens(text))),
                        "metadata": {
                            "url": item["url"],
                            "source": self.provider,
                            "source_metadata": item["source_metadata"],
                        },
                    }
                )
            total_ms = (time.perf_counter() - tool_started) * 1000.0
            return {
                "query": query,
                "retriever": {
                    "name": self.provider,
                    "scope": "live_web",
                    "top_k": top_k,
                    "search_ok": True,
                },
                "documents": docs,
                "errors": [],
                "timing": {
                    "search_api_ms": round(search_ms, 2),
                    "fetch_ms": [],
                    "fetch_total_ms": 0.0,
                    "extract_ms": 0.0,
                    "tool_total_ms": round(total_ms, 2),
                    "success_urls": len(contexts),
                    "failed_urls": 0,
                    "filtered_urls": filtered_urls,
                },
            }
        try:
            results = self._search(query, top_k)
        except requests.RequestException as exc:
            search_ms = (time.perf_counter() - search_started) * 1000.0
            return self._failed_search(query, top_k, exc, search_ms)
        search_ms = (time.perf_counter() - search_started) * 1000.0
        unfiltered_results = results
        results = [x for x in unfiltered_results if not _is_benchmark_leak_url(x.get("url", ""))]
        filtered_urls = len(unfiltered_results) - len(results)
        qtok = _tokens(query)
        candidates: list[tuple[int, str, dict[str, str]]] = []
        errors: list[dict[str, str]] = []
        fetch_timings: list[dict[str, Any]] = []
        success_urls = 0
        extract_ms = 0.0
        for result in results:
            fetch_started = time.perf_counter()
            try:
                text, fetch_info = self._fetch_text(result["url"])
            except Exception as exc:  # noqa: BLE001
                elapsed = (time.perf_counter() - fetch_started) * 1000.0
                errors.append(
                    {
                        "stage": "fetch",
                        "url": result.get("url", ""),
                        "error_type": type(exc).__name__,
                        "elapsed_ms": round(elapsed, 2),
                        "error": str(exc)[:200],
                    }
                )
                fetch_timings.append(
                    {
                        "url": result.get("url", ""),
                        "ok": False,
                        "fetch_ms": round(elapsed, 2),
                        "extract_ms": 0.0,
                    }
                )
                continue
            success_urls += 1
            extract_ms += float(fetch_info.get("extract_ms") or 0.0)
            fetch_timings.append({"url": result.get("url", ""), "ok": True, **fetch_info})
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
        total_ms = (time.perf_counter() - tool_started) * 1000.0
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
            "timing": {
                "search_api_ms": round(search_ms, 2),
                "fetch_ms": fetch_timings,
                "fetch_total_ms": round(
                    sum(float(x.get("fetch_ms") or 0.0) for x in fetch_timings), 2
                ),
                "extract_ms": round(extract_ms, 2),
                "tool_total_ms": round(total_ms, 2),
                "success_urls": success_urls,
                "failed_urls": len(errors),
                "filtered_urls": filtered_urls,
            },
        }

    def _failed_search(
        self, query: str, top_k: int, exc: Exception, search_ms: float
    ) -> dict[str, Any]:
        return {
            "query": query,
            "retriever": {
                "name": self.provider,
                "scope": "live_web",
                "top_k": top_k,
                "search_ok": False,
            },
            "documents": [],
            "errors": [
                {
                    "stage": "search",
                    "error_type": type(exc).__name__,
                    "elapsed_ms": round(search_ms, 2),
                    "error": str(exc)[:300],
                }
            ],
            "timing": {
                "search_api_ms": round(search_ms, 2),
                "fetch_ms": [],
                "fetch_total_ms": 0.0,
                "extract_ms": 0.0,
                "tool_total_ms": round(search_ms, 2),
                "success_urls": 0,
                "failed_urls": 0,
                "filtered_urls": 0,
            },
        }
