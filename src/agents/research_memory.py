"""Bounded episode memory for zero-shot Web research.

This module only consumes tool-returned documents.  It never reads answers,
supporting facts, qrels, or any other oracle field from an evaluation sample.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _norm_query(query: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", query.lower()).strip()


def _canonical_url(url: str) -> str:
    try:
        parsed = urlsplit(str(url or ""))
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))
    except ValueError:
        return str(url or "").strip()


@dataclass
class EvidenceItem:
    evidence_id: str
    title: str
    url: str
    snippet: str
    score: float
    search_turn: int


@dataclass
class SearchStep:
    turn: int
    query: str
    returned_urls: int
    new_urls: int
    new_evidence: int


@dataclass
class ResearchMemory:
    """Compact state that persists within one Web episode only."""

    question: str
    max_searches: int = 5
    evidence_limit: int = 8
    char_budget: int = 5000
    evidence: list[EvidenceItem] = field(default_factory=list)
    searches: list[SearchStep] = field(default_factory=list)
    visited_urls: set[str] = field(default_factory=set)
    _evidence_hashes: set[str] = field(default_factory=set)

    def add_search(self, query: str, documents: list[dict[str, Any]]) -> dict[str, int]:
        """Update memory from one tool response and return novelty counters."""
        turn = len(self.searches) + 1
        new_urls = 0
        new_evidence = 0
        step_urls: set[str] = set()
        for doc in documents:
            metadata = doc.get("metadata") or {}
            url = _canonical_url(metadata.get("url") or "")
            if url:
                step_urls.add(url)
            if url and url not in self.visited_urls:
                self.visited_urls.add(url)
                new_urls += 1
            snippet = _norm_text(doc.get("text") or "")
            digest = hashlib.sha256(snippet.lower().encode("utf-8")).hexdigest()
            if not snippet or digest in self._evidence_hashes:
                continue
            self._evidence_hashes.add(digest)
            new_evidence += 1
            self.evidence.append(
                EvidenceItem(
                    evidence_id=f"E{len(self.evidence) + 1}",
                    title=_norm_text(doc.get("title") or "Untitled"),
                    url=url,
                    snippet=snippet[:700],
                    score=float(doc.get("score") or 0.0),
                    search_turn=turn,
                )
            )
        self.searches.append(
            SearchStep(turn, _norm_text(query), len(step_urls), new_urls, new_evidence)
        )
        return {"new_urls": new_urls, "new_evidence": new_evidence}

    @property
    def remaining_searches(self) -> int:
        return max(0, self.max_searches - len(self.searches))

    @property
    def duplicate_query_count(self) -> int:
        values = [_norm_query(x.query) for x in self.searches]
        return len(values) - len(set(values))

    @property
    def duplicate_url_count(self) -> int:
        return sum(max(0, x.returned_urls - x.new_urls) for x in self.searches)

    def render(self) -> str:
        """Render a bounded, provenance-preserving state for the next policy turn."""
        ranked = sorted(self.evidence, key=lambda x: (-x.score, x.search_turn, x.evidence_id))
        ranked = ranked[: self.evidence_limit]
        suffix = ["Missing information:", f"- Resolve the unanswered parts of: {self.question}"]
        suffix.append("Previous searches:")
        if self.searches:
            for step in self.searches:
                suffix.append(
                    f"- S{step.turn}: {step.query} "
                    f"(new_urls={step.new_urls}, new_evidence={step.new_evidence})"
                )
        else:
            suffix.append("- None.")
        suffix.extend(
            [
                f"Remaining search budget: {self.remaining_searches}",
                "Use cited evidence IDs when forming evidence. Search again only if information is missing.",
                "</research_state>",
            ]
        )
        lines = ["<research_state>", "Known evidence:"]
        reserve = len("\n".join(suffix)) + 2
        for item in ranked:
            source = item.url or item.title
            line = f"- [{item.evidence_id}] {item.snippet[:400]} (source: {source})"
            if len("\n".join(lines + [line])) + reserve > self.char_budget:
                break
            lines.append(line)
        if len(lines) == 2:
            lines.append("- None retained within budget.")
        return "\n".join(lines + suffix)

    def summary(self) -> dict[str, Any]:
        return {
            "searches": len(self.searches),
            "evidence_items": len(self.evidence),
            "unique_urls": len(self.visited_urls),
            "duplicate_query_count": self.duplicate_query_count,
            "duplicate_url_count": self.duplicate_url_count,
            "new_evidence_per_search": round(
                sum(x.new_evidence for x in self.searches) / max(1, len(self.searches)), 4
            ),
            "remaining_searches": self.remaining_searches,
        }
