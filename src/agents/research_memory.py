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
    source_id: str
    document_id: str
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
    _source_ids: dict[str, str] = field(default_factory=dict)
    missing_information: list[str] = field(default_factory=list)
    last_decision: str = ""
    last_next_query: str = ""

    def __post_init__(self) -> None:
        if not self.missing_information:
            self.missing_information = [f"Resolve the unanswered parts of: {self.question}"]

    def update_from_internal(self, internal_text: str) -> None:
        """Persist policy-declared Missing/Decision without trusting it as evidence."""
        text = str(internal_text or "")
        missing = re.search(
            r"(?is)(?:^|\n)\s*Missing\s*:\s*(.*?)(?=\n\s*(?:Decision|Next Query|Known)\s*:|\Z)",
            text,
        )
        if missing:
            values = [
                re.sub(r"^[-*]\s*", "", line).strip()
                for line in missing.group(1).splitlines()
                if line.strip()
            ]
            if len(values) == 1 and values[0].lower().rstrip(".") in {"none", "nothing"}:
                values = []
            self.missing_information = values[:3]
        decision = re.search(r"(?im)^\s*Decision\s*:\s*(SEARCH|ANSWER)\s*$", text)
        if decision:
            self.last_decision = decision.group(1).upper()
        next_query = re.search(r"(?im)^\s*Next Query\s*:\s*(.+?)\s*$", text)
        self.last_next_query = next_query.group(1).strip() if next_query else ""

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
            source_key = url or _norm_text(doc.get("title") or "Untitled")
            if source_key not in self._source_ids:
                self._source_ids[source_key] = f"S{len(self._source_ids) + 1}"
            source_id = self._source_ids[source_key]
            snippet = _norm_text(doc.get("text") or "")
            digest = hashlib.sha256(snippet.lower().encode("utf-8")).hexdigest()
            if not snippet or digest in self._evidence_hashes:
                continue
            self._evidence_hashes.add(digest)
            new_evidence += 1
            self.evidence.append(
                EvidenceItem(
                    evidence_id=f"E{len(self.evidence) + 1}",
                    source_id=source_id,
                    document_id=_norm_text(doc.get("document_id") or source_id),
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
        """Compatibility method; training and runtime call the same serializer."""
        return serialize_research_memory(self)

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
            "missing_information": list(self.missing_information),
            "last_decision": self.last_decision,
            "last_next_query": self.last_next_query,
        }


def serialize_research_memory(memory: ResearchMemory) -> str:
    """Canonical compact ResearchMemory serialization shared by train and runtime."""
    ranked = sorted(
        memory.evidence, key=lambda x: (-x.score, x.search_turn, x.evidence_id)
    )
    ranked = ranked[: memory.evidence_limit]
    suffix = ["Missing:"]
    if memory.missing_information:
        suffix.extend(f"- {x}" for x in memory.missing_information[:3])
    else:
        suffix.append("- None")
    suffix.append("Previous Queries:")
    if memory.searches:
        for step in memory.searches:
            suffix.append(
                f"- S{step.turn}: {step.query} "
                f"(new_urls={step.new_urls}, new_evidence={step.new_evidence})"
            )
    else:
        suffix.append("- None")
    suffix.append("Sources:")
    seen_sources: set[str] = set()
    for item in ranked:
        if item.source_id in seen_sources:
            continue
        seen_sources.add(item.source_id)
        suffix.append(f"- [{item.source_id}] {item.title} | {item.url or 'URL unavailable'}")
    if not seen_sources:
        suffix.append("- None")
    suffix.extend(
        [
            f"Remaining Budget: {memory.remaining_searches}",
            "Use Known evidence to decide whether Missing is empty. Never repeat a previous query.",
            "</research_state>",
        ]
    )
    lines = ["<research_state>", "Known:"]
    reserve = len("\n".join(suffix)) + 2
    for item in ranked:
        line = f"- [{item.source_id}] {item.snippet[:400]}"
        if len("\n".join(lines + [line])) + reserve > memory.char_budget:
            break
        lines.append(line)
    if len(lines) == 2:
        lines.append("- None")
    return "\n".join(lines + suffix)
