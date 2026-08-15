"""In-memory Candidate-BM25 index keyed by sample_id."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.tools.candidate_bm25 import format_observation_text, retrieve_candidate_bm25


class CandidateBM25Index:
    """Maps sample_id -> Hotpot contexts for per-sample BM25."""

    def __init__(self, samples: Dict[str, Dict[str, Any]]):
        self._samples = samples

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "CandidateBM25Index":
        samples: Dict[str, Dict[str, Any]] = {}
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = row.get("sample_id")
                if not sid:
                    continue
                samples[str(sid)] = {
                    "sample_id": str(sid),
                    "question": row.get("question") or "",
                    "contexts": list(row.get("contexts") or []),
                }
        return cls(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def has(self, sample_id: str) -> bool:
        return str(sample_id) in self._samples

    def get(self, sample_id: str) -> Optional[Dict[str, Any]]:
        return self._samples.get(str(sample_id))

    def retrieve(
        self,
        sample_id: str,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        sample = self.get(sample_id)
        if sample is None:
            return {
                "sample_id": sample_id,
                "query": query,
                "error": f"unknown sample_id: {sample_id}",
                "documents": [],
                "observation": "[no documents retrieved]",
            }
        packed = retrieve_candidate_bm25(sample, query, top_k=top_k)
        docs = list(packed.get("documents") or [])
        packed["observation"] = format_observation_text(docs)
        return packed


def write_contexts_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> int:
    """Write retrieval-only rows: sample_id + contexts (+ question for fallback)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            sid = row.get("sample_id")
            if not sid:
                continue
            out = {
                "sample_id": sid,
                "question": row.get("question") or "",
                "contexts": list(row.get("contexts") or []),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    return n
