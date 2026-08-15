"""Candidate-BM25: retrieve inside one HotpotQA sample's contexts.

Diagnostic / controlled environment for Phase 3A — NOT full-corpus Wiki search.
Supports arbitrary model queries (not only the original question).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def retrieve_candidate_bm25(
    sample: Dict[str, Any],
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Rank `sample['contexts']` by BM25 against `query`.

    Returns the same cache-row shape as `scripts/retrieve_candidate_bm25.py`.
    """
    try:
        import bm25s
    except ImportError as exc:
        raise ImportError(
            "Missing bm25s. Install with: python -m pip install bm25s"
        ) from exc

    contexts = list(sample.get("contexts") or [])
    q = (query or "").strip() or sample.get("question") or ""
    if not contexts:
        return {
            "sample_id": sample.get("sample_id"),
            "query": q,
            "retriever": {
                "name": "bm25s",
                "scope": "candidate",
                "top_k": top_k,
                "config": {"stopwords": "en"},
            },
            "documents": [],
        }

    corpus_texts = [f"{c['title']} {c['text']}" for c in contexts]
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    k = min(top_k, len(contexts))
    query_tokens = bm25s.tokenize(q, stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=k)

    documents: List[Dict[str, Any]] = []
    for rank_i in range(results.shape[1]):
        idx = int(results[0, rank_i])
        score = float(scores[0, rank_i])
        ctx = contexts[idx]
        documents.append(
            {
                "document_id": ctx["document_id"],
                "title": ctx["title"],
                "text": ctx["text"],
                "rank": rank_i + 1,
                "score": score,
                "metadata": {"sentences": list(ctx.get("sentences") or [])},
            }
        )

    return {
        "sample_id": sample.get("sample_id"),
        "query": q,
        "retriever": {
            "name": "bm25s",
            "scope": "candidate",
            "top_k": top_k,
            "config": {"stopwords": "en", "indexed_field": "title+text"},
        },
        "documents": documents,
    }


def format_observation_text(documents: List[Dict[str, Any]]) -> str:
    if not documents:
        return "[no documents retrieved]"
    return "\n".join(
        f"[{d['document_id']}] {d['title']}: {d['text']}" for d in documents
    )


def docs_to_schema(
    documents: List[Dict[str, Any]],
    *,
    source: str = "candidate_bm25",
) -> List[Any]:
    from src.eval.trace_schema import Document

    out: List[Document] = []
    for d in documents:
        out.append(
            Document(
                document_id=str(d["document_id"]),
                title=d.get("title") or "",
                text=d.get("text") or "",
                source=source,
                rank=d.get("rank"),
                score=d.get("score"),
                metadata=dict(d.get("metadata") or {}),
            )
        )
    return out
