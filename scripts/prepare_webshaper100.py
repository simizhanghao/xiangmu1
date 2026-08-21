#!/usr/bin/env python3
"""Fetch and isolate a deterministic WebShaper100 question/answer builder pool.

Only question and builder-side gold are emitted. Formalizations and released URLs
never enter the runtime pool consumed by Web or Teacher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import requests

SOURCE_URL = (
    "https://raw.githubusercontent.com/Alibaba-NLP/DeepResearch/main/"
    "WebAgent/WebShaper/data/webshaper.500.jsonl"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(SOURCE_URL, timeout=60)
    response.raise_for_status()
    raw_bytes = response.content
    raw_path = cfg.output_dir / "webshaper.500.source.jsonl"
    raw_path.write_bytes(raw_bytes)
    rows = [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != 500:
        raise SystemExit(f"WEBSHAPER_SOURCE_COUNT_FAIL rows={len(rows)}")
    ranked = sorted(
        rows,
        key=lambda x: hashlib.sha256(f"{cfg.seed}:{x['id']}".encode()).hexdigest(),
    )[: cfg.count]
    isolated: list[dict[str, Any]] = []
    for row in ranked:
        answer = row.get("answer")
        golds = [str(x) for x in answer] if isinstance(answer, list) else [str(answer)]
        isolated.append(
            {
                "sample_id": f"webshaper_{row['id']}",
                "question": str(row["question"]).strip(),
                "gold_answers": golds,
                "supporting_facts": [],
                "source_dataset": "Alibaba-NLP/DeepResearch WebShaper",
                "gold_builder_side_only": True,
            }
        )
    pool_path = cfg.output_dir / "webshaper100.builder_pool.jsonl"
    with pool_path.open("w", encoding="utf-8") as f:
        for row in isolated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    protocol = {
        "gate": "WEBSHAPER100_ISOLATION_PASS",
        "source_url": SOURCE_URL,
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_rows": len(rows),
        "selected_rows": len(isolated),
        "seed": cfg.seed,
        "runtime_fields": ["sample_id", "question"],
        "builder_only_fields": ["gold_answers"],
        "excluded_runtime_fields": ["answer", "formalization", "urls"],
        "teacher_gold_visible": False,
    }
    (cfg.output_dir / "isolation_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
