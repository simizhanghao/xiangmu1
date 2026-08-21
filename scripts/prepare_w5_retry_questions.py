#!/usr/bin/env python3
"""Emit only W5 questions that still lack a valid collected Web state."""

from __future__ import annotations

import glob
import json
from pathlib import Path


def valid(row: dict) -> bool:
    docs = row.get("documents") or []
    steps = row.get("steps") or []
    meta = row.get("metadata") or {}
    return (
        bool(docs)
        and int(meta.get("retrieval_error_count") or 0) == 0
        and all((d.get("metadata") or {}).get("source") == "bocha" for d in docs)
        and sum(s.get("step_type") == "search" for s in steps) == 1
        and any(s.get("step_type") == "observation" for s in steps)
    )


def main() -> None:
    questions = [json.loads(line) for line in Path("data/w5_controller/controller_all5000.jsonl").open()]
    patterns = [
        "results/71_w5_controller/raw_shards/shard*/agent_rollout_*/trace.jsonl",
        "results/71_w5_controller/resume_shards/shard*/agent_rollout_*/trace.jsonl",
        "results/71_w5_controller/retry_shards/attempt*/agent_rollout_*/trace.jsonl",
    ]
    good = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            for line in Path(path).open():
                row = json.loads(line)
                if valid(row):
                    good.add(row["sample_id"])
    pending = [row for row in questions if row["sample_id"] not in good]
    out = Path("data/w5_controller/controller_pending_retry.jsonl")
    with out.open("w") as handle:
        for row in pending:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"target": len(questions), "valid": len(good), "pending": len(pending), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
