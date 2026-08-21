#!/usr/bin/env python3
"""Keep successful W5 states and emit only failed/unseen questions for resume."""

from __future__ import annotations

import glob
import json
from pathlib import Path


def valid(trace: dict) -> bool:
    docs = trace.get("documents") or []
    meta = trace.get("metadata") or {}
    return (
        bool(docs)
        and int(meta.get("retrieval_error_count") or 0) == 0
        and all((d.get("metadata") or {}).get("source") == "bocha" for d in docs)
        and any(s.get("step_type") == "observation" for s in trace.get("steps") or [])
    )


def main() -> None:
    all_path = Path("data/w5_controller/controller_all5000.jsonl")
    out = Path("data/w5_controller/controller_pending_resume.jsonl")
    questions = [json.loads(x) for x in all_path.open()]
    good = set()
    trace_rows = 0
    for path in glob.glob("results/71_w5_controller/raw_shards/shard*/agent_rollout_*/trace.jsonl"):
        for line in Path(path).open():
            trace_rows += 1
            row = json.loads(line)
            if valid(row):
                good.add(row["sample_id"])
    pending = [row for row in questions if row["sample_id"] not in good]
    with out.open("w") as handle:
        for row in pending:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "gate": "W5_RESUME_MANIFEST_PASS",
        "target_questions": len(questions),
        "prior_trace_rows": trace_rows,
        "retained_successful_states": len(good),
        "pending_questions": len(pending),
        "duplicate_paid_successes_scheduled": 0,
        "output": str(out),
    }
    Path("data/w5_controller/resume_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
