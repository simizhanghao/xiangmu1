#!/usr/bin/env python3
"""Merge and audit the fixed W5 natural-state collection shards."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Input trace glob. Repeat to merge initial and resumed collection.",
    )
    p.add_argument("--output-dir", default="results/72_w5_state_dataset/raw_states_5000")
    p.add_argument("--expected", type=int, default=5000)
    args = p.parse_args()

    patterns = args.glob or [
        "results/71_w5_controller/raw_shards/shard*/agent_rollout_*/trace.jsonl",
        "results/71_w5_controller/resume_shards/shard*/agent_rollout_*/trace.jsonl",
        "results/71_w5_controller/retry_shards/attempt*/agent_rollout_*/trace.jsonl",
    ]
    files = sorted({file for pattern in patterns for file in glob.glob(pattern)})

    def valid(row: dict) -> bool:
        steps = row.get("steps") or []
        docs = row.get("documents") or []
        meta = row.get("metadata") or {}
        return (
            sum(x.get("step_type") == "search" for x in steps) == 1
            and any(x.get("step_type") == "observation" for x in steps)
            and bool(docs)
            and all((d.get("metadata") or {}).get("source") == "bocha" for d in docs)
            and int(meta.get("retrieval_error_count") or 0) == 0
        )

    def natural_no_search(row: dict) -> bool:
        steps = row.get("steps") or []
        meta = row.get("metadata") or {}
        return (
            sum(x.get("step_type") == "search" for x in steps) == 0
            and any(x.get("step_type") in {"answer", "evidence"} for x in steps)
            and int(meta.get("retrieval_error_count") or 0) == 0
        )

    by_id: dict[str, dict] = {}
    input_rows = duplicate_attempts = valid_input_rows = 0
    for file in files:
        for line in Path(file).open():
            row = json.loads(line)
            sid = row["sample_id"]
            input_rows += 1
            valid_input_rows += int(valid(row))
            duplicate_attempts += int(sid in by_id)
            # A successful resume must replace the earlier failed/quota-limited row.
            if sid not in by_id or (valid(row) and not valid(by_id[sid])):
                by_id[sid] = row

    selected = [by_id[sid] for sid in sorted(by_id)]
    rows = [row for row in selected if valid(row)]
    valid_rows = len(rows)
    no_search_rows = sum(natural_no_search(row) for row in selected)
    invalid_rows = len(selected) - valid_rows - no_search_rows

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "trace.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    passed = (
        len(selected) == args.expected and valid_rows / args.expected >= 0.995
        and valid_rows + no_search_rows == args.expected and invalid_rows == 0
    )
    summary = {
        "gate": "W5_RAW_STATE_GATE_PASS" if passed else "W5_RAW_STATE_GATE_FAIL",
        "trace_files": len(files),
        "input_rows": input_rows,
        "valid_input_rows": valid_input_rows,
        "unique_sampled_questions": len(selected),
        "expected_sampled_questions": args.expected,
        "duplicate_attempts": duplicate_attempts,
        "valid_selected_states": valid_rows,
        "natural_no_search_questions": no_search_rows,
        "post_observation_coverage": valid_rows / args.expected,
        "invalid_selected_states": invalid_rows,
        "trace": str(out / "trace.jsonl"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
