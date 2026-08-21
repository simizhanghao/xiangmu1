#!/usr/bin/env python3
"""Merge disjoint graph replay and audited live trajectories into the final decision SFT view."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def dump(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--live", action="append", default=[], help="origin:path")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    sources: list[tuple[str, Path]] = [("graph_replay_hotpot", a.graph / "trajectories.jsonl")]
    for value in a.live:
        origin, path = value.split(":", 1)
        sources.append((origin, Path(path)))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicate = 0
    # Prefer live evidence when a source question appears more than once.
    for origin, path in [*sources[1:], sources[0]]:
        for raw in load(path):
            source_id = str(raw["sample_id"]).split("__graph_d", 1)[0]
            if source_id in seen:
                duplicate += 1
                continue
            seen.add(source_id)
            row = copy.deepcopy(raw)
            row["origin"] = origin
            for example in row["decision_examples"]:
                example.setdefault("metadata", {})["origin"] = origin
            rows.append(row)

    examples = [x for row in rows for x in row["decision_examples"]]
    by_type = {
        kind: [x for x in examples if x["metadata"]["decision_type"] == kind]
        for kind in ("initial_search", "post_obs_stop", "post_obs_continue")
    }
    n = min(len(by_type["post_obs_stop"]), len(by_type["post_obs_continue"]))

    def ordered(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(values, key=lambda x: hashlib.sha256(f"{a.seed}:{x['sft_id']}".encode()).hexdigest())

    balanced = []
    for kind in ("initial_search", "post_obs_stop", "post_obs_continue"):
        balanced.extend(ordered(by_type[kind])[:n])
    dump(a.output_dir / "full_trajectories.jsonl", rows)
    dump(a.output_dir / "trajectories.jsonl", rows)
    dump(a.output_dir / "decision_sft.jsonl", examples)
    dump(a.output_dir / "decision_sft_balanced.jsonl", balanced)
    trajectory_origins = Counter(x["origin"] for x in rows)
    balanced_origins = Counter(x["metadata"]["origin"] for x in balanced)
    live_balanced = sum(v for k, v in balanced_origins.items() if not k.startswith("graph_"))
    summary = {
        "gate": "W3_FINAL_MIX_BUILD_PASS",
        "trajectories": len(rows),
        "deduplicated_rows": duplicate,
        "trajectory_origins": dict(trajectory_origins),
        "decision_examples": len(examples),
        "decision_distribution": {k: len(v) for k, v in by_type.items()},
        "balanced_decision_examples": len(balanced),
        "balanced_origins": dict(balanced_origins),
        "live_fraction_balanced": live_balanced / max(1, len(balanced)),
        "source_question_duplicates": 0,
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
