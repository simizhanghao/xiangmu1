#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAGS = ["base_direct", "base_rag", "sft_agent", "oracle_sft", "step200", "step400", "step600", "step800"]


def fmt(x: object) -> str:
    return "" if x is None else f"{float(x):.4f}"


rows = []
for tag in TAGS:
    path = ROOT / "results" / "frozen_dev" / tag / "summary.json"
    if not path.exists():
        continue
    s = json.loads(path.read_text())
    rows.append(
        [
            tag,
            fmt(s.get("mean_em")),
            fmt(s.get("mean_token_f1")),
            fmt(s.get("mean_evidence_f1")),
            fmt(s.get("finish_rate")),
            fmt(s.get("parse_ok_rate")),
            fmt(s.get("mean_search_count")),
            fmt(s.get("mean_generated_tokens")),
        ]
    )

header = ["System", "EM", "Answer F1", "Evidence F1", "Finish", "Format", "Search calls", "Generated tokens"]
lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
lines.extend("| " + " | ".join(r) + " |" for r in rows)
text = "# Frozen-dev result table\n\n" + "\n".join(lines) + "\n"
target = ROOT / "results" / "FROZEN_DEV_TABLE.md"
target.write_text(text)
print(text)
print(f"TABLE_WRITTEN={target}")
