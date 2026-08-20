#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "53_heldout_four_arm"
paths = {
    "Base Direct": OUT / "base_direct" / "summary.json",
    "Fixed RAG": OUT / "base_rag" / "summary.json",
    "SFT Agent": OUT / "sft_agent" / "summary.json",
    "GRPO@400": ROOT / "results/51_heldout_test/n500_grpo400_finalize_v2b/agent_rollout_n500_20260820_125356_heldout_n500_grpo400_finalize_v2b/summary.json",
}

missing = [str(path) for path in paths.values() if not path.is_file()]
if missing:
    raise SystemExit("missing summaries:\n" + "\n".join(missing))

rows = {name: json.loads(path.read_text()) for name, path in paths.items()}
for name, row in rows.items():
    if int(row["num_samples"]) != 500:
        raise SystemExit(f"{name}: expected 500, got {row['num_samples']}")

delta = float(rows["GRPO@400"]["mean_token_f1"]) - float(rows["SFT Agent"]["mean_token_f1"])
summary = {
    "gate": "HELDOUT_FOUR_ARM_COMPLETE",
    "selection_changed": False,
    "best_controlled_policy": "GRPO step400",
    "delta_rl_heldout_f1": round(delta, 4),
    "arms": rows,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

def f(row: dict, key: str) -> str:
    value = row.get(key)
    return "—" if value is None else f"{float(value):.4f}"

lines = [
    "# Held-out four-arm closure",
    "",
    "| Arm | Answer F1 | EM | Evidence F1 | Search | Finish |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, row in rows.items():
    search = row.get("mean_search_count", 0.0)
    evidence = "—" if name in {"Base Direct", "Fixed RAG"} else f(row, "mean_evidence_f1")
    lines.append(
        f"| {name} | {f(row, 'mean_token_f1')} | {f(row, 'mean_em')} | "
        f"{evidence} | {float(search):.4f} | {f(row, 'finish_rate')} |"
    )
lines += ["", f"Held-out ΔRL F1 (GRPO@400 − SFT) = **{delta:+.4f}**.", ""]
(OUT / "TABLE.md").write_text("\n".join(lines))
print("\n".join(lines))
print(f"HELDOUT_FOUR_ARM_COMPLETE delta_rl_f1={delta:+.4f}")
