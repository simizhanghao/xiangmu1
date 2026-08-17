#!/usr/bin/env python3
"""Gate 1.5A: Qwen3-8B Direct/Oracle labels on the frozen 8k HotpotQA pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POOL = Path("/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl")
OLD_DIRECT = Path(
    "/data1/hcc/deepresearch/results/phase2e1_direct_label_n8000_20260807_202826_phase2e1/labels.jsonl"
)
OLD_ORACLE = Path(
    "/data1/hcc/deepresearch/results/phase2e1_base_oracle_n8000_20260807_205154/merged/metrics.json"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels-dir", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def _ok(row: dict) -> bool:
    if row.get("direct_correct") is True:
        return True
    em = row.get("exact_match")
    if em is None and isinstance(row.get("metrics"), dict):
        em = row["metrics"].get("exact_match")
    return float(em or 0) >= 1.0 - 1e-9


def load_jsonl_map(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["sample_id"])] = row
    return out


def load_oracle_map(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        rows = payload["rows"]
    else:
        return {}
    return {str(row["sample_id"]): row for row in rows if "sample_id" in row}


def merge_mode(labels_dir: Path, mode: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(labels_dir.glob(f"{mode}*/metrics.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    by_id = {row["sample_id"]: row for row in rows}
    return [by_id[key] for key in sorted(by_id)]


def shift(old: dict[str, dict], new: dict[str, dict]) -> dict:
    both = set(old) & set(new)
    counts = Counter()
    for sid in both:
        o = "ok" if _ok(old[sid]) else "err"
        n = "ok" if _ok(new[sid]) else "err"
        counts[f"old_{o}__new_{n}"] += 1
    n = max(len(both), 1)
    return {
        "n_compared": len(both),
        "old_err__new_ok": counts["old_err__new_ok"],
        "old_err__new_err": counts["old_err__new_err"],
        "old_ok__new_err": counts["old_ok__new_err"],
        "old_ok__new_ok": counts["old_ok__new_ok"],
        "old_err_to_new_ok_rate": round(counts["old_err__new_ok"] / n, 4),
        "old_labels_found": bool(old),
    }


def main() -> None:
    args = parse_args()
    direct_rows = merge_mode(args.labels_dir, "direct")
    oracle_rows = merge_mode(args.labels_dir, "oracle")
    direct_map = {row["sample_id"]: row for row in direct_rows}
    oracle_map = {row["sample_id"]: row for row in oracle_rows}
    old_direct = load_jsonl_map(OLD_DIRECT)
    old_oracle = load_oracle_map(OLD_ORACLE)
    both_ids = set(direct_map) & set(oracle_map)
    quad = Counter()
    for sid in both_ids:
        d = "ok" if _ok(direct_map[sid]) else "err"
        o = "ok" if _ok(oracle_map[sid]) else "err"
        quad[f"direct_{d}__oracle_{o}"] += 1
    n_both = max(len(both_ids), 1)
    report = {
        "gate": "GATE15A_RELABEL_PASS" if direct_rows and oracle_rows else "GATE15A_PARTIAL",
        "pool": str(POOL),
        "n_direct_8b": len(direct_rows),
        "n_oracle_8b": len(oracle_rows),
        "n_joined": len(both_ids),
        "direct_em_8b": round(
            sum(float(r["exact_match"]) for r in direct_rows) / max(len(direct_rows), 1), 4
        ),
        "oracle_em_8b": round(
            sum(float(r["exact_match"]) for r in oracle_rows) / max(len(oracle_rows), 1), 4
        ),
        "direct_token_f1_8b": round(
            sum(float(r.get("token_f1") or 0) for r in direct_rows) / max(len(direct_rows), 1), 4
        ),
        "oracle_token_f1_8b": round(
            sum(float(r.get("token_f1") or 0) for r in oracle_rows) / max(len(oracle_rows), 1), 4
        ),
        "quadrant_8b": {
            "direct_ok__oracle_ok": quad["direct_ok__oracle_ok"],
            "direct_ok__oracle_err": quad["direct_ok__oracle_err"],
            "direct_err__oracle_ok": quad["direct_err__oracle_ok"],
            "direct_err__oracle_err": quad["direct_err__oracle_err"],
            "internal_rate": round(quad["direct_ok__oracle_ok"] / n_both, 4),
            "search_gap_rate": round(quad["direct_err__oracle_ok"] / n_both, 4),
            "hard_rate": round(quad["direct_err__oracle_err"] / n_both, 4),
            "anomaly_rate": round(quad["direct_ok__oracle_err"] / n_both, 4),
        },
        "shift_direct": shift(old_direct, direct_map),
        "shift_oracle": shift(old_oracle, oracle_map),
    }
    if not old_direct:
        report["shift_direct"]["note"] = "3B Direct labels missing on disk; compare WAIT"
    if not old_oracle:
        report["shift_oracle"]["note"] = "3B Oracle labels missing on disk; compare WAIT"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    labels_out = args.labels_dir / "direct_8b.jsonl"
    with labels_out.open("w", encoding="utf-8") as handle:
        for row in direct_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    oracle_out = args.labels_dir / "oracle_8b.jsonl"
    with oracle_out.open("w", encoding="utf-8") as handle:
        for row in oracle_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))
    if direct_rows and oracle_rows:
        print("GATE15A_RELABEL_PASS")


if __name__ == "__main__":
    main()
