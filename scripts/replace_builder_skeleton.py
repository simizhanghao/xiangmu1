#!/usr/bin/env python3
"""Swap 1 evidence_reasoning skeleton. Keep 17_build immutable. No Teacher fill."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SHARED_DATA = Path("/data1/hcc/deepresearch")

from src.sft.coldstart_builder import assert_train_only, load_frozen_ids
from src.sft.prototype_builder import index_by_sample_id, load_jsonl, validate_sft_row

SRC_CANON = REPO / "results/17_build_8b_coldstart_v2/canonical.jsonl"
SRC_SHARE = REPO / "results/17_build_8b_coldstart_v2/sharegpt.jsonl"
AUDIT = REPO / "results/18_teacher_reasoning_v2/full/replacement_audit.json"
POOL = SHARED_DATA / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
RETRIEVAL = (
    SHARED_DATA / "results/retrieval_candidate_bm25_n8000_20260807_162150/retrieval_results.jsonl"
)
FROZEN_VAL = SHARED_DATA / "data/eval/hotpotqa_200_ids.txt"
BUILDER = REPO / "scripts/build_8b_coldstart_v2.py"
PLACEHOLDER = "__TEACHER_REASONING_PENDING__"
QUOTAS = {
    "internal": 950,
    "search_format": 1250,
    "evidence": 1150,
    "evidence_reasoning": 1200,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--source-canonical", type=Path, default=SRC_CANON)
    p.add_argument("--source-sharegpt", type=Path, default=SRC_SHARE)
    p.add_argument("--replacement-audit", type=Path, default=AUDIT)
    p.add_argument("--pool", type=Path, default=POOL)
    p.add_argument("--retrieval-cache", type=Path, default=RETRIEVAL)
    p.add_argument("--frozen-val-ids", type=Path, default=FROZEN_VAL)
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def load_builder():
    spec = importlib.util.spec_from_file_location("build_8b_coldstart_v2", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_export():
    path = REPO / "scripts/export_coldstart_sharegpt.py"
    spec = importlib.util.spec_from_file_location("export_coldstart_sharegpt", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def swap_lines(src: Path, drop_id: str, new_row: dict) -> tuple[list[str], int, int]:
    lines: list[str] = []
    n_drop = 0
    n_keep = 0
    new_line = json.dumps(new_row, ensure_ascii=False)
    for raw in src.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("sample_id") == drop_id:
            lines.append(new_line)
            n_drop += 1
            continue
        lines.append(raw)
        n_keep += 1
    return lines, n_drop, n_keep


def ids_from_lines(lines: list[str]) -> list[str]:
    return [json.loads(line)["sample_id"] for line in lines]


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.resolve() == args.source_canonical.parent.resolve():
        raise SystemExit("refuse to overwrite 17_build")
    if out.exists() and any(out.iterdir()) and not args.exist_ok:
        raise SystemExit(f"output dir not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    audit = json.loads(args.replacement_audit.read_text(encoding="utf-8"))
    dropped = str(audit["dropped_id"])
    added = str(audit["replacement_id"])
    band = str(audit["band"])
    if dropped == added:
        raise SystemExit("DROP_EQ_ADD")

    builder = load_builder()
    export_mod = load_export()
    pool = index_by_sample_id(load_jsonl(str(args.pool)))
    if added not in pool:
        raise SystemExit(f"REPLACEMENT_NOT_IN_POOL {added}")
    retrieval_map = (
        index_by_sample_id(load_jsonl(str(args.retrieval_cache)))
        if args.retrieval_cache.is_file()
        else {}
    )
    sample = pool[added]
    retr = builder.retrieval_for(sample, retrieval_map)
    new_canon = builder.build_reasoning_pending(sample, args.seed, retr, band)
    errs = validate_sft_row(new_canon)
    if errs:
        raise SystemExit(f"NEW_ROW_INVALID {errs}")
    if PLACEHOLDER not in (new_canon.get("target") or ""):
        raise SystemExit("NEW_ROW_NOT_PENDING")
    new_share = export_mod.to_sharegpt(new_canon)

    canon_lines, n_drop_c, n_keep_c = swap_lines(args.source_canonical, dropped, new_canon)
    share_lines, n_drop_s, n_keep_s = swap_lines(args.source_sharegpt, dropped, new_share)
    if args.max_samples and args.max_samples > 0:
        canon_lines = canon_lines[: args.max_samples]
        share_lines = share_lines[: args.max_samples]

    old_ids = {
        json.loads(line)["sample_id"]
        for line in args.source_canonical.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    new_ids = ids_from_lines(canon_lines)
    share_ids = ids_from_lines(share_lines)
    dropped_set = {dropped}
    added_set = {added}
    id_delta_drop = sorted(old_ids - set(new_ids))
    id_delta_add = sorted(set(new_ids) - old_ids)

    rows = [json.loads(line) for line in canon_lines]
    share_rows = [json.loads(line) for line in share_lines]
    counts = dict(Counter(r["category"] for r in rows))
    reason_ids = [r["sample_id"] for r in rows if r["category"] == "evidence_reasoning"]
    pending = sum(
        1
        for r in rows
        if r["category"] == "evidence_reasoning" and PLACEHOLDER in (r.get("target") or "")
    )
    pending_share = sum(
        1
        for r in share_rows
        if r["category"] == "evidence_reasoning" and PLACEHOLDER in json.dumps(r)
    )
    frozen = load_frozen_ids(str(args.frozen_val_ids)) if args.frozen_val_ids.is_file() else set()
    assert_train_only(rows, frozen)
    obs_in_gpt = 0
    for row in share_rows:
        for turn in row.get("conversations") or []:
            if turn.get("from") == "gpt" and "<observation" in (turn.get("value") or "").lower():
                obs_in_gpt += 1

    hard = {
        "n_4550": len(rows) == 4550,
        "unique_sample_id_4550": len(set(new_ids)) == 4550,
        "quotas": counts == QUOTAS,
        "share_n_match": len(share_rows) == len(rows) and share_ids == new_ids,
        "dropped_absent": dropped not in set(new_ids),
        "replacement_present": added in set(new_ids),
        "id_delta_exact": id_delta_drop == [dropped] and id_delta_add == [added],
        "kept_4549": n_keep_c == 4549 and n_keep_s == 4549 and n_drop_c == 1 and n_drop_s == 1,
        "pending_1200": pending == 1200 and pending_share == 1200,
        "replacement_is_reasoning": added in set(reason_ids),
        "overlap_val200_zero": len(set(new_ids) & frozen) == 0,
        "obs_not_in_gpt": obs_in_gpt == 0,
        "teacher_not_filled": pending == QUOTAS["evidence_reasoning"],
        "source_17_untouched": args.source_canonical.is_file(),
    }
    gate = "GATE_BUILDER_V2_REPLACED_PASS" if all(hard.values()) else "GATE_BUILDER_V2_REPLACED_FAIL"
    report = {
        "gate": gate,
        "hard_gates": hard,
        "counts": counts,
        "n": len(rows),
        "unique": len(set(new_ids)),
        "dropped_id": dropped,
        "replacement_id": added,
        "id_delta_drop": id_delta_drop,
        "id_delta_add": id_delta_add,
        "pending": pending,
        "overlap_val200": len(set(new_ids) & frozen),
        "obs_in_gpt": obs_in_gpt,
        "source_canonical": str(args.source_canonical),
        "source_sharegpt": str(args.source_sharegpt),
        "replacement_audit": str(args.replacement_audit),
        "teacher_filled": 0,
    }

    canon_path = out / "canonical.jsonl"
    share_path = out / "sharegpt.jsonl"
    with canon_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(canon_lines) + "\n")
    with share_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(share_lines) + "\n")
    (out / "ids_evidence_reasoning.json").write_text(
        json.dumps(sorted(reason_ids), indent=2) + "\n", encoding="utf-8"
    )
    (out / "build_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(gate)
    if args.debug:
        print(f"CANON={canon_path} SHARE={share_path} N={len(rows)}")
    if gate != "GATE_BUILDER_V2_REPLACED_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
