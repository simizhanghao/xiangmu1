#!/usr/bin/env python3
"""Gate 1.5D: final 4550 SFT data audit. No training, no Teacher API."""

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

FILLED_CANON = REPO / "results/18_teacher_reasoning_v2/full/canonical_filled.jsonl"
FILLED_SHARE = REPO / "results/18_teacher_reasoning_v2/full/sharegpt_filled.jsonl"
AUDIT = REPO / "results/18_teacher_reasoning_v2/full/replacement_audit.json"
FROZEN_VAL = SHARED_DATA / "data/eval/hotpotqa_200_ids.txt"
PLACEHOLDER = "__TEACHER_REASONING_PENDING__"
QUOTAS = {
    "internal": 950,
    "search_format": 1250,
    "evidence": 1150,
    "evidence_reasoning": 1200,
}
LEGAL_ROLES = {"human", "gpt", "observation"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--canonical", type=Path, default=FILLED_CANON)
    p.add_argument("--sharegpt", type=Path, default=FILLED_SHARE)
    p.add_argument("--replacement-audit", type=Path, default=AUDIT)
    p.add_argument("--frozen-val-ids", type=Path, default=FROZEN_VAL)
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_export():
    path = REPO / "scripts/export_coldstart_sharegpt.py"
    spec = importlib.util.spec_from_file_location("export_coldstart_sharegpt", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def llamafactory_ok(rows: list[dict]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for row in rows:
        conv = row.get("conversations") or []
        if not conv:
            errors.append(f"{row.get('sample_id')}: empty conversations")
            continue
        roles = [c.get("from") for c in conv]
        if any(r not in LEGAL_ROLES for r in roles):
            errors.append(f"{row.get('sample_id')}: illegal role {roles}")
        if "human" not in roles or "gpt" not in roles:
            errors.append(f"{row.get('sample_id')}: missing human/gpt")
        if any(not (c.get("value") or "").strip() for c in conv):
            errors.append(f"{row.get('sample_id')}: empty turn")
        if row.get("category") == "search_format" and "observation" not in roles:
            errors.append(f"{row.get('sample_id')}: search missing observation role")
        if row.get("category") != "search_format" and "observation" in roles:
            errors.append(f"{row.get('sample_id')}: unexpected observation role")
        if len(errors) >= 20:
            break
    return len(errors) == 0, errors


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.exist_ok:
        raise SystemExit(f"output dir not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    canon = load_jsonl(args.canonical)
    share = load_jsonl(args.sharegpt)
    if args.max_samples and args.max_samples > 0:
        canon = canon[: args.max_samples]
        share = share[: args.max_samples]
    audit = json.loads(args.replacement_audit.read_text(encoding="utf-8"))
    dropped = str(audit["dropped_id"])
    added = str(audit["replacement_id"])
    frozen = load_frozen_ids(str(args.frozen_val_ids)) if args.frozen_val_ids.is_file() else set()
    assert_train_only(canon, frozen)

    export_mod = load_export()
    export_stats = export_mod.validate_export(share)
    lf_ok, lf_errors = llamafactory_ok(share)

    c_ids = [r["sample_id"] for r in canon]
    s_ids = [r["sample_id"] for r in share]
    counts = dict(Counter(r["category"] for r in canon))
    reason = [r for r in canon if r["category"] == "evidence_reasoning"]
    pending_c = sum(1 for r in canon if PLACEHOLDER in json.dumps(r, ensure_ascii=False))
    pending_s = sum(1 for r in share if PLACEHOLDER in json.dumps(r, ensure_ascii=False))
    think_reason = sum(1 for r in reason if "<think>" in (r.get("target") or ""))
    answer_reason = sum(1 for r in reason if "<answer>" in (r.get("target") or ""))
    obs_in_gpt = 0
    for row in share:
        for turn in row.get("conversations") or []:
            if turn.get("from") == "gpt" and "<observation" in (turn.get("value") or "").lower():
                obs_in_gpt += 1
    n_obs_role = sum(
        1
        for r in share
        if any(c.get("from") == "observation" for c in r.get("conversations") or [])
    )
    n_search = counts.get("search_format", 0)

    hard = {
        "n_4550": len(canon) == 4550 and len(share) == 4550,
        "unique_4550": len(set(c_ids)) == 4550 and len(set(s_ids)) == 4550,
        "ids_align": c_ids == s_ids,
        "quotas": counts == QUOTAS,
        "teacher_rationale_1200": len(reason) == 1200 and think_reason == 1200 and answer_reason == 1200,
        "pending_zero": pending_c == 0 and pending_s == 0,
        "dropped_absent": dropped not in set(c_ids),
        "replacement_present": added in set(c_ids),
        "overlap_val200_zero": len(set(c_ids) & frozen) == 0,
        "obs_not_in_gpt": obs_in_gpt == 0,
        "observation_role_matches_search": n_obs_role == n_search,
        "sharegpt_parse": export_stats.get("n") == 4550 and export_stats.get("n_reasoning_tag_remaining", 1) == 0,
        "llamafactory_contract": lf_ok,
    }
    gate = "GATE15D_FINAL_DATA_PASS" if all(hard.values()) else "GATE15D_FINAL_DATA_FAIL"
    report = {
        "gate": gate,
        "hard_gates": hard,
        "counts": counts,
        "n": len(canon),
        "unique": len(set(c_ids)),
        "pending": pending_c,
        "teacher_rationale": think_reason,
        "dropped_present": int(dropped in set(c_ids)),
        "replacement_present": int(added in set(c_ids)),
        "overlap_val200": len(set(c_ids) & frozen),
        "obs_in_gpt": obs_in_gpt,
        "n_observation_role": n_obs_role,
        "export": export_stats,
        "llamafactory_errors": lf_errors[:10],
        "canonical": str(args.canonical),
        "sharegpt": str(args.sharegpt),
    }
    (out / "gate15d_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(gate)
    if args.debug:
        print(f"AUDIT={out / 'gate15d_audit.json'}")
    if gate != "GATE15D_FINAL_DATA_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
