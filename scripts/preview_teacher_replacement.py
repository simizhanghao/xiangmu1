#!/usr/bin/env python3
"""Print one deterministic genuine_hard replacement. No API, no writes."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

DEE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEE))
SHARED_DATA = Path("/data1/hcc/deepresearch")

from src.sft.coldstart_builder import load_frozen_ids
from src.sft.prototype_builder import gold_answer_of, load_jsonl, resolve_evidence_refs

DROP = "hotpotqa_distractor_train_5a8bbd2e5542995d1e6f1435"
SEED = 42
POOL = SHARED_DATA / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
MANIFEST = DEE / "results/16_select_8b_coldstart_v2/selection_manifest.jsonl"
FROZEN_VAL = SHARED_DATA / "data/eval/hotpotqa_200_ids.txt"
SEL = DEE / "scripts/select_8b_coldstart_v2.py"


def load_select():
    spec = importlib.util.spec_from_file_location("select_8b_coldstart_v2", SEL)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def stratified_take_with_unused(mod, ids, n, rng, pool_meta):
    if n <= 0:
        return [], list(ids)
    if n >= len(ids):
        return sorted(ids), []
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        groups[mod.stratum_key(sid, pool_meta)].append(sid)
    keys = sorted(groups)
    total = len(ids)
    raw = {k: n * len(groups[k]) / total for k in keys}
    alloc = {k: int(raw[k]) for k in keys}
    leftover = n - sum(alloc.values())
    remainders = sorted(((raw[k] - alloc[k], k) for k in keys), reverse=True)
    for i in range(leftover):
        alloc[remainders[i % len(remainders)][1]] += 1
    chosen: list[str] = []
    unused: list[str] = []
    for key in keys:
        bucket = list(groups[key])
        rng.shuffle(bucket)
        take = min(alloc[key], len(bucket))
        chosen.extend(bucket[:take])
        unused.extend(bucket[take:])
    if len(chosen) < n:
        rng.shuffle(unused)
        extra = n - len(chosen)
        chosen.extend(unused[:extra])
        unused = unused[extra:]
    return sorted(chosen[:n]), unused


def main() -> None:
    mod = load_select()
    direct = mod.load_jsonl_map(mod.DEFAULT_DIRECT)
    oracle = mod.load_jsonl_map(mod.DEFAULT_ORACLE)
    pool_meta = mod.load_pool_meta(POOL)
    ids = sorted(set(direct) & set(oracle) & set(pool_meta))
    f1 = {sid: float(oracle[sid].get("token_f1") or 0.0) for sid in ids}
    d_ok = {sid: mod.em_ok(direct[sid]) for sid in ids}
    o_ok = {sid: mod.em_ok(oracle[sid]) for sid in ids}
    direct_ok = [s for s in ids if d_ok[s]]
    search_gap = [s for s in ids if (not d_ok[s]) and o_ok[s]]
    hard = [s for s in ids if (not d_ok[s]) and (not o_ok[s])]

    rng = random.Random(SEED)
    mod.stratified_take(direct_ok, 950, rng, pool_meta)
    search_format = mod.stratified_take(search_gap, 1250, rng, pool_meta)
    evidence_from_gap = sorted(set(search_gap) - set(search_format))
    need_hard_ev = 1150 - len(evidence_from_gap)
    hard_by_f1 = sorted(hard, key=lambda sid: (-f1[sid], sid))
    evidence_from_hard = hard_by_f1[:need_hard_ev]
    hard_remain = [sid for sid in hard if sid not in set(evidence_from_hard)]
    bands = mod.split_tertiles(hard_remain, f1)
    gh_pool = bands["genuine_hard"]
    taken_gh, leftover_gh = stratified_take_with_unused(
        mod, gh_pool, 400, rng, pool_meta
    )
    for band in ("medium", "near_solved"):
        mod.stratified_take(bands[band], 400, rng, pool_meta)

    man_rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    man_gh = sorted(
        r["sample_id"] for r in man_rows if r.get("reasoning_band") == "genuine_hard"
    )
    used_4550 = {r["sample_id"] for r in man_rows}
    frozen_dev = load_frozen_ids(str(FROZEN_VAL))
    if taken_gh != man_gh:
        raise SystemExit("REPLAY_MISMATCH genuine_hard != manifest")
    if DROP not in taken_gh:
        raise SystemExit(f"DROP_NOT_IN_GH {DROP}")

    blocked = (used_4550 - {DROP}) | frozen_dev
    legal = [sid for sid in leftover_gh if sid not in blocked]
    if not legal:
        raise SystemExit("NO_LEGAL_REPLACEMENT")
    new_id = legal[0]
    sample = next(r for r in load_jsonl(str(POOL)) if r["sample_id"] == new_id)
    refs = resolve_evidence_refs(sample)
    gold = gold_answer_of(sample)
    gold_in_ev = any(gold.lower() in (ref["text"] or "").lower() for ref in refs)

    print("replacement_reason: gold/supporting-evidence inconsistency")
    print("dropped_id:", DROP)
    print("replacement_id:", new_id)
    print("old_band: genuine_hard")
    print("new_band: genuine_hard")
    print("new_oracle_token_f1:", round(f1[new_id], 6))
    print("new_hotpot:", pool_meta[new_id])
    print("n_genuine_hard_pool:", len(gh_pool))
    print("n_taken_gh:", len(taken_gh))
    print("n_leftover_gh:", len(leftover_gh))
    print("n_legal_after_overlap:", len(legal))
    print("replay_matches_manifest_gh:", taken_gh == man_gh)
    print("new_in_4550:", new_id in used_4550)
    print("new_in_frozen_dev:", new_id in frozen_dev)
    print("old_in_frozen_dev:", DROP in frozen_dev)
    print("overlap_check_ok:", new_id not in used_4550 and new_id not in frozen_dev)
    print("gold_substring_in_evidence:", gold_in_ev)
    print("question:", sample["question"])
    print("gold:", gold)
    print("n_evidence:", len(refs))
    for i, ref in enumerate(refs, 1):
        print(f"E{i} [{ref['title']} #{ref['sentence_id']}]: {ref['text']}")


if __name__ == "__main__":
    main()
