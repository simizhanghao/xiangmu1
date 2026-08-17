#!/usr/bin/env python3
"""Gate 1.5C: freeze Qwen3-8B 4550 selection manifest. No SFT text, no Kimi."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POOL = Path("/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl")
DEFAULT_DIRECT = REPO / "results/15_relabel_8b_capability/full/direct_8b.jsonl"
DEFAULT_ORACLE = REPO / "results/15_relabel_8b_capability/full/oracle_8b.jsonl"

QUOTAS = {
    "internal": 950,
    "search_format": 1250,
    "evidence": 1150,
    "evidence_reasoning": 1200,
}
REASONING_PER_BAND = 400
SEED_DEFAULT = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--direct-jsonl", type=Path, default=DEFAULT_DIRECT)
    p.add_argument("--oracle-jsonl", type=Path, default=DEFAULT_ORACLE)
    p.add_argument("--pool", type=Path, default=POOL)
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def em_ok(row: dict) -> bool:
    if row.get("direct_correct") is True:
        return True
    return float(row.get("exact_match") or 0) >= 1.0 - 1e-9


def load_jsonl_map(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["sample_id"])] = row
    return out


def load_pool_meta(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        out[str(row["sample_id"])] = {
            "hotpot_type": str(meta.get("type") or "unknown"),
            "hotpot_level": str(meta.get("level") or "unknown"),
        }
    return out


def stratum_key(sid: str, pool_meta: dict[str, dict]) -> str:
    meta = pool_meta.get(sid, {})
    return f"{meta.get('hotpot_type', 'unknown')}|{meta.get('hotpot_level', 'unknown')}"


def stratified_take(
    ids: list[str],
    n: int,
    rng: random.Random,
    pool_meta: dict[str, dict],
) -> list[str]:
    if n <= 0:
        return []
    if n >= len(ids):
        return sorted(ids)
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in ids:
        groups[stratum_key(sid, pool_meta)].append(sid)
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
        chosen.extend(unused[: n - len(chosen)])
    return sorted(chosen[:n])


def split_tertiles(ids: list[str], f1: dict[str, float]) -> dict[str, list[str]]:
    ordered = sorted(ids, key=lambda sid: (f1[sid], sid))
    n = len(ordered)
    a, b = n // 3, 2 * n // 3
    return {
        "genuine_hard": ordered[:a],
        "medium": ordered[a:b],
        "near_solved": ordered[b:],
    }


def main() -> None:
    args = parse_args()
    out = args.output_dir
    if out.exists() and any(out.iterdir()) and not args.exist_ok:
        raise SystemExit(f"output dir not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    direct = load_jsonl_map(args.direct_jsonl)
    oracle = load_jsonl_map(args.oracle_jsonl)
    pool_meta = load_pool_meta(args.pool)
    ids = sorted(set(direct) & set(oracle) & set(pool_meta))
    if args.max_samples and args.max_samples > 0:
        ids = ids[: args.max_samples]
    if len(ids) < sum(QUOTAS.values()) and not args.max_samples:
        raise SystemExit(f"joined ids {len(ids)} < 4550")

    f1 = {sid: float(oracle[sid].get("token_f1") or 0.0) for sid in ids}
    d_ok = {sid: em_ok(direct[sid]) for sid in ids}
    o_ok = {sid: em_ok(oracle[sid]) for sid in ids}

    buckets = {
        "direct_ok__oracle_ok": [s for s in ids if d_ok[s] and o_ok[s]],
        "direct_ok__oracle_err": [s for s in ids if d_ok[s] and not o_ok[s]],
        "direct_err__oracle_ok": [s for s in ids if (not d_ok[s]) and o_ok[s]],
        "direct_err__oracle_err": [s for s in ids if (not d_ok[s]) and not o_ok[s]],
    }
    direct_ok = buckets["direct_ok__oracle_ok"] + buckets["direct_ok__oracle_err"]
    search_gap = buckets["direct_err__oracle_ok"]
    hard = buckets["direct_err__oracle_err"]

    rng = random.Random(args.seed)
    internal = stratified_take(direct_ok, QUOTAS["internal"], rng, pool_meta)
    search_format = stratified_take(search_gap, QUOTAS["search_format"], rng, pool_meta)
    evidence_from_gap = sorted(set(search_gap) - set(search_format))
    need_hard_ev = QUOTAS["evidence"] - len(evidence_from_gap)
    if need_hard_ev < 0:
        raise SystemExit("search_gap remainder exceeded evidence quota")
    hard_by_f1 = sorted(hard, key=lambda sid: (-f1[sid], sid))
    evidence_from_hard = hard_by_f1[:need_hard_ev]
    evidence = sorted(evidence_from_gap + evidence_from_hard)

    hard_remain = [sid for sid in hard if sid not in set(evidence_from_hard)]
    bands = split_tertiles(hard_remain, f1)
    reasoning: list[str] = []
    reasoning_band: dict[str, str] = {}
    for band, members in bands.items():
        taken = stratified_take(members, REASONING_PER_BAND, rng, pool_meta)
        reasoning.extend(taken)
        for sid in taken:
            reasoning_band[sid] = band
    reasoning = sorted(reasoning)

    selected = {
        "internal": internal,
        "search_format": search_format,
        "evidence": evidence,
        "evidence_reasoning": reasoning,
    }
    flat = [sid for cat in QUOTAS for sid in selected[cat]]
    overlap = len(flat) - len(set(flat))
    counts = {cat: len(selected[cat]) for cat in QUOTAS}
    ok_counts = counts == QUOTAS and overlap == 0 and len(flat) == 4550

    rows: list[dict] = []
    for cat, sids in selected.items():
        for sid in sids:
            q = (
                "direct_ok__oracle_ok"
                if d_ok[sid] and o_ok[sid]
                else "direct_ok__oracle_err"
                if d_ok[sid]
                else "direct_err__oracle_ok"
                if o_ok[sid]
                else "direct_err__oracle_err"
            )
            ev_src = None
            if cat == "evidence":
                ev_src = "search_gap_remainder" if sid in set(evidence_from_gap) else "hard_high_f1"
            rows.append(
                {
                    "sample_id": sid,
                    "category": cat,
                    "quadrant": q,
                    "direct_em": 1.0 if d_ok[sid] else 0.0,
                    "oracle_em": 1.0 if o_ok[sid] else 0.0,
                    "oracle_token_f1": f1[sid],
                    "hotpot_type": pool_meta[sid]["hotpot_type"],
                    "hotpot_level": pool_meta[sid]["hotpot_level"],
                    "reasoning_band": reasoning_band.get(sid),
                    "evidence_source": ev_src,
                }
            )
    rows.sort(key=lambda r: (r["category"], r["sample_id"]))

    def dist(sids: list[str], field: str) -> dict[str, int]:
        return dict(Counter(pool_meta[s][field] for s in sids))

    band_cuts = {
        band: {
            "n_pool": len(members),
            "f1_min": min((f1[s] for s in members), default=0.0),
            "f1_max": max((f1[s] for s in members), default=0.0),
        }
        for band, members in bands.items()
    }
    audit = {
        "gate": "GATE15C_SELECTION_PASS" if ok_counts else "GATE15C_SELECTION_FAIL",
        "seed": args.seed,
        "n_joined": len(ids),
        "quotas": QUOTAS,
        "counts": counts,
        "overlap": overlap,
        "pool_sizes": {k: len(v) for k, v in buckets.items()}
        | {
            "direct_ok": len(direct_ok),
            "search_gap": len(search_gap),
            "hard": len(hard),
            "evidence_from_gap": len(evidence_from_gap),
            "evidence_from_hard": len(evidence_from_hard),
            "hard_remain_after_evidence": len(hard_remain),
        },
        "hard_evidence_f1": {
            "n": len(evidence_from_hard),
            "f1_min": min((f1[s] for s in evidence_from_hard), default=0.0),
            "f1_max": max((f1[s] for s in evidence_from_hard), default=0.0),
        },
        "reasoning_bands": band_cuts,
        "type_by_category": {cat: dist(sids, "hotpot_type") for cat, sids in selected.items()},
        "level_by_category": {cat: dist(sids, "hotpot_level") for cat, sids in selected.items()},
        "selection_signals": ["direct_em", "oracle_em", "oracle_token_f1", "sample_id", "seed"],
        "forbidden_signals": ["evidence_f1"],
        "note": (
            "Direct-ok includes Oracle-err; those items prove Direct exact answer only. "
            "They are not interpreted as Oracle making the model worse."
        ),
    }
    manifest_path = out / "selection_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "selection_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    for cat, sids in selected.items():
        (out / f"ids_{cat}.json").write_text(json.dumps(sids, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print(audit["gate"])
    if args.debug:
        print(f"MANIFEST={manifest_path} N={len(rows)}")
    if not ok_counts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
