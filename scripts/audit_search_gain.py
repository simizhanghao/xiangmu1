#!/usr/bin/env python3
"""CPU-only: is GRPO's Answer-F1 gain just always-search?

Joins existing rollout metrics.jsonl by sample_id. No GPU, no new eval.
Primary split is frozen-dev@200 (official Δ_RL). formal-dev@1000 is confirm-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = {
    "frozen200": {
        "split": "frozen-dev@200",
        "official_delta_rl": True,
        "sft": REPO
        / "results/42_frozen_dev_sft_vllm/n200/agent_rollout_n200_20260819_005714_sft_vllm_n200/metrics.jsonl",
        "grpo200": REPO
        / "results/41_frozen_dev_formal_grpo200/n200/agent_rollout_n200_20260819_000410_formal200_n200/metrics.jsonl",
        "grpo400": REPO
        / "results/45_frozen_dev_formal_grpo400/n200/agent_rollout_n200_20260819_101124_formal400_n200/metrics.jsonl",
        "grpo600": REPO
        / "results/48_frozen_dev_formal_grpo600/n200/agent_rollout_n200_20260819_170302_formal600_n200/metrics.jsonl",
    },
    "formal1000": {
        "split": "formal-dev@1000",
        "official_delta_rl": False,
        "sft": REPO
        / "results/49_formal_dev1000/n1000_sft/agent_rollout_n1000_20260819_215700_fd1000_n1000_sft/metrics.jsonl",
        "grpo200": REPO
        / "results/49_formal_dev1000/n1000_formal200/agent_rollout_n1000_20260819_224558_fd1000_n1000_formal200/metrics.jsonl",
        "grpo400": REPO
        / "results/49_formal_dev1000/n1000_formal400/agent_rollout_n1000_20260819_175758_fd1000_n1000_formal400/metrics.jsonl",
        "grpo600": REPO
        / "results/49_formal_dev1000/n1000_formal600/agent_rollout_n1000_20260819_234418_fd1000_n1000_formal600/metrics.jsonl",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Light always-search vs already-search audit.")
    p.add_argument("--config", type=str, default=str(REPO / "config" / "harness_v1.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO / "results" / "50_audit_search_gain"),
    )
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--splits",
        type=str,
        default="frozen200,formal1000",
        help="Comma list: frozen200 and/or formal1000",
    )
    return p.parse_args()


def load_jsonl(path: Path, max_samples: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or "")
            if not sid:
                continue
            out[sid] = row
            if max_samples and len(out) >= max_samples:
                break
    return out


def searched(row: Dict[str, Any]) -> bool:
    n = int((row.get("metrics") or {}).get("search_count") or 0)
    if n >= 1:
        return True
    return str(row.get("route_first") or "") == "search"


def metric(row: Dict[str, Any], key: str) -> float:
    return float((row.get("metrics") or {}).get(key) or 0.0)


def gen_tokens(row: Dict[str, Any]) -> float:
    cost = row.get("cost_info") or {}
    if "generated_tokens" in cost:
        return float(cost.get("generated_tokens") or 0.0)
    return float((row.get("metrics") or {}).get("generated_tokens") or 0.0)


def search_count(row: Dict[str, Any]) -> int:
    return int((row.get("metrics") or {}).get("search_count") or 0)


def dup_query(row: Dict[str, Any]) -> float:
    return float((row.get("metrics") or {}).get("duplicate_query_count") or 0.0)


def mean(xs: Iterable[float]) -> float:
    vals = list(xs)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def round4(x: float) -> float:
    return round(float(x), 4)


def pair_means(
    ids: List[str],
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    def col(key: str, side: Dict[str, Dict[str, Any]]) -> float:
        return round4(mean(metric(side[i], key) for i in ids))

    return {
        "n": len(ids),
        "share": None,
        "sft_f1": col("token_f1", left),
        "grpo_f1": col("token_f1", right),
        "delta_f1": None,
        "sft_em": col("exact_match", left),
        "grpo_em": col("exact_match", right),
        "delta_em": None,
        "sft_evidence_f1": col("evidence_f1", left),
        "grpo_evidence_f1": col("evidence_f1", right),
        "delta_evidence_f1": None,
    }


def fill_deltas(block: Dict[str, Any], n_all: int) -> Dict[str, Any]:
    block["share"] = round4(block["n"] / n_all) if n_all else 0.0
    block["delta_f1"] = round4(block["grpo_f1"] - block["sft_f1"])
    block["delta_em"] = round4(block["grpo_em"] - block["sft_em"])
    block["delta_evidence_f1"] = round4(block["grpo_evidence_f1"] - block["sft_evidence_f1"])
    return block


def examples(
    ids: List[str],
    a: Dict[str, Dict[str, Any]],
    b: Dict[str, Dict[str, Any]],
    key: str,
    limit: int,
    reverse: bool = False,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[float, str]] = []
    for sid in ids:
        d = metric(b[sid], key) - metric(a[sid], key)
        scored.append((d, sid))
    scored.sort(key=lambda t: t[0], reverse=not reverse)
    out = []
    for d, sid in scored[:limit]:
        out.append(
            {
                "sample_id": sid,
                "delta_f1": round4(d),
                "left_f1": round4(metric(a[sid], "token_f1")),
                "right_f1": round4(metric(b[sid], "token_f1")),
                "left_search": search_count(a[sid]),
                "right_search": search_count(b[sid]),
                "left_gen": round(gen_tokens(a[sid]), 1),
                "right_gen": round(gen_tokens(b[sid]), 1),
            }
        )
    return out


def audit_split(
    name: str,
    spec: Dict[str, Any],
    max_samples: int,
    debug: bool,
) -> Dict[str, Any]:
    sft = load_jsonl(spec["sft"], max_samples)
    g200 = load_jsonl(spec["grpo200"], max_samples)
    g400 = load_jsonl(spec["grpo400"], max_samples)
    g600 = load_jsonl(spec["grpo600"], max_samples)
    ids = sorted(set(sft) & set(g200) & set(g400) & set(g600))
    n = len(ids)
    if n == 0:
        raise SystemExit(f"{name}: no overlapping sample_id")

    already = [i for i in ids if searched(sft[i])]
    flipped = [i for i in ids if (not searched(sft[i])) and searched(g400[i])]
    still_int = [i for i in ids if (not searched(sft[i])) and (not searched(g400[i]))]
    sft_int = [i for i in ids if not searched(sft[i])]

    already_b = fill_deltas(pair_means(already, sft, g400), n)
    flipped_b = fill_deltas(pair_means(flipped, sft, g400), n)
    overall = fill_deltas(pair_means(ids, sft, g400), n)

    # Weighted contribution of each stratum to the overall F1 delta
    contrib_already = round4(already_b["delta_f1"] * already_b["share"])
    contrib_flipped = round4(flipped_b["delta_f1"] * flipped_b["share"])
    residual = round4(overall["delta_f1"] - contrib_already - contrib_flipped)

    better400 = [i for i in ids if metric(g400[i], "token_f1") > metric(g200[i], "token_f1") + 0.01]
    better200 = [i for i in ids if metric(g200[i], "token_f1") > metric(g400[i], "token_f1") + 0.01]
    tie = n - len(better400) - len(better200)

    def length_bucket(row: Dict[str, Any]) -> str:
        sc = search_count(row)
        gt = gen_tokens(row)
        if sc >= 2:
            return "second_search"
        if dup_query(row) > 0:
            return "duplicate_query"
        if gt >= 450:
            return "verbose_single_search"
        return "normal"

    buckets = {"second_search": 0, "duplicate_query": 0, "verbose_single_search": 0, "normal": 0}
    extra_tokens = 0.0
    extra_from_second = 0.0
    extra_from_verbose = 0.0
    for i in ids:
        bkt = length_bucket(g600[i])
        buckets[bkt] += 1
        extra = max(0.0, gen_tokens(g600[i]) - gen_tokens(g400[i]))
        extra_tokens += extra
        if bkt == "second_search":
            extra_from_second += extra
        elif bkt == "verbose_single_search":
            extra_from_verbose += extra

    verdict_parts = []
    if already_b["n"] == 0:
        verdict_parts.append("no SFT-search subset")
    elif already_b["delta_f1"] >= 0.02:
        verdict_parts.append(
            f"GRPO still gains Answer F1 on SFT-already-search items ({already_b['delta_f1']:+.4f})"
        )
    else:
        verdict_parts.append(
            f"little/no Answer F1 gain on SFT-already-search items ({already_b['delta_f1']:+.4f})"
        )
    if flipped_b["n"]:
        verdict_parts.append(
            f"always-search flip n={flipped_b['n']} ΔF1={flipped_b['delta_f1']:+.4f}"
        )
    verdict = "; ".join(verdict_parts)

    report = {
        "split": spec["split"],
        "official_delta_rl": bool(spec["official_delta_rl"]),
        "n_joined": n,
        "files": {k: str(spec[k]) for k in ("sft", "grpo200", "grpo400", "grpo600")},
        "overall_sft_vs_400": overall,
        "sft_already_search": already_b,
        "sft_internal_to_400_search": flipped_b,
        "sft_internal_n": len(sft_int),
        "still_internal_n": len(still_int),
        "delta_f1_contribution": {
            "already_search": contrib_already,
            "search_flip": contrib_flipped,
            "residual_other": residual,
            "note": "share * stratum ΔF1; sums to overall ΔF1 aside from rounding",
        },
        "paired_400_vs_200": {
            "n_400_better_f1": len(better400),
            "n_200_better_f1": len(better200),
            "n_tie": tie,
            "mean_delta_f1_400_minus_200": round4(
                mean(metric(g400[i], "token_f1") - metric(g200[i], "token_f1") for i in ids)
            ),
            "mean_delta_evidence_400_minus_200": round4(
                mean(metric(g400[i], "evidence_f1") - metric(g200[i], "evidence_f1") for i in ids)
            ),
            "examples_400_wins": examples(better400, g200, g400, "token_f1", 3),
            "examples_200_wins": examples(better200, g400, g200, "token_f1", 3),
        },
        "length_600_vs_400": {
            "mean_gen_400": round(mean(gen_tokens(g400[i]) for i in ids), 1),
            "mean_gen_600": round(mean(gen_tokens(g600[i]) for i in ids), 1),
            "p_search_2_400": round4(mean(1.0 if search_count(g400[i]) >= 2 else 0.0 for i in ids)),
            "p_search_2_600": round4(mean(1.0 if search_count(g600[i]) >= 2 else 0.0 for i in ids)),
            "mean_dup_400": round4(mean(dup_query(g400[i]) for i in ids)),
            "mean_dup_600": round4(mean(dup_query(g600[i]) for i in ids)),
            "bucket_n": buckets,
            "bucket_share": {k: round4(v / n) for k, v in buckets.items()},
            "extra_tokens_total": round(extra_tokens, 1),
            "extra_tokens_from_second_search": round(extra_from_second, 1),
            "extra_tokens_from_verbose_single": round(extra_from_verbose, 1),
        },
        "examples_already_search_gain": examples(already, sft, g400, "token_f1", 3),
        "examples_search_flip_gain": examples(flipped, sft, g400, "token_f1", 3),
        "verdict": verdict,
    }
    if debug:
        report["debug_first_id"] = ids[0]
    return report


def print_table(name: str, rep: Dict[str, Any]) -> None:
    o = rep["overall_sft_vs_400"]
    a = rep["sft_already_search"]
    f = rep["sft_internal_to_400_search"]
    c = rep["delta_f1_contribution"]
    p = rep["paired_400_vs_200"]
    L = rep["length_600_vs_400"]
    print(f"\n=== {name}  {rep['split']}  n={rep['n_joined']} ===")
    print(
        f"overall     n={o['n']:4d}  SFT F1={o['sft_f1']:.4f}  @400 F1={o['grpo_f1']:.4f}  "
        f"ΔF1={o['delta_f1']:+.4f}  ΔEv={o['delta_evidence_f1']:+.4f}"
    )
    print(
        f"already-search n={a['n']:4d} ({a['share']:.3f})  "
        f"SFT F1={a['sft_f1']:.4f}  @400 F1={a['grpo_f1']:.4f}  "
        f"ΔF1={a['delta_f1']:+.4f}  ΔEv={a['delta_evidence_f1']:+.4f}  "
        f"contrib={c['already_search']:+.4f}"
    )
    print(
        f"search-flip    n={f['n']:4d} ({f['share']:.3f})  "
        f"SFT F1={f['sft_f1']:.4f}  @400 F1={f['grpo_f1']:.4f}  "
        f"ΔF1={f['delta_f1']:+.4f}  ΔEv={f['delta_evidence_f1']:+.4f}  "
        f"contrib={c['search_flip']:+.4f}"
    )
    print(
        f"@400 vs @200   400-better={p['n_400_better_f1']}  "
        f"200-better={p['n_200_better_f1']}  tie={p['n_tie']}  "
        f"ΔF1={p['mean_delta_f1_400_minus_200']:+.4f}  "
        f"ΔEv={p['mean_delta_evidence_400_minus_200']:+.4f}"
    )
    print(
        f"@600 length    gen {L['mean_gen_400']:.0f}→{L['mean_gen_600']:.0f}  "
        f"search2 {L['p_search_2_400']:.3f}→{L['p_search_2_600']:.3f}  "
        f"buckets={L['bucket_share']}"
    )
    print(f"verdict: {rep['verdict']}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    combined: Dict[str, Any] = {
        "gate": "AUDIT_SEARCH_GAIN",
        "purpose": "Is official GRPO Δ_RL only always-search? CPU join of existing metrics.",
        "best_controlled_policy": "GRPO step400",
        "config": args.config,
        "seed": args.seed,
        "splits": {},
    }
    for name in wanted:
        if name not in DEFAULT_RUNS:
            raise SystemExit(f"unknown split {name}")
        spec = DEFAULT_RUNS[name]
        for key in ("sft", "grpo200", "grpo400", "grpo600"):
            path = spec[key]
            if not path.is_file():
                raise SystemExit(f"missing {name} {key}: {path}")
        rep = audit_split(name, spec, args.max_samples, args.debug)
        combined["splits"][name] = {k: v for k, v in rep.items() if k != "files"}
        (out_dir / f"{name}_summary.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print_table(name, rep)

    official = combined["splits"].get("frozen200", {})
    already = (official.get("sft_already_search") or {}).get("delta_f1")
    flip_c = (official.get("delta_f1_contribution") or {}).get("search_flip")
    already_c = (official.get("delta_f1_contribution") or {}).get("already_search")
    combined["project_read"] = {
        "use_frozen200_for_official_delta": True,
        "already_search_delta_f1": already,
        "already_search_contribution": already_c,
        "search_flip_contribution": flip_c,
        "one_liner": (
            "If already-search ΔF1>0, GRPO is not only always-search; "
            "compare the two contributions."
        ),
    }
    summary_path = out_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
