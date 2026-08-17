#!/usr/bin/env python3
"""Compare vLLM Agent n=8 traces against frozen HF Harness v1.

Does not regenerate. Official Gate 3 HF@200 F1 is not rewritten.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_EMPTY_THINK = re.compile(r"<think>\s*</think>", re.IGNORECASE)
_OBS_TAG = re.compile(r"<observation\b", re.IGNORECASE)
_EXTRA_CONTINUE = re.compile(
    r"Continue\.\s*Prefer|<search> again only if necessary", re.IGNORECASE
)
_PROSE_ONLY = re.compile(r"</(?:search|answer|internal)>", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM vs HF Harness v1 n=8 parity.")
    p.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config" / "harness_v1.json"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "results" / "27_vllm_hf_parity_n8"),
    )
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--hf-dir",
        type=str,
        default=str(
            REPO_ROOT
            / "results"
            / "23_gate3_smoke_n8"
            / "agent_rollout_n8_20260817_095138_g3_agent_n8_fix3"
        ),
    )
    p.add_argument("--vllm-dir", type=str, required=True)
    return p.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def count_in_raw(rows: List[Dict[str, Any]], cre: re.Pattern[str]) -> int:
    n = 0
    for r in rows:
        for g in r.get("raw_generations") or []:
            n += len(cre.findall(g or ""))
    return n


def prose_collapse_n(rows: List[Dict[str, Any]]) -> int:
    bad = 0
    for r in rows:
        gens = [g or "" for g in (r.get("raw_generations") or [])]
        if not gens:
            bad += 1
            continue
        if not any(_PROSE_ONLY.search(g) for g in gens) and not r.get("finished"):
            bad += 1
    return bad


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = Path(args.hf_dir)
    vllm_dir = Path(args.vllm_dir)
    if not str(args.vllm_dir) or not (vllm_dir / "metrics.jsonl").is_file():
        raise SystemExit(
            f"VLLM_DIR_MISSING: {vllm_dir} (need metrics.jsonl). "
            "Do not audit until the vLLM n=8 run finishes."
        )
    hf_rows = load_jsonl(hf_dir / "metrics.jsonl")[: args.max_samples]
    vllm_rows = load_jsonl(vllm_dir / "metrics.jsonl")[: args.max_samples]
    hf_sum = load_summary(hf_dir)
    vllm_sum = load_summary(vllm_dir)

    empty_think = count_in_raw(vllm_rows, _EMPTY_THINK)
    obs_tag = count_in_raw(vllm_rows, _OBS_TAG)
    extra_continue = count_in_raw(vllm_rows, _EXTRA_CONTINUE)
    prose_n = prose_collapse_n(vllm_rows)

    hf_ids = [str(r.get("sample_id")) for r in hf_rows]
    vllm_ids = [str(r.get("sample_id")) for r in vllm_rows]
    id_match = hf_ids == vllm_ids and len(vllm_ids) == args.max_samples

    hf_by = {str(r["sample_id"]): r for r in hf_rows}
    paired: List[Dict[str, Any]] = []
    route_eq = 0
    query_eq = 0
    fin_eq = 0
    for r in vllm_rows:
        sid = str(r.get("sample_id"))
        h = hf_by.get(sid) or {}
        same_route = (r.get("route_first") == h.get("route_first")) if h else False
        same_q = (r.get("search_queries") == h.get("search_queries")) if h else False
        same_fin = (bool(r.get("finished")) == bool(h.get("finished"))) if h else False
        route_eq += int(same_route)
        query_eq += int(same_q)
        fin_eq += int(same_fin)
        paired.append(
            {
                "sample_id": sid,
                "hf_route": h.get("route_first"),
                "vllm_route": r.get("route_first"),
                "hf_finished": h.get("finished"),
                "vllm_finished": r.get("finished"),
                "hf_search_queries": h.get("search_queries"),
                "vllm_search_queries": r.get("search_queries"),
                "hf_em": (h.get("metrics") or {}).get("exact_match"),
                "vllm_em": (r.get("metrics") or {}).get("exact_match"),
                "vllm_errors": r.get("validation_errors") or [],
            }
        )

    finish_rate = float(vllm_sum.get("finish_rate") or 0.0)
    parse_ok = float(vllm_sum.get("parse_ok_rate") or 0.0)
    search_rate = float(vllm_sum.get("search_rate") or 0.0)
    obs_mask = float(vllm_sum.get("observation_mask_ok_rate") or 0.0)

    protocol_ok = empty_think == 0 and obs_tag == 0 and extra_continue == 0
    health_ok = (
        finish_rate >= 0.8
        and parse_ok >= 0.8
        and search_rate > 0
        and obs_mask >= 0.8
        and prose_n == 0
        and id_match
    )
    gate = "VLLM_HF_PARITY_PASS" if protocol_ok and health_ok else "VLLM_HF_PARITY_FAIL"

    report = {
        "gate": gate,
        "n": len(vllm_rows),
        "id_match": id_match,
        "empty_think": empty_think,
        "observation_tag": obs_tag,
        "extra_continue": extra_continue,
        "prose_collapse_n": prose_n,
        "vllm": {
            "finish_rate": finish_rate,
            "parse_ok_rate": parse_ok,
            "search_rate": search_rate,
            "mean_token_f1": vllm_sum.get("mean_token_f1"),
            "mean_evidence_f1": vllm_sum.get("mean_evidence_f1"),
            "observation_mask_ok_rate": obs_mask,
            "backend": vllm_sum.get("backend"),
            "run_dir": str(vllm_dir),
        },
        "hf_ref": {
            "finish_rate": hf_sum.get("finish_rate"),
            "search_rate": hf_sum.get("search_rate"),
            "mean_token_f1": hf_sum.get("mean_token_f1"),
            "run_dir": str(hf_dir),
            "note": "n=8 smoke reference; official Gate3 remains HF@200 F1=0.6649",
        },
        "pairwise": {
            "route_agree": route_eq,
            "query_agree": query_eq,
            "finish_agree": fin_eq,
            "n": len(vllm_rows),
        },
        "note": "F1 need not match HF. Protocol + finish + search presence are the gate.",
    }
    (out_dir / "parity_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "paired_cases.jsonl").open("w", encoding="utf-8") as f:
        for row in paired:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(gate)
    if gate != "VLLM_HF_PARITY_PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
