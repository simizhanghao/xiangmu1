#!/usr/bin/env python3
"""SGLang probability audit: support + Δlogp + IS ratio + ESS. No training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SGLang vs trainer logprob audit.")
    p.add_argument("--config", type=str, default=str(REPO / "config" / "harness_v1.json"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO / "results" / "34_sglang_prob_audit"),
    )
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--dump", type=str, default="")
    p.add_argument("--slice-src", type=str, default="")
    p.add_argument("--slice-dst", type=str, default="")
    p.add_argument("--recompute-hf", action="store_true")
    p.add_argument(
        "--model-path",
        type=str,
        default=str(REPO / "artifacts" / "models" / "qwen3_8b_sft_merged"),
    )
    p.add_argument("--hf-max-rows", type=int, default=0, help="0 = all dump rows")
    return p.parse_args()


def _pct(xs: Sequence[float], q: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round((q / 100.0) * (len(ys) - 1)))))
    return float(ys[i])


def slice_parquet(src: Path, dst: Path, n: int) -> int:
    import pyarrow.parquet as pq

    table = pq.read_table(src)
    if n <= 0 or n > table.num_rows:
        n = table.num_rows
    out = table.slice(0, n)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out.replace_schema_metadata(None), dst, compression="snappy")
    return int(out.num_rows)


def _finite(xs: Optional[Sequence[Any]]) -> bool:
    if xs is None:
        return False
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(v):
            return False
    return True


def analyze_dump(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    search_n = sum(1 for r in rows if int(r.get("search_count") or 0) > 0)
    finish_n = sum(1 for r in rows if int(r.get("finish") or 0) == 1)
    internal_n = sum(1 for r in rows if int(r.get("used_internal") or 0) == 1)
    missing_lp = sum(1 for r in rows if not r.get("response_logprobs"))
    inf_lp = 0
    token_mu: List[float] = []
    for r in rows:
        lp = r.get("response_logprobs")
        mask = r.get("response_mask") or []
        if not lp:
            continue
        if not _finite(lp):
            inf_lp += 1
            continue
        for i, v in enumerate(lp):
            if i < len(mask) and int(mask[i]) != 1:
                continue
            token_mu.append(float(v))
    search_rate = search_n / n if n else 0.0
    support_ok = search_n > 0 and inf_lp == 0 and missing_lp == 0
    verdict = (
        "SGLANG_SUPPORT_PASS"
        if support_ok
        else (
            "SGLANG_FAST_PATH_FAIL_SUPPORT"
            if search_n == 0 or inf_lp > 0
            else "SGLANG_SUPPORT_INCOMPLETE"
        )
    )
    return {
        "n_traj": n,
        "search_traj": search_n,
        "search_rate": round(search_rate, 4),
        "finish_rate": round(finish_n / n, 4) if n else 0.0,
        "internal_rate": round(internal_n / n, 4) if n else 0.0,
        "missing_rollout_logprob_rows": missing_lp,
        "nonfinite_rollout_logprob_rows": inf_lp,
        "n_rollout_tokens": len(token_mu),
        "support_ok": support_ok,
        "support_verdict": verdict,
    }


def recompute_hf(
    rows: List[Dict[str, Any]], model_path: str, max_rows: int
) -> Dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    deltas: List[float] = []
    rhos: List[float] = []
    seq_ws: List[float] = []
    nan_pi = 0
    used = rows if not max_rows else rows[:max_rows]
    for r in used:
        pids = list(r.get("prompt_ids") or [])
        rids = list(r.get("response_ids") or [])
        mask = list(r.get("response_mask") or [])
        mu = list(r.get("response_logprobs") or [])
        if not pids or not rids or not mu or len(mu) != len(rids):
            continue
        ids = pids + rids
        t = torch.tensor([ids], device="cuda:0")
        with torch.no_grad():
            logits = model(t).logits
        logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        nxt = t[0, 1:]
        token_lp = logp.gather(-1, nxt.unsqueeze(-1)).squeeze(-1)
        resp_lp = token_lp[len(pids) - 1 : len(pids) - 1 + len(rids)]
        if resp_lp.numel() != len(rids):
            continue
        seq_log_w = 0.0
        n_w = 0
        for i, (pi, muv, m) in enumerate(zip(resp_lp.tolist(), mu, mask)):
            if int(m) != 1:
                continue
            if not math.isfinite(pi) or not math.isfinite(float(muv)):
                nan_pi += 1
                continue
            d = float(pi) - float(muv)
            rho = math.exp(d)
            deltas.append(abs(d))
            rhos.append(rho)
            seq_log_w += d
            n_w += 1
        if n_w:
            seq_ws.append(math.exp(seq_log_w))
    ess = None
    ess_ratio = None
    if rhos:
        s1 = sum(rhos)
        s2 = sum(w * w for w in rhos)
        ess = (s1 * s1) / s2 if s2 > 0 else 0.0
        ess_ratio = ess / len(rhos)
    mismatch_ok = bool(rhos) and nan_pi == 0 and (ess_ratio or 0) >= 0.2
    return {
        "n_hf_rows": len(used),
        "n_delta_tokens": len(deltas),
        "trainer_nonfinite_tokens": nan_pi,
        "mean_abs_dlogp": round(statistics.mean(deltas), 6) if deltas else None,
        "median_abs_dlogp": round(statistics.median(deltas), 6) if deltas else None,
        "p95_abs_dlogp": round(_pct(deltas, 95) or 0, 6) if deltas else None,
        "p99_abs_dlogp": round(_pct(deltas, 99) or 0, 6) if deltas else None,
        "max_abs_dlogp": round(max(deltas), 6) if deltas else None,
        "rho_mean": round(statistics.mean(rhos), 6) if rhos else None,
        "rho_std": round(statistics.pstdev(rhos), 6) if len(rhos) > 1 else None,
        "rho_p01": round(_pct(rhos, 1) or 0, 6) if rhos else None,
        "rho_p05": round(_pct(rhos, 5) or 0, 6) if rhos else None,
        "rho_p50": round(_pct(rhos, 50) or 0, 6) if rhos else None,
        "rho_p95": round(_pct(rhos, 95) or 0, 6) if rhos else None,
        "rho_p99": round(_pct(rhos, 99) or 0, 6) if rhos else None,
        "rho_max": round(max(rhos), 6) if rhos else None,
        "ess_token": None if ess is None else round(ess, 4),
        "ess_ratio_token": None if ess_ratio is None else round(ess_ratio, 4),
        "n_seq_weights": len(seq_ws),
        "mismatch_ok": mismatch_ok,
        "mismatch_verdict": (
            "SGLANG_MISMATCH_MILD"
            if mismatch_ok
            else "SGLANG_MISMATCH_TOXIC_OR_EMPTY"
        ),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.slice_src:
        n = slice_parquet(Path(args.slice_src), Path(args.slice_dst), args.max_samples)
        print(json.dumps({"sliced_rows": n, "dst": args.slice_dst}), flush=True)
        if not args.dump:
            return
    dump_path = Path(args.dump) if args.dump else out / "sglang_traj_dump.jsonl"
    if not dump_path.is_file():
        raise SystemExit(f"DUMP_MISSING {dump_path}")
    rows = [json.loads(x) for x in dump_path.open(encoding="utf-8") if x.strip()]
    summary: Dict[str, Any] = {
        "dump": str(dump_path),
        "n_dump_lines": len(rows),
        **analyze_dump(rows),
    }
    if args.recompute_hf:
        summary.update(recompute_hf(rows, args.model_path, args.hf_max_rows))
    if summary.get("support_verdict") == "SGLANG_FAST_PATH_FAIL_SUPPORT":
        summary["gate"] = "SGLANG_FAST_PATH_FAIL_SUPPORT"
    elif summary.get("support_ok") and summary.get("mismatch_ok"):
        summary["gate"] = "SGLANG_PROB_AUDIT_PASS"
    elif summary.get("support_ok"):
        summary["gate"] = "SGLANG_SUPPORT_PASS_MISMATCH_PENDING"
    else:
        summary["gate"] = summary.get("support_verdict") or "SGLANG_PROB_AUDIT_INCOMPLETE"
    (out / "sglang_prob_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(summary["gate"], flush=True)


if __name__ == "__main__":
    main()
