#!/usr/bin/env python3
"""Create hidden causal depth labels for the already-frozen web-dev50.

The teacher and Web provider never receive benchmark gold. Gold is used only
inside build_one's deterministic acceptance checks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_web_multiturn_v2 import build_one
from src.sft.prototype_builder import load_jsonl
from src.tools.web_adapter import WebAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, default=ROOT / "data/web_v2/web_dev50.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results/56_web_multiturn_v2/web_dev_depth")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--teacher-base-url", default=os.environ.get("TEACHER_BASE_URL", "http://10.16.137.2:8000/v1"))
    p.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "Kimi-K2.6-CT-FP8KV"))
    p.add_argument("--teacher-api-key", default=os.environ.get("TEACHER_API_KEY", "EMPTY"))
    p.add_argument("--teacher-timeout", type=float, default=180.0)
    p.add_argument("--web-timeout", type=float, default=30.0)
    p.add_argument("--web-retries", type=int, default=2)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = SimpleNamespace(**vars(args))
    cfg.teacher_temperature = 0.0
    cfg.teacher_seed = 42
    cfg.diagnostic_log = None
    web = WebAdapter(
        provider="brave_llm_context",
        cache_dir=args.output_dir / "web_cache",
        timeout_s=args.web_timeout,
        retries=args.web_retries,
        llm_context_tokens=4096,
    )
    annotations: list[dict] = []
    reasons: dict[str, int] = {}
    for index, sample in enumerate(load_jsonl(str(args.pool))[: args.max_samples], 1):
        row1, reason1 = build_one(sample, 1, cfg, web)
        if row1 is not None:
            label, row, reason = 1, row1, "depth1_causal_pass"
        else:
            row2, reason2 = build_one(sample, 2, cfg, web)
            if row2 is not None:
                label, row, reason = 2, row2, "depth2_causal_pass"
            else:
                label, row, reason = 0, None, f"unresolved:d1={reason1};d2={reason2}"
        reasons[reason] = reasons.get(reason, 0) + 1
        annotations.append(
            {
                "sample_id": sample["sample_id"],
                "minimal_depth": label,
                "missing_after_search1": (row or {}).get("missing_after_search1") or [],
                "annotation_reason": reason,
                "gold_visible_to_teacher_or_web": False,
            }
        )
        print(f"[{index}] minimal_depth={label} {sample['sample_id']}", flush=True)
    path = args.output_dir / "hidden_depth_annotations.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in annotations:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {str(d): sum(x["minimal_depth"] == d for x in annotations) for d in (0, 1, 2)}
    summary = {
        "gate": "WEB_DEV_DEPTH_ANNOTATION_PASS" if counts["1"] and counts["2"] else "WEB_DEV_DEPTH_ANNOTATION_INCOMPLETE",
        "samples": len(annotations),
        "minimal_depth_counts": counts,
        "reasons": reasons,
        "gold_visible_to_teacher_or_web": False,
        "annotation_file": str(path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if summary["gate"] != "WEB_DEV_DEPTH_ANNOTATION_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
