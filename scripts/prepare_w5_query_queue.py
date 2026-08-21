#!/usr/bin/env python3
"""Freeze a bounded, mostly-natural CONTINUE queue for W5 query generation."""
import hashlib, json
from pathlib import Path

src = Path("results/74_w5_controller_data/adjudicated_states.jsonl")
rows = [json.loads(x) for x in src.open()]
def order(x): return hashlib.sha256(("42:" + x["state_id"]).encode()).hexdigest()
def take(split, origin, n):
    xs = [x for x in rows if x["decision"] == "CONTINUE" and x["controller_split"] == split and x["state_origin"] == origin]
    return sorted(xs, key=order)[:n]
queue = take("train", "natural_bocha_on_policy", 1400) + take("train", "counterfactual_evidence_mask", 400) + take("dev", "natural_bocha_on_policy", 10000)
out = Path("results/74_w5_controller_data/query_queue.jsonl")
with out.open("w") as f:
    for x in queue: f.write(json.dumps(x, ensure_ascii=False) + "\n")
summary = {"gate":"W5_QUERY_QUEUE_PASS", "total":len(queue), "train_natural":1400, "train_masked":400, "dev_natural":sum(x["controller_split"]=="dev" for x in queue), "gold_fields_exported":False, "output":str(out)}
Path("results/74_w5_controller_data/query_queue_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
