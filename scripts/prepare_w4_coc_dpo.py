#!/usr/bin/env python3
"""Build leakage-free, consequence-aware atomic decision preference pairs."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


BUDGET_RE = re.compile(r"Remaining Budget:\s*\d+")


def sanitize(text: str, budget: int) -> str:
    return BUDGET_RE.sub(f"Remaining Budget: {budget}", text)


def prompt_prefix(example: dict, budget: int) -> list[dict]:
    conversations = example["conversations"][:-1]
    return [
        {"from": item["from"], "value": sanitize(item["value"], budget)}
        for item in conversations
    ]


def response(decision: str, consequence: str, *, query: str = "", answer: str = "") -> str:
    lines = ["<internal>", f"Decision: {decision}"]
    if query:
        lines.append(f"Next Query: {query}")
    lines.append(f"Consequence: {consequence}")
    lines.append("</internal>")
    if decision == "SEARCH":
        lines.extend(["<search>", query, "</search>"])
    elif answer:
        lines.extend(["<answer>", answer, "</answer>"])
    return "\n".join(lines)


def pair(row: dict, example: dict, pair_type: str, chosen: str, rejected: str, budget: int) -> dict:
    return {
        "conversations": prompt_prefix(example, budget),
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
        "system": row["decision_examples"][0]["system"],
        "pair_id": f'{row["sample_id"]}__{pair_type}',
        "sample_id": row["sample_id"],
        "pair_type": pair_type,
        "target_depth": row["target_depth"],
        "fixed_remaining_budget": budget,
    }


def build_pairs(row: dict, budget: int) -> dict[str, dict]:
    by_type = {x["metadata"]["decision_type"]: x for x in row["decision_examples"]}
    final_answer = str(row.get("final_answer") or "").strip()
    q1 = row["turns"][0]["query"].strip()
    stop = response("ANSWER", "current evidence is sufficient for a grounded answer", answer=final_answer)

    if int(row["target_depth"]) == 1:
        unnecessary = response(
            "SEARCH",
            "unnecessary repeated search after the answer is already grounded",
            query=q1,
        )
        return {
            "d1_obs1_stop": pair(row, by_type["post_obs_stop"], "d1_obs1_stop", stop, unnecessary, budget)
        }

    q2 = row["turns"][1]["query"].strip()
    forced = str(row.get("minimal_depth_audit", {}).get("forced1_answer") or final_answer).strip()
    continue_good = response(
        "SEARCH",
        "the new query resolves the missing evidence with a new grounded source",
        query=q2,
    )
    stop_early = response(
        "ANSWER",
        "premature answer while required evidence is still missing",
        answer=forced,
    )
    redundant = response(
        "SEARCH",
        "unnecessary repeated search after the second observation is sufficient",
        query=q2,
    )
    return {
        "d2_obs1_continue": pair(
            row, by_type["post_obs_continue"], "d2_obs1_continue", continue_good, stop_early, budget
        ),
        "d2_obs2_stop": pair(row, by_type["post_obs_stop"], "d2_obs2_stop", stop, redundant, budget),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/60_provider_decoupled_replay/train440_v2_disjoint/full_trajectories.jsonl")
    parser.add_argument("--output-dir", default="results/68_w4_coc/data")
    parser.add_argument("--fixed-remaining-budget", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).open()]
    dev_ids = {
        json.loads(line)["sample_id"]
        for line in Path("results/60_provider_decoupled_replay/behavior_dev40/full_trajectories.jsonl").open()
    }
    overlap = {row["sample_id"] for row in rows} & dev_ids
    if overlap:
        raise RuntimeError(f"behavior-dev leakage: {len(overlap)} ids")

    buckets: dict[str, list[dict]] = {"d1_obs1_stop": [], "d2_obs1_continue": [], "d2_obs2_stop": []}
    for row in rows:
        for key, value in build_pairs(row, args.fixed_remaining_budget).items():
            buckets[key].append(value)

    # Balance the atomic decision label without duplicating examples: 110 STOP
    # from each sufficient-state type and 220 CONTINUE from insufficient D2@Obs1.
    rng = random.Random(args.seed)
    rng.shuffle(buckets["d1_obs1_stop"])
    rng.shuffle(buckets["d2_obs2_stop"])
    rng.shuffle(buckets["d2_obs1_continue"])
    train = (
        buckets["d1_obs1_stop"][:110]
        + buckets["d2_obs2_stop"][:110]
        + buckets["d2_obs1_continue"][:220]
    )
    rng.shuffle(train)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "train.jsonl").open("w") as handle:
        for item in train:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    dataset_info = {
        "w4_coc_train": {
            "file_name": "train.jsonl",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "chosen": "chosen", "rejected": "rejected", "system": "system"},
            "tags": {
                "role_tag": "from", "content_tag": "value", "user_tag": "human",
                "assistant_tag": "gpt", "observation_tag": "observation",
            },
        }
    }
    (out / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n")
    summary = {
        "gate": "W4_COC_DATA_PASS",
        "source_rows": len(rows),
        "behavior_dev_overlap": 0,
        "preference_pairs": len(train),
        "pair_counts": {key: sum(x["pair_type"] == key for x in train) for key in buckets},
        "chosen_decisions": {
            "ANSWER": sum("Decision: ANSWER" in x["chosen"]["value"] for x in train),
            "SEARCH": sum("Decision: SEARCH" in x["chosen"]["value"] for x in train),
        },
        "fixed_remaining_budget": args.fixed_remaining_budget,
        "long_web_text_in_response": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
