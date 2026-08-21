#!/usr/bin/env python3
"""Build the frozen natural-only, decision-only W6 Stage-1 curriculum."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/74_w5_controller_data/adjudicated_states.jsonl"
OUT = ROOT / "results/80_w6_stage1_dataset"
SYSTEM = (
    "You are a research sufficiency controller. Given the question, ResearchMemory, "
    "current observation, previous queries, sources, and remaining budget, decide only "
    "whether the available evidence is sufficient. Output exactly DECISION: STOP if it "
    "is sufficient; otherwise output exactly DECISION: CONTINUE. Do not output a query, "
    "missing fact, evidence, explanation, or answer."
)


def key(row):
    return hashlib.sha256(("w6-stage1-42:" + row["state_id"]).encode()).hexdigest()


def example(row, curriculum_round):
    decision = row["decision"]
    return {
        "system": SYSTEM,
        "conversations": [
            {"from": "human", "value": row["controller_input"]},
            {"from": "gpt", "value": f"DECISION: {decision}"},
        ],
        "metadata": {
            "state_id": row["state_id"],
            "sample_id": row["sample_id"],
            "decision": decision,
            "state_origin": row["state_origin"],
            "curriculum_round": curriculum_round,
        },
    }


def main():
    rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
    natural_train = [
        row
        for row in rows
        if row["controller_split"] == "train"
        and row["state_origin"] == "natural_bocha_on_policy"
    ]
    stops = sorted((row for row in natural_train if row["decision"] == "STOP"), key=key)
    continues = sorted(
        (row for row in natural_train if row["decision"] == "CONTINUE"), key=key
    )
    assert len(stops) == 662 and len(continues) == 3836

    selected = continues[: 3 * len(stops)]
    train = []
    for round_index in range(3):
        subset = selected[round_index * len(stops) : (round_index + 1) * len(stops)]
        train.extend(example(row, round_index + 1) for row in stops)
        train.extend(example(row, round_index + 1) for row in subset)

    dev_ids = {
        row["sample_id"]
        for row in rows
        if row["controller_split"] == "dev"
        and row["state_origin"] == "natural_bocha_on_policy"
    }
    train_ids = {row["metadata"]["sample_id"] for row in train}
    counts = Counter(row["metadata"]["decision"] for row in train)
    round_counts = Counter(
        (row["metadata"]["curriculum_round"], row["metadata"]["decision"])
        for row in train
    )
    manifest = {
        "gate": "W6_STAGE1_DATA_GATE_PASS"
        if len(train) == 3972
        and counts == {"STOP": 1986, "CONTINUE": 1986}
        and not (train_ids & dev_ids)
        else "W6_STAGE1_DATA_GATE_FAIL",
        "source": str(SOURCE.relative_to(ROOT)),
        "natural_only": True,
        "api_calls": 0,
        "physical_epochs": 1,
        "logical_curriculum_rounds": 3,
        "train_rows": len(train),
        "unique_stop_states": len(stops),
        "unique_continue_states": len(selected),
        "decision_counts": dict(counts),
        "round_decision_counts": {
            f"round{r}_{d.lower()}": n for (r, d), n in sorted(round_counts.items())
        },
        "question_overlap_with_frozen_dev500": len(train_ids & dev_ids),
        "masked_siblings": 0,
        "targets_decision_only": all(
            row["conversations"][-1]["value"]
            in {"DECISION: STOP", "DECISION: CONTINUE"}
            for row in train
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train.jsonl").open("w") as handle:
        for row in train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset_info = {
        "w6_stage1_decision_train": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    (OUT / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if manifest["gate"].endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
