#!/usr/bin/env python3
"""Build natural-only compact-evidence W7 sufficiency-and-gap supervision."""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "results/74_w5_controller_data/adjudicated_states.jsonl"
QUERIES = ROOT / "results/75_w5_query_teacher/full/queries.jsonl"
OUT = ROOT / "results/83_w7_s2g_dataset"
SYSTEM = (
    "You are an evidence sufficiency and gap judge. Given a question and a compact "
    "Evidence Context, output exactly SUFFICIENT: YES followed by GAPS: NONE when the "
    "evidence is sufficient. Otherwise output SUFFICIENT: NO followed by GAPS: and one "
    "concise missing fact. Do not produce a search query, explanation, or answer."
)
WORD = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
DOC = re.compile(r"(?ms)^\[(web_[^\]]+)\]\s*(.*?)(?=^\[web_|\Z)")


def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def stable(row):
    return hashlib.sha256(("w7-s2g-42:" + row["state_id"]).encode()).hexdigest()


def terms(text):
    return [token.lower() for token in WORD.findall(text)]


def split_sentences(text):
    pieces = re.split(r"(?<=[.!?。！？；;])\s*|\n+", text)
    output = []
    for piece in pieces:
        piece = " ".join(piece.split()).strip()
        if not piece:
            continue
        output.extend(piece[i : i + 360] for i in range(0, len(piece), 360))
    return [piece for piece in output if len(piece) >= 20]


def parse_state(row):
    text = row["controller_input"]
    match = re.search(r"(?s)^Question:\s*(.*?)\n\nCurrent Observation:\s*(.*?)(?:\n\n<research_state>|\Z)", text)
    if not match:
        raise ValueError(f"cannot parse controller_input: {row['state_id']}")
    question, observation = match.group(1).strip(), match.group(2).strip()
    docs = DOC.findall(observation)
    if not docs:
        docs = [("web_fallback", observation)]
    candidates = []
    for source_id, content in docs:
        for position, sentence in enumerate(split_sentences(content)):
            candidates.append((source_id, position, sentence))
    if not candidates:
        raise ValueError(f"empty observation sentences: {row['state_id']}")
    return question, observation, candidates


def compact(row, top_k=8, per_source=2):
    question, observation, candidates = parse_state(row)
    query_terms = terms(question)
    doc_terms = [terms(sentence) for _, _, sentence in candidates]
    n = len(candidates)
    avgdl = sum(len(tokens) for tokens in doc_terms) / max(1, n)
    df = Counter()
    for tokens in doc_terms:
        df.update(set(tokens))
    scored = []
    for candidate, tokens in zip(candidates, doc_terms):
        tf = Counter(tokens)
        dl = len(tokens)
        score = 0.0
        for term in query_terms:
            if not tf[term]:
                continue
            idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
            score += idf * tf[term] * 2.2 / (tf[term] + 1.2 * (0.25 + 0.75 * dl / max(1, avgdl)))
        source_id, position, sentence = candidate
        scored.append((-score, source_id, position, sentence))
    scored.sort()
    selected, source_counts = [], Counter()
    for neg_score, source_id, position, sentence in scored:
        if source_counts[source_id] >= per_source:
            continue
        selected.append((source_id, position, sentence, -neg_score))
        source_counts[source_id] += 1
        if len(selected) == top_k:
            break
    evidence = "\n".join(f"[{source_id}] {sentence}" for source_id, _, sentence, _ in selected)
    controller_input = f"Question: {question}\n\nEvidence Context:\n{evidence}"
    return controller_input, {
        "raw_observation_chars": len(observation),
        "compact_evidence_chars": len(evidence),
        "candidate_sentences": len(candidates),
        "retained_sentences": len(selected),
        "retained_sources": len({source_id for source_id, _, _, _ in selected}),
    }


def target(row, qmap):
    if row["decision"] == "STOP":
        return "SUFFICIENT: YES\nGAPS: NONE"
    teacher = qmap[row["state_id"]]["teacher"]
    return f"SUFFICIENT: NO\nGAPS:\n- {teacher['missing'].strip()}"


def example(row, compact_input, qmap, curriculum_round):
    return {
        "system": SYSTEM,
        "conversations": [
            {"from": "human", "value": compact_input},
            {"from": "gpt", "value": target(row, qmap)},
        ],
        "metadata": {
            "state_id": row["state_id"],
            "sample_id": row["sample_id"],
            "decision": row["decision"],
            "state_origin": row["state_origin"],
            "curriculum_round": curriculum_round,
        },
    }


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * q))]


def main():
    states = read(STATES)
    qmap = {row["state_id"]: row for row in read(QUERIES)}
    natural_train = [
        row for row in states
        if row["controller_split"] == "train" and row["state_origin"] == "natural_bocha_on_policy"
    ]
    natural_dev = sorted([
        row for row in states
        if row["controller_split"] == "dev" and row["state_origin"] == "natural_bocha_on_policy"
    ], key=stable)
    stops = sorted([row for row in natural_train if row["decision"] == "STOP"], key=stable)
    continues = sorted([
        row for row in natural_train
        if row["decision"] == "CONTINUE"
        and row["state_id"] in qmap
        and qmap[row["state_id"]]["teacher"].get("ok")
        and qmap[row["state_id"]]["teacher"].get("missing", "").strip()
    ], key=stable)
    assert len(stops) == 662 and len(continues) == 1400 and len(natural_dev) == 500
    selected_continues = continues[:1324]
    needed = {row["state_id"]: row for row in stops + selected_continues + natural_dev}
    compacted, diagnostics = {}, {}
    for state_id, row in needed.items():
        compacted[state_id], diagnostics[state_id] = compact(row)

    train = []
    for round_index in range(2):
        subset = selected_continues[round_index * 662 : (round_index + 1) * 662]
        train.extend(example(row, compacted[row["state_id"]], qmap, round_index + 1) for row in stops)
        train.extend(example(row, compacted[row["state_id"]], qmap, round_index + 1) for row in subset)
    dev = []
    for row in natural_dev:
        teacher = qmap.get(row["state_id"], {}).get("teacher", {})
        dev.append({
            "state_id": row["state_id"],
            "sample_id": row["sample_id"],
            "decision": row["decision"],
            "compact_input": compacted[row["state_id"]],
            "gap_target": teacher.get("missing") if row["decision"] == "CONTINUE" else "NONE",
            "compression": diagnostics[row["state_id"]],
        })

    train_ids = {row["metadata"]["sample_id"] for row in train}
    dev_ids = {row["sample_id"] for row in dev}
    counts = Counter(row["metadata"]["decision"] for row in train)
    all_diag = list(diagnostics.values())
    manifest = {
        "gate": "W7_S2G_DATA_GATE_PASS" if len(train) == 2648 and counts == {"STOP": 1324, "CONTINUE": 1324} and not (train_ids & dev_ids) else "W7_S2G_DATA_GATE_FAIL",
        "train_rows": len(train),
        "unique_stop_states": len(stops),
        "unique_continue_states": len(selected_continues),
        "decision_counts": dict(counts),
        "logical_rounds": 2,
        "physical_epochs": 1,
        "natural_only": True,
        "masked_siblings": 0,
        "frozen_dev_rows": len(dev),
        "train_dev_question_overlap": len(train_ids & dev_ids),
        "top_k_sentences": 8,
        "max_sentences_per_source": 2,
        "compressor_inputs": ["question", "current_observation"],
        "compressor_forbidden_inputs": ["gold_answer", "decision", "teacher_missing", "dev_outcome"],
        "raw_chars_p50": percentile([x["raw_observation_chars"] for x in all_diag], 0.50),
        "raw_chars_p95": percentile([x["raw_observation_chars"] for x in all_diag], 0.95),
        "compact_chars_p50": percentile([x["compact_evidence_chars"] for x in all_diag], 0.50),
        "compact_chars_p95": percentile([x["compact_evidence_chars"] for x in all_diag], 0.95),
        "api_calls": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "train.jsonl").open("w") as handle:
        for row in train:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT / "dev500.jsonl").open("w") as handle:
        for row in dev:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    dataset_info = {
        "w7_s2g_train": {
            "file_name": "train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        }
    }
    (OUT / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2) + "\n")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if manifest["gate"].endswith("FAIL"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
