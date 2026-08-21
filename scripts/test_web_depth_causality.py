#!/usr/bin/env python3
"""Offline contract: mined Search1 is reused and D2 has positive forced1 delta."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import build_web_multiturn_v2 as builder


class FakeWeb:
    def retrieve(self, sample, query, top_k):
        if query == sample["question"]:
            text = "Film X was directed by Ada Example."
            doc_id = "web_film"
        elif "occupation" in query:
            text = "Ada Example worked as an artist."
            doc_id = "web_ada_job"
        else:
            text = "Ada Example was born in London."
            doc_id = "web_ada"
        return {
            "documents": [{
                "document_id": doc_id,
                "title": doc_id,
                "text": text,
                "score": 1.0,
                "metadata": {"url": f"https://example.org/{doc_id}"},
            }],
            "errors": [],
        }


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def fake_teacher(cfg, payload):
    instruction = str(payload.get("instruction") or "")
    observation = str(payload.get("latest_observation") or "")
    if "must ANSWER now" in instruction:
        return {"known": [], "missing": [], "decision": "ANSWER", "next_query": "", "answer": "unknown", "source_ids": []}
    if "London" in observation:
        return {"known": ["Ada was born in London [S2]"], "missing": [], "decision": "ANSWER", "next_query": "", "answer": "London", "source_ids": ["S2"]}
    result = {"known": ["Film X was directed by Ada Example [S1]"], "missing": ["Ada's birthplace"], "decision": "SEARCH", "next_query": "Ada Example birthplace", "answer": "", "source_ids": ["S1"]}
    if int(payload.get("query_beam_size") or 1) > 1:
        result["next_query"] = "Ada Example occupation"
        result["next_queries"] = [
            "Ada Example occupation",
            "Ada Example birthplace",
            "Ada Example biography born city",
        ]
    return result


def main() -> None:
    builder.teacher_call = fake_teacher
    sample = {
        "sample_id": "synthetic_d2",
        "question": "Where was the director of Film X born?",
        "gold_answers": ["London"],
    }
    row, reason = builder.build_one(
        sample,
        2,
        SimpleNamespace(top_k=5, memory_tokenizer=FakeTokenizer(), query_beam_size=1),
        FakeWeb(),
        initial_query=sample["question"],
        initial_documents=FakeWeb().retrieve(sample, sample["question"], 5)["documents"],
    )
    assert reason == "accepted" and row is not None
    assert row["turns"][0]["query"] == sample["question"]
    assert row["turns"][1]["query"] == "Ada Example birthplace"
    causal = row["minimal_depth_audit"]
    assert causal["forced1_f1"] == 0.0
    assert causal["forced1_sufficient"] is False
    assert causal["delta_f1_search2"] == 1.0
    assert causal["search2_useful"] is True
    assert row["turns"][1]["query_conditioning"] == "observation_entity"
    assert builder.query_conditioning(
        "Peter Szewczyk nationality",
        "Are Tommy Lee Jones and Peter Szewczyk of the same nationality?",
        "Tommy Lee Jones is American; Peter Szewczyk is not described.",
        ["Peter Szewczyk's nationality"],
    ) == "missing_state_refinement"
    assert builder.acceptance_f1("3rd", ["the third"]) == 1.0

    beam_row, beam_reason = builder.build_one(
        sample,
        2,
        SimpleNamespace(
            top_k=5,
            memory_tokenizer=FakeTokenizer(),
            query_beam_size=3,
            query_beam_diagnostic_log=None,
        ),
        FakeWeb(),
        initial_query=sample["question"],
        initial_documents=FakeWeb().retrieve(sample, sample["question"], 5)["documents"],
    )
    assert beam_reason == "accepted" and beam_row is not None
    assert beam_row["query_beam_audit"]["selected_index"] == 1
    assert beam_row["query_beam_audit"]["selected_query"] == "Ada Example birthplace"
    assert beam_row["query_beam_audit"]["attempts"][0]["reason"] == "teacher_over_depth"
    print("W3_MINIMAL_DEPTH_CAUSALITY_PASS")


if __name__ == "__main__":
    main()
