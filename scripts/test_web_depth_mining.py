#!/usr/bin/env python3
"""Offline unit contract for hidden minimal-depth candidate mining."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mine_web_depth_candidates import classify_candidate


def main() -> None:
    sample = {
        "gold_answers": ["London"],
        "supporting_facts": [
            {"title": "Film X"},
            {"title": "Ada Example"},
        ],
    }
    d1 = [{"title": "Ada Example", "text": "Ada Example was born in London."}]
    d2 = [{"title": "Film X", "text": "Film X was directed by Ada Example."}]
    unresolved = [{"title": "Other", "text": "No relevant information."}]
    assert classify_candidate(sample, d1) == (1, "answer_visible_after_search1")
    assert classify_candidate(sample, d2) == (2, "support_visible_answer_missing")
    assert classify_candidate(sample, unresolved) == (0, "unresolved_or_no_bridge")
    print("W3_DEPTH_MINING_CONTRACT_PASS")


if __name__ == "__main__":
    main()
