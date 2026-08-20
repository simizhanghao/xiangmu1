#!/usr/bin/env python3
"""Offline contract test for bounded Web ResearchMemory."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.research_memory import ResearchMemory


def main() -> None:
    memory = ResearchMemory("Who founded the company and where was the founder born?", max_searches=5)
    first = memory.add_search(
        "company founder",
        [{"title": "Company", "text": "The company was founded by Ada Example.", "score": 3, "metadata": {"url": "https://example.com/company?x=1"}}],
    )
    second = memory.add_search(
        "Ada Example birthplace",
        [{"title": "Ada", "text": "Ada Example was born in Example City.", "score": 4, "metadata": {"url": "https://example.org/ada"}}],
    )
    assert first == {"new_urls": 1, "new_evidence": 1}
    assert second == {"new_urls": 1, "new_evidence": 1}
    assert memory.remaining_searches == 3
    rendered = memory.render()
    assert "Known evidence:" in rendered and "Previous searches:" in rendered
    assert "E1" in rendered and "E2" in rendered and "Remaining search budget: 3" in rendered
    assert len(rendered) <= memory.char_budget
    print("RESEARCH_MEMORY_CONTRACT_PASS", memory.summary())


if __name__ == "__main__":
    main()
