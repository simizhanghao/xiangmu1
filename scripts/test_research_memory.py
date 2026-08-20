#!/usr/bin/env python3
"""Offline contract test for bounded Web ResearchMemory."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.research_memory import ResearchMemory, serialize_research_memory


def main() -> None:
    memory = ResearchMemory("Who founded the company and where was the founder born?", max_searches=5)
    first = memory.add_search(
        "company founder",
        [{"document_id": "web_company", "title": "Company", "text": "The company was founded by Ada Example.", "score": 3, "metadata": {"url": "https://example.com/company?x=1"}}],
    )
    second = memory.add_search(
        "Ada Example birthplace",
        [{"document_id": "web_ada", "title": "Ada", "text": "Ada Example was born in Example City.", "score": 4, "metadata": {"url": "https://example.org/ada"}}],
    )
    assert first == {"new_urls": 1, "new_evidence": 1}
    assert second == {"new_urls": 1, "new_evidence": 1}
    assert memory.remaining_searches == 3
    memory.update_from_internal(
        "Known:\n- Ada founded the company.\n"
        "Missing:\n- Ada Example's birthplace\n"
        "Decision: SEARCH\nNext Query: Ada Example birthplace"
    )
    rendered = memory.render()
    assert rendered == serialize_research_memory(memory)
    assert "Known:" in rendered and "Previous Queries:" in rendered and "Sources:" in rendered
    assert "[S1]" in rendered and "[S2]" in rendered and "Remaining Budget: 3" in rendered
    assert "Ada Example's birthplace" in rendered
    assert memory.last_decision == "SEARCH"
    assert memory.last_next_query == "Ada Example birthplace"
    assert len(rendered) <= memory.char_budget
    print("WEB_MEMORY_PROTOCOL_PARITY_PASS", memory.summary())


if __name__ == "__main__":
    main()
