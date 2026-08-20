#!/usr/bin/env python3
"""Offline contract for combined <internal> + executable action in Web-v2."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.react_loop import _extract_closed_tags, _first_action, protocol_stop_strings
from src.agents.research_memory import ResearchMemory, serialize_research_memory


def main() -> None:
    chunk = (
        "<internal>\nKnown:\n- [S1] Movie X was directed by Ada.\n"
        "Missing:\n- Ada's birthplace\nDecision: SEARCH\n"
        "Next Query: Ada birthplace\n</internal>\n"
        "<search>\nAda birthplace\n</search>"
    )
    tags = _extract_closed_tags(chunk)
    assert _first_action(tags) == "search"
    assert "</internal>" not in protocol_stop_strings("research_v2")
    assert "</search>" in protocol_stop_strings("research_v2")
    memory = ResearchMemory("Where was the director of Movie X born?", max_searches=3)
    memory.update_from_internal(tags["internal"][-1])
    memory.add_search(
        tags["search"][-1],
        [{"title": "Ada", "text": "Ada was born in London.", "score": 1.0, "metadata": {"url": "https://example.org/ada"}}],
    )
    assert memory.last_decision == "SEARCH"
    assert memory.last_next_query == "Ada birthplace"
    assert memory.render() == serialize_research_memory(memory)
    print("WEB_MULTITURN_ACTION_PROTOCOL_PASS")


if __name__ == "__main__":
    main()
