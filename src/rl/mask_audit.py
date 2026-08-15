"""response_mask audit helpers (no veRL dependency — runnable on host)."""

from __future__ import annotations

import re
from typing import Any, Dict, List


_OBS_PATTERNS = [
    re.compile(r"<observation>", re.I),
    re.compile(r"</observation>", re.I),
    re.compile(r"\[hotpotqa_[^\]]+_ctx_\d+\]"),
]


def leak_patterns(text: str) -> List[str]:
    return [p.pattern for p in _OBS_PATTERNS if p.search(text or "")]


def dump_mask_audit(
    tokenizer,
    response_ids: List[int],
    response_mask: List[int],
) -> Dict[str, Any]:
    assert len(response_ids) == len(response_mask)
    kept = [tid for tid, m in zip(response_ids, response_mask) if int(m) == 1]
    dropped = [tid for tid, m in zip(response_ids, response_mask) if int(m) == 0]
    kept_text = tokenizer.decode(kept, skip_special_tokens=True)
    dropped_text = tokenizer.decode(dropped, skip_special_tokens=True)
    full_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    leaks = leak_patterns(kept_text)
    return {
        "num_response_tokens": len(response_ids),
        "num_mask1": len(kept),
        "num_mask0": len(dropped),
        "full_text": full_text,
        "mask1_text": kept_text,
        "mask0_text": dropped_text,
        "observation_leak_patterns_in_mask1": leaks,
        "observation_leak_in_mask1": bool(leaks),
        "pass": len(leaks) == 0,
        "has_observation_in_full": bool(re.search(r"<observation>", full_text or "", re.I)),
    }
