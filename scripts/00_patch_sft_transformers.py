#!/usr/bin/env python3
"""Patch Transformers 5.6 FA2's optional attention-sink handling.

Transformers 5.6 calls ``s_aux.to(...)`` even for model families such as
Qwen3-MoE that do not provide attention sinks. Passing None through to the
lower-level helper is the intended optional-argument behavior.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


spec = importlib.util.find_spec("transformers")
if spec is None or spec.origin is None:
    raise SystemExit("ERROR: transformers is not installed in this Python environment")

target = Path(spec.origin).parent / "integrations/flash_attention.py"
text = target.read_text()
old = "s_aux=s_aux.to(query.dtype),  # FA only accepts half precision"
new = "s_aux=s_aux.to(query.dtype) if s_aux is not None else None,  # optional attention sink"

if new in text:
    print(f"SFT_TRANSFORMERS_FA_PATCH_ALREADY_APPLIED path={target}")
elif old in text:
    backup = target.with_suffix(".py.pre_qwen3_s_aux_fix")
    if not backup.exists():
        backup.write_text(text)
    target.write_text(text.replace(old, new, 1))
    print(f"SFT_TRANSFORMERS_FA_PATCH_APPLIED path={target} backup={backup}")
else:
    raise SystemExit(
        f"ERROR: expected Transformers 5.6 FA2 source pattern not found in {target}; "
        "refusing an unverified patch"
    )

verified = target.read_text()
assert new in verified
print("SFT_TRANSFORMERS_FA_PATCH_PASS")
