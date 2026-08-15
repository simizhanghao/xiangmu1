#!/usr/bin/env python3
"""Minimal VeRL entrypoint for the final project Evidence-GRPO run.

All Hydra overrides are forwarded unchanged. Research-only trainer monkeypatches
from the parent repository are intentionally not loaded here.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.argv = ["verl.trainer.main_ppo", *sys.argv[1:]]
runpy.run_module("verl.trainer.main_ppo", run_name="__main__")

