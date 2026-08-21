#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/eca-verl-vexact/.venv/bin/python
cd "$ROOT"
env -u LD_LIBRARY_PATH "$PY" scripts/merge_w5_state_shards.py
env -u LD_LIBRARY_PATH "$PY" scripts/build_w5_state_candidates.py \
  --trace results/72_w5_state_dataset/raw_states_5000/trace.jsonl \
  --questions data/w5_controller/controller_all5000.jsonl \
  --output-dir results/72_w5_state_dataset/checker_candidates
echo W5_STATE_COLLECTION_FINALIZE_PASS
