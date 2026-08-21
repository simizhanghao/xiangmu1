#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/hcc/deepresearch/Dee
cd "$ROOT"
done_count=$(wc -l < results/73_w5_grounded_checker/full/labels.jsonl 2>/dev/null || echo 0)
state=stopped
tmux has-session -t q8_w5_grounded_checker 2>/dev/null && state=alive
echo "checker_progress=$done_count/6256 tmux=$state"
if [[ -f results/73_w5_grounded_checker/full/summary.json ]]; then
  cat results/73_w5_grounded_checker/full/summary.json
fi
