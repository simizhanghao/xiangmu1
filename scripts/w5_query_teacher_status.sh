#!/usr/bin/env bash
set -euo pipefail
cd /data1/hcc/deepresearch/Dee
n=$(wc -l < results/75_w5_query_teacher/full/queries.jsonl 2>/dev/null || echo 0)
state=stopped; tmux has-session -t q8_w5_query_teacher 2>/dev/null && state=alive
echo "query_progress=$n/2229 tmux=$state"
[[ ! -f results/75_w5_query_teacher/full/summary.json ]] || cat results/75_w5_query_teacher/full/summary.json
