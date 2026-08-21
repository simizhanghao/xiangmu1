#!/usr/bin/env bash
set -euo pipefail
ROOT=/data1/hcc/deepresearch/Dee
SESSION=q8_w5_query_teacher
CONCURRENCY=${W5_QUERY_CONCURRENCY:-8}
cd "$ROOT"
[[ -n "${TEACHER_API_KEY:-}" ]] || { echo "W5_BLOCKED missing TEACHER_API_KEY"; exit 2; }
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "cd '$ROOT'; set -o pipefail; env -u LD_LIBRARY_PATH /data1/hcc/eca-verl-vexact/.venv/bin/python scripts/run_w5_query_teacher.py --output-dir results/75_w5_query_teacher/full --concurrency '$CONCURRENCY' 2>&1 | tee logs/w5_query_teacher_full.log; rc=\${PIPESTATUS[0]}; echo W5_QUERY_TEACHER_EXIT=\$rc; exec bash"
echo "W5_QUERY_TEACHER_STARTED session=$SESSION concurrency=$CONCURRENCY"
