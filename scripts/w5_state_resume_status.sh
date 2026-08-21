#!/usr/bin/env bash
set -u
ROOT=/data1/hcc/deepresearch/Dee
total=0
target=$(wc -l < "$ROOT/data/w5_controller/controller_pending_resume.jsonl" 2>/dev/null || echo 0)
for i in 0 1; do
  log="$ROOT/logs/w5_resume_shard${i}.log"
  progress=$(grep -oE '\[[0-9]+/[0-9]+\]' "$log" 2>/dev/null | tail -1 | tr -d '[]' || true)
  done_n=${progress%%/*}; shard_n=${progress##*/}
  [[ "$done_n" =~ ^[0-9]+$ ]] || done_n=0
  [[ "$shard_n" =~ ^[0-9]+$ ]] || shard_n=0
  total=$((total + done_n))
  summary=$(find "$ROOT/results/71_w5_controller/resume_shards/shard${i}" -type f -name summary.json 2>/dev/null | head -1)
  if [[ -n "$summary" ]]; then state=complete
  elif tmux has-session -t "q8_w5_resume_${i}" 2>/dev/null; then state=running
  else state=stopped
  fi
  printf 'shard=%s progress=%s/%s state=%s\n' "$i" "$done_n" "$shard_n" "$state"
done
printf 'resume_total=%s/%s\n' "$total" "$target"
