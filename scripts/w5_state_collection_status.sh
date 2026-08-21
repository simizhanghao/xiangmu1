#!/usr/bin/env bash
set -u
ROOT=/data1/hcc/deepresearch/Dee
total=0
for i in 0 1 2 3 4 5 6 7; do
  log="$ROOT/logs/w5_states_shard${i}.log"
  progress=$(grep -oE '\[[0-9]+/625\]' "$log" 2>/dev/null | tail -1 | tr -d '[]' || true)
  done_n=${progress%%/*}
  [[ "$done_n" =~ ^[0-9]+$ ]] || done_n=0
  total=$((total + done_n))
  summary=$(find "$ROOT/results/71_w5_controller/raw_shards/shard${i}" -type f -name summary.json 2>/dev/null | head -1)
  if [[ -n "$summary" ]] && grep -q '"num_samples": 625' "$summary"; then
    state=complete
  elif tmux has-session -t "q8_w5_states_${i}" 2>/dev/null; then
    state=running
  else
    state=stopped
  fi
  marker=$(grep -oE 'W5_SHARD_EXIT=[0-9]+' "$log" 2>/dev/null | tail -1 || true)
  printf 'shard=%s progress=%s/625 state=%s %s\n' "$i" "$done_n" "$state" "$marker"
done
printf 'total=%s/5000\n' "$total"
