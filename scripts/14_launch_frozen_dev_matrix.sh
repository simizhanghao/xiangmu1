#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

base_model="/data1/hcc/deepresearch/Qwen3_30B/model"
sft_model="$PROJECT_ROOT/artifacts/models/qwen3_30b_sft_merged"

require_file "$base_model/config.json"
require_file "$sft_model/config.json"
require_file "$PROJECT_ROOT/results/frozen_dev/smoke_base_direct_n2/summary.json"
require_file "$PROJECT_ROOT/results/frozen_dev/smoke_base_rag_n2/summary.json"
require_file "$PROJECT_ROOT/results/frozen_dev/smoke_sft_agent_n2/summary.json"
command -v tmux >/dev/null
command -v nvidia-smi >/dev/null

sessions=(
  q30_eval_base_direct
  q30_eval_base_rag
  q30_eval_sft_agent
  q30_eval_oracle
)
tags=(base_direct base_rag sft_agent oracle_sft)
modes=(direct rag agent oracle)
models=("$base_model" "$base_model" "$sft_model" "$sft_model")
gpus=(0 1 2 3)

for session in "${sessions[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "ERROR tmux session already exists: $session" >&2
    exit 1
  fi
done

for tag in "${tags[@]}"; do
  if [[ -e "$PROJECT_ROOT/results/frozen_dev/$tag" ]]; then
    echo "ERROR result directory already exists: results/frozen_dev/$tag" >&2
    exit 1
  fi
done

for gpu in "${gpus[@]}"; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v target="$gpu" '$1 + 0 == target {gsub(/ /, "", $2); print $2}')
  [[ -n "$used" ]] || { echo "ERROR cannot query GPU $gpu" >&2; exit 1; }
  if (( used > 1024 )); then
    echo "ERROR GPU $gpu is not free: memory.used=${used}MiB" >&2
    exit 1
  fi
done

for i in "${!sessions[@]}"; do
  session=${sessions[$i]}
  tag=${tags[$i]}
  mode=${modes[$i]}
  model=${models[$i]}
  gpu=${gpus[$i]}

  tmux new-session -d -s "$session" -n eval \
    "cd '$PROJECT_ROOT'; EVAL_GPU='$gpu' bash scripts/08_eval_frozen_dev.sh '$tag' '$mode' '$model'; rc=\$?; echo EVAL_EXIT=\$rc; exec bash"
done

cat <<'EOF'
FROZEN_DEV_MATRIX_LAUNCH_PASS
GPU 0  q30_eval_base_direct  base_direct / direct
GPU 1  q30_eval_base_rag     base_rag / one-shot Candidate-BM25
GPU 2  q30_eval_sft_agent    sft_agent / multi-turn agent
GPU 3  q30_eval_oracle       oracle_sft / oracle-retrieval diagnostic

status: bash scripts/15_frozen_dev_matrix_status.sh
attach: tmux attach -t q30_eval_sft_agent
EOF
