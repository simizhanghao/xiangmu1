#!/usr/bin/env bash
set -euo pipefail
PROJECT=/data1/hcc/deepresearch/Dee

declare -A gpu=( [direct]=1 [rag]=2 [sft]=3 )
for arm in direct rag sft; do
  session="q8_heldout_${arm}"
  tmux has-session -t "$session" 2>/dev/null && {
    echo "ERROR session exists: $session" >&2; exit 1;
  }
  used=$(nvidia-smi -i "${gpu[$arm]}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  (( used <= 1024 )) || { echo "ERROR GPU ${gpu[$arm]} busy: ${used}MiB" >&2; exit 1; }
done

for arm in direct rag sft; do
  session="q8_heldout_${arm}"
  log="$PROJECT/logs/heldout_${arm}_$(date +%Y%m%d_%H%M%S).log"
  tmux new-session -d -s "$session" \
    "cd '$PROJECT'; set -o pipefail; bash scripts/run_heldout_closure_arm.sh '$arm' '${gpu[$arm]}' 2>&1 | tee '$log'; rc=\${PIPESTATUS[0]}; echo ARM_EXIT=\$rc; exec bash"
  echo "LAUNCHED arm=$arm gpu=${gpu[$arm]} session=$session log=$log"
done

echo "status: tmux ls | grep q8_heldout"
echo "finish: python3 scripts/summarize_heldout_four_arm.py"
