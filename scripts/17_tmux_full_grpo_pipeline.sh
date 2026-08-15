#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

session=q8_grpo_full
port=${TENSORBOARD_PORT:-6007}
log="$PROJECT_ROOT/logs/grpo_full_pipeline_$(date +%Y%m%d_%H%M%S).log"
tb_dir="$PROJECT_ROOT/tensorboard/qwen3_8b_evidence_800"

command -v tmux >/dev/null
if tmux has-session -t "$session" 2>/dev/null; then
  echo "ERROR tmux session already exists: $session" >&2
  exit 1
fi
curl -fsS "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null || {
  echo "ERROR Candidate-BM25 is not healthy on :$RETRIEVER_PORT" >&2
  exit 1
}
for gpu in 0 1 2 3; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v target="$gpu" '$1 + 0 == target {gsub(/ /, "", $2); print $2}')
  [[ -n "$used" ]] || { echo "ERROR cannot query GPU $gpu" >&2; exit 1; }
  (( used <= 1024 )) || {
    echo "ERROR GPU $gpu is not free: memory.used=${used}MiB" >&2
    exit 1
  }
done

mkdir -p "$tb_dir" "$(dirname "$log")"
tmux new-session -d -s "$session" -n pipeline \
  "cd '$PROJECT_ROOT'; set -o pipefail; bash scripts/16_run_full_grpo_pipeline.sh 2>&1 | tee '$log'; rc=\${PIPESTATUS[0]}; echo PIPELINE_EXIT=\$rc; exec bash"
tmux new-window -t "$session" -n log "touch '$log'; tail -F '$log'"
tmux new-window -t "$session" -n gpu "watch -n 2 nvidia-smi"
tmux new-window -t "$session" -n tensorboard \
  "env -u LD_LIBRARY_PATH '$VEXACT_ROOT/.venv/bin/python' -m tensorboard.main --logdir '$tb_dir' --host 0.0.0.0 --port '$port'; exec bash"
tmux select-window -t "$session:pipeline"

cat <<EOF
FULL_GRPO_TMUX_LAUNCH_PASS
session=$session
log=$log
tensorboard=http://127.0.0.1:$port
attach: tmux attach -t $session
windows: Ctrl-b 0(pipeline), 1(log), 2(gpu), 3(tensorboard)
EOF
