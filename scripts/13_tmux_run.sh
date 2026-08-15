#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

stage=${1:-}
arg=${2:-}
port=${TENSORBOARD_PORT:-6006}
stamp=$(date +%Y%m%d_%H%M%S)

case "$stage" in
  sft)
    session=q8_sft
    log="$PROJECT_ROOT/logs/sft_${stamp}.log"
    tb_dir="$SFT_ADAPTER"
    train="RUN_LOG='$log' bash '$PROJECT_ROOT/scripts/03_train_sft.sh'"
    ;;
  rl-smoke)
    session=q8_rl_smoke
    log="$PROJECT_ROOT/logs/grpo_smoke_to1_${stamp}.log"
    tb_dir="$PROJECT_ROOT/tensorboard/qwen3_8b_evidence_smoke"
    train="RUN_LOG='$log' TENSORBOARD_DIR='$tb_dir' bash '$PROJECT_ROOT/scripts/07_run_evidence_grpo.sh' smoke"
    ;;
  rl-segment)
    case "$arg" in 200|400|600|800) ;; *)
      echo "usage: $0 sft | rl-smoke | rl-segment {200|400|600|800}" >&2
      exit 2
    esac
    session="q8_grpo_${arg}"
    log="$PROJECT_ROOT/logs/grpo_segment_to${arg}_${stamp}.log"
    tb_dir="$PROJECT_ROOT/tensorboard/qwen3_8b_evidence_800"
    train="RUN_LOG='$log' TENSORBOARD_DIR='$tb_dir' bash '$PROJECT_ROOT/scripts/07_run_evidence_grpo.sh' segment '$arg'"
    ;;
  *)
    echo "usage: $0 sft | rl-smoke | rl-segment {200|400|600|800}" >&2
    exit 2
    ;;
esac

command -v tmux >/dev/null
if tmux has-session -t "$session" 2>/dev/null; then
  echo "ERROR tmux session already exists: $session" >&2
  exit 1
fi
mkdir -p "$tb_dir" "$(dirname "$log")"

tmux new-session -d -s "$session" -n train \
  "cd '$PROJECT_ROOT'; set -o pipefail; $train; rc=\$?; echo TRAIN_EXIT=\$rc; exec bash"
tmux new-window -t "$session" -n log \
  "touch '$log'; tail -F '$log'"
tmux new-window -t "$session" -n gpu \
  "watch -n 2 nvidia-smi"
tmux new-window -t "$session" -n tensorboard \
  "env -u LD_LIBRARY_PATH '$VEXACT_ROOT/.venv/bin/python' -m tensorboard.main --logdir '$tb_dir' --host 0.0.0.0 --port '$port'; exec bash"
tmux select-window -t "$session:train"

cat <<EOF
TMUX_LAUNCH_PASS
session=$session
log=$log
tensorboard_logdir=$tb_dir
tensorboard_url=http://127.0.0.1:$port
attach: tmux attach -t $session
windows: Ctrl-b 0(train), 1(log), 2(gpu), 3(tensorboard)
EOF
