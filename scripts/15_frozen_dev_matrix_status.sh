#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

sessions=(q30_eval_base_direct q30_eval_base_rag q30_eval_sft_agent q30_eval_oracle)
tags=(base_direct base_rag sft_agent oracle_sft)

printf '%-24s %-12s %-10s %s\n' SESSION TAG TMUX RESULT
for i in "${!sessions[@]}"; do
  session=${sessions[$i]}
  tag=${tags[$i]}
  if tmux has-session -t "$session" 2>/dev/null; then state=alive; else state=absent; fi
  summary="$PROJECT_ROOT/results/frozen_dev/$tag/summary.json"
  if [[ -f "$summary" ]]; then result=complete; else result=pending; fi
  printf '%-24s %-12s %-10s %s\n' "$session" "$tag" "$state" "$result"
done

echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
  --format=csv,noheader,nounits | awk -F, '$1 + 0 < 4 {print "GPU" $1 ": memory=" $2 "MiB util=" $3 "%"}'

echo
"$VEXACT_ROOT/.venv/bin/python" - "$PROJECT_ROOT/results/frozen_dev" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for tag in ("base_direct", "base_rag", "sft_agent", "oracle_sft"):
    path = root / tag / "summary.json"
    if not path.is_file():
        continue
    data = json.loads(path.read_text())
    keys = (
        "n", "num_samples", "mean_em", "mean_f1", "finish_rate",
        "format_valid_rate", "retrieval_title_recall", "retrieval_hit_all_rate",
        "mean_search_calls", "mean_total_tokens", "runtime_seconds",
    )
    metrics = ", ".join(f"{key}={data[key]}" for key in keys if key in data)
    print(f"{tag}: {metrics}")
PY

echo
echo "Recent terminal markers:"
for session in "${sessions[@]}"; do
  if tmux has-session -t "$session" 2>/dev/null; then
    marker=$(tmux capture-pane -pt "$session:eval" -S -80 2>/dev/null \
      | grep -E 'FROZEN_DEV_PASS|EVAL_EXIT=|Traceback|ERROR|OutOfMemory' | tail -n 2 || true)
    [[ -z "$marker" ]] || printf '%s\n%s\n' "[$session]" "$marker"
  fi
done
