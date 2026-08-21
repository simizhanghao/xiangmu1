#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/hcc/deepresearch/Dee
PY=/data1/hcc/eca-verl-vexact/.venv/bin/python
BASE="$ROOT/results/59_source_first_d2/smoke"
POOL="$BASE/synthesis/builder_pool.jsonl"
MINED="$BASE/depth_mining"
BUILT="$BASE/causal_builder"

: "${BOCHA_API_KEY:?BOCHA_API_KEY is required}"
: "${TEACHER_API_KEY:?TEACHER_API_KEY is required}"
: "${TEACHER_BASE_URL:=https://api.deepseek.com}"
: "${TEACHER_MODEL:=deepseek-v4-flash}"

cd "$ROOT"
mkdir -p "$BASE"

env -u LD_LIBRARY_PATH "$PY" scripts/synthesize_source_first_d2.py \
  --output-dir "$BASE/synthesis" \
  --target 12 \
  --max-seeds 60 \
  --seed 42 \
  --top-k 5 \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-api-key "$TEACHER_API_KEY"

env -u LD_LIBRARY_PATH "$PY" scripts/mine_web_depth_candidates.py \
  --pool "$POOL" \
  --output-dir "$MINED" \
  --max-samples 12 \
  --seed 42 \
  --top-k 5 \
  --provider bocha \
  --candidate-policy answer_absent

read -r d1 d2 < <(env -u LD_LIBRARY_PATH "$PY" -c '
import json,sys
c=json.load(open(sys.argv[1]))["likely_depth_counts"]
print(int(c["1"]), int(c["2"]))
' "$MINED/summary.json")
target=$((d1 + d2))
if [ "$target" -eq 0 ]; then
  echo "SOURCE_FIRST_NO_USABLE_CANDIDATES"
  exit 1
fi

set +e
env -u LD_LIBRARY_PATH "$PY" scripts/build_web_multiturn_v2.py \
  --pool "$POOL" \
  --pool-scope external_isolated \
  --candidate-manifest "$MINED/candidate_manifest.jsonl" \
  --output-dir "$BUILT" \
  --target "$target" \
  --max-candidates 12 \
  --quota-depth1 "$d1" \
  --quota-depth2 "$d2" \
  --quota-depth3 0 \
  --query-beam-size 1 \
  --web-provider bocha \
  --teacher-base-url "$TEACHER_BASE_URL" \
  --teacher-model "$TEACHER_MODEL" \
  --teacher-api-key "$TEACHER_API_KEY" \
  --teacher-temperature 0 \
  --teacher-seed 42 \
  --diagnostic-log "$BUILT/teacher_gold_blind_diagnostics.jsonl" \
  --seed 42
build_rc=$?
set -e

env -u LD_LIBRARY_PATH "$PY" scripts/audit_web_multiturn_v2.py \
  "$BUILT" --allow-incomplete
audit_rc=$?
echo "SOURCE_FIRST_BUILD_EXIT=$build_rc AUDIT_EXIT=$audit_rc"
