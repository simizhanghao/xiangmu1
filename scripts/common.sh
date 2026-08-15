#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
# shellcheck source=../config/project.env
source "$PROJECT_ROOT/config/project.env"
PYTHON_BIN=${PYTHON_BIN:-$VEXACT_ROOT/.venv/bin/python}

mkdir -p "$PROJECT_ROOT/artifacts" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "ERROR missing directory: $1" >&2; exit 1; }
}

run_llamafactory() {
  [[ -x "$LLAMAFACTORY_PYTHON" ]] || {
    echo "ERROR missing LlamaFactory environment: $LLAMAFACTORY_PYTHON" >&2
    echo "Run: bash $PROJECT_ROOT/scripts/00_setup_sft_env.sh" >&2
    return 1
  }
  env -u LD_LIBRARY_PATH \
    PATH="$LLAMAFACTORY_ROOT/.venv/bin:$PATH" \
    PYTHONPATH="$LLAMAFACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$LLAMAFACTORY_PYTHON" -m llamafactory.cli "$@"
}
