#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

lf_data="$LLAMAFACTORY_ROOT/data"
src="$SFT_DATA_DIR"
require_file "$src/eca_coldstart_v1_train.jsonl"
require_file "$src/eca_coldstart_v1_dev.jsonl"
require_file "$src/eca_coldstart_v1_smoke.jsonl"
require_file "$lf_data/dataset_info.json"

ln -sfn "$src/eca_coldstart_v1_train.jsonl" "$lf_data/eca_qwen3_8b_coldstart_train.jsonl"
ln -sfn "$src/eca_coldstart_v1_dev.jsonl" "$lf_data/eca_qwen3_8b_coldstart_dev.jsonl"
ln -sfn "$src/eca_coldstart_v1_smoke.jsonl" "$lf_data/eca_qwen3_8b_coldstart_smoke.jsonl"

"$PYTHON_BIN" - "$lf_data/dataset_info.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
obj = json.loads(p.read_text())
for split in ("train", "dev", "smoke"):
    obj[f"eca_qwen3_8b_coldstart_{split}"] = {
        "file_name": f"eca_qwen3_8b_coldstart_{split}.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
    }
p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
print("registered", ",".join(f"eca_qwen3_8b_coldstart_{s}" for s in ("train", "dev", "smoke")))
PY

"$PYTHON_BIN" - "$src" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for split in ("train", "dev", "smoke"):
    p = root / f"eca_coldstart_v1_{split}.jsonl"
    n = obs = bad = 0
    for line in p.open():
        row = json.loads(line); n += 1
        for msg in row["conversations"]:
            if msg["from"] == "observation": obs += 1
            if msg["from"] == "gpt" and "<observation>" in msg["value"].lower(): bad += 1
    assert bad == 0, (split, bad)
    print(split, "rows=", n, "observation_roles=", obs, "leaks=", bad)
print("SFT_DATA_PROTOCOL_PASS")
PY
