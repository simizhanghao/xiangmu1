#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

venv="$LLAMAFACTORY_ROOT/.venv"
python="$venv/bin/python"
torch_index=https://download.pytorch.org/whl/cu129
flash_attn_wheel=https://github.com/Luosuu/flash-attention3-wheels/releases/download/v0.0.2/flash_attn-2.8.4+cu129torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

command -v uv >/dev/null
cd "$LLAMAFACTORY_ROOT"

if [[ ! -x "$python" ]]; then
  uv venv --python 3.12 "$venv"
fi

# Keep SFT isolated from VeXact: same proven Torch/CUDA ABI, but let the
# LlamaFactory pyproject resolve its own Transformers/PEFT/TRL constraints.
UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10} \
  uv pip install --python "$python" \
    --index-url "$torch_index" \
    'torch==2.10.0+cu129' \
    'torchvision==0.25.0+cu129' \
    'torchaudio==2.10.0+cu129'

UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10} \
  uv pip install --python "$python" -e . tensorboard ninja packaging setuptools wheel

# ZeRO-3 does not require prebuilding optional fused ops. Disabling them makes
# installation deterministic and avoids compiling against the host CUDA SDK.
DS_BUILD_OPS=0 UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10} \
  uv pip install --python "$python" --no-build-isolation 'deepspeed==0.18.4'

UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600} UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-10} \
  uv pip install --python "$python" --no-deps "$flash_attn_wheel"

"$python" "$PROJECT_ROOT/scripts/00_patch_sft_transformers.py"

env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 "$python" - <<'PY'
import json
from pathlib import Path

import accelerate
import datasets
import deepspeed
import flash_attn
import peft
import tensorboard
import torch
import transformers
import trl

versions = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "peft": peft.__version__,
    "trl": trl.__version__,
    "deepspeed": deepspeed.__version__,
    "flash_attn": flash_attn.__version__,
    "tensorboard": tensorboard.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
assert versions["torch"] == "2.10.0+cu129", versions
assert versions["cuda_available"], versions
out = Path("/data1/hcc/deepresearch/Dee/results/sft_env_versions.json")
out.write_text(json.dumps(versions, indent=2) + "\n")
print(json.dumps(versions, indent=2))
print("SFT_ENV_PASS")
PY
