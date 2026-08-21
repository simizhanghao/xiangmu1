# Final Deployment

The shipped system is the frozen **GRPO@400 Evidence-Aware Web Search Agent**. It uses
Bocha and the existing Harness v1 AgentLoop, and emits Search, Evidence, Answer, and
Sources. Query/source provenance is recorded outside the prompt. ResearchMemory prompt
injection and the failed adaptive Controller line are not loaded.

## Offline demo and release check

These commands use no model, GPU, Web request, or API key:

```bash
python3 cli.py --mock
python3 cli.py --mock --question "demo question" --json
bash scripts/smoke_final.sh
```

## Start locally

```bash
cd /data1/hcc/deepresearch/Dee
export BOCHA_API_KEY='your-bocha-key'
export DEE_ASSISTANT_API_KEY='your-deepseek-compatible-key'
bash scripts/start_final_stack.sh
```

The script reuses a healthy vLLM server on port 18120 or starts the frozen GRPO@400
model on `GPU=1`. FastAPI listens on `127.0.0.1:8010`; Swagger is at
`http://127.0.0.1:8010/docs`.

Run in tmux:

```bash
tmux new-session -d -s dee_final \
  "cd /data1/hcc/deepresearch/Dee && bash scripts/start_final_stack.sh 2>&1 | tee logs/final_api.log"
```

## Interactive CLI

```bash
python3 cli.py
```

Default `hybrid` mode uses DeepSeek for query planning/final synthesis and Bocha for
1–3 searches. Use `/multi` to paste a multi-line request and submit with a blank line.
The scientific frozen policy remains available as `python3 cli.py --mode frozen`.

Or ask once:

```bash
python3 cli.py --question "Who directed Troll 2 and where was he born?"
```

## HTTP API

```bash
curl -s http://127.0.0.1:8010/v1/research \
  -H 'Content-Type: application/json' \
  -d '{"question":"Who directed Troll 2 and where was he born?"}'
```

Endpoints:

- `GET /health`: vLLM/provider readiness and frozen policy identity.
- `GET /v1/metrics`: frozen scientific results.
- `POST /v1/research`: one grounded Web research request.

To expose it to other machines, set a service key and bind explicitly:

```bash
export DEE_API_KEY='choose-a-long-random-secret'
API_HOST=0.0.0.0 bash scripts/start_final_stack.sh
```

Clients then send `Authorization: Bearer $DEE_API_KEY`. The launcher refuses a public
bind without this key. Network firewall/reverse-proxy TLS remains the operator's
responsibility; never commit Bocha or service keys.

## Configuration contract

- `config/final.yaml`: human-readable frozen serving contract.
- `config/model_manifest.json`: model/tokenizer/Harness hashes; weights stay outside Git.
- `config/data_manifest.json`: frozen split locations; datasets stay outside Git.
- `.env.example`: variable names only, never credentials.
- `config/final_metrics.json`: immutable public metrics returned by `/v1/metrics`.

The vLLM image is intentionally external to the API package because the validated
rollout environment is GPU/CUDA-specific. `requirements-runtime.txt` covers only the
thin API/CLI layer; use the validated VeXact environment for model serving.

Build the Python wheel without copying model weights or datasets:

```bash
uv build --wheel
# output: dist/evidence_aware_deepresearch-1.0.0-py3-none-any.whl
```

The wheel packages the CLI, API and Agent protocol code. The model path, Web key and
GPU runtime remain explicit deployment dependencies described above.
