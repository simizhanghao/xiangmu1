# 用户执行命令

以下命令按阶段执行。Codex 不会代替用户启动下载、训练或评测。

## -1. 训练前存储审计与清理

先运行只读审计：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/00_storage_audit.sh
```

当前已确认以下六个目录是旧 Phase 18/20 已结束诊断的 checkpoint，不是 Qwen3 主线依赖，合计约 222GB。先关闭仍占用 Phase 18 日志的旧 TensorBoard，再精确删除：

```bash
pkill -f 'tensorboard.main --logdir /data1/hcc/deepresearch/results/18_boundary_exact_rollout/tensorboard' || true

sudo rm -rf -- \
  /data1/hcc/deepresearch/outputs/rl/07_ckpt_boundary_exact \
  /data1/hcc/deepresearch/outputs/rl/08_ckpt_rfpp_baseline \
  /data1/hcc/deepresearch/outputs/rl/09_ckpt_grpo_no_std \
  /data1/hcc/deepresearch/results/18_boundary_exact_rollout/checkpoints/step10_hf \
  /data1/hcc/deepresearch/results/20_rfpp_baseline/checkpoints/step10_hf \
  /data1/hcc/deepresearch/results/20_grpo_no_std/checkpoints/step10_hf

du -sh /data1/hcc/deepresearch
df -h /data1/hcc/deepresearch
```

这些目录删除后不能 resume，只能按保存的配置重新训练；正式的 SFT-v1、Answer-only@100、Evidence@400、所有总结/日志/rollout dump 都保留。

## 0. 提取冻结输入、建立实验契约并预检

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/00_stage_frozen_data.sh
/data1/hcc/eca-verl-vexact/.venv/bin/python scripts/00_build_experiment_contract.py
bash scripts/00_preflight.sh
```

第一条复制冻结 SFT、RL parquet、BM25 index 和 dev；第二条汇总所有历史 ID，并从原始 8k pool 冻结 500 条从未使用的 sealed Test，成功标志是 `P1_DATA_CONTRACT_PASS` 且 `forbidden_overlap_max=0`。现有 RL 对部分 SFT 问题的课程式复用会被如实记录，但 frozen dev/Test 对训练与历史数据的重叠必须为零。之后主线只使用本地副本。P0 必须看到 `P0_PREFLIGHT_PASS`。预检会硬性要求至少 450 GiB 可用空间，因为 30B base、SFT merged 与一套可 resume 的 optimizer state 都很大。

## 1. 下载模型并准备冻结 SFT 数据

先确保 VeXact 的所有锁定 extras 都已安装。这里必须包含 `vllm`，因为 Exact seeded sampling 使用其 Gumbel kernel：

```bash
cd /data1/hcc/eca-verl-vexact
UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
uv sync --frozen \
  --extra gpu \
  --extra vllm \
  --extra verl \
  --extra veomni

env -u LD_LIBRARY_PATH .venv/bin/python -c \
  'import vllm; from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample; print("vllm", vllm.__version__, "SEEDED_SAMPLER_PASS")'
```

不要用无版本约束的 `pip install vllm`，以免覆盖锁定的 Torch、CUDA 或 Transformers 组合。

直接在终端安装 ModelScope 并下载，不需要运行下载脚本：

```bash
python3 -m pip install --user -U modelscope
python3 -c "import modelscope; print(modelscope.__version__)"
mkdir -p /data1/hcc/deepresearch/Qwen3_30B/model
modelscope download Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --local-dir /data1/hcc/deepresearch/Qwen3_30B/model \
  --max-workers 8
```

下载结束后注册本地冻结 SFT 数据：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/02_prepare_sft_data.sh 2>&1 | tee logs/prepare_sft_data.log
```

模型使用前一节给出的 ModelScope 终端命令下载。下载中断后重复相同命令即可续传；模型固定保存在 `/data1/hcc/deepresearch/Qwen3_30B/model`。`01_download_model.sh` 只是可选封装，不是必须入口。

先做 Base 的 VeXact compatibility；该命令只占一张卡：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
COMPAT_GPU=0 bash scripts/05_check_vexact_model.sh \
  /data1/hcc/deepresearch/Qwen3_30B/model qwen3_base
```

只有出现 `VEXACT_MODEL_COMPAT_PASS` 才进入 SFT。

Qwen3-MoE 的 Exact gate 固定使用 `triton-invariant` attention 与 `fused_triton` MoE；rollout 和 VeOmni actor 两侧必须使用同一套算术，且门限为 bitwise `rtol=0, atol=0`，不得通过放宽容差绕过。

## 2. 4 卡 SFT（GPU 0–3）

第一次运行前，创建隔离的 LlamaFactory 训练环境。不要用 base 环境，也不要复用 VeXact `.venv`：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/00_setup_sft_env.sh 2>&1 | tee logs/setup_sft_env.log
```

必须看到 `SFT_ENV_PASS` 后才能启动训练。

Transformers 5.6 对 Qwen3-MoE 的 FA2 可选 `s_aux` 存在空值处理错误。环境安装脚本会自动应用受控补丁；若环境已经建好，可单独执行：

```bash
/data1/hcc/LlamaFactory/.venv/bin/python scripts/00_patch_sft_transformers.py
```

SFT 使用每卡 micro batch 4、梯度累积4、4卡，有效全局 batch仍为64。该配置是在 micro 2 实测峰值约28GB/卡后提升；不得继续直接升到 micro 8，因为4096-token尾部 batch 的峰值仍可能显著高于前几步。

使用统一 tmux 启动器；它自动创建 train/log/GPU/TensorBoard 四个窗口：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/13_tmux_run.sh sft
tmux attach -t q30_sft
```

查看日志/GPU而不干扰训练：

```bash
tail -F /data1/hcc/deepresearch/Qwen3_30B/xiangmu/logs/sft_*.log
watch -n 1 nvidia-smi
```

TensorBoard 默认监听 `0.0.0.0:6006`，浏览器访问 `http://服务器IP:6006`；若端口不能直连，可在本机用 SSH 端口转发后访问 `http://127.0.0.1:6006`。

训练成功后合并；merge 使用 CPU，不要在已有目标上覆盖：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/04_merge_sft.sh
COMPAT_GPU=0 bash scripts/05_check_vexact_model.sh \
  artifacts/models/qwen3_30b_sft_merged qwen3_sft
```

## 3. 冻结 dev 的四种 controlled 协议

每次评测占一张80G GPU。单卡时顺序执行；GPU 0–3 都空闲时可在 smoke 门通过后并行执行。mode 不能省略，否则会混淆 Direct、one-shot RAG 和多轮 Agent：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh base_direct direct \
  /data1/hcc/deepresearch/Qwen3_30B/model
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh base_rag rag \
  /data1/hcc/deepresearch/Qwen3_30B/model
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh sft_agent agent \
  artifacts/models/qwen3_30b_sft_merged
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh oracle_sft oracle \
  artifacts/models/qwen3_30b_sft_merged
```

其中 Oracle 只做检索瓶颈诊断，不属于四个可部署主 baseline。

并行正式评测前，补跑尚缺的 SFT Agent 两题 smoke：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
EVAL_MAX_SAMPLES=2 EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh \
  smoke_sft_agent_n2 agent \
  /data1/hcc/deepresearch/Qwen3_30B/xiangmu/artifacts/models/qwen3_30b_sft_merged
```

smoke 成功后，一次启动四个独立 tmux（GPU 0/1/2/3）：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/14_launch_frozen_dev_matrix.sh
bash scripts/15_frozen_dev_matrix_status.sh
```

需要看最慢的多轮 Agent：

```bash
tmux attach -t q30_eval_sft_agent
```

## 4. 启动 Candidate-BM25

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
tmux new-session -d -s q30_retriever \
  "cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu && bash scripts/06_start_retriever.sh 2>&1 | tee logs/retriever.log"
curl -s http://127.0.0.1:8001/health
```

## 5. 4 卡 Exact RL smoke（GPU 0–3）

先生成 VeRL `datasets==2.21` 兼容副本。该步骤只移除由
`datasets==4.x` 写入的 Hugging Face schema metadata，并通过逐行内容 hash
证明训练样本未改变：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
env -u LD_LIBRARY_PATH /data1/hcc/eca-verl-vexact/.venv/bin/python \
  scripts/02_prepare_rl_compat.py
bash scripts/00_preflight.sh
```

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/13_tmux_run.sh rl-smoke
tmux attach -t q30_rl_smoke
```

必须看到 `GRPO_SEGMENT_PASS step=1`，并确认没有 OOM、NaN、Agent loop error、reward 全零或格式崩溃，才允许正式训练。

若希望无人值守地继续完整正式流程，使用硬门流水线。它仍会先跑 smoke，且在
200/400/600/800 每一段后执行 frozen-dev、健康门、选模和 best 保护；任何失败
立即停止，sealed Test 始终不会被打开：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
TENSORBOARD_PORT=6007 bash scripts/17_tmux_full_grpo_pipeline.sh
tmux attach -t q30_grpo_full
```

## 6. 200/400/600/800 分段训练与评测

每段严格执行同一循环。以下先以 step 200 为例：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/13_tmux_run.sh rl-segment 200
tmux attach -t q30_grpo_200

EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step200 agent \
  artifacts/evidence_grpo_ckpt/global_step_200/actor/huggingface
python3 scripts/09_select_best.py --allow-partial
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh
python3 scripts/12_build_result_table.py
```

确认当前 checkpoint 过 health gate 且 `artifacts/best_hf/FROZEN_GRPO_STEP` 已写好，才继续下一段：

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu

bash scripts/13_tmux_run.sh rl-segment 400
tmux attach -t q30_grpo_400
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step400 agent \
  artifacts/evidence_grpo_ckpt/global_step_400/actor/huggingface
python3 scripts/09_select_best.py --allow-partial
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh

bash scripts/13_tmux_run.sh rl-segment 600
tmux attach -t q30_grpo_600
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step600 agent \
  artifacts/evidence_grpo_ckpt/global_step_600/actor/huggingface
python3 scripts/09_select_best.py --allow-partial
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh

bash scripts/13_tmux_run.sh rl-segment 800
tmux attach -t q30_grpo_800
EVAL_GPU=0 bash scripts/08_eval_frozen_dev.sh step800 agent \
  artifacts/evidence_grpo_ckpt/global_step_800/actor/huggingface
python3 scripts/09_select_best.py
ALLOW_BEST_REPLACE=1 bash scripts/10_promote_best.sh
python3 scripts/12_build_result_table.py
```

## 7. 状态检查

```bash
cd /data1/hcc/deepresearch/Qwen3_30B/xiangmu
bash scripts/11_status.sh
cat results/checkpoint_selection.json
cat results/FROZEN_DEV_TABLE.md
```

到这里先停。不得自行打开 sealed Test；先由我们审计 selection、hash、完整 dev 表和训练曲线，再给 final Test 命令。
