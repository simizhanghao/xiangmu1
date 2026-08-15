# Evidence-Aware Deep Research Agent — Qwen3-30B-A3B

## 项目目标

在不更换数据、retriever、reward 与 Agent protocol 的前提下，把已验证的 Qwen2.5-3B 系统扩展到 `Qwen3-30B-A3B-Instruct-2507`，交付一个强、可运行、可复现的最终 DeepResearch Agent。

主链只有一条：

```text
Qwen3-30B-A3B-Instruct-2507
  → LoRA SFT（原 coldstart_v1）
  → Merge BF16 HF model
  → Evidence GRPO（Exact VeXact + Candidate-BM25）
  → frozen dev @ 200/400/600/800
  → 唯一 best checkpoint
```

## 已冻结的实验定义

- Backbone：`Qwen/Qwen3-30B-A3B-Instruct-2507`，non-thinking template。
- SFT：原 ShareGPT coldstart_v1，2 epochs，LoRA rank 32，effective global batch 64。
- RL：Evidence GRPO，`lambda_e=0.5`、LR `1e-6`、`temperature=0.9`、`top_p=0.95`、`n=4`。
- Environment：Candidate-BM25 top-5、同一 `EcaSearchAgentLoop`、Exact VeXact。
- Budget：最多 800 steps；只在 200/400/600/800 评 frozen dev。
- 选模：先过 finish/format/observation-mask health gate，再按 Answer F1、Evidence F1、EM、少重复 query、较早 checkpoint 的固定顺序排序。
- Test：在唯一 best 冻结前禁止打开 sealed HotpotQA Test。

## 目录

| 路径 | 内容 |
|---|---|
| `config/project.env` | 所有固定路径和运行参数 |
| `config/sft_*.yaml` | 30B SFT 与 merge 配置 |
| `config/rl/` | 本地 Candidate-BM25 tool 与 AgentLoop 注册配置 |
| `src/` | 本地 AgentLoop、retriever、Evidence reward、协议与评测实现 |
| `data/` | 由 staging 命令生成的冻结输入快照（git ignored） |
| `scripts/00`–`05` | 预检、下载、数据、SFT、VeXact compatibility |
| `scripts/06`–`08` | retriever、Evidence-GRPO、frozen-dev |
| `scripts/09`–`12` | 选模、保护 best、状态与结果表 |
| `../model/` | 下载的 Qwen3-30B-A3B Base（git ignored） |
| `artifacts/` | SFT merged、RL checkpoint 与冻结 best（git ignored） |
| `results/` | hash、dev summary、选模记录与结果表 |
| `logs/` | 训练与评测日志 |

## 从零执行流程

```text
冻结输入 staging
  → 环境/磁盘/GPU preflight
  → 下载 Qwen3-30B Base
  → Base HF↔VeXact compatibility
  → 注册 ShareGPT 数据到 LlamaFactory
  → 4 卡 LoRA SFT（GPU 0–3，global batch 64）
  → 合并完整 BF16 HF 模型
  → SFT HF↔VeXact compatibility
  → Base/SFT frozen-dev 基线
  → Candidate-BM25 server
  → 1-step Exact Evidence-GRPO smoke
  → 200 → dev → protect best
  → 400 → dev → protect best
  → 600 → dev → protect best
  → 800 → dev → freeze unique best
  → 审计后首次打开 sealed Test
```

旧大项目只在第一步提供冻结的数据快照。staging 完成后，项目运行不再导入旧仓库的 `scripts/`、`src/` 或 `configs/`。保留的外部依赖只有 LlamaFactory、VeXact/VeOmni/veRL、模型文件和 Python 环境。

## 主线脚本职责

| 脚本 | 作用 |
|---|---|
| `00_stage_frozen_data.sh` | 一次性复制并校验 SFT/RL/dev 冻结输入 |
| `00_preflight.sh` | 环境、数据 hash、GPU、磁盘与框架支持门 |
| `01_download_model.sh` | 可选的模型下载封装；也可直接执行 ModelScope CLI |
| `02_prepare_sft_data.sh` | 校验 observation role，并注册 LlamaFactory 数据 |
| `03_train_sft.sh` | 4 卡 Qwen3-MoE LoRA SFT |
| `04_merge_sft.sh` | LoRA 合并为完整 BF16 HF checkpoint |
| `05_check_vexact_model.sh` | HF/VeOmni/VeXact compatibility gate |
| `06_start_retriever.sh` | 启动本地 Candidate-BM25 HTTP server |
| `07_run_evidence_grpo.sh` | Exact Evidence-GRPO smoke 与四段训练 |
| `08_eval_frozen_dev.sh` | 同协议 frozen-dev Agent 评测 |
| `09_select_best.py` | 按预注册规则选 checkpoint |
| `10_promote_best.sh` | 在旧 optimizer state 清理前保护当前 best HF |
| `11_status.sh` | 只读状态检查 |
| `12_build_result_table.py` | 生成项目主结果表 |
| `launch_grpo.py` | 极简 veRL 入口，不加载旧项目研究 monkeypatch |

代码快照来源和保留理由见 [VENDORED_SOURCES.md](VENDORED_SOURCES.md)。

## 阶段门

| Gate | PASS 条件 | 失败后的动作 |
|---|---|---|
| P0 preflight | 本地冻结数据、4 GPU、框架、≥450 GiB 空间 | 不下载/训练，先解决环境 |
| P1 base compatibility | Qwen3-MoE + VeXact/VeOmni logits 验证成功 | 只修兼容，不改算法 |
| P2 SFT | loss 正常、merge 健康、SFT compatibility PASS | 不进 RL |
| P3 RL smoke | 1 step、完整 agent/tool/reward/update/save 链成功 | 不开长训练 |
| P4 segment | target checkpoint + frozen-dev summary 完整且 health PASS | 停止并分析，不打开 Test |
| P5 freeze | 800 以内唯一 best 被 hard-link 保护并写 hash | 才允许 final Test |

## Checkpoint 空间策略

30B 的 optimizer checkpoint 很大。veRL 只保留最新一套 resumable state；每个 200-step dev 评测后，先运行 partial selector，再把当前 best 的 HF 文件通过同文件系统 hard link 固化到 `artifacts/best_hf`，然后才能继续下一段。这样既能 resume，又不会保存四套 optimizer state。

完整的用户执行顺序见 [COMMANDS.md](COMMANDS.md)。
