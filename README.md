# Evidence-Aware Deep Research Agent — Qwen3-8B

冻结总计划（唯一执行线）：[PLAN.md](PLAN.md)。

**当前进度（2026-08-17）：** Data preparation complete. Gate 1.5D PASS（DeepSeek Teacher-1200 + 1 条同层 replacement）。Next: Qwen3-8B LoRA SFT smoke（≤8）。不要用旧 4550 v1 开训。

## 项目目标

在不更换数据、retriever、reward 与 Agent protocol 的前提下，把已验证的 Qwen2.5-3B 系统升级到 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) dense，交付一个可在 4×A100-80G 上完整跑通 full-parameter Evidence-GRPO 的最终 DeepResearch Agent。

30B-A3B 已放弃：训练时 Adam/梯度按全部 expert 存储，实测 `update_actor` 峰值 74.80/79.25 GiB；CPU offload 能更新但一步约 36 分钟，HF gather 再打爆主机内存。最终基座冻结为 Qwen3-8B。

主链只有一条：

```text
Qwen3-8B (dense, non-thinking)
  → LoRA SFT（8B-aligned coldstart v2，1200 DeepSeek rationale）
  → Merge BF16 HF model
  → Evidence GRPO（Exact VeXact + Candidate-BM25，GPU Adam）
  → 20-step throughput gate
  → frozen dev @ 200/400/600/800
  → 唯一 best checkpoint
```

## 已冻结的实验定义

- Backbone：`Qwen/Qwen3-8B`，`qwen3_nothink`，`enable_thinking=false`。
- SFT：ShareGPT coldstart_v2（4550，`sharegpt_filled.jsonl`），2 epochs，LoRA rank 32，effective global batch 64。旧 v1 仅历史对照。
- RL smoke：当前 128/16 parquet **只用于 Gate 4/5**，不是最终训练集。
- 正式 GRPO：Gate 5.5 构建约 **5,000** HotpotQA questions；fast-dev 200 + formal-dev 1000。
- RL 算法：Evidence GRPO，`lambda_e=0.5`、LR `1e-6`、`temperature=0.9`、`top_p=0.95`、`n=4`。
- Environment：Candidate-BM25 top-5、同一 `EcaSearchAgentLoop`、Exact VeXact。
- Memory：actor param/optimizer offload = false；rollout PP=4；`max_model_len=8192`。
- Budget：正式终点 200/400/600/800；1000 不是 KPI。
- 选模：先过 finish/format/observation-mask health gate，再按 Answer F1、Evidence F1、EM、少重复 query、较早 checkpoint。
- Test：在唯一 best 冻结前禁止打开 sealed HotpotQA Test。

## 目录

| 路径 | 内容 |
|---|---|
| `config/project.env` | 所有固定路径和运行参数 |
| `config/sft_*.yaml` | 8B SFT 与 merge 配置 |
| `config/rl/` | 本地 Candidate-BM25 tool 与 AgentLoop 注册配置 |
| `src/` | AgentLoop、retriever、Evidence reward、协议与评测 |
| `data/` | 冻结输入快照（git ignored） |
| `model/` | 本地 Qwen3-8B 权重（git ignored） |
| `scripts/` | 预检、SFT、GRPO、frozen-dev、选模 |
| `artifacts/` | SFT merged、RL checkpoint、best HF（git ignored） |

## 从零执行流程

```text
冻结输入已在 data/
  → preflight
  → Qwen3-8B 已在 model/
  → Base HF↔VeXact compatibility
  → 注册 ShareGPT 到 LlamaFactory
  → 4 卡 LoRA SFT
  → merge BF16
  → SFT frozen-dev
  → Candidate-BM25
  → 1-step Exact GRPO smoke on 128（必须 GRPO_SEGMENT_PASS）
  → 20-step throughput on 128
  → Gate 5.5 构建正式 HotpotQA-5K
  → 200 → 400 → 600 → 800 + fast-dev 200 / formal-dev 1000
  → 唯一 best 后再开 sealed Test
```

外部依赖：LlamaFactory、VeXact/VeOmni/veRL、本机 8B 权重。

完整命令见 [COMMANDS.md](COMMANDS.md)。代码快照来源见 [VENDORED_SOURCES.md](VENDORED_SOURCES.md)。
