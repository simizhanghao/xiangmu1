# Evidence-Aware Deep Research Agent — Qwen3-8B

冻结总计划（唯一执行线）：[PLAN.md](PLAN.md)。

**当前进度（2026-08-20）：** **Best Controlled Policy = GRPO step400**。held-out 是 regression set。finalize-v2b 后 GRPO@400@500：F1 **0.7506** / EM **0.670** / Evidence **0.7243** / finish **1.0** / 二搜未完成 **0/54**。`p_search_2` 仍 10.8%。下一步：MULTITURN_CAPABILITY_AUDIT。先不跑四臂 / Web。不重训。

## 项目目标

在不更换数据、retriever、reward 与 Agent protocol 的前提下，把已验证的 Qwen2.5-3B 系统升级到 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) dense，交付一个可在 4×A100-80G 上完整跑通 full-parameter Evidence-GRPO 的最终 DeepResearch Agent。

30B-A3B 已放弃：训练时 Adam/梯度按全部 expert 存储，实测 `update_actor` 峰值 74.80/79.25 GiB；CPU offload 能更新但一步约 36 分钟，HF gather 再打爆主机内存。最终基座冻结为 Qwen3-8B。

主链只有一条：

```text
Qwen3-8B (dense, non-thinking)
  → LoRA SFT（8B-aligned coldstart v2，1200 DeepSeek rationale）
  → Merge BF16 HF model
  → Gate 4 Exact VeXact 1-step（正确性锚，已 PASS）
  → Step A/B：SGLang μ/π audit + Decoupled Token-TIS 1-step（已 PASS）
  → Gate 5：SGLang + Token-TIS 20-step（已 PASS）
  → smoke-20 frozen-dev@200 NO_COLLAPSE（诊断，非正式 Δ_RL）
  → Gate 5.5 5K，从 SFT merged 开 Formal GRPO-v1
  → Formal-v1@200 frozen-dev NO_COLLAPSE（首个正式里程碑）
  → Formal-v1@400 frozen-dev NO_COLLAPSE（Best Controlled Policy）
  → Formal-v1@600 frozen-dev NO_COLLAPSE（Answer 回落；停 800）
  → formal-dev 1000 确认（无反转）→ 冻结 step400
  → 轻量行为审计 → held-out test 一次（Direct / RAG / SFT / GRPO）
  → Web adapter zero-shot（不训练、不改 action）
```

## 2026-08-18 新结果（Gate 4 → Step B）

| 门 | 判定 | 关键数字 |
|---|---|---|
| Gate 3 frozen-dev@200 | 已锁 | SFT Agent F1 **0.6649** / EM 0.54；Base RAG 0.6659 / 0.57 |
| Gate 4 Exact 1-step | `GRPO_SEGMENT_PASS` | step **2073s**（gen 1817s）；reward 0.286；Exact pearson 0.976 |
| Step A SGLang μ/π | `SGLANG_PROB_AUDIT_PASS` | search 0.375；ESS **0.999**；ρ mean 0.997；mild mismatch |
| Gate 5 Token-TIS 20-step | `SGLANG_TOKEN_TIS_20STEP_PASS` | 20/20 in **45 min**；median ~2.3 min/step；reward 0.28→0.38；ESS≥0.9998 |
| smoke-20 frozen-dev@200 | `SMOKE20_FROZEN_DEV200_NO_COLLAPSE` | vLLM det F1 **0.7155** / EM 0.575 / Evidence **0.614**；finish=1.0；**不是正式 Δ_RL** |
| Gate 5.5 formal 5K | `GATE55_FORMAL_5K_PASS` | train 5000 / formal-dev 1000；SFT overlap 2809 recorded；BM25 index 5016 |
| Formal-v1@200 | `FORMAL_V1_STEP200_FROZEN_DEV200_NO_COLLAPSE` | 5K、200 step；vLLM F1 **0.6988** / EM 0.545 / Evidence **0.7727**；search=1.0 |
| Formal-v1@400 | `FORMAL_V1_STEP400_FROZEN_DEV200_NO_COLLAPSE` | 5K、400 step；vLLM F1 **0.7338** / EM **0.62** / Evidence **0.7477**；`p_search_2=0.085`；**Best Controlled Policy** |
| Formal-v1@600 | `FORMAL_V1_STEP600_FROZEN_DEV200_NO_COLLAPSE` | vLLM F1 **0.717** / EM **0.59** / Evidence **0.634**；finish 0.965；**停 800** |

正式训练是 **Exact-validated, rollout-corrected Agentic GRPO**，不要叫 Exact。VeXact 只作锚。摘要：`results/45_frozen_dev_formal_grpo400/formal400_dev200_summary.json`、`results/48_frozen_dev_formal_grpo600/formal600_dev200_summary.json`。

## Formal-v1@200（2026-08-19，首个正式里程碑）

5K train、batch 32、n=4、lr 1e-6、T=0.7、SGLang + Token-TIS。Init = `outputs/22_sft_qwen3_8b_merged`。**不是** smoke `global_step_20`。200/200 无 NaN/OOM，ESS≈1。

| 模型 | Answer F1 | EM | Evidence F1 | search | 后端 |
|---|---:|---:|---:|---:|---|
| SFT Agent（Gate 3） | 0.6649 | 0.54 | 0.50 | 0.715 | **HF greedy** |
| Formal-v1@200 | **0.6988** | 0.545 | **0.7727** | 1.0 | **vLLM det** |
| Δ vs Gate 3 HF | +0.0339 | +0.005 | +0.273 | +0.285 | 后端不同 |
| SFT（同 vLLM det） | 0.6693 | 0.545 | 0.4939 | 0.71 | **vLLM det** |
| **同后端 Δ_RL** | **+0.0295** | **0** | **+0.2788** | +0.29 | 正式对照 |

读法：SFT 的 vLLM vs HF 只差 F1 +0.0044，后端不是故事。@200 同后端 Δ_RL 是 Answer F1 **+3.0pp**、Evidence **+27.9pp**、EM **不动**。

## Formal-v1@400（2026-08-19，Best Controlled Policy）

同一 frozen-dev@200、同一 vLLM det。`global_step_400` 从 formal-200 resume，配置冻结。200/200 无 collapse。

| 模型 | Answer F1 | EM | Evidence F1 | search | p_search_2 | 后端 |
|---|---:|---:|---:|---:|---:|---|
| SFT（同 vLLM det） | 0.6693 | 0.545 | 0.4939 | 0.71 | 0 | **vLLM det** |
| Formal-v1@200 | 0.6988 | 0.545 | **0.7727** | 1.0 | 0 | **vLLM det** |
| **Formal-v1@400（已冻结）** | **0.7338** | **0.62** | 0.7477 | 0.995 | **0.085** | **vLLM det** |
| Δ_RL vs SFT | **+0.0645** | **+0.075** | **+0.2538** | +0.285 | +0.085 | 正式对照 |
| Δ vs @200 | **+0.035** | **+0.075** | −0.025 | −0.005 | +0.085 | 趋势 |

400 当时决策是续 600。`p_search_2=0.085` 只观察，**不声称 multi-hop**。

## Formal-v1@600（2026-08-19，停线）

同一 frozen-dev@200、同一 vLLM det。`global_step_600` 从 formal-400 resume，配置冻结。协议未崩（finish 0.965 / parse 1.0 / mask 1.0），但质量回落。

| 模型 | Answer F1 | EM | Evidence F1 | finish | gen tok |
|---|---:|---:|---:|---:|---:|
| SFT（同 vLLM det） | 0.6693 | 0.545 | 0.4939 | 1.0 | — |
| Formal-v1@200 | 0.6988 | 0.545 | **0.7727** | 1.0 | 220 |
| **Formal-v1@400** | **0.7338** | **0.62** | 0.7477 | 0.99 | 289 |
| Formal-v1@600 | 0.717 | 0.59 | 0.6343 | 0.965 | **619** |
| Δ 600 vs 400 | **−0.0168** | **−0.03** | **−0.113** | −0.025 | +330 |

600 决策：**STOP_V1_NO_800**。**Best Controlled Policy = GRPO step400**。正式 Δ_RL 继续引用 @400 vs SFT。摘要：`results/48_frozen_dev_formal_grpo600/formal600_dev200_summary.json`。

## 已冻结的实验定义

- Backbone：`Qwen/Qwen3-8B`，`qwen3_nothink`，`enable_thinking=false`。
- SFT：ShareGPT coldstart_v2（4550，`sharegpt_filled.jsonl`），2 epochs，LoRA rank 32，effective global batch 64。旧 v1 仅历史对照。
- RL smoke：当前 128/16 parquet **只用于 Gate 4/5**，不是最终训练集。
- 正式 GRPO：Gate 5.5 构建约 **5,000** HotpotQA questions；fast-dev 200 + formal-dev 1000。
- RL 算法：Evidence GRPO，`R = EM + 0.5 Evidence + 0.1 Format`，`lambda_e=0.5`、cost λ=0、正式 LR `1e-6`、**T=0.7 / top_p=0.95**、batch 32、`n=4`。
- Rollout：正式路径 **SGLang 0.5.5 + 官方 Decoupled Token-TIS**（`rollout_is=token`，threshold=2.0，无 RS，无 bypass）。Exact VeXact 只作 1-step 正确性锚。
- Environment：Candidate-BM25 top-5、同一 `EcaSearchAgentLoop`、Harness v1。
- Memory：actor param/optimizer offload = false。Exact 锚用 PP=4；正式 SGLang 用 TP=1。
- Budget：Formal-v1 已在 600 停；**不跑 800**。1000 不是训练 KPI，是 formal-dev 确认集。
- 选模：先过 finish/format/observation-mask health gate，再按 **Answer F1**（主）、Evidence F1、EM、少重复 query、较早 checkpoint。Joint F1 **只报告，不改选模**。
- Test：held-out 是 `data/sealed/hotpotqa_test500.jsonl`。先 `scripts/run_heldout_smoke.sh`（GRPO@400 n=8），再四臂各跑一次 500。test 后不换 checkpoint。
- Routing：不在本线再逼少搜。Efficient-Agent-v2 只在 Web 之后、且真实成本有问题时才考虑。
- Web：held-out 之后才做 adapter；不重训，不加 `<open>` / `<find>`。
- 能力边界：v1 主 claim 是 retrieval-grounded answering + evidence quality；**不要**把 adaptive routing 当主结论（@400 search≈0.995）。**不可**声称 query reformulation 或 multi-hop（`p_search_2=0.085` 只观察）。

## 目录

| 路径 | 内容 |
|---|---|
| `config/project.env` | 所有固定路径和运行参数 |
| `config/sft_*.yaml` | 8B SFT 与 merge 配置 |
| `config/harness_v1.json` | Gate 2.5 冻结的推理协议；GRPO 必须复用 |
| `config/rl/` | 本地 Candidate-BM25 tool 与 AgentLoop 注册配置 |
| `src/` | AgentLoop、retriever、Evidence reward、协议与评测 |
| `data/` | 冻结输入快照（git ignored） |
| `model/` | 本地 Qwen3-8B 权重（git ignored） |
| `scripts/` | 预检、SFT、GRPO、frozen-dev、选模 |
| `artifacts/` | SFT merged、RL checkpoint、best HF（git ignored） |

## Gate 2.5 / Gate 3 n=8（2026-08-17）

同一 frozen-dev 前 8 题。n=8 **不是** Answer F1 正式门。正式硬底仍是 frozen-dev@200 上 SFT Agent Answer F1 ≥ 0.55。

| 臂 / 修复 | finish | search | Answer F1 | Evidence F1 | 闭环 |
|---|---:|---:|---:|---:|---:|
| Base Direct | — | — | 0.00 | — | — |
| Base RAG | — | — | 0.625 | — | — |
| SFT Agent 初跑 | 0.50 | 0.125 | 0 | ~0 | 1 |
| fix1 parser | 1.0 | 0.125 | 0 | ~0 | 1 |
| fix2 剥空 think | 0.875 | 0.875 | 0.6875 | 0.14 | 6 |
| **fix3 `<tool_response>`** | **1.0** | **0.875** | **0.75** | **0.7417** | **7** |

因果：模型没坏，SFT 也没白训。坏的是训练/推理协议。

1. **fix1** 只修 parser，search 仍 1/8。
2. **fix2** 去掉 HF 在 `enable_thinking=False` 时仍插入的空 `<think></think>`。search 0.125→0.875，Answer F1 0→0.6875。
3. **fix3** 把第二轮 `<observation>`+`Continue` 改成 LlamaFactory `qwen3_nothink` 的 `<tool_response>` 槽位。Evidence F1 0.14→0.74。
4. **Gate 2.5** CPU token 级对照：`GATE_PROTOCOL_PARITY_PASS`（ROUND0/ROUND1 token 完全一致）。

Harness v1 已冻结。后面 Gate 3 @200 和 GRPO rollout 必须复用 `src/agents/react_loop.py` 这一份，禁止再手写另一套 tool 格式。

## 正式 Gate 3 — frozen-dev@200（2026-08-17 PASS）

同一 200 题、同一 BM25@5。官方数字是 **HF generate**，不再重刷替换。

| 臂 | Answer F1 | EM | 行为 |
|---|---:|---:|---|
| Base Direct | 0.2647 | 0.19 | 无外部文档 |
| Base RAG | 0.6659 | 0.57 | 题目当 query，Top-5 |
| **SFT Agent** | **0.6649** | 0.54 | search 71.5% / internal 28.5%，finish 1.0 |

- 硬底 ≥0.55、目标 ≥0.60、finish 健康线 ≥0.90 → **过目标线**。
- SFT 把裸 8B（0.26）训成可自主 search→tool_response→evidence→answer 的 Agent，并**对齐** one-shot RAG。Δ(Agent−RAG) = −0.001。
- Evidence F1 = 0.50；`p_search_2 = 0`。漏搜 10/79；RAG对/Agent错 19；Agent对/RAG错 17。
- 这是 cold-start 成绩，不是项目终局。超过 RAG 交给 Evidence-aware GRPO。第一版 reward 保持 `R_answer + 0.5 R_evidence + 0.1 R_format`，`cost λ = 0`。

## vLLM ↔ HF n=8（2026-08-17 PASS）

`vllm/vllm-openai:latest` serve + Harness v1 拼 prompt + `/v1/completions`。对照 HF fix3 同一 8 题。

| 检查 | 结果 |
|---|---|
| empty think / `<observation>` / extra Continue | 0 / 0 / 0 |
| finish / parse / search | 1.0 / 1.0 / 0.875 |
| Answer F1 / Evidence F1 | 0.75 / 0.7417（与 HF 相同） |
| route / query / finish 对齐 | 8/8 / 8/8 / 8/8 |

以后 eval / rollout / GRPO 生成走 vLLM 服务，不走 HF `model.generate()`，也不走 vLLM chat template。

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
  → smoke-20 frozen-dev@200（NO_COLLAPSE；不从 step20 续训）
  → Gate 5.5 构建正式 HotpotQA-5K
  → Formal-v1@200 frozen-dev（NO_COLLAPSE；从 formal-200 续 400）
  → Formal-v1@400 frozen-dev（NO_COLLAPSE；Best Controlled Policy）
  → Formal-v1@600 frozen-dev（Answer 回落；停 800）
  → formal-dev 1000 确认 → 轻量审计 → held-out test 一次
```

外部依赖：LlamaFactory、VeXact/VeOmni/veRL、本机 8B 权重。

完整命令见 [COMMANDS.md](COMMANDS.md)。代码快照来源见 [VENDORED_SOURCES.md](VENDORED_SOURCES.md)。
