# 提纯代码来源

这些文件从父项目的已验证主线复制到 `xiangmu`，目的是让最终项目不再运行时引用父项目代码。复制基线 commit 为 `8e81cc6782445ce34bcf9cfa46d5304a47cce5cf`；后续修改只在 `xiangmu` 内进行。

| 本地路径 | 父项目来源 | 保留原因 |
|---|---|---|
| `src/rl/eca_search_agent_loop.py` | `src/rl/eca_search_agent_loop.py` | 已通过 Exact rollout closure 的多轮 AgentLoop |
| `src/rl/candidate_bm25_tool.py` | `src/rl/candidate_bm25_tool.py` | veRL BaseTool 接口 |
| `src/rl/retrieval_server.py`、`candidate_index.py` | 同路径 | sample-scoped Candidate-BM25 服务 |
| `src/rl/rewards_evidence.py`、`reward_breakdown.py` | 同路径 | 冻结 Evidence reward |
| `src/rl/mask_audit.py` | 同路径 | AgentLoop 的 loss-mask 完整性依赖 |
| `src/agents/react_loop.py` | 同路径 | frozen-dev 的同协议离线 Agent evaluator |
| `src/eval/*` | 同路径 | Answer/Evidence/format/trace 指标 |
| `src/sft/prototype_builder.py` | 同路径 | 冻结 prompt/protocol 与 JSONL loader |
| `src/sft/teacher_reasoning.py` | 同路径 | DeepSeek Teacher-1200 validator / prompt（2026-08-17 冻结） |
| `src/sft/coldstart_builder.py` | 同路径 | frozen-dev overlap 与 train-only 断言 |
| `scripts/generate_teacher_reasoning.py` | 同路径 | Teacher API helper；只从 Dee `src/` 导入 |
| `scripts/export_coldstart_sharegpt.py` | 同路径 | ShareGPT export / observation role |
| `src/tools/candidate_bm25.py` | 同路径 | BM25 排序和 observation 格式 |
| `scripts/run_agent_rollout_smoke.py` | 同路径 | frozen-dev 入口 |
| `scripts/start_candidate_retrieval_server.py` | 同路径 | retriever 入口 |
| `config/rl/*` | `configs/rl/*` | tool/AgentLoop Hydra 注册 |

没有复制 CUR、DSSR、Root Pivot、Step Gate、Boundary reward、optimizer sweep 或对应控制器。`launch_grpo.py` 是新的极简入口，明确不加载父项目的研究型 trainer monkeypatch。

`eca_search_agent_loop.py` 目前保留了已验证快照中的 dormant audit hooks，因为直接删改会改变 Exact 验证对象；最终项目运行没有设置这些 audit/CUR 环境变量，因此它们不生效。待 8B compatibility smoke 通过后，可另做行为等价的精简，不在 smoke 之前改动关键 AgentLoop。
