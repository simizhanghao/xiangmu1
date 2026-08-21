# Final Frozen Metrics

## Controlled held-out 500

| Arm | Answer F1 | EM | Evidence F1 | Search | Finish |
|---|---:|---:|---:|---:|---:|
| Base Direct | 0.2210 | 0.148 | 0.0000 | 0.000 | 0.988 |
| Fixed RAG | 0.3944 | 0.250 | 0.0000 | 1.000 | 0.998 |
| SFT Agent | 0.6064 | 0.532 | 0.4652 | 0.718 | 0.998 |
| GRPO@400 | 0.7506 | 0.670 | 0.7243 | 1.108 | 1.000 |

GRPO@400 improves Answer F1 over SFT by **14.42 percentage points**.

## Real-Web zero-shot n=40

Provider for this frozen evaluation: `brave_llm_context`. Final serving uses Bocha; these metrics are not relabeled as a Bocha benchmark.

| Memory | Answer F1 | EM | Evidence F1 | Search | Finish | Latency ms |
|---|---:|---:|---:|---:|---:|---:|
| No Memory | 0.4529 | 0.300 | 0.2625 | 1.000 | 1.000 | 3100.3 |
| ResearchMemory | 0.4113 | 0.300 | 0.2693 | 1.200 | 0.925 | 3750.5 |

## Adaptive-depth study

| Method | AUROC | STOP/D1 | CONTINUE/D2 | Balanced Acc | Gate |
|---|---:|---:|---:|---:|---|
| WebMT-v2 | — | 0.6500 | 0.2500 | — | WEBMT_BEHAVIOR_FAIL |
| Atomic Controller | — | 0.8500 | 0.7000 | — | W4_BEHAVIOR_GATE_FAIL |
| CoC/DPO | — | 0.8500 | 0.7000 | — | W4_BEHAVIOR_GATE_FAIL |
| W5 Controller | 0.8564 | 0.7042 | 0.8019 | 0.7530 | W5_CONTROLLER_OFFLINE_GATE_FAIL |
| W5.5 Linear Probe | 0.8505 | 0.7746 | 0.7762 | 0.7754 | W55_LINEAR_PROBE_FAIL |
| W6 Decision-only | 0.8333 | 0.6901 | 0.8042 | 0.7472 | W6_STAGE1_DECISION_GATE_FAIL |
| W7 Structured Gap | 0.7120 | 0.5352 | 0.7110 | 0.6231 | W7_S2G_STATIC_GATE_FAIL |

## Frozen conclusion

Controlled Agentic RL and Real-Web transfer pass. Observation-conditioned second-hop retrieval is a partial pass. The deployable adaptive-depth Controller is a terminal fail and is not part of the final serving path.
