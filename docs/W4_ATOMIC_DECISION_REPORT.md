# W4 Atomic Decision Calibration Report

Date: 2026-08-21
Parent commit: `c7a38df`

## Question

W3 LoRA-v2 improved the selected second-query subset but traded D1 stopping for D2
continuation. W4 tests whether this is only an operating-threshold problem, then permits
exactly one consequence-aware preference calibration attempt.

## Leakage discovered and corrected

The first behavior-dev score audit appeared strong (AUROC 0.910), but the graph builder
rendered `Remaining Budget: 0` for every D1 state and `Remaining Budget: 1` for every D2
state. That result is invalid as sufficiency evidence. Every authoritative W4 prompt now
replaces the field with the constant `Remaining Budget: 4`.

With leakage removed, LoRA-v2 scores were:

| Backend | AUROC | AUPRC | Selected STOP@D1 | Selected CONT@D2 |
|---|---:|---:|---:|---:|
| HF | 0.775 | 0.825 | 85% | 70% |
| vLLM | 0.780 | 0.830 | 85% | 70% |

The HF forced behavior test achieved duplicate Q2=5% and finish=100%, but only 50% of
all D2 questions produced an observation-conditioned Q2. Manual inspection confirmed the
four failures were generic/repeated queries, not audit false negatives. A natural
Candidate-BM25 n=4 integration smoke sent all eight post-observation decisions to SEARCH;
all four questions hit the three-search cap, with mean duplicate count 1.0 and EM 75%.
Thus an offline threshold does not transfer to natural Agent states.

## Single W4-1 Chain-of-Calibration attempt

The data builder uses only the disjoint, audited 440 graph-replay trajectories and has no
behavior-dev overlap. It creates 440 short consequence-aware DPO pairs:

- 110 D1@Obs1: `ANSWER > unnecessary SEARCH`;
- 220 D2@Obs1: `grounded SEARCH/Q2 > premature ANSWER`;
- 110 D2@Obs2: `ANSWER > redundant SEARCH`.

Chosen decisions are exactly balanced (220 SEARCH / 220 ANSWER), budget is fixed to 4,
and long Web/tool responses are excluded from the trainable preference response.

Training starts from merged LoRA-v2 and uses one fixed configuration only: LoRA rank 16,
beta 0.1, LR 5e-6, one epoch, four GPUs, global batch 16. It completed 28/28 optimizer
steps in 251.5 seconds. Mean loss was 0.6577; step loss moved from 0.6931 to 0.6312.

## Independent post-training result

| Metric | LoRA-v2 before CoC | W4 CoC | Delta |
|---|---:|---:|---:|
| AUROC | 0.7750 | 0.7825 | +0.0075 |
| AUPRC | 0.8246 | 0.8282 | +0.0036 |
| STOP@D1 | 85% | 85% | 0pp |
| CONTINUE@D2 | 70% | 70% | 0pp |
| Duplicate Q2 | 5% | 5% | 0pp |
| Obs-conditioned Q2 | 50% | 50% | 0pp |
| Finish | 100% | 100% | 0pp |

Gate: **`W4_BEHAVIOR_GATE_FAIL`**.

The small DPO run optimized its local preference objective, but did not materially change
the held-out decision geometry or the full behavior. This is a useful negative result:
the remaining issue is not fixed by one global atomic threshold or a small low-rank
calibration over graph-replay states. Natural Web evaluation remains blocked.

## Scientific interpretation

Small LoRA/DPO runs can work when the base policy already contains the desired capability
and only a low-dimensional style, protocol, or decision boundary must move. They are not
expected to create a missing capability or repair a large train/deployment state shift.
Here 440 pairs were enough to test the narrow calibration hypothesis, but the unchanged
held-out behavior shows that hypothesis was insufficient. More epochs or a beta/LR sweep
would weaken the causal story and are explicitly disallowed.
