# W6 Multi-stage Controller — Terminal Report

## Protocol

W6 tested whether a narrow decision-only curriculum could learn grounded sufficiency
before Query generation. Stage 1 used Qwen3-1.7B with one fixed LoRA configuration
(`r=32`, `alpha=64`, `lr=1e-5`) and no API calls. Its 3,972 natural-only rows comprised
three balanced curriculum rounds: 662 train-natural STOP states repeated once per round
and 1,986 disjoint train-natural CONTINUE states. Frozen dev500 had zero train overlap.

Training completed 249 optimizer steps in 7m11s. TensorBoard was recorded; loss declined
from roughly 6.11 initially to roughly 0.21 at the end (full-run mean 0.8433). This shows
optimization succeeded, but without a non-gating validation split it does not by itself
prove or disprove classical overfitting.

## One-shot frozen-dev500 Gate

| Metric | W5 mixed SFT | W5.5 linear probe | W6 decision-only | W6 Gate |
|---|---:|---:|---:|---:|
| AUROC | 0.8564 | 0.8505 | 0.8333 | diagnostic |
| STOP recall | 0.7042 | 0.7746 | 0.6901 | >=0.80 |
| CONTINUE recall | 0.8019 | 0.7762 | 0.8042 | >=0.80 |
| Balanced accuracy | 0.7530 | 0.7754 | 0.7472 | >=0.80 |
| Parse-valid | 0.9440 | 1.0000 | 1.0000 | diagnostic |

W6 improved formatting and increased CONTINUE recall by only 0.23 percentage points
over W5, while STOP recall fell 1.41 points, balanced accuracy fell 0.59 points, and
AUROC fell 2.31 points. The narrower target therefore did not improve the deployable
sufficiency boundary. The falling train loss together with failed frozen-dev behavior is
consistent with a generalization/representation gap, but the protocol does not isolate
classical overfitting as its unique cause.

## Decision

`W6_STAGE1_DECISION_GATE_FAIL` is terminal under the precommitted protocol. Stage 2,
Stage 3, DPO, alternate ranks, threshold tuning, and repeated dev evaluation are not run.
The Controlled GRPO@400 result remains valid; only the modular adaptive-depth Controller
claim remains unmet.
