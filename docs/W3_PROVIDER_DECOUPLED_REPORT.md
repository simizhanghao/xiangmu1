# W3 Provider-Decoupled Multi-Turn Report

## Outcome

W3 established that causally valid D1/D2 supervision can be built without live-provider
mining, but two controlled LoRA attempts did not meet the frozen behavior gate. The branch
is closed; GRPO@400 remains the frozen deployable policy.

## Data and causal audit

- Graph replay smoke: 8 D1 + 8 D2, all memory/leak/decision checks passed.
- Frozen behavior dev: 20 D1 + 20 D2, disjoint from train/eval/sealed IDs.
- Final train mix: 611 trajectories (D1=324, D2=287), 1,509 decisions.
- Balanced v1 view: 861 rows; graph/live decision share 74.9%/25.1%.
- Duplicate, leak, invalid reference and oracle-field rates were zero.
- D2 forced-one-search insufficiency, positive delta, observation-conditioned Q2, new
  source and new evidence checks were all 100% by construction and replay audit.

## LoRA-v1

Initialization was GRPO@400. LoRA r=16/alpha=32, assistant-only CE, masked observations,
one epoch, global batch 32, cutoff 6144. It completed 27 steps in 299.5s:

- train loss: 0.1923
- eval loss: 0.1124
- STOP@D1: 75%
- CONTINUE@D2: 15% (GRPO@400 baseline: 5%)

The row distribution was 1:1:1, but target character mass was initial=14.3%, STOP=67.9%,
CONTINUE=17.7%. Long evidence/answer targets dominated token-level CE.

## Single token-balanced repair

The only authorized repair restarted from GRPO@400 and repeated CONTINUE four times. This
gave 1,722 rows and post-observation target mass STOP=467,214 vs CONTINUE=488,144 chars.
Everything else stayed frozen. It completed 54 steps in 505.3s:

- train loss: 0.2009
- eval loss: 0.1073 → 0.0766 → 0.0737
- STOP@D1: 65%
- CONTINUE@D2: 25%
- duplicate Q2: 0%
- observation-conditioned Q2: 25% over all D2 states (100% conditional on searching)

Token balancing improved continuation another 10pp and eliminated duplicate Q2, but moved
the policy along a stop/continue trade-off: STOP fell below its 70% gate while CONTINUE
remained far below 60%. Nine of 40 diagnostic generations hit the 192-token cap before an
action, so raw finish=77.5% is truncation-confounded. Even the optimistic upper bound that
counts both truncated D2 rows as SEARCH is 35%, which cannot change the gate decision.

## Scientific conclusion

The failure is not attributable to provider instability, leakage, observation masking,
serializer mismatch, missing D2 support, or duplicate-query targets. CE over the same LM
policy improved one side of routing at the expense of the other and did not learn the
frozen adaptive-depth boundary. No natural Web-dev50 or third hyperparameter sweep is
licensed. Preserve these artifacts as a negative result and use GRPO@400 for the existing
zero-shot Web line.
