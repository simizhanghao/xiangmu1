# W5 Modular Research Controller — Final Report

## Outcome

W5 is closed as a failed offline deployment gate. No additional model training,
threshold sweep, live-Web integration, or `web-final50` evaluation is authorized.

## Data and training

- 5,000 frozen formal-train questions; 4,998 natural post-Search1 states and two
  reproducible direct-answer states.
- 6,256 grounded-checker states: 4,998 natural plus 1,258 evidence-masked siblings.
- DeepSeek checker: 6,256/6,256 successful; 854 accepted STOP after conservative
  adjudication, 19 weak STOP labels demoted to CONTINUE; no train/dev overlap or
  structural gold leakage.
- Query teacher: 2,229/2,229 successful, 99.87% valid, 0.09% duplicate, 99.96%
  state-conditioned.
- Qwen3-1.7B Controller LoRA: 3,084 balanced rows (1,542 STOP / 1,542 CONTINUE),
  two epochs, global batch 16, 386 steps. Runtime 13m26s, train loss 0.7485,
  CE-dev loss 0.4595; no OOM/NaN.

## Frozen natural-dev500 gate

The question-disjoint natural set was not label-resampled after collection
(STOP=71, CONTINUE=429); the gate therefore uses class recalls and balanced accuracy.

| Metric | Result | Required |
|---|---:|---:|
| AUROC | 0.8564 | >=0.90 |
| STOP recall | 0.7042 | >=0.80 |
| CONTINUE recall | 0.8019 | >=0.80 |
| Balanced accuracy | 0.7530 | >=0.80 |
| Parse-valid rate | 0.9440 | diagnostic |
| Duplicate Query | 0.0357 | <=0.10 |
| State-conditioned Query | 1.0000 | >=0.70 |

The model learned query construction but not a sufficiently separable stopping
boundary. A complete score-threshold audit cannot rescue it: the best balanced
accuracy is 0.7883 (STOP 0.7887, CONTINUE 0.7879), and zero thresholds satisfy
80/80 recall simultaneously.

## Scientific conclusion

The project demonstrates strong Controlled Agentic GRPO and partial adaptive second-hop
retrieval, but the final modular adaptive-depth deployment claim is not met. W5 isolates
the remaining failure to grounded sufficiency/stopping discrimination rather than query
reformulation. This negative result is retained; further tuning would violate the frozen
terminal rule and weaken the causal story.

## W5.5 authorized frozen-backbone probe

The sole precommitted exception to the W5 terminal rule was completed. The frozen
Qwen3-1.7B + W5 LoRA produced 4,998 final-layer vectors (`hidden_dim=2048`); 471
prompts (9.42%) required the fixed source-prefix-plus-`DECISION:` truncation rule.
No backbone parameter was updated and no Web or Teacher API was called.

| Metric | Calibration / train | Frozen dev500 | Required |
|---|---:|---:|---:|
| Weighted train loss | 0.1550 | — | diagnostic |
| AUROC | — | 0.8505 | >=0.90 |
| STOP recall | 0.8136 | 0.7746 | >=0.80 |
| CONTINUE recall | 0.8094 | 0.7762 | >=0.80 |
| Balanced accuracy | — | 0.7754 | >=0.80 |
| Parse-valid | — | 1.0000 | 1.00 |

The original logged `train_loss=0.8947` was LBFGS's returned pre-optimization closure
value; read-only recomputation from the saved head gives the final 0.1550 value above.
Thus the failure is not lack of train-set fit or output parsing. The frozen representation
does not provide a sufficiently generalizable linear STOP/CONTINUE boundary. W5.5 is
therefore terminally closed: no integration, MLP, pooling/truncation ablation, new data,
threshold rescue, DPO, RL, or repeated dev evaluation.
