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
