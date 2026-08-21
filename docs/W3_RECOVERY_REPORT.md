# W3 Recovery Report

Status: closed after strict causal evaluation on Bocha + DeepSeek-v4-flash.

## Changes since `394ebbb`

- Added deterministic, gold-blind post-Obs1 Q2 Beam=3 with per-branch diagnostics.
- Added external-pool isolation and official WebShaper100 preparation.
- Added `answer_absent` candidate policy for datasets without supporting-title labels;
  final D1/D2 labels still come only from the unchanged causal Builder.
- Added Bocha-native source-first synthesis with grounded Source A/Bridge/Source B,
  bridge/answer hiding, URL separation, and fresh Search1 revalidation.
- Fixed orchestration so an incomplete but nonempty synthesis subset is always audited.
- Extended the offline causality test to verify deterministic Beam selection.

## Results

| Experiment | Candidate supply | Strict result | Decision |
|---|---:|---:|---|
| Bocha + DeepSeek single Q2 | 112 D2 candidates | 9 D2 | baseline |
| Same state, Q2 Beam=3 | 112 D2 candidates | 16 D2 | causal subset PASS; gain too small |
| WebShaper100 | 6 D1 / 94 apparent D2 | 1 D1 / 1 D2 | quantity FAIL |
| Source-first v1 | 12 synthesized; 6 apparent D2 | 4 D1 / 0 D2 | bridge visibility missing |
| Source-first v2 | 120 discovery seeds → 5 frozen D2 candidates | 1 strict D2 | causal subset PASS; yield too low |

Beam3 selected proposal ranks 1/2/3 for 11/3/2 accepted rows. All 16 accepted Beam rows
passed forced1 insufficiency, positive grounded delta, state-conditioned Query2, new
source/evidence, memory parity and leakage checks. This rules out gate corruption while
showing that extra query proposals do not recover enough trajectories.

WebShaper Tool mining labeled 94 questions as answer-absent after Search1, but strict
construction accepted only one D2. “Answer absent” is candidate recall, not proof of a
recoverable bridge or minimal depth.

Source-first v2 corrected that issue by requiring a fresh question-based Search1 to expose
the hidden bridge while hiding the answer, then freezing the exact Obs1. One of five rows
survived the unchanged Builder. Its audit is clean; the end-to-end yield is 1/120 (0.83%).

## Conclusion

The protocol is working: it rejects plausible-looking but non-causal multi-turn examples.
The bottleneck is data-environment alignment, not Teacher strength, Web reliability, or an
overly strict gate. Do not run Beam5, another Teacher, N5000 brute-force mining, or LoRA on
these sparse rows. A future reopening requires a materially different source graph/index
that constructs reliable bridge paths before question generation.
