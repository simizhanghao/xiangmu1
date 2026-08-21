# Web-MultiTurn-v2: formal mining and Bocha migration report

Date: 2026-08-21  
Baseline commit: `e97db2b` (`Add causal depth mining for Web MultiTurn`)

## Scope

This report records all work after `e97db2b`. The upper-level policy remains frozen:
GRPO@400, XML action grammar, ResearchMemory, deterministic grounded causal gates and
decision-SFT schema were not changed. Web-provider credentials are environment-only and
are not stored in the repository.

## Formal data-engineering additions

- Search1 provenance now freezes provider, query, top-k, context budget, leak-filter
  version, source IDs/URLs, timestamp and an SHA256 of the exact normalized documents.
- Mining excludes frozen dev/final IDs, current eval/sealed sets, W2 traces and all
  protocol-development/Pilot questions. A dry-run exclusion manifest verifies zero
  selected overlap.
- Provider failures are no longer silently displayed as ordinary D0. Mining prints the
  reason, preserves a partial artifact and aborts after five consecutive Web errors.
- A configurable minimum request interval prevents provider QPS errors.
- Each trajectory exports `initial_search`, `post_obs_stop` and `post_obs_continue`
  decision classes. `decision_sft_balanced.jsonl` deterministically balances the three
  classes at the smallest STOP/CONTINUE count.
- Exact tokenizer-based ResearchMemory token p50/p95 and replay parity are audited.
- Web-dev depth annotation reuses one immutable Search1 observation for both D1 and D2
  counterfactual checks.

## Brave formal results completed after the baseline commit

### Tool-only N=400

Artifact: `results/56_web_multiturn_v2/depth_mining_n400/summary.json`

| Metric | Result |
|---|---:|
| D0 | 37 |
| D1 candidate | 243 |
| D2 candidate | 120 |
| Usable yield | 90.75% |
| Web errors | 7 |
| Excluded-ID overlap | 0 |

### Mixed causal Pilot

Artifact: `results/56_web_multiturn_v2/mixed_pilot_n120/audit_summary.json`

- Accepted 102 trajectories: D1=54, D2=48.
- 252 decision examples; balanced view=144 (48/48/48).
- ResearchMemory tokens p50=754, p95=1277.
- Duplicate Query2=0; observation-conditioned Query2, new source and new evidence=100%.
- D1 STOP correctness, D2 forced1 insufficiency and D2 positive grounded delta=100%.
- Memory replay, leak, empty observation, invalid reference and oracle-field audits pass.

### Frozen natural web-dev50 annotation

Artifact: `results/56_web_multiturn_v2/web_dev_depth/summary.json`

- D1=22, D2=2, unresolved=26.
- The natural set remains frozen. D2=2 is insufficient for a stable standalone
  `CONTINUE@D2` estimate, so a disjoint strict-causal validation split is required before
  training. `web-final50` was not opened.

## Bocha provider implementation and Tool gate

`WebAdapter(provider="bocha")` calls the Bocha Web Search endpoint with a Bearer token,
normalizes `name/url/summary/snippet/siteName/datePublished` into the existing document
shape, applies the same benchmark leak filtering and leaves AgentLoop untouched. Smoke,
rollout, benchmarking, mining, building and dev annotation CLIs now accept `bocha`.

Fixed n=20 comparison on the same HotpotQA questions:

| Metric | Brave LLM Context | Bocha top-5 | Bocha top-10 |
|---|---:|---:|---:|
| Mean tool latency | 507.33 ms | 141.83 ms | 158.15 ms |
| Nonempty context | 100% | 100% | 100% |
| Error rate | 0% | 0% | 0% |
| Answer-string hit | 55.0% | 20.0% | 20.0% |
| Supporting-title recall | 60.0% | 27.5% | 27.5% |
| Approx. context tokens | 422.75 | 399.2 | 783.5 |

Top-10 doubled context without improving retrieval metrics, so top-5 remained frozen.
Bocha is faster and operationally stable but has materially weaker first-search recall.

## Bocha N=400 mining

Artifact: `results/57_bocha_migration/depth_mining_n400/summary.json`

| Metric | Bocha | Brave historical |
|---|---:|---:|
| D0 | 180 | 37 |
| D1 candidate | 108 | 243 |
| D2 candidate | 112 | 120 |
| Usable yield | 55.0% | 90.75% |
| Web errors | 0 | 7 |

All 400 observations were nonempty, provider=`bocha`, SHA-valid and free of benchmark
leak matches. Bocha preserved the coarse D2 candidate supply, but strict causal building
showed that the heuristic candidate count overestimated usable D2.

## Same-manifest Teacher comparison

Both Teachers replayed the exact Bocha N=400 manifest and frozen Obs1 with identical
prompts, T=0, seed=42, quotas and grounded causal gates. Outputs are isolated and must
never be concatenated.

| Metric | Kimi K2.6 | DeepSeek v4 flash |
|---|---:|---:|
| Candidates attempted | 220 | 220 |
| Accepted | 42 | 59 |
| D1 accepted | 30 | 50 |
| D2 accepted | 12 | 9 |
| Balanced decision examples | 36 | 27 |
| Memory-token p50 / p95 | 1036 / 1815 | 1015 / 1740 |
| Accepted-subset causal audit | PASS | PASS |
| Quantity gate | FAIL | FAIL |

Kimi's dominant rejection was `teacher_over_depth` (118). DeepSeek reduced that to 18
and improved D1, but D2 remained lower; its largest rejects were answer mismatch (47),
forced1 already correct (41), unresolved missing (31) and depth mismatch (21).

Every accepted trajectory under both Teachers has zero duplicate Query2, 100%
observation-conditioned Query2/new source/new evidence, complete memory replay, no leaks
or invalid references, and 100% D1/D2 causal checks. This establishes high subset quality
but insufficient quantity.

## Scientific conclusion and current decision

The project must not start Bocha N=5000 under either tested Teacher. Observed strict rates
project approximately 375 D1 + 150 D2 from 5000 raw questions under Kimi; DeepSeek improves
total D1 but does not repair D2. The bottleneck is therefore not only Teacher latency or
format compliance. It is the interaction between Bocha first-search observations and the
strict grounded second-hop construction.

Current state: provider infrastructure PASS, accepted-subset causal quality PASS, formal
data quantity FAIL. N=5000, decision LoRA, Bocha web-dev relabeling and `web-final50` remain
blocked until a frozen retrieval/data construction path can supply sufficient strict D2
without relaxing the causal gate.
