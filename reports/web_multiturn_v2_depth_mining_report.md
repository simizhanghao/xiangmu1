# Web-MultiTurn-v2: Depth Mining and Causal Gate Report

Date: 2026-08-20
Base commit: `f835357`

## Outcome

W3 trajectory format, shared ResearchMemory, action protocol, and replay parity were
already working. This phase isolated and fixed the missing causal data-selection layer.
No LoRA or Web RL was run.

## Experiments

| Run | Result | Scientific read |
|---|---|---|
| Initial random 2/2/2 smoke | 4/6; D1=2, D2=2, D3=0 | Accepted format quality passed, but minimal-depth causality was not measured and forced D3 conflicted with Hotpot/Brave. |
| Tool-only depth mining smoke | 20 attempted; likely D1=12, D2=7, unresolved=1; 95% usable | Pre-mining removes the 6.7% random-builder yield bottleneck. Labels remain candidates, not training truth. |
| Causal smoke v1 | D1=2 accepted, D2=0 | Miner and builder used different Search1 queries. Invalid causal comparison. |
| Causal smoke v2 | stopped | Same Query1 still re-fetched live Obs1, causing evidence drift/Web errors. Invalid causal comparison. |
| Causal smoke v3 | D1 STOP 2/2, D2=0 | First valid immutable-Obs1 run. D2 failure isolated to Teacher Query2/finalization and overly narrow gates. |
| D2 diagnostic | 1/7 accepted | Proved a genuine D2 path: forced1 F1 0.083 → final F1 1.0. Exposed state-conditioning and answer-normalization false rejects. |
| Stochastic retry2 | 0/7 | Demonstrated temperature 0.2 and answer-only forced1 scoring were unstable/incorrect. |
| Deterministic grounded D2 | **4/7 accepted** | `W3_ACCEPTED_SUBSET_GATE_PASS`; causal D2 construction is feasible. |

## Final D2 gate

- accepted: 4/7; 12 step-level decision examples;
- forced1 grounded-insufficient: 4/4;
- Search2 positive grounded-acceptance delta: 4/4;
- state-conditioned Query2: 4/4;
- new source and new evidence: 4/4;
- duplicate extra query: 0/4;
- memory replay: 8/8;
- empty observations, leaks, invalid evidence references, oracle fields: all 0.

Two accepted examples had raw Answer-F1 delta 0: forced1 guessed the gold but admitted
unresolved Missing or cited no source. They correctly count as grounded-insufficient.

## Fixes frozen by this phase

1. Mine likely depth before calling the strong Teacher.
2. Store and replay the exact mined Search1 documents; same query alone is insufficient
   with a live provider.
3. Define state-conditioned Query2 as either a new Obs1 entity or a non-duplicate query
   targeting the serialized Missing state.
4. Define Search1 sufficiency conjunctively: accepted answer, Missing empty, and valid
   current-memory source IDs.
5. Use deterministic Teacher generation (`temperature=0`, `seed=42`).
6. Export `decision_sft.jsonl`; do not rely on a monolithic long-trajectory CE signal.

## Remaining limitations

- The successful causal set is only four D2 examples; it proves feasibility, not a stable
  population yield.
- Three of seven candidates still failed (wrong answer, over-depth, or early answer).
- Hotpot plus Brave LLM Context is suitable for D1/D2 but not forced D3.
- Hidden web-dev50 depth labels and the mixed n=120 pilot are not complete.
- No evidence yet that the 8B policy learns these decisions; training remains gated.

## Next gate

Run medium Tool-only mining on unused train questions, then a mixed n=120 pilot targeting
54 D1 / 66 D2 but accepting any >=100 high-quality set with at least 20 from each class.
Only after the full causal/data gate passes may the dataset scale to 1K–1.5K and one-epoch
WebMT LoRA begin.
