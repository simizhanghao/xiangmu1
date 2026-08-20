# Qwen3-8B DeepResearch — Frozen Master Plan

Status: **FROZEN 2026-08-15**. Unique execution line: `/data1/hcc/deepresearch/Dee`.
Parent 3B CUR / DSSR / Boundary docs are historical. They are not this project's runtime.

This is not “30B failed, try a smaller model.”
System, algorithm, and protocol are already frozen. The 30B MoE backbone could not land on 4×A100-80G. Replace it with trainable **Qwen3-8B dense**, then close one complete Agentic RL loop.

## Unique line

```text
Dee/model Qwen3-8B
        │
        ▼
[0] 8B Contract Freeze
        │
        ▼
[1] Base Compatibility
        │
        ▼
[1.5] 8B capability relabel + SFT v2 selection
        │
        ▼
[2] Agentic LoRA SFT (8B-aligned 4550 v2)
        │
        ▼
[3] Controlled SFT Evaluation
        │
        ▼
[4] Exact GRPO 1-step Smoke
        │
        ▼
[5] 20-step Throughput Gate (128 smoke set)
        │
        ▼
[5.5] HotpotQA-5K formal RL contract
        │
        ▼
[6] Formal-v1 GRPO 200 → 400 → 600 (STOP; no 800)
        │
        ▼
[7] formal-dev 1000 confirm → Best Controlled Policy = GRPO step400
        │
        ▼
[7.5] Light behavior audit (already-search vs always-search)
        │
        ▼
[8] Held-out test once: Direct / Fixed-RAG / SFT / GRPO@400
        │
        ▼
====== CONTROLLED AGENT DONE ======
        │
        ▼
[9] Live-Web adapter (same policy, no train, no new actions)
        │
        ▼
[10] Controlled BM25 vs Web zero-shot
        │
        ▼
[11] Interview / README / Demo
        │
       STOP
```

Hard split: **train the policy and run Web in separate milestones.**
Do not mix RL gains with search-backend gains.

## Two milestones only

### Milestone A — Trainable Agentic RL

```text
8B Contract → Compatibility → 8B relabel / SFT v2 → SFT → SFT Eval
  → GRPO Smoke → Throughput → 200/400/600 (stop)
  → formal-dev 1000 confirm → Best Controlled Policy = step400
  → light search-gain audit → held-out test once
  (Direct / Fixed-RAG / SFT / GRPO@400)
```

### Milestone B — Same policy, Live Web

```text
@400 frozen
  → Web adapter (tool-internal SEARCH / fetch / extract)
  → still <search> + <tool_response>
  → Controlled BM25 vs Web zero-shot
  → README / Demo → STOP
```

Do **not** add `<open>` / `<find>` on v1 Web. Do **not** retrain.

Web starts only after Best Controlled Policy freeze + light audit + held-out test.

---

## Frozen algorithm / data / protocol

Do not reopen these while executing Gates 0–8.

| Item | Frozen value |
|---|---|
| Backbone | `Qwen/Qwen3-8B` dense, `Qwen3ForCausalLM`, `model_type=qwen3` |
| Local weights | `/data1/hcc/deepresearch/Dee/model` |
| Thinking | `qwen3_nothink`, `enable_thinking=false` |
| SFT data | original 4550 coldstart_v1 (internal / search_format / evidence / evidence_reasoning). Do not expand. |
| SFT method | LoRA rank 32, alpha 64, 2 epochs, global batch 64 |
| Merged SFT | `Dee/artifacts/models/qwen3_8b_sft_merged` |
| Reward | `R = R_answer + 0.5 R_evidence + 0.1 R_format` |
| Current RL parquet | **128/16 is smoke / throughput only.** Not the formal GRPO train set. |
| Formal GRPO train | **~5,000 HotpotQA questions** (build at Gate 5.5, after smoke+throughput PASS) |
| Fast dev | frozen-dev 200, used at every 200/400/600/800 |
| Formal dev | **1,000** HotpotQA, used to pick unique best (Gate 5.5) |
| Sealed test | current 500 stays unread; Gate 5.5 may add a larger unseen final set without opening it |
| RL algorithm | Evidence GRPO, `n=4`, `lr=1e-6`, KL=0.001, `T=0.9`, `top_p=0.95` |
| Env | Candidate-BM25 top-5, `EcaSearchAgentLoop`, Exact VeXact |
| Memory | GPU Adam: param/optimizer/FSDP offload = false; PP=4; `max_model_len=8192` |
| Formal budget | 200 / 400 / 600 / 800. **1000 is not a KPI.** |
| GPUs | physical 0–1–2–3 only |
| Test | sealed HotpotQA Test stays unread until unique best is frozen |
| Web | PAUSED until held-out test of GRPO@400 |
| Fallbacks | Qwen3-8B → Qwen2.5-7B-Instruct (framework fail) → Qwen3-4B (throughput RED) |

Do **not** reuse the 30B LoRA adapter. Do **not** regenerate Gold / BM25. Do **not** let Kimi write full 4550 Agent trajectories.
**Do** rebuild SFT *selection* with Qwen3-8B Direct/Oracle labels, then write **all 1200** new `evidence_reasoning` rationales with local Kimi 2.6 (Gate 1.5B).

---

## Gate table

| Gate | What runs | What it proves | Next |
|---|---|---|---|
| **0 Contract** | Retarget `Dee` identity/SFT/GRPO/checkpoint contracts; full-repo preflight | No live path still sources `Qwen3_30B` | Gate 1 |
| **1 Compatibility** | HF + tokenizer + non-thinking + VeOmni/VeXact min init | 8B can enter the frozen Exact stack | Gate 1.5 |
| **1.5 Relabel** | 8B Direct+Oracle on 8k pool; rebuild 4550 v2 selection; **1200/1200** Kimi grounded rationales (thinking ON, save 2–6 sentence final only) | Cold-start matches 8B capability | Gate 2 |
| **2 SFT** | LoRA on **8B-aligned 4550 v2**; merge BF16 | Agent protocol is re-learned | Gate 3 |
| **3 SFT Eval** | Base Direct / Base RAG / SFT Agent @ frozen-dev 200 | Qualified RL start | Gate 4 |
| **4 GRPO Smoke** | Exact GRPO 1 real update on the **128** smoke set | rollout→reward→Adam→ckpt | Gate 5 |
| **5 Throughput** | 20 consecutive steps on the **128** set | 800 is deliverable | Gate 5.5 |
| **5.5 Formal data** | Build HotpotQA-5K train + formal-dev 1000; keep 128 as smoke | Formal RL contract exists | Gate 6 |
| **6 GRPO** | 200→400→600 on **5K**, fast-dev 200 each; **no 800** | RL improves the Agent; sweet spot @400 | Gate 7 |
| **7 Selection** | 1000 confirm → freeze **Best Controlled Policy = GRPO step400** → light audit → held-out test once (Direct / RAG / SFT / GRPO) | Controlled Agent done | Web zero-shot adapter |

One gate at a time. No new research line.

---

## Gate 0 — Contract freeze

Four contract classes.

### Identity

```text
PROJECT_ROOT = /data1/hcc/deepresearch/Dee
MODEL_ID     = Qwen/Qwen3-8B
BASE_MODEL   = /data1/hcc/deepresearch/Dee/model
model_type   == qwen3
model_type   != qwen3_moe
SFT_MERGED   = Dee/artifacts/models/qwen3_8b_sft_merged
```

### SFT (change paths/names only)

Keep: 4550 content, splits, LoRA 32, Agent protocol, observation masking, `qwen3_nothink`.

### GRPO memory

```text
actor param offload      = false
actor optimizer offload  = false
enable_fsdp_offload      = false
gradient checkpointing   = true
VeXact PP                = 4
max_model_len            = 8192
```

Adam stays on GPU. That is a primary reason to use 8B.

### Checkpoint

During training save only `model / optimizer / extra_state` for resume.
HF gather is an eval/deploy action, not a per-step training action.
Export HF only at **200 / 400 / 600 / 800** for frozen-dev.

### Gate 0 PASS

Full-repo preflight must show:

```text
0 live Qwen3_30B paths in runtime configs/scripts
MODEL_ID = Qwen/Qwen3-8B
BASE_MODEL = Dee/model
model_type = qwen3
thinking disabled
GRPO FSDP optimizer offload = false
GRPO max_model_len = 8192
GRPO PP = 4
milestones = 200 / 400 / 600 / 800
```

Allowed leftover strings: historical README, 30B failure notes, old outputs that are not sourced at runtime.

One sentence: **no official entrypoint still knows `Qwen3_30B` exists.**

---

## Gate 1 — Compatibility (no SFT)

Three levels:

1. Transformers load
2. Chat template + `enable_thinking=false` short generation
3. VeOmni / VeXact recognize the same model

Question: can Qwen3-8B enter the frozen Exact Agentic RL runtime unchanged?
Not: does it answer well?

PASS: HF load, tokenizer, chat template, non-thinking, short generation, VeOmni recognition, VeXact init.

---

## Gate 1.5 — 8B capability relabel (before SFT)

Priority: stale **3B Direct/Oracle selection** > Kimi coverage of `evidence_reasoning` > 4550 size.
Old 400 Kimi + 800 template was a cost-sensitive hard-case booster, **not** a coverage claim. Local Kimi (`http://10.16.137.2:8000/v1`, `Kimi-K2.6-CT-FP8KV`) removes that constraint. Do **not** rerun the old 400 alone. Do **not** Kimi-write internal / search_format / evidence.

### 1.5A — Qwen3-8B Direct + Oracle on the same 8k pool

```text
8k HotpotQA pool
  → 8B Direct  (question only)
  → 8B Oracle  (question + gold supporting docs)
  → compare vs 3B labels
```

Publish the 3B→8B shift matrix, especially `3B Direct wrong → 8B Direct correct`.
If that cell is large (tens of percent), old search_format teaching is misaligned and v2 is mandatory.

### 1.5B — After 8B selection: Kimi writes all 1200 `evidence_reasoning`

Order is locked:

```text
8k Direct/Oracle  →  1.5C select 4550  →  1.5B Kimi on the NEW 1200
```

Teacher: local Kimi 2.6, thinking **ON** internally. Save only the final concise grounded justification (typically 2–6 sentences, minimum sufficient). Never store hidden chain-of-thought / “First I need to think…” noise.

Input = question + gold supporting facts + gold answer. Code still owns evidence tags and `<answer>`. Kimi only writes the `<think>` rationale.

Keep the old 400 as cache/audit only. Do not treat reuse-rate ≥250 as a skip gate.

Conceptual mix inside the 1200 (exact counts after 8B map, not frozen now): easy bridge / normal / oracle-hard. Do not restrict Kimi to hardest-only.

### 1.5C — Rebuild selection, not Gold/BM25/protocol

Reuse Question / gold / BM25 / protocol templates. Only re-assign categories with 8B labels. Keep mix **950 / 1250 / 1150 / 1200**. Output `qwen3_8b_coldstart_v2`.

Selection signals only: Direct EM, Oracle EM, Oracle token F1, `sample_id`, seed=42.
**Do not** use `evidence_f1`. This gate writes a manifest/audit only — no Builder text, no Kimi.

```text
Direct EM=1 (1778)                    → internal 950   (stratified type×level)
Direct=0 Oracle=1 (1853)              → search_format 1250 + evidence 603
Direct=0 Oracle=0 (4369)
  highest Oracle F1                   → evidence 补 547
  remaining tertiles 400+400+400      → evidence_reasoning 1200
```

1227 Direct-ok / Oracle-err stay in the **internal candidate pool**. They only prove: 8B already produced the exact answer without search. Do not read them as “Oracle made the model worse.”

Reasoning bands (after removing the 547 evidence hard items), equal-count tertiles of remaining Oracle token F1:
`near_solved` / `medium` / `genuine_hard` × 400.

Script: `Dee/scripts/select_8b_coldstart_v2.py`.
Manifest: `Dee/results/16_select_8b_coldstart_v2/`.

Then Builder v2 (deterministic text) → DeepSeek-V4-Flash on the frozen 1200
ids → Gate 2 LoRA. Query rewriting stays for on-policy GRPO.

Builder v2 script: `Dee/scripts/build_8b_coldstart_v2.py` — PASS.
Teacher-1200: `Dee/scripts/fill_teacher_reasoning_v2.py` fills the frozen
1200 ids only. Teacher is **DeepSeek `deepseek-v4-flash` only**
(`https://api.deepseek.com`, thinking ON, `reasoning_effort=max`,
`response_format=json_object`, `max_tokens=4096`). Input = question +
gold supporting evidence + reference answer. Never feed distractor
documents. Save only the 2–6 sentence JSON `reasoning`; never persist
`reasoning_content`. Coverage heuristic is a soft audit. DeepSeek
30-sample stratified smoke (10/10/10 bands) is
`GATE_DEEPSEEK_TEACHER_SMOKE_PASS` (local rescore, 0 API calls;
hard_fail=1 leftover surname alias). Teacher contract is frozen.
Do not start Gate 2 until Teacher-1200 Gate + 1.5D audit PASS.

Teacher hard gate (n=1200): unique=1200, pending=0, grounded ≥98%,
answer derivation ≥98%, hard-fail ≤1%, XML/meta leak=0, sample-id delta=0,
`reasoning_content_saved=0`. Lexical evidence-coverage is a **soft audit**.

## Gate 2 — Agentic LoRA SFT

```text
Qwen3-8B + 4550 coldstart + LoRA rank 32
  → merge qwen3_8b_sft_merged
```

SFT teaches **how to act**. GRPO teaches **how to act better**.

---

## Gate 2.5 — Protocol Parity（2026-08-17 PASS）

Training–inference protocol must match before official Gate 3 / GRPO.

- Round 0: no empty `<think></think>` after `<|im_start|>assistant`
- Round 1: ShareGPT `observation` → LF `<tool_response>` slot; no extra `Continue`
- Frozen contract: `config/harness_v1.json` + `src/agents/react_loop.py`
- n=8 fix3 (not official): Answer F1 0.75 / Evidence F1 0.7417 / search 0.875 / finish 1.0

## Gate 3 — SFT frozen-dev@200

Rerun the first three protocols on the **same 8B backbone**:

1. Base Direct
2. Base one-shot RAG
3. SFT Agent

Do not compare 8B GRPO against 30B SFT.

| Line | Answer F1 |
|---|---|
| Hard floor | `>= 0.55` — below this, **STOP, no GRPO** |
| Target | `>= 0.60` — healthy RL start |

30B SFT reference only (not a must-copy): Answer F1 0.63 / Evidence F1 0.54 / search 0.80 / finish 0.935.

### Official Gate 3 result（2026-08-17 PASS, HF generate）

| Arm | Answer F1 | EM |
|---|---:|---:|
| Base Direct | 0.2647 | 0.19 |
| Base RAG | 0.6659 | 0.57 |
| SFT Agent | **0.6649** | 0.54 |

`GATE3_TARGET_PASS`. Artifact: `Dee/results/26_gate3_frozen_dev_200/gate3_summary.json`.

SFT taught the Agent loop and matched one-shot RAG. It did **not** beat RAG. Do not retrain SFT. Do not change BM25. Official 0.6649 is frozen; do not rerun 200 to replace it.

Residual for RL (not SFT): missed-search 10/79; RAG-ok/Agent-bad 19; Agent-ok/RAG-bad 17; Evidence F1 0.50; `p_search_2=0`.

Inference backend after this gate: **vLLM** (generate only). Harness v1 prompt/parser stay.

---

## Gate 3.5 — GRPO Exploration Audit（after vLLM parity）

Do **not** enter 5K GRPO or even 1-step update until this passes.

1. **n=8 vLLM ↔ HF Harness v1 parity**（2026-08-17 `VLLM_HF_PARITY_PASS`）. Image `vllm/vllm-openai:latest` serve + `/v1/completions`. Same 8 IDs: empty-think=0, no `<observation>`, no extra Continue, finish 1.0, search 0.875, F1 0.75 / Evidence 0.7417, route/query/finish agree 8/8. Official Gate 3 remains HF@200 F1=0.6649.
2. **32×8 T=0.7 / top_p=0.95**（2026-08-17 `GATE35_UNDEREXPLORE`）. Artifact: `Dee/results/28_gate35_exploration_32x8/`. 256/256, finish=1.0, parse=1.0, group reward-std nonzero **0.75**, nonzero advantage **0.75**, route diversity **0.53**. `query_diversity=0`, `p_search_2=0`. Reward is healthy. Residual is **search-query peakedness**, not SFT failure and not a flat reward.
3. Read-only query audit (2026-08-17): `query_diversity=0` is **not** a stats bug (exact string set, no normalize). Searchable groups copy the **full question** as the query. Route still flips internal/search. Do not retrain SFT.
4. **Gate 3.5B — global T/top_p sweep DONE（2026-08-18 `STOP_SWEEP`）.** Same first 16 smoke IDs × 8, seed 42, one knob at a time. Artifacts: `29_gate35b_trial_a_16x8`, `30_gate35b_trial_b_16x8`, `31_gate35b_trial_c_16x8`.

| | A T=0.9 p=.95 | B T=1.1 p=.95 | C T=1.1 p=1.0 |
|---|---:|---:|---:|
| n | 128 | 128 | 128 |
| finish / parse | 1.0 / 1.0 | 1.0 / 1.0 | 0.992 / 1.0 |
| empty think / obs / Continue | 0 | 0 | 0 |
| group reward-std nonzero | 0.75 | 0.81 | 0.94 |
| nonzero advantage | 0.74 | 0.81 | 0.94 |
| route diversity | 0.56 | 0.63 | 0.63 |
| **conditional query div** | **0** | **0** | **0** |
| **exact question copy** | **1.0** | **1.0** | **1.0** |
| unique query / searchable group | 1.0 | 1.0 | 1.0 |
| search_2 | 0 | 0 | 0 |
| branch | TRIAL_B | TRIAL_C | **STOP_SWEEP** |

Raising T/top_p only opened **route** (and reward variance). Search query stayed a **byte-identical question copy**. Do **not** try T≥1.5. Do **not** implement round-specific sampling. Still no SFT retrain.
5. **CPU Query Supervision Audit DONE（2026-08-18 `QUERY_COPY_IS_SUPERVISION_INDUCED`）.** Artifact: `Dee/results/32_audit_sft_query_supervision/`. 4550 mix unchanged (`internal=950`, `search_format=1250`, `evidence=1150`, `evidence_reasoning=1200`). All **1250/1250** `search_format` targets have `search_query == original_question` (`exact_copy_rate=1.0`, `normalized_copy_rate=1.0`, `non_copy_rate=0.0`, length ratio=1.0, `query_source=question_copy`). `query_diversity=0` is **supervision-induced**, not a sampling bug. Stop all temperature / top_p / round-specific sampling work.
6. **Frozen Gate 3.5 verdict: `GATE35_GRPO_READY_WITH_QUERY_LIMITATION`.** Hard keep only: finish≥0.95, parse≥0.95, group reward-std nonzero ≥0.30, nonzero advantage ≥0.20, route diversity observed (32×8 = 0.53). `conditional_query_diversity` and `search_2` are **capability limits**, not GRPO-v1 hard doors. Record them; do not block training.
7. Reward stays `R_answer + 0.5 R_evidence + 0.1 R_format`. `cost λ = 0`. Frozen: SFT ckpt, 4550, Teacher, BM25 top-5, Harness v1, vLLM completions. Gate 4 sampling restores **T=0.7 / top_p=0.95**.

**GRPO v1 may claim:** routing, evidence use, answer.  
**GRPO v1 must not claim:** query reformulation, multi-hop / second search. Those are GRPO-v2 / enhancement.

Project target after later GRPO (not this gate): Answer F1 ≥ 0.70 or Δ ≥ +0.03 vs SFT; Evidence F1 clearly > 0.50; finish ≥ 0.95.

---

## Gate 4 — 1-step Exact GRPO smoke

Must complete the full chain:

```text
question → n=4 on-policy rollout → tool + Candidate-BM25
  → observation → trajectory
  → R_answer + 0.5 R_evidence + 0.1 R_format
  → group-relative advantage → policy loss
  → backward → update_actor → Adam.step()
  → FSDP checkpoint → GRPO_SEGMENT_PASS step=1
```

Missing any link is FAIL. No new cost-aware routing. Sampling for this smoke: **temperature=0.7, top_p=0.95** (restore Gate 3.5 32×8 baseline; do not use Trial C). Judge only `optimizer_step=1` + checkpoint + no NaN/OOM. Ignore F1. Query diversity is not a Gate 4 fail.

**DONE 2026-08-18 `GRPO_SEGMENT_PASS step=1`.** Artifact: `Dee/results/33_gate4_grpo_1step/gate4_summary.json`. Wall 2073s (gen 1817s, update_actor 39s, save 167s). Reward mean/max/min 0.286/1.10/0.10; advantage not all-zero; Exact pearson 0.976; `grad_norm=1.05`; aborted=0. Teardown printed a DataLoader `Killed` then `GRPO_EXIT=0`. Do not read F1.

---

## SGLang Probability Audit (Step A) — PASS

**DONE 2026-08-18 `SGLANG_PROB_AUDIT_PASS`.** Artifact: `Dee/results/34_sglang_prob_audit/sglang_prob_summary.json`.

8 prompts × n=4, `val_only`, no backward, inside historical `eca-verl` (SGLang 0.5.5, TF 4.57.1). Weights = official merge `outputs/22_sft_qwen3_8b_merged` via `model_view` (relative symlinks: 22_ shards + `Dee/model` tokenizer/config). Do not mutate `22_`. Do not edit `07_run_evidence_grpo.sh`.

3B-era “SGLang search-prob=0” **does not hold** on current 8B SFT + Harness v1.

| Metric | Value |
|---|---|
| n | 32 |
| search_rate | 0.375 (12/32) |
| finish | 1.0 |
| missing / nonfinite μ | 0 / 0 |
| mean / median \|Δlogp\| | 0.005554 / 1.2e-5 |
| p95 / p99 / max \|Δlogp\| | 0.023 / 0.127 / 0.49 |
| ρ mean / p01 / max | 0.997 / 0.90 / 1.64 |
| ESS (token) | 4453 / 4457 = **0.999** |

Verdict: μ vs π is a **correctable mismatch** (`SGLANG_MISMATCH_MILD`), not `π/0` support failure. Token-level truncated IS is in-principle valid. Sequence IS is not the first choice. Honest claim if adopted: VeXact = Exact correctness anchor; formal scale = SGLang + explicit correction. Do not call formal SGLang+IS training “Exact”.

---

## Gate 5 — 20-step SGLang + Decoupled Token-TIS

**Locked 2026-08-18.** VeXact stays the Exact correctness anchor. Formal training is **SGLang + official Decoupled Token-TIS**. This is not “speed over correctness”: Gate 4 proved the GRPO chain; Step A showed mild μ/π mismatch; Step B applied official token-TIS (`π_old / π_rollout`) and stayed GREEN. Do not further tune the backend. Launch: `scripts/run_sglang_token_tis_20step.sh`. Smoke parquet is 128 prompts / batch 32 → **4 steps/epoch**; `total_epochs` must be ≥20 so `total_training_steps=20` can finish (same reason `07` uses `total_epochs=800`). Do not edit `07_run_evidence_grpo.sh`.

20-step validates **continuous stability after raising lr 1e-8 → 1e-6**, not F1≥0.70. After PASS: frozen-dev@200 vs SFT, then formal 200→400→600→800 on 5K.

Frozen (do not change any of these in this gate):

```text
policy = 22_ SFT merge
rollout = SGLang async multi-turn
batch=32 n=4  → 128 traj/step
T=0.7 top_p=0.95 top_k=-1
R = EM + 0.5 EvidenceF1 + 0.1 Format   cost λ=0
Harness v1 + Candidate-BM25 top-5   max_search_turns=2
Token-TIS threshold=2.0   RS=OFF   bypass=OFF
lr=1e-6
```

Do **not** change batch / reward / T / BM25 / query; do **not** add RS, bypass, Sequence IS, query-rewrite, or SFT retrain.

Watch four groups. Soft drift is OK. Hard-stop only: nonfinite μ, support loss, ESS collapse, mass clamp, NaN, OOM.

1. Reward trend (not monotonic): `R_answer` not collapse, `R_evidence` up, `R_total` up. SFT baseline Answer F1 0.6649 / Evidence ~0.50 — Evidence is the main headroom.
2. IS health (record every step). Official alarms: ESS<0.3, IS std>1, IS mean<0.5 or >2, |corr KL|>0.1, chi2_token>1. ESS 0.999→0.94 is **not** a stop.
3. Agent behavior: finish/parse, search vs internal, search_2, dup query, length/trunc. search_rate→0 or →1 is collapse. search_2=0 does **not** block GRPO-v1.
4. Throughput: median of **steps 3–20**. Accept ~3–4 min/step. Do not optimize 30s.

After 20 steps, eval **frozen-dev@200** on the verified vLLM deterministic path (not train reward). PASS if no collapse (Answer ~SFT and Evidence/reward not down). Modest Evidence-only gains still continue. Collapse (e.g. Answer 0.665→0.55) stops the line.

**DONE 2026-08-18 `SGLANG_TOKEN_TIS_20STEP_PASS`.** Artifact: `Dee/results/36_sglang_token_tis_20step/gate5_summary.json`. Second launch used `total_epochs=20` (first launch stopped at 4 steps because 128/32=4). Progress bar **20/20 in 45.3 min** (mean 136s/it). ckpt `global_step_10` and `global_step_20`.

| | step 1 | step 20 | 3–20 median |
|---|---:|---:|---:|
| reward mean | 0.281 | **0.378** | last-4 mean **0.400** |
| ESS | 0.9999 | 0.9999 | min **0.99980** |
| IS mean / max | 1.00 / 1.30 | 1.00 / 1.89 | peak IS max=2.0, clamp ≤2e-5 |
| gen / step wall | 46s / 155s | 50s / 171s (incl. save) | **~66s / 136s (2.3 min)** |
| aborted / μ valid | 0 / 1.0 | 0 / 1.0 | always |

Reward on the 128 smoke set rose 0.28→~0.38 and did **not** collapse. IS stayed GREEN vs official alarms (ESS<0.3, IS std>1, |KL|>0.1, chi2>1). Length/turns rose mid-run (len 482→820, turns 5.1→6.0) then eased (672 / 5.55). Entropy 0.023→0.013. This is **5 epochs of the same 128 prompts** — do **not** read F1 from train reward, do **not** call it formal GRPO.

**DONE 2026-08-18 `SMOKE20_FROZEN_DEV200_NO_COLLAPSE`.** Artifact: `Dee/results/38_frozen_dev_grpo_smoke20/smoke20_dev200_summary.json`. Same 200 IDs, Harness v1, Candidate-BM25@5, **vLLM det** (`dee-vllm-grpo20`, T=0). 200/200 in 458s. finish=1.0, parse=1.0, observation mask=1.0.

| Model | Answer F1 | EM | Evidence F1 | search |
|---|---:|---:|---:|---:|
| SFT Agent (Gate 3, **HF**) | 0.6649 | 0.54 | 0.50 | 0.715 |
| GRPO-smoke-20 (vLLM det) | **0.7155** | 0.575 | **0.6141** | 0.855 |

Δ vs Gate 3 SFT: F1 **+0.0506**, EM +0.035, Evidence **+0.114**. `p_search_2` still 0. **NO_COLLAPSE** — this is the only decision this eval is allowed to make.

Do **not** treat 0.7155 as formal Δ_RL: (1) official SFT number is HF, this run is vLLM det; (2) the ckpt saw the same 128 prompts ~5 epochs. **Formal GRPO-v1 restarts from `outputs/22_sft_qwen3_8b_merged`**, not `global_step_20`. Do not retune T / entropy / batch / TIS / reward.

---

## Gate 5.5 — Formal HotpotQA-5K contract

**DONE 2026-08-18 `GATE55_FORMAL_5K_PASS`.** Artifact: `Dee/data/rl/formal_5k/freeze_manifest.json`. Smoke FSDP ckpts deleted (~415G). SFT merged kept.

| Split | n | Overlap |
|---|---:|---|
| Formal train | **5000** unique | frozen-dev 0, sealed 0, SFT **2809** (curriculum, recorded) |
| Formal-dev IDs | 1000 | disjoint from train / frozen-dev / sealed |
| veRL val | 16 | disjoint; index n=5016 |

Policy prompt = system + `Question:` only. Gold answer + supporting facts are reward-only. No extra filtering. Human preview: `data/rl/formal_5k/human_preview_32.jsonl`.

Next: reload BM25 to the 5K index, then Formal-v1 from `outputs/22_sft_qwen3_8b_merged` (not smoke-20). Segment 200 → frozen-dev@200 → 400/600/800. Sealed test once after unique best.

Do **not** treat the 128/16 parquet as the final GRPO train set.
That split exists only to debug leakage, Candidate-BM25, reward, AgentLoop, and Exact GRPO.

| Split | Size | Role |
|---|---:|---|
| SFT coldstart | 4550 | Keep. Teaches protocol, not HotpotQA coverage. |
| GRPO smoke / throughput | 128 / 16 | Keep. Deterministic engineering set. |
| **Formal GRPO train** | **~5000** | New main train. Build only after Gate 5 PASS. |
| Fast frozen-dev | 200 | Every 200-step checkpoint. |
| Formal dev | 1000 | Rank SFT / 400 / 600 / 800 before choosing best. |
| Sealed / final test | 500 now; target 1000–2000+ | Open once after unique best. Report ΔF1 with bootstrap CI. |

Why 5K, not 90K or 128: recent search-agent RL (s3) shows a few thousand examples can train a search policy; 5K is conservative vs 2.4K and ~40× the smoke set. 800 steps × global batch 32 ≈ 25.6K prompt instances → ~5 repeats/question on 5K, vs ~200 repeats on 128.

OOD (2Wiki / MuSiQue / Bamboogle) and Web stay after held-out test. Do not build the 5K set during Gate 2–5.

## Gate 6 — Formal GRPO

Runs on the **Gate 5.5 5K set**, not the 128 smoke set.

```text
200 → fast-dev 200 → 400 → fast-dev 200 → 600 → fast-dev 200 → 800 → fast-dev 200
```

800 is the **maximum budget**, not the model name.
Watch Answer F1 / Evidence F1 / EM up; Finish / Format / tool behavior must not collapse.
Best may be 200, 400, 600, or 800. Confirm on formal-dev 1000 before freeze.

**DONE 2026-08-19 Formal-v1@200 frozen-dev:** `FORMAL_V1_STEP200_FROZEN_DEV200_NO_COLLAPSE`. Artifact: `Dee/results/41_frozen_dev_formal_grpo200/formal200_dev200_summary.json`. vLLM det vs Gate 3 HF: Answer F1 0.6988 vs 0.6649, EM 0.545 vs 0.54, Evidence F1 0.7727 vs 0.50, search 1.0 vs 0.715. Protocol healthy; `p_search_2` still 0.

**DONE 2026-08-19 Formal-v1@400 frozen-dev:** `FORMAL_V1_STEP400_FROZEN_DEV200_NO_COLLAPSE`. Artifact: `Dee/results/45_frozen_dev_formal_grpo400/formal400_dev200_summary.json`. Same vLLM det / same 200 IDs. finish=0.99, parse=1.0, obs mask=1.0. Answer F1 **0.7338**, EM **0.62**, Evidence **0.7477**, search 0.995, **`p_search_2=0.085`**.

Same-backend Δ_RL vs SFT-vLLM 0.6693/0.545/0.4939: F1 **+0.0645**, EM **+0.075**, Evidence **+0.2538**.
vs Formal-v1@200: F1 **+0.035**, EM **+0.075**, Evidence **−0.025**. Current leader = `global_step_400`. **Not unique best yet.**

**400 decision = CONTINUE_TO_600.** Answer was clearly up. Reward / GDPO / GSPO stayed frozen.

**DONE 2026-08-19 Formal-v1@600 frozen-dev:** `FORMAL_V1_STEP600_FROZEN_DEV200_NO_COLLAPSE`. Artifact: `Dee/results/48_frozen_dev_formal_grpo600/formal600_dev200_summary.json`. Same vLLM det / same 200 IDs. finish=0.965, parse=1.0, obs mask=1.0. Answer F1 **0.717**, EM **0.59**, Evidence **0.6343**, search 0.995, `p_search_2=0.07`, generated tokens **618.9**.

vs Formal-v1@400: F1 **−0.0168**, EM **−0.03**, Evidence **−0.1134**, finish **−0.025**. vs SFT-vLLM still +4.77pp F1 / +4.5pp EM / +14.0pp Evidence.

**600 decision = STOP_V1_NO_800.** Not a trainer/TIS/SGLang bug. Reading: 5K × batch 32 reaches ~2.56 epochs at 400 and ~3.84 at 600. Train surrogate (Evidence-driven, `ans_nz=0`) kept optimizing while greedy Answer/Evidence/finish fell and generations doubled (289→619) as entropy collapsed (0.014→0.0085). That is over-optimization / verbosity degeneration on a peaked policy, not a system failure.

**Best Controlled Policy = GRPO step400** after 1000 confirm. Do **not** train 800. Do **not** change reward / GDPO / DAPO on this line.

Formal-v2 is a **side branch after freeze**, not a replacement for v1. Order if opened later: **A unique RL data 5K→10K/20K → B dense Answer reward → C GDPO only if multi-reward still drowns Answer → D Clip-Higher / soft overlong only if 289→619 repeats**. One variable per step. Do not swap GSPO/DPPO/OPO to “fix” this signal.

Main claim for v1: Agentic GRPO improved **autonomous retrieval-grounded answering and supporting-evidence quality**. Do **not** lead with adaptive routing (`search≈0.995` on Hotpot). Do **not** claim multi-hop from `p_search_2=0.085` until a cheap query1→obs→query2 audit after freeze.

---

## Gate 7 — Best Controlled Policy freeze + light audit + held-out test

**CONTROLLED CLOSED 2026-08-20. NOW = Web zero-shot adapter. No more Controlled training.**
Held-out four-arm closure on the same corrected 500 is complete: Direct F1 0.2210, Fixed-RAG 0.3944, SFT 0.6064, GRPO@400 **0.7506**. Same-backend ΔRL held-out = **+0.1442 F1 / +0.138 EM / +0.2591 Evidence F1**. Selection is unchanged.

**DONE 2026-08-20 L3 counterfactual n=54 — `L3_PARTIAL_PASS`:**
mean ΔF1 **+0.354**; helps **22/54 (40.7%)**, hurts **2/54 (3.7%)**.
Strict query rewrite + Obs1-conditioned + new document + ΔF1>0 = **18/54 (33.3%)**.
Obs-conditioned hop mean ΔF1 = **+0.562**. New gold supporting fact + ΔF1>0 = **7/54 (13.0%)**.
Duplicate retry remains **25/54 (46.3%)**. This proves partial causal second-hop ability, not mature fully adaptive DeepResearch.

**DONE 2026-08-20 L3 offline n=54:**
exact dup **25/54 = 46.3%** (all labeled `duplicate_retry`, no new docs).
rewrite **29/54 = 53.7%** → `obs_conditioned_hop` **28**, `rewrite_not_obs_conditioned` **1**.
new_doc 53.7% / new supporting-fact **16.7%** / two-search F1 **0.6043** (harder slice than overall 0.7506).
Heuristic only. Do **not** announce adaptive multi-hop until ΔF1.

**Capability ladder (locked):**
| Level | Meaning | Status |
| L1 | Calls search | DONE |
| L2 | Uses obs/evidence to answer | DONE (main GRPO gain) |
| L3 | After Obs1: stop or write a useful Query2 | **PARTIAL PASS** |
| L4 | Real Web, adaptive depth, stop when enough | **NOT RUN** |

Multi-turn means the policy chooses depth under a budget, not “must search twice.”

**After this audit (locked fork):**
- Real hops (rewrite + obs-conditioned + new evidence + ΔF1>0) → Web zero-shot, raise budget as safety only.
- Mostly duplicate retry → Controlled stays closed; optional MultiTurn-v2 later (plan SFT + IGPO-style turn credit). Do not reopen Formal-v1 / GDPO / DAPO / search-cost.

**DONE 2026-08-20 finalize-v2b held-out 500:** `HELDOUT_GRPO400_DONE`.
`results/51_heldout_test/n500_grpo400_finalize_v2b/.../summary.json`. 1802s.
F1 **0.7506** / EM **0.670** / Ev **0.7243** / Joint **0.5804** / finish **1.0** / `p_search_2=0.108` / gen 312.7.
Unfinished **0/54 search=2 = 0%** (old 22.2%, v1 20.4%). F1 not down vs 0.7493.
This is a **corrected eval**, not a new policy. Still a held-out regression set, not Δ_RL,test.

**DONE 2026-08-20 finalize-v2b n=11 (the v1 unfinished set):**
`search=2` on **11/11**, finish **11/11**, parse/obs-mask 1.0. Duplicate query **0.818**. EM/F1 **0.3636 / 0.4242** (same as v2a 1-search finish — preview: extra hop may not add answer gain). This is the reserved-answer contract, not a policy upgrade.

**DONE 2026-08-20 finalize-fix n=8:** same first 8 sealed IDs as the broken-harness smoke.
Old: F1 0.65 / EM 0.50 / Ev 0.75; item1 `search=2 EM=0`.
New: F1 **0.775** / EM **0.625** / Ev **0.7917**; item1 `search=2 EM=1`. finish=1.0. Protocol healthy. n=8 is not the test score.

**Multi-turn claim (locked wording):**
Level 1 tool re-use is shown (`p_search_2=0.108` on held-out 500). Levels 2–4 (query novelty, obs-conditioned rewrite, stop-when-enough) are **not** established. Say: *GRPO induced second-search execution; adaptive multi-hop is not yet proven.*

**DONE 2026-08-20 finalize-fix v1 n=500:** `FINALIZE_FIX_V1_INSUFFICIENT`.
`results/51_heldout_test/n500_grpo400_finalize_fix/.../summary.json`. 1801s.
F1 **0.7392** / EM **0.660** / Ev 0.7132 / finish 0.978 / `p_search_2=0.108`.
Unfinished **11/54 search=2 = 20.4%** (old 12/54=22.2%). F1 down vs 0.7493.
V1 only added one extra generate; evidence and answer still share 512 tokens.
This 500 is a **held-out regression set**, not a pristine sealed test.

**finalize-v2 (eval only):** after search2, `phase=evidence` (512, stop `</evidence>`), then reserved `phase=answer` (256). No 3rd search. No gold answers.

**Order (locked 2026-08-20):**
```text
held-out@500 raw              DONE (search2 unfinished 22.2%)
        ↓
finalize-v1 + same 500        DONE → V1_INSUFFICIENT (20.4%)
        ↓
finalize-v2b split budgets    DONE on the 11 unfinished IDs
        ↓
rerun @400 same 500           DONE (finish 1.0, search2 unf 0/54, F1 0.7506)
        ↓
MULTITURN_CAPABILITY_AUDIT L3  DONE → PARTIAL PASS
  1 novelty  2 obs1 dependence  3 new evidence  4 forced-1-search ΔF1
        ↓
fork: Web zero-shot   or   optional MultiTurn-v2
        ↓
held-out four arms             ← this step
        ↓
L4 Real-Web adaptive depth (budget is a cap, not a quota)
```

### Gate 9 — Real-Web Zero-Shot + bounded ResearchMemory

**ACTIVE 2026-08-20. Policy remains GRPO@400; no training and no new action tags.**

Execution order is frozen:

```text
provider smoke
  → Web adapter raw zero-shot smoke (`memory_mode=none`)
  → Web Harness v1.1 (`memory_mode=research`, max_search_turns=5 as cap)
  → paired No-Memory vs Research-Memory evaluation
  → L4 decision
```

Web-v1 memory is episode-local and tool-derived only:

- raw observation buffer: temporary; older page text is compacted;
- evidence memory: bounded snippets with evidence ID, title and canonical URL;
- search memory: previous queries, visited URLs, novelty and remaining budget;
- research state: Known evidence / Missing information / Previous searches / Remaining budget.

Never read gold answers, supporting facts or qrels into memory. Do not add a vector DB,
long-term user memory, knowledge graph or multi-agent shared memory. Search budget 5 is a
safety ceiling, not a quota.

Primary paired metrics: duplicate query, duplicate URL, new evidence/search,
observation-conditioned Query2/3, prompt tokens, finish, Answer F1 and evidence quality.
ResearchMemory succeeds only if redundancy/prompt growth fall while Answer and finish do
not regress materially. If zero-shot already shows variable depth, evidence gain and active
stopping, stop optimization. Only persistent repeat/non-planning may open the optional
1–2K MultiTurn-v2 branch.

**IN PROGRESS 2026-08-20 Web smoke n=8:** No-Memory arm completed on frozen-dev's
first 8 IDs: finish/parse/observation-mask **1.0**, Answer F1 **0.2292**, EM **0.125**,
Evidence F1 **0.10**, search exactly **1.0** on every item, latency **205.0s/item**.
Each episode still had **5.75 page-fetch errors** on average, although empty retrieval was
0. This is an infrastructure-qualified smoke, not an L4 score and not comparable to the
held-out-500 Controlled table. ResearchMemory paired arm is still running; no paired
decision until it completes.

**STOPPED 2026-08-20 — `P0_WEB_INFRA_FAIL`:** ResearchMemory arm was manually stopped
before completion. Do not interpret it. The No-Memory diagnostic showed 205s/item and
5.75 failed page fetches/item; therefore Layer 2 (URL→content) is broken enough to confound
Layer 3 (policy). Freeze model and memory work until Web Tool passes.

Web Infra recovery order:

```text
v1.1 per-stage profile (search API / per-URL fetch / extract / model / total)
  → v1.2 Brave LLM Context (pre-extracted snippets, 5 URLs, ~4096 tokens)
  → tool-only same-query A/B, first n=3 then n=20
  → require low errors + nonempty/relevant context + practical latency
  → Agent n=8 protocol smoke
  → Agent n=30–50 depth/memory evaluation
```

Do not add proxy first. If Brave API/LLM Context itself is slow or resets, route only the
Web Tool through a stable proxy/VPS. Never proxy vLLM, local retrieval or training traffic.
Do not reopen Query planning or MultiTurn-v2 until Web Tool passes.

**DONE 2026-08-20 Web Tool A/B n=3 (leak-filtered):** `WEB_TOOL_N3_PASS`.
Brave URL+local-fetch: mean/p50/p95 **23.24/16.47/42.12s**, failed URLs **2.67/q**,
answer-string hit **0/3**, supporting-title recall **0.167**. Brave LLM Context:
mean/p50/p95 **0.65/0.62/0.76s**, failed URLs **0**, nonempty **3/3**,
answer-string hit **3/3**, supporting-title recall **0.667**. Two Hugging Face dataset
mirror URLs were filtered before retrieval. Decision: continue tool-only n=20; do not run
Agent yet.

**DONE 2026-08-20 Web Tool A/B n=20 (same-query, leak-filtered):**
`WEB_TOOL_N20_PASS`. Brave Search + local page fetch mean/p50/p95 latency is
**29.16/26.33/51.09s**, with **3.50 failed URLs/query**, **75%** nonempty context,
answer-string hit **0.15**, and supporting-title recall **0.20**. Brave LLM Context is
**0.507/0.471/0.644s**, with **0 errors**, **100%** nonempty context, answer-string hit
**0.55**, and supporting-title recall **0.60**. Mean latency falls **98.26% (57.5x)**.
Local fetch consumes **97.4%** of the old Tool latency, while the Brave Search API itself
is only 0.73s; therefore the causal bottleneck is arbitrary-page fetching, not model or
search API. Freeze Web-v1 provider to `brave_llm_context`; no proxy is justified now.
This is a Tool-only fixed-slice diagnostic, not an Agent/L4 score. Next gate: frozen
GRPO@400 Agent n=8 protocol smoke; only after health passes run n=30–50.

**DONE 2026-08-20 held-out GRPO@400 n=500:** `HELDOUT_GRPO400_DONE`.
`results/51_heldout_test/n500_grpo400/.../summary.json`. 1789s.
Answer F1 **0.7493** / EM **0.668** / Evidence **0.7132** / Joint **0.5825** / finish **0.976** / search=1.0 / `p_search_2=0.108` / gen **312.5**.
Not Δ_RL,test until SFT is on the same 500.

**Harness finding (from metrics + `react_loop.py`, confirm with audit script):**
12 unfinished, **all search=2**. search=1 unfinished = 0 / 446. search=2 unfinished = 12 / 54 (**22.2%**).
After 2nd obs the loop does **one** generate then `break`. Stop strings are `</search></answer></internal>` — **not** `</evidence>`. Unfinished gens are ~1060–1100 tok (≈ two short searches + one 512-token evidence dump). `hit_max_search_turns` is false on all 12: not a 3rd-search cap, a **missing forced answer phase**.
Do **not** retrain @400. Fix inference loop only, then rerun the same 500.

**DONE 2026-08-20 held-out smoke n=8:** `HELDOUT_SMOKE_PASS`. Same sealed file, vLLM det. finish=1.0 / parse=1.0 / obs-mask=1.0 / F1=0.65 / EM=0.5 / Evidence=0.75 / gen=294 / search=1.0. Protocol healthy. Do **not** treat n=8 F1 as the test score.

**Locked 2026-08-20 — do not chase selective search on this line.**
Always-search is a known limitation of Formal-v1, not the explanation of +6.45pp (65% of Answer F1 is on SFT-already-search items). Do **not** reopen uniform search penalty, static router, confidence gate, or Routing-v2. Those stay an optional **Efficient-Agent-v2 after Web**, only if live latency/cost actually hurts. Do not train. Do not open GDPO / DAPO. Do not retune Formal-v1.

**Best Controlled Policy = GRPO step400** (`results/39_formal_grpo_v1/ckpt/global_step_400`).
`STOP_V1_NO_800` stays. Do not treat “FINAL_POLICY” as a paper term; this is the project best checkpoint.

Next project order (locked):

```text
Best Controlled Policy = GRPO step400
        ↓
light audit (existing metrics only)
  already-search vs always-search
  @400-right / @200-wrong
  @600 length 288→609
        ↓
held-out test once
  Base Direct / Fixed RAG / SFT Agent / GRPO@400
  same split, same vLLM, same Harness v1
  do not pick a checkpoint from test
        ↓
Controlled Agent stage done
        ↓
Web adapter (no train, no new actions)
```

Protocol (locked):

```text
SFT-vLLM + Formal@200 + Formal@400 + Formal@600
same 1000 IDs (data/eval/hotpotqa_formal_dev_1000.jsonl)
same vLLM det (T=0)
same Harness v1 / Candidate-BM25@5 / max_search_turns=2
```

Primary selection metric stays **Answer F1**. Do not change it after seeing numbers.

**DONE 2026-08-19 Formal@400 @1000:** `FORMAL_V1_STEP400_FORMAL_DEV1000_SCORED`.
Artifact: `Dee/results/49_formal_dev1000/formal400_dev1000_summary.json`.
Answer F1 **0.816** / EM 0.75 / Evidence 0.7636 / Joint F1 0.6634 / finish 0.988 / gen **288.0** / `p_search_2=0.071`.
The important fact is **policy-shape stability vs frozen-dev@200** (search=1, gen≈288, search₂≈0.07, finish≈0.99) — not the 0.816 absolute. Do **not** replace official Δ_RL (still frozen-dev@200: +6.45pp F1). This 1000 is a train-pool confirm split.

**DONE 2026-08-19 SFT-vLLM @1000:** `SFT_VLLM_FORMAL_DEV1000_SCORED`.
Artifact: `Dee/results/49_formal_dev1000/sft_dev1000_summary.json`.
Answer F1 **0.6871** / EM 0.623 / Evidence 0.4575 / Joint 0.3843 / finish 1.0 / search **0.628** / internal 0.372 / gen 168.2.
Same-split vs @400: F1 **+0.1289**, EM +0.127, Evidence +0.306, Joint +0.279. Confirm-only; official Δ_RL unchanged.

**DONE 2026-08-19 Formal@200 @1000:** `FORMAL_V1_STEP200_FORMAL_DEV1000_SCORED`.
Artifact: `Dee/results/49_formal_dev1000/formal200_dev1000_summary.json`.
Answer F1 **0.808** / EM 0.732 / Evidence **0.7948** / Joint **0.6762** / finish 1.0 / search=1.0 / `p_search_2=0` / gen 225.3.
Same-split rank so far by Answer F1: **@400 0.816 > @200 0.808 > SFT 0.6871**. @200 still wins Evidence/Joint (same pattern as frozen-dev@200). Do not change primary metric.

**DONE 2026-08-20 Formal@600 @1000:** `FORMAL_V1_STEP600_FORMAL_DEV1000_SCORED`.
Artifact: `Dee/results/49_formal_dev1000/formal600_dev1000_summary.json`.
Answer F1 **0.7985** / EM 0.729 / Evidence 0.619 / Joint 0.5253 / finish 0.984 / search=1.0 / `p_search_2=0.054` / gen **609**.
vs @400: F1 **−0.0175**, EM −0.021, Evidence **−0.145**, Joint **−0.138**, gen 288→609.

**1000 decision = CASE_A, now frozen.** Answer F1 rank: **@400 0.816 > @200 0.808 > @600 0.7985 > SFT 0.6871**. No split reversal. **Best Controlled Policy = GRPO step400.**

Report also (secondary, not for ranking):

```text
Answer EM
Evidence / supporting-fact F1
Evidence EM
Joint F1
Joint EM
finish / parse / obs-mask / gen tokens
```

Joint uses Hotpot official form: `P_joint = P_ans * P_sp`, `R_joint = R_ans * R_sp`, then F1; `EM_joint = EM_ans * EM_sp`. Report-only.

Health gate first: no format collapse, no broken tools. Then rank:

1. Answer F1
2. Evidence F1 if Answer F1 is tied
3. More stable Finish / tool behavior
4. Earlier checkpoint if still tied

@200 = strongest grounding so far; @400 = strongest final answer. That is expected. Do not swap primary to Joint/Evidence.

**@600@1000 decision (locked before seeing the number):**

- **A — @400 still Answer F1 #1:** freeze **Best Controlled Policy = GRPO step400**. No more Formal-v1 train.
- **B — @600 ≈ @400 (no clear win):** still pick **@400**.
- **C — @600 clearly beats @400:** do **not** auto-switch to 600, do **not** train 800. (Did not happen.)

Observed: **A**. Held-out test is one final unread split after the audit, not a paper preregistration ritual. Run four arms **once**:

```text
split: data/sealed/hotpotqa_test500.jsonl   (n=500, first open)
Base Direct | Fixed RAG     = HF greedy, scripts/run_controlled_baseline.py (Gate 3 protocol)
SFT Agent   | GRPO@400      = vLLM det + Harness v1 (same as official Δ_RL)
smoke first: scripts/run_heldout_smoke.sh   (GRPO@400 n=8)
then each arm once at n=500. Never pick a checkpoint from test.
```

Δ_RL,test = F1_GRPO400 − F1_SFT. **Never pick a checkpoint from test.**

After that: **Web zero-shot adapter, no train, no new policy actions.** Formal-v2 (dense Answer → GDPO / DAPO) and Efficient-Agent-v2 (OTC-style minimal-search) stay optional and off the main line. Do not block Web on “learn when not to search”.

**Gate 7.5 light audit — DONE 2026-08-20.** `AUDIT_SEARCH_GAIN`. Artifact: `results/50_audit_search_gain/audit_summary.json`. CPU join, no new eval.

Official read is **frozen-dev@200** (n=200, ΔF1 **+0.0645** matches @400 vs SFT):

| Stratum | n | share | SFT F1 | @400 F1 | ΔF1 | Δ Evidence | F1 contrib |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT already search | 142 | 0.710 | 0.6871 | 0.7460 | **+0.0589** | +0.0511 | **+0.0418 (65%)** |
| SFT internal → @400 search | 57 | 0.285 | 0.6193 | 0.6985 | +0.0792 | +0.7636 | +0.0226 (35%) |
| still internal | 1 | — | — | — | — | — | — |

Read: **+6.45pp is not only always-search.** Most official Answer-F1 points come from items SFT already searched (+5.9pp on that 71%). The big Evidence jump (+25pp overall) is mostly the 57 flip items (Evidence 0 → 0.76; SFT internal has no `<evidence>`). formal-dev@1000 agrees on already-search ΔF1 **+0.0500**, but that split has more SFT-internal (37%) so flip looks larger — do not replace official Δ_RL with 0.1289.

@400 vs @200: 24 / 17 / 159 tie on val@200; 71 / 62 / 867 on 1000. Same policy family; @400 is a small Answer-F1 edge, not a rewrite.

@600 length: gen 289→619; **77% verbose_single_search**, 7% second-search, 0% duplicate query. Degeneration is rambling, not extra hops.

Next: held-out test once (Direct / Fixed-RAG / SFT / GRPO@400). Do not train.

---

## How we prove GRPO helped

Interview story is the four-arm ladder: **Direct → Fixed RAG → SFT Agent → GRPO@400**.
Main numeric claim is still **Δ_RL = Metric_GRPO − Metric_SFT** on the same backend. The audit shows this is **not** only always-search: 65% of official +6.45pp F1 is on SFT-already-search items. Do not lead with GRPO vs Base.

| Model | Answer F1 | EM | Evidence F1 | Finish | Search/Tool |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Direct | 0.2647 | 0.19 | – | – | none |
| Qwen3-8B + RAG | 0.6659 | 0.57 | – | – | fixed |
| Qwen3-8B SFT Agent | **0.6649** | 0.54 | **0.50** | 1.0 | autonomous |
| GRPO-smoke-20 *(diagnostic)* | 0.7155 | 0.575 | 0.614 | 1.0 | autonomous |
| Formal-v1@200 (vLLM) | 0.6988 | 0.545 | 0.7727 | 1.0 | autonomous |
| **Formal-v1@400 (vLLM)** | **0.7338** | **0.62** | 0.7477 | 0.99 | autonomous |
| Formal-v1@600 (vLLM) | 0.717 | 0.59 | 0.6343 | 0.965 | autonomous |

Base→SFT = learned to be an Agent.
SFT→GRPO = on-policy Agentic RL improved retrieval-grounded answering and evidence quality.
Do not lead with “learned when not to search”: @400 search≈0.995 on this Hotpot setup.

GRPO-smoke-20 is a **collapse check only** (128-set ~5 epoch ckpt, vLLM vs Gate 3 HF). Formal Δ_RL comes from Formal GRPO-v1 restarted at SFT merged. Do not fill that row from `global_step_20`.

---

## Milestone B — Web (after freeze only)

Same policy. Tool backend changes from Candidate-BM25 to Live Web.

```text
User Question → frozen GRPO policy
  → <search>query</search>
  → tool-internal Web search → fetch → clean/chunk → rank/dedupe/truncate
  → <tool_response>evidence chunks + URLs</tool_response>
  → reason + cite → answer
```

First Web version keeps the **single existing policy action** `<search>`. OPEN/FIND are
tool-internal operations, not new model actions. No retraining and no multi-agent
browser/planner/writer/critic stack.

**Web adapter status 2026-08-20:** implementation + offline fixture PASS. Supports
DuckDuckGo HTML, Brave API, and SearXNG; includes public-URL checks, timeout,
bounded fetch, visible-text extraction, relevant-chunk ranking, dedupe, and cache.
Host TLS to DuckDuckGo is reset, so live smoke awaits `BRAVE_SEARCH_API_KEY` or
`SEARXNG_URL`; this is the current external-provider gate, not a policy blocker.

Harness additions (not new algorithms): Evidence Workspace, context budget, cache/retry/timeout, Streamlit trace viewer.

Provider is **not** frozen today. Re-check API / stability / price when Milestone B starts.

### Three eval layers (final)

1. Controlled RL — sealed HotpotQA Test: did RL help?
2. Controlled DeepResearch — fixed corpus / qrels: harder than HotpotQA?
3. Live Web — does the BM25-trained search policy transfer?

Report Answer, Retrieval, Evidence, Agent Reliability, and Efficiency. Not Answer F1 alone.

---

## 30B closed — do not reopen

| ID | Fact | Lesson |
|---|---|---|
| F0 | `lr=0` → VeOmni cosine `ZeroDivisionError` | smoke `lr=1e-8` |
| F1 | HF `max_position_embeddings=262144` → VeXact KV OOM | `max_model_len=8192` |
| F2 | PP=1 hybrid: Actor + VeXact on one 80G | `PP=4` |
| F3 | first Adam `exp_avg` → Actor 74.80/79.25 GiB | MoE A3B does not shrink optimizer |
| F4 | FSDP CPU offload: update ~36 min, HF gather host-RAM OOM | 8B keeps GPU Adam; HF export only at eval steps |

Official VeXact 30B recipe wants 8 GPUs. This machine has 4 usable 80G cards.
Do not retry 30B. Do not pick 14B or Qwen3.5-9B as the primary line.

---

## Current freeze (this document)

```text
Qwen3-8B weights: READY (Dee/model)
algorithm / data / reward / AgentLoop / Candidate-BM25 / Exact VeXact: FROZEN
Best Controlled Policy: GRPO step400
Web: ACTIVE — zero-shot adapter, same frozen GRPO@400 policy, no new actions
30B project tree: DELETE (not a runtime source)
```

Gate 4 **`GRPO_SEGMENT_PASS step=1`**. Step A **`SGLANG_PROB_AUDIT_PASS`**.

**Formal path (locked 2026-08-18):** VeXact is the **Exact correctness anchor only**. Do **not** run VeXact 20-step / 200–800. Formal candidate is **SGLang + official Decoupled Token-TIS**. IS does not “turn μ back into π”; it reweights SGLang trajectories so gradients estimate updates under π. Reward / group advantage / BM25 / Harness / 4550 stay frozen. Do not edit `07_run_evidence_grpo.sh`. Do not claim formal SGLang+TIS is Exact. Do not enable SGLang deterministic inference or upgrade 0.5.5 in this step.

```text
Gate 4 Exact VeXact 1-step     PASS
Step A SGLang μ/π audit        PASS   (search=0.375, ESS=0.999)
Step B SGLang + Token-TIS 1-step   PASS
Gate 5 SGLang+TIS 20-step      PASS
Formal 200→600 (no 800)        DONE; Best Controlled Policy = step400
```

---

## Step B — 1-step SGLang + Decoupled Token-TIS

Apples-to-apples vs Gate 4. Only two algorithm changes: rollout `VeXact → SGLang`, correction `OFF → Decoupled Token-TIS`.

Frozen with Gate 4: SFT `22_` merge, batch 32, n 4, T=0.7, top_p=0.95, Harness v1, BM25 top-5, `R = EM + 0.5 Evidence + 0.1 Format`, clip 0.2, actor KL 0.001, smoke `lr=1e-8`. Do not drop batch. Do not add RS. Do not set `bypass_mode=true` (old-logprob was 11s; the 1817s rollout is the bottleneck).

Official keys (verl docs):

```text
algorithm.rollout_correction.rollout_is=token
algorithm.rollout_correction.rollout_is_threshold=2.0
algorithm.rollout_correction.rollout_rs=null
algorithm.rollout_correction.bypass_mode=false
actor_rollout_ref.rollout.calculate_log_probs=true
```

`eca-verl` already contains official files (`verl/trainer/config/algorithm.py`, `rollout_corr_helper.py`, tests). Launch: `scripts/run_sglang_token_tis_1step.sh`. Confirm `decoupled_token_is()` imports before starting the 1-step. Do not hand-write IS loss.

**DONE 2026-08-18 `SGLANG_TOKEN_TIS_1STEP_PASS`.** Artifact: `Dee/results/35_sglang_token_tis_1step/stepb_summary.json`. Launch: `scripts/run_sglang_token_tis_1step.sh`.

| Metric | Gate 4 Exact | Step B SGLang+TIS |
|---|---:|---:|
| step wall | 2073s | **222s (9.3×)** |
| gen | 1817s | **54s** |
| old_log_prob / ref / update | 11 / 10 / 39s | 6 / 52 / 22s |
| reward mean / max / min | 0.286 / 1.10 / 0.10 | 0.281 / 1.10 / 0.10 |
| IS mean / ESS / clamp | n/a (Exact) | **0.9999 / 0.9999 / 0** |
| IS max | n/a | 1.27 (threshold 2.0) |
| pearson / μ valid | 0.976 / 1.0 | 0.998 / 1.0 |
| grad_norm / aborted | 1.05 / 0 | 0.59 / 0 |

GREEN: optimizer_step=1, ckpt `global_step_1`, 100% finite μ, IS mean≈1, ESS≥0.90, clamp=0, no toxic tail, no NaN/OOM. Speed far above 2×. Do **not** call this Exact.

Gate 5 is now the only main task: 20-step at `lr=1e-6`, still 32×4. No new backend audits. Watch reward / evidence / search-rate / ESS drift; do not stop for mild ESS drift. Then frozen-dev@200 vs SFT 0.6649 / Evidence ~0.50 — do not require F1>0.70 at 20 steps; require no collapse. Formal claim: **Exact-validated, rollout-corrected Agentic GRPO**.
