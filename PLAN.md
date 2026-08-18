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
[6] GRPO 200 → 400 → 600 → 800 (5K train)
        │
        ▼
[7] Frozen-dev Unique Best
        │
        ▼
[8] Sealed Test
        │
        ▼
====== FINAL POLICY FREEZE ======
        │
        ▼
[9] Live-Web Agent Harness
        │
        ▼
[10] Controlled + Web Evaluation
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
  → GRPO Smoke → Throughput → 200/400/600/800
  → unique best → sealed Test → FINAL POLICY FREEZE
```

### Milestone B — Same policy, Live Web

```text
Policy Freeze → SEARCH/OPEN/FIND → Evidence Workspace
  → Context Management → Citation → Trace Viewer
  → Web Eval → README / Demo → STOP
```

Web starts only after unique best + sealed Test + Final Policy Freeze.

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
| Web | PAUSED until Final Policy Freeze |
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
| **6 GRPO** | 200→400→600→800 on **5K**, fast-dev 200 each | RL improves the Agent | Gate 7 |
| **7 Selection** | Health + fast-dev, then formal-dev 1000 → unique best → sealed Test | Final Agent Policy | OOD / Web |

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

Next: Gate 5.5 freeze ~5K RL train (question-only to policy; gold only for reward). Then Formal-v1 200→400→600→800 with frozen-dev@200 at each milestone. Sealed test once after unique best.

---

## Gate 5.5 — Formal HotpotQA-5K contract (after smoke+throughput)

Do **not** treat the current 128/16 parquet as the final GRPO train set.
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

OOD (2Wiki / MuSiQue / Bamboogle) and Web stay after Final Policy Freeze. Do not build the 5K set during Gate 2–5.

## Gate 6 — Formal GRPO

Runs on the **Gate 5.5 5K set**, not the 128 smoke set.

```text
200 → fast-dev 200 → 400 → fast-dev 200 → 600 → fast-dev 200 → 800 → fast-dev 200
```

800 is the **maximum budget**, not the model name.
Watch Answer F1 / Evidence F1 / EM up; Finish / Format / tool behavior must not collapse.
Best may be 200, 400, 600, or 800. Confirm on formal-dev 1000 before freeze.

---

## Gate 7 — Unique best, then sealed Test

Health gate first: no Finish regression, no format collapse, no broken tools.

Then rank:

1. Answer F1
2. Evidence F1 if Answer F1 is tied
3. More stable Finish / tool behavior
4. Earlier checkpoint if still tied

Then write `FINAL_POLICY` and open sealed Test **once**.
Test answers generalization of the frozen model. It does not choose 600 vs 800.

---

## How we prove GRPO helped

Main claim is **Δ_RL = Metric_GRPO − Metric_SFT**, not GRPO vs Base.

| Model | Answer F1 | EM | Evidence F1 | Finish | Search/Tool |
|---|---:|---:|---:|---:|---:|
| Qwen3-8B Direct | 0.2647 | 0.19 | – | – | none |
| Qwen3-8B + RAG | 0.6659 | 0.57 | – | – | fixed |
| Qwen3-8B SFT Agent | **0.6649** | 0.54 | **0.50** | 1.0 | autonomous |
| GRPO-smoke-20 *(diagnostic)* | 0.7155 | 0.575 | 0.614 | 1.0 | autonomous |
| **Qwen3-8B Formal GRPO** | | | | | |

Base→SFT = learned to be an Agent.
SFT→GRPO = on-policy Agentic RL improved the policy.

GRPO-smoke-20 is a **collapse check only** (128-set ~5 epoch ckpt, vLLM vs Gate 3 HF). Formal Δ_RL comes from Formal GRPO-v1 restarted at SFT merged. Do not fill that row from `global_step_20`.

---

## Milestone B — Web (after freeze only)

Same policy. Tool backend changes from Candidate-BM25 to Live Web.

```text
User Question → frozen GRPO policy
  → SEARCH → result list
  → OPEN → clean / chunk
  → FIND → Evidence Workspace
  → reason + cite → answer
```

First Web version has **three actions only**: SEARCH / OPEN / FIND.
No multi-agent browser/planner/writer/critic stack.

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
Web: PAUSED
30B project tree: DELETE (not a runtime source)
```

Gate 4 **`GRPO_SEGMENT_PASS step=1`**. Step A **`SGLANG_PROB_AUDIT_PASS`**.

**Formal path (locked 2026-08-18):** VeXact is the **Exact correctness anchor only**. Do **not** run VeXact 20-step / 200–800. Formal candidate is **SGLang + official Decoupled Token-TIS**. IS does not “turn μ back into π”; it reweights SGLang trajectories so gradients estimate updates under π. Reward / group advantage / BM25 / Harness / 4550 stay frozen. Do not edit `07_run_evidence_grpo.sh`. Do not claim formal SGLang+TIS is Exact. Do not enable SGLang deterministic inference or upgrade 0.5.5 in this step.

```text
Gate 4 Exact VeXact 1-step     PASS
Step A SGLang μ/π audit        PASS   (search=0.375, ESS=0.999)
Step B SGLang + Token-TIS 1-step   PASS
Gate 5 SGLang+TIS 20-step      PASS
Formal 200→800                 after frozen-dev@200 + 5K
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
