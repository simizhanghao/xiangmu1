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

Missing any link is FAIL. No new cost-aware routing.

---

## Gate 5 — 20-step throughput

`1-step` = correctness. `20-step` = engineering feasibility.
Ignore step 1–2 (load / compile / Ray / VeXact warmup). Use **median step wall time** after warmup.

| Median min/step | Color | Action |
|---|---|---|
| `<= 3.2` | GREEN | Approve 200→400→600→800 |
| `3.2–3.6` | YELLOW | Do not promise 800 in 48h; deliver 600 first |
| `> 3.6` | RED | Stop betting 8B throughput; Qwen3-4B fallback |

Do not reopen “maybe it will get faster if we keep tuning.”

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
| Qwen3-8B Direct | | | – | | 0 |
| Qwen3-8B + RAG | | | – | | 1 |
| Qwen3-8B SFT Agent | | | | | |
| **Qwen3-8B Evidence-GRPO** | | | | | |

Base→SFT = learned to be an Agent.
SFT→GRPO = on-policy Agentic RL improved the policy.

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

Next authorized action after this plan: **Gate 0 preflight only**. No SFT until Gate 0 PASS.
Then Gate 1 compatibility. Never skip ahead.
