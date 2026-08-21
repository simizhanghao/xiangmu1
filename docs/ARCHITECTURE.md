# Final Architecture

## Production path

```text
User Question
    │
    ▼
Frozen Qwen3-8B GRPO@400 policy
    │  <search>query</search>
    ▼
Bocha Web Search → normalized tool_response
    │
    ▼
Evidence selection → grounded Answer → Sources
```

The serving path reuses the frozen Harness v1 AgentLoop. Application state records
queries, retrieved documents and URLs for provenance, but is **not injected into the
policy prompt**. This preserves the input distribution validated by the no-memory W2
arm, which outperformed ResearchMemory on Answer F1 and finish rate.

## Research-only path

```text
ResearchMemory → WebMT SFT → Atomic/DPO → Modular Controller
               → Linear probe → Decision-only SFT → Structured-gap judge
               → frozen natural-state gate FAIL
```

W3–W7 remain reproducible research artifacts. They are not imported by `src/app`, and
`adaptive_controller.enabled=false` is recorded in the final manifest.
