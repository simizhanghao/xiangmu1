# Adaptive Retrieval Research Branch

This branch is retained as a negative scientific result, not shipped behavior.

| Stage | Finding |
|---|---|
| W2 ResearchMemory | OOD prompt changed behavior; F1/finish did not improve |
| WebMT-v2 | Better query generation, poor STOP/CONTINUE balance |
| Atomic + DPO | Preference optimization did not improve frozen behavior |
| W5 controller | BA 0.7530; frozen gate fail |
| W5.5 linear probe | BA 0.7754; best result, still below gate |
| W6 decision-only SFT | BA 0.7472; no improvement |
| W7 structured-gap judge | BA 0.6231; clear regression |

Conclusion: the experiments detect useful latent signal but do not establish a
deployable static supervised stopping controller under natural on-policy state shift.
Code and reports are preserved; the final CLI/API intentionally excludes the module.

See `docs/W5_MODULAR_CONTROLLER_REPORT.md`,
`docs/W6_MULTISTAGE_CONTROLLER_REPORT.md`, and `docs/FINAL_METRICS.md`.
