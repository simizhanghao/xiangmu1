# v1.0.0

Final stable deployment:

- Qwen3-8B GRPO@400 frozen policy
- Bocha real-Web search adapter
- grounded Evidence, Answer and source provenance
- interactive CLI, cached offline mock and FastAPI service
- frozen metrics/manifests and offline release smoke

Controlled held-out 500: Answer F1 **0.7506**, EM **0.670**, Evidence F1
**0.7243**, finish **1.000**. Adaptive-depth Controller research is preserved but
not deployed because no variant passed the pre-registered frozen natural-state gate.
