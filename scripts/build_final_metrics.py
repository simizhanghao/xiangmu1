#!/usr/bin/env python3
"""Freeze the final, source-linked project metrics artifact and Markdown table."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/final_release"


def load(path):
    return json.loads((ROOT / path).read_text())


def main():
    heldout_path = "results/53_heldout_four_arm/summary.json"
    heldout = load(heldout_path)
    web_none_path = "results/54_web_zero_shot/none/agent_rollout_n40_20260820_194942_web_brave_llm_context_none_n40/summary.json"
    web_memory_path = "results/54_web_zero_shot/research/agent_rollout_n40_20260820_195153_web_brave_llm_context_research_n40/summary.json"
    capability_path = "results/52_multiturn_capability/n54_counterfactual/summary.json"
    adaptive_sources = {
        "WebMT-v2": "results/66_webmt_behavior_v2/webmt_lora_v2/summary.json",
        "Atomic Controller": "results/67_w4_atomic_decision/behavior_gate_v2_sanitized/summary.json",
        "CoC/DPO": "results/70_w4_coc_eval/behavior_gate_hf/summary.json",
        "W5 Controller": "results/78_w5_controller_offline/summary.json",
        "W5.5 Linear Probe": "results/79_w55_linear_probe/summary.json",
        "W6 Decision-only": "results/82_w6_stage1_offline/summary.json",
        "W7 Structured Gap": "results/85_w7_s2g_offline/summary.json",
    }
    controlled = {}
    for name, row in heldout["arms"].items():
        controlled[name] = {
            key: row.get(key)
            for key in (
                "num_samples", "mean_em", "mean_token_f1", "mean_evidence_f1",
                "mean_joint_f1", "mean_search_count", "finish_rate", "parse_ok_rate",
                "mean_latency_ms",
            )
        }
    web = {}
    for name, path in (("No Memory", web_none_path), ("ResearchMemory", web_memory_path)):
        row = load(path)
        web[name] = {
            key: row.get(key)
            for key in (
                "num_samples", "mean_em", "mean_token_f1", "mean_evidence_f1",
                "mean_search_count", "finish_rate", "parse_ok_rate", "mean_latency_ms",
                "p_search_1", "p_search_2", "p_search_ge3", "mean_new_evidence_per_search",
            )
        }
    adaptive = {}
    for name, path in adaptive_sources.items():
        row = load(path)
        adaptive[name] = {
            "gate": row.get("gate"),
            "auroc": row.get("auroc", row.get("auroc_diagnostic")),
            "stop_recall": row.get("stop_recall", row.get("stop_at_d1")),
            "continue_recall": row.get("continue_recall", row.get("continue_at_d2")),
            "balanced_accuracy": row.get("balanced_accuracy"),
            "parse_valid_rate": row.get("parse_valid_rate"),
            "finish_rate": row.get("finish_rate"),
            "source": path,
        }
    capability = load(capability_path)
    payload = {
        "status": "FROZEN_FINAL",
        "best_policy": "GRPO@400",
        "deployment": "GRPO@400 + Web Search + provenance-only state; prompt memory and adaptive Controller disabled",
        "deployment_contract": {
            "policy_prompt_memory": False,
            "provenance_state": True,
            "adaptive_controller": False,
        },
        "controlled_heldout500": controlled,
        "controlled_rl_delta_f1_vs_sft": heldout["delta_rl_heldout_f1"],
        "multiturn_capability": {
            key: capability.get(key)
            for key in (
                "n_search2", "obs_conditioned_rate", "new_doc_rate", "new_sf_rate",
                "mean_two_search_f1", "mean_forced1_f1", "mean_delta_f1_search2",
                "share_search2_helps", "share_search2_hurts",
            )
        },
        "real_web_n40": web,
        "real_web_n40_provider": "brave_llm_context",
        "adaptive_depth_study": adaptive,
        "claims": {
            "controlled_agentic_rl": "PASS",
            "real_web_zero_shot_transfer": "PASS",
            "observation_conditioned_second_hop": "PARTIAL_PASS",
            "deployable_adaptive_depth_controller": "FAIL_CLOSED",
        },
        "sources": {
            "controlled": heldout_path,
            "multiturn": capability_path,
            "web_no_memory": web_none_path,
            "web_research_memory": web_memory_path,
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    (ROOT / "config/final_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Final Frozen Metrics", "", "## Controlled held-out 500", "",
        "| Arm | Answer F1 | EM | Evidence F1 | Search | Finish |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in controlled.items():
        lines.append(f"| {name} | {row['mean_token_f1']:.4f} | {row['mean_em']:.3f} | {(row['mean_evidence_f1'] or 0):.4f} | {row['mean_search_count']:.3f} | {row['finish_rate']:.3f} |")
    lines += [
        "", "GRPO@400 improves Answer F1 over SFT by **14.42 percentage points**.",
        "", "## Real-Web zero-shot n=40", "",
        "Provider for this frozen evaluation: `brave_llm_context`. Final serving uses Bocha; these metrics are not relabeled as a Bocha benchmark.", "",
        "| Memory | Answer F1 | EM | Evidence F1 | Search | Finish | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in web.items():
        lines.append(f"| {name} | {row['mean_token_f1']:.4f} | {row['mean_em']:.3f} | {row['mean_evidence_f1']:.4f} | {row['mean_search_count']:.3f} | {row['finish_rate']:.3f} | {row['mean_latency_ms']:.1f} |")
    lines += [
        "", "## Adaptive-depth study", "",
        "| Method | AUROC | STOP/D1 | CONTINUE/D2 | Balanced Acc | Gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in adaptive.items():
        value = lambda key: "—" if row[key] is None else f"{row[key]:.4f}"
        lines.append(f"| {name} | {value('auroc')} | {value('stop_recall')} | {value('continue_recall')} | {value('balanced_accuracy')} | {row['gate']} |")
    lines += [
        "", "## Frozen conclusion", "",
        "Controlled Agentic RL and Real-Web transfer pass. Observation-conditioned second-hop retrieval is a partial pass. The deployable adaptive-depth Controller is a terminal fail and is not part of the final serving path.", "",
    ]
    (ROOT / "docs/FINAL_METRICS.md").write_text("\n".join(lines))
    print(json.dumps({"gate": "FINAL_METRICS_FROZEN", "output": str(OUT / 'metrics.json')}, indent=2))


if __name__ == "__main__":
    main()
