"""Configurable reward components for Phase 3C/3D (weights only; no trainer coupling)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class RewardWeights:
    answer_weight: float = 1.0
    evidence_weight: float = 0.5
    format_weight: float = 0.1
    search_cost_weight: float = 0.0  # 3D
    duplicate_weight: float = 0.0  # 3D


@dataclass
class RewardBreakdown:
    answer_reward: float = 0.0
    evidence_reward: float = 0.0
    format_reward: float = 0.0
    cost_reward: float = 0.0
    duplicate_penalty: float = 0.0
    total: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def weights_from_mapping(m: Optional[Mapping[str, Any]] = None) -> RewardWeights:
    m = m or {}
    return RewardWeights(
        answer_weight=float(m.get("answer_weight", 1.0)),
        evidence_weight=float(m.get("evidence_weight", 0.5)),
        format_weight=float(m.get("format_weight", 0.1)),
        search_cost_weight=float(m.get("search_cost_weight", 0.0)),
        duplicate_weight=float(m.get("duplicate_weight", 0.0)),
    )


def combine_rewards(
    *,
    answer: float,
    evidence: float,
    format_r: float,
    cost: float = 0.0,
    duplicate: float = 0.0,
    weights: Optional[RewardWeights] = None,
) -> RewardBreakdown:
    w = weights or RewardWeights()
    total = (
        w.answer_weight * answer
        + w.evidence_weight * evidence
        + w.format_weight * format_r
        - w.search_cost_weight * cost
        - w.duplicate_weight * duplicate
    )
    return RewardBreakdown(
        answer_reward=float(answer),
        evidence_reward=float(evidence),
        format_reward=float(format_r),
        cost_reward=float(cost),
        duplicate_penalty=float(duplicate),
        total=float(total),
    )
