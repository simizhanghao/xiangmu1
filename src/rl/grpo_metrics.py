"""Formal GRPO logs: answer EM / evidence F1 / format / total. No reward change."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def _as_float(x: Any, default: float = 0.0) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _reward_info(extra: Mapping[str, Any]) -> Mapping[str, Any]:
    info = extra.get("reward_extra_info") if isinstance(extra, Mapping) else None
    return info if isinstance(info, Mapping) else {}


def _agent_metrics(extra: Mapping[str, Any]) -> Mapping[str, Any]:
    m = extra.get("metrics") if isinstance(extra, Mapping) else None
    return m if isinstance(m, Mapping) else {}


def compute_grpo_batch_metrics(
    extra_fields: Sequence[Mapping[str, Any]],
    *,
    uids: Optional[Sequence[Any]] = None,
    sequence_scores: Optional[Sequence[float]] = None,
    zero_std_eps: float = 1e-6,
) -> Dict[str, float]:
    n = len(extra_fields)
    if n == 0:
        return {}

    answer, evidence, fmt, total = [], [], [], []
    finish, search_counts, search_route, internal = [], [], [], []
    gen_len: List[float] = []

    for extra in extra_fields:
        extra = extra or {}
        rinfo = _reward_info(extra)
        am = _agent_metrics(extra)

        em = rinfo.get("answer_em", rinfo.get("em", rinfo.get("answer_reward")))
        ev = rinfo.get("evidence_f1", rinfo.get("evidence_reward"))
        fo = rinfo.get("format_reward", rinfo.get("format"))
        tot = rinfo.get("total_reward", rinfo.get("score"))
        if em is not None:
            answer.append(_as_float(em))
        if ev is not None:
            evidence.append(_as_float(ev))
        if fo is not None:
            fmt.append(_as_float(fo))
        if tot is not None:
            total.append(_as_float(tot))

        fin = extra.get("finish", am.get("finish", 0))
        finish.append(1.0 if _as_float(fin) >= 0.5 else 0.0)
        sc = _as_float(extra.get("search_count", am.get("search_count", rinfo.get("search_count", 0))))
        search_counts.append(sc)
        if sc > 0:
            search_route.append(1.0)
            internal.append(0.0)
        else:
            search_route.append(0.0)
            internal.append(1.0 if _as_float(am.get("used_internal", extra.get("used_internal", 0))) >= 0.5 else 0.0)
        for key in ("response_length", "generated_tokens", "mean_generated_tokens"):
            if key in extra or key in am:
                gen_len.append(_as_float(extra.get(key, am.get(key, 0))))
                break

    out: Dict[str, float] = {}

    def _mean_std(name: str, vals: List[float]) -> None:
        if not vals:
            return
        arr = np.asarray(vals, dtype=np.float64)
        out[f"{name}/mean"] = float(arr.mean())
        out[f"{name}/std"] = float(arr.std())
        out[f"{name}/max"] = float(arr.max())
        out[f"{name}/min"] = float(arr.min())

    _mean_std("reward/answer_em", answer)
    _mean_std("reward/evidence_f1", evidence)
    _mean_std("reward/format", fmt)
    _mean_std("reward/total", total)
    if finish:
        out["agent/finish_rate"] = float(np.mean(finish))
    _mean_std("agent/search_count", search_counts)
    if search_route:
        out["agent/search_rate"] = float(np.mean(search_route))
    if internal:
        out["agent/internal_rate"] = float(np.mean(internal))
    _mean_std("agent/response_len", gen_len)

    scores = list(sequence_scores) if sequence_scores is not None else list(total)
    if uids is not None and scores and len(uids) == len(scores):
        by_uid: dict[Any, list[float]] = defaultdict(list)
        for u, s in zip(uids, scores):
            by_uid[u].append(float(s))
        group_stds = []
        zero = 0
        for vals in by_uid.values():
            if len(vals) < 2:
                continue
            std = float(np.std(vals))
            group_stds.append(std)
            if std <= zero_std_eps:
                zero += 1
        if group_stds:
            out["grpo/zero_std_group_rate"] = zero / len(group_stds)
            out["grpo/group_reward_std/mean"] = float(np.mean(group_stds))
            out["grpo/num_groups"] = float(len(group_stds))

        # Observation only: which reward still has within-group advantage.
        if len(uids) == len(answer) == len(evidence):
            for key, vals in (
                ("answer", answer),
                ("evidence", evidence),
                ("format", fmt if len(fmt) == len(uids) else []),
            ):
                if not vals:
                    continue
                by_c: dict[Any, list[float]] = defaultdict(list)
                for u, v in zip(uids, vals):
                    by_c[u].append(float(v))
                rates = []
                for gvals in by_c.values():
                    if len(gvals) < 2:
                        continue
                    rates.append(1.0 if float(np.std(gvals)) > zero_std_eps else 0.0)
                if rates:
                    out[f"grpo/group_nonzero_var/{key}"] = float(np.mean(rates))
    out["grpo/num_trajectories"] = float(n)
    return out


def summarize_console_line(metrics: Mapping[str, float]) -> str:
    labeled = [
        ("answer_em", "reward/answer_em/mean"),
        ("evidence_f1", "reward/evidence_f1/mean"),
        ("format", "reward/format/mean"),
        ("total", "reward/total/mean"),
        ("finish", "agent/finish_rate"),
        ("search_rate", "agent/search_rate"),
        ("zero_std", "grpo/zero_std_group_rate"),
        ("ans_nz", "grpo/group_nonzero_var/answer"),
        ("ev_nz", "grpo/group_nonzero_var/evidence"),
    ]
    return " | ".join(f"{lab}={metrics[k]:.4g}" for lab, k in labeled if k in metrics)
