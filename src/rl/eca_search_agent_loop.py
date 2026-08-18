"""ECA Search AgentLoop for veRL — preserves SFT-v1 XML dialect.

Flow:
  prompt (system+question, no gold/contexts)
    → generate until </search>|</answer>|</internal>
    → if <search>: CandidateBM25Tool(sample_id, query) → <observation> (mask=0)
    → continue (max_search_turns=2)
    → terminate on <answer> or budgets

Registered via configs/rl/eca_agent_loop.yaml as agent_name=eca_search_agent.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import ToolListWrap
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel("INFO")

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_INTERNAL_RE = re.compile(r"<internal>(.*?)</internal>", re.DOTALL | re.IGNORECASE)

_STOP_STRINGS = ["</search>", "</answer>", "</internal>"]


def _audit_enabled() -> bool:
    return (os.environ.get("ECA_ROUTING_MISMATCH_AUDIT") or "").strip() in ("1", "true", "yes")


def _audit_first_generate_only() -> bool:
    return (os.environ.get("ECA_AUDIT_FIRST_GENERATE_ONLY") or "").strip() in ("1", "true", "yes")


def _audit_stop_mode() -> str:
    # current = last-token stop_ids (production); none = strip custom stop_token_ids
    return (os.environ.get("ECA_AUDIT_STOP_MODE") or "current").strip().lower()


def _audit_max_new_tokens() -> int | None:
    """Optional per-generate cap for Path B forensic (avoid filling 2048)."""
    raw = (os.environ.get("ECA_AUDIT_MAX_NEW_TOKENS") or "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
    except ValueError:
        return None
    return v if v > 0 else None


def _audit_path_label() -> str:
    return (os.environ.get("ECA_AUDIT_PATH") or "").strip() or (
        "B" if _audit_first_generate_only() else "C"
    )


def _rollout_backend() -> str:
    """Backend label/compatibility switch set explicitly by launchers."""
    return (os.environ.get("ECA_ROLLOUT_BACKEND") or "sglang").strip().lower()


def _positive_env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return value if value > 0 else default


def _truncate_at_complete_sequence(
    token_ids: list[int],
    sequences: dict[str, list[int]],
    max_tokens: int,
    tokenizer: Any | None = None,
) -> tuple[list[int], str | None, bool]:
    """Keep through the earliest complete closing tag, bounded by max_tokens.

    Exact token matching remains the cheap/default path.  With a tokenizer,
    also inspect decoded prefixes because BPE tokenization of ``</tag>`` can
    differ when the tag follows a newline or ordinary text.
    """
    best_end: int | None = None
    best_tag: str | None = None
    for tag, sequence in sequences.items():
        if not sequence:
            continue
        width = len(sequence)
        for start in range(0, len(token_ids) - width + 1):
            if token_ids[start : start + width] == sequence:
                end = start + width
                if best_end is None or end < best_end:
                    best_end, best_tag = end, tag
                break
    if tokenizer is not None:
        limit = min(len(token_ids), max_tokens)
        bounded_text = tokenizer.decode(token_ids[:limit], skip_special_tokens=True)
        decoded_candidates: list[tuple[int, int, str]] = []
        for tag in sequences:
            char_pos = bounded_text.find(tag)
            if char_pos < 0:
                continue
            lo, hi = 1, limit
            while lo < hi:
                mid = (lo + hi) // 2
                prefix_text = tokenizer.decode(token_ids[:mid], skip_special_tokens=True)
                if tag in prefix_text:
                    hi = mid
                else:
                    lo = mid + 1
            decoded_candidates.append((lo, char_pos, tag))
        if decoded_candidates:
            best_end, _, best_tag = min(decoded_candidates)

    keep = min(len(token_ids), max_tokens, best_end if best_end is not None else len(token_ids))
    matched = best_tag if best_end is not None and best_end <= max_tokens else None
    return token_ids[:keep], matched, keep < len(token_ids)


def _token_output_meta(output: Any) -> dict[str, Any]:
    """Best-effort finish_reason / meta from TokenOutput (engine-dependent)."""
    meta: dict[str, Any] = {}
    for key in ("finish_reason", "stop_reason", "matched_stop", "meta_info"):
        if hasattr(output, key):
            val = getattr(output, key)
            if val is not None:
                meta[key] = val
    extra = getattr(output, "extra_fields", None) or {}
    if isinstance(extra, dict):
        for key in ("finish_reason", "stop_reason", "matched_stop", "meta_info"):
            if key in extra and extra[key] is not None:
                meta[key] = extra[key]
    return meta


def _stop_tokenization_report(tokenizer: Any) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    last_ids: list[int] = []
    for s in _STOP_STRINGS:
        ids = list(tokenizer.encode(s, add_special_tokens=False))
        last = ids[-1] if ids else None
        tags[s] = {
            "full_token_ids": ids,
            "last_token_id": last,
            "decoded_last_token": (
                tokenizer.decode([last], skip_special_tokens=False) if last is not None else None
            ),
        }
        if last is not None:
            last_ids.append(int(last))
    collision: dict[str, list[str]] = {}
    for s, info in tags.items():
        lid = info["last_token_id"]
        if lid is None:
            continue
        collision.setdefault(str(lid), []).append(s)
    collision = {k: v for k, v in collision.items() if len(v) > 1}
    return {
        "closing_tags": tags,
        "last_token_collision": collision,
        "unique_last_token_ids": sorted(set(last_ids)),
    }


def _mismatch_dump_row(row: dict[str, Any]) -> None:
    """Append forensic row when ECA_ROUTING_MISMATCH_DUMP is set."""
    path = (os.environ.get("ECA_ROUTING_MISMATCH_DUMP") or "").strip()
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with out.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@register("eca_search_agent")
class EcaSearchAgentLoop(AgentLoopBase):
    """Multi-turn search agent using Candidate-BM25 BaseTool + XML tags."""

    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        super().__init__(*args, **kwargs)
        tool_list = tools.tools if tools else []
        self.tools = {t.name: t for t in tool_list}
        self.search_tool = self.tools.get("search")
        if self.search_tool is None:
            raise ValueError(
                "EcaSearchAgentLoop requires a BaseTool named 'search' "
                "(see configs/rl/candidate_bm25_tool.yaml)"
            )

        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = int(mt.max_assistant_turns or 6)
        self.max_user_turns = int(mt.max_user_turns or 4)
        self.max_tool_response_length = int(mt.max_tool_response_length or 2048)
        self.tool_response_truncate_side = mt.tool_response_truncate_side or "middle"

        # Locked for 3B0/3B1; raise later via config if needed.
        self.max_search_turns = 2
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        self.max_assistant_turn_tokens = _positive_env_int("ECA_MAX_ASSISTANT_TURN_TOKENS", 256)
        self.final_answer_reserve = _positive_env_int("ECA_FINAL_ANSWER_RESERVE", 256)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {}) or {}
        tools_kwargs = kwargs.get("tools_kwargs", {}) or extra_info.get("tools_kwargs") or {}
        forced_root_action = str(extra_info.get("cur_forced_arm") or "").strip().lower()
        if forced_root_action not in {"", "search", "internal"}:
            raise ValueError(f"invalid cur_forced_arm={forced_root_action!r}")

        sample_id = (
            extra_info.get("sample_id")
            or (tools_kwargs.get("search") or {}).get("create_kwargs", {}).get("sample_id")
        )
        if not sample_id:
            raise ValueError("eca_search_agent requires extra_info.sample_id")

        create_kwargs = {"sample_id": str(sample_id)}
        # Prefer dataset tools_kwargs if present.
        if "search" in tools_kwargs and isinstance(tools_kwargs["search"], dict):
            ck = tools_kwargs["search"].get("create_kwargs") or {}
            if ck.get("sample_id"):
                create_kwargs["sample_id"] = str(ck["sample_id"])

        metrics: dict[str, Any] = {
            "sample_id": create_kwargs["sample_id"],
            "search_count": 0,
            "duplicate_query_count": 0,
            "observation_tokens": 0,
            "finish": 0,
            "used_internal": 0,
            "final_answer_missing": 1,
            "final_answer_reserve_violations": 0,
            "cur_forced_arm": forced_root_action,
            "cur_forbidden_search_attempts": 0,
        }
        request_id = uuid4().hex

        # Initial prompt — DO NOT inject OpenAI tool schemas (SFT-v1 never saw them).
        prompt_ids = await self.apply_chat_template(messages, tools=None)
        canonical_prompt_ids = list(prompt_ids)
        root_prompt_len = len(canonical_prompt_ids)
        canonical_prompt_sha256 = hashlib.sha256(
            json.dumps(canonical_prompt_ids).encode("utf-8")
        ).hexdigest()
        forced_open_tag = f"<{forced_root_action}>" if forced_root_action else ""
        forced_prefix_ids = (
            list(self.tokenizer.encode(forced_open_tag, add_special_tokens=False))
            if forced_open_tag
            else []
        )
        stop_tok_report = _stop_tokenization_report(self.tokenizer) if _audit_enabled() else {}
        stop_sequences = {
            tag: list(self.tokenizer.encode(tag, add_special_tokens=False)) for tag in _STOP_STRINGS
        }

        async def build_observation_ids(body: str, suffix: str, token_cap: int) -> list[int]:
            """Format one observation turn while enforcing an exact token cap."""

            async def render(rendered_body: str) -> list[int]:
                content = f"<observation>\n{rendered_body}\n</observation>\n{suffix}"
                ids = list(
                    await self.apply_chat_template(
                        [{"role": "user", "content": content}],
                        tools=None,
                        remove_system_prompt=True,
                    )
                )
                return list(self.turn_separator or []) + ids

            full_ids = await render(body)
            if len(full_ids) <= token_cap:
                return full_ids

            body_ids = list(self.tokenizer.encode(body, add_special_tokens=False))
            keep = max(0, len(body_ids) - (len(full_ids) - token_cap) - 8)
            while keep >= 0:
                if self.tool_response_truncate_side == "left":
                    kept = body_ids[-keep:] if keep else []
                    marker_left, marker_right = "(truncated)...", ""
                elif self.tool_response_truncate_side == "right":
                    kept = body_ids[:keep]
                    marker_left, marker_right = "", "...(truncated)"
                else:
                    left = keep // 2
                    right = keep - left
                    kept = []
                    marker_left, marker_right = "", ""
                if self.tool_response_truncate_side == "middle":
                    rendered = (
                        self.tokenizer.decode(body_ids[:left], skip_special_tokens=False)
                        + "...(truncated)..."
                        + self.tokenizer.decode(body_ids[-right:] if right else [], skip_special_tokens=False)
                    )
                else:
                    rendered = self.tokenizer.decode(kept, skip_special_tokens=False)
                    rendered = marker_left + rendered + marker_right
                ids = await render(rendered)
                if len(ids) <= token_cap:
                    return ids
                keep -= max(1, len(ids) - token_cap)
            return []

        response_mask: list[int] = []
        response_logprobs: list[float] = []
        search_queries: list[str] = []
        assistant_turns = 0
        user_turns = 0
        finished = False
        route_first = forced_root_action or "none"  # search | internal | both | answer | none
        first_gen_ids: list[int] = []
        first_gen_text = ""
        # Propagate weight-version tags from generate → TransferQueue metrics.
        # Missing these makes trainer_base._compute_metrics crash (None → int).
        min_global_steps: int | None = None
        max_global_steps: int | None = None

        # Bind trajectory tool instance once.
        instance_id, _ = await self.search_tool.create(create_kwargs=create_kwargs)

        try:
            sp = dict(sampling_params)
            rollout_backend = _rollout_backend()
            # Prefer stop_token_ids. String `stop` crashes SGLang 0.5.5 under
            # veRL hybrid (skip_tokenizer_init → scheduler tokenizer is None).
            sp.pop("stop", None)
            stop_ids: list[int] = []
            stop_mode = _audit_stop_mode()
            if stop_mode == "current":
                for s in _STOP_STRINGS:
                    ids = self.tokenizer.encode(s, add_special_tokens=False)
                    if ids:
                        # Use last token as stop signal (Qwen often splits tags).
                        stop_ids.append(ids[-1])
                if stop_ids:
                    existing = list(sp.get("stop_token_ids") or [])
                    sp["stop_token_ids"] = sorted(set(existing + stop_ids))
            elif stop_mode in ("none", "sequence"):
                sp.pop("stop_token_ids", None)
            else:
                raise ValueError(f"unknown ECA_AUDIT_STOP_MODE={stop_mode!r}")

            audit_cap = _audit_max_new_tokens()
            if audit_cap is not None:
                sp["max_new_tokens"] = int(audit_cap)

            turn_gen_lens: list[int] = []
            turn_gen_raw_lens: list[int] = []
            turn_close_tags: list[str | None] = []
            turn_was_truncated: list[bool] = []
            turn_continued: list[bool] = []
            observation_turn_lens: list[int] = []
            first_gen_meta: dict[str, Any] = {}

            while (
                assistant_turns < self.max_assistant_turns
                and user_turns < self.max_user_turns
                and len(response_mask) < self.response_length
            ):
                # CUR intervention is applied *after* hashing the canonical
                # prompt.  Its tokens are externally forced, hence masked out
                # of policy loss/logprob while remaining in the response span.
                if assistant_turns == 0 and forced_prefix_ids:
                    prompt_ids = prompt_ids + forced_prefix_ids
                    response_mask.extend([0] * len(forced_prefix_ids))
                    response_logprobs.extend([0.0] * len(forced_prefix_ids))
                # Prefer remaining trajectory budget when engine honors max_new_tokens.
                remain_budget = self.response_length - len(response_mask)
                if remain_budget <= 0:
                    break
                sp_turn = dict(sp)
                if rollout_backend == "vexact":
                    # VeXact derives the remaining decode budget from the current
                    # prompt length and rejects per-request max_new_tokens.
                    sp_turn.pop("max_new_tokens", None)
                elif audit_cap is not None:
                    sp_turn["max_new_tokens"] = min(int(audit_cap), int(remain_budget))
                else:
                    # Do not request more tokens than remaining response slot.
                    existing_mnt = sp_turn.get("max_new_tokens")
                    if existing_mnt is None:
                        sp_turn["max_new_tokens"] = int(remain_budget)
                    else:
                        sp_turn["max_new_tokens"] = min(int(existing_mnt), int(remain_budget))

                with simple_timer("generate_sequences", metrics):
                    output: TokenOutput = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=sp_turn,
                    )

                # Track trajectory weight versions across multi-turn generate calls.
                out_extra = getattr(output, "extra_fields", None) or {}
                step = out_extra.get("min_global_steps", out_extra.get("global_steps"))
                if step is not None:
                    step_i = int(step)
                    min_global_steps = step_i if min_global_steps is None else min(min_global_steps, step_i)
                    max_g = out_extra.get("max_global_steps", step_i)
                    max_global_steps = (
                        int(max_g)
                        if max_global_steps is None
                        else max(max_global_steps, int(max_g))
                    )

                raw_gen_ids = list(output.token_ids or [])
                if not raw_gen_ids:
                    break

                # A2 preserves raw first-generate output. Formal multi-turn uses
                # complete closing-sequence truncation plus a per-turn token cap.
                remain = self.response_length - len(response_mask)
                if _audit_enabled() and _audit_first_generate_only() and assistant_turns == 0:
                    gen_ids = raw_gen_ids[:remain]
                    matched_close = None
                    was_truncated = len(gen_ids) < len(raw_gen_ids)
                else:
                    gen_ids, matched_close, was_truncated = _truncate_at_complete_sequence(
                        raw_gen_ids,
                        stop_sequences,
                        min(remain, self.max_assistant_turn_tokens),
                        tokenizer=self.tokenizer,
                    )
                turn_gen_raw_lens.append(len(raw_gen_ids))
                turn_gen_lens.append(len(gen_ids))
                turn_close_tags.append(matched_close)
                turn_was_truncated.append(was_truncated)
                turn_continued.append(False)
                if assistant_turns == 0:
                    first_gen_ids = list(forced_prefix_ids) + list(gen_ids)
                    first_gen_text = forced_open_tag + self.tokenizer.decode(
                        gen_ids, skip_special_tokens=True
                    )
                    first_gen_meta = _token_output_meta(output)

                # Path B: dump raw first generate then terminate (no parse/tool/continue).
                if _audit_enabled() and _audit_first_generate_only() and assistant_turns == 0:
                    prompt_ids = prompt_ids + gen_ids
                    response_mask.extend([1] * len(gen_ids))
                    if output.log_probs:
                        response_logprobs.extend(list(output.log_probs[: len(gen_ids)]))
                    assistant_turns += 1
                    text = first_gen_text
                    has_answer = bool(_ANSWER_RE.search(text))
                    search_hits = list(_SEARCH_RE.finditer(text))
                    has_internal = bool(_INTERNAL_RE.search(text))
                    if search_hits and has_internal:
                        route_first = "both"
                    elif search_hits:
                        route_first = "search"
                    elif has_internal:
                        route_first = "internal"
                    elif has_answer:
                        route_first = "answer"
                    # Still record a "raw" label from decode for attribution, but
                    # Path B verdict uses first_generate_* primarily.
                    _mismatch_dump_row(
                        {
                            "backend": f"{rollout_backend}_eca_search_agent_loop",
                            "audit_path": _audit_path_label(),
                            "first_generate_only": True,
                            "sample_id": create_kwargs["sample_id"],
                            "canonical_prompt_ids": canonical_prompt_ids,
                            "canonical_prompt_sha256": canonical_prompt_sha256,
                            "canonical_prompt_len": root_prompt_len,
                            "root_prompt_len": root_prompt_len,
                            "canonical_prompt_text": self.tokenizer.decode(
                                canonical_prompt_ids, skip_special_tokens=False
                            ),
                            "first_generate_token_ids": first_gen_ids,
                            "first_generate_text": first_gen_text,
                            "first_generate_text_prefix": first_gen_text[:64],
                            "first_generate_len": len(first_gen_ids),
                            "generated_len": len(first_gen_ids),
                            "response_mask_len": len(response_mask),
                            "max_new_tokens_requested": sp_turn.get("max_new_tokens"),
                            "route_first": route_first,
                            "route_token_start": root_prompt_len,
                            "route_token_end": root_prompt_len + len(first_gen_ids),
                            "sampling_params_full": {
                                k: (list(v) if isinstance(v, (set, tuple)) else v)
                                for k, v in sorted(sp_turn.items())
                                if k != "stop"  # avoid huge/unhashable
                            },
                            "sampling_temperature": sp_turn.get("temperature"),
                            "sampling_top_p": sp_turn.get("top_p"),
                            "sampling_top_k": sp_turn.get("top_k"),
                            "sampling_min_p": sp_turn.get("min_p"),
                            "stop_strings": list(_STOP_STRINGS),
                            "stop_token_ids": list(sp_turn.get("stop_token_ids") or []),
                            "stop_mode": stop_mode,
                            "stop_tokenization": stop_tok_report,
                            "generate_meta": first_gen_meta,
                            "sglang_log_probs_first": (
                                list(output.log_probs[: len(first_gen_ids)])
                                if output.log_probs
                                else None
                            ),
                            "rollout_log_probs_first": (
                                list(output.log_probs[: len(first_gen_ids)])
                                if output.log_probs
                                else None
                            ),
                            "parsed_despite_bypass": True,
                        }
                    )
                    break

                prompt_ids = prompt_ids + gen_ids
                response_mask.extend([1] * len(gen_ids))
                if output.log_probs:
                    response_logprobs.extend(list(output.log_probs[: len(gen_ids)]))
                assistant_turns += 1

                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                if assistant_turns == 1 and forced_open_tag:
                    text = forced_open_tag + text
                # Some engines omit stop string; tolerate either.
                has_answer = bool(_ANSWER_RE.search(text))
                search_hits = list(_SEARCH_RE.finditer(text))
                has_internal = bool(_INTERNAL_RE.search(text))

                if route_first == "none":
                    if search_hits and has_internal:
                        route_first = "both"
                    elif search_hits:
                        route_first = "search"
                    elif has_internal:
                        route_first = "internal"
                    elif has_answer:
                        route_first = "answer"

                if has_answer:
                    finished = True
                    metrics["finish"] = 1
                    metrics["final_answer_missing"] = 0
                    break

                if search_hits:
                    if forced_root_action == "internal":
                        # Tools remain disabled for the complete internal arm.
                        # Retain this as a policy failure instead of executing.
                        metrics["cur_forbidden_search_attempts"] += len(search_hits)
                        break
                    if metrics["search_count"] >= self.max_search_turns:
                        # Budget exhausted — stop without executing more search.
                        break
                    query = search_hits[-1].group(1).strip()
                    if query in search_queries:
                        metrics["duplicate_query_count"] += 1
                    search_queries.append(query)

                    tool_resp, _, tool_metrics = await self.search_tool.execute(
                        instance_id, {"query": query}
                    )
                    metrics["search_count"] = int(metrics["search_count"]) + 1
                    obs_body = (tool_resp.text or "[no documents retrieved]").strip()
                    suffix = (
                        "Continue. Prefer <evidence> then <think> then <answer>. "
                        f"You may <search> again only if necessary "
                        f"(searches used: {metrics['search_count']}/{self.max_search_turns})."
                    )
                    room_before_reserve = (
                        self.response_length - len(response_mask) - self.final_answer_reserve
                    )
                    obs_cap = min(self.max_tool_response_length, room_before_reserve)
                    if obs_cap <= 0:
                        metrics["final_answer_reserve_violations"] += 1
                        break
                    obs_ids = await build_observation_ids(obs_body, suffix, obs_cap)
                    if not obs_ids:
                        metrics["final_answer_reserve_violations"] += 1
                        break

                    prompt_ids = prompt_ids + obs_ids
                    response_mask.extend([0] * len(obs_ids))
                    if response_logprobs:
                        response_logprobs.extend([0.0] * len(obs_ids))
                    metrics["observation_tokens"] += len(obs_ids)
                    observation_turn_lens.append(len(obs_ids))
                    user_turns += 1
                    metrics["last_tool_metrics"] = tool_metrics
                    continue

                if has_internal:
                    metrics["used_internal"] = 1
                    # Ask for final answer without search (same as Phase 3A).
                    nudge = (
                        "You chose <internal>. Now give the final answer in "
                        "<answer>...</answer> (optionally with short <think>)."
                    )
                    nudge_ids = await self.apply_chat_template(
                        [{"role": "user", "content": nudge}],
                        tools=None,
                        remove_system_prompt=True,
                    )
                    if self.turn_separator:
                        nudge_ids = list(self.turn_separator) + list(nudge_ids)
                    room_before_reserve = (
                        self.response_length - len(response_mask) - self.final_answer_reserve
                    )
                    if len(nudge_ids) > room_before_reserve:
                        metrics["final_answer_reserve_violations"] += 1
                        break
                    prompt_ids = prompt_ids + nudge_ids
                    response_mask.extend([0] * len(nudge_ids))
                    if response_logprobs:
                        response_logprobs.extend([0.0] * len(nudge_ids))
                    user_turns += 1
                    continue

                if was_truncated and len(gen_ids) >= min(remain, self.max_assistant_turn_tokens):
                    # Preserve the per-turn cap without discarding an otherwise
                    # valid long answer.  Give the model another bounded turn to
                    # finish and emit a complete closing tag.
                    nudge = (
                        "Continue exactly where the previous response stopped. "
                        "Finish briefly and emit the required complete closing tag; "
                        "do not repeat earlier text."
                    )
                    nudge_ids = await self.apply_chat_template(
                        [{"role": "user", "content": nudge}],
                        tools=None,
                        remove_system_prompt=True,
                    )
                    if self.turn_separator:
                        nudge_ids = list(self.turn_separator) + list(nudge_ids)
                    room_before_reserve = (
                        self.response_length - len(response_mask) - self.final_answer_reserve
                    )
                    if len(nudge_ids) > room_before_reserve:
                        metrics["final_answer_reserve_violations"] += 1
                        break
                    prompt_ids = prompt_ids + list(nudge_ids)
                    response_mask.extend([0] * len(nudge_ids))
                    if response_logprobs:
                        response_logprobs.extend([0.0] * len(nudge_ids))
                    user_turns += 1
                    turn_continued[-1] = True
                    continue

                # No closed action — terminate.
                break
        finally:
            await self.search_tool.release(instance_id)

        response_ids = prompt_ids[-len(response_mask) :] if response_mask else []
        prompt_ids_out = prompt_ids[: len(prompt_ids) - len(response_mask)] if response_mask else prompt_ids

        # Store masked decode helpers for audit.
        # Fallback to 0 for sync smoke if generate never stamped versions
        # (empty gen / early break) — still must be int for metrics.
        if min_global_steps is None:
            min_global_steps = 0
        if max_global_steps is None:
            max_global_steps = min_global_steps
        # Flatten agent counters for GRPO TensorBoard (see grpo_metrics.py).
        metrics["route_first"] = route_first
        metrics["response_tokens"] = len(response_mask)
        metrics["assistant_tokens"] = int(sum(response_mask))
        metrics["hit_response_cap"] = int(len(response_mask) >= self.response_length)
        metrics["max_assistant_turn_tokens"] = max(turn_gen_lens, default=0)
        metrics["max_observation_turn_tokens"] = max(observation_turn_lens, default=0)
        metrics["final_answer_reserve"] = self.final_answer_reserve
        if forced_root_action == "search":
            forced_action_valid = int(int(metrics.get("search_count") or 0) >= 1)
        elif forced_root_action == "internal":
            forced_action_valid = int(
                int(metrics.get("used_internal") or 0) == 1
                and int(metrics.get("search_count") or 0) == 0
                and int(metrics.get("cur_forbidden_search_attempts") or 0) == 0
            )
        else:
            forced_action_valid = 1
        metrics["cur_forced_action_valid"] = forced_action_valid
        metrics["cur_policy_failure"] = int(bool(forced_root_action) and not forced_action_valid)
        extra_fields = {
            "turn_scores": [],
            "tool_rewards": [],
            "min_global_steps": min_global_steps,
            "max_global_steps": max_global_steps,
            "sample_id": create_kwargs["sample_id"],
            "search_queries": search_queries,
            "finish": int(finished or metrics.get("finish") or 0),
            "search_count": int(metrics.get("search_count") or 0),
            "duplicate_query_count": int(metrics.get("duplicate_query_count") or 0),
            "observation_tokens": int(metrics.get("observation_tokens") or 0),
            "max_search_turns": int(self.max_search_turns),
            "used_internal": int(metrics.get("used_internal") or 0),
            "route_first": route_first,
            "cur_forced_arm": forced_root_action,
            "cur_forced_prefix_ids": forced_prefix_ids,
            "cur_forced_action_valid": forced_action_valid,
            "cur_policy_failure": int(metrics["cur_policy_failure"]),
            "cur_forbidden_search_attempts": int(
                metrics.get("cur_forbidden_search_attempts") or 0
            ),
            # Exact fixed-policy attribution needs the untouched root prompt
            # and the first assistant segment.  Keep these in TransferQueue so
            # the post-reward trainer hook can persist them with estimator
            # tensors in the same row order.
            "canonical_prompt_ids": canonical_prompt_ids,
            "canonical_prompt_sha256": canonical_prompt_sha256,
            "root_prompt_len": root_prompt_len,
            "first_generate_token_ids": first_gen_ids,
            "first_generate_len": len(first_gen_ids),
            "route_token_start": 0,
            "route_token_end": len(first_gen_ids),
            "response_tokens": int(metrics["response_tokens"]),
            "hit_response_cap": int(metrics["hit_response_cap"]),
            "final_answer_missing": int(metrics["final_answer_missing"]),
            "final_answer_reserve_violations": int(metrics["final_answer_reserve_violations"]),
            "metrics": metrics,
        }
        _parity_dump_row(
            {
                "sample_id": create_kwargs["sample_id"],
                # Counterfactual gradient attribution is valid only when all
                # branches consume exactly the same sampled trajectory.
                "trajectory_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "response_ids": response_ids[: self.response_length],
                            "response_mask": response_mask[: self.response_length],
                        },
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "route_first": route_first,
                "canonical_prompt_sha256": canonical_prompt_sha256,
                "canonical_prompt_len": root_prompt_len,
                "cur_forced_arm": forced_root_action,
                "cur_forced_prefix_ids": forced_prefix_ids,
                "cur_forced_action_valid": forced_action_valid,
                "cur_policy_failure": int(metrics["cur_policy_failure"]),
                "cur_forbidden_search_attempts": int(
                    metrics.get("cur_forbidden_search_attempts") or 0
                ),
                "action": (
                    "search"
                    if route_first == "search"
                    else ("internal" if route_first == "internal" else "other")
                ),
                "search_count": int(metrics.get("search_count") or 0),
                "used_internal": int(metrics.get("used_internal") or 0),
                "finish": int(finished or metrics.get("finish") or 0),
                "response_tokens": int(metrics["response_tokens"]),
                "hit_response_cap": int(metrics["hit_response_cap"]),
                "final_answer_missing": int(metrics["final_answer_missing"]),
                "final_answer_reserve_violations": int(metrics["final_answer_reserve_violations"]),
                "max_assistant_turn_tokens": int(metrics["max_assistant_turn_tokens"]),
                "max_observation_turn_tokens": int(metrics["max_observation_turn_tokens"]),
                "max_search_turns": int(self.max_search_turns),
                "max_assistant_turns": int(self.max_assistant_turns),
                "prompt_ids": prompt_ids_out,
                "response_ids": response_ids[: self.response_length],
                "response_mask": response_mask[: self.response_length],
                "response_logprobs": (
                    response_logprobs[: self.response_length] if response_logprobs else None
                ),
                "backend": f"{_rollout_backend()}_eca_search_agent_loop",
            }
        )
        if _audit_enabled() and not _audit_first_generate_only():
            _mismatch_dump_row(
                {
                    "backend": f"{rollout_backend}_eca_search_agent_loop",
                    "audit_path": _audit_path_label(),
                    "first_generate_only": False,
                    "sample_id": create_kwargs["sample_id"],
                    "canonical_prompt_ids": canonical_prompt_ids,
                    "canonical_prompt_sha256": canonical_prompt_sha256,
                    "canonical_prompt_len": root_prompt_len,
                    "root_prompt_len": root_prompt_len,
                    "canonical_prompt_text": self.tokenizer.decode(
                        canonical_prompt_ids, skip_special_tokens=False
                    ),
                    "first_generate_token_ids": first_gen_ids,
                    "first_generate_text": first_gen_text,
                    "first_generate_text_prefix": first_gen_text[:64],
                    "first_generate_len": len(first_gen_ids),
                    "turn_gen_lens": turn_gen_lens,
                    "turn_gen_raw_lens": turn_gen_raw_lens,
                    "turn_close_tags": turn_close_tags,
                    "turn_was_truncated": turn_was_truncated,
                    "turn_continued": turn_continued,
                    "observation_turn_lens": observation_turn_lens,
                    "response_mask_len": len(response_mask),
                    "assistant_turns": int(assistant_turns),
                    "user_turns": int(user_turns),
                    "observation_tokens": int(metrics.get("observation_tokens") or 0),
                    "route_first": route_first,
                    "route_token_start": root_prompt_len,
                    "route_token_end": root_prompt_len + len(first_gen_ids),
                    "sampling_temperature": (sampling_params or {}).get("temperature"),
                    "sampling_top_p": (sampling_params or {}).get("top_p"),
                    "stop_strings": list(_STOP_STRINGS),
                    "stop_token_ids": sorted(set(stop_ids)) if stop_ids else [],
                    "stop_mode": stop_mode,
                    "stop_tokenization": stop_tok_report,
                    "generate_meta_first": first_gen_meta,
                    "search_count": int(metrics.get("search_count") or 0),
                    "used_internal": int(metrics.get("used_internal") or 0),
                    "finish": int(finished or metrics.get("finish") or 0),
                    "hit_response_cap": int(metrics["hit_response_cap"]),
                    "final_answer_missing": int(metrics["final_answer_missing"]),
                    "final_answer_reserve_violations": int(metrics["final_answer_reserve_violations"]),
                    "max_assistant_turn_tokens": int(metrics["max_assistant_turn_tokens"]),
                    "max_observation_turn_tokens": int(metrics["max_observation_turn_tokens"]),
                }
            )

        return AgentLoopOutput(
            prompt_ids=prompt_ids_out,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            num_turns=assistant_turns + user_turns + 1,
            metrics=metrics,
            extra_fields=extra_fields,
        )


def _parity_dump_row(row: dict[str, Any]) -> None:
    """Append one trajectory row when ECA_PARITY_DUMP is set (rollout-only audit)."""
    path = (os.environ.get("ECA_PARITY_DUMP") or "").strip()
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with out.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# Re-export for convenience inside the container.
from src.rl.mask_audit import dump_mask_audit  # noqa: E402
