"""Phase 3A: multi-turn Search Agent loop (generate → tool → observe → continue).

No training. Observation tokens are environment-injected (loss_mask=False in Trace).
Harness v1: Qwen3 no-think prefix + LlamaFactory qwen3_nothink tool_response slot.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.eval.metrics import basic_metrics, exact_match, token_f1
from src.eval.protocol import score_evidence_use
from src.eval.trace_schema import (
    EXPECTED_LOSS_MASK,
    CostInfo,
    TraceRecord,
    TraceStep,
    validate_trace_record,
)
from src.sft.prototype_builder import AGENT_SYSTEM_PROMPT
from src.tools.candidate_bm25 import (
    docs_to_schema,
    format_observation_text,
    retrieve_candidate_bm25,
)

_STOP_STRINGS = ("</search>", "</answer>", "</internal>")

_TAG_BODY = {
    "think": re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    "internal": re.compile(r"<internal>(.*?)</internal>", re.DOTALL | re.IGNORECASE),
    "search": re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE),
    "evidence": re.compile(r"<evidence>(.*?)</evidence>", re.DOTALL | re.IGNORECASE),
    "answer": re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE),
}


@dataclass
class RolloutConfig:
    top_k: int = 5
    max_search_turns: int = 2
    max_new_tokens: int = 512
    max_rounds: int = 8
    temperature: float = 0.0
    system_prompt: str = AGENT_SYSTEM_PROMPT


@dataclass
class RolloutResult:
    trace: TraceRecord
    finished: bool
    route_first: str  # internal | search | none | both
    search_queries: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw_generations: List[str] = field(default_factory=list)


_LOOSE_ANSWER_RE = re.compile(r"(?is)(?:<answer>\s*)?(.*?)\s*</answer>")
# HF Qwen3 still appends an empty think block when enable_thinking=False.
# SFT used LlamaFactory qwen3_nothink, which starts generation at assistant\n.
_EMPTY_THINK_TAIL = re.compile(r"(?:<think>\s*</think>\s*)$", re.IGNORECASE)
_FORMAT_NUDGE = (
    "Your last message had no closed action tag. "
    "Reply with either <search>query</search> or "
    "<answer>short answer</answer> only."
)


# LlamaFactory qwen3_nothink format_observation (training slot). Body only.
_LF_OBS_SLOT = (
    "<|im_start|>user\n<tool_response>\n{content}\n"
    "</tool_response><|im_end|>\n<|im_start|>assistant\n"
)


def lf_observation_slot(obs_body: str) -> str:
    """Exact qwen3_nothink observation serialization. Do not add Continue."""
    return _LF_OBS_SLOT.replace("{content}", obs_body)


def tool_response_user_content(obs_body: str) -> str:
    """Tag wrap only; inference should use role=observation + lf_observation_slot."""
    return f"<tool_response>\n{obs_body}\n</tool_response>"


def _hf_nothink(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    return _EMPTY_THINK_TAIL.sub("", prompt)


def lf_serialize_messages(messages: List[Dict[str, str]]) -> str:
    """Full qwen3_nothink string (system/user/assistant/observation)."""
    rest = list(messages)
    system = ""
    if rest and rest[0].get("role") == "system":
        system = rest[0]["content"]
        rest = rest[1:]
    out = f"<|im_start|>system\n{system}<|im_end|>\n"
    for m in rest:
        role = m["role"]
        if role == "user":
            out += (
                f"<|im_start|>user\n{m['content']}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        elif role == "assistant":
            out += m["content"] + "<|im_end|>\n"
        elif role == "observation":
            out += lf_observation_slot(m["content"])
        else:
            raise ValueError(f"unsupported role in lf_serialize: {role}")
    if rest and rest[-1]["role"] == "assistant":
        out += "<|im_start|>assistant\n"
    return out


def render_nothink_prompt(
    tokenizer: PreTrainedTokenizerBase, messages: List[Dict[str, str]]
) -> str:
    """Round-0: HF no-think + empty-think strip. After tool: LF serialize."""
    if any(m.get("role") == "observation" for m in messages):
        return lf_serialize_messages(messages)
    return _hf_nothink(tokenizer, messages, add_generation_prompt=True)


def _extract_closed_tags(text: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name, cre in _TAG_BODY.items():
        hits = [m.group(1).strip() for m in cre.finditer(text or "") if m.group(1).strip()]
        if hits:
            out[name] = hits
    return out


def _recover_loose_answer(text: str) -> Optional[str]:
    """Accept a closed <answer> pair, or a dangling </answer> without opener."""
    tags = _extract_closed_tags(text)
    if tags.get("answer"):
        return tags["answer"][0]
    if "</answer>" not in (text or "").lower():
        return None
    m = _LOOSE_ANSWER_RE.search(text or "")
    body = (m.group(1).strip() if m else "")
    return body or None


def _first_action(tags: Dict[str, List[str]]) -> Optional[str]:
    """Priority: answer > search > internal (answer ends episode)."""
    if "answer" in tags:
        return "answer"
    if "search" in tags:
        return "search"
    if "internal" in tags:
        return "internal"
    return None


def _append_policy_steps(
    steps: List[TraceStep],
    chunk: str,
    *,
    prefer_order: Sequence[str] = ("think", "internal", "search", "evidence", "answer"),
) -> None:
    """Append TraceSteps for closed tags in a generation chunk (policy tokens)."""
    events: List[Tuple[int, str, str]] = []
    for name in prefer_order:
        cre = _TAG_BODY[name]
        for m in cre.finditer(chunk or ""):
            body = m.group(1).strip()
            if body:
                events.append((m.start(), name, body))
    events.sort(key=lambda x: x[0])
    for _, name, body in events:
        steps.append(
            TraceStep(
                step_id=len(steps),
                step_type=name,
                content=body,
                loss_mask=EXPECTED_LOSS_MASK[name],
            )
        )


@torch.inference_mode()
def _generate_until_action(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
) -> Tuple[str, int, int]:
    """One generate call; stop at first closed action tag when supported."""
    prompt = render_nothink_prompt(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    try:
        out_ids = model.generate(
            **inputs,
            **gen_kwargs,
            stop_strings=list(_STOP_STRINGS),
            tokenizer=tokenizer,
        )
    except TypeError:
        out_ids = model.generate(**inputs, **gen_kwargs)

    gen_ids = out_ids[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    cut = len(text)
    for s in _STOP_STRINGS:
        idx = text.find(s)
        if idx >= 0:
            cut = min(cut, idx + len(s))
    text = text[:cut]
    return text, prompt_len, int(gen_ids.shape[-1])


def run_search_agent_rollout(
    sample: Dict[str, Any],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[RolloutConfig] = None,
) -> RolloutResult:
    cfg = config or RolloutConfig()
    t0 = time.perf_counter()

    question = sample["question"]
    gold = list(sample.get("gold_answers") or [])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": cfg.system_prompt},
        {"role": "user", "content": f"Question: {question}"},
    ]

    steps: List[TraceStep] = []
    all_docs: List[Any] = []
    seen_doc_ids: set = set()
    search_queries: List[str] = []
    raw_gens: List[str] = []
    prompt_tokens = 0
    generated_tokens = 0
    observation_tokens = 0
    search_turns = 0
    finished = False
    route_first = "none"
    hit_max_search = False
    format_nudge_used = False

    for _round in range(cfg.max_rounds):
        chunk, ptok, gtok = _generate_until_action(
            model,
            tokenizer,
            messages,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
        )
        raw_gens.append(chunk)
        prompt_tokens += ptok
        generated_tokens += gtok
        tags = _extract_closed_tags(chunk)
        action = _first_action(tags)

        if route_first == "none":
            has_i = "internal" in tags
            has_s = "search" in tags
            if has_i and has_s:
                route_first = "both"
            elif has_i:
                route_first = "internal"
            elif has_s:
                route_first = "search"

        if action is None:
            loose = _recover_loose_answer(chunk)
            if loose:
                steps.append(
                    TraceStep(
                        step_id=len(steps),
                        step_type="answer",
                        content=loose,
                        loss_mask=True,
                        metadata={"loose_answer": True},
                    )
                )
                finished = True
                break
            if chunk.strip():
                steps.append(
                    TraceStep(
                        step_id=len(steps),
                        step_type="think",
                        content=chunk.strip(),
                        loss_mask=True,
                        metadata={"unparsed_chunk": True},
                    )
                )
            if not format_nudge_used:
                format_nudge_used = True
                messages.append({"role": "assistant", "content": chunk})
                messages.append({"role": "user", "content": _FORMAT_NUDGE})
                continue
            break

        _append_policy_steps(steps, chunk)

        if action == "answer":
            finished = True
            break

        if action == "internal":
            if "answer" in tags:
                finished = True
                break
            messages.append({"role": "assistant", "content": chunk})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You chose <internal>. Now give the final answer in "
                        "<answer>...</answer> (optionally with short <think>)."
                    ),
                }
            )
            continue

        query = tags["search"][-1]
        search_queries.append(query)
        search_turns += 1
        if search_turns > cfg.max_search_turns:
            hit_max_search = True
            break

        packed = retrieve_candidate_bm25(sample, query, top_k=cfg.top_k)
        docs = list(packed.get("documents") or [])
        for d in docs_to_schema(docs):
            if d.document_id not in seen_doc_ids:
                seen_doc_ids.add(d.document_id)
                all_docs.append(d)

        obs_body = format_observation_text(docs)
        observation_tokens += len(tokenizer.encode(obs_body, add_special_tokens=False))

        steps.append(
            TraceStep(
                step_id=len(steps),
                step_type="observation",
                content=obs_body,
                loss_mask=False,
                document_ids=[str(d["document_id"]) for d in docs],
                metadata={
                    "query": query,
                    "retriever": packed.get("retriever"),
                    "search_turn": search_turns,
                },
            )
        )

        messages.append({"role": "assistant", "content": chunk})
        messages.append({"role": "observation", "content": obs_body})

        if search_turns >= cfg.max_search_turns:
            chunk2, ptok2, gtok2 = _generate_until_action(
                model,
                tokenizer,
                messages,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
            )
            raw_gens.append(chunk2)
            prompt_tokens += ptok2
            generated_tokens += gtok2
            tags2 = _extract_closed_tags(chunk2)
            loose2 = _recover_loose_answer(chunk2)
            if "search" in tags2 and "answer" not in tags2 and not loose2:
                hit_max_search = True
            _append_policy_steps(
                steps,
                chunk2,
                prefer_order=("think", "evidence", "answer", "internal"),
            )
            if "answer" not in tags2 and loose2:
                steps.append(
                    TraceStep(
                        step_id=len(steps),
                        step_type="answer",
                        content=loose2,
                        loss_mask=True,
                        metadata={"loose_answer": True},
                    )
                )
            while steps and steps[-1].step_type == "search":
                steps.pop()
                for i, st in enumerate(steps):
                    st.step_id = i
            if any(s.step_type == "answer" for s in steps):
                finished = True
            break

    if not any(s.step_type == "answer" for s in steps):
        steps.append(
            TraceStep(
                step_id=len(steps),
                step_type="answer",
                content="[UNFINISHED]",
                loss_mask=True,
                metadata={"unfinished": True},
            )
        )
    else:
        ans_i = next(i for i, s in enumerate(steps) if s.step_type == "answer")
        steps = steps[: ans_i + 1]
        for i, st in enumerate(steps):
            st.step_id = i

    latency_ms = (time.perf_counter() - t0) * 1000.0
    uniq = sorted(set(search_queries))
    dup = max(0, len(search_queries) - len(uniq))

    pred = ""
    for s in reversed(steps):
        if s.step_type == "answer" and s.content != "[UNFINISHED]":
            pred = s.content
            break

    cost = CostInfo(
        search_count=len(search_queries),
        unique_query_count=len(uniq),
        duplicate_query_count=dup,
        retrieved_document_count=len(all_docs),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        observation_tokens=observation_tokens,
        latency_ms=round(latency_ms, 1),
    )

    evid_meta: Dict[str, Any] = {}
    evid_texts = [s.content for s in steps if s.step_type == "evidence"]
    sf = sample.get("supporting_facts")
    if hasattr(sf, "tolist"):
        sf = list(sf.tolist())
    if evid_texts and sf:
        try:
            joined = "\n\n".join(evid_texts)
            fake_gen = f"<evidence>\n{joined}\n</evidence><answer>\n{pred}\n</answer>"
            evid_meta = score_evidence_use(fake_gen, {**sample, "supporting_facts": sf})
        except Exception as exc:  # noqa: BLE001
            evid_meta = {"evidence_score_error": str(exc)}

    trace = TraceRecord(
        question=question,
        gold_answers=gold,
        trace_id=f"rollout_{uuid.uuid4().hex[:12]}",
        sample_id=sample.get("sample_id") or "",
        steps=steps,
        documents=all_docs,
        cost_info=cost,
        metadata={
            "phase": "3A",
            "finished": finished,
            "route_first": route_first,
            "search_queries": search_queries,
            "hit_max_search_turns": hit_max_search,
            "max_search_turns": cfg.max_search_turns,
            "top_k": cfg.top_k,
            "retriever_scope": "candidate",
            "evidence": evid_meta,
        },
    )
    errors = validate_trace_record(trace)
    if not finished:
        errors.append("rollout_unfinished_no_answer")

    m = {
        "exact_match": exact_match(pred, gold) if pred else 0.0,
        "token_f1": token_f1(pred, gold) if pred else 0.0,
        "finished": float(finished),
        "search_count": float(cost.search_count),
        "duplicate_query_count": float(cost.duplicate_query_count),
        "format_valid": float(len([e for e in errors if "rollout_unfinished" not in e]) == 0),
        "evidence_f1": float((evid_meta or {}).get("evidence_f1") or 0.0),
    }
    if finished and pred:
        try:
            bm = basic_metrics(trace)
            m["exact_match"] = float(bm.get("exact_match", m["exact_match"]))
            m["token_f1"] = float(bm.get("token_f1", m["token_f1"]))
            m["format_valid"] = float(bm.get("format_valid", m["format_valid"]))
        except Exception:
            pass

    return RolloutResult(
        trace=trace,
        finished=finished,
        route_first=route_first,
        search_queries=search_queries,
        validation_errors=errors,
        metrics=m,
        raw_generations=raw_gens,
    )
