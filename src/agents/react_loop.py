"""Phase 3A: multi-turn Search Agent loop (generate → tool → observe → continue).

No training. Observation tokens are environment-injected (loss_mask=False in Trace).
Harness v1: Qwen3 no-think prefix + LlamaFactory qwen3_nothink tool_response slot.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.eval.metrics import basic_metrics, exact_match, hotpot_joint, token_f1, token_prf
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

_STOP_STRINGS = ("</search>", "</answer>", "</internal>", "</evidence>")

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
    max_evidence_tokens: int = 512
    max_answer_tokens: int = 256
    max_rounds: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
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
_FINALIZE_NUDGE = (
    "Search budget is used up. Do not output <search>. "
    "Write <evidence> if needed, then <answer>short answer</answer>."
)
_ANSWER_ONLY_NUDGE = (
    "Now give only the final answer in <answer>...</answer>."
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


def _recover_open_evidence(text: str) -> Optional[str]:
    """Unclosed <evidence> body after the evidence budget is exhausted."""
    raw = text or ""
    low = raw.lower()
    if "<evidence>" not in low or "</evidence>" in low:
        return None
    idx = low.find("<evidence>")
    body = raw[idx + len("<evidence>") :].strip()
    return body or None


def _token_budget(phase: str, cfg: RolloutConfig) -> int:
    """Search actions share max_new_tokens; finalize splits evidence vs answer."""
    if phase == "answer":
        return int(cfg.max_answer_tokens)
    if phase == "evidence":
        return int(cfg.max_evidence_tokens)
    return int(cfg.max_new_tokens)


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


GenerateFn = Callable[..., Tuple[str, int, int]]


def _truncate_at_stop(text: str) -> str:
    cut = len(text or "")
    for s in _STOP_STRINGS:
        idx = (text or "").find(s)
        if idx >= 0:
            cut = min(cut, idx + len(s))
    return (text or "")[:cut]


def make_vllm_generate_fn(llm: Any) -> GenerateFn:
    """vLLM backend: consume already-rendered Harness v1 prompts. No chat template."""
    from vllm import SamplingParams

    def _fn(
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        kwargs: Dict[str, Any] = {
            "temperature": 0.0 if temperature <= 0 else float(temperature),
            "max_tokens": int(max_new_tokens),
            "stop": list(_STOP_STRINGS),
        }
        if temperature > 0:
            kwargs["top_p"] = float(top_p)
        if seed is not None:
            kwargs["seed"] = int(seed)
        try:
            params = SamplingParams(**kwargs, include_stop_str_in_output=True)
        except TypeError:
            params = SamplingParams(**kwargs)
        outs = llm.generate([prompt], params, use_tqdm=False)
        out = outs[0]
        text = _truncate_at_stop(out.outputs[0].text)
        ptok = len(out.prompt_token_ids or [])
        gtok = len(out.outputs[0].token_ids or [])
        return text, ptok, gtok

    return _fn


def make_openai_completions_fn(base_url: str, model: str) -> GenerateFn:
    """Call a running vLLM OpenAI server. Prompt is already Harness v1 text."""
    import json
    from urllib.request import Request, urlopen

    url = base_url.rstrip("/") + "/completions"

    def _fn(
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[str, int, int]:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0 if temperature <= 0 else float(temperature),
            "stop": list(_STOP_STRINGS),
            "include_stop_str_in_output": True,
        }
        if temperature > 0:
            payload["top_p"] = float(top_p)
        if seed is not None:
            payload["seed"] = int(seed)
        req = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = _truncate_at_stop((data.get("choices") or [{}])[0].get("text") or "")
        usage = data.get("usage") or {}
        return text, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    return _fn


@torch.inference_mode()
def _generate_until_action(
    model: Optional[PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    *,
    max_new_tokens: int,
    temperature: float,
    generate_fn: Optional[GenerateFn] = None,
    top_p: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[str, int, int]:
    """One generate call; stop at first closed action tag when supported."""
    prompt = render_nothink_prompt(tokenizer, messages)
    if generate_fn is not None:
        return generate_fn(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    if model is None:
        raise ValueError("HF generate needs a model when generate_fn is None")
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
        gen_kwargs["top_p"] = top_p

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
    text = _truncate_at_stop(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return text, prompt_len, int(gen_ids.shape[-1])


def run_search_agent_rollout(
    sample: Dict[str, Any],
    model: Optional[PreTrainedModel],
    tokenizer: PreTrainedTokenizerBase,
    config: Optional[RolloutConfig] = None,
    generate_fn: Optional[GenerateFn] = None,
    generation_seed: Optional[int] = None,
) -> RolloutResult:
    cfg = config or RolloutConfig()
    if generate_fn is None and model is None:
        raise ValueError("run_search_agent_rollout needs model or generate_fn")
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
    finalize_only = False
    phase = "act"  # act | evidence | answer
    answer_reserve_used = False

    def _go_answer_reserve(chunk: str) -> bool:
        nonlocal finalize_only, phase, answer_reserve_used
        if answer_reserve_used:
            return False
        messages.append({"role": "assistant", "content": chunk})
        messages.append({"role": "user", "content": _ANSWER_ONLY_NUDGE})
        finalize_only = True
        phase = "answer"
        answer_reserve_used = True
        return True

    def _keep_truncated_evidence(chunk: str, tags: Dict[str, List[str]]) -> None:
        open_ev = _recover_open_evidence(chunk)
        if not open_ev or tags.get("evidence"):
            return
        steps.append(
            TraceStep(
                step_id=len(steps),
                step_type="evidence",
                content=open_ev,
                loss_mask=True,
                metadata={"truncated_evidence": True},
            )
        )

    for _round in range(cfg.max_rounds):
        chunk, ptok, gtok = _generate_until_action(
            model,
            tokenizer,
            messages,
            max_new_tokens=_token_budget(phase, cfg),
            temperature=cfg.temperature,
            generate_fn=generate_fn,
            top_p=cfg.top_p,
            seed=generation_seed,
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
            closed_ev = "evidence" in tags
            open_ev = _recover_open_evidence(chunk)
            search_exhausted = (
                finalize_only
                or phase in ("evidence", "answer")
                or search_turns >= cfg.max_search_turns
            )
            # Closed evidence = model chose to finalize. Unclosed evidence
            # only finalizes after the search budget is gone; otherwise it
            # would turn a second-search attempt into a 1-search answer.
            if closed_ev or (open_ev and search_exhausted):
                _append_policy_steps(steps, chunk)
                _keep_truncated_evidence(chunk, tags)
                if phase == "answer" or not _go_answer_reserve(chunk):
                    break
                continue
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
            if phase == "answer":
                break
            if finalize_only or phase == "evidence":
                if not _go_answer_reserve(chunk):
                    break
                continue
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
            if phase == "answer" or not _go_answer_reserve(chunk):
                break
            continue

        if finalize_only or search_turns >= cfg.max_search_turns or phase != "act":
            hit_max_search = True
            while steps and steps[-1].step_type == "search":
                steps.pop()
            for i, st in enumerate(steps):
                st.step_id = i
            if phase == "answer" or not _go_answer_reserve(chunk):
                break
            continue

        query = tags["search"][-1]
        search_queries.append(query)
        search_turns += 1

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
            finalize_only = True
            phase = "evidence"
        continue

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

    ans_prf = token_prf(pred, gold) if pred else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    ev_p = float((evid_meta or {}).get("evidence_precision") or 0.0)
    ev_r = float((evid_meta or {}).get("evidence_recall") or 0.0)
    ev_f1 = float((evid_meta or {}).get("evidence_f1") or 0.0)
    ev_em = 1.0 if ev_p >= 1.0 and ev_r >= 1.0 and ev_f1 >= 1.0 else 0.0
    ans_em = exact_match(pred, gold) if pred else 0.0
    joint = hotpot_joint(ans_em, ans_prf["precision"], ans_prf["recall"], ev_em, ev_p, ev_r)
    m = {
        "exact_match": ans_em,
        "token_f1": float(ans_prf["f1"]),
        "token_precision": float(ans_prf["precision"]),
        "token_recall": float(ans_prf["recall"]),
        "finished": float(finished),
        "search_count": float(cost.search_count),
        "duplicate_query_count": float(cost.duplicate_query_count),
        "format_valid": float(len([e for e in errors if "rollout_unfinished" not in e]) == 0),
        "evidence_f1": ev_f1,
        "evidence_precision": ev_p,
        "evidence_recall": ev_r,
        "evidence_em": ev_em,
        "joint_f1": float(joint["joint_f1"]),
        "joint_em": float(joint["joint_em"]),
    }
    if finished and pred:
        try:
            bm = basic_metrics(trace)
            m["exact_match"] = float(bm.get("exact_match", m["exact_match"]))
            m["token_f1"] = float(bm.get("token_f1", m["token_f1"]))
            m["format_valid"] = float(bm.get("format_valid", m["format_valid"]))
        except Exception:
            pass
        joint = hotpot_joint(
            m["exact_match"],
            m["token_precision"],
            m["token_recall"],
            ev_em,
            ev_p,
            ev_r,
        )
        m["joint_f1"] = float(joint["joint_f1"])
        m["joint_em"] = float(joint["joint_em"])

    return RolloutResult(
        trace=trace,
        finished=finished,
        route_first=route_first,
        search_queries=search_queries,
        validation_errors=errors,
        metrics=m,
        raw_generations=raw_gens,
    )
