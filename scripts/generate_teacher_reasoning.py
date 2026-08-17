"""Phase 2E2: Kimi teacher → structured JSON rationale (code wraps <think>).

Preferred (mode A):
  thinking=disabled + json_schema + max_tokens=512

Smoke A/B/C (5 each):
  python scripts/generate_teacher_reasoning.py --mode abc --max-samples 5 \\
    --run-tag smoke_abc --concurrency 4

Smoke A only (20):
  python scripts/generate_teacher_reasoning.py --mode A --max-samples 20 \\
    --run-tag smoke20_A --concurrency 8

Full over-generate (~550 → filter to ~400 later):
  python scripts/generate_teacher_reasoning.py --mode A \\
    --n-persistent 440 --n-other 110 --concurrency 16 --run-tag teacher550

Env overrides:
  KIMI_BASE_URL / KIMI_API_KEY / KIMI_MODEL / KIMI_CONCURRENCY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.sft.prototype_builder import (  # noqa: E402
    gold_answer_of,
    load_jsonl,
    resolve_evidence_refs,
)
from src.sft.teacher_reasoning import (  # noqa: E402
    DEFAULT_TEACHER_MODEL,
    PROMPT_VERSION,
    REASONING_JSON_SCHEMA,
    SYSTEM_PROMPT,
    format_teacher_user_prompt,
    mine_hard_candidates,
    oracle_em_map_from_metrics,
    validate_teacher_reasoning,
)

DEFAULT_TRAIN = (
    REPO_ROOT / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
)
DEFAULT_DIRECT = (
    REPO_ROOT
    / "results/phase2e1_direct_label_n8000_20260807_202826_phase2e1/labels.jsonl"
)
DEFAULT_BASE_ORACLE = (
    REPO_ROOT
    / "results/phase2e1_base_oracle_n8000_20260807_205154/merged/metrics.json"
)
DEFAULT_SFT_ORACLE = (
    REPO_ROOT
    / "results/phase2e1_sftv0_oracle_n8000_20260807_211627/merged/metrics.json"
)

_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Primary: non-thinking + JSON Schema
    "A": {
        "thinking": "disabled",
        "response_format": "json_schema",
        "temperature": 0.3,
        "max_tokens": 512,
    },
    # Fallback if schema unsupported
    "B": {
        "thinking": "disabled",
        "response_format": "json_object",
        "temperature": 0.3,
        "max_tokens": 512,
    },
    # Ablation: deep thinking + large budget; still consume content JSON only
    "C": {
        "thinking": "enabled",
        "response_format": "json_schema",
        "temperature": 1.0,
        "max_tokens": 16384,
    },
}

_print_lock = threading.Lock()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Kimi grounded rationale cache (JSON → code <think>)."
    )
    p.add_argument("--train-file", type=str, default=str(DEFAULT_TRAIN))
    p.add_argument("--direct-labels", type=str, default=str(DEFAULT_DIRECT))
    p.add_argument("--base-oracle-metrics", type=str, default=str(DEFAULT_BASE_ORACLE))
    p.add_argument("--sft-oracle-metrics", type=str, default=str(DEFAULT_SFT_ORACLE))
    p.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "results"))
    p.add_argument("--run-tag", type=str, default="phase2e2_teacher")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-persistent", type=int, default=320)
    p.add_argument("--n-other", type=int, default=80)
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap chosen candidates (use 5 for abc smoke, 20 for mode smoke).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="A",
        choices=["A", "B", "C", "a", "b", "c", "abc", "ABC"],
        help="A=json_schema+thinking_off; B=json_object+thinking_off; "
        "C=thinking_on ablation; abc=run A/B/C sequentially.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override mode preset temperature.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override mode preset max_tokens / completion budget.",
    )
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--retry-backoff", type=float, default=3.0)
    p.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("KIMI_CONCURRENCY", "8")),
        help="Parallel request workers (default 8; keep modest for VPN endpoint).",
    )
    p.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("KIMI_BASE_URL", "http://10.16.137.2:8000/v1"),
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("KIMI_API_KEY", "EMPTY"),
    )
    p.add_argument(
        "--model",
        type=str,
        default=os.environ.get("KIMI_MODEL", DEFAULT_TEACHER_MODEL),
    )
    p.add_argument(
        "--min-accept-score",
        type=int,
        default=4,
        help="Minimum quality_score (0-5) to accept.",
    )
    return p.parse_args()


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else REPO_ROOT / p


def load_direct(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["sample_id"]] = r
    return rows


def build_response_format(kind: str) -> Dict[str, Any]:
    if kind == "json_schema":
        return {"type": "json_schema", "json_schema": REASONING_JSON_SCHEMA}
    if kind == "json_object":
        return {"type": "json_object"}
    raise ValueError(f"unknown response_format kind: {kind}")


def chat_complete(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    thinking: str,
    response_format_kind: str,
    retries: int = 3,
    retry_backoff: float = 3.0,
    quiet: bool = False,
    allow_thinking_degrade: bool = True,
    reasoning_effort: str = "",
) -> Dict[str, Any]:
    """Return full diagnostic payload (content + reasoning_content + usage...)."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": build_response_format(response_format_kind),
    }
    # Kimi K2.6 thinking switch (ignored harmlessly if server strips unknown fields).
    if thinking in {"disabled", "enabled"}:
        payload["thinking"] = {"type": thinking}
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code >= 400:
                # Fallback: some servers reject json_schema or thinking.
                body_preview = (resp.text or "")[:400]
                if attempt < retries and (
                    "response_format" in body_preview.lower()
                    or "json_schema" in body_preview.lower()
                    or "thinking" in body_preview.lower()
                ):
                    # Progressive degrade within this call.
                    if (
                        allow_thinking_degrade
                        and "thinking" in payload
                        and "thinking" in body_preview.lower()
                    ):
                        payload.pop("thinking", None)
                    elif payload.get("response_format", {}).get("type") == "json_schema":
                        payload["response_format"] = {"type": "json_object"}
                    if not quiet:
                        with _print_lock:
                            print(
                                f"[teacher] HTTP {resp.status_code}; "
                                f"degrading payload and retrying: {body_preview}",
                                flush=True,
                            )
                    time.sleep(retry_backoff * attempt)
                    continue
                resp.raise_for_status()

            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = (
                usage.get("reasoning_tokens")
                or details.get("reasoning_tokens")
                or details.get("reasoning")
            )
            content = message.get("content") or ""
            rc = message.get("reasoning_content")
            # Keep diagnostics even when content is empty (e.g. finish_reason=length).
            # Never persist reasoning_content text; only its length.
            return {
                "content": content,
                "reasoning_content": None,
                "reasoning_content_present": bool(rc),
                "reasoning_content_len": len(rc or ""),
                "content_len": len(content),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "reasoning_tokens": reasoning_tokens,
                },
                "model": data.get("model") or model,
                "raw_response_keys": sorted(list(data.keys())),
                "request_thinking": thinking,
                "thinking_payload_kept": "thinking" in payload,
                "request_response_format": payload.get("response_format", {}).get("type"),
                "http_status": resp.status_code,
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt >= retries:
                break
            sleep_s = retry_backoff * attempt
            if not quiet:
                with _print_lock:
                    print(
                        f"[teacher] retry {attempt}/{retries} after error: {exc}; "
                        f"sleep {sleep_s}s",
                        flush=True,
                    )
            time.sleep(sleep_s)
    assert last_err is not None
    raise last_err


def _pool_tag(sid: str, direct: Dict[str, Any], base_oracle: Dict[str, Any]) -> str:
    d = direct.get(sid) or {}
    d_ok = bool(d.get("direct_correct")) or float(d.get("exact_match") or 0) >= 1.0 - 1e-9
    o_ok = float((base_oracle.get(sid) or {}).get("exact_match") or 0) >= 1.0 - 1e-9
    if (not d_ok) and (not o_ok):
        return "persistent_c_like"
    return "other_hard"


def process_one(
    idx: int,
    total: int,
    sid: str,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    direct: Dict[str, Dict[str, Any]],
    base_oracle: Dict[str, Dict[str, Any]],
    mode_cfg: Dict[str, Any],
    mode_name: str,
) -> Tuple[int, Dict[str, Any]]:
    refs = resolve_evidence_refs(sample)
    gold = gold_answer_of(sample)
    user = format_teacher_user_prompt(sample, refs, gold)
    t0 = time.time()
    err = None
    api_meta: Dict[str, Any] = {}
    raw = ""
    try:
        api_meta = chat_complete(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=mode_cfg["temperature"],
            max_tokens=mode_cfg["max_tokens"],
            timeout=args.timeout,
            thinking=mode_cfg["thinking"],
            response_format_kind=mode_cfg["response_format"],
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            quiet=args.concurrency > 1,
        )
        raw = api_meta.get("content") or ""
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
    latency_ms = round((time.time() - t0) * 1000, 1)

    if err is None:
        validation = validate_teacher_reasoning(
            raw,
            gold_answer=gold,
            question=sample["question"],
            refs=refs,
            min_accept_score=args.min_accept_score,
        )
    else:
        validation = {
            "parse_ok": False,
            "format_valid": False,
            "answer_consistent": False,
            "grounding_valid": False,
            "evidence1_used": False,
            "evidence2_used": False,
            "length_valid": False,
            "meta_clean": False,
            "bridge_ok": False,
            "n_words": 0,
            "novel_proper_nouns": [],
            "quality_score": 0,
            "errors": [err or "api_error"],
            "accepted": False,
            "think": None,
            "reasoning": None,
            "think_wrapped": None,
        }

    # Diagnostic only — never use reasoning_content as SFT target.
    rc = api_meta.get("reasoning_content")
    rc_preview = None
    if isinstance(rc, str) and rc.strip():
        rc_preview = rc.strip()[:500]

    row = {
        "sample_id": sid,
        "question": sample["question"],
        "gold_answer": gold,
        "gold_answers": list(sample.get("gold_answers") or []),
        "evidence_refs": refs,
        "q_type": (sample.get("metadata") or {}).get("type"),
        "teacher_model": args.model,
        "teacher_prompt_version": PROMPT_VERSION,
        "teacher_base_url": args.base_url,
        "teacher_mode": mode_name,
        "teacher_mode_config": mode_cfg,
        "teacher_raw_output": raw,
        "teacher_api": {
            "finish_reason": api_meta.get("finish_reason"),
            "usage": api_meta.get("usage"),
            "model": api_meta.get("model"),
            "reasoning_content_present": api_meta.get("reasoning_content_present"),
            "reasoning_content_preview": rc_preview,
            "request_thinking": api_meta.get("request_thinking"),
            "request_response_format": api_meta.get("request_response_format"),
            "http_status": api_meta.get("http_status"),
        },
        "teacher_validation": {
            k: validation.get(k)
            for k in (
                "parse_ok",
                "format_valid",
                "answer_consistent",
                "grounding_valid",
                "evidence1_used",
                "evidence2_used",
                "length_valid",
                "meta_clean",
                "bridge_ok",
                "n_words",
                "novel_proper_nouns",
                "quality_score",
                "errors",
                "accepted",
            )
        },
        "think": validation.get("think"),
        "think_wrapped": validation.get("think_wrapped"),
        "latency_ms": latency_ms,
        "api_error": err,
        "reasoning_source": "kimi2.6",
        "pool": _pool_tag(sid, direct, base_oracle),
    }
    status = "OK" if validation["accepted"] else "REJECT"
    with _print_lock:
        print(
            f"[{mode_name} {idx}/{total}] {sid} {status} "
            f"score={validation.get('quality_score')} "
            f"words={validation.get('n_words')} "
            f"finish={api_meta.get('finish_reason')} "
            f"lat={latency_ms}ms "
            f"err={validation.get('errors')[:2]}",
            flush=True,
        )
    return idx, row


def run_mode(
    *,
    mode_name: str,
    chosen: List[str],
    by_id: Dict[str, Dict[str, Any]],
    direct: Dict[str, Dict[str, Any]],
    base_oracle: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
    mine_stats: Dict[str, Any],
) -> Dict[str, Any]:
    preset = dict(_MODE_PRESETS[mode_name])
    if args.temperature is not None:
        preset["temperature"] = args.temperature
    if args.max_tokens is not None:
        preset["max_tokens"] = args.max_tokens

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = resolve(args.output_dir) / (
        f"teacher_reasoning_n{len(chosen)}_{stamp}_{args.run_tag}_mode{mode_name}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_dir / "reasoning_cache.jsonl"
    summary_path = run_dir / "summary.json"

    workers = min(args.concurrency, max(len(chosen), 1))
    print(f"\n[teacher] ===== MODE {mode_name} =====", flush=True)
    print(f"[teacher] config={preset}", flush=True)
    print(f"[teacher] model={args.model} base_url={args.base_url}", flush=True)
    print(f"[teacher] run_dir={run_dir}", flush=True)
    print(
        f"[teacher] generating n={len(chosen)} concurrency={workers}",
        flush=True,
    )

    t_all = time.time()
    n_ok = 0
    n_fail = 0
    n_done = 0
    n_parse = 0
    finish_reasons: Dict[str, int] = {}
    score_hist: Dict[str, int] = {}
    rows_out: List[Dict[str, Any]] = []

    with cache_path.open("w", encoding="utf-8") as out, ThreadPoolExecutor(
        max_workers=workers
    ) as ex:
        futs = [
            ex.submit(
                process_one,
                i,
                len(chosen),
                sid,
                by_id[sid],
                args,
                direct,
                base_oracle,
                preset,
                mode_name,
            )
            for i, sid in enumerate(chosen, 1)
        ]
        print(f"[teacher] submitted {len(futs)} jobs", flush=True)
        for fut in as_completed(futs):
            _idx, row = fut.result()
            rows_out.append(row)
            n_done += 1
            tv = row["teacher_validation"]
            if tv.get("accepted"):
                n_ok += 1
            else:
                n_fail += 1
            if tv.get("parse_ok"):
                n_parse += 1
            fr = (row.get("teacher_api") or {}).get("finish_reason") or "unknown"
            finish_reasons[str(fr)] = finish_reasons.get(str(fr), 0) + 1
            sc = str(tv.get("quality_score"))
            score_hist[sc] = score_hist.get(sc, 0) + 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if n_done % max(1, min(5, len(chosen))) == 0 or n_done == len(chosen):
                elapsed_so_far = time.time() - t_all
                print(
                    f"[teacher] progress {n_done}/{len(chosen)} "
                    f"ok={n_ok} reject={n_fail} parse={n_parse} "
                    f"elapsed={elapsed_so_far:.1f}s",
                    flush=True,
                )

    elapsed = round(time.time() - t_all, 2)
    n = max(len(chosen), 1)
    avg_words = 0.0
    word_vals = [
        r["teacher_validation"].get("n_words") or 0
        for r in rows_out
        if r["teacher_validation"].get("accepted")
    ]
    if word_vals:
        avg_words = round(sum(word_vals) / len(word_vals), 2)

    summary = {
        "mode": mode_name,
        "mode_config": preset,
        "num_requested": len(chosen),
        "num_accepted": n_ok,
        "num_rejected": n_fail,
        "num_parse_ok": n_parse,
        "accept_rate": round(n_ok / n, 4),
        "parse_rate": round(n_parse / n, 4),
        "answer_consistency_rate": round(
            sum(
                1
                for r in rows_out
                if r["teacher_validation"].get("answer_consistent")
            )
            / n,
            4,
        ),
        "grounding_rate": round(
            sum(1 for r in rows_out if r["teacher_validation"].get("grounding_valid"))
            / n,
            4,
        ),
        "avg_accepted_words": avg_words,
        "finish_reasons": finish_reasons,
        "quality_score_hist": score_hist,
        "concurrency": workers,
        "elapsed_seconds": elapsed,
        "throughput_qps": round(len(chosen) / max(elapsed, 1e-6), 3),
        "mine_stats": mine_stats,
        "teacher_model": args.model,
        "teacher_prompt_version": PROMPT_VERSION,
        "cache_path": str(cache_path),
        "run_dir": str(run_dir),
        "phase": "2E2",
        "purpose": "kimi_json_rationale_smoke"
        if (args.max_samples or 0) <= 50
        else "kimi_json_rationale",
        "io_contract": "teacher_json_reasoning__code_wraps_think",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[teacher] artifacts -> {run_dir}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    mode_arg = args.mode.upper()
    modes = ["A", "B", "C"] if mode_arg == "ABC" else [mode_arg]

    train_path = resolve(args.train_file)
    samples = load_jsonl(str(train_path))
    by_id = {s["sample_id"]: s for s in samples}
    direct = load_direct(resolve(args.direct_labels))
    base_oracle = oracle_em_map_from_metrics(
        json.loads(resolve(args.base_oracle_metrics).read_text(encoding="utf-8"))
    )
    sft_oracle = oracle_em_map_from_metrics(
        json.loads(resolve(args.sft_oracle_metrics).read_text(encoding="utf-8"))
    )

    chosen, mine_stats = mine_hard_candidates(
        samples_by_id=by_id,
        direct=direct,
        base_oracle=base_oracle,
        sft_oracle=sft_oracle,
        seed=args.seed,
        n_persistent=args.n_persistent,
        n_other=args.n_other,
    )
    if args.max_samples is not None:
        chosen = chosen[: args.max_samples]

    print(f"[teacher] mine_stats={mine_stats}", flush=True)
    print(f"[teacher] modes={modes} n={len(chosen)} seed={args.seed}", flush=True)

    summaries = []
    for m in modes:
        summaries.append(
            run_mode(
                mode_name=m,
                chosen=chosen,
                by_id=by_id,
                direct=direct,
                base_oracle=base_oracle,
                args=args,
                mine_stats=mine_stats,
            )
        )

    if len(summaries) > 1:
        compare_path = (
            resolve(args.output_dir)
            / f"teacher_mode_compare_{time.strftime('%Y%m%d_%H%M%S')}_{args.run_tag}.json"
        )
        compare = {
            "modes": summaries,
            "selection_hint": (
                "Prefer A if parse_rate>=0.95 and accept_rate>=0.70; "
                "else B; use C only if clearly better rationale quality."
            ),
        }
        compare_path.write_text(json.dumps(compare, ensure_ascii=False, indent=2) + "\n")
        print(f"[teacher] mode compare -> {compare_path}", flush=True)
        for s in summaries:
            print(
                f"[teacher] MODE {s['mode']}: parse={s['parse_rate']} "
                f"accept={s['accept_rate']} "
                f"ans={s['answer_consistency_rate']} "
                f"ground={s['grounding_rate']} "
                f"finish={s['finish_reasons']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
