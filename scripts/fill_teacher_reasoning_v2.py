#!/usr/bin/env python3
"""Fill frozen 1200 evidence_reasoning slots with DeepSeek-V4-Flash.

Teacher writes only a 2-6 sentence grounded rationale. Internal thinking is
enabled but reasoning_content is never persisted. Do not change sample ids.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SHARED_DATA = Path("/data1/hcc/deepresearch")

from src.sft.prototype_builder import gold_answer_of, load_jsonl, resolve_evidence_refs
from src.sft.teacher_reasoning import (
    DEFAULT_TEACHER_MODEL,
    PROMPT_VERSION_V4,
    SYSTEM_PROMPT_V4,
    format_teacher_user_prompt,
    validate_teacher_reasoning,
    wrap_think,
)

PLACEHOLDER = "__TEACHER_REASONING_PENDING__"
FROZEN_IDS = REPO / "results/17_build_8b_coldstart_v2/ids_evidence_reasoning.json"
MANIFEST = REPO / "results/16_select_8b_coldstart_v2/selection_manifest.jsonl"
POOL = SHARED_DATA / "data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
_SENT_RE = re.compile(r"[.!?]+")
_XML_RE = re.compile(r"</?(?:search|observation|evidence|think|answer|internal)\b", re.I)
BANDS = ("genuine_hard", "medium", "near_solved")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument(
        "--smoke-per-band",
        type=int,
        default=0,
        help="If >0, take this many ids from each reasoning band (10+10+10=30).",
    )
    p.add_argument("--debug", action="store_true")
    p.add_argument("--ids-json", type=Path, default=FROZEN_IDS)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--pool", type=Path, default=POOL)
    p.add_argument("--canonical", type=Path, default=REPO / "results/17_build_8b_coldstart_v2/canonical.jsonl")
    p.add_argument("--sharegpt", type=Path, default=REPO / "results/17_build_8b_coldstart_v2/sharegpt.jsonl")
    p.add_argument(
        "--base-url",
        default=os.environ.get("TEACHER_BASE_URL", "https://api.deepseek.com"),
    )
    p.add_argument(
        "--api-key",
        default=(os.environ.get("DEEPSEEK_API_KEY") or "EMPTY").strip(),
    )
    p.add_argument(
        "--model",
        default=os.environ.get("TEACHER_MODEL", DEFAULT_TEACHER_MODEL),
    )
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("TEACHER_CONCURRENCY", "8")))
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument(
        "--reasoning-effort",
        default=os.environ.get("TEACHER_REASONING_EFFORT", "max"),
    )
    p.add_argument("--apply", action="store_true")
    p.add_argument("--exist-ok", action="store_true")
    p.add_argument(
        "--rescore-only",
        action="store_true",
        help="Re-validate an existing teacher_cache.jsonl; do not call the API.",
    )
    p.add_argument(
        "--retry-ids",
        type=Path,
        default=None,
        help="JSON list of sample_ids to regenerate even if cached. Empty only.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print FROZEN/CACHED/RETRY/TODO and exit. Never call the API.",
    )
    p.add_argument(
        "--replacement-audit",
        type=Path,
        default=None,
        help="QC replacement JSON. Default: <output-dir>/replacement_audit.json if present.",
    )
    return p.parse_args()


def load_gen():
    path = REPO / "scripts/generate_teacher_reasoning.py"
    spec = importlib.util.spec_from_file_location("generate_teacher_reasoning", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def n_sentences(text: str) -> int:
    return len([p for p in _SENT_RE.split(text or "") if p.strip()])


def load_cache(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row["sample_id"])] = row
    return out


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite_cache(path: Path, cache: dict[str, dict], frozen: list[str]) -> None:
    """Write one row per frozen id so retries do not leave duplicate lines."""
    with path.open("w", encoding="utf-8") as handle:
        for sid in frozen:
            row = cache.get(sid)
            if row is None:
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_replacement_audit(args: argparse.Namespace) -> dict | None:
    path = args.replacement_audit
    if path is None:
        path = args.output_dir / "replacement_audit.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dropped_id", "replacement_id", "band"):
        if not data.get(key):
            raise SystemExit(f"REPLACEMENT_AUDIT_MISSING {key}")
    return data


def overlay_replacement_band(band_map: dict[str, str], audit: dict | None) -> dict[str, str]:
    if audit:
        band_map[str(audit["replacement_id"])] = str(audit["band"])
    return band_map


def assert_replacement_ids(frozen: list[str], audit: dict | None) -> None:
    if not audit:
        return
    dropped = str(audit["dropped_id"])
    added = str(audit["replacement_id"])
    original = json.loads(FROZEN_IDS.read_text(encoding="utf-8"))
    expected = sorted((set(original) - {dropped}) | {added})
    if sorted(frozen) != expected:
        raise SystemExit(
            f"REPLACEMENT_IDS_MISMATCH frozen={len(frozen)} expected={len(expected)} "
            f"dropped_in_frozen={dropped in set(frozen)} "
            f"replacement_in_frozen={added in set(frozen)}"
        )
    print(f"REPLACEMENT_OK DROP={dropped} ADD={added} FROZEN={len(frozen)}", flush=True)


def load_retry_ids(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.is_file():
        raise SystemExit(f"RETRY_IDS_MISSING {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise SystemExit("RETRY_IDS must be a non-empty JSON list")
    ids = [str(x) for x in data]
    if len(ids) != len(set(ids)):
        raise SystemExit("RETRY_IDS_DUPLICATE")
    return ids


def plan_todo(
    frozen: list[str],
    cache: dict[str, dict],
    retry_ids: list[str],
) -> tuple[list[str], list[str]]:
    frozen_set = set(frozen)
    retry_set = set(retry_ids)
    for sid in retry_ids:
        if sid not in frozen_set:
            raise SystemExit(f"RETRY_ID_NOT_FROZEN {sid}")
        old = cache.get(sid) or {}
        if (old.get("reasoning") or "").strip():
            raise SystemExit(f"REFUSE_OVERWRITE_NONEMPTY {sid}")
    todo: list[str] = []
    for sid in frozen:
        if sid in retry_set or sid not in cache:
            todo.append(sid)
    kept = [sid for sid in frozen if sid not in todo]
    if retry_ids:
        if len(todo) != len(retry_ids):
            raise SystemExit(
                f"TODO_NE_RETRY todo={len(todo)} retry={len(retry_ids)}; "
                "refusing to expand beyond the retry list"
            )
        if not all(not (cache.get(sid) or {}).get("reasoning", "").strip() for sid in retry_ids):
            raise SystemExit("RETRY_IDS_NOT_ALL_EMPTY")
    return todo, kept


def load_band_map(manifest: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not manifest.is_file():
        return out
    for row in load_jsonl(str(manifest)):
        if row.get("category") != "evidence_reasoning":
            continue
        band = row.get("reasoning_band")
        if band:
            out[str(row["sample_id"])] = str(band)
    return out


def pick_stratified(
    frozen: list[str],
    band_map: dict[str, str],
    per_band: int,
    seed: int,
) -> list[str]:
    by_band: dict[str, list[str]] = {b: [] for b in BANDS}
    for sid in frozen:
        band = band_map.get(sid)
        if band in by_band:
            by_band[band].append(sid)
    rng = random.Random(seed)
    picked: list[str] = []
    for band in BANDS:
        ids = list(by_band[band])
        rng.shuffle(ids)
        if len(ids) < per_band:
            raise SystemExit(f"band {band} has {len(ids)} < {per_band}")
        picked.extend(ids[:per_band])
    return picked


def teacher_gate(rows: list[dict], frozen: list[str], audit: dict | None = None) -> dict:
    n = len(frozen)
    got = {r["sample_id"] for r in rows}
    accepted = [r for r in rows if r.get("accepted")]
    grounded = [r for r in rows if r.get("grounding_valid")]
    derived = [r for r in rows if r.get("answer_consistent")]
    covered = [r for r in rows if r.get("evidence1_used") and r.get("evidence2_used")]
    xml = [r for r in rows if _XML_RE.search(r.get("reasoning") or "")]
    leak = [r for r in rows if not r.get("meta_clean", True)]
    sent_ok = [r for r in rows if 2 <= int(r.get("n_sentences") or 0) <= 6]
    pending = sum(1 for r in rows if PLACEHOLDER in (r.get("reasoning") or ""))
    saved_rc = [r for r in rows if r.get("reasoning_content_saved")]
    present_rc = [r for r in rows if r.get("reasoning_content_present")]
    empty = [r for r in rows if not (r.get("reasoning") or "").strip()]
    hard_fail = [r for r in rows if r.get("hard_reject")]
    comps = [int((r.get("usage") or {}).get("completion_tokens") or 0) for r in rows]
    rtoks = [int((r.get("usage") or {}).get("reasoning_tokens") or 0) for r in rows]
    rtoks_sorted = sorted(rtoks)
    def _pct(xs: list[int], p: float) -> float:
        if not xs:
            return 0.0
        idx = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
        return float(xs[idx])
    rate = lambda xs: round(len(xs) / max(len(rows), 1), 4)
    by_tier: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("reasoning_band") or "unknown")].append(row)
    for band, items in grouped.items():
        bt = [int((r.get("usage") or {}).get("reasoning_tokens") or 0) for r in items]
        lt = [float(r.get("latency_s") or 0) for r in items]
        by_tier[band] = {
            "n": len(items),
            "mean_reasoning_tokens": round(sum(bt) / max(len(bt), 1), 1),
            "mean_latency_s": round(sum(lt) / max(len(lt), 1), 2),
        }
    thinking_signal = rate(present_rc) > 0 or sum(rtoks) > 0
    hard = {
        "n_success": len(rows) == n,
        "unique": len(got) == n and got == set(frozen),
        "pending_zero": pending == 0,
        "empty_zero": len(empty) == 0,
        "xml_or_meta_leak_zero": len(xml) + len(leak) == 0,
        "ids_unchanged": got <= set(frozen),
        "reasoning_content_not_saved": len(saved_rc) == 0,
        "thinking_on": thinking_signal,
    }
    if n >= 1200:
        hard.update(
            {
                "grounded_ge_98": rate(grounded) >= 0.98,
                "answer_ge_98": rate(derived) >= 0.98,
                "hard_fail_le_1pct": rate(hard_fail) <= 0.01,
            }
        )
        if audit:
            dropped = str(audit["dropped_id"])
            added = str(audit["replacement_id"])
            hard["dropped_absent"] = dropped not in got and dropped not in set(frozen)
            hard["replacement_present"] = added in got
        gate = "GATE_TEACHER1200_PASS" if all(hard.values()) else "GATE_TEACHER1200_FAIL"
    elif n == 30:
        hard.update(
            {
                "grounded_ge_29": len(grounded) >= 29,
                "answer_ge_29": len(derived) >= 29,
                "hard_fail_le_1": len(hard_fail) <= 1,
                "len_2to6_ge_90": rate(sent_ok) >= 0.90,
            }
        )
        gate = (
            "GATE_DEEPSEEK_TEACHER_SMOKE_PASS"
            if all(hard.values())
            else "GATE_DEEPSEEK_TEACHER_SMOKE_FAIL"
        )
    else:
        hard["generation_success"] = len(rows) == n
        gate = (
            "GATE_TEACHER1200_SMOKE_PASS"
            if hard["n_success"] and hard["unique"] and hard["xml_or_meta_leak_zero"]
            and hard["reasoning_content_not_saved"]
            else "GATE_TEACHER1200_FAIL"
        )
    return {
        "gate": gate,
        "n_input": n,
        "n_success": len(rows),
        "unique": len(got),
        "pending": pending,
        "empty_rationale": len(empty),
        "hard_fail": len(hard_fail),
        "accepted_rate": rate(accepted),
        "grounded_rate": rate(grounded),
        "answer_derivation_rate": rate(derived),
        "evidence_coverage_rate": rate(covered),
        "xml_leak": len(xml),
        "meta_leak": len(leak),
        "len_2to6_rate": rate(sent_ok),
        "sample_id_delta": len(got - set(frozen)),
        "reasoning_content_saved": len(saved_rc),
        "reasoning_content_present_rate": rate(present_rc),
        "mean_completion_tokens": round(sum(comps) / max(len(comps), 1), 1),
        "mean_reasoning_tokens": round(sum(rtoks) / max(len(rtoks), 1), 1),
        "median_reasoning_tokens": _pct(rtoks_sorted, 50),
        "p90_reasoning_tokens": _pct(rtoks_sorted, 90),
        "max_reasoning_tokens": max(rtoks) if rtoks else 0,
        "thinking_signal": thinking_signal,
        "coverage_is_soft_audit": True,
        "dropped_present": int(bool(audit) and str(audit["dropped_id"]) in got),
        "replacement_present": int(bool(audit) and str(audit["replacement_id"]) in got),
        "by_tier": by_tier,
        "hard_gates": hard,
        "prompt_version": PROMPT_VERSION_V4,
        "teacher_model": "deepseek-v4-flash",
        "request_thinking": "enabled",
        "reasoning_effort": "max",
        "response_format": "json_object",
        "max_tokens": 4096,
    }


def apply_fill(canonical: list[dict], sharegpt: list[dict], cache: dict[str, dict]) -> tuple[list[dict], list[dict], int]:
    filled_n = 0
    by_id = {r["sample_id"]: cache[r["sample_id"]] for r in canonical if r["sample_id"] in cache}

    def fill_target(text: str, reasoning: str) -> str:
        return text.replace(PLACEHOLDER, reasoning.strip())

    new_canon = []
    for row in canonical:
        if row["category"] != "evidence_reasoning":
            new_canon.append(row)
            continue
        hit = by_id.get(row["sample_id"])
        if not hit or not hit.get("reasoning"):
            new_canon.append(row)
            continue
        row = dict(row)
        row["target"] = fill_target(row["target"], hit["reasoning"])
        row["provenance"] = dict(row.get("provenance") or {})
        row["provenance"]["reasoning_source"] = "deepseek-v4-flash"
        row["provenance"]["teacher_id"] = hit.get("model")
        row["metadata"] = dict(row.get("metadata") or {})
        row["metadata"]["teacher_pending"] = False
        new_canon.append(row)
        filled_n += 1
    new_share = []
    for row in sharegpt:
        if row["category"] != "evidence_reasoning":
            new_share.append(row)
            continue
        hit = by_id.get(row["sample_id"])
        if not hit or not hit.get("reasoning"):
            new_share.append(row)
            continue
        row = dict(row)
        conv = []
        for turn in row["conversations"]:
            turn = dict(turn)
            if turn.get("from") == "gpt" and PLACEHOLDER in (turn.get("value") or ""):
                turn["value"] = fill_target(turn["value"], hit["reasoning"])
            conv.append(turn)
        row["conversations"] = conv
        row["metadata"] = dict(row.get("metadata") or {})
        row["metadata"]["reasoning_source"] = "deepseek-v4-flash"
        new_share.append(row)
    return new_canon, new_share, filled_n


def probe_teacher(base_url: str, api_key: str, model: str) -> None:
    key = (api_key or "").strip()
    if not key or key == "EMPTY":
        raise SystemExit("MISSING_DEEPSEEK_API_KEY: export a new key; do not paste it into chat.")
    if len(key) < 16:
        raise SystemExit(f"TEACHER_KEY_TOO_SHORT len={len(key)}")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with PING_OK only."}],
        "max_tokens": 16,
        "thinking": {"type": "disabled"},
    }
    try:
        resp = __import__("requests").post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            json=payload,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"TEACHER_UNREACHABLE {url}: {exc}") from exc
    if resp.status_code == 401:
        raise SystemExit(
            "TEACHER_AUTH_401: DeepSeek rejected the key. "
            "Revoke the leaked key, create a new one, `read -s` import it, "
            "then rerun. Do not paste the key into chat."
        )
    if resp.status_code >= 400:
        raise SystemExit(f"TEACHER_PROBE_HTTP {resp.status_code}: {resp.text[:300]}")
    print(f"TEACHER_PROBE_OK key_len={len(key)} model={model}", flush=True)


def run_rescore_only(args: argparse.Namespace) -> None:
    """Re-score an existing cache. Must never import/call the Teacher API."""
    out = args.output_dir
    cache_path = out / "teacher_cache.jsonl"
    if not cache_path.is_file():
        raise SystemExit(f"RESCORE_NO_CACHE missing {cache_path}")
    cache = load_cache(cache_path)
    audit = resolve_replacement_audit(args)
    smoke_ids = out / "smoke_ids.json"
    # Explicit --ids-json wins so a leftover smoke_ids.json cannot shrink a 1200 rescore.
    if args.ids_json.is_file():
        frozen = json.loads(args.ids_json.read_text(encoding="utf-8"))
    elif smoke_ids.is_file():
        frozen = json.loads(smoke_ids.read_text(encoding="utf-8"))
    else:
        frozen = list(cache.keys())
    assert_replacement_ids(frozen, audit)
    missing = [sid for sid in frozen if sid not in cache]
    print(
        f"RESCORE_ONLY CACHED={len(cache)} FROZEN={len(frozen)} TODO=0 API_CALLS=0",
        flush=True,
    )
    if missing:
        raise SystemExit(f"RESCORE_MISSING {len(missing)} cache rows; refuse API fallback")
    pool = {r["sample_id"]: r for r in load_jsonl(str(args.pool))}
    band_map = overlay_replacement_band(load_band_map(args.manifest), audit)
    rescored: dict[str, dict] = {}
    for sid in frozen:
        row = dict(cache[sid])
        sample = pool[sid]
        refs = resolve_evidence_refs(sample)
        gold = gold_answer_of(sample)
        raw = json.dumps({"reasoning": row.get("reasoning") or ""})
        val = validate_teacher_reasoning(
            raw,
            gold_answer=gold,
            question=sample["question"],
            refs=refs,
            min_words=20,
            max_words=180,
            min_accept_score=4,
        )
        reasoning = val.get("reasoning") or row.get("reasoning") or ""
        row.update(
            {
                "reasoning": reasoning,
                "think_wrapped": wrap_think(reasoning) if reasoning else None,
                "accepted": bool(val.get("accepted")),
                "hard_reject": bool(val.get("hard_reject")),
                "grounding_valid": bool(val.get("grounding_valid")),
                "answer_consistent": bool(val.get("answer_consistent")),
                "evidence1_used": bool(val.get("evidence1_used")),
                "evidence2_used": bool(val.get("evidence2_used")),
                "meta_clean": bool(val.get("meta_clean")),
                "n_words": val.get("n_words"),
                "n_sentences": n_sentences(reasoning),
                "quality_score": val.get("quality_score"),
                "errors": val.get("errors"),
                "soft_warnings": val.get("soft_warnings") or [],
                "reasoning_band": row.get("reasoning_band") or band_map.get(sid),
                "rescored": True,
            }
        )
        rescored[sid] = row
    rescored_path = out / "teacher_cache.rescored.jsonl"
    with rescored_path.open("w", encoding="utf-8") as handle:
        for sid in frozen:
            handle.write(json.dumps(rescored[sid], ensure_ascii=False) + "\n")
    ordered = [rescored[sid] for sid in frozen]
    gate = teacher_gate(ordered, frozen, audit)
    gate["rescore_only"] = True
    gate["api_calls"] = 0
    (out / "teacher_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    print(
        "RESCORE_SUMMARY "
        f"HARD_FAIL={gate.get('hard_fail')} "
        f"EMPTY={gate.get('empty_rationale')} "
        f"GROUND={gate.get('grounded_rate')} "
        f"ANSWER={gate.get('answer_derivation_rate')} "
        "API_CALLS=0",
        flush=True,
    )
    print(gate["gate"])
    if "FAIL" in gate["gate"]:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.rescore_only:
        run_rescore_only(args)
        return
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.ids_json.read_text(encoding="utf-8"))
    audit = resolve_replacement_audit(args)
    assert_replacement_ids(frozen, audit)
    band_map = overlay_replacement_band(load_band_map(args.manifest), audit)
    if args.smoke_per_band and args.smoke_per_band > 0:
        frozen = pick_stratified(frozen, band_map, args.smoke_per_band, args.seed)
        (out / "smoke_ids.json").write_text(json.dumps(frozen, indent=2) + "\n")
    elif args.max_samples and args.max_samples > 0:
        frozen = frozen[: args.max_samples]
    pool = {r["sample_id"]: r for r in load_jsonl(str(args.pool))}
    cache_path = out / "teacher_cache.jsonl"
    cache = load_cache(cache_path)
    retry_ids = load_retry_ids(args.retry_ids)
    todo, kept = plan_todo(frozen, cache, retry_ids)
    print(
        f"FROZEN={len(frozen)} CACHED={len(kept)} RETRY={len(retry_ids)} TODO={len(todo)}",
        flush=True,
    )
    if args.dry_run:
        print("DRY_RUN API_CALLS=0", flush=True)
        return
    rescored_path = out / "teacher_cache.rescored.jsonl"
    if not todo and rescored_path.is_file():
        cache = load_cache(rescored_path)
        print(f"USING_RESCORED {rescored_path} N={len(cache)} API_CALLS=0", flush=True)
    if todo:
        gen = load_gen()
        args.api_key = (args.api_key or "").strip()
        probe_teacher(args.base_url, args.api_key, args.model)
    else:
        gen = None
        print("SKIP_TEACHER_PROBE TODO=0 API_CALLS=0", flush=True)
    fail_log = out / "retry_failures.jsonl"

    def log_attempt_failure(rec: dict) -> None:
        append_jsonl(fail_log, rec)
        print(
            "FAIL_TELEMETRY "
            f"sid={rec.get('sample_id')} attempt={rec.get('attempt')} "
            f"error={rec.get('error_type')} finish={rec.get('finish_reason')} "
            f"rtok={rec.get('reasoning_tokens')} ctok={rec.get('completion_tokens')} "
            f"content_len={rec.get('content_len')}",
            flush=True,
        )

    def work(sid: str) -> dict:
        sample = pool[sid]
        refs = resolve_evidence_refs(sample)
        gold = gold_answer_of(sample)
        user = format_teacher_user_prompt(sample, refs, gold)
        last_row: dict | None = None
        last_err = "unknown"
        for attempt in range(1, args.retries + 1):
            t0 = time.time()
            try:
                api = gen.chat_complete(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_V4},
                        {"role": "user", "content": user},
                    ],
                    temperature=1.0,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    thinking="enabled",
                    response_format_kind="json_object",
                    retries=2,
                    retry_backoff=3.0,
                    quiet=args.concurrency > 1,
                    allow_thinking_degrade=False,
                    reasoning_effort=args.reasoning_effort,
                )
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
                log_attempt_failure(
                    {
                        "sample_id": sid,
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "exception_message": str(exc)[:400],
                        "http_status": getattr(exc, "response", None)
                        and getattr(exc.response, "status_code", None),
                        "finish_reason": None,
                        "content_len": 0,
                        "reasoning_content_len": 0,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "reasoning_tokens": None,
                        "latency_s": round(time.time() - t0, 3),
                        "max_tokens": args.max_tokens,
                    }
                )
                time.sleep(3.0 * attempt)
                continue
            raw = api.get("content") or ""
            val = validate_teacher_reasoning(
                raw,
                gold_answer=gold,
                question=sample["question"],
                refs=refs,
                min_words=20,
                max_words=180,
                min_accept_score=4,
            )
            reasoning = val.get("reasoning") or ""
            usage = api.get("usage") or {}
            row = {
                "sample_id": sid,
                "model": api.get("model") or args.model,
                "prompt_version": PROMPT_VERSION_V4,
                "reasoning_band": band_map.get(sid),
                "reasoning": reasoning,
                "think_wrapped": wrap_think(reasoning) if reasoning else None,
                "accepted": bool(val.get("accepted")),
                "hard_reject": bool(val.get("hard_reject")),
                "grounding_valid": bool(val.get("grounding_valid")),
                "answer_consistent": bool(val.get("answer_consistent")),
                "evidence1_used": bool(val.get("evidence1_used")),
                "evidence2_used": bool(val.get("evidence2_used")),
                "meta_clean": bool(val.get("meta_clean")),
                "n_words": val.get("n_words"),
                "n_sentences": n_sentences(reasoning),
                "quality_score": val.get("quality_score"),
                "errors": val.get("errors"),
                "soft_warnings": val.get("soft_warnings") or [],
                "request_thinking": api.get("request_thinking"),
                "reasoning_effort": args.reasoning_effort or None,
                "thinking_payload_kept": bool(api.get("thinking_payload_kept", True)),
                "reasoning_content_present": bool(api.get("reasoning_content_present")),
                "reasoning_content_saved": False,
                "finish_reason": api.get("finish_reason"),
                "usage": usage,
                "latency_s": round(time.time() - t0, 3),
                "retry_count": attempt,
            }
            last_row = row
            if row["hard_reject"]:
                log_attempt_failure(
                    {
                        "sample_id": sid,
                        "attempt": attempt,
                        "error_type": "empty_content"
                        if not reasoning.strip()
                        else "hard_reject",
                        "exception_message": ",".join(row.get("errors") or []),
                        "http_status": api.get("http_status"),
                        "finish_reason": api.get("finish_reason"),
                        "content_len": int(api.get("content_len") or len(raw)),
                        "reasoning_content_len": int(api.get("reasoning_content_len") or 0),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": usage.get("reasoning_tokens"),
                        "latency_s": row["latency_s"],
                        "max_tokens": args.max_tokens,
                    }
                )
            if row["hard_reject"] and attempt < args.retries:
                last_err = "hard_reject:" + ",".join(row.get("errors") or [])
                time.sleep(1.0)
                continue
            return row
        if last_row is not None:
            last_row["hard_reject"] = True
            last_row["accepted"] = False
            last_row["errors"] = list(last_row.get("errors") or []) + [f"retry_exhausted:{last_err}"]
            return last_row
        return {
            "sample_id": sid,
            "model": args.model,
            "prompt_version": PROMPT_VERSION_V4,
            "reasoning_band": band_map.get(sid),
            "reasoning": "",
            "think_wrapped": None,
            "accepted": False,
            "hard_reject": True,
            "grounding_valid": False,
            "answer_consistent": False,
            "evidence1_used": False,
            "evidence2_used": False,
            "meta_clean": True,
            "n_words": 0,
            "n_sentences": 0,
            "quality_score": 0,
            "errors": [f"api_failed:{last_err}"],
            "soft_warnings": [],
            "request_thinking": "enabled",
            "reasoning_effort": args.reasoning_effort or None,
            "thinking_payload_kept": True,
            "reasoning_content_present": False,
            "reasoning_content_saved": False,
            "usage": {},
            "latency_s": 0.0,
            "retry_count": args.retries,
        }

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool_ex:
            futs = {pool_ex.submit(work, sid): sid for sid in todo}
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                append_jsonl(cache_path, row)
                cache[row["sample_id"]] = row
                done += 1
                print(
                    f"[{done}/{len(todo)}] {row['sample_id']} "
                    f"band={row.get('reasoning_band')} accepted={row['accepted']} "
                    f"hard={row.get('hard_reject')} rtok={(row.get('usage') or {}).get('reasoning_tokens')}",
                    flush=True,
                )
        if retry_ids:
            rewrite_cache(cache_path, cache, frozen)

    ordered = [cache[sid] for sid in frozen if sid in cache]
    gate = teacher_gate(ordered, frozen, audit)
    (out / "teacher_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    print(gate["gate"])

    if args.apply:
        if len(frozen) != 1200:
            raise SystemExit("refuse --apply unless filling the full frozen 1200")
        canon = load_jsonl(str(args.canonical))
        share = load_jsonl(str(args.sharegpt))
        new_c, new_s, n_fill = apply_fill(canon, share, cache)
        pending = sum(1 for r in new_s if PLACEHOLDER in json.dumps(r))
        with (out / "canonical_filled.jsonl").open("w", encoding="utf-8") as handle:
            for row in new_c:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with (out / "sharegpt_filled.jsonl").open("w", encoding="utf-8") as handle:
            for row in new_s:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        filled_ids = [r["sample_id"] for r in new_c]
        dropped = str(audit["dropped_id"]) if audit else ""
        added = str(audit["replacement_id"]) if audit else ""
        apply_audit = {
            "filled": n_fill,
            "pending_left": pending,
            "ids_unchanged": filled_ids == [r["sample_id"] for r in canon],
            "n_canon": len(new_c),
            "unique": len(set(filled_ids)),
            "dropped_present": int(bool(dropped) and dropped in set(filled_ids)),
            "replacement_present": int(bool(added) and added in set(filled_ids)),
            "missing_teacher": pending,
            "api_calls": 0 if not todo else len(todo),
        }
        (out / "apply_audit.json").write_text(json.dumps(apply_audit, indent=2) + "\n")
        print(json.dumps(apply_audit, indent=2))
        if pending != 0 or n_fill != 1200:
            raise SystemExit("apply did not clear all 1200 pending slots")
        if audit and (apply_audit["dropped_present"] or not apply_audit["replacement_present"]):
            raise SystemExit("apply replacement set mismatch")

    if "FAIL" in gate["gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
