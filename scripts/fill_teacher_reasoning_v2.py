#!/usr/bin/env python3
"""Fill frozen 1200 evidence_reasoning slots with Kimi 2.6. Do not change ids."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PARENT = Path("/data1/hcc/deepresearch")
sys.path.insert(0, str(PARENT))

from src.sft.prototype_builder import gold_answer_of, load_jsonl, resolve_evidence_refs
from src.sft.teacher_reasoning import (
    DEFAULT_TEACHER_MODEL,
    PROMPT_VERSION_V3,
    SYSTEM_PROMPT_V3,
    format_teacher_user_prompt,
    validate_teacher_reasoning,
    wrap_think,
)

REPO = Path(__file__).resolve().parents[1]
PLACEHOLDER = "__TEACHER_REASONING_PENDING__"
FROZEN_IDS = REPO / "results/17_build_8b_coldstart_v2/ids_evidence_reasoning.json"
POOL = Path("/data1/hcc/deepresearch/data/sft/source/hotpotqa_distractor_train_pool_n8000.jsonl")
_SENT_RE = re.compile(r"[.!?]+")
_XML_RE = re.compile(r"</?(?:search|observation|evidence|think|answer|internal)\b", re.I)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--ids-json", type=Path, default=FROZEN_IDS)
    p.add_argument("--pool", type=Path, default=POOL)
    p.add_argument("--canonical", type=Path, default=REPO / "results/17_build_8b_coldstart_v2/canonical.jsonl")
    p.add_argument("--sharegpt", type=Path, default=REPO / "results/17_build_8b_coldstart_v2/sharegpt.jsonl")
    p.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", "http://10.16.137.2:8000/v1"))
    p.add_argument("--api-key", default=os.environ.get("KIMI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.environ.get("KIMI_MODEL", DEFAULT_TEACHER_MODEL))
    p.add_argument("--concurrency", type=int, default=int(os.environ.get("KIMI_CONCURRENCY", "8")))
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--exist-ok", action="store_true")
    return p.parse_args()


def load_gen():
    path = PARENT / "scripts/generate_teacher_reasoning.py"
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


def teacher_gate(rows: list[dict], frozen: list[str]) -> dict:
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
    rate = lambda xs: round(len(xs) / max(len(rows), 1), 4)
    hard = {
        "n_input_1200_or_smoke": n == len(frozen),
        "n_success": len(rows) == n,
        "unique": len(got) == n and got == set(frozen),
        "pending_zero": pending == 0,
        "accepted_ge_95": rate(accepted) >= 0.95 if n >= 1200 else True,
        "grounded_ge_95": rate(grounded) >= 0.95 if n >= 1200 else True,
        "answer_ge_95": rate(derived) >= 0.95 if n >= 1200 else True,
        "coverage_ge_90": rate(covered) >= 0.90 if n >= 1200 else True,
        "xml_or_meta_leak_zero": len(xml) + len(leak) == 0,
        "ids_unchanged": got <= set(frozen),
    }
    # smoke: still report rates; full 1200 uses the ≥95/90 bars
    if n < 1200:
        hard["accepted_ge_95"] = len(rows) == n
        hard["grounded_ge_95"] = True
        hard["answer_ge_95"] = True
        hard["coverage_ge_90"] = True
    gate = "GATE_TEACHER1200_PASS" if n >= 1200 and all(hard.values()) else (
        "GATE_TEACHER1200_SMOKE_PASS" if n < 1200 and hard["n_success"] and hard["unique"]
        else "GATE_TEACHER1200_FAIL"
    )
    return {
        "gate": gate,
        "n_input": n,
        "n_success": len(rows),
        "unique": len(got),
        "pending": pending,
        "accepted_rate": rate(accepted),
        "grounded_rate": rate(grounded),
        "answer_derivation_rate": rate(derived),
        "evidence_coverage_rate": rate(covered),
        "xml_leak": len(xml),
        "meta_leak": len(leak),
        "len_2to6_rate": rate(sent_ok),
        "sample_id_delta": len(got - set(frozen)),
        "hard_gates": hard,
        "prompt_version": PROMPT_VERSION_V3,
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
        row["provenance"]["reasoning_source"] = "kimi2.6"
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
        row["metadata"]["reasoning_source"] = "kimi2.6"
        new_share.append(row)
    return new_canon, new_share, filled_n


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(args.ids_json.read_text(encoding="utf-8"))
    if args.max_samples and args.max_samples > 0:
        frozen = frozen[: args.max_samples]
    pool = {r["sample_id"]: r for r in load_jsonl(str(args.pool))}
    cache_path = out / "teacher_cache.jsonl"
    cache = load_cache(cache_path)
    gen = load_gen()
    models_url = args.base_url.rstrip("/") + "/models"
    try:
        probe = __import__("requests").get(models_url, timeout=10)
        probe.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"KIMI_UNREACHABLE {models_url}: {exc}\n"
            "Start Kimi-K2.6-CT-FP8KV on that host:port, then rerun smoke."
        ) from exc
    todo = [sid for sid in frozen if sid not in cache]
    if args.debug:
        print(f"FROZEN={len(frozen)} CACHED={len(cache)} TODO={len(todo)}", flush=True)

    def work(sid: str) -> dict:
        sample = pool[sid]
        refs = resolve_evidence_refs(sample)
        gold = gold_answer_of(sample)
        user = format_teacher_user_prompt(sample, refs, gold)
        api = gen.chat_complete(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V3},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
            max_tokens=8192,
            timeout=args.timeout,
            thinking="enabled",
            response_format_kind="json_schema",
            retries=args.retries,
            retry_backoff=3.0,
            quiet=args.concurrency > 1,
        )
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
        row = {
            "sample_id": sid,
            "model": api.get("model") or args.model,
            "prompt_version": PROMPT_VERSION_V3,
            "reasoning": reasoning,
            "think_wrapped": wrap_think(reasoning) if reasoning else None,
            "accepted": bool(val.get("accepted")),
            "grounding_valid": bool(val.get("grounding_valid")),
            "answer_consistent": bool(val.get("answer_consistent")),
            "evidence1_used": bool(val.get("evidence1_used")),
            "evidence2_used": bool(val.get("evidence2_used")),
            "meta_clean": bool(val.get("meta_clean")),
            "n_words": val.get("n_words"),
            "n_sentences": n_sentences(reasoning),
            "quality_score": val.get("quality_score"),
            "errors": val.get("errors"),
            "reasoning_content_saved": False,
            "usage": api.get("usage"),
        }
        return row

    if todo:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool_ex:
            futs = {pool_ex.submit(work, sid): sid for sid in todo}
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                append_jsonl(cache_path, row)
                cache[row["sample_id"]] = row
                done += 1
                print(f"[{done}/{len(todo)}] {row['sample_id']} accepted={row['accepted']}", flush=True)

    ordered = [cache[sid] for sid in frozen if sid in cache]
    gate = teacher_gate(ordered, frozen)
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
        apply_audit = {
            "filled": n_fill,
            "pending_left": pending,
            "ids_unchanged": [r["sample_id"] for r in new_c] == [r["sample_id"] for r in canon],
        }
        (out / "apply_audit.json").write_text(json.dumps(apply_audit, indent=2) + "\n")
        print(json.dumps(apply_audit, indent=2))
        if pending != 0 or n_fill != 1200:
            raise SystemExit("apply did not clear all 1200 pending slots")

    if gate["gate"] == "GATE_TEACHER1200_FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
