#!/usr/bin/env python3
"""Freeze an untouched HotpotQA test and write a leakage-audited experiment contract.

This script deliberately does not print test IDs or examples. It scans all historical
data artifacts, excludes every observed sample ID, selects deterministically from the
original 8k pool, and writes the sealed payload with owner-only permissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


ID_RE = re.compile(r"hotpotqa_distractor_(?:train|validation|test)_[0-9a-f]+")
TEXT_SUFFIXES = {".json", ".jsonl", ".txt"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.chmod(tmp, mode)
    tmp.replace(path)


def ids_in_parquet(path: Path) -> set[str]:
    found: set[str] = set()
    for row in pq.read_table(path).to_pylist():
        found.update(ID_RE.findall(json.dumps(row, ensure_ascii=False)))
    return found


def ids_in_text(path: Path) -> set[str]:
    return set(ID_RE.findall(path.read_text(errors="ignore")))


def ids_in_path(path: Path) -> set[str]:
    if not path.exists():
        return set()
    if path.suffix == ".parquet":
        return ids_in_parquet(path)
    if path.suffix in TEXT_SUFFIXES:
        return ids_in_text(path)
    return set()


def scan_historical_ids(data_root: Path, excluded_files: set[Path]) -> tuple[set[str], list[str]]:
    ids: set[str] = set()
    scanned: list[str] = []
    for path in sorted(data_root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded_files:
            continue
        if path.suffix not in TEXT_SUFFIXES | {".parquet"}:
            continue
        ids.update(ids_in_path(path))
        scanned.append(str(path.relative_to(data_root.parent)))
    return ids, scanned


def pairwise_overlap(named: dict[str, set[str]]) -> dict[str, int]:
    names = sorted(named)
    return {
        f"{left}__{right}": len(named[left] & named[right])
        for i, left in enumerate(names)
        for right in names[i + 1 :]
    }


def git_value(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNAVAILABLE"


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if "sample_id" not in row:
                    raise ValueError(f"missing sample_id: {path}:{line_number}")
                yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    sealed_existing = project / f"data/sealed/hotpotqa_test{args.size}.jsonl"
    if sealed_existing.is_file():
        raise SystemExit(
            "REFUSE: sealed test already frozen at "
            f"{sealed_existing}. Do not rerun this script."
        )
    repo = project.parents[1]
    data_root = repo / "data"
    pool = data_root / "sft/source/hotpotqa_distractor_train_pool_n8000.jsonl"
    pool_ids_file = data_root / "sft/source/hotpotqa_distractor_train_pool_n8000_ids.txt"
    if not pool.is_file():
        raise FileNotFoundError(pool)

    excluded_from_scan = {pool.resolve(), pool_ids_file.resolve()}
    historical, scanned_files = scan_historical_ids(data_root, excluded_from_scan)

    active_paths = {
        "sft_train": project / "data/sft/eca_coldstart_v1_train.jsonl",
        "sft_dev": project / "data/sft/eca_coldstart_v1_dev.jsonl",
        "sft_smoke": project / "data/sft/eca_coldstart_v1_smoke.jsonl",
        "rl_train": project / "data/rl/train.parquet",
        "rl_val": project / "data/rl/val.parquet",
        "frozen_dev": project / "data/eval/hotpotqa_200.jsonl",
    }
    fallback_paths = {
        "sft_train": data_root / "sft/llamafactory/eca_coldstart_v1_train.jsonl",
        "sft_dev": data_root / "sft/llamafactory/eca_coldstart_v1_dev.jsonl",
        "sft_smoke": data_root / "sft/llamafactory/eca_coldstart_v1_smoke.jsonl",
        "rl_train": data_root / "rl/train_smoke_128/train.parquet",
        "rl_val": data_root / "rl/train_smoke_128/val.parquet",
        "frozen_dev": data_root / "eval/hotpotqa_200.jsonl",
    }
    resolved_active = {
        name: path if path.is_file() else fallback_paths[name] for name, path in active_paths.items()
    }
    for path in resolved_active.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    active_ids = {name: ids_in_path(path) for name, path in resolved_active.items()}
    active_overlap = pairwise_overlap(active_ids)
    # SFT smoke is intentionally a train subset, and the historical RL set reuses
    # some SFT questions. Record those curriculum overlaps, but gate evaluation
    # isolation: frozen dev must not occur in any training split.
    forbidden_active_pairs = {
        "frozen_dev__rl_train",
        "frozen_dev__rl_val",
        "frozen_dev__sft_dev",
        "frozen_dev__sft_smoke",
        "frozen_dev__sft_train",
        "rl_train__rl_val",
        "sft_dev__sft_train",
    }
    bad_active = {
        name: count
        for name, count in active_overlap.items()
        if name in forbidden_active_pairs and count
    }
    if bad_active:
        raise RuntimeError(f"forbidden active split overlap detected: {bad_active}")

    pool_rows = list(load_jsonl(pool))
    pool_ids = [row["sample_id"] for row in pool_rows]
    if len(pool_ids) != len(set(pool_ids)):
        raise RuntimeError("duplicate sample_id in source pool")
    eligible = [row for row in pool_rows if row["sample_id"] not in historical]
    eligible.sort(
        key=lambda row: hashlib.sha256(f"{args.seed}:{row['sample_id']}".encode()).hexdigest()
    )
    if len(eligible) < args.size:
        raise RuntimeError(f"only {len(eligible)} unseen samples remain; requested {args.size}")
    selected = eligible[: args.size]
    selected_ids = {row["sample_id"] for row in selected}
    if selected_ids & historical:
        raise AssertionError("sealed test overlaps historical IDs")

    sealed_dir = project / "data/sealed"
    sealed_dir.mkdir(parents=True, exist_ok=True)
    test_path = sealed_dir / f"hotpotqa_test{args.size}.jsonl"
    ids_path = sealed_dir / f"hotpotqa_test{args.size}_ids.txt"
    with test_path.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    ids_path.write_text("\n".join(sorted(selected_ids)) + "\n")
    os.chmod(test_path, 0o600)
    os.chmod(ids_path, 0o600)

    all_named = dict(active_ids)
    all_named["sealed_test"] = selected_ids
    overlap = pairwise_overlap(all_named)
    forbidden_pairs = forbidden_active_pairs | {
        name for name in overlap if "sealed_test" in name
    }
    bad = {
        name: count for name, count in overlap.items() if name in forbidden_pairs and count
    }
    if bad:
        raise RuntimeError(f"frozen split overlap detected: {bad}")

    manifest = {
        "gate": "SEALED_TEST_FREEZE_PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "sha256(seed:sample_id), first N after historical exclusion",
        "seed": args.seed,
        "count": args.size,
        "source_pool": str(pool),
        "source_pool_sha256": sha256_file(pool),
        "source_pool_count": len(pool_rows),
        "historical_unique_ids": len(historical),
        "historical_pool_overlap": len(set(pool_ids) & historical),
        "eligible_unseen_count": len(eligible),
        "historical_scan_file_count": len(scanned_files),
        "historical_scan_files_sha256": hashlib.sha256(
            "\n".join(scanned_files).encode()
        ).hexdigest(),
        "sealed_payload": str(test_path),
        "sealed_payload_sha256": sha256_file(test_path),
        "sealed_ids_sha256": sha256_file(ids_path),
        "pairwise_overlap_counts": overlap,
        "allowed_training_overlap_note": (
            "SFT smoke is a train subset; historical RL may reuse SFT questions. "
            "These are recorded but are not evaluation leakage."
        ),
        "forbidden_overlap_max": max(
            (overlap[name] for name in forbidden_pairs), default=0
        ),
        "test_read_policy": "Do not read examples or evaluate until unique best checkpoint is frozen.",
    }
    manifest_path = project / "results/sealed_test_manifest.json"
    write_json(manifest_path, manifest)

    tracked_inputs = {
        name: {"path": str(path), "sha256": sha256_file(path), "count": len(active_ids[name])}
        for name, path in resolved_active.items()
    }
    tracked_inputs["sealed_test"] = {
        "path": str(test_path),
        "sha256": sha256_file(test_path),
        "count": args.size,
    }
    contract = {
        "gate": "EXPERIMENT_CONTRACT_PASS",
        "project_definition": "Evidence-Aware DeepResearch Agent with Exact On-Policy GRPO",
        "git_commit": git_value(project, "rev-parse", "HEAD"),
        "git_dirty": bool(git_value(project, "status", "--porcelain")),
        "model": {
            "path": str(project / "model"),
            "config_sha256": sha256_file(project / "model/config.json"),
            "tokenizer_sha256": sha256_file(project / "model/tokenizer.json"),
        },
        "inputs": tracked_inputs,
        "pairwise_overlap_counts": overlap,
        "allowed_training_overlap_note": manifest["allowed_training_overlap_note"],
        "forbidden_overlap_max": manifest["forbidden_overlap_max"],
        "selection_rule": "Dev only: health gate, Answer F1, Evidence F1, EM, duplicate query, earlier step.",
        "test_policy": "One opening after the unique best model and all baseline protocols are frozen.",
    }
    write_json(project / "results/experiment_contract.json", contract)

    print(
        json.dumps(
            {
                "gate": "P1_DATA_CONTRACT_PASS",
                "source_pool": len(pool_rows),
                "historical_unique": len(historical),
                "eligible_unseen": len(eligible),
                "sealed_count": args.size,
                "forbidden_overlap_max": manifest["forbidden_overlap_max"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
