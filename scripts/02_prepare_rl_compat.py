#!/usr/bin/env python3
"""Create VeRL/datasets-2.x compatible Parquet files without changing rows.

The source Parquet files were written by datasets 4.x and contain Hugging Face
schema metadata with `_type: List`.  datasets 2.21 cannot deserialize that
metadata even though the Arrow columns themselves are valid.  The compatible
copies preserve the Arrow schema and values but omit only schema metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "rl"
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_digest(table: pa.Table) -> str:
    digest = hashlib.sha256()
    for row in table.to_pylist():
        payload = json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def convert(source: Path, target: Path) -> dict[str, object]:
    source_table = pq.read_table(source)
    source_row_digest = row_digest(source_table)
    clean_table = source_table.replace_schema_metadata(None)

    temp = target.with_suffix(target.suffix + ".tmp")
    temp.unlink(missing_ok=True)
    pq.write_table(clean_table, temp, compression="snappy")

    check_table = pq.read_table(temp)
    if check_table.schema.metadata:
        raise RuntimeError(f"metadata was not removed from {temp}")
    if check_table.schema != clean_table.schema:
        raise RuntimeError(f"Arrow schema changed for {source}")
    if check_table.num_rows != source_table.num_rows:
        raise RuntimeError(f"row count changed for {source}")
    if row_digest(check_table) != source_row_digest:
        raise RuntimeError(f"row values changed for {source}")

    os.replace(temp, target)

    # This is the exact loader used by VeRL and is the compatibility gate.
    loaded = datasets.load_dataset(
        "parquet", data_files=str(target), split="train"
    )
    if len(loaded) != source_table.num_rows:
        raise RuntimeError(f"datasets row count mismatch for {target}")

    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "compatible": str(target),
        "compatible_sha256": sha256(target),
        "rows": source_table.num_rows,
        "columns": source_table.column_names,
        "row_content_sha256": source_row_digest,
        "source_metadata_keys": sorted(
            key.decode("utf-8", errors="replace")
            for key in (source_table.schema.metadata or {})
        ),
        "compatible_metadata_keys": [],
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = [
        convert(DATA / "train.parquet", DATA / "train_verl.parquet"),
        convert(DATA / "val.parquet", DATA / "val_verl.parquet"),
    ]
    manifest = {
        "gate": "RL_PARQUET_COMPAT_PASS",
        "reason": "strip datasets-4.x Hugging Face schema metadata for datasets-2.21",
        "datasets_version": datasets.__version__,
        "pyarrow_version": pa.__version__,
        "content_invariant": "row_content_sha256 source == compatible",
        "files": records,
    }
    target = RESULTS / "rl_parquet_compat_manifest.json"
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"RL_PARQUET_COMPAT_PASS manifest={target}")


if __name__ == "__main__":
    main()
