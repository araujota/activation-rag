#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activation_rag.supervised_reranking import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize behavior pair telemetry JSONL into per-row cache JSON files.")
    parser.add_argument("--capture-jsonl", required=True)
    parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    summary = materialize_cache(capture_jsonl=Path(args.capture_jsonl), cache_dir=Path(args.cache_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))


def materialize_cache(*, capture_jsonl: Path, cache_dir: Path) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(capture_jsonl)
    written = 0
    valid = 0
    for row in rows:
        chunk_id = str(row["chunk_id"])
        if row.get("telemetry_valid", True) and row.get("sae_feature_values"):
            valid += 1
        (cache_dir / f"{chunk_id}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written += 1
    return {
        "schema_version": "activation_rag.behavior_telemetry_cache_materialization.v1",
        "capture_jsonl": str(capture_jsonl),
        "cache_dir": str(cache_dir),
        "row_count": len(rows),
        "written_count": written,
        "valid_count": valid,
    }


if __name__ == "__main__":
    main()
