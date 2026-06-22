#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.supervised_reranking import load_jsonl
from scripts.prepare_vertical_reranker_groups import _write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge query-disjoint reranker group JSONL files.")
    parser.add_argument("--input", action="append", required=True, help="Input group JSONL. Repeat for multiple sources.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset-name", help="Optional combined dataset_name to stamp while preserving source_dataset_name.")
    parser.add_argument("--prefix-query-ids", action="store_true", help="Prefix merged query IDs with source dataset names to avoid task-local collisions.")
    args = parser.parse_args()

    summary = merge_group_files(
        input_paths=[Path(path) for path in args.input],
        out_path=Path(args.out),
        dataset_name=str(args.dataset_name) if args.dataset_name else None,
        prefix_query_ids=bool(args.prefix_query_ids),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def merge_group_files(*, input_paths: list[Path], out_path: Path, dataset_name: str | None = None, prefix_query_ids: bool = False) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()
    source_counts: dict[str, int] = {}
    for input_path in input_paths:
        for row in load_jsonl(input_path):
            merged = dict(row)
            source_name = str(merged.get("dataset_name") or input_path.parent.name)
            query_id = str(merged["query_id"])
            merged_query_id = f"{source_name}:{query_id}" if prefix_query_ids else query_id
            if merged_query_id in seen_query_ids:
                raise ValueError(f"duplicate query_id across merged groups: {merged_query_id}")
            seen_query_ids.add(merged_query_id)
            if prefix_query_ids:
                merged["source_query_id"] = query_id
                merged["query_id"] = merged_query_id
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
            if dataset_name is not None:
                merged["source_dataset_name"] = source_name
                merged["dataset_name"] = dataset_name
            rows.append(merged)
    _write_jsonl(out_path, rows)
    return {
        "schema_version": "activation_rag.reranker_group_merge.v1",
        "input_paths": [str(path) for path in input_paths],
        "out": str(out_path),
        "dataset_name": dataset_name,
        "prefix_query_ids": prefix_query_ids,
        "group_count": len(rows),
        "source_counts": source_counts,
        "positive_in_pool_count": sum(1 for row in rows if row.get("positive_in_candidate_pool")),
    }


if __name__ == "__main__":
    main()
