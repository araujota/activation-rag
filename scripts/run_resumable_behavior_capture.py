#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Run behavior-pair telemetry capture in resumable batches.")
    parser.add_argument("--requests", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-aggregate-output", action="store_true", help="Do not append rows to the aggregate JSONL; cache files remain the durable output.")
    parser.add_argument("--delete-batch-rows-after-cache", action="store_true", help="Delete per-batch row JSONL after rows are materialized into cache.")
    parser.add_argument("capture_args", nargs=argparse.REMAINDER, help="Arguments passed after -- to capture_qwen_sae_prefill.py")
    args = parser.parse_args()

    capture_args = list(args.capture_args)
    if capture_args and capture_args[0] == "--":
        capture_args = capture_args[1:]
    summary = run_resumable_capture(
        requests_path=Path(args.requests),
        output_jsonl=Path(args.output_jsonl),
        cache_dir=Path(args.cache_dir),
        batch_dir=Path(args.batch_dir),
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        plan_only=args.plan_only,
        aggregate_output=not args.no_aggregate_output,
        delete_batch_rows_after_cache=args.delete_batch_rows_after_cache,
        capture_args=capture_args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_resumable_capture(
    *,
    requests_path: Path,
    output_jsonl: Path,
    cache_dir: Path,
    batch_dir: Path,
    batch_size: int,
    max_batches: int,
    plan_only: bool,
    capture_args: list[str] | None = None,
    aggregate_output: bool = True,
    delete_batch_rows_after_cache: bool = False,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    capture_args = list(capture_args or [])
    requests = load_jsonl(requests_path)
    completed = completed_chunk_ids(cache_dir)
    all_batches = [
        (batch_index, requests[start : start + batch_size])
        for batch_index, start in enumerate(range(0, len(requests), batch_size), start=1)
    ]
    missing_batches = [
        (batch_index, [row for row in batch if str(row["chunk_id"]) not in completed])
        for batch_index, batch in all_batches
    ]
    missing_batches = [(batch_index, batch) for batch_index, batch in missing_batches if batch]
    if max_batches > 0:
        missing_batches = missing_batches[:max_batches]
    missing_count = sum(len(batch) for _batch_index, batch in missing_batches)
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.resumable_behavior_capture.v1",
        "requests": str(requests_path),
        "output_jsonl": str(output_jsonl),
        "cache_dir": str(cache_dir),
        "batch_dir": str(batch_dir),
        "batch_size": batch_size,
        "total_request_count": len(requests),
        "completed_before_count": len(completed),
        "missing_before_count": missing_count,
        "planned_batch_count": len(missing_batches),
        "plan_only": plan_only,
        "aggregate_output": aggregate_output,
        "delete_batch_rows_after_cache": delete_batch_rows_after_cache,
        "captured_batch_count": 0,
        "captured_row_count": 0,
    }
    if plan_only:
        return summary

    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    if aggregate_output:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    capture_script = Path(__file__).with_name("capture_qwen_sae_prefill.py")
    for batch_index, batch in missing_batches:
        request_batch = batch_dir / f"batch-{batch_index:05d}.requests.jsonl"
        output_batch = batch_dir / f"batch-{batch_index:05d}.rows.jsonl"
        write_jsonl(request_batch, batch)
        if output_batch.exists() and output_batch.stat().st_size > 0:
            rows = load_jsonl(output_batch)
        else:
            command = [sys.executable, str(capture_script), str(request_batch), str(output_batch), *capture_args]
            completed_process = subprocess.run(command, text=True, check=False)
            if completed_process.returncode != 0:
                summary["failed_batch_index"] = batch_index
                summary["failed_batch_requests"] = str(request_batch)
                summary["failed_batch_output"] = str(output_batch)
                summary["returncode"] = completed_process.returncode
                return summary
            rows = load_jsonl(output_batch)
        if aggregate_output:
            append_jsonl(output_jsonl, rows)
        for row in rows:
            chunk_id = str(row["chunk_id"])
            (cache_dir / f"{chunk_id}.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if delete_batch_rows_after_cache:
            output_batch.unlink(missing_ok=True)
            request_batch.unlink(missing_ok=True)
        summary["captured_batch_count"] = int(summary["captured_batch_count"]) + 1
        summary["captured_row_count"] = int(summary["captured_row_count"]) + len(rows)
    summary["completed_after_count"] = len(completed_chunk_ids(cache_dir))
    return summary


def completed_chunk_ids(cache_dir: Path) -> set[str]:
    if not cache_dir.exists():
        return set()
    completed: set[str] = set()
    for path in cache_dir.glob("*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if row.get("telemetry_valid", True) and row.get("sae_feature_values"):
            completed.add(str(row.get("chunk_id") or path.stem))
    return completed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
