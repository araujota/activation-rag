#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.schema import ChunkRecord, stable_hash
from activation_rag.supervised_reranking import load_jsonl
from activation_rag.telemetry import CommandPrefillTelemetryProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture prefill telemetry for the query/candidate union in reranker groups.")
    parser.add_argument("--groups", action="append", required=True, help="Candidate group JSONL. Can be repeated.")
    parser.add_argument("--telemetry-command", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--provider-id", default="qwen-rmt-sae-prefill")
    parser.add_argument("--model-id", default="qwen3-4b-rmt-sae")
    parser.add_argument("--site-id", default="l07_resid_pre")
    parser.add_argument("--layer-selection-policy", default="qwen3_rmt_l07_resid_pre_core245")
    parser.add_argument("--prompt-template-id", default="raw_chunk_v1")
    parser.add_argument("--normalization-policy", default="qwen_sae_checkpoint_mean_rms_topk64")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    summary = run_capture(
        group_paths=[Path(path) for path in args.groups],
        telemetry_command=tuple(shlex.split(args.telemetry_command)),
        cache_dir=Path(args.cache_dir),
        batch_size=args.batch_size,
        limit=args.limit,
        provider_id=args.provider_id,
        model_id=args.model_id,
        site_id=args.site_id,
        layer_selection_policy=args.layer_selection_policy,
        prompt_template_id=args.prompt_template_id,
        normalization_policy=args.normalization_policy,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_capture(
    *,
    group_paths: list[Path],
    telemetry_command: tuple[str, ...],
    cache_dir: Path,
    batch_size: int,
    limit: int | None,
    provider_id: str,
    model_id: str,
    site_id: str,
    layer_selection_policy: str,
    prompt_template_id: str,
    normalization_policy: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chunks = _chunks_from_groups(group_paths)
    provider = CommandPrefillTelemetryProvider(
        command=telemetry_command,
        provider_id=provider_id,
        model_id=model_id,
        site_id=site_id,
        layer_selection_policy=layer_selection_policy,
        prompt_template_id=prompt_template_id,
        normalization_policy=normalization_policy,
        timeout_seconds=timeout_seconds,
        cache_dir=cache_dir,
    )
    uncached_all = [chunk for chunk in chunks if provider._read_cache_row(chunk) is None]
    uncached = uncached_all[:limit] if limit is not None else uncached_all
    captured = 0
    batch_count = 0
    for start in range(0, len(uncached), batch_size):
        batch = uncached[start : start + batch_size]
        provider.capture_prefill(batch)
        captured += len(batch)
        batch_count += 1
        print(
            json.dumps(
                {
                    "batch_count": batch_count,
                    "batch_size": len(batch),
                    "captured_count": captured,
                    "remaining_count": len(uncached) - captured,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "schema_version": "activation_rag.group_telemetry_cache_capture.v1",
        "group_paths": [str(path) for path in group_paths],
        "candidate_count": len(chunks),
        "cache_hit_count": len(chunks) - len(uncached_all),
        "uncached_count": len(uncached_all),
        "attempted_uncached_count": len(uncached),
        "captured_count": captured,
        "batch_count": batch_count,
        "batch_size": batch_size,
        "cache_dir": str(cache_dir),
    }


def _chunks_from_groups(group_paths: list[Path]) -> list[ChunkRecord]:
    by_id: dict[str, ChunkRecord] = {}
    for path in group_paths:
        for group in load_jsonl(path):
            query_text = str(group.get("query_text") or "")
            query_chunk_id = str(group.get("query_activation_chunk_id") or stable_hash(f"query\n{query_text}", 32))
            by_id.setdefault(
                query_chunk_id,
                ChunkRecord(
                    chunk_id=query_chunk_id,
                    document_id="query",
                    ordinal=0,
                    text=query_text,
                    text_hash=stable_hash(query_text, 32),
                    char_start=0,
                    char_end=len(query_text),
                    token_count_estimate=max(1, len(query_text.split())),
                    chunker="query-as-chunk-v1",
                    chunk_size=512,
                    chunk_overlap=0,
                ),
            )
            for candidate in group.get("candidates", []):
                chunk_id = str(candidate["chunk_id"])
                if chunk_id in by_id:
                    continue
                text = str(candidate.get("text") or "")
                by_id[chunk_id] = ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=str(candidate.get("doc_id") or "document"),
                    ordinal=int(candidate.get("dense_rank") or 0),
                    text=text,
                    text_hash=stable_hash(text, 32),
                    char_start=0,
                    char_end=len(text),
                    token_count_estimate=max(1, len(text.split())),
                    chunker="candidate-group-text-v1",
                    chunk_size=512,
                    chunk_overlap=0,
                )
    return list(by_id.values())


if __name__ == "__main__":
    main()
