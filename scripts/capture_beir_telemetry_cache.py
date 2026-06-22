#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from activation_rag.benchmarks import load_beir_dataset
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.schema import ChunkRecord, DocumentRecord, stable_hash
from activation_rag.telemetry import CommandPrefillTelemetryProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture BEIR document/query prefill telemetry into a durable cache in batches.")
    parser.add_argument("--beir-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--telemetry-command", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--documents", action="store_true", help="Capture document chunks.")
    parser.add_argument("--queries", action="store_true", help="Capture query pseudo-chunks.")
    parser.add_argument("--limit", type=int, help="Optional max chunks/queries to attempt, after cache filtering.")
    parser.add_argument("--provider-id", default="sidecar-command-prefill-zero-section-filtered")
    parser.add_argument("--model-id", default="DeepSeek-R1-Distill-Llama-8B-Q6_K")
    parser.add_argument("--site-id", default="selected_resid_pre")
    parser.add_argument("--layer-selection-policy", default="selected_runtime_summary_prompt_prefill_only")
    parser.add_argument("--prompt-template-id", default="rag_raw_chunk_prefill_v1_strict_zero_section_v2")
    parser.add_argument("--normalization-policy", default="raw_summary_values_v2_prompt_prefill_filtered")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    args = parser.parse_args()

    summary = run_capture(
        beir_dir=Path(args.beir_dir),
        dataset_name=args.dataset_name,
        split=args.split,
        telemetry_command=tuple(shlex.split(args.telemetry_command)),
        cache_dir=Path(args.cache_dir),
        batch_size=args.batch_size,
        include_documents=bool(args.documents or not args.queries),
        include_queries=bool(args.queries),
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
    beir_dir: Path,
    dataset_name: str,
    split: str,
    telemetry_command: tuple[str, ...],
    cache_dir: Path,
    batch_size: int,
    include_documents: bool,
    include_queries: bool,
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
    dataset = load_beir_dataset(beir_dir, name=dataset_name, split=split)
    chunks: list[ChunkRecord] = []
    if include_documents:
        chunks.extend(_document_chunks(dataset_name=dataset.name, corpus=dataset.corpus))
    if include_queries:
        chunks.extend(_query_chunks(dataset.queries))
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
    uncached = uncached_all
    if limit is not None:
        uncached = uncached[:limit]
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
                    "captured_count": captured,
                    "remaining_count": len(uncached) - captured,
                    "batch_count": batch_count,
                    "batch_size": len(batch),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "schema_version": "activation_rag.beir_telemetry_cache_capture.v1",
        "dataset_name": dataset.name,
        "split": dataset.split,
        "candidate_count": len(chunks),
        "captured_count": captured,
        "cache_hit_count": len(chunks) - len(uncached_all),
        "uncached_count": len(uncached_all),
        "attempted_uncached_count": len(uncached),
        "batch_count": batch_count,
        "batch_size": batch_size,
        "cache_dir": str(cache_dir),
        "included_documents": include_documents,
        "included_queries": include_queries,
    }


def _document_chunks(*, dataset_name: str, corpus: dict[str, str]) -> list[ChunkRecord]:
    documents = [
        DocumentRecord.from_text(
            source_uri=f"benchmark://{dataset_name}/{doc_id}",
            title=doc_id,
            text=text,
            metadata={"benchmark_doc_id": doc_id},
        )
        for doc_id, text in corpus.items()
    ]
    return Chunker(ChunkerSettings(chunk_size=512, chunk_overlap=0)).split(documents)


def _query_chunks(queries: dict[str, str]) -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id=stable_hash(f"query\n{query}", 32),
            document_id="query",
            ordinal=index,
            text=query,
            text_hash=stable_hash(query, 32),
            char_start=0,
            char_end=len(query),
            token_count_estimate=max(1, len(query.split())),
            chunker="query-as-chunk-v1",
            chunk_size=512,
            chunk_overlap=0,
        )
        for index, (query_id, query) in enumerate(queries.items())
    ]


if __name__ == "__main__":
    main()
