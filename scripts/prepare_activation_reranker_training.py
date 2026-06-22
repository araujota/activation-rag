#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activation_rag.benchmarks import load_beir_dataset
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.schema import DocumentRecord, stable_hash


SCHEMA_VERSION = "activation_rag.activation_reranker_training_row.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare supervised activation reranker/projection JSONL rows.")
    parser.add_argument("--beir-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--target-mode",
        choices=("teacher_forced_answer_span", "pooled_answer_chunk", "contrastive_answer_chunk"),
        default="contrastive_answer_chunk",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--fallback-hard-negative-count", type=int, default=20)
    args = parser.parse_args()

    rows = prepare_training_rows(
        beir_dir=Path(args.beir_dir),
        dataset_name=args.dataset_name,
        split=args.split,
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        target_mode=args.target_mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        fallback_hard_negative_count=args.fallback_hard_negative_count,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "row_count": len(rows)}, indent=2, sort_keys=True))


def prepare_training_rows(
    *,
    beir_dir: Path,
    dataset_name: str,
    split: str,
    telemetry_cache_dir: Path,
    target_mode: str = "contrastive_answer_chunk",
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    fallback_hard_negative_count: int = 20,
    available_chunk_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    available = available_chunk_ids if available_chunk_ids is not None else _available_cache_chunk_ids(telemetry_cache_dir)
    dataset = load_beir_dataset(beir_dir, name=dataset_name, split=split)
    chunks_by_doc_id = _chunks_by_benchmark_doc_id(
        dataset_name=dataset_name,
        corpus=dataset.corpus,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    all_cached_doc_chunk_ids = sorted(
        chunk_id
        for chunk_ids in chunks_by_doc_id.values()
        for chunk_id in chunk_ids
        if chunk_id in available
    )
    rows: list[dict[str, Any]] = []
    for query_id, query_text in dataset.queries.items():
        query_chunk_id = stable_hash(f"query\n{query_text}", 32)
        if query_chunk_id not in available:
            continue
        positive_doc_ids = sorted(doc_id for doc_id, grade in dataset.qrels.get(query_id, {}).items() if grade > 0)
        positive_chunk_ids = sorted(
            chunk_id
            for doc_id in positive_doc_ids
            for chunk_id in chunks_by_doc_id.get(doc_id, [])
            if chunk_id in available
        )
        if not positive_chunk_ids:
            continue
        positive_set = set(positive_chunk_ids)
        hard_negative_chunk_ids = [
            chunk_id
            for chunk_id in all_cached_doc_chunk_ids
            if chunk_id not in positive_set
        ][:fallback_hard_negative_count]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_name": dataset.name,
                "split": dataset.split,
                "query_id": query_id,
                "query_text": query_text,
                "query_activation_chunk_id": query_chunk_id,
                "target_mode": target_mode,
                "qrel_positive_doc_ids": positive_doc_ids,
                "positive_chunk_ids": positive_chunk_ids,
                "hard_negative_chunk_ids": hard_negative_chunk_ids,
                "activation_feature_reference": "telemetry_cache_json_by_chunk_id",
                "telemetry_cache_dir": str(telemetry_cache_dir),
            }
        )
    return rows


def _available_cache_chunk_ids(cache_dir: Path) -> set[str]:
    chunk_ids: set[str] = set()
    for path in sorted(cache_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if _valid_telemetry_row(row):
            chunk_ids.add(str(row["chunk_id"]))
    return chunk_ids


def _valid_telemetry_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("telemetry_valid", True)
        and not row.get("invalid_reason")
        and row.get("sae_feature_values")
    )


def _chunks_by_benchmark_doc_id(
    *,
    dataset_name: str,
    corpus: dict[str, str],
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, list[str]]:
    documents = [
        DocumentRecord.from_text(
            source_uri=f"benchmark://{dataset_name}/{doc_id}",
            title=doc_id,
            text=text,
            metadata={"benchmark_doc_id": doc_id},
        )
        for doc_id, text in corpus.items()
    ]
    document_id_to_doc_id = {document.document_id: str(document.metadata["benchmark_doc_id"]) for document in documents}
    chunks = Chunker(ChunkerSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)).split(documents)
    chunks_by_doc_id: dict[str, list[str]] = {}
    for chunk in chunks:
        chunks_by_doc_id.setdefault(document_id_to_doc_id[chunk.document_id], []).append(chunk.chunk_id)
    return chunks_by_doc_id


if __name__ == "__main__":
    main()
