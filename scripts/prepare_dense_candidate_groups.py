#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.benchmarks import load_beir_dataset
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.schema import DocumentRecord, stable_hash


SCHEMA_VERSION = "activation_rag.dense_candidate_group.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BEIR dense top-k candidate groups without requiring telemetry.")
    parser.add_argument("--beir-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dense-embedding-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    args = parser.parse_args()

    groups = prepare_dense_candidate_groups(
        beir_dir=Path(args.beir_dir),
        dataset_name=args.dataset_name,
        split=args.split,
        dense_embedding_cache=Path(args.dense_embedding_cache),
        candidate_k=args.candidate_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, sort_keys=True) + "\n")
    print(json.dumps(_summary(groups, args.candidate_k, out), indent=2, sort_keys=True))


def prepare_dense_candidate_groups(
    *,
    beir_dir: Path,
    dataset_name: str,
    split: str,
    dense_embedding_cache: Path,
    candidate_k: int,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
) -> list[dict[str, Any]]:
    dataset = load_beir_dataset(beir_dir, name=dataset_name, split=split)
    chunks, chunk_to_doc_id = _build_chunks(
        dataset_name=dataset_name,
        corpus=dataset.corpus,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    dense = np.load(dense_embedding_cache, allow_pickle=False)
    dense_chunk_ids = [str(value) for value in dense["chunk_ids"]]
    dense_query_ids = [str(value) for value in dense["query_ids"]]
    doc_vectors = np.array(dense["doc_vectors"], dtype=np.float64)
    query_vectors = np.array(dense["query_vectors"], dtype=np.float64)
    query_index = {query_id: index for index, query_id in enumerate(dense_query_ids)}
    groups: list[dict[str, Any]] = []
    for query_id, query_text in dataset.queries.items():
        if query_id not in query_index:
            continue
        positives = {doc_id for doc_id, grade in dataset.qrels.get(query_id, {}).items() if grade > 0}
        dense_scores = _cosine_scores(query_vectors[query_index[query_id]], doc_vectors)
        candidate_indices = _top_indices(dense_scores, min(candidate_k, len(dense_scores)))
        candidates: list[dict[str, Any]] = []
        for rank, dense_index in enumerate(candidate_indices, start=1):
            chunk_id = dense_chunk_ids[int(dense_index)]
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            doc_id = chunk_to_doc_id[chunk_id]
            label = 1 if doc_id in positives else 0
            dense_score = float(dense_scores[int(dense_index)])
            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": chunk.text,
                    "dense_rank": rank,
                    "dense_score": dense_score,
                    "label": label,
                    "negative_source": None if label else "dense_hard_negative",
                    "negative_trust": None if label else "unjudged_assumed_negative",
                    "features": {
                        "dense_score": dense_score,
                        "dense_rank_reciprocal": 1.0 / rank,
                    },
                }
            )
        if candidates:
            groups.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_name": dataset.name,
                    "split": dataset.split,
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_activation_chunk_id": stable_hash(f"query\n{query_text}", 32),
                    "candidate_k": candidate_k,
                    "positive_doc_ids": sorted(positives),
                    "positive_in_candidate_pool": any(candidate["label"] > 0 for candidate in candidates),
                    "false_negative_policy": "unjudged_dense_candidates_assumed_negative_except_qrel_positives",
                    "candidates": candidates,
                }
            )
    return groups


def _build_chunks(*, dataset_name: str, corpus: dict[str, str], chunk_size: int, chunk_overlap: int):
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
    return chunks, {chunk.chunk_id: document_id_to_doc_id[chunk.document_id] for chunk in chunks}


def _cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    denominator = np.maximum(query_norm * doc_norms, 1e-12)
    return docs @ query / denominator


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    unordered = np.argpartition(-scores, kth=k - 1)[:k]
    return unordered[np.argsort(-scores[unordered])]


def _summary(groups: list[dict[str, Any]], candidate_k: int, out: Path) -> dict[str, Any]:
    candidate_counts = [len(group["candidates"]) for group in groups]
    positive_hits = sum(1 for group in groups if group["positive_in_candidate_pool"])
    return {
        "schema_version": "activation_rag.dense_candidate_groups.summary.v1",
        "out": str(out),
        "group_count": len(groups),
        "candidate_k": candidate_k,
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "positive_in_candidate_pool_count": positive_hits,
    }


if __name__ == "__main__":
    main()
