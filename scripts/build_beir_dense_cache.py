#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np

from activation_rag.benchmarks import load_beir_dataset
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.embedding import CommandEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider
from activation_rag.schema import DocumentRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qid-aligned dense embedding cache for BEIR candidate groups.")
    parser.add_argument("--beir-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--reuse-doc-cache", help="Existing .npz with matching chunk_ids/doc_vectors to reuse.")
    parser.add_argument("--embedding-command", help="Command with {input_jsonl}/{output_jsonl} placeholders.")
    parser.add_argument("--embedding-timeout-seconds", type=int, default=3600)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--hash-dimension", type=int, help="Use hash embeddings instead of command embeddings.")
    args = parser.parse_args()
    if args.hash_dimension:
        embedder: EmbeddingProvider = HashEmbeddingProvider(dimension=args.hash_dimension)
    elif args.embedding_command:
        embedder = CommandEmbeddingProvider(
            command=shlex.split(args.embedding_command),
            model_id="command:beir-dense-cache",
            timeout_seconds=args.embedding_timeout_seconds,
        )
    else:
        raise SystemExit("Pass --embedding-command or --hash-dimension")
    summary = build_dense_cache(
        beir_dir=Path(args.beir_dir),
        dataset_name=args.dataset_name,
        split=args.split,
        out=Path(args.out),
        embedder=embedder,
        reuse_doc_cache=Path(args.reuse_doc_cache) if args.reuse_doc_cache else None,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_dense_cache(
    *,
    beir_dir: Path,
    dataset_name: str,
    split: str,
    out: Path,
    embedder: EmbeddingProvider,
    reuse_doc_cache: Path | None = None,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
) -> dict:
    dataset = load_beir_dataset(beir_dir, name=dataset_name, split=split)
    documents = [
        DocumentRecord.from_text(
            source_uri=f"benchmark://{dataset_name}/{doc_id}",
            title=doc_id,
            text=text,
            metadata={"benchmark_doc_id": doc_id},
        )
        for doc_id, text in dataset.corpus.items()
    ]
    chunks = Chunker(ChunkerSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)).split(documents)
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    doc_vectors = None
    reused_doc_vectors = False
    if reuse_doc_cache and reuse_doc_cache.exists():
        cached = np.load(reuse_doc_cache, allow_pickle=False)
        if list(cached["chunk_ids"]) == chunk_ids:
            doc_vectors = np.array(cached["doc_vectors"], dtype=np.float64)
            reused_doc_vectors = True
    if doc_vectors is None:
        doc_vectors = np.array([record.vector for record in embedder.embed_chunks(chunks)], dtype=np.float64)
    query_items = list(dataset.queries.items())
    query_ids = [query_id for query_id, _ in query_items]
    query_vectors = np.array(embedder.embed_texts([query for _, query in query_items]), dtype=np.float64)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        chunk_ids=np.array(chunk_ids),
        query_ids=np.array(query_ids),
        doc_vectors=doc_vectors,
        query_vectors=query_vectors,
    )
    return {
        "out": str(out),
        "split": split,
        "chunk_count": len(chunk_ids),
        "query_count": len(query_ids),
        "reused_doc_vectors": reused_doc_vectors,
    }


if __name__ == "__main__":
    main()
