#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.schema import DocumentRecord, stable_hash


SCHEMA_VERSION = "activation_rag.qrel_positive_capture_group.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare train-only qrel-positive evidence groups for telemetry capture.")
    parser.add_argument("--beir-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--limit-queries", type=int)
    args = parser.parse_args()

    groups = prepare_qrel_positive_capture_groups(
        beir_dir=Path(args.beir_dir),
        dataset_name=args.dataset_name,
        split=args.split,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        limit_queries=args.limit_queries,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps(_summary(groups, out), indent=2, sort_keys=True))


def prepare_qrel_positive_capture_groups(
    *,
    beir_dir: Path,
    dataset_name: str,
    split: str,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    limit_queries: int | None = None,
) -> list[dict[str, Any]]:
    qrels = _load_qrels(beir_dir / "qrels" / f"{split}.tsv")
    if limit_queries is not None:
        qrels = dict(list(qrels.items())[:limit_queries])
    queries = _load_selected_queries(beir_dir / "queries.jsonl", set(qrels))
    needed_doc_ids = {doc_id for doc_ids in qrels.values() for doc_id in doc_ids}
    documents = _load_selected_documents(beir_dir / "corpus.jsonl", needed_doc_ids, dataset_name=dataset_name)
    chunker = Chunker(ChunkerSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    chunks_by_doc_id: dict[str, list[Any]] = {}
    for doc_id, document in documents.items():
        chunks_by_doc_id[doc_id] = chunker.split([document])

    groups: list[dict[str, Any]] = []
    for query_id, positive_doc_ids in qrels.items():
        query_text = queries.get(query_id)
        if query_text is None:
            continue
        candidates: list[dict[str, Any]] = []
        rank = 1
        for doc_id in sorted(positive_doc_ids):
            for chunk in chunks_by_doc_id.get(doc_id, []):
                candidates.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "doc_id": doc_id,
                        "text": chunk.text,
                        "dense_rank": rank,
                        "dense_score": 1.0 / rank,
                        "label": 1,
                        "negative_source": None,
                        "negative_trust": None,
                        "features": {
                            "qrel_positive_rank_reciprocal": 1.0 / rank,
                            "dense_score": 1.0 / rank,
                        },
                    }
                )
                rank += 1
        if candidates:
            groups.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset_name": dataset_name,
                    "split": split,
                    "query_id": query_id,
                    "query_text": query_text,
                    "query_activation_chunk_id": stable_hash(f"query\n{query_text}", 32),
                    "candidate_k": len(candidates),
                    "positive_doc_ids": sorted(positive_doc_ids),
                    "positive_in_candidate_pool": True,
                    "false_negative_policy": "qrel_positive_capture_only_no_negatives",
                    "candidates": candidates,
                }
            )
    return groups


def _load_qrels(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    qrels: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if int(row.get("score") or 0) <= 0:
                continue
            qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    return qrels


def _load_selected_queries(path: Path, query_ids: set[str]) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row["_id"])
            if query_id in query_ids:
                queries[query_id] = str(row.get("text") or "")
    return queries


def _load_selected_documents(path: Path, doc_ids: set[str], *, dataset_name: str) -> dict[str, DocumentRecord]:
    documents: dict[str, DocumentRecord] = {}
    remaining = set(doc_ids)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not remaining:
                break
            row = json.loads(line)
            doc_id = str(row["_id"])
            if doc_id not in remaining:
                continue
            title = str(row.get("title") or doc_id)
            text = "\n\n".join(part for part in [title if title != doc_id else "", str(row.get("text") or "")] if part).strip()
            documents[doc_id] = DocumentRecord.from_text(
                source_uri=f"benchmark://{dataset_name}/{doc_id}",
                title=title,
                text=text,
                metadata={"benchmark_doc_id": doc_id},
            )
            remaining.remove(doc_id)
    return documents


def _summary(groups: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    candidate_counts = [len(group["candidates"]) for group in groups]
    unique_docs = {candidate["doc_id"] for group in groups for candidate in group["candidates"]}
    unique_chunks = {candidate["chunk_id"] for group in groups for candidate in group["candidates"]}
    return {
        "schema_version": "activation_rag.qrel_positive_capture_groups.summary.v1",
        "out": str(out),
        "group_count": len(groups),
        "candidate_count": sum(candidate_counts),
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "unique_positive_doc_count": len(unique_docs),
        "unique_positive_chunk_count": len(unique_chunks),
    }


if __name__ == "__main__":
    main()
