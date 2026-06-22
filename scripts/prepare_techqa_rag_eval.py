#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
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

from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.embedding import CommandEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider
from activation_rag.schema import DocumentRecord, stable_hash


SCHEMA_VERSION = "activation_rag.techqa_rag_eval_groups.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TechQA-RAG-Eval as chunk-level documentation retrieval groups.")
    parser.add_argument("--out-dir", default="data/benchmarks/techqa-rag-eval")
    parser.add_argument("--groups-out", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--embedding-command", help="Command with {input_jsonl}/{output_jsonl} placeholders.")
    parser.add_argument("--embedding-timeout-seconds", type=int, default=3600)
    parser.add_argument("--hash-dimension", type=int, help="Use hash embeddings instead of command embeddings.")
    parser.add_argument("--dense-cache-out")
    args = parser.parse_args()

    if args.hash_dimension:
        embedder: EmbeddingProvider = HashEmbeddingProvider(dimension=args.hash_dimension)
    elif args.embedding_command:
        embedder = CommandEmbeddingProvider(
            command=shlex.split(args.embedding_command),
            model_id="command:techqa-rag-eval",
            timeout_seconds=args.embedding_timeout_seconds,
        )
    else:
        raise SystemExit("Pass --embedding-command or --hash-dimension")

    summary = prepare_techqa_groups(
        out_dir=Path(args.out_dir),
        groups_out=Path(args.groups_out),
        split=args.split,
        candidate_k=args.candidate_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedder=embedder,
        dense_cache_out=Path(args.dense_cache_out) if args.dense_cache_out else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def prepare_techqa_groups(
    *,
    out_dir: Path,
    groups_out: Path,
    split: str,
    candidate_k: int,
    chunk_size: int,
    chunk_overlap: int,
    embedder: EmbeddingProvider,
    dense_cache_out: Path | None,
) -> dict[str, Any]:
    rows = _load_hf_rows()
    answerable_rows = [row for row in rows if not bool(row.get("is_impossible")) and row.get("contexts")]
    documents, doc_id_by_filename = _build_documents(answerable_rows)
    chunker = Chunker(ChunkerSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    chunks = chunker.split(documents)
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    chunks_by_source_doc_id: dict[str, list[Any]] = {}
    for chunk in chunks:
        source_doc_id = str(chunk_by_id[chunk.chunk_id].document_id)
        chunks_by_source_doc_id.setdefault(source_doc_id, []).append(chunk)

    chunk_vectors = np.array([record.vector for record in embedder.embed_chunks(chunks)], dtype=np.float64)
    query_ids = [str(row["id"]) for row in answerable_rows]
    query_texts = [str(row["question"]) for row in answerable_rows]
    query_vectors = np.array(embedder.embed_texts(query_texts), dtype=np.float64)

    groups: list[dict[str, Any]] = []
    positive_policy_counts: dict[str, int] = {}
    for row_index, row in enumerate(answerable_rows):
        context = row["contexts"][0]
        filename = str(context["filename"])
        source_doc_id = doc_id_by_filename[filename]
        source_chunks = chunks_by_source_doc_id.get(source_doc_id, [])
        positive_chunk_ids, policy = select_positive_chunk_ids(
            answer=str(row.get("answer") or ""),
            context_text=str(context.get("text") or ""),
            chunks=source_chunks,
        )
        positive_policy_counts[policy] = positive_policy_counts.get(policy, 0) + 1
        dense_scores = _cosine_scores(query_vectors[row_index], chunk_vectors)
        candidate_indices = _top_indices(dense_scores, min(candidate_k, len(chunks)))
        if not any(chunks[int(index)].chunk_id in positive_chunk_ids for index in candidate_indices):
            positive_indices = [idx for idx, chunk in enumerate(chunks) if chunk.chunk_id in positive_chunk_ids]
            candidate_indices = _merge_positive_indices(candidate_indices, positive_indices, dense_scores, candidate_k)
        candidates: list[dict[str, Any]] = []
        for rank, dense_index in enumerate(candidate_indices, start=1):
            chunk = chunks[int(dense_index)]
            label = 1 if chunk.chunk_id in positive_chunk_ids else 0
            dense_score = float(dense_scores[int(dense_index)])
            source_doc = next(document for document in documents if document.document_id == chunk.document_id)
            candidates.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.chunk_id,
                    "source_doc_id": source_doc.metadata["techqa_filename"],
                    "text": chunk.text,
                    "dense_rank": rank,
                    "dense_score": dense_score,
                    "label": label,
                    "negative_source": None if label else "global_techqa_dense_hard_negative",
                    "negative_trust": None if label else "unjudged_assumed_negative",
                    "features": {
                        "dense_score": dense_score,
                        "dense_rank_reciprocal": 1.0 / rank,
                    },
                }
            )
        groups.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_name": "techqa-rag-eval",
                "split": split,
                "query_id": str(row["id"]),
                "query_text": str(row["question"]),
                "query_activation_chunk_id": stable_hash(f"query\n{row['question']}", 32),
                "candidate_k": candidate_k,
                "positive_doc_ids": sorted(positive_chunk_ids),
                "positive_source_doc_id": filename,
                "positive_label_policy": policy,
                "positive_in_candidate_pool": any(candidate["label"] > 0 for candidate in candidates),
                "false_negative_policy": "non_gold_techqa_chunks_assumed_negative",
                "answer": str(row.get("answer") or ""),
                "candidates": candidates,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "corpus.chunks.jsonl", [_chunk_row(chunk, documents) for chunk in chunks])
    _write_jsonl(out_dir / "queries.answerable.jsonl", [{"_id": str(row["id"]), "text": str(row["question"])} for row in answerable_rows])
    _write_qrels(out_dir / "qrels" / f"{split}.tsv", groups)
    groups_out.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(groups_out, groups)
    if dense_cache_out:
        dense_cache_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dense_cache_out,
            chunk_ids=np.array([chunk.chunk_id for chunk in chunks]),
            query_ids=np.array(query_ids),
            doc_vectors=chunk_vectors,
            query_vectors=query_vectors,
        )
    manifest = {
        "schema_version": "activation_rag.techqa_rag_eval_manifest.v1",
        "dataset": "nvidia/TechQA-RAG-Eval",
        "split": split,
        "answerable_query_count": len(answerable_rows),
        "source_document_count": len(documents),
        "chunk_count": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "candidate_k": candidate_k,
        "positive_policy_counts": positive_policy_counts,
        "groups": str(groups_out),
        "dense_cache": str(dense_cache_out) if dense_cache_out else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **manifest,
        "positive_in_candidate_pool_count": sum(1 for group in groups if group["positive_in_candidate_pool"]),
        "groups_out": str(groups_out),
    }


def select_positive_chunk_ids(*, answer: str, context_text: str, chunks: list[Any]) -> tuple[set[str], str]:
    answer_norm = _normalize_text(answer)
    if not chunks:
        return set(), "no_chunks"
    if answer_norm and answer_norm != "-":
        context_norm = _normalize_text(context_text)
        start_norm = context_norm.find(answer_norm)
        if start_norm >= 0:
            span = _normalized_span_to_original_span(context_text, start_norm, start_norm + len(answer_norm))
            overlapping = {
                chunk.chunk_id
                for chunk in chunks
                if chunk.char_start < span[1] and chunk.char_end > span[0]
            }
            if overlapping:
                return overlapping, "answer_span_overlap"
        line_hits = _chunks_with_answer_line_hits(answer, chunks)
        if line_hits:
            return line_hits, "answer_line_substring"
    return {_best_lexical_overlap_chunk(answer, chunks).chunk_id}, "best_lexical_overlap_fallback"


def _load_hf_rows() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("nvidia/TechQA-RAG-Eval")["train"]
    return [dict(row) for row in dataset]


def _build_documents(rows: list[dict[str, Any]]) -> tuple[list[DocumentRecord], dict[str, str]]:
    filename_to_text: dict[str, str] = {}
    for row in rows:
        for context in row.get("contexts") or []:
            filename = str(context["filename"])
            text = str(context["text"])
            filename_to_text.setdefault(filename, text)
    documents: list[DocumentRecord] = []
    doc_id_by_filename: dict[str, str] = {}
    for filename, text in sorted(filename_to_text.items()):
        document = DocumentRecord.from_text(
            source_uri=f"techqa-rag-eval://{filename}",
            title=filename,
            text=text,
            metadata={"techqa_filename": filename},
        )
        documents.append(document)
        doc_id_by_filename[filename] = document.document_id
    return documents, doc_id_by_filename


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _normalized_span_to_original_span(text: str, norm_start: int, norm_end: int) -> tuple[int, int]:
    norm_index = 0
    original_start: int | None = None
    original_end = len(text)
    in_space = False
    for original_index, char in enumerate(text):
        if char.isspace():
            if in_space:
                continue
            normalized_char = " "
            in_space = True
        else:
            normalized_char = char.lower()
            in_space = False
        if norm_index == norm_start and original_start is None:
            original_start = original_index
        norm_index += len(normalized_char)
        if norm_index >= norm_end:
            original_end = original_index + 1
            break
    return (original_start if original_start is not None else 0, original_end)


def _chunks_with_answer_line_hits(answer: str, chunks: list[Any]) -> set[str]:
    hits: set[str] = set()
    snippets = [
        _normalize_text(line)
        for line in re.split(r"[\n\r]+", answer)
        if len(_normalize_text(line)) >= 30
    ]
    for chunk in chunks:
        chunk_norm = _normalize_text(chunk.text)
        if any(snippet in chunk_norm for snippet in snippets):
            hits.add(chunk.chunk_id)
    return hits


def _best_lexical_overlap_chunk(answer: str, chunks: list[Any]) -> Any:
    answer_terms = set(re.findall(r"[a-z0-9_./-]{3,}", answer.lower()))
    if not answer_terms:
        return chunks[0]
    best = chunks[0]
    best_score = -1.0
    for chunk in chunks:
        chunk_terms = set(re.findall(r"[a-z0-9_./-]{3,}", chunk.text.lower()))
        score = len(answer_terms & chunk_terms) / max(len(answer_terms), 1)
        if score > best_score:
            best = chunk
            best_score = score
    return best


def _cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    denominator = np.maximum(query_norm * doc_norms, 1e-12)
    return docs @ query / denominator


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    unordered = np.argpartition(-scores, kth=k - 1)[:k]
    return unordered[np.argsort(-scores[unordered])]


def _merge_positive_indices(candidate_indices: np.ndarray, positive_indices: list[int], scores: np.ndarray, candidate_k: int) -> np.ndarray:
    merged = list(dict.fromkeys([int(index) for index in candidate_indices] + positive_indices))
    merged = sorted(merged, key=lambda index: float(scores[index]), reverse=True)
    positives = set(positive_indices)
    if positives and not any(index in positives for index in merged[:candidate_k]):
        merged = merged[: max(candidate_k - 1, 0)] + [positive_indices[0]]
    return np.array(merged[:candidate_k], dtype=np.int64)


def _chunk_row(chunk: Any, documents: list[DocumentRecord]) -> dict[str, Any]:
    doc_by_id = {document.document_id: document for document in documents}
    document = doc_by_id[chunk.document_id]
    return {
        "_id": chunk.chunk_id,
        "source_doc_id": document.metadata["techqa_filename"],
        "text": chunk.text,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_qrels(path: Path, groups: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("query-id\tcorpus-id\tscore\n")
        for group in groups:
            for chunk_id in group["positive_doc_ids"]:
                handle.write(f"{group['query_id']}\t{chunk_id}\t1\n")


if __name__ == "__main__":
    main()
