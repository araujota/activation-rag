#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from activation_rag.embedding import CommandEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider

from prepare_vertical_reranker_groups import (
    RetrievalComponents,
    _candidate_row,
    _cosine_scores,
    _group_row,
    _rows_to_qrels,
    _rows_to_text_map,
    _top_indices,
    _write_jsonl,
)


PRESETS: dict[str, dict[str, str]] = {
    "cosqa": {
        "queries_corpus_dataset": "CoIR-Retrieval/cosqa-queries-corpus",
        "qrels_dataset": "CoIR-Retrieval/cosqa-qrels",
        "dataset_name": "coir-cosqa",
        "train_split": "train",
        "dev_split": "valid",
        "test_split": "test",
    },
    "codetrans-dl": {
        "queries_corpus_dataset": "CoIR-Retrieval/codetrans-dl-queries-corpus",
        "qrels_dataset": "CoIR-Retrieval/codetrans-dl-qrels",
        "dataset_name": "coir-codetrans-dl",
        "train_split": "train",
        "dev_split": "valid",
        "test_split": "test",
    },
    "codesearchnet-python": {
        "queries_corpus_dataset": "CoIR-Retrieval/CodeSearchNet-python-queries-corpus",
        "qrels_dataset": "CoIR-Retrieval/CodeSearchNet-python-qrels",
        "dataset_name": "coir-codesearchnet-python",
        "train_split": "train",
        "dev_split": "valid",
        "test_split": "test",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CoIR retrieval candidate groups from paired queries/corpus and qrels datasets.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="cosqa")
    parser.add_argument("--queries-corpus-dataset")
    parser.add_argument("--qrels-dataset")
    parser.add_argument("--dataset-name")
    parser.add_argument("--train-split")
    parser.add_argument("--dev-split")
    parser.add_argument("--test-split")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--append-qrel-positives", action="store_true")
    parser.add_argument("--embedding-command", help="Command with {input_jsonl}/{output_jsonl} placeholders for dense candidate generation.")
    parser.add_argument("--embedding-timeout-seconds", type=int, default=7200)
    parser.add_argument("--hash-dimension", type=int)
    parser.add_argument("--dense-cache-out")
    parser.add_argument("--reuse-dense-cache")
    parser.add_argument("--train-limit", type=int, help="Optional deterministic cap for train groups, useful for diagnostics.")
    parser.add_argument("--dev-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--dev-out", required=True)
    parser.add_argument("--test-out", required=True)
    args = parser.parse_args()

    config = _config_from_args(args)
    if int(args.candidate_k) <= 0:
        raise SystemExit("--candidate-k must be positive")
    embedder = _build_embedder(args)
    bundle = load_coir_components(config)
    train_rows, dev_rows, test_rows = prepare_coir_groups(
        bundle,
        candidate_k=int(args.candidate_k),
        embedder=embedder,
        dense_cache_out=Path(args.dense_cache_out) if args.dense_cache_out else None,
        reuse_dense_cache=Path(args.reuse_dense_cache) if args.reuse_dense_cache else None,
        append_qrel_positives=bool(args.append_qrel_positives),
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
    )
    _write_jsonl(Path(args.train_out), train_rows)
    _write_jsonl(Path(args.dev_out), dev_rows)
    _write_jsonl(Path(args.test_out), test_rows)
    print(
        json.dumps(
            {
                "schema_version": "activation_rag.coir_group_preparation.v1",
                "preset": args.preset,
                "dataset_name": config["dataset_name"],
                "candidate_k": int(args.candidate_k),
                "train_count": len(train_rows),
                "dev_count": len(dev_rows),
                "test_count": len(test_rows),
                "train_positive_in_pool_count": sum(1 for group in train_rows if group["positive_in_candidate_pool"]),
                "dev_positive_in_pool_count": sum(1 for group in dev_rows if group["positive_in_candidate_pool"]),
                "test_positive_in_pool_count": sum(1 for group in test_rows if group["positive_in_candidate_pool"]),
                "train_out": args.train_out,
                "dev_out": args.dev_out,
                "test_out": args.test_out,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_coir_components(config: dict[str, str]) -> dict[str, Any]:
    from datasets import load_dataset

    queries = _rows_to_text_map(load_dataset(config["queries_corpus_dataset"], split="queries"))
    corpus = _rows_to_text_map(load_dataset(config["queries_corpus_dataset"], split="corpus"))
    train_qrels = _rows_to_qrels(load_dataset(config["qrels_dataset"], split=config["train_split"]))
    dev_qrels = _rows_to_qrels(load_dataset(config["qrels_dataset"], split=config["dev_split"]))
    test_qrels = _rows_to_qrels(load_dataset(config["qrels_dataset"], split=config["test_split"]))
    return {
        "dataset_name": config["dataset_name"],
        "queries": queries,
        "corpus": corpus,
        "qrels_by_split": {
            "train": train_qrels,
            "dev": dev_qrels,
            "test": test_qrels,
        },
    }


def prepare_coir_groups(
    bundle: dict[str, Any],
    *,
    candidate_k: int,
    embedder: EmbeddingProvider,
    dense_cache_out: Path | None = None,
    reuse_dense_cache: Path | None = None,
    append_qrel_positives: bool = False,
    train_limit: int | None = None,
    dev_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    qrels_by_split: dict[str, dict[str, dict[str, float]]] = bundle["qrels_by_split"]
    all_qrels: dict[str, dict[str, float]] = {}
    for split_qrels in qrels_by_split.values():
        for query_id, doc_scores in split_qrels.items():
            all_qrels.setdefault(query_id, {}).update(doc_scores)
    components = RetrievalComponents(
        dataset_name=str(bundle["dataset_name"]),
        queries={query_id: bundle["queries"][query_id] for query_id in sorted(all_qrels) if query_id in bundle["queries"]},
        corpus=dict(bundle["corpus"]),
        qrels=all_qrels,
    )
    doc_ids = list(components.corpus)
    query_ids = list(components.queries)
    doc_vectors, query_vectors = _embed_or_load_vectors(
        components=components,
        doc_ids=doc_ids,
        query_ids=query_ids,
        embedder=embedder,
        dense_cache_out=dense_cache_out,
        reuse_dense_cache=reuse_dense_cache,
    )
    query_index = {query_id: offset for offset, query_id in enumerate(query_ids)}
    doc_index = {doc_id: offset for offset, doc_id in enumerate(doc_ids)}
    train_rows = _prepare_split_groups(
        components,
        qrels_by_split["train"],
        split="train",
        candidate_k=candidate_k,
        doc_ids=doc_ids,
        doc_index=doc_index,
        query_index=query_index,
        doc_vectors=doc_vectors,
        query_vectors=query_vectors,
        append_qrel_positives=append_qrel_positives,
        limit=train_limit,
    )
    dev_rows = _prepare_split_groups(
        components,
        qrels_by_split["dev"],
        split="dev",
        candidate_k=candidate_k,
        doc_ids=doc_ids,
        doc_index=doc_index,
        query_index=query_index,
        doc_vectors=doc_vectors,
        query_vectors=query_vectors,
        append_qrel_positives=append_qrel_positives,
        limit=dev_limit,
    )
    test_rows = _prepare_split_groups(
        components,
        qrels_by_split["test"],
        split="test",
        candidate_k=candidate_k,
        doc_ids=doc_ids,
        doc_index=doc_index,
        query_index=query_index,
        doc_vectors=doc_vectors,
        query_vectors=query_vectors,
        append_qrel_positives=append_qrel_positives,
        limit=test_limit,
    )
    return train_rows, dev_rows, test_rows


def _prepare_split_groups(
    components: RetrievalComponents,
    split_qrels: dict[str, dict[str, float]],
    *,
    split: str,
    candidate_k: int,
    doc_ids: list[str],
    doc_index: dict[str, int],
    query_index: dict[str, int],
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    append_qrel_positives: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_id in sorted(split_qrels):
        if query_id not in components.queries or query_id not in query_index:
            continue
        dense_scores = _cosine_scores(query_vectors[query_index[query_id]], doc_vectors)
        top_indices = _top_indices(dense_scores, min(candidate_k, len(doc_ids)))
        positives = split_qrels.get(query_id, {})
        candidates = []
        for rank, dense_index in enumerate(top_indices, start=1):
            doc_id = doc_ids[int(dense_index)]
            candidates.append(
                _candidate_row(
                    components.dataset_name,
                    doc_id,
                    components.corpus[doc_id],
                    rank,
                    float(dense_scores[int(dense_index)]),
                    float(positives.get(doc_id, 0.0)),
                )
            )
        if append_qrel_positives:
            present = {candidate["doc_id"] for candidate in candidates}
            for doc_id, score in sorted(positives.items()):
                if score <= 0 or doc_id in present or doc_id not in doc_index:
                    continue
                candidates.append(
                    _candidate_row(
                        components.dataset_name,
                        doc_id,
                        components.corpus[doc_id],
                        len(candidates) + 1,
                        float(dense_scores[doc_index[doc_id]]),
                        float(score),
                    )
                )
        group = _group_row(components, query_id, candidate_k, candidates, dense_score_source="embedding_cosine")
        group["split"] = split
        group["split_policy"] = "coir_native_qrels_split"
        rows.append(group)
        if limit is not None and len(rows) >= int(limit):
            break
    return rows


def _embed_or_load_vectors(
    *,
    components: RetrievalComponents,
    doc_ids: list[str],
    query_ids: list[str],
    embedder: EmbeddingProvider,
    dense_cache_out: Path | None,
    reuse_dense_cache: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    from prepare_vertical_reranker_groups import _embed_or_load_vectors as embed_or_load

    return embed_or_load(
        components=components,
        doc_ids=doc_ids,
        query_ids=query_ids,
        embedder=embedder,
        dense_cache_out=dense_cache_out,
        reuse_dense_cache=reuse_dense_cache,
    )


def _build_embedder(args: argparse.Namespace) -> EmbeddingProvider:
    if args.hash_dimension:
        return HashEmbeddingProvider(dimension=int(args.hash_dimension))
    if args.embedding_command:
        return CommandEmbeddingProvider(
            shlex.split(str(args.embedding_command)),
            model_id="command:coir-dense-cache",
            timeout_seconds=int(args.embedding_timeout_seconds),
        )
    raise SystemExit("CoIR preparation requires --embedding-command or --hash-dimension")


def _config_from_args(args: argparse.Namespace) -> dict[str, str]:
    config = dict(PRESETS.get(str(args.preset), {}))
    for key in ("queries_corpus_dataset", "qrels_dataset", "dataset_name", "train_split", "dev_split", "test_split"):
        value = getattr(args, key, None)
        if value:
            config[key] = str(value)
    missing = [key for key in ("queries_corpus_dataset", "qrels_dataset", "dataset_name", "train_split", "dev_split", "test_split") if not config.get(key)]
    if missing:
        raise SystemExit(f"missing required CoIR settings: {', '.join(missing)}")
    return config


if __name__ == "__main__":
    main()
