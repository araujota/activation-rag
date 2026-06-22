#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activation_rag.benchmarks import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from activation_rag.supervised_reranking import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a normal text cross-encoder reranker over prepared candidate groups.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scores-out", help="Optional JSONL cache of per candidate text-reranker scores.")
    parser.add_argument("--scores-jsonl", help="Use existing score JSONL instead of running a model.")
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    groups = load_jsonl(args.groups)
    if args.scores_jsonl:
        scores = load_score_jsonl(Path(args.scores_jsonl))
    else:
        scores = score_with_cross_encoder(
            groups,
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        if args.scores_out:
            write_score_jsonl(Path(args.scores_out), scores)
    metrics = evaluate_with_scores(groups, scores=scores, top_k=args.top_k)
    summary = {
        "schema_version": "activation_rag.text_reranker_baseline.v1",
        "groups": args.groups,
        "model": args.model if not args.scores_jsonl else "precomputed_scores",
        "device": args.device,
        "max_length": args.max_length,
        "query_count": len(groups),
        "top_k": args.top_k,
        "metrics": metrics,
    }
    write_json(args.out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def score_with_cross_encoder(
    groups: list[dict[str, Any]],
    *,
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[tuple[str, str], float]:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device, max_length=max_length)
    pairs: list[tuple[str, str]] = []
    keys: list[tuple[str, str]] = []
    for group in groups:
        query_id = str(group["query_id"])
        query_text = str(group["query_text"])
        for candidate in group.get("candidates", []):
            pairs.append((query_text, str(candidate.get("text", ""))))
            keys.append((query_id, str(candidate["chunk_id"])))
    raw_scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    return {key: float(score) for key, score in zip(keys, raw_scores, strict=True)}


def evaluate_with_scores(
    groups: list[dict[str, Any]],
    *,
    scores: dict[tuple[str, str], float],
    top_k: int,
) -> dict[str, dict[str, float]]:
    dense_rankings: dict[str, list[str]] = {}
    rerank_rankings: dict[str, list[str]] = {}
    qrels: dict[str, dict[str, int]] = {}
    for group in groups:
        query_id = str(group["query_id"])
        candidates = list(group.get("candidates", []))
        qrels[query_id] = {
            str(candidate["doc_id"]): int(candidate.get("label", 0))
            for candidate in candidates
            if int(candidate.get("label", 0)) > 0
        }
        dense_ordered = sorted(candidates, key=lambda item: int(item.get("dense_rank", 10**9)))
        rerank_ordered = sorted(
            candidates,
            key=lambda item: scores.get((query_id, str(item["chunk_id"])), float("-inf")),
            reverse=True,
        )
        dense_rankings[query_id] = [str(candidate["doc_id"]) for candidate in dense_ordered]
        rerank_rankings[query_id] = [str(candidate["doc_id"]) for candidate in rerank_ordered]
    return {
        "dense": _aggregate(dense_rankings, qrels, top_k),
        "text_reranker": _aggregate(rerank_rankings, qrels, top_k),
    }


def load_score_jsonl(path: Path) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = {}
    for row in load_jsonl(path):
        scores[(str(row["query_id"]), str(row["chunk_id"]))] = float(row["score"])
    return scores


def write_score_jsonl(path: Path, scores: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (query_id, chunk_id), score in sorted(scores.items()):
            handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")


def _aggregate(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], top_k: int) -> dict[str, float]:
    if not rankings:
        return {}
    mrr = []
    recall = []
    ndcg = []
    for query_id, ranked_doc_ids in rankings.items():
        qrel = qrels.get(query_id, {})
        mrr.append(mean_reciprocal_rank(ranked_doc_ids, qrel, min(10, top_k)))
        recall.append(recall_at_k(ranked_doc_ids, qrel, top_k))
        ndcg.append(ndcg_at_k(ranked_doc_ids, qrel, min(10, top_k)))
    return {
        f"mrr@{min(10, top_k)}": float(sum(mrr) / len(mrr)),
        f"recall@{top_k}": float(sum(recall) / len(recall)),
        f"ndcg@{min(10, top_k)}": float(sum(ndcg) / len(ndcg)),
    }


if __name__ == "__main__":
    main()
