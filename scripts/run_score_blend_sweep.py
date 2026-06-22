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

from activation_rag.benchmarks import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from activation_rag.supervised_reranking import load_jsonl, write_json
from scripts.audit_reranker_comparison import load_score_jsonl, write_score_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dense-score interpolation with a reranker score JSONL.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--scores-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--best-scores-out")
    parser.add_argument("--method-name", default="reranker")
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    summary = run_blend_sweep(
        groups_path=Path(args.groups),
        scores_path=Path(args.scores_jsonl),
        out_path=Path(args.out),
        best_scores_out=Path(args.best_scores_out) if args.best_scores_out else None,
        method_name=args.method_name,
        alpha_grid=tuple(float(item) for item in args.alpha_grid.split(",") if item.strip()),
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_blend_sweep(
    *,
    groups_path: Path,
    scores_path: Path,
    out_path: Path,
    best_scores_out: Path | None,
    method_name: str,
    alpha_grid: tuple[float, ...],
    top_k: int,
) -> dict[str, Any]:
    groups = load_jsonl(groups_path)
    scores = load_score_jsonl(scores_path)
    rows = []
    best_row: dict[str, Any] | None = None
    best_score_map: dict[tuple[str, str], float] | None = None
    metric_key = f"ndcg@{min(10, top_k)}"
    for alpha in alpha_grid:
        blended = _blend_scores(groups, scores, alpha=alpha)
        metrics = _evaluate(groups, blended, top_k=top_k)
        row = {"alpha": float(alpha), "metrics": metrics}
        rows.append(row)
        if best_row is None or metrics["model"][metric_key] > best_row["metrics"]["model"][metric_key]:
            best_row = row
            best_score_map = blended
    if best_scores_out and best_score_map is not None:
        write_score_jsonl(best_scores_out, best_score_map)
    summary = {
        "schema_version": "activation_rag.score_blend_sweep.v1",
        "groups": str(groups_path),
        "scores_jsonl": str(scores_path),
        "method_name": method_name,
        "query_count": len(groups),
        "top_k": top_k,
        "sweep": rows,
        "best": best_row,
        "best_scores_out": str(best_scores_out) if best_scores_out else None,
    }
    write_json(out_path, summary)
    return summary


def _blend_scores(groups: list[dict[str, Any]], scores: dict[tuple[str, str], float], *, alpha: float) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for group in groups:
        query_id = str(group["query_id"])
        candidates = list(group.get("candidates", []))
        dense = np.array([float(candidate.get("dense_score", (candidate.get("features") or {}).get("dense_score", 0.0))) for candidate in candidates], dtype=np.float64)
        rerank = np.array([float(scores.get((query_id, str(candidate["chunk_id"])), float("-inf"))) for candidate in candidates], dtype=np.float64)
        finite = np.isfinite(rerank)
        if not bool(finite.all()):
            rerank = np.where(finite, rerank, float(np.min(rerank[finite])) if bool(finite.any()) else 0.0)
        final = ((1.0 - alpha) * _zscore(dense)) + (alpha * _zscore(rerank))
        for candidate, score in zip(candidates, final, strict=True):
            out[(query_id, str(candidate["chunk_id"]))] = float(score)
    return out


def _evaluate(groups: list[dict[str, Any]], scores: dict[tuple[str, str], float], *, top_k: int) -> dict[str, dict[str, float]]:
    dense_rankings: dict[str, list[str]] = {}
    model_rankings: dict[str, list[str]] = {}
    qrels: dict[str, dict[str, int]] = {}
    for group in groups:
        query_id = str(group["query_id"])
        candidates = list(group.get("candidates", []))
        qrels[query_id] = {
            str(candidate["doc_id"]): int(candidate.get("label", 0))
            for candidate in candidates
            if int(candidate.get("label", 0)) > 0
        }
        dense_ordered = sorted(candidates, key=lambda candidate: int(candidate.get("dense_rank", 10**9)))
        model_ordered = sorted(candidates, key=lambda candidate: scores.get((query_id, str(candidate["chunk_id"])), float("-inf")), reverse=True)
        dense_rankings[query_id] = [str(candidate["doc_id"]) for candidate in dense_ordered]
        model_rankings[query_id] = [str(candidate["doc_id"]) for candidate in model_ordered]
    return {"dense": _aggregate(dense_rankings, qrels, top_k), "model": _aggregate(model_rankings, qrels, top_k)}


def _aggregate(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], top_k: int) -> dict[str, float]:
    mrr = []
    recall = []
    ndcg = []
    for query_id, ranked_doc_ids in rankings.items():
        qrel = qrels.get(query_id, {})
        mrr.append(mean_reciprocal_rank(ranked_doc_ids, qrel, min(10, top_k)))
        recall.append(recall_at_k(ranked_doc_ids, qrel, top_k))
        ndcg.append(ndcg_at_k(ranked_doc_ids, qrel, min(10, top_k)))
    return {
        f"mrr@{min(10, top_k)}": float(sum(mrr) / len(mrr)) if mrr else 0.0,
        f"ndcg@{min(10, top_k)}": float(sum(ndcg) / len(ndcg)) if ndcg else 0.0,
        f"recall@{top_k}": float(sum(recall) / len(recall)) if recall else 0.0,
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


if __name__ == "__main__":
    main()
