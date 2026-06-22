#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activation_rag.supervised_reranking import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect per-query candidate movements between dense and reranker score arms.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--candidate-scores-jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--query-limit", type=int, default=25)
    parser.add_argument("--snippet-chars", type=int, default=360)
    parser.add_argument("--sort-by", choices=("abs_ndcg_delta", "harm", "improve", "query_id"), default="abs_ndcg_delta")
    args = parser.parse_args()
    summary = audit_candidate_movements(
        groups_path=Path(args.groups),
        candidate_scores_path=Path(args.candidate_scores_jsonl),
        out_path=Path(args.out),
        top_n=args.top_n,
        query_limit=args.query_limit,
        snippet_chars=args.snippet_chars,
        sort_by=args.sort_by,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def audit_candidate_movements(
    *,
    groups_path: Path,
    candidate_scores_path: Path,
    out_path: Path,
    top_n: int,
    query_limit: int,
    snippet_chars: int,
    sort_by: str,
) -> dict[str, Any]:
    groups = load_jsonl(groups_path)
    scores = _load_scores(candidate_scores_path)
    rows = []
    for group in groups:
        dense_ranked = sorted(group.get("candidates", []), key=lambda candidate: (-float(candidate.get("dense_score", 0.0)), int(candidate.get("dense_rank", 10**9))))
        model_ranked = sorted(
            group.get("candidates", []),
            key=lambda candidate: (-scores.get((str(group["query_id"]), str(candidate["chunk_id"])), float("-inf")), int(candidate.get("dense_rank", 10**9))),
        )
        dense_ndcg = _dcg(dense_ranked[:10]) / max(_ideal_dcg(group.get("candidates", []), 10), 1e-12)
        model_ndcg = _dcg(model_ranked[:10]) / max(_ideal_dcg(group.get("candidates", []), 10), 1e-12)
        rows.append(
            {
                "query_id": str(group["query_id"]),
                "query_text": str(group.get("query_text") or ""),
                "positive_doc_ids": list(group.get("positive_doc_ids") or []),
                "dense_ndcg@10": dense_ndcg,
                "model_ndcg@10": model_ndcg,
                "ndcg_delta": model_ndcg - dense_ndcg,
                "dense_top": [_candidate_view(candidate, scores, str(group["query_id"]), snippet_chars) for candidate in dense_ranked[:top_n]],
                "model_top": [_candidate_view(candidate, scores, str(group["query_id"]), snippet_chars) for candidate in model_ranked[:top_n]],
                "first_positive_dense_rank": _first_positive_rank(dense_ranked),
                "first_positive_model_rank": _first_positive_rank(model_ranked),
            }
        )
    ordered = _sort_rows(rows, sort_by)[:query_limit]
    summary = {
        "schema_version": "activation_rag.candidate_movement_audit.v1",
        "groups": str(groups_path),
        "candidate_scores_jsonl": str(candidate_scores_path),
        "query_count": len(groups),
        "top_n": top_n,
        "sort_by": sort_by,
        "mean_ndcg_delta": sum(row["ndcg_delta"] for row in rows) / len(rows) if rows else 0.0,
        "improved_count": sum(1 for row in rows if row["ndcg_delta"] > 1e-12),
        "harmed_count": sum(1 for row in rows if row["ndcg_delta"] < -1e-12),
        "unchanged_count": sum(1 for row in rows if abs(row["ndcg_delta"]) <= 1e-12),
        "examples": ordered,
    }
    write_json(out_path, summary)
    return summary


def _load_scores(path: Path) -> dict[tuple[str, str], float]:
    return {(str(row["query_id"]), str(row["chunk_id"])): float(row["score"]) for row in load_jsonl(path)}


def _candidate_view(candidate: dict[str, Any], scores: dict[tuple[str, str], float], query_id: str, snippet_chars: int) -> dict[str, Any]:
    text = " ".join(str(candidate.get("text") or "").split())
    return {
        "doc_id": str(candidate.get("doc_id") or ""),
        "chunk_id": str(candidate.get("chunk_id") or ""),
        "label": int(candidate.get("label", 0)),
        "dense_rank": int(candidate.get("dense_rank", 10**9)),
        "dense_score": float(candidate.get("dense_score", 0.0)),
        "model_score": scores.get((query_id, str(candidate.get("chunk_id")))),
        "snippet": text[:snippet_chars],
    }


def _dcg(candidates: list[dict[str, Any]]) -> float:
    import math

    return sum((2.0 ** float(candidate.get("label_score", candidate.get("label", 0))) - 1.0) / math.log2(rank + 1) for rank, candidate in enumerate(candidates, start=1))


def _ideal_dcg(candidates: list[dict[str, Any]], k: int) -> float:
    ideal = sorted(candidates, key=lambda candidate: -float(candidate.get("label_score", candidate.get("label", 0))))[:k]
    return _dcg(ideal)


def _first_positive_rank(candidates: list[dict[str, Any]]) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if int(candidate.get("label", 0)) > 0:
            return rank
    return None


def _sort_rows(rows: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if sort_by == "harm":
        return sorted(rows, key=lambda row: row["ndcg_delta"])
    if sort_by == "improve":
        return sorted(rows, key=lambda row: -row["ndcg_delta"])
    if sort_by == "query_id":
        return sorted(rows, key=lambda row: row["query_id"])
    return sorted(rows, key=lambda row: -abs(row["ndcg_delta"]))


if __name__ == "__main__":
    main()
