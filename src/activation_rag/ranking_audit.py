from __future__ import annotations

import random
from typing import Any

from activation_rag.benchmarks import mean_reciprocal_rank, ndcg_at_k, recall_at_k

ScoreMap = dict[tuple[str, str], float]


def paired_randomization_test(
    baseline_values: list[float],
    candidate_values: list[float],
    *,
    iterations: int = 10000,
    seed: int = 13,
) -> dict[str, Any]:
    if len(baseline_values) != len(candidate_values):
        raise ValueError("baseline and candidate vectors must have the same length")
    differences = [candidate - baseline for baseline, candidate in zip(baseline_values, candidate_values, strict=True)]
    if not differences:
        return {"query_count": 0, "mean_delta": 0.0, "p_value": 1.0, "iterations": 0}
    observed = abs(sum(differences) / len(differences))
    if observed == 0.0:
        return {"query_count": len(differences), "mean_delta": 0.0, "p_value": 1.0, "iterations": 0}
    rng = random.Random(seed)
    more_extreme = 0
    total = max(1, iterations)
    for _ in range(total):
        randomized_delta = sum(diff if rng.random() < 0.5 else -diff for diff in differences) / len(differences)
        if abs(randomized_delta) >= observed - 1e-15:
            more_extreme += 1
    return {
        "query_count": len(differences),
        "mean_delta": sum(differences) / len(differences),
        "p_value": (more_extreme + 1) / (total + 1),
        "iterations": total,
    }


def compare_ranked_systems(
    groups: list[dict[str, Any]],
    *,
    baseline_name: str,
    candidate_name: str,
    baseline_scores: ScoreMap | None,
    candidate_scores: ScoreMap,
    top_k: int = 10,
    randomization_iterations: int = 10000,
    changed_query_limit: int = 10,
    seed: int = 13,
) -> dict[str, Any]:
    baseline_per_query: dict[str, dict[str, float]] = {}
    candidate_per_query: dict[str, dict[str, float]] = {}
    changed_rows: list[dict[str, Any]] = []
    for group in groups:
        query_id = str(group["query_id"])
        baseline_ordered = rank_candidates(group, scores=baseline_scores)
        candidate_ordered = rank_candidates(group, scores=candidate_scores)
        qrels = {
            str(candidate["doc_id"]): int(candidate.get("label", 0))
            for candidate in group.get("candidates", [])
            if int(candidate.get("label", 0)) > 0
        }
        baseline_metrics = _per_query_metrics(baseline_ordered, qrels, top_k)
        candidate_metrics = _per_query_metrics(candidate_ordered, qrels, top_k)
        baseline_per_query[query_id] = baseline_metrics
        candidate_per_query[query_id] = candidate_metrics
        changed_rows.append(
            {
                "query_id": query_id,
                "query_text": group.get("query_text", ""),
                "metric_deltas": {
                    metric: candidate_metrics[metric] - baseline_metrics[metric]
                    for metric in baseline_metrics
                },
                "baseline_top": summarize_top_candidates(baseline_ordered, limit=top_k),
                "candidate_top": summarize_top_candidates(candidate_ordered, limit=top_k),
            }
        )
    metric_names = sorted(next(iter(baseline_per_query.values())).keys()) if baseline_per_query else []
    return {
        "schema_version": "activation_rag.ranking_audit.v1",
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "query_count": len(baseline_per_query),
        "top_k": top_k,
        "baseline_metrics": _aggregate_per_query(baseline_per_query),
        "candidate_metrics": _aggregate_per_query(candidate_per_query),
        "metric_deltas": {
            metric: _aggregate_per_query(candidate_per_query)[metric] - _aggregate_per_query(baseline_per_query)[metric]
            for metric in metric_names
        },
        "paired_significance": {
            metric: paired_randomization_test(
                [baseline_per_query[query_id][metric] for query_id in baseline_per_query],
                [candidate_per_query[query_id][metric] for query_id in baseline_per_query],
                iterations=randomization_iterations,
                seed=seed,
            )
            for metric in metric_names
        },
        "changed_query_summary": {
            metric: _changed_query_summary(changed_rows, metric)
            for metric in metric_names
        },
        "changed_queries": {
            "helped": sorted(
                changed_rows,
                key=lambda row: (row["metric_deltas"].get(f"ndcg@{min(10, top_k)}", 0.0), row["query_id"]),
                reverse=True,
            )[:changed_query_limit],
            "harmed": sorted(
                changed_rows,
                key=lambda row: (row["metric_deltas"].get(f"ndcg@{min(10, top_k)}", 0.0), row["query_id"]),
            )[:changed_query_limit],
        },
    }


def rank_candidates(group: dict[str, Any], *, scores: ScoreMap | None) -> list[dict[str, Any]]:
    candidates = list(group.get("candidates", []))
    if scores is None:
        return sorted(candidates, key=lambda item: int(item.get("dense_rank", 10**9)))
    query_id = str(group["query_id"])
    return sorted(
        candidates,
        key=lambda item: scores.get((query_id, str(item["chunk_id"])), float("-inf")),
        reverse=True,
    )


def summarize_top_candidates(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for rank, candidate in enumerate(candidates[:limit], start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": str(candidate["chunk_id"]),
                "doc_id": str(candidate["doc_id"]),
                "label": int(candidate.get("label", 0)),
                "dense_rank": int(candidate.get("dense_rank", 10**9)),
                "text": str(candidate.get("text", ""))[:500],
            }
        )
    return rows


def _per_query_metrics(candidates: list[dict[str, Any]], qrels: dict[str, int], top_k: int) -> dict[str, float]:
    ranked_doc_ids = [str(candidate["doc_id"]) for candidate in candidates]
    metric_k = min(10, top_k)
    return {
        f"mrr@{metric_k}": mean_reciprocal_rank(ranked_doc_ids, qrels, metric_k),
        f"ndcg@{metric_k}": ndcg_at_k(ranked_doc_ids, qrels, metric_k),
        f"recall@{top_k}": recall_at_k(ranked_doc_ids, qrels, top_k),
    }


def _aggregate_per_query(per_query: dict[str, dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    metric_names = sorted(next(iter(per_query.values())).keys())
    return {
        metric: sum(row[metric] for row in per_query.values()) / len(per_query)
        for metric in metric_names
    }


def _changed_query_summary(changed_rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    deltas = [float(row["metric_deltas"].get(metric, 0.0)) for row in changed_rows]
    improved = sum(1 for delta in deltas if delta > 1e-12)
    harmed = sum(1 for delta in deltas if delta < -1e-12)
    unchanged = len(deltas) - improved - harmed
    return {
        "improved_query_count": improved,
        "harmed_query_count": harmed,
        "unchanged_query_count": unchanged,
        "net_changed_query_count": improved - harmed,
        "max_improvement": max(deltas) if deltas else 0.0,
        "max_harm": min(deltas) if deltas else 0.0,
    }
