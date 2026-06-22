#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from activation_rag.ranking_audit import ScoreMap, rank_candidates
from activation_rag.supervised_reranking import load_jsonl, write_json


DEFAULT_DATASETS = {
    "scifact": {
        "groups": "runs/supervised/scifact-test-reranker-groups.k100.jsonl",
        "actpred_scores": "runs/supervised/direct-blend-rerankers-20260615/scifact/test-scores.full-fallback.jsonl",
        "ettin_scores": "runs/supervised/scifact-ettin-reranker-150m-v1-baseline.scores.jsonl",
        "metrics": "runs/supervised/direct-blend-rerankers-20260615/scifact/metrics.json",
    },
    "legalbenchrag": {
        "groups": "runs/supervised/verticals/legalbenchrag/test-groups.k100.bge.appendpos.jsonl",
        "actpred_scores": "runs/supervised/verticals/crossfit/legalbenchrag-scaled/final/test-scores.jsonl",
        "ettin_scores": "runs/supervised/verticals/ettin-baselines/legalbenchrag/test-ettin-reranker-150m-v1.scores.jsonl",
        "metrics": "runs/supervised/verticals/crossfit/legalbenchrag-scaled/final/metrics.json",
    },
    "r2med-all": {
        "groups": "runs/supervised/verticals/r2med-all/test-groups.k100.appendpos.jsonl",
        "actpred_scores": "runs/supervised/verticals/crossfit/r2med-all-scaled/final/test-scores.jsonl",
        "metrics": "runs/supervised/verticals/crossfit/r2med-all-scaled/final/metrics.json",
    },
    "coir-cosqa": {
        "groups": "runs/supervised/verticals/coir-cosqa/test-groups.k100.bge.appendpos.jsonl",
        "actpred_scores": "runs/supervised/verticals/crossfit/coir-cosqa-scaled/final/test-scores.jsonl",
        "metrics": "runs/supervised/verticals/crossfit/coir-cosqa-scaled/final/metrics.json",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep diagnostics for activation-aware reranker behavior.")
    parser.add_argument("--out", default="runs/supervised/verticals/deep-activation-reranker-diagnostics-20260617.json")
    parser.add_argument("--example-limit", type=int, default=4)
    parser.add_argument("--snippet-chars", type=int, default=420)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "schema_version": "activation_rag.deep_activation_reranker_diagnostics.v1",
        "datasets": {},
        "cross_dataset_summary": {},
    }
    for dataset_name, config in DEFAULT_DATASETS.items():
        report["datasets"][dataset_name] = diagnose_dataset(
            dataset_name=dataset_name,
            config=config,
            example_limit=args.example_limit,
            snippet_chars=args.snippet_chars,
        )
    report["cross_dataset_summary"] = summarize_cross_dataset(report["datasets"])
    write_json(args.out, report)
    print(json.dumps(report["cross_dataset_summary"], indent=2, sort_keys=True))


def diagnose_dataset(
    *,
    dataset_name: str,
    config: dict[str, str],
    example_limit: int,
    snippet_chars: int,
) -> dict[str, Any]:
    groups_path = Path(config["groups"])
    if not groups_path.exists():
        return {"error": f"missing groups: {groups_path}"}
    groups = load_jsonl(groups_path)
    actpred_scores = load_scores(config.get("actpred_scores"))
    ettin_scores = load_scores(config.get("ettin_scores"))
    metrics = load_json(config.get("metrics"))
    systems: dict[str, ScoreMap | None] = {"dense": None}
    if actpred_scores:
        systems["actpred"] = actpred_scores
    if ettin_scores:
        systems["ettin"] = ettin_scores

    rows = [per_query_row(group, systems) for group in groups]
    score_separation = {
        name: score_separation_for(groups, scores)
        for name, scores in systems.items()
    }
    examples = {
        "actpred_improves_dense": examples_for(groups, baseline_scores=None, candidate_scores=actpred_scores, limit=example_limit, snippet_chars=snippet_chars, descending=True) if actpred_scores else [],
        "actpred_harms_dense": examples_for(groups, baseline_scores=None, candidate_scores=actpred_scores, limit=example_limit, snippet_chars=snippet_chars, descending=False) if actpred_scores else [],
    }
    if ettin_scores and actpred_scores:
        examples["ettin_beats_actpred"] = examples_for(
            groups,
            baseline_scores=actpred_scores,
            candidate_scores=ettin_scores,
            limit=example_limit,
            snippet_chars=snippet_chars,
            descending=True,
        )
        examples["actpred_beats_ettin"] = examples_for(
            groups,
            baseline_scores=ettin_scores,
            candidate_scores=actpred_scores,
            limit=example_limit,
            snippet_chars=snippet_chars,
            descending=True,
        )

    return {
        "dataset_name": dataset_name,
        "groups": str(groups_path),
        "query_count": len(groups),
        "candidate_stats": candidate_stats(groups),
        "training_generalization": training_generalization(metrics),
        "rank_movement": aggregate_rank_movement(rows),
        "score_separation": score_separation,
        "source_overlap": source_overlap_stats(groups),
        "lexical_shape": lexical_shape_stats(groups),
        "examples": examples,
    }


def per_query_row(group: dict[str, Any], systems: dict[str, ScoreMap | None]) -> dict[str, Any]:
    candidates = list(group.get("candidates", []))
    positives = {str(candidate["chunk_id"]) for candidate in candidates if is_positive(candidate)}
    row: dict[str, Any] = {"query_id": str(group["query_id"]), "positive_count": len(positives)}
    for name, scores in systems.items():
        ranked = rank_candidates(group, scores=scores)
        row[f"{name}_ndcg"] = ndcg(ranked, 10)
        row[f"{name}_first_positive_rank"] = first_positive_rank(ranked)
        row[f"{name}_positive_at_1"] = 1.0 if ranked and is_positive(ranked[0]) else 0.0
        row[f"{name}_positive_at_10"] = 1.0 if any(is_positive(candidate) for candidate in ranked[:10]) else 0.0
    return row


def candidate_stats(groups: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_counts = [len(group.get("candidates", [])) for group in groups]
    positive_counts = [sum(1 for candidate in group.get("candidates", []) if is_positive(candidate)) for group in groups]
    dense_first_positive = [first_positive_rank(rank_candidates(group, scores=None)) for group in groups]
    return {
        "candidate_count": describe(candidate_counts),
        "positive_count": describe(positive_counts),
        "queries_with_positive": sum(1 for count in positive_counts if count > 0),
        "dense_first_positive_rank": describe([rank for rank in dense_first_positive if rank is not None]),
        "dense_positive_at_1_rate": mean([1.0 if rank == 1 else 0.0 for rank in dense_first_positive]),
        "dense_positive_at_10_rate": mean([1.0 if rank is not None and rank <= 10 else 0.0 for rank in dense_first_positive]),
    }


def aggregate_rank_movement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for candidate_name in ("actpred", "ettin"):
        if not rows or f"{candidate_name}_ndcg" not in rows[0]:
            continue
        deltas = [row[f"{candidate_name}_ndcg"] - row["dense_ndcg"] for row in rows]
        rank_deltas = [
            (row["dense_first_positive_rank"] or 10**6) - (row[f"{candidate_name}_first_positive_rank"] or 10**6)
            for row in rows
        ]
        summary[f"{candidate_name}_vs_dense"] = {
            "mean_ndcg_delta": mean(deltas),
            "improved_queries": sum(1 for delta in deltas if delta > 1e-12),
            "harmed_queries": sum(1 for delta in deltas if delta < -1e-12),
            "unchanged_queries": sum(1 for delta in deltas if abs(delta) <= 1e-12),
            "mean_first_positive_rank_improvement": mean(rank_deltas),
            "positive_at_1_delta": mean([row[f"{candidate_name}_positive_at_1"] - row["dense_positive_at_1"] for row in rows]),
            "positive_at_10_delta": mean([row[f"{candidate_name}_positive_at_10"] - row["dense_positive_at_10"] for row in rows]),
        }
    return summary


def score_separation_for(groups: list[dict[str, Any]], scores: ScoreMap | None) -> dict[str, Any]:
    query_aucs = []
    pos_scores = []
    neg_scores = []
    pos_z = []
    neg_z = []
    coverage = 0
    total = 0
    for group in groups:
        values = []
        for candidate in group.get("candidates", []):
            total += 1
            score = score_for(group, candidate, scores)
            if score is None:
                continue
            coverage += 1
            values.append((score, is_positive(candidate)))
            if is_positive(candidate):
                pos_scores.append(score)
            else:
                neg_scores.append(score)
        if not values:
            continue
        mu = mean([value for value, _ in values])
        sigma = statistics.pstdev([value for value, _ in values]) or 1.0
        for value, positive in values:
            (pos_z if positive else neg_z).append((value - mu) / sigma)
        query_auc = pairwise_auc(values)
        if query_auc is not None:
            query_aucs.append(query_auc)
    return {
        "candidate_score_coverage": coverage / total if total else 0.0,
        "mean_query_pairwise_auc": mean(query_aucs),
        "median_query_pairwise_auc": median(query_aucs),
        "positive_score_mean": mean(pos_scores),
        "negative_score_mean": mean(neg_scores),
        "positive_minus_negative_score": mean(pos_scores) - mean(neg_scores) if pos_scores and neg_scores else 0.0,
        "positive_z_mean": mean(pos_z),
        "negative_z_mean": mean(neg_z),
        "positive_minus_negative_z": mean(pos_z) - mean(neg_z) if pos_z and neg_z else 0.0,
    }


def source_overlap_stats(groups: list[dict[str, Any]]) -> dict[str, Any]:
    per_query_negative_same_source = []
    per_query_top10_negative_same_source = []
    for group in groups:
        positives = {source_prefix(str(candidate.get("doc_id", ""))) for candidate in group.get("candidates", []) if is_positive(candidate)}
        if not positives:
            continue
        negatives = [candidate for candidate in group.get("candidates", []) if not is_positive(candidate)]
        same_negative = [candidate for candidate in negatives if source_prefix(str(candidate.get("doc_id", ""))) in positives]
        dense_top10_negatives = [candidate for candidate in rank_candidates(group, scores=None)[:10] if not is_positive(candidate)]
        dense_top10_same = [candidate for candidate in dense_top10_negatives if source_prefix(str(candidate.get("doc_id", ""))) in positives]
        per_query_negative_same_source.append(len(same_negative) / len(negatives) if negatives else 0.0)
        per_query_top10_negative_same_source.append(len(dense_top10_same) / len(dense_top10_negatives) if dense_top10_negatives else 0.0)
    return {
        "mean_negative_same_source_as_positive_rate": mean(per_query_negative_same_source),
        "mean_dense_top10_negative_same_source_as_positive_rate": mean(per_query_top10_negative_same_source),
    }


def lexical_shape_stats(groups: list[dict[str, Any]]) -> dict[str, Any]:
    query_lengths = [len(str(group.get("query_text") or "").split()) for group in groups]
    positive_lengths = []
    negative_lengths = []
    top_dense_negative_lengths = []
    for group in groups:
        for candidate in group.get("candidates", []):
            target = positive_lengths if is_positive(candidate) else negative_lengths
            target.append(len(str(candidate.get("text") or "").split()))
        for candidate in rank_candidates(group, scores=None)[:10]:
            if not is_positive(candidate):
                top_dense_negative_lengths.append(len(str(candidate.get("text") or "").split()))
    return {
        "query_word_count": describe(query_lengths),
        "positive_word_count": describe(positive_lengths),
        "negative_word_count": describe(negative_lengths),
        "dense_top10_negative_word_count": describe(top_dense_negative_lengths),
    }


def training_generalization(metrics: dict[str, Any]) -> dict[str, Any]:
    selected = metrics.get("selected") if metrics else None
    if not isinstance(selected, dict):
        return {}
    train = selected.get("train_metrics", {})
    dev = selected.get("dev_metrics", {})
    test = selected.get("test_metrics", {})
    out = {
        "selected_alpha": selected.get("alpha"),
        "best_epoch": selected.get("best_epoch"),
    }
    for split_name, split in (("train", train), ("dev", dev), ("test", test)):
        dense = split.get("dense", {}).get("ndcg@10")
        model = split.get("model", {}).get("ndcg@10")
        if dense is not None and model is not None:
            out[f"{split_name}_ndcg_delta"] = float(model) - float(dense)
            out[f"{split_name}_dense_ndcg"] = float(dense)
            out[f"{split_name}_model_ndcg"] = float(model)
    return out


def examples_for(
    groups: list[dict[str, Any]],
    *,
    baseline_scores: ScoreMap | None,
    candidate_scores: ScoreMap,
    limit: int,
    snippet_chars: int,
    descending: bool,
) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        baseline_ranked = rank_candidates(group, scores=baseline_scores)
        candidate_ranked = rank_candidates(group, scores=candidate_scores)
        baseline_ndcg = ndcg(baseline_ranked, 10)
        candidate_ndcg = ndcg(candidate_ranked, 10)
        rows.append(
            {
                "query_id": str(group["query_id"]),
                "query_text": str(group.get("query_text") or ""),
                "ndcg_delta": candidate_ndcg - baseline_ndcg,
                "baseline_first_positive_rank": first_positive_rank(baseline_ranked),
                "candidate_first_positive_rank": first_positive_rank(candidate_ranked),
                "baseline_top": candidate_views(group, baseline_ranked[:5], baseline_scores, candidate_scores, snippet_chars),
                "candidate_top": candidate_views(group, candidate_ranked[:5], baseline_scores, candidate_scores, snippet_chars),
            }
        )
    return sorted(rows, key=lambda row: (row["ndcg_delta"], row["query_id"]), reverse=descending)[:limit]


def candidate_views(
    group: dict[str, Any],
    candidates: list[dict[str, Any]],
    baseline_scores: ScoreMap | None,
    candidate_scores: ScoreMap,
    snippet_chars: int,
) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": str(candidate.get("doc_id", "")),
            "chunk_id": str(candidate.get("chunk_id", "")),
            "label": int(candidate.get("label", 0)),
            "dense_rank": int(candidate.get("dense_rank", 10**9)),
            "dense_score": float(candidate.get("dense_score", 0.0)),
            "baseline_score": score_for(group, candidate, baseline_scores),
            "candidate_score": score_for(group, candidate, candidate_scores),
            "snippet": " ".join(str(candidate.get("text", "")).split())[:snippet_chars],
        }
        for candidate in candidates
    ]


def summarize_cross_dataset(datasets: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for name, payload in datasets.items():
        if "error" in payload:
            rows[name] = payload
            continue
        rows[name] = {
            "query_count": payload["query_count"],
            "dense_positive_at_1_rate": payload["candidate_stats"]["dense_positive_at_1_rate"],
            "dense_positive_at_10_rate": payload["candidate_stats"]["dense_positive_at_10_rate"],
            "actpred_ndcg_delta": payload["rank_movement"].get("actpred_vs_dense", {}).get("mean_ndcg_delta"),
            "ettin_ndcg_delta": payload["rank_movement"].get("ettin_vs_dense", {}).get("mean_ndcg_delta"),
            "actpred_pairwise_auc": payload["score_separation"].get("actpred", {}).get("mean_query_pairwise_auc"),
            "ettin_pairwise_auc": payload["score_separation"].get("ettin", {}).get("mean_query_pairwise_auc"),
            "negative_same_source_rate": payload["source_overlap"]["mean_negative_same_source_as_positive_rate"],
            "train_ndcg_delta": payload["training_generalization"].get("train_ndcg_delta"),
            "dev_ndcg_delta": payload["training_generalization"].get("dev_ndcg_delta"),
            "test_ndcg_delta": payload["training_generalization"].get("test_ndcg_delta"),
        }
    return rows


def load_scores(path_value: str | None) -> ScoreMap:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    return {(str(row["query_id"]), str(row["chunk_id"])): float(row["score"]) for row in load_jsonl(path)}


def load_json(path_value: str | None) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def score_for(group: dict[str, Any], candidate: dict[str, Any], scores: ScoreMap | None) -> float | None:
    if scores is None:
        return float(candidate.get("dense_score", 0.0))
    return scores.get((str(group["query_id"]), str(candidate["chunk_id"])))


def pairwise_auc(values: list[tuple[float, bool]]) -> float | None:
    positives = [score for score, positive in values if positive]
    negatives = [score for score, positive in values if not positive]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = 0
    for positive in positives:
        for negative in negatives:
            total += 1
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total if total else None


def source_prefix(doc_id: str) -> str:
    parts = doc_id.rsplit(":", 3)
    if len(parts) == 4 and all(part.isdigit() for part in parts[1:]):
        return parts[0]
    return doc_id


def ndcg(candidates: list[dict[str, Any]], k: int) -> float:
    ideal = sorted(candidates, key=lambda candidate: -gain(candidate))[:k]
    denom = dcg(ideal)
    return dcg(candidates[:k]) / denom if denom > 0 else 0.0


def dcg(candidates: list[dict[str, Any]]) -> float:
    return sum((2.0**gain(candidate) - 1.0) / math.log2(rank + 1) for rank, candidate in enumerate(candidates, start=1))


def gain(candidate: dict[str, Any]) -> float:
    return float(candidate.get("label_score", candidate.get("label", 0)))


def first_positive_rank(candidates: list[dict[str, Any]]) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if is_positive(candidate):
            return rank
    return None


def is_positive(candidate: dict[str, Any]) -> bool:
    return int(candidate.get("label", 0)) > 0 or float(candidate.get("label_score", 0.0)) > 0.0


def describe(values: list[int | float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def mean(values: list[int | float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[int | float]) -> float:
    return float(statistics.median(values)) if values else 0.0


if __name__ == "__main__":
    main()
