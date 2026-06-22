#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.benchmarks import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from activation_rag.supervised_reranking import infer_feature_names, load_jsonl, write_json
from scripts.train_activation_mlp_reranker import run_mlp_training


DENSE_FEATURES = ("dense_score", "dense_rank_reciprocal")
MATCHER_PREFIX = "core245_matcher:"
CATEGORY_PREFIX = "core245_category:"
COUNTERFACTUAL_PREFIX = "core245_counterfactual:"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Core245 hubness and feature-source isolation diagnostics.")
    parser.add_argument("--sweep-dir", required=True, help="Existing core245 permutation sweep directory.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("listwise_softmax", "pairwise_logistic"), default="listwise_softmax")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    report = run_diagnostics(
        sweep_dir=Path(args.sweep_dir),
        out_dir=Path(args.out_dir),
        report_out=Path(args.report_out),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def run_diagnostics(
    *,
    sweep_dir: Path,
    out_dir: Path,
    report_out: Path,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    top_k: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    category_paths = _variant_paths(sweep_dir, "qwen_l07_core245_causal_weighted_category")
    counterfactual_paths = _variant_paths(sweep_dir, "qwen_l07_core245_counterfactual_matched")
    category_train = load_jsonl(category_paths["train"])
    counterfactual_train = load_jsonl(counterfactual_paths["train"])
    category_features = infer_feature_names(category_train)
    counterfactual_features = infer_feature_names(counterfactual_train)

    jobs = [
        _job("dense_only_from_category", category_paths, category_features, lambda name: name in DENSE_FEATURES),
        _job("matcher_only_from_category", category_paths, category_features, lambda name: name in DENSE_FEATURES or name.startswith(MATCHER_PREFIX)),
        _job("category_aggregates_only", category_paths, category_features, lambda name: name in DENSE_FEATURES or name.startswith(CATEGORY_PREFIX)),
        _job("category_aggregates_plus_matcher", category_paths, category_features, lambda name: name in DENSE_FEATURES or name.startswith(CATEGORY_PREFIX) or name.startswith(MATCHER_PREFIX)),
        _job("matcher_only_from_counterfactual", counterfactual_paths, counterfactual_features, lambda name: name in DENSE_FEATURES or name.startswith(MATCHER_PREFIX)),
        _job("counterfactual_aggregates_only", counterfactual_paths, counterfactual_features, lambda name: name in DENSE_FEATURES or name.startswith(COUNTERFACTUAL_PREFIX)),
        _job("counterfactual_aggregates_plus_matcher", counterfactual_paths, counterfactual_features, lambda name: name in DENSE_FEATURES or name.startswith(COUNTERFACTUAL_PREFIX) or name.startswith(MATCHER_PREFIX)),
    ]

    reranker_runs: list[dict[str, Any]] = []
    for job in jobs:
        run_dir = out_dir / job["name"]
        metrics = run_mlp_training(
            train_path=job["paths"]["train"],
            dev_path=job["paths"]["dev"],
            test_path=job["paths"]["test"],
            model_out=run_dir / "model.pt",
            metrics_out=run_dir / "metrics.json",
            feature_names=tuple(job["feature_names"]),
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            loss_name=loss_name,
            top_k=top_k,
            device=device,
            seed=seed,
        )
        reranker_runs.append(_summarize_training_run(job["name"], job["feature_names"], metrics))

    test_groups = load_jsonl(category_paths["test"])
    matcher_features = [name for name in infer_feature_names(test_groups) if name.startswith(MATCHER_PREFIX)]
    rank_diagnostics = {
        "dense_rank": _score_diagnostics(test_groups, score_name="dense_rank", scorer=lambda candidate: -float(candidate.get("dense_rank", 10**9)), top_k=top_k),
    }
    for feature_name in matcher_features:
        rank_diagnostics[feature_name] = _score_diagnostics(
            test_groups,
            score_name=feature_name,
            scorer=lambda candidate, name=feature_name: float((candidate.get("features") or {}).get(name, 0.0)),
            top_k=top_k,
        )

    report: dict[str, Any] = {
        "schema_version": "activation_rag.core245_hubness_diagnostics.v1",
        "sweep_dir": str(sweep_dir),
        "out_dir": str(out_dir),
        "top_k": top_k,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "loss": loss_name,
        "device": device,
        "reranker_runs": reranker_runs,
        "rank_diagnostics": rank_diagnostics,
        "interpretation_hints": {
            "matcher_only_test_delta_vs_dense": _metric_delta(
                _by_name(reranker_runs, "matcher_only_from_category")["test_metrics"]["model"],
                _by_name(reranker_runs, "matcher_only_from_category")["test_metrics"]["dense"],
                top_k,
            ),
            "category_aggregates_only_test_delta_vs_dense": _metric_delta(
                _by_name(reranker_runs, "category_aggregates_only")["test_metrics"]["model"],
                _by_name(reranker_runs, "category_aggregates_only")["test_metrics"]["dense"],
                top_k,
            ),
            "counterfactual_aggregates_only_test_delta_vs_dense": _metric_delta(
                _by_name(reranker_runs, "counterfactual_aggregates_only")["test_metrics"]["model"],
                _by_name(reranker_runs, "counterfactual_aggregates_only")["test_metrics"]["dense"],
                top_k,
            ),
        },
    }
    write_json(report_out, report)
    return report


def _variant_paths(sweep_dir: Path, variant: str) -> dict[str, Path]:
    root = sweep_dir / variant
    paths = {split: root / f"{split}.jsonl" for split in ("train", "dev", "test")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing variant files for {variant}: {missing}")
    return paths


def _job(
    name: str,
    paths: dict[str, Path],
    feature_names: tuple[str, ...],
    predicate: Callable[[str], bool],
) -> dict[str, Any]:
    selected = tuple(name for name in feature_names if predicate(name))
    if not selected:
        raise ValueError(f"diagnostic job {name} has no features")
    return {"name": name, "paths": paths, "feature_names": selected}


def _summarize_training_run(name: str, feature_names: tuple[str, ...], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "feature_count": len(feature_names),
        "feature_prefix_counts": _feature_prefix_counts(feature_names),
        "best_epoch": metrics.get("best_epoch"),
        "best_dev_score": metrics.get("best_dev_score"),
        "train_metrics": metrics.get("train_metrics"),
        "dev_metrics": metrics.get("dev_metrics"),
        "test_metrics": metrics.get("test_metrics"),
    }


def _feature_prefix_counts(feature_names: tuple[str, ...]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for name in feature_names:
        if name in DENSE_FEATURES:
            counts["dense"] += 1
        elif name.startswith(MATCHER_PREFIX):
            counts["matcher"] += 1
        elif name.startswith(CATEGORY_PREFIX):
            counts["category"] += 1
        elif name.startswith(COUNTERFACTUAL_PREFIX):
            counts["counterfactual"] += 1
        else:
            counts["other"] += 1
    return dict(sorted(counts.items()))


def _score_diagnostics(
    groups: list[dict[str, Any]],
    *,
    score_name: str,
    scorer: Callable[[dict[str, Any]], float],
    top_k: int,
) -> dict[str, Any]:
    qrels: dict[str, dict[str, int]] = {}
    rankings: dict[str, list[str]] = {}
    top1_chunks: list[str] = []
    topk_chunks: list[str] = []
    reciprocal_top1 = 0
    for group in groups:
        query_id = str(group["query_id"])
        candidates = list(group.get("candidates", []))
        qrels[query_id] = {
            str(candidate["doc_id"]): int(candidate.get("label", 0))
            for candidate in candidates
            if int(candidate.get("label", 0)) > 0
        }
        ordered = sorted(candidates, key=scorer, reverse=True)
        rankings[query_id] = [str(candidate["doc_id"]) for candidate in ordered]
        if ordered:
            top1_chunks.append(str(ordered[0].get("chunk_id") or ordered[0].get("doc_id")))
            top_chunk_ids = [str(candidate.get("chunk_id") or candidate.get("doc_id")) for candidate in ordered[:top_k]]
            topk_chunks.extend(top_chunk_ids)
            query_chunk_id = str(group.get("query_activation_chunk_id") or "")
            if query_chunk_id and query_chunk_id in top_chunk_ids:
                reciprocal_top1 += int(top_chunk_ids[0] == query_chunk_id)
    metrics = _aggregate(rankings, qrels, top_k)
    return {
        "score_name": score_name,
        "metrics": metrics,
        "top1_hubness": _occurrence_stats(top1_chunks),
        f"top{top_k}_hubness": _occurrence_stats(topk_chunks),
        "self_top1_count": reciprocal_top1,
    }


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


def _occurrence_stats(items: list[str]) -> dict[str, Any]:
    counts = Counter(items)
    values = list(counts.values())
    total = sum(values)
    return {
        "total_occurrences": total,
        "unique_items": len(counts),
        "max_occurrences": max(values) if values else 0,
        "max_fraction": float(max(values) / total) if total else 0.0,
        "gini": _gini(values),
        "entropy": _entropy(values),
        "top_items": [{"id": item, "count": count} for item, count in counts.most_common(10)],
    }


def _gini(values: list[int]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    total = sum(sorted_values)
    if total <= 0.0:
        return 0.0
    n = len(sorted_values)
    weighted = sum((index + 1) * value for index, value in enumerate(sorted_values))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def _entropy(values: list[int]) -> float:
    total = sum(values)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        probability = value / total
        entropy -= probability * math.log(probability)
    return float(entropy)


def _metric_delta(model_metrics: dict[str, float], dense_metrics: dict[str, float], top_k: int) -> dict[str, float]:
    keys = (f"mrr@{min(10, top_k)}", f"ndcg@{min(10, top_k)}", f"recall@{top_k}")
    return {key: float(model_metrics.get(key, 0.0) - dense_metrics.get(key, 0.0)) for key in keys}


def _by_name(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row["name"] == name:
            return row
    raise KeyError(name)


if __name__ == "__main__":
    main()
