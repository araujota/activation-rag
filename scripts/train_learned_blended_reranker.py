#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
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


FEATURE_NAMES = (
    "z_dense",
    "z_actpred",
    "dense_rank_reciprocal",
    "z_dense_x_z_actpred",
    "actpred_minus_dense",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dataset-specific learned dense/activation blended reranker.")
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--dev-groups", required=True)
    parser.add_argument("--test-groups", required=True)
    parser.add_argument("--train-actpred-scores", required=True)
    parser.add_argument("--dev-actpred-scores", required=True)
    parser.add_argument("--test-actpred-scores", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--test-scores-out", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--max-pairs-per-query", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    summary = run_training(
        train_groups_path=Path(args.train_groups),
        dev_groups_path=Path(args.dev_groups),
        test_groups_path=Path(args.test_groups),
        train_scores_path=Path(args.train_actpred_scores),
        dev_scores_path=Path(args.dev_actpred_scores),
        test_scores_path=Path(args.test_actpred_scores),
        model_out=Path(args.model_out),
        metrics_out=Path(args.metrics_out),
        test_scores_out=Path(args.test_scores_out),
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        max_pairs_per_query=args.max_pairs_per_query,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_training(
    *,
    train_groups_path: Path,
    dev_groups_path: Path,
    test_groups_path: Path,
    train_scores_path: Path,
    dev_scores_path: Path,
    test_scores_path: Path,
    model_out: Path,
    metrics_out: Path,
    test_scores_out: Path,
    dataset_name: str,
    epochs: int,
    learning_rate: float,
    l2: float,
    max_pairs_per_query: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    np.random.seed(seed)
    train_groups = load_jsonl(train_groups_path)
    dev_groups = load_jsonl(dev_groups_path)
    test_groups = load_jsonl(test_groups_path)
    train_scores = load_score_jsonl(train_scores_path)
    dev_scores = load_score_jsonl(dev_scores_path)
    test_scores = load_score_jsonl(test_scores_path)

    train_examples = _build_feature_groups(train_groups, train_scores)
    dev_examples = _build_feature_groups(dev_groups, dev_scores)
    test_examples = _build_feature_groups(test_groups, test_scores)
    if not train_examples:
        raise ValueError("no train groups with at least one positive, one negative, and actpred scores")
    if not dev_examples:
        raise ValueError("no dev groups with at least one positive, one negative, and actpred scores")

    weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    weights[FEATURE_NAMES.index("z_dense")] = 1.0
    bias = 0.0
    best_weights = weights.copy()
    best_bias = bias
    best_epoch = 0
    metric_key = f"ndcg@{min(10, top_k)}"
    initial_dev_metrics, _ = _evaluate(dev_examples, weights, bias, top_k=top_k)
    best_dev_score = float(initial_dev_metrics["model"][metric_key])
    epoch_tail: list[dict[str, float]] = []

    for epoch in range(1, max(1, epochs) + 1):
        rng.shuffle(train_examples)
        losses: list[float] = []
        for example in train_examples:
            pairs = _pair_indices(example["labels"], example["dense_ranks"], max_pairs_per_query=max_pairs_per_query, rng=rng)
            if not pairs:
                continue
            pos_indices = np.array([pos_idx for pos_idx, _ in pairs], dtype=np.int64)
            neg_indices = np.array([neg_idx for _, neg_idx in pairs], dtype=np.int64)
            deltas = example["features"][pos_indices] - example["features"][neg_indices]
            margins = deltas @ weights
            losses.extend(float(value) for value in _softplus_array(-margins))
            # Bias cancels in pairwise differences, so it is only persisted for future extensibility.
            scales = _sigmoid_array(-margins)
            grad = (-(deltas * scales[:, None]).mean(axis=0)) + (l2 * weights)
            weights -= learning_rate * grad
        dev_metrics, _ = _evaluate(dev_examples, weights, bias, top_k=top_k)
        dev_score = float(dev_metrics["model"][metric_key])
        epoch_tail.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, "dev_ndcg": dev_score})
        epoch_tail = epoch_tail[-10:]
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            best_epoch = epoch
            best_weights = weights.copy()
            best_bias = bias

    train_metrics, _ = _evaluate(train_examples, best_weights, best_bias, top_k=top_k)
    dev_metrics, _ = _evaluate(dev_examples, best_weights, best_bias, top_k=top_k)
    test_metrics, test_score_map = _evaluate(test_examples, best_weights, best_bias, top_k=top_k)

    model = {
        "schema_version": "activation_rag.learned_blended_reranker.v1",
        "dataset_name": dataset_name,
        "feature_names": list(FEATURE_NAMES),
        "weights": {name: float(value) for name, value in zip(FEATURE_NAMES, best_weights, strict=True)},
        "bias": float(best_bias),
        "best_epoch": best_epoch,
        "best_dev_score": best_dev_score,
        "learning_rate": learning_rate,
        "l2": l2,
        "max_pairs_per_query": max_pairs_per_query,
        "seed": seed,
    }
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_score_jsonl(test_scores_out, test_score_map)
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.learned_blended_reranker_run.v1",
        "dataset_name": dataset_name,
        "train_groups": str(train_groups_path),
        "dev_groups": str(dev_groups_path),
        "test_groups": str(test_groups_path),
        "train_actpred_scores": str(train_scores_path),
        "dev_actpred_scores": str(dev_scores_path),
        "test_actpred_scores": str(test_scores_path),
        "feature_names": list(FEATURE_NAMES),
        "weights": model["weights"],
        "best_epoch": best_epoch,
        "best_dev_score": best_dev_score,
        "train_query_count": len(train_examples),
        "dev_query_count": len(dev_examples),
        "test_query_count": len(test_examples),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "epoch_summaries_tail": epoch_tail,
        "model_out": str(model_out),
        "metrics_out": str(metrics_out),
        "test_scores_out": str(test_scores_out),
    }
    write_json(metrics_out, summary)
    return summary


def _build_feature_groups(groups: list[dict[str, Any]], scores: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for group in groups:
        query_id = str(group["query_id"])
        candidates = list(group.get("candidates", []))
        kept = [candidate for candidate in candidates if (query_id, str(candidate["chunk_id"])) in scores]
        if not kept:
            continue
        labels = np.array([int(candidate.get("label", 0)) for candidate in kept], dtype=np.int64)
        if not bool((labels > 0).any()) or not bool((labels <= 0).any()):
            continue
        dense = np.array(
            [float(candidate.get("dense_score", (candidate.get("features") or {}).get("dense_score", 0.0))) for candidate in kept],
            dtype=np.float64,
        )
        act = np.array([float(scores[(query_id, str(candidate["chunk_id"]))]) for candidate in kept], dtype=np.float64)
        z_dense = _zscore(dense)
        z_act = _zscore(act)
        rank_recip = np.array([1.0 / max(1.0, float(candidate.get("dense_rank", 10**9))) for candidate in kept], dtype=np.float64)
        features = np.column_stack(
            [
                z_dense,
                z_act,
                rank_recip,
                z_dense * z_act,
                z_act - z_dense,
            ]
        ).astype(np.float64)
        examples.append(
            {
                "query_id": query_id,
                "features": features,
                "labels": labels,
                "dense_ranks": np.array([int(candidate.get("dense_rank", 10**9)) for candidate in kept], dtype=np.int64),
                "doc_ids": [str(candidate["doc_id"]) for candidate in kept],
                "chunk_ids": [str(candidate["chunk_id"]) for candidate in kept],
            }
        )
    return examples


def _pair_indices(labels: np.ndarray, dense_ranks: np.ndarray, *, max_pairs_per_query: int, rng: random.Random) -> list[tuple[int, int]]:
    positives = [idx for idx, label in enumerate(labels) if label > 0]
    negatives = [idx for idx, label in enumerate(labels) if label <= 0]
    if not positives or not negatives:
        return []
    negatives = sorted(negatives, key=lambda idx: int(dense_ranks[idx]))
    hard_negatives = negatives[: max(1, min(len(negatives), max_pairs_per_query))]
    pairs = [(pos_idx, neg_idx) for pos_idx in positives for neg_idx in hard_negatives]
    if len(pairs) > max_pairs_per_query:
        pairs = rng.sample(pairs, max_pairs_per_query)
    return pairs


def _evaluate(examples: list[dict[str, Any]], weights: np.ndarray, bias: float, *, top_k: int) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    dense_rankings: dict[str, list[str]] = {}
    model_rankings: dict[str, list[str]] = {}
    qrels: dict[str, dict[str, int]] = {}
    score_map: dict[tuple[str, str], float] = {}
    for example in examples:
        query_id = str(example["query_id"])
        model_scores = (example["features"] @ weights) + bias
        for chunk_id, score in zip(example["chunk_ids"], model_scores, strict=True):
            score_map[(query_id, str(chunk_id))] = float(score)
        qrels[query_id] = {
            doc_id: int(label)
            for doc_id, label in zip(example["doc_ids"], example["labels"], strict=True)
            if int(label) > 0
        }
        dense_order = sorted(range(len(example["doc_ids"])), key=lambda idx: int(example["dense_ranks"][idx]))
        model_order = sorted(range(len(example["doc_ids"])), key=lambda idx: float(model_scores[idx]), reverse=True)
        dense_rankings[query_id] = [example["doc_ids"][idx] for idx in dense_order]
        model_rankings[query_id] = [example["doc_ids"][idx] for idx in model_order]
    return {"dense": _aggregate(dense_rankings, qrels, top_k), "model": _aggregate(model_rankings, qrels, top_k)}, score_map


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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _softplus(value: float) -> float:
    if value > 30:
        return value
    if value < -30:
        return math.exp(value)
    return math.log1p(math.exp(value))


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _softplus_array(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(values, 0.0)


if __name__ == "__main__":
    main()
