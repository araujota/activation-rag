from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from activation_rag.benchmarks import mean_reciprocal_rank, ndcg_at_k, recall_at_k


@dataclass(frozen=True)
class PairwiseLinearRanker:
    feature_names: tuple[str, ...]
    weights: dict[str, float]
    bias: float
    means: dict[str, float]
    scales: dict[str, float]
    schema_version: str = "activation_rag.pairwise_linear_ranker.v1"

    def score_candidate(self, candidate: dict[str, Any]) -> float:
        features = candidate.get("features") or {}
        score = self.bias
        for name in self.feature_names:
            value = float(features.get(name, 0.0))
            normalized = (value - self.means[name]) / self.scales[name]
            score += self.weights[name] * normalized
        return float(score)

    def score_group(self, group: dict[str, Any]) -> dict[str, float]:
        return {
            str(candidate["chunk_id"]): self.score_candidate(candidate)
            for candidate in group.get("candidates", [])
        }

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_names": list(self.feature_names),
            "weights": self.weights,
            "bias": self.bias,
            "means": self.means,
            "scales": self.scales,
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> "PairwiseLinearRanker":
        return cls(
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            weights={str(key): float(value) for key, value in payload["weights"].items()},
            bias=float(payload.get("bias", 0.0)),
            means={str(key): float(value) for key, value in payload["means"].items()},
            scales={str(key): float(value) for key, value in payload["scales"].items()},
        )

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_pairwise_linear_ranker(
    groups: list[dict[str, Any]],
    *,
    feature_names: Iterable[str] | None = None,
    epochs: int = 100,
    learning_rate: float = 0.05,
    l2: float = 1e-4,
) -> PairwiseLinearRanker:
    feature_names = tuple(feature_names or infer_feature_names(groups))
    if not feature_names:
        raise ValueError("at least one feature is required")
    candidate_matrix = _candidate_matrix(groups, feature_names)
    if candidate_matrix.size == 0:
        raise ValueError("no candidates available for training")
    means = candidate_matrix.mean(axis=0)
    scales = candidate_matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    examples = _pairwise_examples(groups, feature_names, means=means, scales=scales)
    if examples.size == 0:
        raise ValueError("no positive-vs-negative pairs available for training")
    x = examples
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(max(1, epochs)):
        margins = x @ weights + bias
        probabilities = _sigmoid(margins)
        gradient_factor = probabilities - 1.0
        grad_w = (gradient_factor[:, None] * x).mean(axis=0) + (l2 * weights)
        grad_b = float(gradient_factor.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
    return PairwiseLinearRanker(
        feature_names=feature_names,
        weights={name: float(weights[index]) for index, name in enumerate(feature_names)},
        bias=float(bias),
        means={name: float(means[index]) for index, name in enumerate(feature_names)},
        scales={name: float(scales[index]) for index, name in enumerate(feature_names)},
    )


def infer_feature_names(groups: list[dict[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for group in groups:
        for candidate in group.get("candidates", []):
            names.update(str(key) for key in (candidate.get("features") or {}))
    return tuple(sorted(names))


def evaluate_group_rankings(
    groups: list[dict[str, Any]],
    *,
    model: PairwiseLinearRanker | None = None,
    top_k: int = 10,
) -> dict[str, dict[str, float]]:
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
        dense_ordered = sorted(candidates, key=lambda item: int(item.get("dense_rank", 10**9)))
        dense_rankings[query_id] = [str(candidate["doc_id"]) for candidate in dense_ordered]
        if model is not None:
            model_ordered = sorted(candidates, key=model.score_candidate, reverse=True)
            model_rankings[query_id] = [str(candidate["doc_id"]) for candidate in model_ordered]
    metrics = {"dense": _aggregate(dense_rankings, qrels, top_k)}
    if model is not None:
        metrics["model"] = _aggregate(model_rankings, qrels, top_k)
    return metrics


def _candidate_matrix(groups: list[dict[str, Any]], feature_names: tuple[str, ...]) -> np.ndarray:
    vectors = [
        _feature_vector(candidate, feature_names)
        for group in groups
        for candidate in group.get("candidates", [])
    ]
    if not vectors:
        return np.zeros((0, len(feature_names)), dtype=np.float64)
    return np.vstack(vectors)


def _pairwise_examples(
    groups: list[dict[str, Any]],
    feature_names: tuple[str, ...],
    *,
    means: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    examples: list[list[float]] = []
    for group in groups:
        positives = [candidate for candidate in group.get("candidates", []) if int(candidate.get("label", 0)) > 0]
        negatives = [candidate for candidate in group.get("candidates", []) if int(candidate.get("label", 0)) <= 0]
        for positive in positives:
            positive_vector = (_feature_vector(positive, feature_names) - means) / scales
            for negative in negatives:
                negative_vector = (_feature_vector(negative, feature_names) - means) / scales
                examples.append((positive_vector - negative_vector).tolist())
    if not examples:
        return np.zeros((0, len(feature_names)), dtype=np.float64)
    return np.array(examples, dtype=np.float64)


def _feature_vector(candidate: dict[str, Any], feature_names: tuple[str, ...]) -> np.ndarray:
    features = candidate.get("features") or {}
    return np.array([float(features.get(name, 0.0)) for name in feature_names], dtype=np.float64)


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


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))
