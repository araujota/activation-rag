#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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
from scripts.build_core245_permutation_groups import _load_cache_rows, _load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a query+candidate behavior-latent support reranker.")
    parser.add_argument("--train-groups", action="append", required=True)
    parser.add_argument("--dev-groups", action="append", required=True)
    parser.add_argument("--test-groups", action="append", required=True)
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--activation-transform", choices=("raw", "log1p", "log1p_l2", "binary"), default="raw")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    summary = run_training(
        train_group_paths=[Path(path) for path in args.train_groups],
        dev_group_paths=[Path(path) for path in args.dev_groups],
        test_group_paths=[Path(path) for path in args.test_groups],
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        feature_manifest_path=Path(args.feature_manifest),
        model_out=Path(args.model_out),
        metrics_out=Path(args.metrics_out),
        scores_out=Path(args.scores_out),
        dataset_name=args.dataset_name,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        alpha_grid=tuple(float(item) for item in args.alpha_grid.split(",") if item.strip()),
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        activation_transform=args.activation_transform,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_training(
    *,
    train_group_paths: list[Path],
    dev_group_paths: list[Path],
    test_group_paths: list[Path],
    telemetry_cache_dir: Path,
    feature_manifest_path: Path,
    model_out: Path,
    metrics_out: Path,
    scores_out: Path,
    dataset_name: str,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    alpha_grid: tuple[float, ...],
    dropout: float,
    label_smoothing: float,
    activation_transform: str,
    grad_clip: float,
    early_stopping_patience: int,
    top_k: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    from torch import nn
    import torch.nn.functional as F

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows_by_chunk = _load_cache_rows(telemetry_cache_dir)
    manifest = _load_manifest(feature_manifest_path)
    feature_ids = manifest["feature_ids"]
    train_groups = load_groups(train_group_paths)
    dev_groups = load_groups(dev_group_paths)
    test_groups = load_groups(test_group_paths)
    train_examples = build_examples(train_groups, rows_by_chunk, feature_ids, activation_transform=activation_transform)
    dev_examples = build_examples(dev_groups, rows_by_chunk, feature_ids, activation_transform=activation_transform)
    test_examples = build_examples(test_groups, rows_by_chunk, feature_ids, activation_transform=activation_transform)
    if not train_examples:
        raise ValueError("no train groups with behavior telemetry and positive/negative labels")
    if not dev_examples:
        raise ValueError("no dev groups with behavior telemetry and positive/negative labels")
    normalizer = fit_normalizer(train_examples)
    input_dim = len(train_examples[0]["vectors"][0])
    train_tensors = [example_tensors(example, normalizer, device=device) for example in train_examples]
    metric_key = f"ndcg@{min(10, top_k)}"
    selected: dict[str, Any] | None = None
    alpha_summaries: list[dict[str, Any]] = []

    for alpha in alpha_grid:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_state: dict[str, Any] | None = None
        best_dev_score = float("-inf")
        best_epoch = 0
        stale = 0
        epoch_tail: list[dict[str, float]] = []
        for epoch in range(1, max(1, epochs) + 1):
            random.shuffle(train_tensors)
            losses: list[float] = []
            model.train()
            for vectors, labels, dense_scores in train_tensors:
                if not bool((labels > 0).any()) or not bool((labels <= 0).any()):
                    continue
                optimizer.zero_grad()
                support_scores = model(vectors).squeeze(-1)
                final_scores = ((1.0 - alpha) * torch_zscore(dense_scores)) + (alpha * torch_zscore(support_scores))
                target = smoothed_target(labels, label_smoothing=label_smoothing)
                loss = -(target * F.log_softmax(final_scores / max(temperature, 1e-6), dim=0)).sum()
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            dev_metrics, _ = evaluate_examples(dev_examples, model, normalizer, alpha=alpha, top_k=top_k, device=device)
            dev_score = float(dev_metrics["model"][metric_key])
            epoch_tail.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, "dev_ndcg": dev_score})
            epoch_tail = epoch_tail[-10:]
            if dev_score > best_dev_score + 1e-8:
                best_dev_score = dev_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if early_stopping_patience > 0 and stale >= early_stopping_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        train_metrics, _ = evaluate_examples(train_examples, model, normalizer, alpha=alpha, top_k=top_k, device=device)
        dev_metrics, _ = evaluate_examples(dev_examples, model, normalizer, alpha=alpha, top_k=top_k, device=device)
        test_metrics, test_scores = evaluate_examples(test_examples, model, normalizer, alpha=alpha, top_k=top_k, device=device)
        row = {
            "alpha": float(alpha),
            "best_epoch": best_epoch,
            "best_dev_score": best_dev_score,
            "train_metrics": train_metrics,
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics,
            "epoch_summaries_tail": epoch_tail,
            "state_dict": copy.deepcopy(model.state_dict()),
            "test_scores": test_scores,
        }
        alpha_summaries.append({key: value for key, value in row.items() if key not in {"state_dict", "test_scores"}})
        if selected is None or dev_metrics["model"][metric_key] > selected["dev_metrics"]["model"][metric_key]:
            selected = row
    assert selected is not None
    test_scores = selected.pop("test_scores")
    state_dict = selected.pop("state_dict")
    write_scores(scores_out, test_scores)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "activation_rag.behavior_latent_support_reranker.v1",
            "dataset_name": dataset_name,
            "feature_set_id": manifest["feature_set_id"],
            "feature_ids": feature_ids,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "normalizer": normalizer,
            "state_dict": state_dict,
            "selected_alpha": selected["alpha"],
            "selected_epoch": selected["best_epoch"],
            "prompt_representation": "query_candidate_pair_core245_max_prefill",
            "activation_transform": activation_transform,
        },
        model_out,
    )
    summary = {
        "schema_version": "activation_rag.behavior_latent_support_reranker_run.v1",
        "dataset_name": dataset_name,
        "train_groups": [str(path) for path in train_group_paths],
        "dev_groups": [str(path) for path in dev_group_paths],
        "test_groups": [str(path) for path in test_group_paths],
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "feature_manifest": str(feature_manifest_path),
        "prompt_representation": "query_candidate_pair_core245_max_prefill",
        "activation_transform": activation_transform,
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "alpha_summaries": alpha_summaries,
        "grad_clip": grad_clip,
        "selected": selected,
        "train_query_count": len(train_examples),
        "dev_query_count": len(dev_examples),
        "test_query_count": len(test_examples),
        "model_out": str(model_out),
        "metrics_out": str(metrics_out),
        "scores_out": str(scores_out),
    }
    write_json(metrics_out, summary)
    return summary


def load_groups(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def build_examples(
    groups: list[dict[str, Any]],
    rows_by_chunk: dict[str, dict[str, Any]],
    feature_ids: list[str],
    *,
    activation_transform: str = "raw",
) -> list[dict[str, Any]]:
    examples = []
    for group in groups:
        candidates = []
        vectors = []
        labels = []
        dense_scores = []
        for candidate in group.get("candidates", []):
            behavior_chunk_id = str(candidate.get("behavior_chunk_id") or "")
            row = rows_by_chunk.get(behavior_chunk_id)
            if row is None:
                continue
            candidates.append(candidate)
            vectors.append(vector_for(row, candidate, feature_ids, activation_transform=activation_transform))
            labels.append(int(candidate.get("label", 0)))
            dense_scores.append(float(candidate.get("dense_score", 0.0)))
        if any(label > 0 for label in labels) and any(label <= 0 for label in labels):
            examples.append(
                {
                    "query_id": str(group["query_id"]),
                    "candidates": candidates,
                    "vectors": np.array(vectors, dtype=np.float64),
                    "labels": np.array(labels, dtype=np.int64),
                    "dense_scores": np.array(dense_scores, dtype=np.float64),
                }
            )
    return examples


def vector_for(row: dict[str, Any], candidate: dict[str, Any], feature_ids: list[str], *, activation_transform: str = "raw") -> list[float]:
    values = row.get("sae_feature_values") or {}
    features = candidate.get("features") or {}
    sae_vector = [float(values.get(feature_id, values.get(f"sae.feature.{feature_id}", 0.0))) for feature_id in feature_ids]
    vector = transform_activation_values(sae_vector, activation_transform)
    vector.extend(
        [
            float(candidate.get("dense_score", 0.0)),
            float(features.get("dense_z", 0.0)),
            float(features.get("dense_rank_reciprocal", 1.0 / max(1, int(candidate.get("dense_rank", 10**9))))),
        ]
    )
    return vector


def transform_activation_values(values: list[float], activation_transform: str) -> list[float]:
    array = np.array(values, dtype=np.float64)
    if activation_transform == "raw":
        transformed = array
    elif activation_transform == "log1p":
        transformed = np.sign(array) * np.log1p(np.abs(array))
    elif activation_transform == "log1p_l2":
        transformed = np.sign(array) * np.log1p(np.abs(array))
        norm = float(np.linalg.norm(transformed))
        if norm > 1e-12:
            transformed = transformed / norm
    elif activation_transform == "binary":
        transformed = (array != 0.0).astype(np.float64)
    else:
        raise ValueError(f"unknown activation transform: {activation_transform}")
    return transformed.tolist()


def fit_normalizer(examples: list[dict[str, Any]]) -> dict[str, list[float]]:
    matrix = np.vstack([example["vectors"] for example in examples])
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def example_tensors(example: dict[str, Any], normalizer: dict[str, list[float]], *, device: str) -> tuple[Any, Any, Any]:
    import torch

    mean = np.array(normalizer["mean"], dtype=np.float64)
    scale = np.array(normalizer["scale"], dtype=np.float64)
    vectors = (example["vectors"] - mean) / scale
    return (
        torch.tensor(vectors, dtype=torch.float32, device=device),
        torch.tensor(example["labels"], dtype=torch.long, device=device),
        torch.tensor(example["dense_scores"], dtype=torch.float32, device=device),
    )


def evaluate_examples(examples: list[dict[str, Any]], model: Any, normalizer: dict[str, list[float]], *, alpha: float, top_k: int, device: str) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    import torch

    dense_rankings: dict[str, list[str]] = {}
    model_rankings: dict[str, list[str]] = {}
    qrels: dict[str, dict[str, int]] = {}
    scores: dict[tuple[str, str], float] = {}
    model.eval()
    with torch.no_grad():
        for example in examples:
            query_id = str(example["query_id"])
            candidates = list(example["candidates"])
            qrels[query_id] = {str(candidate["doc_id"]): int(candidate.get("label", 0)) for candidate in candidates if int(candidate.get("label", 0)) > 0}
            dense_ordered = sorted(candidates, key=lambda item: int(item.get("dense_rank", 10**9)))
            dense_rankings[query_id] = [str(candidate["doc_id"]) for candidate in dense_ordered]
            vectors, _labels, dense_scores_tensor = example_tensors(example, normalizer, device=device)
            support = model(vectors).squeeze(-1)
            final = ((1.0 - alpha) * torch_zscore(dense_scores_tensor)) + (alpha * torch_zscore(support))
            final_values = final.detach().cpu().numpy().tolist()
            for candidate, score in zip(candidates, final_values, strict=True):
                scores[(query_id, str(candidate["chunk_id"]))] = float(score)
            model_ordered = [candidate for _score, candidate in sorted(zip(final_values, candidates, strict=True), key=lambda item: item[0], reverse=True)]
            model_rankings[query_id] = [str(candidate["doc_id"]) for candidate in model_ordered]
    return {"dense": aggregate(dense_rankings, qrels, top_k), "model": aggregate(model_rankings, qrels, top_k)}, scores


def aggregate(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], top_k: int) -> dict[str, float]:
    if not rankings:
        return {}
    metric_k = min(10, top_k)
    return {
        f"mrr@{metric_k}": float(sum(mean_reciprocal_rank(ranking, qrels.get(query_id, {}), metric_k) for query_id, ranking in rankings.items()) / len(rankings)),
        f"ndcg@{metric_k}": float(sum(ndcg_at_k(ranking, qrels.get(query_id, {}), metric_k) for query_id, ranking in rankings.items()) / len(rankings)),
        f"recall@{top_k}": float(sum(recall_at_k(ranking, qrels.get(query_id, {}), top_k) for query_id, ranking in rankings.items()) / len(rankings)),
    }


def torch_zscore(values: Any) -> Any:
    std = values.std(unbiased=False)
    if float(std.detach().cpu()) < 1e-8:
        return values * 0.0
    return (values - values.mean()) / std


def smoothed_target(labels: Any, *, label_smoothing: float) -> Any:
    target = labels.float()
    target = target / target.sum()
    if label_smoothing <= 0.0:
        return target
    uniform = target.new_full(target.shape, 1.0 / max(1, target.numel()))
    return ((1.0 - label_smoothing) * target) + (label_smoothing * uniform)


def write_scores(path: Path, scores: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (query_id, chunk_id), score in sorted(scores.items()):
            handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
