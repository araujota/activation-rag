#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from activation_rag.supervised_reranking import evaluate_group_rankings, infer_feature_names, load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a torch MLP activation-aware reranker over dense candidate groups.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--feature", action="append", dest="features")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("listwise_softmax", "pairwise_logistic"), default="listwise_softmax")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--selection-metric", help="Dev metric used to select the saved checkpoint. Defaults to nDCG at top-k.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    summary = run_mlp_training(
        train_path=Path(args.train),
        dev_path=Path(args.dev),
        test_path=Path(args.test) if args.test else None,
        model_out=Path(args.model_out),
        metrics_out=Path(args.metrics_out),
        feature_names=tuple(args.features) if args.features else None,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        top_k=args.top_k,
        selection_metric=args.selection_metric,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_mlp_training(
    *,
    train_path: Path,
    dev_path: Path,
    test_path: Path | None,
    model_out: Path,
    metrics_out: Path,
    feature_names: tuple[str, ...] | None,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    top_k: int,
    selection_metric: str | None = None,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_groups = load_jsonl(train_path)
    dev_groups = load_jsonl(dev_path)
    test_groups = load_jsonl(test_path) if test_path else []
    selected_features = tuple(feature_names or infer_feature_names(train_groups))
    normalizer = _fit_normalizer(train_groups, selected_features)
    model = nn.Sequential(
        nn.Linear(len(selected_features), hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_tensors = [_group_tensors(group, selected_features, normalizer, device=device) for group in train_groups]
    train_tensors = [item for item in train_tensors if item is not None]
    if not train_tensors:
        raise ValueError("no train groups with positive and negative candidates")
    metric_name = selection_metric or f"ndcg@{min(10, top_k)}"
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, Any] | None = None
    for epoch in range(1, max(1, epochs) + 1):
        random.shuffle(train_tensors)
        model.train()
        for features, labels in train_tensors:
            optimizer.zero_grad()
            logits = model(features).squeeze(-1)
            loss = _loss(logits, labels, loss_name)
            loss.backward()
            optimizer.step()
        scorer = TorchMlpScorer(model=model, feature_names=selected_features, normalizer=normalizer, device=device)
        dev_metrics = evaluate_group_rankings(dev_groups, model=scorer, top_k=top_k)
        score = float(dev_metrics["model"].get(metric_name, float("-inf")))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    scorer = TorchMlpScorer(model=model, feature_names=selected_features, normalizer=normalizer, device=device)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "activation_rag.activation_mlp_reranker.v1",
            "state_dict": model.state_dict(),
            "feature_names": selected_features,
            "normalizer": normalizer,
            "hidden_dim": hidden_dim,
            "loss": loss_name,
            "selection_metric": metric_name,
            "best_epoch": best_epoch,
            "best_dev_score": best_score,
        },
        model_out,
    )
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.activation_mlp_training_run.v1",
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "test_path": str(test_path) if test_path else None,
        "model_out": str(model_out),
        "feature_names": list(selected_features),
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss": loss_name,
        "selection_metric": metric_name,
        "best_epoch": best_epoch,
        "best_dev_score": best_score,
        "device": device,
        "train_query_count": len(train_groups),
        "dev_query_count": len(dev_groups),
        "test_query_count": len(test_groups),
        "top_k": top_k,
        "train_metrics": evaluate_group_rankings(train_groups, model=scorer, top_k=top_k),
        "dev_metrics": evaluate_group_rankings(dev_groups, model=scorer, top_k=top_k),
    }
    if test_groups:
        summary["test_metrics"] = evaluate_group_rankings(test_groups, model=scorer, top_k=top_k)
    write_json(metrics_out, summary)
    return summary


class TorchMlpScorer:
    def __init__(self, *, model: Any, feature_names: tuple[str, ...], normalizer: dict[str, dict[str, float]], device: str) -> None:
        self.model = model
        self.feature_names = feature_names
        self.normalizer = normalizer
        self.device = device

    def score_candidate(self, candidate: dict[str, Any]) -> float:
        import torch

        features = _candidate_vector(candidate, self.feature_names, self.normalizer)
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.model(tensor).squeeze().detach().cpu())


def _loss(logits: Any, labels: Any, loss_name: str) -> Any:
    import torch
    import torch.nn.functional as F

    if loss_name == "listwise_softmax":
        positives = labels > 0
        if not bool(positives.any()):
            return logits.sum() * 0.0
        target = labels.float()
        target = target / target.sum()
        return -(target * F.log_softmax(logits, dim=0)).sum()
    if loss_name == "pairwise_logistic":
        pos = logits[labels > 0]
        neg = logits[labels <= 0]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.sum() * 0.0
        margins = pos[:, None] - neg[None, :]
        return F.binary_cross_entropy_with_logits(margins, torch.ones_like(margins))
    raise ValueError(f"unknown loss: {loss_name}")


def _fit_normalizer(groups: list[dict[str, Any]], feature_names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    matrix = np.array(
        [
            [float((candidate.get("features") or {}).get(name, 0.0)) for name in feature_names]
            for group in groups
            for candidate in group.get("candidates", [])
        ],
        dtype=np.float64,
    )
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    return {
        "means": {name: float(means[index]) for index, name in enumerate(feature_names)},
        "scales": {name: float(scales[index]) for index, name in enumerate(feature_names)},
    }


def _group_tensors(group: dict[str, Any], feature_names: tuple[str, ...], normalizer: dict[str, dict[str, float]], *, device: str):
    import torch

    candidates = list(group.get("candidates", []))
    labels = np.array([int(candidate.get("label", 0)) for candidate in candidates], dtype=np.int64)
    if labels.max(initial=0) <= 0 or (labels <= 0).sum() == 0:
        return None
    features = np.array([_candidate_vector(candidate, feature_names, normalizer) for candidate in candidates], dtype=np.float32)
    return (
        torch.tensor(features, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
    )


def _candidate_vector(candidate: dict[str, Any], feature_names: tuple[str, ...], normalizer: dict[str, dict[str, float]]) -> list[float]:
    features = candidate.get("features") or {}
    means = normalizer["means"]
    scales = normalizer["scales"]
    return [
        (float(features.get(name, 0.0)) - means[name]) / scales[name]
        for name in feature_names
    ]


if __name__ == "__main__":
    main()
