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

from activation_rag.supervised_reranking import load_jsonl, write_json
from scripts.build_core245_permutation_groups import _load_cache_rows, _load_manifest
from scripts.train_activation_representation_searcher import (
    _build_examples,
    _fit_normalizer,
    _make_vectorizer,
    _select_blend_alpha_from_sweep,
    evaluate_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an answer-activation predictor directly on blended dense/activation ranking loss.")
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--dev-groups", required=True)
    parser.add_argument("--test-groups", required=True)
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--representation", default="raw")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha-grid", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    summary = run_training(
        train_groups_path=Path(args.train_groups),
        dev_groups_path=Path(args.dev_groups),
        test_groups_path=Path(args.test_groups),
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        feature_manifest_path=Path(args.feature_manifest),
        representation=args.representation,
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
        input_noise_std=args.input_noise_std,
        label_smoothing=args.label_smoothing,
        grad_clip_norm=args.grad_clip_norm,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_training(
    *,
    train_groups_path: Path,
    dev_groups_path: Path,
    test_groups_path: Path,
    telemetry_cache_dir: Path,
    feature_manifest_path: Path,
    representation: str,
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
    input_noise_std: float,
    label_smoothing: float,
    grad_clip_norm: float,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
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
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if input_noise_std < 0.0:
        raise ValueError("input_noise_std must be non-negative")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    if grad_clip_norm < 0.0:
        raise ValueError("grad_clip_norm must be non-negative")

    rows_by_chunk = _load_cache_rows(telemetry_cache_dir)
    manifest = _load_manifest(feature_manifest_path)
    feature_ids = manifest["feature_ids"]
    vectorizer = _make_vectorizer(
        representation=representation,
        feature_ids=feature_ids,
        feature_meta=manifest["feature_meta"],
        seed=seed,
    )
    train_examples = _build_examples(load_jsonl(train_groups_path), rows_by_chunk, feature_ids, vectorizer)
    dev_examples = _build_examples(load_jsonl(dev_groups_path), rows_by_chunk, feature_ids, vectorizer)
    test_examples = _build_examples(load_jsonl(test_groups_path), rows_by_chunk, feature_ids, vectorizer)
    if not train_examples:
        raise ValueError("no train examples with query telemetry, positive candidates, and negative candidates")
    if not dev_examples:
        raise ValueError("no dev examples with query telemetry, positive candidates, and negative candidates")

    normalizer = _fit_normalizer(train_examples)
    input_dim = int(train_examples[0].query_vector.shape[0])
    train_tensors = [_example_tensors(example, normalizer, device=device) for example in train_examples]
    metric_key = f"ndcg@{min(10, top_k)}"
    best_summary: dict[str, Any] | None = None

    for alpha in alpha_grid:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_state: dict[str, Any] | None = None
        best_dev_score = float("-inf")
        best_epoch = 0
        stale_epochs = 0
        epoch_tail: list[dict[str, float]] = []
        for epoch in range(1, max(1, epochs) + 1):
            random.shuffle(train_tensors)
            losses: list[float] = []
            model.train()
            for query, candidates, labels, dense_scores in train_tensors:
                positives = labels > 0
                if not bool(positives.any()) or bool((labels <= 0).sum() == 0):
                    continue
                optimizer.zero_grad()
                noisy_query = query
                if input_noise_std > 0.0:
                    noisy_query = query + torch.randn_like(query) * input_noise_std
                prediction = _forward_with_training_dropout(model, noisy_query, dropout=dropout)
                act_scores = (F.normalize(prediction.unsqueeze(0), dim=1) @ F.normalize(candidates, dim=1).T).squeeze(0)
                final_scores = ((1.0 - alpha) * _torch_zscore(dense_scores)) + (alpha * _torch_zscore(act_scores))
                target = _smoothed_target(labels, label_smoothing=label_smoothing)
                loss = -(target * F.log_softmax(final_scores / max(temperature, 1e-6), dim=0)).sum()
                loss.backward()
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            dev_metrics, _ = evaluate_examples(dev_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=alpha, device=device)
            dev_score = float(dev_metrics["model"].get(metric_key, float("-inf")))
            epoch_tail.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, "dev_ndcg": dev_score})
            epoch_tail = epoch_tail[-10:]
            if dev_score > best_dev_score + early_stopping_min_delta:
                best_dev_score = dev_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        train_metrics, _ = evaluate_examples(train_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=alpha, device=device)
        dev_metrics, _ = evaluate_examples(dev_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=alpha, device=device)
        test_metrics, test_scores = evaluate_examples(test_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=alpha, device=device)
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
        if best_summary is None or dev_metrics["model"][metric_key] > best_summary["dev_metrics"]["model"][metric_key]:
            best_summary = row

    assert best_summary is not None
    selected_scores = best_summary.pop("test_scores")
    selected_state = best_summary.pop("state_dict")
    _write_scores(scores_out, selected_scores)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "activation_rag.direct_blend_answer_representation_searcher.v1",
            "dataset_name": dataset_name,
            "state_dict": selected_state,
            "representation": representation,
            "feature_set_id": manifest["feature_set_id"],
            "feature_ids": feature_ids,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "normalizer": normalizer,
            "selected_alpha": float(best_summary["alpha"]),
            "selected_epoch": int(best_summary["best_epoch"]),
            "temperature": temperature,
            "dropout": dropout,
            "input_noise_std": input_noise_std,
            "label_smoothing": label_smoothing,
            "grad_clip_norm": grad_clip_norm,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        },
        model_out,
    )
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.direct_blend_answer_representation_searcher_run.v1",
        "dataset_name": dataset_name,
        "representation": representation,
        "feature_set_id": manifest["feature_set_id"],
        "train_groups": str(train_groups_path),
        "dev_groups": str(dev_groups_path),
        "test_groups": str(test_groups_path),
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "alpha_grid": [float(alpha) for alpha in alpha_grid],
        "dropout": dropout,
        "input_noise_std": input_noise_std,
        "label_smoothing": label_smoothing,
        "grad_clip_norm": grad_clip_norm,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "selected": best_summary,
        "train_query_count": len(train_examples),
        "dev_query_count": len(dev_examples),
        "test_query_count": len(test_examples),
        "model_out": str(model_out),
        "metrics_out": str(metrics_out),
        "scores_out": str(scores_out),
    }
    write_json(metrics_out, summary)
    return summary


def _example_tensors(example: Any, normalizer: dict[str, list[float]], *, device: str) -> tuple[Any, Any, Any, Any]:
    import torch

    mean = np.array(normalizer["mean"], dtype=np.float64)
    scale = np.array(normalizer["scale"], dtype=np.float64)
    query = (example.query_vector - mean) / scale
    candidates = (example.candidate_vectors - mean) / scale
    return (
        torch.tensor(query, dtype=torch.float32, device=device),
        torch.tensor(candidates, dtype=torch.float32, device=device),
        torch.tensor(example.labels, dtype=torch.long, device=device),
        torch.tensor(example.dense_scores, dtype=torch.float32, device=device),
    )


def _torch_zscore(values: Any) -> Any:
    std = values.std(unbiased=False)
    if float(std.detach().cpu()) < 1e-8:
        return values * 0.0
    return (values - values.mean()) / std


def _forward_with_training_dropout(model: Any, query: Any, *, dropout: float) -> Any:
    import torch.nn.functional as F

    if dropout <= 0.0:
        return model(query)
    x = model[0](query)
    x = F.relu(x)
    x = F.dropout(x, p=dropout, training=True)
    x = model[2](x)
    x = F.relu(x)
    x = F.dropout(x, p=dropout, training=True)
    return model[4](x)


def _smoothed_target(labels: Any, *, label_smoothing: float) -> Any:
    target = labels.float()
    target = target / target.sum()
    if label_smoothing <= 0.0:
        return target
    uniform = target.new_full(target.shape, 1.0 / max(1, target.numel()))
    return ((1.0 - label_smoothing) * target) + (label_smoothing * uniform)


def _write_scores(path: Path, scores: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (query_id, chunk_id), score in sorted(scores.items()):
            handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
