#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activation_rag.supervised_reranking import (
    evaluate_group_rankings,
    infer_feature_names,
    load_jsonl,
    train_pairwise_linear_ranker,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supervised activation-aware pairwise reranker.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test")
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--feature", action="append", dest="features", help="Feature to use; repeatable. Defaults to all.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    summary = run_training(
        train_path=Path(args.train),
        dev_path=Path(args.dev),
        test_path=Path(args.test) if args.test else None,
        model_out=Path(args.model_out),
        metrics_out=Path(args.metrics_out),
        feature_names=tuple(args.features) if args.features else None,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_training(
    *,
    train_path: Path,
    dev_path: Path,
    test_path: Path | None,
    model_out: Path,
    metrics_out: Path,
    feature_names: tuple[str, ...] | None = None,
    epochs: int,
    learning_rate: float,
    l2: float,
    top_k: int,
) -> dict[str, Any]:
    train_groups = load_jsonl(train_path)
    dev_groups = load_jsonl(dev_path)
    test_groups = load_jsonl(test_path) if test_path else []
    selected_features = feature_names or infer_feature_names(train_groups)
    model = train_pairwise_linear_ranker(
        train_groups,
        feature_names=selected_features,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
    )
    model.save_json(model_out)
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.activation_reranker_training_run.v1",
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "test_path": str(test_path) if test_path else None,
        "model_out": str(model_out),
        "feature_names": list(model.feature_names),
        "train_query_count": len(train_groups),
        "dev_query_count": len(dev_groups),
        "test_query_count": len(test_groups),
        "top_k": top_k,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "train_metrics": evaluate_group_rankings(train_groups, model=model, top_k=top_k),
        "dev_metrics": evaluate_group_rankings(dev_groups, model=model, top_k=top_k),
    }
    if test_groups:
        summary["test_metrics"] = evaluate_group_rankings(test_groups, model=model, top_k=top_k)
    write_json(metrics_out, summary)
    return summary


if __name__ == "__main__":
    main()
