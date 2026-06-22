#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activation_rag.supervised_reranking import infer_feature_names, load_jsonl, write_json
from scripts.train_activation_mlp_reranker import run_mlp_training


DENSE_FEATURES = ("dense_rank_reciprocal", "dense_score")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic activation feature-family reranker diagnostics.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("listwise_softmax", "pairwise_logistic"), default="listwise_softmax")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--plan-only", action="store_true", help="Only write the variant manifest; do not train.")
    args = parser.parse_args()
    summary = run_sweep(
        train_path=Path(args.train),
        dev_path=Path(args.dev),
        test_path=Path(args.test) if args.test else None,
        out_dir=Path(args.out_dir),
        manifest_out=Path(args.manifest_out),
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        loss_name=args.loss,
        top_k=args.top_k,
        device=args.device,
        seed=args.seed,
        plan_only=args.plan_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_sweep(
    *,
    train_path: Path,
    dev_path: Path,
    test_path: Path | None,
    out_dir: Path,
    manifest_out: Path,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    top_k: int,
    device: str,
    seed: int,
    plan_only: bool,
) -> dict[str, Any]:
    train_groups = load_jsonl(train_path)
    variants = build_feature_variants(train_groups)
    manifest: dict[str, Any] = {
        "schema_version": "activation_rag.semantic_activation_feature_sweep.v1",
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "test_path": str(test_path) if test_path else None,
        "out_dir": str(out_dir),
        "variant_count": len(variants),
        "variants": {name: list(features) for name, features in variants.items()},
        "runs": [],
        "plan_only": plan_only,
    }
    if not plan_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        for variant_name, feature_names in variants.items():
            model_out = out_dir / f"{variant_name}.pt"
            metrics_out = out_dir / f"{variant_name}.metrics.json"
            run_summary = run_mlp_training(
                train_path=train_path,
                dev_path=dev_path,
                test_path=test_path,
                model_out=model_out,
                metrics_out=metrics_out,
                feature_names=feature_names,
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                loss_name=loss_name,
                top_k=top_k,
                device=device,
                seed=seed,
            )
            manifest["runs"].append(
                {
                    "variant_name": variant_name,
                    "feature_count": len(feature_names),
                    "model_out": str(model_out),
                    "metrics_out": str(metrics_out),
                    "best_dev_score": run_summary.get("best_dev_score"),
                }
            )
    write_json(manifest_out, manifest)
    return manifest


def build_feature_variants(groups: list[dict[str, Any]]) -> dict[str, tuple[str, ...]]:
    names = tuple(infer_feature_names(groups))
    dense = tuple(name for name in DENSE_FEATURES if name in names)
    semantic_labels = _prefixed(names, "activation_semantic:semantic_label:")
    categories = _prefixed(names, "activation_semantic:category:")
    causal_confidence = _prefixed(names, "activation_semantic:causal_confidence:")
    polarity = _prefixed(names, "activation_semantic:polarity:")
    validation = _prefixed(names, "activation_semantic:validation_status:")
    counterfactual = _prefixed(names, "activation_semantic:counterfactual:")
    variants: dict[str, tuple[str, ...]] = {}
    if dense:
        variants["dense_only"] = dense
    if semantic_labels:
        variants["semantic_labels_only"] = semantic_labels
        variants["dense_plus_semantic_labels"] = _merge(dense, semantic_labels)
    if categories:
        variants["categories_only"] = categories
        variants["dense_plus_categories"] = _merge(dense, categories)
    if causal_confidence:
        variants["causal_confidence_only"] = causal_confidence
        variants["dense_plus_causal_confidence"] = _merge(dense, causal_confidence)
    if polarity:
        variants["polarity_only"] = polarity
        variants["dense_plus_polarity"] = _merge(dense, polarity)
    if validation:
        variants["validation_status_only"] = validation
        variants["dense_plus_validation_status"] = _merge(dense, validation)
    if counterfactual:
        variants["counterfactual_only"] = counterfactual
        variants["dense_plus_counterfactual"] = _merge(dense, counterfactual)
    all_semantic = tuple(
        name
        for name in names
        if name.startswith("activation_semantic:") and not name.startswith("activation_semantic:counterfactual:")
    )
    if all_semantic:
        variants["dense_plus_all_semantic"] = _merge(dense, all_semantic)
    return variants


def _prefixed(names: tuple[str, ...], prefix: str) -> tuple[str, ...]:
    return tuple(name for name in names if name.startswith(prefix))


def _merge(*groups: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return tuple(out)


if __name__ == "__main__":
    main()
