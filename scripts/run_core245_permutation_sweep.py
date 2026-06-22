#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.supervised_reranking import infer_feature_names, load_jsonl, write_json
from scripts.build_core245_permutation_groups import VARIANTS, build_permutation_groups
from scripts.train_activation_mlp_reranker import run_mlp_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ordered longmem core245 supervised reranker ablation sweep.")
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--dev-groups", required=True)
    parser.add_argument("--test-groups", required=True)
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--top-effect-k", type=int, default=64)
    parser.add_argument("--df-min-fraction", type=float, default=0.0)
    parser.add_argument("--df-max-fraction", type=float, default=0.80)
    parser.add_argument("--counterfactual-seed", type=int, default=13)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=("listwise_softmax", "pairwise_logistic"), default="listwise_softmax")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    summary = run_sweep(
        train_groups=Path(args.train_groups),
        dev_groups=Path(args.dev_groups),
        test_groups=Path(args.test_groups),
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        feature_manifest=Path(args.feature_manifest),
        out_dir=Path(args.out_dir),
        manifest_out=Path(args.manifest_out),
        top_effect_k=args.top_effect_k,
        df_min_fraction=args.df_min_fraction,
        df_max_fraction=args.df_max_fraction,
        counterfactual_seed=args.counterfactual_seed,
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
    train_groups: Path,
    dev_groups: Path,
    test_groups: Path,
    telemetry_cache_dir: Path,
    feature_manifest: Path,
    out_dir: Path,
    manifest_out: Path,
    top_effect_k: int,
    df_min_fraction: float,
    df_max_fraction: float,
    counterfactual_seed: int,
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
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": "activation_rag.core245_permutation_sweep.v1",
        "train_groups": str(train_groups),
        "dev_groups": str(dev_groups),
        "test_groups": str(test_groups),
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "feature_manifest": str(feature_manifest),
        "out_dir": str(out_dir),
        "variants": [],
        "plan_only": plan_only,
    }
    for variant in VARIANTS:
        variant_dir = out_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        built_paths = {
            "train": variant_dir / "train.jsonl",
            "dev": variant_dir / "dev.jsonl",
            "test": variant_dir / "test.jsonl",
        }
        build_summaries = {}
        for split_name, source in (("train", train_groups), ("dev", dev_groups), ("test", test_groups)):
            built = build_permutation_groups(
                groups_path=source,
                telemetry_cache_dir=telemetry_cache_dir,
                feature_manifest_path=feature_manifest,
                variant=variant,
                out_path=built_paths[split_name],
                top_effect_k=top_effect_k,
                df_min_fraction=df_min_fraction,
                df_max_fraction=df_max_fraction,
                counterfactual_seed=counterfactual_seed,
            )
            build_summaries[split_name] = built["summary"]
        train_rows = load_jsonl(built_paths["train"])
        feature_names = infer_feature_names(train_rows)
        run_entry: dict[str, Any] = {
            "variant": variant,
            "paths": {key: str(value) for key, value in built_paths.items()},
            "build_summaries": build_summaries,
            "feature_count": len(feature_names),
        }
        if not plan_only:
            metrics_out = variant_dir / "metrics.json"
            model_out = variant_dir / "model.pt"
            run_summary = run_mlp_training(
                train_path=built_paths["train"],
                dev_path=built_paths["dev"],
                test_path=built_paths["test"],
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
            run_entry.update(
                {
                    "model_out": str(model_out),
                    "metrics_out": str(metrics_out),
                    "best_epoch": run_summary.get("best_epoch"),
                    "best_dev_score": run_summary.get("best_dev_score"),
                    "dev_metrics": run_summary.get("dev_metrics"),
                    "test_metrics": run_summary.get("test_metrics"),
                }
            )
        manifest["variants"].append(run_entry)
    write_json(manifest_out, manifest)
    return manifest


if __name__ == "__main__":
    main()
