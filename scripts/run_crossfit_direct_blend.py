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

from activation_rag.supervised_reranking import load_jsonl, write_json
from scripts.prepare_vertical_reranker_groups import _split_score, _write_jsonl
from scripts.train_direct_blend_activation_searcher import run_training


SCHEMA_VERSION = "activation_rag.crossfit_direct_blend_run.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate out-of-fold direct-blend activation scores and train the final regularized artifact.")
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--dev-groups", required=True)
    parser.add_argument("--test-groups", required=True)
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--representation", default="raw")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--alpha-grid", default="0.05,0.1,0.2,0.3,0.4")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--input-noise-std", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--skip-folds", action="store_true", help="Train only the final model; useful after OOF scores already exist.")
    args = parser.parse_args()

    summary = run_crossfit(
        train_groups_path=Path(args.train_groups),
        dev_groups_path=Path(args.dev_groups),
        test_groups_path=Path(args.test_groups),
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        feature_manifest_path=Path(args.feature_manifest),
        representation=str(args.representation),
        dataset_name=str(args.dataset_name),
        out_dir=Path(args.out_dir),
        folds=int(args.folds),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        temperature=float(args.temperature),
        alpha_grid=tuple(float(item) for item in str(args.alpha_grid).split(",") if item.strip()),
        dropout=float(args.dropout),
        input_noise_std=float(args.input_noise_std),
        label_smoothing=float(args.label_smoothing),
        grad_clip_norm=float(args.grad_clip_norm),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        top_k=int(args.top_k),
        device=str(args.device),
        seed=int(args.seed),
        skip_folds=bool(args.skip_folds),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_crossfit(
    *,
    train_groups_path: Path,
    dev_groups_path: Path,
    test_groups_path: Path,
    telemetry_cache_dir: Path,
    feature_manifest_path: Path,
    representation: str,
    dataset_name: str,
    out_dir: Path,
    folds: int,
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
    skip_folds: bool = False,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    train_rows = load_jsonl(train_groups_path)
    fold_rows = split_rows_into_folds(train_rows, folds=folds, seed=str(seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    oof_scores_path = out_dir / "oof-train-scores.jsonl"
    fold_summaries: list[dict[str, Any]] = []
    if not skip_folds:
        score_paths = []
        for fold_index, heldout_rows in enumerate(fold_rows):
            heldout_ids = {str(row["query_id"]) for row in heldout_rows}
            fold_train_rows = [row for row in train_rows if str(row["query_id"]) not in heldout_ids]
            if not fold_train_rows or not heldout_rows:
                raise ValueError(f"fold {fold_index} has empty train or heldout rows")
            fold_dir = out_dir / "folds" / f"fold-{fold_index:02d}"
            fold_train_path = fold_dir / "train.jsonl"
            fold_heldout_path = fold_dir / "heldout.jsonl"
            _write_jsonl(fold_train_path, fold_train_rows)
            _write_jsonl(fold_heldout_path, heldout_rows)
            fold_summary = run_training(
                train_groups_path=fold_train_path,
                dev_groups_path=dev_groups_path,
                test_groups_path=fold_heldout_path,
                telemetry_cache_dir=telemetry_cache_dir,
                feature_manifest_path=feature_manifest_path,
                representation=representation,
                model_out=fold_dir / "model.pt",
                metrics_out=fold_dir / "metrics.json",
                scores_out=fold_dir / "heldout-scores.jsonl",
                dataset_name=f"{dataset_name}-fold-{fold_index:02d}",
                hidden_dim=hidden_dim,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                temperature=temperature,
                alpha_grid=alpha_grid,
                dropout=dropout,
                input_noise_std=input_noise_std,
                label_smoothing=label_smoothing,
                grad_clip_norm=grad_clip_norm,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                top_k=top_k,
                device=device,
                seed=seed + fold_index,
            )
            score_paths.append(fold_dir / "heldout-scores.jsonl")
            fold_summaries.append(_compact_fold_summary(fold_index, fold_summary, len(fold_train_rows), len(heldout_rows)))
        concat_score_files(score_paths, oof_scores_path)

    final_dir = out_dir / "final"
    final_summary = run_training(
        train_groups_path=train_groups_path,
        dev_groups_path=dev_groups_path,
        test_groups_path=test_groups_path,
        telemetry_cache_dir=telemetry_cache_dir,
        feature_manifest_path=feature_manifest_path,
        representation=representation,
        model_out=final_dir / "model.pt",
        metrics_out=final_dir / "metrics.json",
        scores_out=final_dir / "test-scores.jsonl",
        dataset_name=dataset_name,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        temperature=temperature,
        alpha_grid=alpha_grid,
        dropout=dropout,
        input_noise_std=input_noise_std,
        label_smoothing=label_smoothing,
        grad_clip_norm=grad_clip_norm,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        top_k=top_k,
        device=device,
        seed=seed,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "train_groups": str(train_groups_path),
        "dev_groups": str(dev_groups_path),
        "test_groups": str(test_groups_path),
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "feature_manifest": str(feature_manifest_path),
        "representation": representation,
        "folds": folds,
        "fold_summaries": fold_summaries,
        "oof_scores_out": str(oof_scores_path) if not skip_folds else None,
        "final_metrics": final_summary,
        "regularization": {
            "dropout": dropout,
            "input_noise_std": input_noise_std,
            "label_smoothing": label_smoothing,
            "grad_clip_norm": grad_clip_norm,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
        },
    }
    write_json(out_dir / "crossfit-summary.json", summary)
    return summary


def split_rows_into_folds(rows: list[dict[str, Any]], *, folds: int, seed: str) -> list[list[dict[str, Any]]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    ordered = sorted(rows, key=lambda row: _split_score(str(row["query_id"]), seed))
    out = [[] for _ in range(folds)]
    for index, row in enumerate(ordered):
        out[index % folds].append(row)
    return out


def concat_score_files(score_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    with out_path.open("w", encoding="utf-8") as output:
        for score_path in score_paths:
            for row in load_jsonl(score_path):
                key = (str(row["query_id"]), str(row["chunk_id"]))
                if key in seen:
                    raise ValueError(f"duplicate OOF score key: {key}")
                seen.add(key)
                output.write(json.dumps(row, sort_keys=True) + "\n")


def _compact_fold_summary(fold_index: int, summary: dict[str, Any], train_count: int, heldout_count: int) -> dict[str, Any]:
    selected = summary.get("selected") or {}
    return {
        "fold": fold_index,
        "train_count": train_count,
        "heldout_count": heldout_count,
        "selected_alpha": selected.get("alpha"),
        "best_epoch": selected.get("best_epoch"),
        "best_dev_score": selected.get("best_dev_score"),
        "heldout_metrics": selected.get("test_metrics"),
        "metrics_out": summary.get("metrics_out"),
        "scores_out": summary.get("scores_out"),
    }


if __name__ == "__main__":
    main()
