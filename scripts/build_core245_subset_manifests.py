#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filtered Core245 feature manifests from diagnostic feature-id sets.")
    parser.add_argument("--base-manifest", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weak-threshold", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=32)
    args = parser.parse_args()

    base = json.loads(Path(args.base_manifest).read_text(encoding="utf-8"))
    diagnostics = json.loads(Path(args.diagnostics).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_by_id = {str(row["feature_id"]): row for row in base.get("features", [])}
    written: list[dict[str, Any]] = []
    for stability in diagnostics.get("stability", []):
        dataset = str(stability["dataset"])
        ids = _candidate_ids(
            diagnostics=diagnostics,
            dataset=dataset,
            weak_threshold=float(args.weak_threshold),
            top_n=int(args.top_n),
        )
        if not ids:
            continue
        subset = dict(base)
        subset_features = [feature_by_id[feature_id] for feature_id in ids if feature_id in feature_by_id]
        subset["artifact_id"] = f"{base.get('artifact_id', 'core245')}-{dataset}-diagnostic-subset"
        subset["feature_set_id"] = f"{base.get('feature_set_id', 'core245')}_{dataset}_diagnostic_subset"
        subset["features"] = subset_features
        subset["sae_feature_ids"] = [str(row["feature_id"]) for row in subset_features]
        subset["sae_feature_count"] = len(subset_features)
        subset["sae_feature_labels"] = {str(row["feature_id"]): row.get("label") for row in subset_features}
        subset["sae_feature_categories"] = {str(row["feature_id"]): row.get("categories", []) for row in subset_features}
        retained_order = [name for name in base.get("feature_order", []) if _keep_feature_order_name(name, set(subset["sae_feature_ids"]))]
        subset["feature_order"] = retained_order
        subset["subset_policy"] = {
            "schema_version": "activation_rag.core245_subset_policy.v1",
            "source_diagnostics": str(Path(args.diagnostics)),
            "dataset": dataset,
            "weak_threshold": float(args.weak_threshold),
            "top_n": int(args.top_n),
            "feature_ids": subset["sae_feature_ids"],
        }
        out_path = out_dir / f"{dataset}-diagnostic-subset-top{len(subset_features)}.feature_manifest.json"
        out_path.write_text(json.dumps(subset, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append({"dataset": dataset, "path": str(out_path), "feature_count": len(subset_features), "feature_ids": subset["sae_feature_ids"]})
    summary_path = out_dir / "subset-manifests-summary.json"
    summary_path.write_text(json.dumps({"schema_version": "activation_rag.core245_subset_manifests.v1", "written": written}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "written": written}, indent=2, sort_keys=True))


def _candidate_ids(*, diagnostics: dict[str, Any], dataset: str, weak_threshold: float, top_n: int) -> list[str]:
    split_maps = {
        split["split"]: {str(row["feature_id"]): row for row in split.get("all_features", [])}
        for split in diagnostics.get("splits", [])
        if split.get("dataset") == dataset
    }
    if not split_maps:
        return []
    feature_ids = sorted(set().union(*(set(values) for values in split_maps.values())))
    stable: list[tuple[float, float, str]] = []
    positive_any: list[tuple[float, float, str]] = []
    harmful: set[str] = set()
    for feature_id in feature_ids:
        margins = [float(split_maps[split][feature_id]["directional_margin"]) for split in split_maps if feature_id in split_maps[split]]
        if not margins:
            continue
        if all(margin <= -weak_threshold for margin in margins):
            harmful.add(feature_id)
            continue
        min_margin = min(margins)
        mean_margin = sum(margins) / len(margins)
        if all(margin >= weak_threshold for margin in margins):
            stable.append((min_margin, mean_margin, feature_id))
        if mean_margin > 0.0 and feature_id not in harmful:
            positive_any.append((min_margin, mean_margin, feature_id))
    selected = [feature_id for _, _, feature_id in sorted(stable, reverse=True)]
    for _, _, feature_id in sorted(positive_any, reverse=True):
        if len(selected) >= top_n:
            break
        if feature_id not in selected:
            selected.append(feature_id)
    return selected[:top_n]


def _keep_feature_order_name(name: str, subset_ids: set[str]) -> bool:
    if name.startswith("sae.feature."):
        return name.removeprefix("sae.feature.") in subset_ids
    return not name.startswith("ix.source.")


if __name__ == "__main__":
    main()
