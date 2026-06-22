#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from activation_rag.supervised_reranking import load_jsonl
from scripts.build_core245_permutation_groups import _load_cache_rows, _load_manifest, _row_vector


PRIMITIVES = ("doc_value", "query_doc_product", "negative_absdiff", "shared_min")
CATEGORY_NAMES = (
    "entity_domain",
    "event_action",
    "quantity_math_code",
    "relation_discourse",
    "state_affect",
    "task_instruction",
    "uncategorized_content",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose named Core245 SAE feature signal by benchmark split.")
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="Dataset spec: name:split:groups_jsonl:telemetry_cache_dir. Repeat for train/dev/test.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--max-train-groups", type=int, default=2000)
    args = parser.parse_args()

    manifest = _load_manifest(Path(args.feature_manifest))
    feature_ids = manifest["feature_ids"]
    feature_meta = manifest["feature_meta"]
    specs = [_parse_dataset_spec(raw) for raw in args.dataset]
    cache_dirs = sorted({spec["telemetry_cache_dir"] for spec in specs})
    rows_by_cache = {cache_dir: _load_cache_rows(Path(cache_dir)) for cache_dir in cache_dirs}

    split_reports: list[dict[str, Any]] = []
    for spec in specs:
        groups = load_jsonl(spec["groups_path"])
        if spec["split"] == "train" and args.max_train_groups > 0 and len(groups) > args.max_train_groups:
            groups = groups[: args.max_train_groups]
        split_reports.append(
            _diagnose_split(
                dataset_name=spec["name"],
                split=spec["split"],
                groups=groups,
                rows_by_chunk=rows_by_cache[spec["telemetry_cache_dir"]],
                feature_ids=feature_ids,
                feature_meta=feature_meta,
                top_k=args.top_k,
            )
        )

    stability = _stability_report(split_reports, feature_ids, feature_meta, top_k=args.top_k)
    output = {
        "schema_version": "activation_rag.core245_feature_signal_diagnostics.v1",
        "feature_manifest": str(Path(args.feature_manifest)),
        "feature_set_id": manifest["feature_set_id"],
        "source_feature_count": len(feature_ids),
        "primitives": list(PRIMITIVES),
        "train_group_cap": args.max_train_groups,
        "splits": split_reports,
        "stability": stability,
        "interpretation_notes": [
            "AUC is computed within query groups over positive-vs-negative candidate pairs.",
            "Feature diagnostics use labels for explanation only; do not use heldout test rankings here for model selection.",
            "doc_value reflects generic candidate activation mass, query_doc_product and shared_min test query-document co-activation, and negative_absdiff tests direct similarity.",
            "A feature is marked stable_useful only when its best primitive beats 0.53 AUC on both train/dev and heldout test for the same dataset.",
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_printable_summary(output), indent=2, sort_keys=True))


def _parse_dataset_spec(raw: str) -> dict[str, str]:
    parts = raw.split(":", 3)
    if len(parts) != 4:
        raise SystemExit(f"invalid --dataset spec {raw!r}; expected name:split:groups_jsonl:telemetry_cache_dir")
    name, split, groups_path, telemetry_cache_dir = parts
    return {
        "name": name,
        "split": split,
        "groups_path": groups_path,
        "telemetry_cache_dir": telemetry_cache_dir,
    }


def _diagnose_split(
    *,
    dataset_name: str,
    split: str,
    groups: list[dict[str, Any]],
    rows_by_chunk: dict[str, dict[str, Any]],
    feature_ids: list[str],
    feature_meta: dict[str, dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    feature_count = len(feature_ids)
    primitive_wins = {name: np.zeros(feature_count, dtype=np.float64) for name in PRIMITIVES}
    primitive_pairs = {name: np.zeros(feature_count, dtype=np.float64) for name in PRIMITIVES}
    category_wins = {name: np.zeros(len(CATEGORY_NAMES), dtype=np.float64) for name in PRIMITIVES}
    category_pairs = {name: np.zeros(len(CATEGORY_NAMES), dtype=np.float64) for name in PRIMITIVES}
    pos_nonzero = np.zeros(feature_count, dtype=np.float64)
    neg_nonzero = np.zeros(feature_count, dtype=np.float64)
    pos_count = np.zeros(feature_count, dtype=np.float64)
    neg_count = np.zeros(feature_count, dtype=np.float64)
    coverage = Counter()
    changed_groups = 0

    category_indices = _category_indices(feature_ids, feature_meta)

    for group in groups:
        query_row = rows_by_chunk.get(str(group.get("query_activation_chunk_id") or ""))
        if query_row is None:
            coverage["missing_query_telemetry"] += 1
            continue
        query = _row_vector(query_row, feature_ids)
        candidates: list[dict[str, Any]] = []
        docs: list[np.ndarray] = []
        labels: list[int] = []
        for candidate in group.get("candidates", []):
            row = rows_by_chunk.get(str(candidate.get("chunk_id") or ""))
            if row is None:
                coverage["missing_candidate_telemetry"] += 1
                continue
            candidates.append(candidate)
            docs.append(_row_vector(row, feature_ids))
            labels.append(1 if int(candidate.get("label", 0)) > 0 else 0)
        if not docs:
            coverage["empty_after_telemetry_join"] += 1
            continue
        if not any(labels):
            coverage["no_positive_in_group"] += 1
            continue
        if all(labels):
            coverage["no_negative_in_group"] += 1
            continue
        changed_groups += 1
        doc_matrix = np.vstack(docs)
        label_array = np.array(labels, dtype=np.int64)
        positive_mask = label_array > 0
        negative_mask = ~positive_mask
        pos_values = doc_matrix[positive_mask]
        neg_values = doc_matrix[negative_mask]
        pos_nonzero += (np.abs(pos_values) > 0.0).sum(axis=0)
        neg_nonzero += (np.abs(neg_values) > 0.0).sum(axis=0)
        pos_count += float(pos_values.shape[0])
        neg_count += float(neg_values.shape[0])

        feature_scores = _primitive_feature_scores(query, doc_matrix)
        for primitive_name, scores in feature_scores.items():
            wins, pairs = _pairwise_auc_counts(scores, positive_mask)
            primitive_wins[primitive_name] += wins
            primitive_pairs[primitive_name] += pairs

        query_categories = _category_values(query, category_indices)
        doc_categories = np.vstack([_category_values(row, category_indices) for row in doc_matrix])
        category_scores = _primitive_feature_scores(query_categories, doc_categories)
        for primitive_name, scores in category_scores.items():
            wins, pairs = _pairwise_auc_counts(scores, positive_mask)
            category_wins[primitive_name] += wins
            category_pairs[primitive_name] += pairs

    feature_rows = []
    for index, feature_id in enumerate(feature_ids):
        primitive_aucs = {
            name: _safe_auc(primitive_wins[name][index], primitive_pairs[name][index])
            for name in PRIMITIVES
        }
        best_name, best_auc = max(primitive_aucs.items(), key=lambda item: abs(item[1] - 0.5))
        meta = feature_meta[feature_id]
        feature_rows.append(
            {
                "feature_id": feature_id,
                "label": meta.get("label", feature_id),
                "categories": list(meta.get("categories") or []),
                "causal_effect": float(meta.get("causal_effect") or 0.0),
                "primitive_aucs": primitive_aucs,
                "best_primitive": best_name,
                "best_auc": best_auc,
                "directional_margin": best_auc - 0.5,
                "positive_nonzero_rate": _safe_rate(pos_nonzero[index], pos_count[index]),
                "negative_nonzero_rate": _safe_rate(neg_nonzero[index], neg_count[index]),
            }
        )
    category_rows = []
    for index, category_name in enumerate(CATEGORY_NAMES):
        primitive_aucs = {
            name: _safe_auc(category_wins[name][index], category_pairs[name][index])
            for name in PRIMITIVES
        }
        best_name, best_auc = max(primitive_aucs.items(), key=lambda item: abs(item[1] - 0.5))
        category_rows.append(
            {
                "category": category_name,
                "feature_count": len(category_indices[category_name]),
                "primitive_aucs": primitive_aucs,
                "best_primitive": best_name,
                "best_auc": best_auc,
                "directional_margin": best_auc - 0.5,
            }
        )

    top_positive = sorted(feature_rows, key=lambda row: row["directional_margin"], reverse=True)[:top_k]
    top_negative = sorted(feature_rows, key=lambda row: row["directional_margin"])[:top_k]
    most_noisy = sorted(feature_rows, key=lambda row: abs(row["directional_margin"]))[:top_k]
    high_mass_noisy = sorted(
        [
            row
            for row in feature_rows
            if abs(row["directional_margin"]) < 0.015
            and max(row["positive_nonzero_rate"], row["negative_nonzero_rate"]) >= 0.15
        ],
        key=lambda row: max(row["positive_nonzero_rate"], row["negative_nonzero_rate"]),
        reverse=True,
    )[:top_k]

    return {
        "dataset": dataset_name,
        "split": split,
        "group_count": len(groups),
        "diagnostic_group_count": changed_groups,
        "coverage": dict(coverage),
        "category_summary": sorted(category_rows, key=lambda row: row["directional_margin"], reverse=True),
        "top_positive_features": top_positive,
        "top_negative_features": top_negative,
        "most_noisy_features": most_noisy,
        "high_mass_noisy_features": high_mass_noisy,
        "all_features": feature_rows,
    }


def _primitive_feature_scores(query: np.ndarray, docs: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "doc_value": docs,
        "query_doc_product": docs * query[None, :],
        "negative_absdiff": -np.abs(docs - query[None, :]),
        "shared_min": np.minimum(docs, query[None, :]),
    }


def _pairwise_auc_counts(scores: np.ndarray, positive_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positives = scores[positive_mask]
    negatives = scores[~positive_mask]
    wins = np.zeros(scores.shape[1], dtype=np.float64)
    pairs = np.zeros(scores.shape[1], dtype=np.float64)
    if positives.size == 0 or negatives.size == 0:
        return wins, pairs
    for positive in positives:
        wins += (positive[None, :] > negatives).sum(axis=0)
        wins += 0.5 * (positive[None, :] == negatives).sum(axis=0)
        pairs += negatives.shape[0]
    return wins, pairs


def _category_indices(feature_ids: list[str], feature_meta: dict[str, dict[str, Any]]) -> dict[str, list[int]]:
    out = {name: [] for name in CATEGORY_NAMES}
    for index, feature_id in enumerate(feature_ids):
        categories = feature_meta[feature_id].get("categories") or ["uncategorized_content"]
        for category in categories:
            if category in out:
                out[category].append(index)
    return out


def _category_values(values: np.ndarray, category_indices: dict[str, list[int]]) -> np.ndarray:
    out = np.zeros(len(CATEGORY_NAMES), dtype=np.float64)
    positive = np.maximum(values, 0.0)
    for index, category in enumerate(CATEGORY_NAMES):
        indices = category_indices[category]
        if indices:
            out[index] = float(positive[indices].sum())
    return out


def _stability_report(
    split_reports: list[dict[str, Any]],
    feature_ids: list[str],
    feature_meta: dict[str, dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for report in split_reports:
        by_dataset[report["dataset"]][report["split"]] = report
    stability = []
    for dataset, splits in sorted(by_dataset.items()):
        train = _feature_map(splits.get("train"))
        dev = _feature_map(splits.get("dev"))
        test = _feature_map(splits.get("test"))
        rows = []
        for feature_id in feature_ids:
            split_values = {
                split_name: split_map.get(feature_id, {}).get("directional_margin")
                for split_name, split_map in (("train", train), ("dev", dev), ("test", test))
            }
            available = [value for value in split_values.values() if value is not None]
            if not available:
                continue
            stable_positive = all((split_values.get(name) or 0.0) >= 0.03 for name in ("train", "dev", "test") if name in splits)
            unstable = (
                split_values.get("train") is not None
                and split_values.get("test") is not None
                and split_values["train"] * split_values["test"] < 0.0
            )
            meta = feature_meta[feature_id]
            rows.append(
                {
                    "feature_id": feature_id,
                    "label": meta.get("label", feature_id),
                    "categories": list(meta.get("categories") or []),
                    "split_margins": split_values,
                    "mean_margin": float(sum(available) / len(available)),
                    "min_margin": float(min(available)),
                    "stable_useful": stable_positive,
                    "train_test_sign_flip": unstable,
                }
            )
        stability.append(
            {
                "dataset": dataset,
                "stable_useful_features": sorted([row for row in rows if row["stable_useful"]], key=lambda row: row["min_margin"], reverse=True)[:top_k],
                "train_test_sign_flip_features": sorted([row for row in rows if row["train_test_sign_flip"]], key=lambda row: abs(row["mean_margin"]), reverse=True)[:top_k],
                "recommended_label_subsets": _recommended_label_subsets(dataset, rows, top_k=top_k),
            }
        )
    return stability


def _feature_map(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    return {str(row["feature_id"]): row for row in report.get("all_features", [])}


def _recommended_label_subsets(dataset: str, rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    useful = [row for row in rows if row.get("stable_useful")]
    if not useful:
        useful = sorted(rows, key=lambda row: row["mean_margin"], reverse=True)[:top_k]
    category_counts = Counter(cat for row in useful for cat in row.get("categories", []))
    label_terms = Counter()
    for row in useful:
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", str(row.get("label", "")).lower()):
            if token not in {"context", "tokens", "segment", "initial", "reference"}:
                label_terms[token] += 1
    return {
        "candidate_feature_ids": [row["feature_id"] for row in useful[:top_k]],
        "candidate_feature_labels": [row["label"] for row in useful[:top_k]],
        "dominant_categories": category_counts.most_common(),
        "repeated_label_terms": label_terms.most_common(16),
        "human_readable_hypothesis": _subset_hypothesis(dataset, category_counts, label_terms),
    }


def _subset_hypothesis(dataset: str, category_counts: Counter[str], label_terms: Counter[str]) -> str:
    categories = ", ".join(name for name, _ in category_counts.most_common(3)) or "none"
    terms = ", ".join(name for name, _ in label_terms.most_common(6)) or "none"
    return f"{dataset}: prioritize categories [{categories}] and label terms [{terms}] before retraining broader Core245 models."


def _safe_auc(wins: float, pairs: float) -> float:
    if pairs <= 0:
        return 0.5
    value = wins / pairs
    if not math.isfinite(value):
        return 0.5
    return float(value)


def _safe_rate(count: float, total: float) -> float:
    return float(count / total) if total > 0 else 0.0


def _printable_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "out": payload.get("out"),
        "feature_set_id": payload["feature_set_id"],
        "source_feature_count": payload["source_feature_count"],
        "splits": [
            {
                "dataset": split["dataset"],
                "split": split["split"],
                "group_count": split["group_count"],
                "diagnostic_group_count": split["diagnostic_group_count"],
                "top_positive": [
                    {
                        "feature_id": row["feature_id"],
                        "label": row["label"],
                        "margin": round(row["directional_margin"], 4),
                        "primitive": row["best_primitive"],
                    }
                    for row in split["top_positive_features"][:8]
                ],
                "top_categories": [
                    {
                        "category": row["category"],
                        "margin": round(row["directional_margin"], 4),
                        "primitive": row["best_primitive"],
                    }
                    for row in split["category_summary"][:5]
                ],
            }
            for split in payload["splits"]
        ],
        "stability": payload["stability"],
    }


if __name__ == "__main__":
    main()
