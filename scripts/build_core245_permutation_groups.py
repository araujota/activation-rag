#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from activation_rag.supervised_reranking import load_jsonl


VARIANTS = (
    "qwen_l07_core245_raw_max",
    "qwen_l07_core245_log1p_l2",
    "qwen_l07_core245_causal_weighted_category",
    "qwen_l07_core245_high_effect_topk",
    "qwen_l07_core245_df_filtered",
    "qwen_l07_core245_counterfactual_matched",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supervised reranker groups for longmem core245 SAE permutations.")
    parser.add_argument("--groups", required=True, help="Dense candidate groups JSONL.")
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top-effect-k", type=int, default=64)
    parser.add_argument("--df-min-fraction", type=float, default=0.0)
    parser.add_argument("--df-max-fraction", type=float, default=0.80)
    parser.add_argument("--counterfactual-seed", type=int, default=13)
    args = parser.parse_args()

    result = build_permutation_groups(
        groups_path=Path(args.groups),
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        feature_manifest_path=Path(args.feature_manifest),
        variant=args.variant,
        out_path=Path(args.out),
        top_effect_k=args.top_effect_k,
        df_min_fraction=args.df_min_fraction,
        df_max_fraction=args.df_max_fraction,
        counterfactual_seed=args.counterfactual_seed,
    )
    printable = dict(result["summary"])
    printable["out"] = str(args.out)
    print(json.dumps(printable, indent=2, sort_keys=True))


def build_permutation_groups(
    *,
    groups_path: Path,
    telemetry_cache_dir: Path,
    feature_manifest_path: Path,
    variant: str,
    out_path: Path,
    top_effect_k: int,
    df_min_fraction: float,
    df_max_fraction: float,
    counterfactual_seed: int,
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown core245 permutation: {variant}")
    groups = load_jsonl(groups_path)
    rows_by_chunk = _load_cache_rows(telemetry_cache_dir)
    manifest = _load_manifest(feature_manifest_path)
    feature_ids = manifest["feature_ids"]
    feature_meta = manifest["feature_meta"]
    doc_chunk_ids = _candidate_chunk_ids(groups, rows_by_chunk)
    doc_matrix = _matrix(rows_by_chunk, doc_chunk_ids, feature_ids)
    df_fractions = _document_frequency_fractions(doc_matrix)
    selected_ids, selected_matrix = _select_variant_matrix(
        variant=variant,
        feature_ids=feature_ids,
        feature_meta=feature_meta,
        doc_matrix=doc_matrix,
        df_fractions=df_fractions,
        top_effect_k=top_effect_k,
        df_min_fraction=df_min_fraction,
        df_max_fraction=df_max_fraction,
    )
    transformed_groups = _transform_groups(
        groups=groups,
        rows_by_chunk=rows_by_chunk,
        feature_ids=feature_ids,
        selected_ids=selected_ids,
        selected_doc_chunk_ids=doc_chunk_ids,
        selected_doc_matrix=selected_matrix,
        feature_meta=feature_meta,
        variant=variant,
        counterfactual_seed=counterfactual_seed,
    )
    summary = {
        "schema_version": "activation_rag.core245_permutation_groups.summary.v1",
        "groups": str(groups_path),
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "feature_manifest": str(feature_manifest_path),
        "variant": variant,
        "input_group_count": len(groups),
        "output_group_count": len(transformed_groups),
        "feature_set_id": manifest["feature_set_id"],
        "source_feature_count": len(feature_ids),
        "selected_feature_count": len(selected_ids),
        "top_effect_k": top_effect_k,
        "df_min_fraction": df_min_fraction,
        "df_max_fraction": df_max_fraction,
        "counterfactual_seed": counterfactual_seed,
    }
    _write_jsonl(out_path, transformed_groups)
    return {"summary": summary, "groups": transformed_groups}


def _load_cache_rows(cache_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("telemetry_valid", True) and not row.get("invalid_reason") and row.get("sae_feature_values"):
            rows[str(row["chunk_id"])] = row
    return rows


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("features") or []
    ids = [str(row["feature_id"]) for row in rows if row.get("feature_id") is not None]
    if not ids:
        ids = [str(value) for value in payload.get("sae_feature_ids") or []]
    meta = {
        str(row.get("feature_id")): {
            "categories": [str(item) for item in row.get("categories") or []],
            "causal_effect": float(row.get("causal_effect") or 0.0),
            "label": str(row.get("label") or row.get("short_label") or row.get("feature_id")),
        }
        for row in rows
        if row.get("feature_id") is not None
    }
    for feature_id in ids:
        meta.setdefault(feature_id, {"categories": ["uncategorized"], "causal_effect": 0.0, "label": feature_id})
    return {
        "feature_set_id": str(payload.get("feature_set_id") or ""),
        "feature_ids": ids,
        "feature_meta": meta,
    }


def _candidate_chunk_ids(groups: list[dict[str, Any]], rows_by_chunk: dict[str, dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for group in groups:
        for candidate in group.get("candidates", []):
            chunk_id = str(candidate.get("chunk_id"))
            if chunk_id in rows_by_chunk and chunk_id not in seen:
                seen.add(chunk_id)
                out.append(chunk_id)
    return out


def _matrix(rows_by_chunk: dict[str, dict[str, Any]], chunk_ids: list[str], feature_ids: list[str]) -> np.ndarray:
    matrix = np.zeros((len(chunk_ids), len(feature_ids)), dtype=np.float64)
    for row_index, chunk_id in enumerate(chunk_ids):
        values = rows_by_chunk[chunk_id].get("sae_feature_values") or {}
        for col_index, feature_id in enumerate(feature_ids):
            matrix[row_index, col_index] = float(values.get(feature_id, values.get(f"sae.feature.{feature_id}", 0.0)))
    return matrix


def _document_frequency_fractions(doc_matrix: np.ndarray) -> np.ndarray:
    if doc_matrix.shape[0] == 0:
        return np.zeros(doc_matrix.shape[1], dtype=np.float64)
    return (np.abs(doc_matrix) > 0.0).sum(axis=0) / float(doc_matrix.shape[0])


def _select_variant_matrix(
    *,
    variant: str,
    feature_ids: list[str],
    feature_meta: dict[str, dict[str, Any]],
    doc_matrix: np.ndarray,
    df_fractions: np.ndarray,
    top_effect_k: int,
    df_min_fraction: float,
    df_max_fraction: float,
) -> tuple[list[str], np.ndarray]:
    indices = np.arange(len(feature_ids))
    if variant == "qwen_l07_core245_high_effect_topk":
        effects = np.array([abs(float(feature_meta[feature_id].get("causal_effect") or 0.0)) for feature_id in feature_ids])
        keep = set(np.argsort(-effects)[: max(1, min(top_effect_k, len(feature_ids)))].tolist())
        indices = np.array([index for index in indices if int(index) in keep], dtype=np.int64)
    elif variant == "qwen_l07_core245_df_filtered":
        indices = np.array(
            [
                index
                for index, fraction in enumerate(df_fractions)
                if df_min_fraction <= float(fraction) <= df_max_fraction
            ],
            dtype=np.int64,
        )
    selected = [feature_ids[int(index)] for index in indices]
    matrix = doc_matrix[:, indices] if len(indices) else np.zeros((doc_matrix.shape[0], 0), dtype=np.float64)
    if variant == "qwen_l07_core245_log1p_l2":
        matrix = _l2_normalize(_signed_log1p(matrix))
    return selected, matrix


def _transform_groups(
    *,
    groups: list[dict[str, Any]],
    rows_by_chunk: dict[str, dict[str, Any]],
    feature_ids: list[str],
    selected_ids: list[str],
    selected_doc_chunk_ids: list[str],
    selected_doc_matrix: np.ndarray,
    feature_meta: dict[str, dict[str, Any]],
    variant: str,
    counterfactual_seed: int,
) -> list[dict[str, Any]]:
    selected_index = {feature_id: index for index, feature_id in enumerate(selected_ids)}
    doc_index = {chunk_id: index for index, chunk_id in enumerate(selected_doc_chunk_ids)}
    category_groups = _category_groups(selected_ids, feature_meta)
    counterfactual_groups = _matched_counterfactual_groups(selected_ids, category_groups, seed=counterfactual_seed)
    matcher_index = _build_matcher_index(selected_doc_matrix, selected_doc_chunk_ids)
    transformed: list[dict[str, Any]] = []
    for group in groups:
        query_chunk_id = str(group.get("query_activation_chunk_id") or "")
        query_row = rows_by_chunk.get(query_chunk_id)
        if query_row is None:
            continue
        query_vector = _row_vector(query_row, feature_ids)
        query_selected = np.array([query_vector[feature_ids.index(feature_id)] for feature_id in selected_ids], dtype=np.float64)
        if variant == "qwen_l07_core245_log1p_l2":
            query_selected = _l2_normalize(_signed_log1p(query_selected[None, :]))[0]
        matcher_scores = _matcher_scores(query_selected, matcher_index)
        new_candidates: list[dict[str, Any]] = []
        for candidate in group.get("candidates", []):
            chunk_id = str(candidate.get("chunk_id"))
            row = rows_by_chunk.get(chunk_id)
            if row is None or chunk_id not in doc_index:
                continue
            doc_selected = selected_doc_matrix[doc_index[chunk_id]]
            features = _base_dense_features(candidate)
            if variant == "qwen_l07_core245_causal_weighted_category":
                features.update(_group_features("core245_category", query_selected, doc_selected, selected_index, category_groups, feature_meta))
            elif variant == "qwen_l07_core245_counterfactual_matched":
                features.update(_group_features("core245_counterfactual", query_selected, doc_selected, selected_index, counterfactual_groups, feature_meta))
            else:
                prefix = _value_prefix(variant)
                features.update(_vector_interaction_features(prefix, selected_ids, query_selected, doc_selected))
            features.update({key: scores.get(chunk_id, 0.0) for key, scores in matcher_scores.items()})
            updated = dict(candidate)
            updated["features"] = features
            new_candidates.append(updated)
        if new_candidates:
            _add_group_zscores(new_candidates, "core245_matcher:cosine", "core245_matcher:cosine_z")
            updated_group = dict(group)
            updated_group["candidates"] = new_candidates
            updated_group["activation_permutation"] = {
                "schema_version": "activation_rag.core245_permutation_group.v1",
                "variant": variant,
                "selected_feature_count": len(selected_ids),
                "counterfactual_seed": counterfactual_seed if variant == "qwen_l07_core245_counterfactual_matched" else None,
            }
            transformed.append(updated_group)
    return transformed


def _row_vector(row: dict[str, Any], feature_ids: list[str]) -> np.ndarray:
    values = row.get("sae_feature_values") or {}
    return np.array([float(values.get(feature_id, values.get(f"sae.feature.{feature_id}", 0.0))) for feature_id in feature_ids], dtype=np.float64)


def _base_dense_features(candidate: dict[str, Any]) -> dict[str, float]:
    source = candidate.get("features") or {}
    dense_rank = float(candidate.get("dense_rank") or 0.0)
    return {
        "dense_score": float(candidate.get("dense_score", source.get("dense_score", 0.0))),
        "dense_rank_reciprocal": float(source.get("dense_rank_reciprocal", 1.0 / dense_rank if dense_rank > 0 else 0.0)),
    }


def _value_prefix(variant: str) -> str:
    return {
        "qwen_l07_core245_raw_max": "core245",
        "qwen_l07_core245_log1p_l2": "core245_log1p_l2",
        "qwen_l07_core245_high_effect_topk": "core245_high_effect",
        "qwen_l07_core245_df_filtered": "core245_df",
    }[variant]


def _vector_interaction_features(prefix: str, feature_ids: list[str], query: np.ndarray, doc: np.ndarray) -> dict[str, float]:
    diff = query - doc
    features = {
        f"{prefix}:cosine": _cosine(query, doc),
        f"{prefix}:l2_distance": float(np.linalg.norm(diff)),
        f"{prefix}:abs_diff_mean": float(np.abs(diff).mean()) if diff.size else 0.0,
        f"{prefix}:product_mean": float((query * doc).mean()) if diff.size else 0.0,
    }
    for index, feature_id in enumerate(feature_ids):
        q = float(query[index])
        d = float(doc[index])
        features[f"{prefix}:{feature_id}:product"] = q * d
    return features


def _category_groups(feature_ids: list[str], feature_meta: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for feature_id in feature_ids:
        categories = feature_meta[feature_id].get("categories") or ["uncategorized"]
        for category in categories:
            groups.setdefault(_slug(str(category)), []).append(feature_id)
    return {key: sorted(value) for key, value in sorted(groups.items())}


def _matched_counterfactual_groups(feature_ids: list[str], category_groups: dict[str, list[str]], *, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    out: dict[str, list[str]] = {}
    for category, members in category_groups.items():
        width = min(len(members), len(feature_ids))
        out[category] = sorted(rng.sample(feature_ids, width)) if width else []
    return out


def _group_features(
    prefix: str,
    query: np.ndarray,
    doc: np.ndarray,
    selected_index: dict[str, int],
    groups: dict[str, list[str]],
    feature_meta: dict[str, dict[str, Any]],
) -> dict[str, float]:
    features: dict[str, float] = {}
    for group_name, feature_ids in groups.items():
        indices = [selected_index[feature_id] for feature_id in feature_ids if feature_id in selected_index]
        if not indices:
            continue
        q = query[indices]
        d = doc[indices]
        effects = np.array([abs(float(feature_meta[feature_id].get("causal_effect") or 0.0)) for feature_id in feature_ids if feature_id in selected_index], dtype=np.float64)
        if effects.size == 0:
            effects = np.ones_like(q)
        features[f"{prefix}:{group_name}:cosine"] = _cosine(q, d)
        features[f"{prefix}:{group_name}:l2_distance"] = float(np.linalg.norm(q - d))
        features[f"{prefix}:{group_name}:abs_diff_mean"] = float(np.abs(q - d).mean())
        features[f"{prefix}:{group_name}:product_mean"] = float((q * d).mean())
        features[f"{prefix}:{group_name}:query_mass"] = float(np.abs(q).sum())
        features[f"{prefix}:{group_name}:doc_mass"] = float(np.abs(d).sum())
        features[f"{prefix}:{group_name}:weighted_product"] = float((q * d * effects).sum() / max(float(effects.sum()), 1e-12))
    return features


def _build_matcher_index(docs: np.ndarray, doc_chunk_ids: list[str]) -> dict[str, Any]:
    if docs.size == 0:
        return {"doc_chunk_ids": doc_chunk_ids, "empty": True}
    doc_cosine = _l2_normalize(docs)
    doc_doc_sim = doc_cosine @ doc_cosine.T
    np.fill_diagonal(doc_doc_sim, -np.inf)
    local_k = min(10, max(1, docs.shape[0] - 1))
    doc_csls_radius = np.partition(doc_doc_sim, -local_k, axis=1)[:, -local_k:].mean(axis=1)
    doc_doc_distance = 1.0 - doc_doc_sim
    np.fill_diagonal(doc_doc_distance, np.inf)
    doc_nicdm_radius = np.partition(doc_doc_distance, local_k - 1, axis=1)[:, :local_k].mean(axis=1)
    top_pc = _fit_top_pc_transform(docs, remove_components=3)
    whiten = _fit_whiten_transform(docs, dimensions=min(64, docs.shape[1]))
    return {
        "doc_chunk_ids": doc_chunk_ids,
        "empty": False,
        "docs": docs,
        "doc_cosine": doc_cosine,
        "local_k": local_k,
        "doc_csls_radius": doc_csls_radius,
        "doc_nicdm_radius": doc_nicdm_radius,
        "top_pc": top_pc,
        "top_pc_docs": _apply_top_pc_transform(docs, top_pc),
        "whiten": whiten,
        "whiten_docs": _apply_linear_transform(docs, whiten),
    }


def _matcher_scores(query: np.ndarray, matcher_index: dict[str, Any]) -> dict[str, dict[str, float]]:
    doc_chunk_ids = matcher_index["doc_chunk_ids"]
    if matcher_index.get("empty") or query.size == 0:
        return {name: {chunk_id: 0.0 for chunk_id in doc_chunk_ids} for name in _matcher_names()}
    query_row = query[None, :]
    raw = _cosine_scores(query_row, matcher_index["docs"])
    local_k = min(max(1, matcher_index["local_k"]), raw.shape[1])
    query_csls_radius = np.partition(raw, -local_k, axis=1)[:, -local_k:].mean(axis=1)
    csls = (2.0 * raw) - query_csls_radius[:, None] - matcher_index["doc_csls_radius"][None, :]
    distances = 1.0 - raw
    query_nicdm_radius = np.partition(distances, local_k - 1, axis=1)[:, :local_k].mean(axis=1)
    nicdm = -(distances / np.sqrt(np.maximum(query_nicdm_radius[:, None] * matcher_index["doc_nicdm_radius"][None, :], 1e-12)))
    top_pc_query = _apply_top_pc_transform(query_row, matcher_index["top_pc"])
    whiten_query = _apply_linear_transform(query_row, matcher_index["whiten"])
    matrices = {
        "core245_matcher:cosine": raw,
        "core245_matcher:csls": csls,
        "core245_matcher:nicdm": nicdm,
        "core245_matcher:top_pc_removed": _cosine_scores(top_pc_query, matcher_index["top_pc_docs"]),
        "core245_matcher:whiten_l2": _cosine_scores(whiten_query, matcher_index["whiten_docs"]),
    }
    return {
        name: {chunk_id: float(matrix[0, index]) for index, chunk_id in enumerate(doc_chunk_ids)}
        for name, matrix in matrices.items()
    }


def _matcher_names() -> tuple[str, ...]:
    return (
        "core245_matcher:cosine",
        "core245_matcher:csls",
        "core245_matcher:nicdm",
        "core245_matcher:top_pc_removed",
        "core245_matcher:whiten_l2",
    )


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return 0.0
    return float(a @ b / denominator)


def _cosine_scores(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    return _l2_normalize(queries) @ _l2_normalize(docs).T


def _csls_scores(queries: np.ndarray, docs: np.ndarray, *, k: int) -> np.ndarray:
    sim = _cosine_scores(queries, docs)
    query_k = min(max(1, k), sim.shape[1])
    query_radius = np.partition(sim, -query_k, axis=1)[:, -query_k:].mean(axis=1)
    doc_doc = _cosine_scores(docs, docs)
    np.fill_diagonal(doc_doc, -np.inf)
    doc_k = min(max(1, k), max(1, docs.shape[0] - 1))
    doc_radius = np.partition(doc_doc, -doc_k, axis=1)[:, -doc_k:].mean(axis=1)
    return (2.0 * sim) - query_radius[:, None] - doc_radius[None, :]


def _nicdm_scores(queries: np.ndarray, docs: np.ndarray, *, k: int) -> np.ndarray:
    sim = _cosine_scores(queries, docs)
    distances = 1.0 - sim
    query_k = min(max(1, k), distances.shape[1])
    query_radius = np.partition(distances, query_k - 1, axis=1)[:, :query_k].mean(axis=1)
    doc_distances = 1.0 - _cosine_scores(docs, docs)
    np.fill_diagonal(doc_distances, np.inf)
    doc_k = min(max(1, k), max(1, docs.shape[0] - 1))
    doc_radius = np.partition(doc_distances, doc_k - 1, axis=1)[:, :doc_k].mean(axis=1)
    denominator = np.sqrt(np.maximum(query_radius[:, None] * doc_radius[None, :], 1e-12))
    return -(distances / denominator)


def _fit_whiten_transform(docs: np.ndarray, *, dimensions: int) -> dict[str, np.ndarray]:
    mean = docs.mean(axis=0, keepdims=True)
    centered_docs = docs - mean
    covariance = np.atleast_2d(np.cov(centered_docs, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    keep = min(dimensions, docs.shape[1], len(order))
    if keep <= 0:
        return {"mean": mean, "components": np.eye(docs.shape[1]), "scale": np.ones(docs.shape[1])}
    order = order[:keep]
    components = eigenvectors[:, order]
    scale = np.sqrt(np.maximum(eigenvalues[order], 1e-8))
    return {"mean": mean, "components": components, "scale": scale}


def _fit_top_pc_transform(docs: np.ndarray, *, remove_components: int) -> dict[str, np.ndarray]:
    mean = docs.mean(axis=0, keepdims=True)
    centered_docs = docs - mean
    remove = min(max(0, remove_components), max(0, docs.shape[1] - 1), docs.shape[0])
    if remove == 0:
        return {"mean": mean, "top": np.zeros((docs.shape[1], 0), dtype=np.float64)}
    _, _, vt = np.linalg.svd(centered_docs, full_matrices=False)
    top = vt[:remove].T
    return {"mean": mean, "top": top}


def _apply_linear_transform(matrix: np.ndarray, transform: dict[str, np.ndarray]) -> np.ndarray:
    centered = matrix - transform["mean"]
    return centered @ transform["components"] / transform["scale"]


def _apply_top_pc_transform(matrix: np.ndarray, transform: dict[str, np.ndarray]) -> np.ndarray:
    centered = matrix - transform["mean"]
    top = transform["top"]
    if top.shape[1] == 0:
        return centered
    return centered - centered @ top @ top.T


def _signed_log1p(matrix: np.ndarray) -> np.ndarray:
    return np.sign(matrix) * np.log1p(np.abs(matrix))


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def _add_group_zscores(candidates: list[dict[str, Any]], source_feature: str, target_feature: str) -> None:
    values = np.array([candidate["features"].get(source_feature, 0.0) for candidate in candidates], dtype=np.float64)
    mean = float(values.mean()) if values.size else 0.0
    std = float(values.std()) if values.size else 1.0
    if std < 1e-8:
        std = 1.0
    for candidate, value in zip(candidates, values, strict=True):
        candidate["features"][target_feature] = float((value - mean) / std)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _slug(value: str) -> str:
    out = []
    for char in value.strip().lower():
        out.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(out).strip("_") or "unknown"


if __name__ == "__main__":
    main()
