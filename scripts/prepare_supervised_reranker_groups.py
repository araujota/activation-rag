#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from activation_rag.benchmarks import load_beir_dataset
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.feature_catalog import FeatureCatalog, load_feature_catalog
from activation_rag.schema import DocumentRecord, stable_hash


SCHEMA_VERSION = "activation_rag.supervised_reranker_group.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dense-candidate supervised activation reranker groups.")
    parser.add_argument("--beir-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--dense-embedding-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--feature-catalog", help="Optional semantic activation feature catalog JSON/JSONL.")
    args = parser.parse_args()

    groups = prepare_reranker_groups(
        beir_dir=Path(args.beir_dir),
        dataset_name=args.dataset_name,
        split=args.split,
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        dense_embedding_cache=Path(args.dense_embedding_cache),
        candidate_k=args.candidate_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        feature_catalog_path=Path(args.feature_catalog) if args.feature_catalog else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, sort_keys=True) + "\n")
    print(json.dumps(_summary(groups, args.candidate_k, out), indent=2, sort_keys=True))


def prepare_reranker_groups(
    *,
    beir_dir: Path,
    dataset_name: str,
    split: str,
    telemetry_cache_dir: Path,
    dense_embedding_cache: Path,
    candidate_k: int,
    chunk_size: int = 512,
    chunk_overlap: int = 0,
    feature_catalog_path: Path | None = None,
) -> list[dict[str, Any]]:
    dataset = load_beir_dataset(beir_dir, name=dataset_name, split=split)
    chunks, chunk_to_doc_id = _build_chunks(
        dataset_name=dataset_name,
        corpus=dataset.corpus,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    rows_by_chunk = _load_cache_rows(telemetry_cache_dir)
    dense = np.load(dense_embedding_cache, allow_pickle=False)
    dense_chunk_ids = [str(value) for value in dense["chunk_ids"]]
    dense_query_ids = [str(value) for value in dense["query_ids"]]
    doc_vectors = np.array(dense["doc_vectors"], dtype=np.float64)
    query_vectors = np.array(dense["query_vectors"], dtype=np.float64)
    chunk_index = {chunk_id: index for index, chunk_id in enumerate(dense_chunk_ids)}
    query_index = {query_id: index for index, query_id in enumerate(dense_query_ids)}
    usable_chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.chunk_id in rows_by_chunk and chunk.chunk_id in chunk_index]
    activation_feature_names = sorted(
        {
            key
            for chunk_id in usable_chunk_ids
            for key in rows_by_chunk[chunk_id].get("sae_feature_values", {})
        }
    )
    feature_catalog = load_feature_catalog(feature_catalog_path) if feature_catalog_path is not None else None
    semantic_feature_groups = (
        feature_catalog.groups_for_feature_names(activation_feature_names)
        if feature_catalog is not None
        else {}
    )
    groups: list[dict[str, Any]] = []
    for query_id, query_text in dataset.queries.items():
        if query_id not in query_index:
            continue
        query_chunk_id = stable_hash(f"query\n{query_text}", 32)
        query_row = rows_by_chunk.get(query_chunk_id)
        if query_row is None:
            continue
        dense_scores = _cosine_scores(query_vectors[query_index[query_id]], doc_vectors)
        candidate_dense_indices = _top_indices(dense_scores, min(candidate_k, len(dense_scores)))
        positives = {doc_id for doc_id, grade in dataset.qrels.get(query_id, {}).items() if grade > 0}
        query_activation = _activation_vector(query_row, activation_feature_names)
        raw_candidates: list[dict[str, Any]] = []
        for rank, dense_index in enumerate(candidate_dense_indices, start=1):
            chunk_id = dense_chunk_ids[int(dense_index)]
            if chunk_id not in rows_by_chunk or chunk_id not in chunk_by_id:
                continue
            doc_id = chunk_to_doc_id[chunk_id]
            label = 1 if doc_id in positives else 0
            doc_activation = _activation_vector(rows_by_chunk[chunk_id], activation_feature_names)
            features = _activation_features(
                query_activation=query_activation,
                doc_activation=doc_activation,
                query_values=query_row.get("sae_feature_values", {}),
                doc_values=rows_by_chunk[chunk_id].get("sae_feature_values", {}),
                semantic_feature_groups=semantic_feature_groups,
            )
            features["dense_score"] = float(dense_scores[int(dense_index)])
            features["dense_rank_reciprocal"] = 1.0 / rank
            raw_candidates.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": chunk_by_id[chunk_id].text,
                    "dense_rank": rank,
                    "dense_score": float(dense_scores[int(dense_index)]),
                    "label": label,
                    "negative_source": None if label else "dense_hard_negative",
                    "negative_trust": None if label else "unjudged_assumed_negative",
                    "features": features,
                }
            )
        if not raw_candidates:
            continue
        _add_group_zscores(raw_candidates, "dense_score", "dense_z")
        _add_group_zscores(raw_candidates, "activation_cosine", "activation_cosine_z")
        groups.append(
            {
                "schema_version": SCHEMA_VERSION,
                "dataset_name": dataset.name,
                "split": dataset.split,
                "query_id": query_id,
                "query_text": query_text,
                "query_activation_chunk_id": query_chunk_id,
                "candidate_k": candidate_k,
                "positive_doc_ids": sorted(positives),
                "positive_in_candidate_pool": any(candidate["label"] > 0 for candidate in raw_candidates),
                "false_negative_policy": "unjudged_dense_candidates_assumed_negative_except_qrel_positives",
                "activation_feature_catalog": feature_catalog.summary() if feature_catalog is not None else None,
                "candidates": raw_candidates,
            }
        )
    return groups


def _build_chunks(*, dataset_name: str, corpus: dict[str, str], chunk_size: int, chunk_overlap: int):
    documents = [
        DocumentRecord.from_text(
            source_uri=f"benchmark://{dataset_name}/{doc_id}",
            title=doc_id,
            text=text,
            metadata={"benchmark_doc_id": doc_id},
        )
        for doc_id, text in corpus.items()
    ]
    document_id_to_doc_id = {document.document_id: str(document.metadata["benchmark_doc_id"]) for document in documents}
    chunks = Chunker(ChunkerSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)).split(documents)
    return chunks, {chunk.chunk_id: document_id_to_doc_id[chunk.document_id] for chunk in chunks}


def _load_cache_rows(cache_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(cache_dir.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if _valid_telemetry_row(row):
            rows[str(row["chunk_id"])] = row
    return rows


def _valid_telemetry_row(row: dict[str, Any]) -> bool:
    return bool(
        row.get("telemetry_valid", True)
        and not row.get("invalid_reason")
        and row.get("sae_feature_values")
    )


def _activation_vector(row: dict[str, Any], feature_names: list[str]) -> np.ndarray:
    values = row.get("sae_feature_values") or {}
    return np.array([float(values.get(name, 0.0)) for name in feature_names], dtype=np.float64)


def _activation_features(
    *,
    query_activation: np.ndarray,
    doc_activation: np.ndarray,
    query_values: dict[str, float],
    doc_values: dict[str, float],
    semantic_feature_groups: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    diff = query_activation - doc_activation
    product = query_activation * doc_activation
    features = {
        "activation_cosine": _cosine(query_activation, doc_activation),
        "activation_l2_distance": float(np.linalg.norm(diff)),
        "activation_abs_diff_mean": float(np.abs(diff).mean()) if diff.size else 0.0,
        "activation_abs_diff_max": float(np.abs(diff).max()) if diff.size else 0.0,
        "activation_product_mean": float(product.mean()) if product.size else 0.0,
    }
    for site in _sites(query_values, doc_values):
        names = sorted(name for name in set(query_values) | set(doc_values) if _site_from_feature_name(name) == site)
        query_site = np.array([float(query_values.get(name, 0.0)) for name in names], dtype=np.float64)
        doc_site = np.array([float(doc_values.get(name, 0.0)) for name in names], dtype=np.float64)
        features[f"activation_site:{site}"] = _cosine(query_site, doc_site)
    for group_name, names in sorted((semantic_feature_groups or {}).items()):
        query_group = np.array([float(query_values.get(name, 0.0)) for name in names], dtype=np.float64)
        doc_group = np.array([float(doc_values.get(name, 0.0)) for name in names], dtype=np.float64)
        group_diff = query_group - doc_group
        group_product = query_group * doc_group
        prefix = f"activation_semantic:{group_name}"
        features[f"{prefix}:cosine"] = _cosine(query_group, doc_group)
        features[f"{prefix}:l2_distance"] = float(np.linalg.norm(group_diff))
        features[f"{prefix}:abs_diff_mean"] = float(np.abs(group_diff).mean()) if group_diff.size else 0.0
        features[f"{prefix}:product_mean"] = float(group_product.mean()) if group_product.size else 0.0
        features[f"{prefix}:query_abs_mass"] = float(np.abs(query_group).sum())
        features[f"{prefix}:doc_abs_mass"] = float(np.abs(doc_group).sum())
    return features


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(a @ b / denominator)


def _cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    denominator = np.maximum(query_norm * doc_norms, 1e-12)
    return docs @ query / denominator


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    unordered = np.argpartition(-scores, kth=k - 1)[:k]
    return unordered[np.argsort(-scores[unordered])]


def _add_group_zscores(candidates: list[dict[str, Any]], source_feature: str, target_feature: str) -> None:
    values = np.array([candidate["features"].get(source_feature, 0.0) for candidate in candidates], dtype=np.float64)
    std = float(values.std())
    if std < 1e-8:
        std = 1.0
    mean = float(values.mean())
    for candidate, value in zip(candidates, values, strict=True):
        candidate["features"][target_feature] = float((value - mean) / std)


def _sites(*value_maps: dict[str, float]) -> list[str]:
    sites = {
        site
        for values in value_maps
        for name in values
        if (site := _site_from_feature_name(name)) is not None
    }
    return sorted(sites)


def _site_from_feature_name(name: str) -> str | None:
    parts = name.split(":")
    if len(parts) >= 4 and parts[0] == "act":
        return parts[1]
    return None


def _summary(groups: list[dict[str, Any]], candidate_k: int, out: Path) -> dict[str, Any]:
    candidate_counts = [len(group["candidates"]) for group in groups]
    positive_hits = sum(1 for group in groups if group["positive_in_candidate_pool"])
    return {
        "out": str(out),
        "group_count": len(groups),
        "candidate_k": candidate_k,
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "positive_in_candidate_pool_count": positive_hits,
    }


if __name__ == "__main__":
    main()
