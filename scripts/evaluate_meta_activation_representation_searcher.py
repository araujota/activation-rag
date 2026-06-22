#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.benchmarks import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from activation_rag.supervised_reranking import load_jsonl, write_json
from scripts.build_core245_permutation_groups import _load_cache_rows, _row_vector, _signed_log1p
from scripts.train_meta_activation_representation_searcher import ResidualActivationPredictor, apply_geometry


@dataclass(frozen=True)
class ScoredGroup:
    query_id: str
    qrels: dict[str, int]
    dense_ranked_doc_ids: list[str]
    rerank_scores: dict[tuple[str, str], float]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained meta answer-activation predictor on heldout groups.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scores-out")
    parser.add_argument("--pure-index-scores-out")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = evaluate_meta_searcher(
        groups_path=Path(args.groups),
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        model_path=Path(args.model),
        out_path=Path(args.out),
        scores_out=Path(args.scores_out) if args.scores_out else None,
        pure_index_scores_out=Path(args.pure_index_scores_out) if args.pure_index_scores_out else None,
        top_k=args.top_k,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def evaluate_meta_searcher(
    *,
    groups_path: Path,
    telemetry_cache_dir: Path,
    model_path: Path,
    out_path: Path,
    scores_out: Path | None,
    pure_index_scores_out: Path | None,
    top_k: int,
    device: str,
) -> dict[str, Any]:
    import torch

    groups = load_jsonl(groups_path)
    rows_by_chunk = _load_cache_rows(telemetry_cache_dir)
    payload = torch.load(model_path, map_location="cpu")
    model = ResidualActivationPredictor(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        dropout=float(payload.get("dropout", 0.0)),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    feature_ids = [str(feature_id) for feature_id in payload["feature_ids"]]
    representation = str(payload.get("representation") or "raw")
    geometry = payload["geometry"]

    scored_groups, rerank_score_map = _score_dense_candidate_groups(
        groups=groups,
        rows_by_chunk=rows_by_chunk,
        model=model,
        feature_ids=feature_ids,
        representation=representation,
        geometry=geometry,
        device=device,
    )
    pure_rankings, pure_score_map, pure_index_chunk_count = _score_pure_activation_index(
        groups=groups,
        rows_by_chunk=rows_by_chunk,
        model=model,
        feature_ids=feature_ids,
        representation=representation,
        geometry=geometry,
        device=device,
        collect_scores=pure_index_scores_out is not None,
    )

    if scores_out:
        _write_score_map(scores_out, rerank_score_map)
    if pure_index_scores_out:
        _write_score_map(pure_index_scores_out, pure_score_map)

    qrels_by_query = {group.query_id: group.qrels for group in scored_groups}
    dense_rankings = {group.query_id: group.dense_ranked_doc_ids for group in scored_groups}
    groups_by_query = {str(group["query_id"]): group for group in groups}
    rerank_rankings = {
        group.query_id: [
            doc_id
            for doc_id, _, _ in sorted(
                _candidate_score_rows(group, groups_by_query[str(group.query_id)]),
                key=lambda row: row[2],
                reverse=True,
            )
        ]
        for group in scored_groups
    }
    summary = {
        "schema_version": "activation_rag.meta_answer_representation_searcher_eval.v1",
        "groups": str(groups_path),
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "model": str(model_path),
        "query_count": len(scored_groups),
        "top_k": top_k,
        "representation": representation,
        "feature_set_id": str(payload.get("feature_set_id") or ""),
        "metrics": {
            "pure_dense_candidates": _aggregate(dense_rankings, qrels_by_query, top_k),
            "dense_candidates_actpred_rerank": _aggregate(rerank_rankings, qrels_by_query, top_k),
            "pure_actpred_available_activation_index": _aggregate(pure_rankings, qrels_by_query, top_k),
        },
        "score_outputs": {
            "dense_candidate_actpred_scores": str(scores_out) if scores_out else None,
            "pure_index_actpred_scores": str(pure_index_scores_out) if pure_index_scores_out else None,
        },
        "pure_index_scope": {
            "description": "Ranks the union of captured heldout candidate chunks for this dataset, not an uncaptured full BEIR corpus.",
            "unique_index_chunk_count": pure_index_chunk_count,
        },
    }
    write_json(out_path, summary)
    return summary


def _score_dense_candidate_groups(
    *,
    groups: list[dict[str, Any]],
    rows_by_chunk: dict[str, dict[str, Any]],
    model: Any,
    feature_ids: list[str],
    representation: str,
    geometry: dict[str, Any],
    device: str,
) -> tuple[list[ScoredGroup], dict[tuple[str, str], float]]:
    scored_groups: list[ScoredGroup] = []
    score_map: dict[tuple[str, str], float] = {}
    for group in groups:
        query_id = str(group["query_id"])
        query_prediction = _predict_query(
            group,
            rows_by_chunk=rows_by_chunk,
            model=model,
            feature_ids=feature_ids,
            representation=representation,
            geometry=geometry,
            device=device,
        )
        if query_prediction is None:
            continue
        candidate_rows: list[tuple[str, str, float, int, int]] = []
        for candidate in group.get("candidates", []):
            chunk_id = str(candidate["chunk_id"])
            row = rows_by_chunk.get(chunk_id)
            if row is None:
                continue
            score = _score_vector(
                query_prediction,
                _geometry_vector(row, feature_ids=feature_ids, representation=representation, geometry=geometry),
            )
            score_map[(query_id, chunk_id)] = score
            candidate_rows.append(
                (
                    str(candidate["doc_id"]),
                    chunk_id,
                    score,
                    int(candidate.get("dense_rank", 10**9)),
                    int(candidate.get("label", 0)),
                )
            )
        if not candidate_rows:
            continue
        scored_groups.append(
            ScoredGroup(
                query_id=query_id,
                qrels={doc_id: label for doc_id, _, _, _, label in candidate_rows if label > 0},
                dense_ranked_doc_ids=[doc_id for doc_id, _, _, _, _ in sorted(candidate_rows, key=lambda row: row[3])],
                rerank_scores=score_map,
            )
        )
    return scored_groups, score_map


def _score_pure_activation_index(
    *,
    groups: list[dict[str, Any]],
    rows_by_chunk: dict[str, dict[str, Any]],
    model: Any,
    feature_ids: list[str],
    representation: str,
    geometry: dict[str, Any],
    device: str,
    collect_scores: bool,
) -> tuple[dict[str, list[str]], dict[tuple[str, str], float], int]:
    index: list[tuple[str, str, np.ndarray]] = []
    seen_chunks: set[str] = set()
    for group in groups:
        for candidate in group.get("candidates", []):
            chunk_id = str(candidate["chunk_id"])
            if chunk_id in seen_chunks:
                continue
            row = rows_by_chunk.get(chunk_id)
            if row is None:
                continue
            seen_chunks.add(chunk_id)
            index.append(
                (
                    str(candidate["doc_id"]),
                    chunk_id,
                    _geometry_vector(row, feature_ids=feature_ids, representation=representation, geometry=geometry),
                )
            )
    if index:
        index_doc_ids = [doc_id for doc_id, _, _ in index]
        index_chunk_ids = [chunk_id for _, chunk_id, _ in index]
        index_matrix = np.vstack([vector for _, _, vector in index]).astype(np.float64)
        norms = np.linalg.norm(index_matrix, axis=1, keepdims=True)
        index_matrix = index_matrix / np.maximum(norms, 1e-12)
    else:
        index_doc_ids = []
        index_chunk_ids = []
        index_matrix = np.zeros((0, len(feature_ids)), dtype=np.float64)
    rankings: dict[str, list[str]] = {}
    score_map: dict[tuple[str, str], float] = {}
    for group in groups:
        query_id = str(group["query_id"])
        prediction = _predict_query(
            group,
            rows_by_chunk=rows_by_chunk,
            model=model,
            feature_ids=feature_ids,
            representation=representation,
            geometry=geometry,
            device=device,
        )
        if prediction is None:
            continue
        prediction = prediction / max(float(np.linalg.norm(prediction)), 1e-12)
        scores = index_matrix @ prediction
        order = np.argsort(-scores)
        if collect_scores:
            for index_pos, score in enumerate(scores):
                score_map[(query_id, index_chunk_ids[index_pos])] = float(score)
        rankings[query_id] = [index_doc_ids[int(index_pos)] for index_pos in order]
    return rankings, score_map, len(index)


def _predict_query(
    group: dict[str, Any],
    *,
    rows_by_chunk: dict[str, dict[str, Any]],
    model: Any,
    feature_ids: list[str],
    representation: str,
    geometry: dict[str, Any],
    device: str,
) -> np.ndarray | None:
    import torch

    query_row = rows_by_chunk.get(str(group.get("query_activation_chunk_id") or ""))
    if query_row is None:
        return None
    query = _geometry_vector(query_row, feature_ids=feature_ids, representation=representation, geometry=geometry)
    with torch.no_grad():
        tensor = torch.tensor(query, dtype=torch.float32, device=device).unsqueeze(0)
        prediction = model(tensor).squeeze(0).detach().cpu().numpy().astype(np.float64)
    return prediction


def _geometry_vector(
    row: dict[str, Any],
    *,
    feature_ids: list[str],
    representation: str,
    geometry: dict[str, Any],
) -> np.ndarray:
    vector = _row_vector(row, feature_ids)
    if representation == "log1p":
        vector = _signed_log1p(vector[None, :])[0]
    return apply_geometry(vector, geometry).astype(np.float64)


def _score_vector(query_prediction: np.ndarray, candidate_vector: np.ndarray) -> float:
    left = query_prediction / max(float(np.linalg.norm(query_prediction)), 1e-12)
    right = candidate_vector / max(float(np.linalg.norm(candidate_vector)), 1e-12)
    return float(left @ right)


def _candidate_score_rows(group: ScoredGroup, original_group: dict[str, Any]) -> list[tuple[str, str, float]]:
    rows = []
    for candidate in original_group.get("candidates", []):
        chunk_id = str(candidate["chunk_id"])
        score = group.rerank_scores.get((group.query_id, chunk_id), float("-inf"))
        rows.append((str(candidate["doc_id"]), chunk_id, score))
    return rows


def _aggregate(rankings: dict[str, list[str]], qrels_by_query: dict[str, dict[str, int]], top_k: int) -> dict[str, float]:
    metric_k = min(10, top_k)
    mrr = []
    ndcg = []
    recall = []
    for query_id, qrels in qrels_by_query.items():
        ranked_doc_ids = rankings.get(query_id, [])
        mrr.append(mean_reciprocal_rank(ranked_doc_ids, qrels, metric_k))
        ndcg.append(ndcg_at_k(ranked_doc_ids, qrels, metric_k))
        recall.append(recall_at_k(ranked_doc_ids, qrels, top_k))
    return {
        f"mrr@{metric_k}": float(sum(mrr) / len(mrr)) if mrr else 0.0,
        f"ndcg@{metric_k}": float(sum(ndcg) / len(ndcg)) if ndcg else 0.0,
        f"recall@{top_k}": float(sum(recall) / len(recall)) if recall else 0.0,
    }


def _write_score_map(path: Path, scores: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (query_id, chunk_id), score in sorted(scores.items()):
            handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")

if __name__ == "__main__":
    main()
