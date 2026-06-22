#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
from scripts.build_core245_permutation_groups import (
    _category_groups,
    _load_cache_rows,
    _load_manifest,
    _l2_normalize,
    _matched_counterfactual_groups,
    _row_vector,
    _signed_log1p,
)


REPRESENTATIONS = ("raw", "log1p_l2", "category_aggregate", "counterfactual_aggregate")


@dataclass(frozen=True)
class SearchExample:
    query_id: str
    query_vector: np.ndarray
    candidate_vectors: np.ndarray
    labels: np.ndarray
    doc_ids: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    dense_scores: np.ndarray
    dense_ranks: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a query-to-answer activation representation searcher.")
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--dev-groups", required=True)
    parser.add_argument("--test-groups", required=True)
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--representation", choices=REPRESENTATIONS, required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--scores-out")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--mse-weight", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--blend-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
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
        scores_out=Path(args.scores_out) if args.scores_out else None,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        mse_weight=args.mse_weight,
        top_k=args.top_k,
        blend_grid=tuple(float(item) for item in args.blend_grid.split(",") if item.strip()),
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
    scores_out: Path | None,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    mse_weight: float,
    top_k: int,
    blend_grid: tuple[float, ...],
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    from torch import nn
    import torch.nn.functional as F

    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rows_by_chunk = _load_cache_rows(telemetry_cache_dir)
    manifest = _load_manifest(feature_manifest_path)
    feature_ids = manifest["feature_ids"]
    feature_meta = manifest["feature_meta"]
    vectorizer = _make_vectorizer(
        representation=representation,
        feature_ids=feature_ids,
        feature_meta=feature_meta,
        seed=seed,
    )
    train_examples = _build_examples(load_jsonl(train_groups_path), rows_by_chunk, feature_ids, vectorizer)
    dev_examples = _build_examples(load_jsonl(dev_groups_path), rows_by_chunk, feature_ids, vectorizer)
    test_examples = _build_examples(load_jsonl(test_groups_path), rows_by_chunk, feature_ids, vectorizer)
    if not train_examples:
        raise ValueError("no train examples with query telemetry, positive candidates, and negative candidates")
    normalizer = _fit_normalizer(train_examples)
    input_dim = int(train_examples[0].query_vector.shape[0])
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_tensors = [_example_tensors(example, normalizer, device=device) for example in train_examples]
    metric_key = f"ndcg@{min(10, top_k)}"
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_dev_score = float("-inf")
    epoch_summaries: list[dict[str, float]] = []

    for epoch in range(1, max(1, epochs) + 1):
        random.shuffle(train_tensors)
        model.train()
        losses: list[float] = []
        for query, candidates, labels in train_tensors:
            positives = labels > 0
            if not bool(positives.any()) or bool((labels <= 0).sum() == 0):
                continue
            optimizer.zero_grad()
            prediction = model(query)
            logits = F.normalize(prediction.unsqueeze(0), dim=1) @ F.normalize(candidates, dim=1).T
            logits = logits.squeeze(0) / max(temperature, 1e-6)
            target = labels.float()
            target = target / target.sum()
            loss = -(target * F.log_softmax(logits, dim=0)).sum()
            if mse_weight > 0.0:
                target_mean = F.normalize(candidates[positives].mean(dim=0, keepdim=True), dim=1).squeeze(0)
                loss = loss + (mse_weight * F.mse_loss(F.normalize(prediction, dim=0), target_mean))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics, _ = evaluate_examples(dev_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=None, device=device)
        score = float(dev_metrics["model"].get(metric_key, float("-inf")))
        epoch_summaries.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)) if losses else 0.0, "dev_score": score})
        if score > best_dev_score:
            best_dev_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics_model_only, _ = evaluate_examples(train_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=None, device=device)
    dev_metrics_model_only, _ = evaluate_examples(dev_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=None, device=device)
    test_metrics_model_only, test_model_only_scores = evaluate_examples(test_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=None, device=device)
    blend_sweep = _blend_sweep(dev_examples, test_examples, model=model, normalizer=normalizer, top_k=top_k, blend_grid=blend_grid, device=device)
    best_alpha = _select_blend_alpha_from_sweep(blend_sweep, top_k=top_k)
    train_metrics_blend, _ = evaluate_examples(train_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=best_alpha, device=device)
    dev_metrics_blend, _ = evaluate_examples(dev_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=best_alpha, device=device)
    test_metrics_blend, test_blend_scores = evaluate_examples(test_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=best_alpha, device=device)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "activation_rag.answer_representation_searcher.v1",
            "state_dict": model.state_dict(),
            "representation": representation,
            "feature_set_id": manifest["feature_set_id"],
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "normalizer": normalizer,
            "temperature": temperature,
            "mse_weight": mse_weight,
            "best_epoch": best_epoch,
            "best_dev_score": best_dev_score,
            "blend_alpha": best_alpha,
        },
        model_out,
    )
    if scores_out:
        _write_scores(scores_out, test_blend_scores)
        _write_scores(scores_out.with_suffix(".model_only.jsonl"), test_model_only_scores)
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.answer_representation_searcher_run.v1",
        "representation": representation,
        "feature_set_id": manifest["feature_set_id"],
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "mse_weight": mse_weight,
        "device": device,
        "seed": seed,
        "train_query_count": len(train_examples),
        "dev_query_count": len(dev_examples),
        "test_query_count": len(test_examples),
        "best_epoch": best_epoch,
        "best_dev_score": best_dev_score,
        "train_metrics_model_only": train_metrics_model_only,
        "dev_metrics_model_only": dev_metrics_model_only,
        "test_metrics_model_only": test_metrics_model_only,
        "best_blend_alpha": best_alpha,
        "blend_sweep": blend_sweep,
        "train_metrics_selected_blend": train_metrics_blend,
        "dev_metrics_selected_blend": dev_metrics_blend,
        "test_metrics_selected_blend": test_metrics_blend,
        "train_metrics": train_metrics_blend,
        "test_metrics": test_metrics_blend,
        "epoch_summaries_tail": epoch_summaries[-10:],
        "model_out": str(model_out),
        "metrics_out": str(metrics_out),
        "scores_out": str(scores_out) if scores_out else None,
    }
    write_json(metrics_out, summary)
    return summary


def _make_vectorizer(
    *,
    representation: str,
    feature_ids: list[str],
    feature_meta: dict[str, dict[str, Any]],
    seed: int,
) -> Any:
    if representation == "raw":
        return lambda vector: vector.astype(np.float64)
    if representation == "log1p_l2":
        return lambda vector: _l2_normalize(_signed_log1p(vector[None, :]))[0]
    if representation in {"category_aggregate", "counterfactual_aggregate"}:
        groups = _category_groups(feature_ids, feature_meta)
        if representation == "counterfactual_aggregate":
            groups = _matched_counterfactual_groups(feature_ids, groups, seed=seed)
        ordered_groups = list(groups.items())
        feature_index = {feature_id: index for index, feature_id in enumerate(feature_ids)}

        def vectorize(vector: np.ndarray) -> np.ndarray:
            out: list[float] = []
            for _, members in ordered_groups:
                indices = [feature_index[feature_id] for feature_id in members if feature_id in feature_index]
                values = vector[indices] if indices else np.zeros(0, dtype=np.float64)
                effects = np.array(
                    [abs(float(feature_meta[feature_id].get("causal_effect") or 0.0)) for feature_id in members if feature_id in feature_index],
                    dtype=np.float64,
                )
                if values.size == 0:
                    out.extend([0.0, 0.0, 0.0, 0.0])
                    continue
                out.append(float(values.sum()))
                out.append(float(np.abs(values).sum()))
                out.append(float(np.max(values)))
                out.append(float((np.abs(values) * effects).sum() / max(float(effects.sum()), 1e-12)) if effects.size else 0.0)
            return np.array(out, dtype=np.float64)

        return vectorize
    raise ValueError(f"unknown representation: {representation}")


def _build_examples(
    groups: list[dict[str, Any]],
    rows_by_chunk: dict[str, dict[str, Any]],
    feature_ids: list[str],
    vectorizer: Any,
) -> list[SearchExample]:
    examples: list[SearchExample] = []
    for group in groups:
        query_row = rows_by_chunk.get(str(group.get("query_activation_chunk_id") or ""))
        if query_row is None:
            continue
        candidates = []
        labels = []
        doc_ids = []
        chunk_ids = []
        dense_scores = []
        dense_ranks = []
        for candidate in group.get("candidates", []):
            row = rows_by_chunk.get(str(candidate.get("chunk_id")))
            if row is None:
                continue
            candidates.append(vectorizer(_row_vector(row, feature_ids)))
            labels.append(int(candidate.get("label", 0)))
            doc_ids.append(str(candidate.get("doc_id")))
            chunk_ids.append(str(candidate.get("chunk_id")))
            dense_scores.append(float(candidate.get("dense_score", (candidate.get("features") or {}).get("dense_score", 0.0))))
            dense_ranks.append(float(candidate.get("dense_rank", 10**9)))
        if not candidates or max(labels, default=0) <= 0 or sum(1 for label in labels if label <= 0) == 0:
            continue
        examples.append(
            SearchExample(
                query_id=str(group["query_id"]),
                query_vector=vectorizer(_row_vector(query_row, feature_ids)),
                candidate_vectors=np.vstack(candidates),
                labels=np.array(labels, dtype=np.int64),
                doc_ids=tuple(doc_ids),
                chunk_ids=tuple(chunk_ids),
                dense_scores=np.array(dense_scores, dtype=np.float64),
                dense_ranks=np.array(dense_ranks, dtype=np.float64),
            )
        )
    return examples


def _fit_normalizer(examples: list[SearchExample]) -> dict[str, list[float]]:
    matrix = np.vstack(
        [example.query_vector for example in examples]
        + [candidate for example in examples for candidate in example.candidate_vectors]
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def _normalize_vector(vector: np.ndarray, normalizer: dict[str, list[float]]) -> np.ndarray:
    return (vector - np.array(normalizer["mean"], dtype=np.float64)) / np.array(normalizer["scale"], dtype=np.float64)


def _example_tensors(example: SearchExample, normalizer: dict[str, list[float]], *, device: str) -> tuple[Any, Any, Any]:
    import torch

    query = _normalize_vector(example.query_vector, normalizer)
    candidates = np.vstack([_normalize_vector(vector, normalizer) for vector in example.candidate_vectors])
    return (
        torch.tensor(query, dtype=torch.float32, device=device),
        torch.tensor(candidates, dtype=torch.float32, device=device),
        torch.tensor(example.labels, dtype=torch.long, device=device),
    )


def evaluate_examples(
    examples: list[SearchExample],
    *,
    model: Any,
    normalizer: dict[str, list[float]],
    top_k: int,
    blend_alpha: float | None,
    device: str,
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], float]]:
    scores_by_query: dict[str, list[tuple[str, str, float, float, int]]] = {}
    score_map: dict[tuple[str, str], float] = {}
    for example in examples:
        model_scores = _predict_scores(example, model=model, normalizer=normalizer, device=device)
        if blend_alpha is None:
            final_scores = model_scores
        else:
            final_scores = _zscore(example.dense_scores) * (1.0 - blend_alpha) + _zscore(model_scores) * blend_alpha
        rows = []
        for index, doc_id in enumerate(example.doc_ids):
            score = float(final_scores[index])
            rows.append((doc_id, example.chunk_ids[index], score, float(example.dense_ranks[index]), int(example.labels[index])))
            score_map[(example.query_id, example.chunk_ids[index])] = score
        scores_by_query[example.query_id] = rows
    return {
        "dense": _metrics_from_scores(scores_by_query, top_k=top_k, use_dense=True),
        "model": _metrics_from_scores(scores_by_query, top_k=top_k, use_dense=False),
    }, score_map


def _predict_scores(example: SearchExample, *, model: Any, normalizer: dict[str, list[float]], device: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    query, candidates, _ = _example_tensors(example, normalizer, device=device)
    model.eval()
    with torch.no_grad():
        prediction = F.normalize(model(query).unsqueeze(0), dim=1)
        scores = prediction @ F.normalize(candidates, dim=1).T
    return scores.squeeze(0).detach().cpu().numpy().astype(np.float64)


def _blend_sweep(
    dev_examples: list[SearchExample],
    test_examples: list[SearchExample],
    *,
    model: Any,
    normalizer: dict[str, list[float]],
    top_k: int,
    blend_grid: tuple[float, ...],
    device: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for alpha in blend_grid:
        dev_metrics, _ = evaluate_examples(dev_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=alpha, device=device)
        test_metrics, _ = evaluate_examples(test_examples, model=model, normalizer=normalizer, top_k=top_k, blend_alpha=alpha, device=device)
        rows.append({"alpha": float(alpha), "dev_metrics": dev_metrics, "test_metrics": test_metrics})
    return rows


def _select_blend_alpha_from_sweep(blend_sweep: list[dict[str, Any]], *, top_k: int) -> float:
    metric_key = f"ndcg@{min(10, top_k)}"
    best_alpha = 1.0
    best_score = float("-inf")
    for row in blend_sweep:
        score = float(row["dev_metrics"]["model"].get(metric_key, float("-inf")))
        if score > best_score:
            best_score = score
            best_alpha = float(row["alpha"])
    return best_alpha


def _metrics_from_scores(scores_by_query: dict[str, list[tuple[str, str, float, float, int]]], *, top_k: int, use_dense: bool) -> dict[str, float]:
    mrr = []
    recall = []
    ndcg = []
    for rows in scores_by_query.values():
        qrels = {doc_id: label for doc_id, _, _, _, label in rows if label > 0}
        if use_dense:
            ranked = [doc_id for doc_id, _, _, _, _ in sorted(rows, key=lambda row: row[3])]
        else:
            ranked = [doc_id for doc_id, _, _, _, _ in sorted(rows, key=lambda row: row[2], reverse=True)]
        mrr.append(mean_reciprocal_rank(ranked, qrels, min(10, top_k)))
        recall.append(recall_at_k(ranked, qrels, top_k))
        ndcg.append(ndcg_at_k(ranked, qrels, min(10, top_k)))
    return {
        f"mrr@{min(10, top_k)}": float(sum(mrr) / len(mrr)) if mrr else 0.0,
        f"ndcg@{min(10, top_k)}": float(sum(ndcg) / len(ndcg)) if ndcg else 0.0,
        f"recall@{top_k}": float(sum(recall) / len(recall)) if recall else 0.0,
    }


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _write_scores(path: Path, scores: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (query_id, chunk_id), score in sorted(scores.items()):
            handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
