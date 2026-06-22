#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from activation_rag.supervised_reranking import load_jsonl, write_json
from scripts.build_core245_permutation_groups import _load_cache_rows, _load_manifest, _row_vector, _signed_log1p


REPRESENTATIONS = ("raw", "log1p")
GEOMETRY_POLICIES = ("zscore_center_top_pc_l2", "zscore_whiten_l2")


@dataclass(frozen=True)
class SourceSpec:
    dataset_name: str
    groups_path: Path
    telemetry_cache_dir: Path


@dataclass(frozen=True)
class RawExample:
    dataset_name: str
    query_id: str
    query_vector: np.ndarray
    candidate_vectors: np.ndarray
    labels: np.ndarray
    dense_scores: np.ndarray


@dataclass(frozen=True)
class TrainExample:
    dataset_name: str
    query_id: str
    query_vector: np.ndarray
    candidate_vectors: np.ndarray
    labels: np.ndarray
    dense_scores: np.ndarray
    dataset_weight: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a train-only meta-BEIR answer-activation predictor without heldout validation.")
    parser.add_argument("--train-source", action="append", required=True, help="dataset_name:groups_jsonl:telemetry_cache_dir")
    parser.add_argument("--locked-validation-source", action="append", default=[], help="dataset_name:groups_jsonl:telemetry_cache_dir recorded but not evaluated")
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument(
        "--allow-positive-only-groups",
        action="store_true",
        help="Allow qrel-positive-only groups. Default remains dense groups with at least one negative.",
    )
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="raw")
    parser.add_argument("--geometry-policy", choices=GEOMETRY_POLICIES, default="zscore_center_top_pc_l2")
    parser.add_argument("--top-pc-removal", type=int, default=3)
    parser.add_argument("--whitening-rank", type=int, default=0)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--centroid-weight", type=float, default=0.10)
    parser.add_argument("--inbatch-weight", type=float, default=0.25)
    parser.add_argument("--margin-weight", type=float, default=0.10)
    parser.add_argument("--uniformity-weight", type=float, default=0.01)
    parser.add_argument("--hard-margin", type=float, default=0.10)
    parser.add_argument("--hubness-sample-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    summary = run_training(
        train_sources=[parse_source_spec(value) for value in args.train_source],
        locked_validation_sources=[parse_source_spec(value) for value in args.locked_validation_source],
        feature_manifest_path=Path(args.feature_manifest),
        allow_positive_only_groups=args.allow_positive_only_groups,
        representation=args.representation,
        geometry_policy=args.geometry_policy,
        top_pc_removal=args.top_pc_removal,
        whitening_rank=args.whitening_rank,
        model_out=Path(args.model_out),
        metrics_out=Path(args.metrics_out),
        manifest_out=Path(args.manifest_out),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        centroid_weight=args.centroid_weight,
        inbatch_weight=args.inbatch_weight,
        margin_weight=args.margin_weight,
        uniformity_weight=args.uniformity_weight,
        hard_margin=args.hard_margin,
        hubness_sample_size=args.hubness_sample_size,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_training(
    *,
    train_sources: list[SourceSpec],
    locked_validation_sources: list[SourceSpec],
    feature_manifest_path: Path,
    allow_positive_only_groups: bool,
    representation: str,
    geometry_policy: str,
    top_pc_removal: int,
    whitening_rank: int,
    model_out: Path,
    metrics_out: Path,
    manifest_out: Path,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    temperature: float,
    centroid_weight: float,
    inbatch_weight: float,
    margin_weight: float,
    uniformity_weight: float,
    hard_margin: float,
    hubness_sample_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    if not train_sources:
        raise ValueError("at least one train source is required")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    if geometry_policy not in GEOMETRY_POLICIES:
        raise ValueError(f"unknown geometry policy: {geometry_policy}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    manifest = _load_manifest(feature_manifest_path)
    feature_ids = manifest["feature_ids"]
    raw_examples = load_raw_examples(
        train_sources,
        feature_ids=feature_ids,
        representation=representation,
        allow_positive_only_groups=allow_positive_only_groups,
    )
    if not raw_examples:
        if allow_positive_only_groups:
            raise ValueError("no train examples with query telemetry and positive candidates")
        raise ValueError("no train examples with query telemetry, positive candidates, and negative candidates")
    geometry = fit_geometry(
        raw_examples,
        geometry_policy=geometry_policy,
        top_pc_removal=top_pc_removal,
        whitening_rank=whitening_rank,
    )
    train_examples = prepare_examples(raw_examples, geometry=geometry)
    dataset_counts = _dataset_counts(train_examples)
    train_examples = [
        TrainExample(
            dataset_name=example.dataset_name,
            query_id=example.query_id,
            query_vector=example.query_vector,
            candidate_vectors=example.candidate_vectors,
            labels=example.labels,
            dense_scores=example.dense_scores,
            dataset_weight=len(train_examples) / (len(dataset_counts) * dataset_counts[example.dataset_name]),
        )
        for example in train_examples
    ]

    input_dim = int(train_examples[0].query_vector.shape[0])
    model = ResidualActivationPredictor(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tensor_examples = [_example_tensors(example, device=device) for example in train_examples]
    epoch_summaries: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    best_train_loss = float("inf")

    for epoch in range(1, max(1, epochs) + 1):
        order = list(range(len(tensor_examples)))
        random.shuffle(order)
        model.train()
        loss_rows: list[dict[str, float]] = []
        for start in range(0, len(order), batch_size):
            batch = [tensor_examples[index] for index in order[start : start + batch_size]]
            optimizer.zero_grad()
            components = _batch_loss(
                model,
                batch,
                temperature=temperature,
                centroid_weight=centroid_weight,
                inbatch_weight=inbatch_weight,
                margin_weight=margin_weight,
                uniformity_weight=uniformity_weight,
                hard_margin=hard_margin,
            )
            components["loss"].backward()
            optimizer.step()
            loss_rows.append({key: float(value.detach().cpu()) for key, value in components.items()})
        epoch_summary = _aggregate_loss_rows(epoch, loss_rows)
        epoch_summaries.append(epoch_summary)
        if epoch_summary["loss"] < best_train_loss:
            best_train_loss = float(epoch_summary["loss"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(epoch_summary, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    train_diagnostics = train_only_diagnostics(
        model,
        tensor_examples,
        sample_size=hubness_sample_size,
        device=device,
        seed=seed,
    )
    locked_manifest = {
        "schema_version": "activation_rag.meta_beir_locked_validation_manifest.v1",
        "status": "locked_not_executed",
        "note": "These sources are recorded for later validation. This training run did not score or evaluate them.",
        "sources": [source_to_json(source) for source in locked_validation_sources],
    }

    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "activation_rag.meta_answer_representation_searcher.v1",
            "state_dict": model.state_dict(),
            "feature_set_id": manifest["feature_set_id"],
            "feature_ids": feature_ids,
            "representation": representation,
            "geometry": geometry,
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "loss_weights": {
                "centroid_weight": centroid_weight,
                "inbatch_weight": inbatch_weight,
                "margin_weight": margin_weight,
                "uniformity_weight": uniformity_weight,
                "hard_margin": hard_margin,
            "temperature": temperature,
        },
        "allow_positive_only_groups": allow_positive_only_groups,
        "dataset_counts": dataset_counts,
        "best_epoch_by_train_loss": best_epoch,
            "best_train_loss": best_train_loss,
        },
        model_out,
    )
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.meta_answer_representation_searcher_run.v1",
        "status": "training_complete_no_heldout_validation_executed",
        "train_sources": [source_to_json(source) for source in train_sources],
        "locked_validation_manifest": str(manifest_out),
        "feature_set_id": manifest["feature_set_id"],
        "representation": representation,
        "geometry_policy": geometry_policy,
        "top_pc_removal": top_pc_removal,
        "whitening_rank": whitening_rank,
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_weights": {
            "centroid_weight": centroid_weight,
            "inbatch_weight": inbatch_weight,
            "margin_weight": margin_weight,
            "uniformity_weight": uniformity_weight,
            "hard_margin": hard_margin,
            "temperature": temperature,
        },
        "allow_positive_only_groups": allow_positive_only_groups,
        "train_query_count": len(train_examples),
        "dataset_counts": dataset_counts,
        "best_epoch_by_train_loss": best_epoch,
        "best_train_loss": best_train_loss,
        "epoch_summaries": epoch_summaries,
        "epoch_summaries_tail": epoch_summaries[-10:],
        "train_only_diagnostics": train_diagnostics,
        "model_out": str(model_out),
        "metrics_out": str(metrics_out),
    }
    write_json(metrics_out, summary)
    write_json(manifest_out, locked_manifest)
    return summary


class ResidualActivationPredictor:
    def __new__(cls, *, input_dim: int, hidden_dim: int, dropout: float):
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.norm = nn.LayerNorm(input_dim)
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, input_dim),
                )
                self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

            def forward(self, query):
                normalized = self.norm(query)
                return self.net(normalized) + self.residual_scale * query

        return _Model()


def parse_source_spec(value: str) -> SourceSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("source must be dataset_name:groups_jsonl:telemetry_cache_dir")
    return SourceSpec(parts[0], Path(parts[1]), Path(parts[2]))


def source_to_json(source: SourceSpec) -> dict[str, Any]:
    return {
        "dataset_name": source.dataset_name,
        "groups_path": str(source.groups_path),
        "telemetry_cache_dir": str(source.telemetry_cache_dir),
        "groups_sha256": _file_sha256(source.groups_path) if source.groups_path.exists() else None,
        "groups_line_count": _line_count(source.groups_path) if source.groups_path.exists() else None,
    }


def load_raw_examples(
    sources: list[SourceSpec],
    *,
    feature_ids: list[str],
    representation: str,
    allow_positive_only_groups: bool = False,
) -> list[RawExample]:
    examples: list[RawExample] = []
    for source in sources:
        rows_by_chunk = _load_cache_rows(source.telemetry_cache_dir)
        for group in load_jsonl(source.groups_path):
            example = _raw_example_from_group(
                source.dataset_name,
                group,
                rows_by_chunk=rows_by_chunk,
                feature_ids=feature_ids,
                representation=representation,
                allow_positive_only_groups=allow_positive_only_groups,
            )
            if example is not None:
                examples.append(example)
    return examples


def _raw_example_from_group(
    dataset_name: str,
    group: dict[str, Any],
    *,
    rows_by_chunk: dict[str, dict[str, Any]],
    feature_ids: list[str],
    representation: str,
    allow_positive_only_groups: bool,
) -> RawExample | None:
    query_row = rows_by_chunk.get(str(group.get("query_activation_chunk_id") or ""))
    if query_row is None:
        return None
    candidates: list[np.ndarray] = []
    labels: list[int] = []
    dense_scores: list[float] = []
    for candidate in group.get("candidates", []):
        row = rows_by_chunk.get(str(candidate.get("chunk_id")))
        if row is None:
            continue
        candidates.append(_represent(_row_vector(row, feature_ids), representation))
        labels.append(int(candidate.get("label", 0)))
        dense_scores.append(float(candidate.get("dense_score", (candidate.get("features") or {}).get("dense_score", 0.0))))
    if not candidates or max(labels, default=0) <= 0:
        return None
    if not allow_positive_only_groups and sum(1 for label in labels if label <= 0) == 0:
        return None
    return RawExample(
        dataset_name=dataset_name,
        query_id=f"{dataset_name}:{group['query_id']}",
        query_vector=_represent(_row_vector(query_row, feature_ids), representation),
        candidate_vectors=np.vstack(candidates).astype(np.float64),
        labels=np.array(labels, dtype=np.int64),
        dense_scores=np.array(dense_scores, dtype=np.float64),
    )


def _represent(vector: np.ndarray, representation: str) -> np.ndarray:
    if representation == "raw":
        return vector.astype(np.float64)
    if representation == "log1p":
        return _signed_log1p(vector[None, :])[0].astype(np.float64)
    raise ValueError(f"unknown representation: {representation}")


def fit_geometry(
    examples: list[RawExample],
    *,
    geometry_policy: str,
    top_pc_removal: int,
    whitening_rank: int,
) -> dict[str, Any]:
    matrix = np.vstack(
        [example.query_vector for example in examples]
        + [candidate for example in examples for candidate in example.candidate_vectors]
    ).astype(np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    scaled = (matrix - mean) / scale
    _, singular_values, vt = np.linalg.svd(scaled, full_matrices=False)
    variance = (singular_values**2) / max(1, scaled.shape[0] - 1)
    explained = variance / max(float(variance.sum()), 1e-12)
    if geometry_policy == "zscore_whiten_l2":
        rank = whitening_rank if whitening_rank > 0 else min(scaled.shape[1], 128)
        components = vt[:rank]
        component_variance = variance[:rank]
        output_dim = rank
    else:
        remove_count = max(0, min(top_pc_removal, vt.shape[0]))
        components = vt[:remove_count]
        component_variance = variance[:remove_count]
        output_dim = scaled.shape[1]
    return {
        "schema_version": "activation_rag.activation_geometry_transform.v1",
        "geometry_policy": geometry_policy,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "components": components.tolist(),
        "component_variance": component_variance.tolist(),
        "explained_variance": explained[: max(len(components), 10)].tolist(),
        "output_dim": int(output_dim),
        "train_vector_count": int(matrix.shape[0]),
    }


def apply_geometry(vector: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    mean = np.array(geometry["mean"], dtype=np.float64)
    scale = np.array(geometry["scale"], dtype=np.float64)
    transformed = (vector.astype(np.float64) - mean) / scale
    components = np.array(geometry.get("components") or [], dtype=np.float64)
    if str(geometry["geometry_policy"]) == "zscore_whiten_l2":
        variances = np.array(geometry.get("component_variance") or [], dtype=np.float64)
        if components.size == 0:
            return transformed
        return (transformed @ components.T) / np.sqrt(np.maximum(variances, 1e-6))
    if components.size:
        transformed = transformed - (transformed @ components.T) @ components
    return transformed


def prepare_examples(raw_examples: list[RawExample], *, geometry: dict[str, Any]) -> list[TrainExample]:
    return [
        TrainExample(
            dataset_name=example.dataset_name,
            query_id=example.query_id,
            query_vector=apply_geometry(example.query_vector, geometry).astype(np.float32),
            candidate_vectors=np.vstack([apply_geometry(vector, geometry) for vector in example.candidate_vectors]).astype(np.float32),
            labels=example.labels,
            dense_scores=example.dense_scores,
            dataset_weight=1.0,
        )
        for example in raw_examples
    ]


def _dataset_counts(examples: list[TrainExample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.dataset_name] = counts.get(example.dataset_name, 0) + 1
    return counts


def _example_tensors(example: TrainExample, *, device: str) -> dict[str, Any]:
    import torch

    return {
        "dataset_name": example.dataset_name,
        "query_id": example.query_id,
        "query": torch.tensor(example.query_vector, dtype=torch.float32, device=device),
        "candidates": torch.tensor(example.candidate_vectors, dtype=torch.float32, device=device),
        "labels": torch.tensor(example.labels, dtype=torch.float32, device=device),
        "dataset_weight": torch.tensor(example.dataset_weight, dtype=torch.float32, device=device),
    }


def _batch_loss(
    model: Any,
    batch: list[dict[str, Any]],
    *,
    temperature: float,
    centroid_weight: float,
    inbatch_weight: float,
    margin_weight: float,
    uniformity_weight: float,
    hard_margin: float,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    queries = torch.stack([row["query"] for row in batch])
    predictions = model(queries)
    listwise_losses = []
    centroid_losses = []
    margin_losses = []
    centroids = []
    weights = torch.stack([row["dataset_weight"] for row in batch])
    for index, row in enumerate(batch):
        labels = row["labels"]
        positive_mask = labels > 0
        candidates = row["candidates"]
        candidate_norm = F.normalize(candidates, dim=1)
        prediction_norm = F.normalize(predictions[index], dim=0)
        logits = (candidate_norm @ prediction_norm) / max(temperature, 1e-6)
        target = labels.clamp_min(0.0)
        target = target / target.sum().clamp_min(1e-12)
        listwise_losses.append(-(target * F.log_softmax(logits, dim=0)).sum())
        positive_vectors = candidates[positive_mask]
        centroid = F.normalize(positive_vectors.mean(dim=0), dim=0)
        centroids.append(centroid)
        centroid_losses.append(1.0 - torch.dot(prediction_norm, centroid))
        positive_logits = logits[positive_mask]
        negative_logits = logits[~positive_mask]
        if positive_logits.numel() and negative_logits.numel():
            margin_losses.append(F.softplus(negative_logits.max() - positive_logits.min() + hard_margin))
        else:
            margin_losses.append(torch.zeros((), dtype=torch.float32, device=logits.device))
    listwise = _weighted_mean(torch.stack(listwise_losses), weights)
    centroid_loss = _weighted_mean(torch.stack(centroid_losses), weights)
    margin_loss = _weighted_mean(torch.stack(margin_losses), weights)
    pred_norm = F.normalize(predictions, dim=1)
    centroid_norm = F.normalize(torch.stack(centroids), dim=1)
    inbatch_logits = pred_norm @ centroid_norm.T / max(temperature, 1e-6)
    inbatch_targets = torch.arange(len(batch), dtype=torch.long, device=inbatch_logits.device)
    inbatch_loss = F.cross_entropy(inbatch_logits, inbatch_targets)
    uniformity_loss = _uniformity_loss(pred_norm)
    loss = (
        listwise
        + centroid_weight * centroid_loss
        + inbatch_weight * inbatch_loss
        + margin_weight * margin_loss
        + uniformity_weight * uniformity_loss
    )
    return {
        "loss": loss,
        "listwise_loss": listwise,
        "centroid_loss": centroid_loss,
        "inbatch_loss": inbatch_loss,
        "margin_loss": margin_loss,
        "uniformity_loss": uniformity_loss,
    }


def _weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp_min(1e-12)


def _uniformity_loss(normalized_predictions):
    import torch

    if normalized_predictions.shape[0] < 2:
        return torch.zeros((), dtype=normalized_predictions.dtype, device=normalized_predictions.device)
    distances = torch.pdist(normalized_predictions, p=2).pow(2)
    return torch.log(torch.exp(-2.0 * distances).mean().clamp_min(1e-12))


def _aggregate_loss_rows(epoch: int, rows: list[dict[str, float]]) -> dict[str, Any]:
    keys = sorted(rows[0]) if rows else []
    return {
        "epoch": epoch,
        **{key: float(sum(row[key] for row in rows) / len(rows)) for key in keys},
    }


def train_only_diagnostics(
    model: Any,
    tensor_examples: list[dict[str, Any]],
    *,
    sample_size: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    rng = random.Random(seed)
    sample = list(tensor_examples)
    rng.shuffle(sample)
    sample = sample[: max(1, min(sample_size, len(sample)))]
    model.eval()
    predictions = []
    positive_centroids = []
    with torch.no_grad():
        for row in sample:
            prediction = model(row["query"].unsqueeze(0)).squeeze(0)
            positives = row["candidates"][row["labels"] > 0]
            predictions.append(F.normalize(prediction, dim=0))
            positive_centroids.append(F.normalize(positives.mean(dim=0), dim=0))
    pred = torch.stack(predictions).to(device)
    centroids = torch.stack(positive_centroids).to(device)
    scores = pred @ centroids.T
    nearest = scores.argmax(dim=1).detach().cpu().numpy().tolist()
    counts: dict[int, int] = {}
    for index in nearest:
        counts[index] = counts.get(index, 0) + 1
    count_values = sorted(counts.values(), reverse=True)
    pairwise_cos = pred @ pred.T
    off_diagonal = pairwise_cos[~torch.eye(pairwise_cos.shape[0], dtype=torch.bool, device=pairwise_cos.device)]
    return {
        "sample_size": len(sample),
        "prediction_pairwise_cosine_mean": float(off_diagonal.mean().detach().cpu()) if off_diagonal.numel() else 0.0,
        "prediction_pairwise_cosine_std": float(off_diagonal.std().detach().cpu()) if off_diagonal.numel() else 0.0,
        "nearest_positive_centroid_unique_count": len(counts),
        "nearest_positive_centroid_max_count": max(count_values) if count_values else 0,
        "nearest_positive_centroid_top10_counts": count_values[:10],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


if __name__ == "__main__":
    main()
