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
from scripts.build_core245_permutation_groups import _load_cache_rows
from scripts.train_behavior_latent_reranker import build_examples, evaluate_examples, write_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frozen query+candidate groups with a trained behavior-latent reranker.")
    parser.add_argument("--groups", required=True, help="Behavior-pair candidate groups JSONL.")
    parser.add_argument("--telemetry-cache-dir", required=True, help="Directory of per behavior_chunk_id telemetry JSON rows.")
    parser.add_argument("--checkpoint", required=True, help="Trained behavior-latent model.pt checkpoint.")
    parser.add_argument("--scores-out", required=True, help="Output JSONL of query_id/chunk_id/score rows.")
    parser.add_argument("--metrics-out", required=True, help="Output metrics JSON.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = score_checkpoint(
        groups_path=Path(args.groups),
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        checkpoint_path=Path(args.checkpoint),
        scores_out=Path(args.scores_out),
        metrics_out=Path(args.metrics_out),
        top_k=args.top_k,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def score_checkpoint(
    *,
    groups_path: Path,
    telemetry_cache_dir: Path,
    checkpoint_path: Path,
    scores_out: Path,
    metrics_out: Path,
    top_k: int,
    device: str,
) -> dict[str, Any]:
    import torch

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_ids = [str(feature_id) for feature_id in payload["feature_ids"]]
    activation_transform = str(payload.get("activation_transform") or "raw")
    rows_by_chunk = _load_cache_rows(telemetry_cache_dir)
    groups = load_jsonl(groups_path)
    examples = build_examples(groups, rows_by_chunk, feature_ids, activation_transform=activation_transform)
    if not examples:
        raise ValueError("no scorable groups with behavior telemetry and positive/negative labels")

    model = build_model(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        state_dict=payload["state_dict"],
        device=device,
    )
    alpha = float(payload.get("selected_alpha", 1.0))
    metrics, scores = evaluate_examples(
        examples,
        model,
        payload["normalizer"],
        alpha=alpha,
        top_k=top_k,
        device=device,
    )
    write_scores(scores_out, scores)
    summary = {
        "schema_version": "activation_rag.behavior_latent_checkpoint_score.v1",
        "groups": str(groups_path),
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "checkpoint": str(checkpoint_path),
        "prompt_representation": payload.get("prompt_representation"),
        "activation_transform": activation_transform,
        "selected_alpha": alpha,
        "top_k": top_k,
        "source_query_count": len(groups),
        "scored_query_count": len(examples),
        "source_candidate_count": sum(len(group.get("candidates", [])) for group in groups),
        "scored_candidate_count": sum(len(example["candidates"]) for example in examples),
        "metrics": metrics,
        "scores_out": str(scores_out),
    }
    write_json(metrics_out, summary)
    return summary


def build_model(*, input_dim: int, hidden_dim: int, state_dict: dict[str, Any], device: str) -> Any:
    import torch
    from torch import nn

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == "__main__":
    main()
