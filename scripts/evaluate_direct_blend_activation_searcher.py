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
from scripts.build_core245_permutation_groups import _load_cache_rows, _load_manifest
from scripts.train_activation_representation_searcher import _build_examples, _make_vectorizer, evaluate_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a direct-blend answer-activation predictor checkpoint to candidate groups.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--telemetry-cache-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--model-only-scores-out")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = evaluate_direct_blend_checkpoint(
        groups_path=Path(args.groups),
        telemetry_cache_dir=Path(args.telemetry_cache_dir),
        feature_manifest_path=Path(args.feature_manifest),
        model_path=Path(args.model),
        out_path=Path(args.out),
        scores_out=Path(args.scores_out),
        model_only_scores_out=Path(args.model_only_scores_out) if args.model_only_scores_out else None,
        top_k=args.top_k,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def evaluate_direct_blend_checkpoint(
    *,
    groups_path: Path,
    telemetry_cache_dir: Path,
    feature_manifest_path: Path,
    model_path: Path,
    out_path: Path,
    scores_out: Path,
    model_only_scores_out: Path | None,
    top_k: int,
    device: str,
) -> dict[str, Any]:
    import torch
    from torch import nn

    payload = torch.load(model_path, map_location="cpu")
    if payload.get("schema_version") != "activation_rag.direct_blend_answer_representation_searcher.v1":
        raise ValueError(f"unsupported checkpoint schema: {payload.get('schema_version')}")
    manifest = _load_manifest(feature_manifest_path)
    feature_ids = [str(feature_id) for feature_id in payload.get("feature_ids") or manifest["feature_ids"]]
    vectorizer = _make_vectorizer(
        representation=str(payload["representation"]),
        feature_ids=feature_ids,
        feature_meta=manifest["feature_meta"],
        seed=13,
    )
    model = nn.Sequential(
        nn.Linear(int(payload["input_dim"]), int(payload["hidden_dim"])),
        nn.ReLU(),
        nn.Linear(int(payload["hidden_dim"]), int(payload["hidden_dim"])),
        nn.ReLU(),
        nn.Linear(int(payload["hidden_dim"]), int(payload["input_dim"])),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    examples = _build_examples(load_jsonl(groups_path), _load_cache_rows(telemetry_cache_dir), feature_ids, vectorizer)
    selected_alpha = float(payload["selected_alpha"])
    metrics, scores = evaluate_examples(
        examples,
        model=model,
        normalizer=payload["normalizer"],
        top_k=top_k,
        blend_alpha=selected_alpha,
        device=device,
    )
    model_only_metrics, model_only_scores = evaluate_examples(
        examples,
        model=model,
        normalizer=payload["normalizer"],
        top_k=top_k,
        blend_alpha=None,
        device=device,
    )
    _write_scores(scores_out, scores)
    if model_only_scores_out:
        _write_scores(model_only_scores_out, model_only_scores)
    summary: dict[str, Any] = {
        "schema_version": "activation_rag.direct_blend_answer_representation_searcher_eval.v1",
        "groups": str(groups_path),
        "telemetry_cache_dir": str(telemetry_cache_dir),
        "feature_manifest": str(feature_manifest_path),
        "model": str(model_path),
        "source_dataset_name": str(payload.get("dataset_name") or ""),
        "query_count": len(examples),
        "top_k": top_k,
        "representation": str(payload["representation"]),
        "feature_set_id": str(payload.get("feature_set_id") or ""),
        "selected_alpha": selected_alpha,
        "metrics": metrics,
        "model_only_metrics": model_only_metrics,
        "scores_out": str(scores_out),
        "model_only_scores_out": str(model_only_scores_out) if model_only_scores_out else None,
    }
    write_json(out_path, summary)
    return summary


def _write_scores(path: Path, scores: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (query_id, chunk_id), score in sorted(scores.items()):
            handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
