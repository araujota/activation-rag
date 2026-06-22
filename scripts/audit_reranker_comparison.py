#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from activation_rag.ranking_audit import ScoreMap, compare_ranked_systems
from activation_rag.supervised_reranking import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit paired per-query reranker differences over frozen candidate groups.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-name", default="dense")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--baseline-scores-jsonl")
    parser.add_argument("--candidate-scores-jsonl")
    parser.add_argument("--candidate-mlp-checkpoint")
    parser.add_argument("--candidate-scores-out")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--randomization-iterations", type=int, default=10000)
    parser.add_argument("--changed-query-limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    summary = run_audit(
        groups_path=Path(args.groups),
        out_path=Path(args.out),
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        baseline_scores_path=Path(args.baseline_scores_jsonl) if args.baseline_scores_jsonl else None,
        candidate_scores_path=Path(args.candidate_scores_jsonl) if args.candidate_scores_jsonl else None,
        candidate_mlp_checkpoint=Path(args.candidate_mlp_checkpoint) if args.candidate_mlp_checkpoint else None,
        candidate_scores_out=Path(args.candidate_scores_out) if args.candidate_scores_out else None,
        top_k=args.top_k,
        randomization_iterations=args.randomization_iterations,
        changed_query_limit=args.changed_query_limit,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_audit(
    *,
    groups_path: Path,
    out_path: Path,
    baseline_name: str,
    candidate_name: str,
    baseline_scores_path: Path | None,
    candidate_scores_path: Path | None,
    candidate_mlp_checkpoint: Path | None,
    candidate_scores_out: Path | None,
    top_k: int,
    randomization_iterations: int,
    changed_query_limit: int,
    seed: int,
) -> dict[str, Any]:
    groups = load_jsonl(groups_path)
    baseline_scores = load_score_jsonl(baseline_scores_path) if baseline_scores_path else None
    if candidate_scores_path:
        candidate_scores = load_score_jsonl(candidate_scores_path)
    elif candidate_mlp_checkpoint:
        candidate_scores = score_with_mlp_checkpoint(groups, candidate_mlp_checkpoint)
    else:
        raise ValueError("pass --candidate-scores-jsonl or --candidate-mlp-checkpoint")
    if candidate_scores_out:
        write_score_jsonl(candidate_scores_out, candidate_scores)
    summary = compare_ranked_systems(
        groups,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        baseline_scores=baseline_scores,
        candidate_scores=candidate_scores,
        top_k=top_k,
        randomization_iterations=randomization_iterations,
        changed_query_limit=changed_query_limit,
        seed=seed,
    )
    summary["groups"] = str(groups_path)
    if baseline_scores_path:
        summary["baseline_scores_jsonl"] = str(baseline_scores_path)
    if candidate_scores_path:
        summary["candidate_scores_jsonl"] = str(candidate_scores_path)
    if candidate_mlp_checkpoint:
        summary["candidate_mlp_checkpoint"] = str(candidate_mlp_checkpoint)
    write_json(out_path, summary)
    return summary


def load_score_jsonl(path: Path | None) -> ScoreMap:
    if path is None:
        return {}
    scores: ScoreMap = {}
    for row in load_jsonl(path):
        scores[(str(row["query_id"]), str(row["chunk_id"]))] = float(row["score"])
    return scores


def write_score_jsonl(path: Path, scores: ScoreMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for (query_id, chunk_id), score in sorted(scores.items()):
            handle.write(json.dumps({"query_id": query_id, "chunk_id": chunk_id, "score": score}, sort_keys=True) + "\n")


def score_with_mlp_checkpoint(groups: list[dict[str, Any]], checkpoint_path: Path) -> ScoreMap:
    import torch
    from torch import nn

    payload = torch.load(checkpoint_path, map_location="cpu")
    feature_names = tuple(str(name) for name in payload["feature_names"])
    hidden_dim = int(payload["hidden_dim"])
    model = nn.Sequential(
        nn.Linear(len(feature_names), hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalizer = payload["normalizer"]
    scores: ScoreMap = {}
    with torch.no_grad():
        for group in groups:
            query_id = str(group["query_id"])
            for candidate in group.get("candidates", []):
                vector = [
                    (float((candidate.get("features") or {}).get(name, 0.0)) - float(normalizer["means"][name]))
                    / float(normalizer["scales"][name])
                    for name in feature_names
                ]
                tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
                scores[(query_id, str(candidate["chunk_id"]))] = float(model(tensor).squeeze())
    return scores


if __name__ == "__main__":
    main()
