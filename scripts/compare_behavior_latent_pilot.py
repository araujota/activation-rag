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

from activation_rag.ranking_audit import ScoreMap, compare_ranked_systems
from activation_rag.supervised_reranking import load_jsonl, write_json
from scripts.audit_reranker_comparison import load_score_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare behavior-latent pilot scores against dense, actpred, and Ettin.")
    parser.add_argument("--groups", required=True)
    parser.add_argument("--behavior-scores", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--actpred-scores")
    parser.add_argument("--ettin-scores")
    parser.add_argument("--randomization-iterations", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    summary = compare_pilot(
        groups_path=Path(args.groups),
        behavior_scores_path=Path(args.behavior_scores),
        out_path=Path(args.out),
        actpred_scores_path=Path(args.actpred_scores) if args.actpred_scores else None,
        ettin_scores_path=Path(args.ettin_scores) if args.ettin_scores else None,
        randomization_iterations=args.randomization_iterations,
        top_k=args.top_k,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def compare_pilot(
    *,
    groups_path: Path,
    behavior_scores_path: Path,
    out_path: Path,
    actpred_scores_path: Path | None,
    ettin_scores_path: Path | None,
    randomization_iterations: int,
    top_k: int,
) -> dict[str, Any]:
    groups = load_jsonl(groups_path)
    score_maps: dict[str, ScoreMap | None] = {"dense": None}
    score_maps["behavior"] = load_score_jsonl(behavior_scores_path)
    if actpred_scores_path and actpred_scores_path.exists():
        score_maps["actpred"] = load_score_jsonl(actpred_scores_path)
    if ettin_scores_path and ettin_scores_path.exists():
        score_maps["ettin"] = load_score_jsonl(ettin_scores_path)
    dense_vs_behavior = compare_ranked_systems(
        groups,
        baseline_name="dense",
        candidate_name="behavior",
        baseline_scores=None,
        candidate_scores=score_maps["behavior"] or {},
        top_k=top_k,
        randomization_iterations=randomization_iterations,
        changed_query_limit=10,
    )
    systems: dict[str, Any] = {"dense": dense_vs_behavior["baseline_metrics"], "behavior": dense_vs_behavior["candidate_metrics"]}
    paired: dict[str, Any] = {"behavior_minus_dense": compact_audit(dense_vs_behavior)}
    for name in ("actpred", "ettin"):
        scores = score_maps.get(name)
        if not scores:
            continue
        audit = compare_ranked_systems(
            groups,
            baseline_name="dense",
            candidate_name=name,
            baseline_scores=None,
            candidate_scores=scores,
            top_k=top_k,
            randomization_iterations=randomization_iterations,
            changed_query_limit=10,
        )
        systems[name] = audit["candidate_metrics"]
        paired[f"{name}_minus_dense"] = compact_audit(audit)
        against_behavior = compare_ranked_systems(
            groups,
            baseline_name="behavior",
            candidate_name=name,
            baseline_scores=score_maps["behavior"],
            candidate_scores=scores,
            top_k=top_k,
            randomization_iterations=randomization_iterations,
            changed_query_limit=10,
        )
        paired[f"{name}_minus_behavior"] = compact_audit(against_behavior)
    summary = {
        "schema_version": "activation_rag.behavior_latent_pilot_comparison.v1",
        "groups": str(groups_path),
        "behavior_scores": str(behavior_scores_path),
        "query_count": len(groups),
        "metrics": systems,
        "paired": paired,
    }
    write_json(out_path, summary)
    return summary


def compact_audit(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "metric_deltas": audit["metric_deltas"],
        "paired_significance": audit["paired_significance"],
        "changed_query_summary": audit["changed_query_summary"],
    }


if __name__ == "__main__":
    main()
