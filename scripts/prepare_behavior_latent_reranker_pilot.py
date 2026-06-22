#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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

from activation_rag.schema import stable_hash
from activation_rag.supervised_reranking import load_jsonl


PROMPT_TEMPLATE_ID = "behavior_support_pair_v1"
PROMPT_TEMPLATE = """Query:
{query}

Candidate evidence:
{evidence}

Task:
Decide whether the candidate evidence directly supports answering the query. Focus on exact support, not topical similarity.

Answer support:"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare query+candidate behavior-latent pilot groups and capture requests.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--dev-groups", required=True)
    parser.add_argument("--test-groups", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-limit", type=int, default=200)
    parser.add_argument("--dev-limit", type=int, default=80)
    parser.add_argument("--test-limit", type=int, default=120)
    parser.add_argument("--candidates-per-query", type=int, default=8)
    parser.add_argument("--seed", default="13")
    args = parser.parse_args()

    summary = prepare_pilot(
        dataset_name=args.dataset_name,
        train_groups_path=Path(args.train_groups),
        dev_groups_path=Path(args.dev_groups),
        test_groups_path=Path(args.test_groups),
        out_dir=Path(args.out_dir),
        train_limit=args.train_limit,
        dev_limit=args.dev_limit,
        test_limit=args.test_limit,
        candidates_per_query=args.candidates_per_query,
        seed=str(args.seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def prepare_pilot(
    *,
    dataset_name: str,
    train_groups_path: Path,
    dev_groups_path: Path,
    test_groups_path: Path,
    out_dir: Path,
    train_limit: int,
    dev_limit: int,
    test_limit: int,
    candidates_per_query: int,
    seed: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    split_specs = {
        "train": (train_groups_path, train_limit),
        "dev": (dev_groups_path, dev_limit),
        "test": (test_groups_path, test_limit),
    }
    all_requests: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for split, (path, limit) in split_specs.items():
        groups, requests = build_split_groups(
            dataset_name=dataset_name,
            split=split,
            source_groups=load_jsonl(path),
            limit=limit,
            candidates_per_query=candidates_per_query,
            prompt_hash=prompt_hash,
            seed=seed,
        )
        groups_path = out_dir / f"{split}-groups.behavior-pair.jsonl"
        requests_path = out_dir / f"{split}-capture-requests.jsonl"
        write_jsonl(groups_path, groups)
        write_jsonl(requests_path, requests)
        for request in requests:
            all_requests.setdefault(str(request["chunk_id"]), request)
        summaries[split] = {
            "source_groups": str(path),
            "groups_out": str(groups_path),
            "requests_out": str(requests_path),
            "query_count": len(groups),
            "request_count": len(requests),
            "positive_query_count": sum(1 for group in groups if any(int(c.get("label", 0)) > 0 for c in group["candidates"])),
            "candidate_count_mean": (sum(len(group["candidates"]) for group in groups) / len(groups)) if groups else 0.0,
        }
    all_requests_path = out_dir / "all-capture-requests.jsonl"
    write_jsonl(all_requests_path, list(all_requests.values()))
    manifest = {
        "schema_version": "activation_rag.behavior_latent_pilot_prepare.v1",
        "dataset_name": dataset_name,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_template_hash": prompt_hash,
        "candidates_per_query": candidates_per_query,
        "seed": seed,
        "splits": summaries,
        "all_requests_out": str(all_requests_path),
        "unique_request_count": len(all_requests),
        "capture_note": "Pair prompt uses Q + candidate evidence and existing Qwen/RMT/SAE Core245 max-over-prefill-token capture.",
    }
    (out_dir / "prepare-summary.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_split_groups(
    *,
    dataset_name: str,
    split: str,
    source_groups: list[dict[str, Any]],
    limit: int,
    candidates_per_query: int,
    prompt_hash: str,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if candidates_per_query < 2:
        raise ValueError("candidates_per_query must be at least 2")
    selected_source = select_groups(source_groups, limit=limit, seed=f"{seed}:{dataset_name}:{split}")
    groups: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []
    for group in selected_source:
        selected_candidates = select_candidates(group, candidates_per_query=candidates_per_query)
        if not selected_candidates:
            continue
        query_id = str(group["query_id"])
        query_text = str(group.get("query_text") or "")
        updated_candidates = []
        for candidate in selected_candidates:
            evidence = str(candidate.get("text") or "")
            behavior_chunk_id = behavior_pair_id(
                dataset_name=dataset_name,
                query_id=query_id,
                chunk_id=str(candidate["chunk_id"]),
                prompt_hash=prompt_hash,
            )
            prompt_text = PROMPT_TEMPLATE.format(query=query_text.strip(), evidence=evidence.strip())
            updated = dict(candidate)
            updated["behavior_chunk_id"] = behavior_chunk_id
            updated["behavior_prompt_template_id"] = PROMPT_TEMPLATE_ID
            updated["behavior_prompt_template_hash"] = prompt_hash
            updated["features"] = dict(candidate.get("features") or {})
            updated["features"].setdefault("dense_score", float(candidate.get("dense_score", 0.0)))
            updated["features"].setdefault("dense_rank_reciprocal", 1.0 / max(1, int(candidate.get("dense_rank", 10**9))))
            updated_candidates.append(updated)
            requests.append(
                {
                    "schema_version": "activation_rag.prefill_capture_request.v1",
                    "chunk_id": behavior_chunk_id,
                    "document_id": f"behavior_pair:{dataset_name}:{split}:{query_id}",
                    "capture_run_id": f"behavior-pair-{dataset_name}-{split}",
                    "provider_id": "qwen-rmt-sae-prefill",
                    "model_id": "qwen3-4b-rmt-sae",
                    "site_id": "l07_resid_pre",
                    "layer_selection_policy": "qwen3_rmt_l07_resid_pre_core245",
                    "prompt_template_id": PROMPT_TEMPLATE_ID,
                    "prompt_template_hash": prompt_hash,
                    "normalization_policy": "qwen_sae_checkpoint_mean_rms_topk64",
                    "requested_prompt_section_label": "query_candidate_behavior_prompt",
                    "prompt_section_label": "query_candidate_behavior_prompt",
                    "text": prompt_text,
                    "prompt_text": prompt_text,
                    "text_hash": stable_hash(prompt_text, 32),
                    "token_count_estimate": max(1, len(prompt_text.split())),
                    "metadata": {
                        "dataset_name": dataset_name,
                        "split": split,
                        "query_id": query_id,
                        "source_chunk_id": str(candidate["chunk_id"]),
                        "source_doc_id": str(candidate.get("doc_id") or ""),
                        "label": int(candidate.get("label", 0)),
                        "label_score": float(candidate.get("label_score", candidate.get("label", 0))),
                    },
                }
            )
        add_group_zscores(updated_candidates, "dense_score", "dense_z")
        updated_group = dict(group)
        updated_group["behavior_prompt_template_id"] = PROMPT_TEMPLATE_ID
        updated_group["behavior_prompt_template_hash"] = prompt_hash
        updated_group["candidates"] = updated_candidates
        groups.append(updated_group)
    return groups, dedupe_requests(requests)


def select_groups(groups: list[dict[str, Any]], *, limit: int, seed: str) -> list[dict[str, Any]]:
    eligible = [group for group in groups if has_positive_and_negative(group)]
    ordered = sorted(eligible, key=lambda group: stable_hash(f"{seed}:{group['query_id']}", 32))
    return ordered[:limit] if limit > 0 else ordered


def has_positive_and_negative(group: dict[str, Any]) -> bool:
    positives = [candidate for candidate in group.get("candidates", []) if int(candidate.get("label", 0)) > 0]
    negatives = [candidate for candidate in group.get("candidates", []) if int(candidate.get("label", 0)) <= 0]
    return bool(positives and negatives)


def select_candidates(group: dict[str, Any], *, candidates_per_query: int) -> list[dict[str, Any]]:
    candidates = list(group.get("candidates", []))
    positives = sorted([candidate for candidate in candidates if int(candidate.get("label", 0)) > 0], key=dense_order_key)
    negatives = sorted([candidate for candidate in candidates if int(candidate.get("label", 0)) <= 0], key=dense_order_key)
    if not positives or not negatives:
        return []
    budget_for_negatives = max(1, candidates_per_query - len(positives))
    selected = positives + negatives[:budget_for_negatives]
    return sorted(selected, key=dense_order_key)


def dense_order_key(candidate: dict[str, Any]) -> tuple[int, float]:
    return (int(candidate.get("dense_rank", 10**9)), -float(candidate.get("dense_score", 0.0)))


def behavior_pair_id(*, dataset_name: str, query_id: str, chunk_id: str, prompt_hash: str) -> str:
    return stable_hash(f"behavior_pair_v1\n{dataset_name}\n{query_id}\n{chunk_id}\n{prompt_hash}", 32)


def dedupe_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        by_id.setdefault(str(request["chunk_id"]), request)
    return list(by_id.values())


def add_group_zscores(candidates: list[dict[str, Any]], source: str, target: str) -> None:
    values = [float((candidate.get("features") or {}).get(source, candidate.get(source, 0.0))) for candidate in candidates]
    if not values:
        return
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = variance**0.5 or 1.0
    for candidate, value in zip(candidates, values, strict=True):
        candidate.setdefault("features", {})[target] = (value - mean) / std


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
