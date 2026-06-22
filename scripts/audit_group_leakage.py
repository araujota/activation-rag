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

from activation_rag.supervised_reranking import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit train/dev/test candidate groups for leakage and memorization controls.")
    parser.add_argument("--train-groups", required=True)
    parser.add_argument("--dev-groups")
    parser.add_argument("--test-groups", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--max-shingle-postings", type=int, default=250)
    args = parser.parse_args()

    summary = audit_group_leakage(
        train_groups_path=Path(args.train_groups),
        dev_groups_path=Path(args.dev_groups) if args.dev_groups else None,
        test_groups_path=Path(args.test_groups),
        out_path=Path(args.out),
        example_limit=args.example_limit,
        near_duplicate_threshold=args.near_duplicate_threshold,
        shingle_size=args.shingle_size,
        max_shingle_postings=args.max_shingle_postings,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def audit_group_leakage(
    *,
    train_groups_path: Path,
    dev_groups_path: Path | None,
    test_groups_path: Path,
    out_path: Path,
    example_limit: int = 20,
    near_duplicate_threshold: float = 0.85,
    shingle_size: int = 5,
    max_shingle_postings: int = 250,
) -> dict[str, Any]:
    splits = {
        "train": _index_groups(load_jsonl(train_groups_path)),
        "test": _index_groups(load_jsonl(test_groups_path)),
    }
    paths = {"train": str(train_groups_path), "test": str(test_groups_path)}
    if dev_groups_path:
        splits["dev"] = _index_groups(load_jsonl(dev_groups_path))
        paths["dev"] = str(dev_groups_path)

    comparisons: dict[str, Any] = {}
    for left, right in _comparison_pairs(tuple(splits.keys())):
        comparisons[f"{left}_vs_{right}"] = _compare_splits(
            left_name=left,
            left=splits[left],
            right_name=right,
            right=splits[right],
            example_limit=example_limit,
            near_duplicate_threshold=near_duplicate_threshold,
            shingle_size=shingle_size,
            max_shingle_postings=max_shingle_postings,
        )

    summary = {
        "schema_version": "activation_rag.group_leakage_audit.v1",
        "paths": paths,
        "splits": {
            name: {
                "query_count": len(index["query_ids"]),
                "unique_query_text_count": len(index["query_text_hashes"]),
                "positive_doc_count": len(index["positive_doc_ids"]),
                "candidate_doc_count": len(index["candidate_doc_ids"]),
                "positive_candidate_pair_count": len(index["positive_pairs"]),
                "unique_candidate_text_count": len(index["candidate_text_hashes"]),
                "unique_positive_text_count": len(index["positive_text_hashes"]),
            }
            for name, index in splits.items()
        },
        "near_duplicate_policy": {
            "threshold": near_duplicate_threshold,
            "shingle_size": shingle_size,
            "max_shingle_postings": max_shingle_postings,
        },
        "comparisons": comparisons,
    }
    write_json(out_path, summary)
    return summary


def _index_groups(groups: list[dict[str, Any]]) -> dict[str, Any]:
    query_ids: set[str] = set()
    query_text_hashes: dict[str, list[str]] = {}
    positive_doc_ids: set[str] = set()
    candidate_doc_ids: set[str] = set()
    positive_pairs: set[tuple[str, str]] = set()
    candidate_pairs: set[tuple[str, str]] = set()
    query_id_to_text: dict[str, str] = {}
    candidate_text_hashes: dict[str, list[dict[str, str]]] = {}
    positive_text_hashes: dict[str, list[dict[str, str]]] = {}
    candidate_text_items: list[dict[str, Any]] = []
    positive_text_items: list[dict[str, Any]] = []
    query_text_items: list[dict[str, Any]] = []
    for group in groups:
        query_id = str(group["query_id"])
        query_text = _canonical_text(str(group.get("query_text", "")))
        query_hash = _sha256(query_text)
        query_ids.add(query_id)
        query_id_to_text[query_id] = query_text
        query_text_hashes.setdefault(query_hash, []).append(query_id)
        query_text_items.append({"kind": "query", "query_id": query_id, "text": query_text})
        positives = {str(doc_id) for doc_id in group.get("positive_doc_ids", [])}
        positive_doc_ids.update(positives)
        for candidate in group.get("candidates", []):
            doc_id = str(candidate["doc_id"])
            chunk_id = str(candidate.get("chunk_id") or "")
            text = _canonical_text(str(candidate.get("text", "")))
            text_hash = _sha256(text)
            text_record = {"query_id": query_id, "doc_id": doc_id, "chunk_id": chunk_id, "text": text}
            candidate_doc_ids.add(doc_id)
            candidate_pairs.add((query_id, doc_id))
            candidate_text_hashes.setdefault(text_hash, []).append(text_record)
            candidate_text_items.append({**text_record, "kind": "candidate"})
            if int(candidate.get("label", 0)) > 0:
                positive_doc_ids.add(doc_id)
                positive_pairs.add((query_id, doc_id))
                positive_text_hashes.setdefault(text_hash, []).append(text_record)
                positive_text_items.append({**text_record, "kind": "positive"})
    return {
        "query_ids": query_ids,
        "query_id_to_text": query_id_to_text,
        "query_text_hashes": query_text_hashes,
        "positive_doc_ids": positive_doc_ids,
        "candidate_doc_ids": candidate_doc_ids,
        "positive_pairs": positive_pairs,
        "candidate_pairs": candidate_pairs,
        "candidate_text_hashes": candidate_text_hashes,
        "positive_text_hashes": positive_text_hashes,
        "candidate_text_items": candidate_text_items,
        "positive_text_items": positive_text_items,
        "query_text_items": query_text_items,
    }


def _comparison_pairs(names: tuple[str, ...]) -> list[tuple[str, str]]:
    ordered = [name for name in ("train", "dev", "test") if name in names]
    return [(ordered[left], ordered[right]) for left in range(len(ordered)) for right in range(left + 1, len(ordered))]


def _compare_splits(
    *,
    left_name: str,
    left: dict[str, Any],
    right_name: str,
    right: dict[str, Any],
    example_limit: int,
    near_duplicate_threshold: float,
    shingle_size: int,
    max_shingle_postings: int,
) -> dict[str, Any]:
    query_id_overlap = sorted(left["query_ids"] & right["query_ids"])
    query_text_overlap_hashes = sorted(set(left["query_text_hashes"]) & set(right["query_text_hashes"]))
    positive_doc_overlap = sorted(left["positive_doc_ids"] & right["positive_doc_ids"])
    candidate_doc_overlap = sorted(left["candidate_doc_ids"] & right["candidate_doc_ids"])
    positive_pair_overlap = sorted(left["positive_pairs"] & right["positive_pairs"])
    candidate_pair_overlap = sorted(left["candidate_pairs"] & right["candidate_pairs"])
    train_positive_in_eval_candidates = sorted(left["positive_doc_ids"] & right["candidate_doc_ids"])
    candidate_text_overlap_hashes = sorted(set(left["candidate_text_hashes"]) & set(right["candidate_text_hashes"]))
    positive_text_overlap_hashes = sorted(set(left["positive_text_hashes"]) & set(right["positive_text_hashes"]))
    left_positive_text_in_right_candidate_hashes = sorted(set(left["positive_text_hashes"]) & set(right["candidate_text_hashes"]))
    near_duplicate_queries = _near_duplicate_pairs(
        left["query_text_items"],
        right["query_text_items"],
        threshold=near_duplicate_threshold,
        shingle_size=shingle_size,
        max_shingle_postings=max_shingle_postings,
        example_limit=example_limit,
    )
    near_duplicate_left_positive_in_right_candidate = _near_duplicate_pairs(
        left["positive_text_items"],
        right["candidate_text_items"],
        threshold=near_duplicate_threshold,
        shingle_size=shingle_size,
        max_shingle_postings=max_shingle_postings,
        example_limit=example_limit,
    )

    return {
        "left_split": left_name,
        "right_split": right_name,
        "query_id_overlap_count": len(query_id_overlap),
        "query_text_overlap_count": len(query_text_overlap_hashes),
        "positive_doc_overlap_count": len(positive_doc_overlap),
        "candidate_doc_overlap_count": len(candidate_doc_overlap),
        "positive_pair_overlap_count": len(positive_pair_overlap),
        "candidate_pair_overlap_count": len(candidate_pair_overlap),
        "left_positive_doc_in_right_candidate_doc_count": len(train_positive_in_eval_candidates),
        "candidate_text_overlap_count": len(candidate_text_overlap_hashes),
        "positive_text_overlap_count": len(positive_text_overlap_hashes),
        "left_positive_text_in_right_candidate_text_count": len(left_positive_text_in_right_candidate_hashes),
        "near_duplicate_query_text_count": near_duplicate_queries["count"],
        "near_duplicate_left_positive_in_right_candidate_text_count": near_duplicate_left_positive_in_right_candidate["count"],
        "rates": {
            "query_id_overlap_right": _rate(len(query_id_overlap), len(right["query_ids"])),
            "query_text_overlap_right": _rate(len(query_text_overlap_hashes), len(right["query_text_hashes"])),
            "positive_doc_overlap_right": _rate(len(positive_doc_overlap), len(right["positive_doc_ids"])),
            "candidate_doc_overlap_right": _rate(len(candidate_doc_overlap), len(right["candidate_doc_ids"])),
            "positive_pair_overlap_right": _rate(len(positive_pair_overlap), len(right["positive_pairs"])),
            "candidate_pair_overlap_right": _rate(len(candidate_pair_overlap), len(right["candidate_pairs"])),
            "candidate_text_overlap_right": _rate(len(candidate_text_overlap_hashes), len(right["candidate_text_hashes"])),
            "positive_text_overlap_right": _rate(len(positive_text_overlap_hashes), len(right["positive_text_hashes"])),
            "left_positive_text_in_right_candidate_text_right": _rate(
                len(left_positive_text_in_right_candidate_hashes),
                len(right["candidate_text_hashes"]),
            ),
        },
        "examples": {
            "query_id_overlap": query_id_overlap[:example_limit],
            "query_text_overlap": _query_text_examples(left, right, query_text_overlap_hashes, example_limit),
            "positive_doc_overlap": positive_doc_overlap[:example_limit],
            "positive_pair_overlap": [{"query_id": query_id, "doc_id": doc_id} for query_id, doc_id in positive_pair_overlap[:example_limit]],
            "candidate_text_overlap": _text_hash_examples(left["candidate_text_hashes"], right["candidate_text_hashes"], candidate_text_overlap_hashes, example_limit),
            "positive_text_overlap": _text_hash_examples(left["positive_text_hashes"], right["positive_text_hashes"], positive_text_overlap_hashes, example_limit),
            "left_positive_text_in_right_candidate_text": _text_hash_examples(
                left["positive_text_hashes"],
                right["candidate_text_hashes"],
                left_positive_text_in_right_candidate_hashes,
                example_limit,
            ),
            "near_duplicate_query_text": near_duplicate_queries["examples"],
            "near_duplicate_left_positive_in_right_candidate_text": near_duplicate_left_positive_in_right_candidate["examples"],
        },
    }


def _query_text_examples(
    left: dict[str, Any],
    right: dict[str, Any],
    overlap_hashes: list[str],
    example_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query_hash in overlap_hashes[:example_limit]:
        left_ids = sorted(left["query_text_hashes"][query_hash])
        right_ids = sorted(right["query_text_hashes"][query_hash])
        text = left["query_id_to_text"].get(left_ids[0], "") if left_ids else ""
        rows.append({"query_text_hash": query_hash, "left_query_ids": left_ids, "right_query_ids": right_ids, "query_text": text})
    return rows


def _canonical_text(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _text_hash_examples(
    left_hashes: dict[str, list[dict[str, str]]],
    right_hashes: dict[str, list[dict[str, str]]],
    overlap_hashes: list[str],
    example_limit: int,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for text_hash in overlap_hashes[:example_limit]:
        left_rows = left_hashes[text_hash]
        right_rows = right_hashes[text_hash]
        text = left_rows[0].get("text", "")
        examples.append(
            {
                "text_hash": text_hash,
                "left": [_compact_text_record(row) for row in left_rows[:example_limit]],
                "right": [_compact_text_record(row) for row in right_rows[:example_limit]],
                "text_preview": text[:240],
            }
        )
    return examples


def _compact_text_record(row: dict[str, str]) -> dict[str, str]:
    return {
        "query_id": str(row.get("query_id", "")),
        "doc_id": str(row.get("doc_id", "")),
        "chunk_id": str(row.get("chunk_id", "")),
    }


def _near_duplicate_pairs(
    left_items: list[dict[str, Any]],
    right_items: list[dict[str, Any]],
    *,
    threshold: float,
    shingle_size: int,
    max_shingle_postings: int,
    example_limit: int,
) -> dict[str, Any]:
    if not left_items or not right_items:
        return {"count": 0, "examples": []}
    left_sets: list[set[str]] = [_shingles(str(item.get("text", "")), shingle_size) for item in left_items]
    right_sets: list[set[str]] = [_shingles(str(item.get("text", "")), shingle_size) for item in right_items]
    shingle_index: dict[str, list[int]] = {}
    for index, shingles in enumerate(left_sets):
        for shingle in shingles:
            shingle_index.setdefault(shingle, []).append(index)
    shingle_index = {key: values for key, values in shingle_index.items() if len(values) <= max_shingle_postings}

    count = 0
    examples: list[dict[str, Any]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for right_index, right_shingles in enumerate(right_sets):
        candidate_counts: dict[int, int] = {}
        for shingle in right_shingles:
            for left_index in shingle_index.get(shingle, []):
                candidate_counts[left_index] = candidate_counts.get(left_index, 0) + 1
        for left_index, intersection_count in candidate_counts.items():
            pair = (left_index, right_index)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            left_shingles = left_sets[left_index]
            union_count = len(left_shingles) + len(right_shingles) - intersection_count
            jaccard = float(intersection_count) / float(union_count) if union_count else 0.0
            if jaccard >= threshold:
                count += 1
                if len(examples) < example_limit:
                    examples.append(
                        {
                            "jaccard": jaccard,
                            "left": _compact_near_duplicate_item(left_items[left_index]),
                            "right": _compact_near_duplicate_item(right_items[right_index]),
                        }
                    )
    return {"count": count, "examples": examples}


def _compact_near_duplicate_item(item: dict[str, Any]) -> dict[str, str]:
    text = str(item.get("text", ""))
    return {
        "kind": str(item.get("kind", "")),
        "query_id": str(item.get("query_id", "")),
        "doc_id": str(item.get("doc_id", "")),
        "chunk_id": str(item.get("chunk_id", "")),
        "text_preview": text[:240],
    }


def _shingles(text: str, shingle_size: int) -> set[str]:
    tokens = text.split()
    if not tokens:
        return set()
    if len(tokens) <= shingle_size:
        return {" ".join(tokens)}
    return {" ".join(tokens[index : index + shingle_size]) for index in range(0, len(tokens) - shingle_size + 1)}


if __name__ == "__main__":
    main()
