#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.supervised_reranking import load_jsonl


SCHEMA_VERSION = "activation_rag.caa_prefill_request.v1"
PROMPT_TEMPLATE_ID = "raw_chunk_v1_strict_zero_caa8"
PROMPT_TEMPLATE = "{chunk_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unique strict-zero CAA/EM prefill requests from reranker groups.")
    parser.add_argument("--group-file", action="append", required=True, help="JSONL candidate group file. Repeat for train/dev/test.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out")
    parser.add_argument("--seed", default="13")
    parser.add_argument("--max-groups-per-file", type=int, default=0)
    parser.add_argument("--request-prefix", default="activation-rag-caa")
    args = parser.parse_args()

    result = build_requests(
        group_files=[Path(item) for item in args.group_file],
        max_groups_per_file=int(args.max_groups_per_file),
        seed=str(args.seed),
        request_prefix=str(args.request_prefix),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in result["requests"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = dict(result["summary"])
    summary["out"] = str(out_path)
    if args.summary_out:
        path = Path(args.summary_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def build_requests(
    *,
    group_files: list[Path],
    max_groups_per_file: int,
    seed: str,
    request_prefix: str,
) -> dict[str, Any]:
    requests: dict[str, dict[str, Any]] = {}
    per_file: list[dict[str, Any]] = []
    for group_file in group_files:
        groups = load_jsonl(group_file)
        original_group_count = len(groups)
        if max_groups_per_file > 0 and len(groups) > max_groups_per_file:
            rng = random.Random(_stable_int(f"{seed}:{group_file}"))
            groups = sorted(rng.sample(groups, max_groups_per_file), key=lambda row: str(row.get("query_id")))
        for group in groups:
            query_chunk_id = str(group.get("query_activation_chunk_id") or "")
            query_text = str(group.get("query_text") or "")
            if query_chunk_id and query_text:
                _add_request(
                    requests,
                    chunk_id=query_chunk_id,
                    document_id="query",
                    text=query_text,
                    prompt_section_label="query",
                    request_prefix=request_prefix,
                )
            for candidate in group.get("candidates", []):
                chunk_id = str(candidate.get("chunk_id") or "")
                text = str(candidate.get("text") or "")
                if not chunk_id or not text:
                    continue
                _add_request(
                    requests,
                    chunk_id=chunk_id,
                    document_id=str(candidate.get("doc_id") or ""),
                    text=text,
                    prompt_section_label="document_chunk",
                    request_prefix=request_prefix,
                )
        per_file.append(
            {
                "group_file": str(group_file),
                "input_group_count": original_group_count,
                "used_group_count": len(groups),
            }
        )
    rows = [requests[key] for key in sorted(requests)]
    summary = {
        "schema_version": "activation_rag.caa_prefill_request_summary.v1",
        "group_files": per_file,
        "request_count": len(rows),
        "query_count": sum(1 for row in rows if row["prompt_section_label"] == "query"),
        "document_chunk_count": sum(1 for row in rows if row["prompt_section_label"] == "document_chunk"),
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_template_hash": _hash(PROMPT_TEMPLATE, 24),
        "max_groups_per_file": max_groups_per_file,
        "seed": seed,
    }
    return {"requests": rows, "summary": summary}


def _add_request(
    rows: dict[str, dict[str, Any]],
    *,
    chunk_id: str,
    document_id: str,
    text: str,
    prompt_section_label: str,
    request_prefix: str,
) -> None:
    if chunk_id in rows:
        return
    text_hash = _hash(text, 24)
    rows[chunk_id] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_id": chunk_id,
        "document_id": document_id,
        "capture_run_id": _hash(f"{request_prefix}:{chunk_id}:{text_hash}", 24),
        "prompt_text": text,
        "text_hash": text_hash,
        "prompt_section_label": prompt_section_label,
        "requested_prompt_section_label": prompt_section_label,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_template_hash": _hash(PROMPT_TEMPLATE, 24),
        "provider_id": "qwen-caa8-compact-prefill",
        "model_id": "qwen3-4b-q8_0",
    }


def _hash(value: str, length: int = 32) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def _stable_int(value: str) -> int:
    return int(_hash(value, 16), 16)


if __name__ == "__main__":
    main()
