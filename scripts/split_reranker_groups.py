#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from activation_rag.supervised_reranking import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically split reranker query groups into train/dev JSONL.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--dev-out", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", default="activation-rag")
    args = parser.parse_args()
    summary = split_groups(
        source=Path(args.source),
        train_out=Path(args.train_out),
        dev_out=Path(args.dev_out),
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def split_groups(
    *,
    source: Path,
    train_out: Path,
    dev_out: Path,
    dev_fraction: float,
    seed: str,
) -> dict[str, Any]:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between 0 and 1")
    groups = load_jsonl(source)
    ordered = sorted(groups, key=lambda group: _score(str(group["query_id"]), seed))
    dev_count = max(1, round(len(ordered) * dev_fraction)) if ordered else 0
    dev_ids = {str(group["query_id"]) for group in ordered[:dev_count]}
    train_rows = [group for group in groups if str(group["query_id"]) not in dev_ids]
    dev_rows = [group for group in groups if str(group["query_id"]) in dev_ids]
    _write_jsonl(train_out, train_rows)
    _write_jsonl(dev_out, dev_rows)
    return {
        "source": str(source),
        "train_out": str(train_out),
        "dev_out": str(dev_out),
        "input_count": len(groups),
        "train_count": len(train_rows),
        "dev_count": len(dev_rows),
        "dev_fraction": dev_fraction,
        "seed": seed,
    }


def _score(query_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\n{query_id}".encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
